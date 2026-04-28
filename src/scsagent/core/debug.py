from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Annotated, Optional, List, Dict, Literal, Union
import logging
import re
import json
import ipdb
import os
import glob
from datetime import datetime


from scsagent.utils.log_decorators import log_func, log_io
from scsagent.core.tool_call import (
    inspect_data_tool,
    interact_with_user_tool,
    python_repl_tool,
    bash_tool,
)
from utils.sandbox import get_docker_manager
from utils.database import Database
from scsagent.ingestion.get_doc import summary_doc


class AgentState(TypedDict):
    task: str
    tool: str
    data_info: str
    docs: List[Dict[str, Any]]  # ← 改这里！原来是 list[str]
    host_data_path: str
    container_work_dir: str
    codes: list[str]
    outputs: list[str]
    done: bool


logger = logging.getLogger(__name__)

qwen3max = ChatOpenAI(
    model="qwen3-max",
    # model="deepseek-v3-2-251201",
    # model="qwen3-coder-plus",
    base_url="",
    api_key="",
)


llm = ChatOpenAI(
    model="qwen3-max",
    # model="deepseek-v3-2-251201",
    # model="qwen3-coder-plus",
    base_url="",
    api_key="",
)

tools = [python_repl_tool, bash_tool]
tools_by_name = {tool.name: tool for tool in tools}


@task
def call_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])


def _extract_llm_text(raw_response) -> str:
    """
    兼容不同 LLM 的返回格式，提取文本内容。

    Args:
        raw_response: LLM 的原始响应对象

    Returns:
        str: 提取的文本内容
    """
    if not raw_response or not hasattr(raw_response, "content"):
        logger.warning("LLM 响应为空或没有 content 属性")
        return ""

    content = raw_response.content

    # Gemini 格式：content 是列表 [{'type': 'text', 'text': '...'}]
    if isinstance(content, list):
        if len(content) == 0:
            logger.warning("Gemini 返回空列表")
            return ""
        first_item = content[0]
        if isinstance(first_item, dict):
            return first_item.get("text", "")
        elif isinstance(first_item, str):
            return first_item
        else:
            logger.warning(f"未知的 Gemini content 类型：{type(first_item)}")
            return str(first_item)

    # Qwen/OpenAI 格式：content 是字符串
    elif isinstance(content, str):
        return content

    # 其他未知格式，尝试转换
    else:
        logger.warning(f"未知的 content 类型：{type(content)}，尝试转换为字符串")
        return str(content)


def invoke_llm(llm_instance, prompt: str) -> str:
    """
    统一封装 LLM 调用，自动处理不同 API 的返回格式。

    Args:
        llm_instance: LLM 实例（如 ChatOpenAI、ChatGoogleGenerativeAI）
        prompt: 提示词

    Returns:
        str: 提取的文本内容
    """
    raw_response = llm_instance.invoke(prompt)
    return _extract_llm_text(raw_response)


def extract_json(text: str, ensure_type=None):
    # 注意这个函数复制于wkf\src\scsagent\core\debug.py，如果修改了请同步至原处
    """
    从文本中提取最外层的 JSON（对象或数组），支持去除 Markdown 代码块。

    Args:
        text: 输入字符串
        ensure_type: 可选，'object' 或 'array'，用于校验类型

    Returns:
        dict 或 list（根据 JSON 内容）

    Raises:
        ValueError: 无法提取或解析 JSON，或类型不符
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    # 去除 Markdown 代码块（支持 ```json、```JSON、``` 等）
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text.strip(), re.IGNORECASE)
    if match:
        text = match.group(1)
    else:
        # 如果没有代码块，就取整个文本（但去掉首尾空白）
        text = text.strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e

    # 类型校验（可选）
    if ensure_type == "object":
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object (dict), but got something else")
    elif ensure_type == "array":
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array (list), but got something else")
    elif ensure_type is not None:
        raise ValueError("ensure_type must be 'object', 'array', or None")

    return data


def generate_solutions(final_goal, code, error):
    solutions_prompt = f"""任务：请针对这个错误给出解决方案，并按要求返回JSON数组。
要求：确保每个方案的描述包含了所有执行细节。返回一个JSON数组。

### 用户目标
{final_goal}
### 任务代码
```
{code}
```
### 错误信息
{error}

### 输出示例
["方案1的详细描述", "方案2的详细描述", "方案3的详细描述", ...]

### 输出格式：请只返回纯净的JSON数组，不要有```json等格式。
"""
    response_text = invoke_llm(llm, solutions_prompt)

    try:
        solutions_list = extract_json(response_text)
        if not isinstance(solutions_list, list):
            logger.warning(f"extract_json 返回的不是列表：{solutions_list}")
            solutions_list = []
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"解析 solutions JSON 失败：{e}")
        solutions_list = []

    solutions = [{"description": e, "status": "pending"} for e in solutions_list]
    return solutions


def check_task_done(task, code, output):
    if "Error" in output or "Traceback" in output or "Exception" in output:
        return False, "执行输出中包含错误或异常信息"
    # 构造判断 prompt
    check_task_done_prompt = f"""## 任务：请根据以下信息判断，代码是否成功完成了用户要求的任务，按这个格式输出JSON串，不要输出```格式或其他解释文本：
{{"task_done": true, "reason": 'xxx'}} 或 {{"task_done": false, "reason": 'xxx'}}

## 判断标准：
成功的标准：
代码运行未报错（仅有警告信息warnings视作未报错！），并且使用task中的工具完成了任务。
常见失败的例子：
- output 中有报错信息（如 Traceback, Error, Exception等）

## 用户任务：
{task}

## 当前执行代码：
{code}

## 执行结果：
{output}

## 输出要求
只返回JSON字符串本身，不要输出```格式或其他解释文本"""
    response_text = invoke_llm(qwen3max, check_task_done_prompt)
    logger.info("<check_task_done>" + str(response_text) + "</check_task_done>")
    # 解析 LLM 返回内容
    try:
        result = extract_json(response_text.strip())
        task_done = bool(result["task_done"])
        reason = result["reason"]
    except json.JSONDecodeError:
        logger.info("check_task_done 解析失败")
        task_done = False  # 解析失败视为未完成
        reason = "check_task_done 流程解析错误"
    return task_done, reason


def debug(state: AgentState):
    """Main debugging agent orchestrator with iterative solution exploration."""
    final_goal = state["task"]
    max_attempts = 6
    docker_manager = get_docker_manager()

    # Initialize with latest user code
    task_code = state["codes"][-1]
    task_output = python_repl_tool.invoke(task_code)

    for attempt in range(max_attempts):
        docker_manager.clear_directory()
        logger.info(f"[<Attempt {attempt + 1}/{max_attempts}>]")

        # Generate solution candidates for current error state
        solutions = generate_solutions(final_goal, task_code, task_output)
        logger.info(f"<solutions>{solutions}</solutions>")

        # Try each solution candidate with state isolation
        result = _explore_solutions(
            final_goal=final_goal,
            original_code=task_code,  # 传递初始状态用于方案隔离
            original_output=task_output,
            solutions=solutions,
        )

        if result["done"]:
            return _format_response(True, result["task_code"], result["task_output"])

        # Update state for next iteration using LAST modified state
        task_code = result["task_code"]
        task_output = result["task_output"]

        if attempt == max_attempts - 1:
            logger.info(f"[<DebugAgent reached attempt limit: {max_attempts}>]")

        # TODO 改进为判断之前的目标是否有与现在的错误相同的，在当时的尝试方案历史上进行追加修改
    return _format_response(False, task_code, task_output)


def _explore_solutions(
    final_goal: str, original_code: str, original_output: str, solutions: list
) -> dict:
    """Explore solution candidates with strict state isolation per solution."""
    task_code = original_code
    task_output = original_output
    sol_idx = 0

    # Use while loop to support 'continue' decision semantics (retry current solution)
    while sol_idx < len(solutions):
        solution = solutions[sol_idx]
        solution["status"] = "running"
        logger.info(f"<solution-{sol_idx}>: {solution}")

        # Plan executable steps for this solution
        steps = _plan_solution_steps(final_goal, task_code, task_output, solution)
        logger.info(f"<steps>{steps}</steps>")

        # Execute planned steps
        exec_result = _execute_solution_steps(
            final_goal=final_goal,
            task_code=task_code,
            task_output=task_output,
            steps=steps,
        )

        # Update state with execution results
        task_code = exec_result["task_code"]
        task_output = exec_result["task_output"]

        # Check for immediate task completion
        if exec_result["done"]:
            return exec_result

        # Evaluate solution progress and decide next action
        decision = _evaluate_solution_progress(
            original_error=original_output,
            current_steps=steps,
            execution_history=exec_result["compressed_context"],
            solutions=solutions,
        )
        logger.info(f"<decision>{decision}</decision>")

        next_action = decision.get("next", "next_solution")
        if next_action == "exit":
            # Exit entire solution exploration (but continue outer attempt loop)
            return {"done": False, "task_code": task_code, "task_output": task_output}
        elif next_action == "continue":
            # Retry CURRENT solution (do not increment index) - matches original semantics
            logger.info(f"<decision>continue: retrying solution-{sol_idx}</decision>")
            continue  # Re-attempt same solution index
        else:  # "next_solution" or unknown
            sol_idx += 1  # Move to next solution candidate
            if sol_idx < len(solutions):
                task_code = original_code
                task_output = original_output

    # All solutions exhausted without success
    return {"done": False, "task_code": task_code, "task_output": task_output}


def _plan_solution_steps(
    final_goal: str, task_code: str, task_output: str, solution: dict
) -> List[str]:
    """将解决方案描述转换为可执行步骤序列（带 JSON 解析重试）。"""
    base_prompt = f"""# 任务：请将解决方案分成多个步骤返回：
## 要求
1. 步骤要求：每个步骤只执行一个动作，操作内容必须包括方案中提到的所有细节。  
2. 落到实处：确保修改作用到实际环境中，或者修改实际的代码，而不是只有可行性验证。  
3. 最终验证：在计划的结尾根据执行历史决定是否对用户代码进行修改，之后运行用户代码并验证方案结果。

## 环境信息
你正在一个基于 Ubuntu 22.04 的 Docker 容器中工作。该容器具有以下关键特性：
- 当前用户是普通用户 `STOmics_test`（UID=1013），执行权限命令必须使用sudo；
- conda 环境目录为 /home/STOmics_test/.conda/envs/，执行前需先运行 conda env list 检查可用环境：若存在非 py310 的环境则优先使用，仅当无其他环境时才使用 py310（已预装 Cython、scanpy、leidenalg、umap-learn）；
- 系统已预装：git、build-essential、python3-dev、wget、vim、curl、iputils-ping 等基础开发工具；
- 用户代码应该是一个Python文件，不能有Jupyter Notebook的语法。

## 主线用户任务
{final_goal}

## 用户代码
```python
{task_code}
```

## 报错信息
{task_output}

## 解决方案
{solution['description']}

## 示例输出1
["1. 将用户代码中原来的<code snippet>修改为<code snippet1>", "2. 使用python_repl_tool工具运行新的用户代码进行验证"]

## 示例输出2
["1. 使用bash_tool工具执行命令`pip show xxx`检查xxx的版本", "2. 使用bash_tool工具执行命令xxx安装xxx包", "3. 使用python_repl_tool工具运行用户代码进行验证"]

输出格式：请只返回纯净的JSON数组（如 ["1. ...", "2. ..."]），不要包含任何其他文字、注释或Markdown包裹（如 ```json）。
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_response = invoke_llm(llm, base_prompt)
            steps = extract_json(raw_response)

            # 校验：必须是 list，且每个元素是字符串
            if not isinstance(steps, list):
                raise ValueError("Response is not a list")
            if not steps:
                raise ValueError("Step list is empty")
            if not all(isinstance(step, str) and step.strip() for step in steps):
                raise ValueError("All steps must be non-empty strings")

            # 成功解析，持久化并返回
            solution["description"] = str(steps)
            return steps

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"[_plan_solution_steps] JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(
                    "[_plan_solution_steps] All retries failed. Falling back to empty step list."
                )
                # 可选：抛出异常 or 返回空列表（这里选择空列表以避免中断流程）
                solution["description"] = "[]"
                return []

            # 追加纠错提示
            base_prompt += (
                "\n\n注意：你上次的回复无法被解析为合法的 JSON 数组。"
                '请严格只输出一个 JSON 数组，格式如：["1. 第一步", "2. 第二步"]，不要任何其他内容。'
            )


def _execute_solution_steps(
    final_goal: str,
    task_code: str,
    task_output: str,
    steps: list,
) -> dict:
    """Execute planned steps with error recovery and context compression."""
    event_limit = len(steps) * 2 if steps else 0

    compressed_context = []
    current_output = task_output
    current_run = None
    action_type = ""

    for event_idx in range(event_limit):
        # Analyze current state and determine next action
        analysis = _analyze_execution_state(
            steps=steps,
            context_history=compressed_context,
            current_code=task_code,
            last_action=current_run,
            last_output=current_output,
            action_type=action_type,
        )
        logger.info(f"<analysis>{analysis}</analysis>")

        # Decide concrete action to execute
        action = _decide_next_action(
            current_code=task_code,
            last_action=current_run,
            last_output=current_output,
            analysis=analysis,
            action_type=action_type,
        )
        current_run = action["content"]
        action_type = action["type"]

        # 执行动作，只获取输出
        current_output = _execute_action(action_type, current_run)

        # 显式更新 task_code（仅当是 task_code 类型）
        if action_type == "task_code":
            task_code = current_run  # ← 主线代码被替换
            # 检查任务是否完成
            task_done, _ = check_task_done(final_goal, task_code, current_output)
            if task_done:
                return {
                    "done": True,
                    "task_code": task_code,
                    "task_output": current_output,
                    "compressed_context": compressed_context,
                }

        # Compress execution context for history
        compressed_action = _compress_execution_context(
            analysis=analysis, executed_action=current_run, result=current_output
        )
        compressed_context.append(compressed_action)
        logger.info(f"<compressed_action>{compressed_action}</compressed_action>")

    return {
        "done": False,
        "task_code": task_code,
        "task_output": current_output,
        "compressed_context": compressed_context,
    }


def _analyze_execution_state(
    steps: list,
    context_history: list,
    current_code: str,
    last_action: str,
    last_output: str,
    action_type: str,
) -> str:
    """Generate error analysis based on execution history."""
    prompt = f"""# 任务：请根据当前执行轨迹和上次的执行结果，分析错误的原因并给出解决方案。你必须沿着当前的规划方向进行探索，不要再规划新的方向。

## 当前规划
{steps}

## 执行轨迹
{str(context_history[:-1]) if len(context_history) > 1 else '[]'}

## 最新用户任务代码
```python
{current_code}
```

{f'## 上次执行\n{last_action}' if action_type in ('bash', 'test_code') else ''}

## 上次执行结果
{last_output}

现在开始执行任务：根据当前执行轨迹和上次的执行结果，分析上次执行错误的原因并给出解决方案。你必须沿着当前的规划方向进行探索，不要再规划新的方向。
"""
    return invoke_llm(llm, prompt)


def _decide_next_action(
    current_code: str,
    last_action: str,
    last_output: str,
    analysis: str,
    action_type: str,
) -> dict:
    """从分析结果中选择一个具体动作执行，并严格指定动作类型（带 JSON 解析重试）。"""
    last_execution = (
        f"## 上次执行\n{last_action}"
        if action_type in ("bash", "test_code") and last_action
        else ""
    )

    base_prompt = f"""请从<分析过程>中选择最合适的一个方案，按照如下要求返回。
要求：
1. 只选择方案，不要自己重新规划。
2. **类型必须严格匹配内容**：
   - 如果返回的是用于**完成用户主线任务的完整 Python 代码**，则 type 必须为 `"task_code"`；
   - 如果返回的是用于**验证、调试或测试的小段 Python 代码**（非最终任务代码），则 type 必须为 `"test_code"`；
   - 如果返回的是 **bash 命令**（如 pip install、ls、sudo 等），则 type 必须为 `"bash"`；
3. 不得将 Python 代码标记为 "bash"，也不得将 bash 命令标记为 "task_code" 或 "test_code"。
4. **只输出一个合法的 JSON 对象，不要任何其他文字、注释或 Markdown 包裹（如 ```json）**。

## 用户任务代码
```python
{current_code}
```

{last_execution}

## 上次执行结果
{last_output}

## 分析过程
<分析过程>{analysis}</分析过程>

## 输出格式（严格遵守）
{{"content": "string", "type": "task_code|test_code|bash"}}
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_response = invoke_llm(llm, base_prompt)
            result = extract_json(raw_response)

            # 额外校验：确保必要字段存在且类型合法
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            if "content" not in result or "type" not in result:
                raise ValueError("Missing 'content' or 'type'")
            if result["type"] not in {"task_code", "test_code", "bash"}:
                raise ValueError(f"Invalid type: {result['type']}")
            if not isinstance(result["content"], str) or not result["content"].strip():
                raise ValueError("Content is empty or not a string")

            return result

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"[_decide_next_action] JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                # 最后一次失败，抛出异常或返回安全默认值
                logger.error(
                    "[_decide_next_action] All retries failed. Falling back to test_code with error message."
                )
                raise

            # 构造更强的纠错提示
            base_prompt += f"""\n\n注意：你上次的回复无法被解析为合法 JSON。请严格只输出如下格式的 JSON 对象，不要任何其他内容：
{{"content": "string", "type": "task_code|test_code|bash"}}"""


def _execute_action(action_type: str, action_content: str) -> str:
    """Execute the given action and return its output string."""
    if action_type == "task_code":
        return python_repl_tool.invoke(action_content)
    elif action_type == "bash":
        return bash_tool.invoke(action_content)
    elif action_type == "test_code":
        return python_repl_tool.invoke(action_content)
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def _compress_execution_context(
    analysis: str, executed_action: str, result: str
) -> str:
    """Summarize execution event for history compression."""
    prompt = f"""请你对上下文进行压缩，简要描述清楚本次做出了什么样的代码修改或者命令，结果是什么。
                    
## 上次结果的分析
{analysis}

## 本次执行的命令
{executed_action}

## 本次的执行结果或者错误
{result}

现在输出压缩的内容："""
    return invoke_llm(llm, prompt).strip()


def _evaluate_solution_progress(
    original_error: str,
    current_steps: List[str],
    execution_history: List[str],
    solutions: List[Dict],
) -> Dict[str, str]:
    """基于执行轨迹评估方案进展，并决定下一步策略（带 JSON 解析重试）。"""
    base_prompt = f"""## 任务
我当前正在按照下面这个方案来解决用户代码的报错。你需要根据执行轨迹判断下一步的动作，并返回标准JSON串。

## 原始报错信息
{original_error}

## 全部解决方案
{str(solutions)}

## 当前解决方案
{str(current_steps)}

## 执行轨迹
{str(execution_history)}

**判断规则**
- 如果执行轨迹显示已经解决了[原始报错信息]（绕过也可以），需要结束任务或者重新规划解决方案：
    返回 {{"next": "exit", "reason": "具体原因"}}
- 如果当前解决方案尚未执行完，仍有探索空间：
    返回 {{"next": "continue", "reason": "具体原因"}}
- 如果当前解决方案已失败或无望，应转而尝试其他待处理方案：
    返回 {{"next": "next_solution", "reason": "具体原因"}}

**输出要求**
1. 只返回一个合法的 JSON 对象，不要任何其他文字、注释或 Markdown 包裹（如 ```json）。
2. "next" 字段值必须是以下三者之一：`"exit"`、`"continue"`、`"next_solution"`。
3. "reason" 字段必须是非空字符串，说明判断依据。
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_response = invoke_llm(llm, base_prompt)
            result = extract_json(raw_response)

            # 校验结构
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            if "next" not in result or "reason" not in result:
                raise ValueError("Missing required keys: 'next' and/or 'reason'")
            if result["next"] not in {"exit", "continue", "next_solution"}:
                raise ValueError(f"Invalid 'next' value: {result['next']}")
            if not isinstance(result["reason"], str) or not result["reason"].strip():
                raise ValueError("'reason' must be a non-empty string")

            return result

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"[_evaluate_solution_progress] JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(
                    "[_evaluate_solution_progress] All retries failed. Falling back to 'next_solution'."
                )
                # 安全兜底：默认切换到下一个方案，避免卡死
                return {
                    "next": "next_solution",
                    "reason": "Failed to parse evaluation decision after retries; defaulting to next solution.",
                }

            # 追加纠错提示
            base_prompt += (
                "\n\n注意：你上次的回复无法被解析为合法 JSON 对象。"
                "请严格只输出如下格式的 JSON："
                '{"next": "exit|continue|next_solution", "reason": "具体原因"}'
                "不要包含任何其他内容。"
            )


def _format_response(done, code: str, output: str) -> dict:
    """Format debugging result."""
    return {"done": done, "codes": [code], "outputs": [output]}


@log_func
def inspect_data(state: AgentState) -> AgentState:
    prompt = f"""从以下用户任务中识别所有数据文件路径，仅输出标准 JSON 数组格式（即 ["path1", "path2"]），不要输出代码块符号```和其他文本：

任务：
{state['task']}"""

    # 调用大模型
    res = invoke_llm(llm, prompt).strip()

    # 尝试解析 JSON
    try:
        file_paths = json.loads(res)
        if not isinstance(file_paths, list):
            raise ValueError("输出不是数组")
    except (json.JSONDecodeError, ValueError) as e:
        logger.info(f"解析失败或格式错误：{e}")
        file_paths = []

    # ipdb.set_trace()
    container_data_path = os.path.join(
        state.get("container_work_dir"), "input"
    )  # e.g., /workspace/input
    logger.info(f"container_data_path: {container_data_path}")
    host_data_path = state.get("host_data_path")

    normalized_paths = []
    for path in file_paths:
        # 仅当路径以 container_data_path + 分隔符 开头时才替换（确保是子路径）
        if path.startswith(container_data_path + os.sep):
            rel_part = path[len(container_data_path) + 1 :]  # 跳过分隔符
            new_path = os.path.join(host_data_path, rel_part)
            normalized_paths.append(new_path)
        else:
            # 不匹配前缀的路径保留原样（可选：也可跳过或报错）
            normalized_paths.append(path)

    output_lines = []
    for i, file in enumerate(normalized_paths):
        if file.endswith(".h5ad"):
            res = inspect_data_tool.invoke(file)
            output_lines.append(f"{i}. {file_paths[i]}")
            output_lines.append(f"{res}")

    result_text = "\n".join(output_lines)
    return {"data_info": result_text}


@log_func
def ensure_llm_summaries(db, model, tool_name: str) -> None:
    """确保该工具下所有文档都有 llm_summary，缺失的则生成并更新"""
    # 查询所有缺少 llm_summary 的文档
    sql_missing = """
        SELECT id, doc 
        FROM docs 
        WHERE tool_id = (SELECT id FROM tools WHERE name = %s)
          AND (llm_summary IS NULL OR TRIM(llm_summary) = '')
    """
    missing_docs = db.execute_query(sql_missing, (tool_name,))

    for doc_id, doc_content in missing_docs:
        try:
            logger.info(f"正在为文档 ID {doc_id} 生成 LLM 摘要...")
            summary = summary_doc(model, doc_content)
            update_sql = "UPDATE docs SET llm_summary = %s WHERE id = %s"
            db.execute_update(update_sql, (summary, doc_id))
            logger.info(f"成功更新文档 ID {doc_id} 的 llm_summary")
        except Exception as e:
            logger.error(f"生成或更新摘要失败（文档 ID {doc_id}）: {e}")


@log_func
def rank_docs_by_relevance(
    model, db, tool_name: str, user_request: str, n_doc_limit: int = 1
) -> List[str]:
    """根据用户需求对文档摘要打分排序，过滤低相关性文档，返回 top-k 原始内容"""
    # 获取所有有效文档
    sql_summaries = """
        SELECT id, llm_summary 
        FROM docs 
        WHERE tool_id = (SELECT id FROM tools WHERE name = %s)
          AND llm_summary IS NOT NULL 
          AND TRIM(llm_summary) != ''
    """
    db_res = db.execute_query(sql_summaries, (tool_name,))

    if not db_res:
        logger.warning(f"未找到工具 '{tool_name}' 的有效文档摘要")
        return []

    summaries_text = "\n\n".join(
        f"- 文档ID: {row[0]}\n  摘要: {row[1]}" for row in db_res
    )

    # 更清晰的评分定义 + JSON 输出要求
    ranking_prompt = f"""你是一个文档相关性评估助手。请根据用户需求，对以下文档摘要进行相关性评分（0.0 到 1.0）。

# 相关性定义：
- 0.0~0.3：完全无关，或仅含通用背景，无法用于当前任务 → **应忽略**
- 0.4~0.6：部分相关，但缺少关键细节
- 0.7~0.8：高度相关，包含可复用步骤
- 0.9~1.0：非常相关，几乎可直接解决问题

# 用户需求：
{user_request}

# 候选文档：
{summaries_text}

# 输出要求：
- 仅输出一个 JSON 数组，不要任何其他文字。
- 每个元素为 {{"id": 文档ID, "score": 分数}}。
- 分数保留两位小数。
- 按 score 从高到低排序。
- **即使所有文档都无关，也请如实打分（可能全 ≤0.3）**。

示例输出：
[{{"id": 5, "score": 0.95}}, {{"id": 2, "score": 0.78}}]
"""

    try:
        llm_response = invoke_llm(qwen3max, ranking_prompt)
        ranked_list = extract_json(llm_response)  # 返回 list of dict
        logger.info(f"文档相关性排序:{ranked_list}")

        if not isinstance(ranked_list, list):
            raise ValueError("extract_json did not return a list")

        # 解析并过滤：只保留 score > 0.3 的文档
        valid_docs = []
        seen_ids = set()
        for item in ranked_list:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("id")
            score = item.get("score")
            if (
                isinstance(doc_id, int)
                and isinstance(score, (int, float))
                and score > 0.3
                and doc_id not in seen_ids
            ):
                valid_docs.append((doc_id, float(score)))
                seen_ids.add(doc_id)

        # 按分数降序（LLM 应已排好，但保险起见）
        valid_docs.sort(key=lambda x: x[1], reverse=True)

        # 取前 n_doc_limit 个 ID
        selected_ids = [doc_id for doc_id, _ in valid_docs[:n_doc_limit]]

        if not selected_ids:
            logger.info("所有文档相关性 ≤ 0.3，无有效文档返回")
            return []

    except Exception as e:
        logger.error(f"LLM 排序或解析失败: {e}")
        # 回退策略：尝试用原始 rate 排序，但也要过滤？这里保守返回空或默认
        # 也可选择不过滤回退结果，但建议保持一致性 → 返回空
        return []

    # 查询原始文档内容，保持顺序
    placeholders = ",".join(["%s"] * len(selected_ids))
    sql_docs = f"""
        SELECT id, doc 
        FROM docs 
        WHERE id IN ({placeholders})
        ORDER BY FIELD(id, {','.join(map(str, selected_ids))})
    """
    doc_results = db.execute_query(sql_docs, selected_ids)

    # 返回 [{"id": ..., "content": ...}, ...]
    return [{"id": row[0], "content": row[1]} for row in doc_results]


@log_func
def retrieve_docs(state: AgentState, n_doc_limit: int = 1) -> AgentState:
    """主入口：检索并返回与当前任务最相关的文档"""
    tool_name = state["tool"]
    user_request = state["task"]

    with Database() as db:
        # Step 1: 确保所有文档都有 llm_summary
        ensure_llm_summaries(db, llm, tool_name)

        # Step 2: 根据用户需求对文档排序并获取 top-k 原始内容
        docs = rank_docs_by_relevance(
            model=llm,
            db=db,
            tool_name=tool_name,
            user_request=user_request,
            n_doc_limit=n_doc_limit,
        )

    # 日志输出
    for i, doc in enumerate(docs, start=1):
        logger.info(
            f"<检索到的第 {i} 份文档 (ID={doc['id']})>\n{doc['content']}\n</第 {i} 份文档>"
        )

    return {"docs": docs}  # 现在是 list of dict


def get_code_notes():
    return """- 不要有jupyter语法，这是纯python脚本。
- 如果代码中涉及模型训练，请将epoch设置为1，不要使用gpu，只能使用cpu。
- 展示和保存结果：代码要展示并保存必要的中间结果和最终结果，保存文件名要清晰表明其内容。    
"""


@log_func
def generate_code(state: AgentState) -> AgentState:
    # 步骤1 - 提取完整代码
    #     EXTRACT_CODE_PROMPT = """你是代码提取专家。请从以下文档中提取完整的可运行代码：

    # ## 参考文档
    # ```
    # {doc}
    # ```

    # ## 要求：
    # 1. 提取所有相关的代码块
    # 2. 将必要的解释信息写为注释
    # 3. 保持代码的完整性和顺序

    # ## 请提取代码：
    # """

    # 步骤2 - 精简代码
    PRUNE_CODE_PROMPT = """# 任务：你是代码优化专家。请精简以下代码，删除与【用户任务】无关的部分。只输出替换后的完整代码，不要输出```符号和其他任何解释信息：

## 用户任务
{task}    

## 原始代码
```
{code}
```

## 要求
1. 保留与当前分析任务直接相关的核心流程
2. 删除与当前数据类型不匹配的参数设置
3. 保持必要的导入语句和基本错误处理
4. 保持代码逻辑完整性

请只输出代码，不要输出```符号和其他任何解释信息：
"""

    # 步骤3 - 参数替换
    REPLACE_PARAMS_PROMPT = """# 任务：你是代码适配专家。请将以下代码中的路径、参数字段替换为用户提供的具体值。请只输出替换后的完整代码，不要输出```符号和其他任何解释信息：

## 替换规则
1. 将代码中的输入输出文件路径替换为以下真实路径
2. 根据真实数据信息调整相关参数字段（如细胞类型、基因数量等）
3. 如果代码中要求的某个字段【输入数据信息】中没有提供，则修改或者删除对应的代码语句。
4. 保持代码其他部分不变

## 注意事项
{code_notes}

## 【输入数据信息】
{task}
{data_info}

## 输出结果路径
{container_work_dir}

## 待处理代码
```
{code}
```

## 输出要求
请只输出替换后的完整代码，不要输出```符号和其他任何解释信息"""

    # 获取当前状态信息
    task = state["task"]
    docs = state["docs"]
    data_info = state["data_info"]
    container_work_dir = state["container_work_dir"]

    if not docs:
        # 没有检索到相关文档，直接让 LLM 从头生成代码
        logger.info("未检索到相关文档，将基于任务和数据信息直接生成代码。")

        direct_gen_prompt = f"""# 任务：请根据用户任务和输入数据信息，编写一个完整的、可运行的 Python 脚本。
        
## 要求
1. 不要有 Jupyter Notebook 语法（如 %matplotlib、display 等），必须是纯 Python 脚本。
2. 如果涉及模型训练，请将 epoch 设置为 1，仅使用 CPU。
3. 所有结果（图、表、中间文件等）必须保存到指定输出目录，文件名需清晰表明内容。
4. 代码应能独立运行，包含必要的导入和错误处理。

## 用户任务
{task}

## 输入数据信息
{data_info}

## 输出结果路径
{container_work_dir}

## 注意事项
{get_code_notes()}

请只输出完整的 Python 代码，不要任何解释、Markdown 代码块（如 ```python）或其他文本。
"""
        final_code = invoke_llm(llm, direct_gen_prompt)

    else:
        doc_content = "\n\n".join(d["content"] for d in state["docs"])

        # 步骤 1: 提取完整代码
        # extract_prompt = EXTRACT_CODE_PROMPT.format(doc=doc)
        # extracted_code = llm.invoke(extract_prompt).content

        # 步骤 2: 精简代码
        prune_prompt = PRUNE_CODE_PROMPT.format(task=task, code=doc_content)
        pruned_code = invoke_llm(llm, prune_prompt)

        # 步骤 3: 参数替换
        replace_prompt = REPLACE_PARAMS_PROMPT.format(
            code=pruned_code,
            data_info=data_info,
            container_work_dir=container_work_dir,
            task=task,
            code_notes=get_code_notes(),
        )
        final_code = invoke_llm(llm, replace_prompt)

    # 更新状态
    return {
        "codes": [final_code],
    }


def debug_workflow():
    workflow = StateGraph(AgentState)

    # 节点定义
    workflow.add_node("inspect_data", inspect_data)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("debug", debug)

    # 流程连接
    workflow.set_entry_point("inspect_data")
    workflow.add_edge("inspect_data", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "generate_code")
    workflow.add_edge("generate_code", "debug")
    workflow.add_edge("debug", END)

    return workflow.compile()
