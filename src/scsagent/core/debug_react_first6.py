from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    ToolMessage,
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph, START, MessagesState, END
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langmem.short_term import SummarizationNode, RunningSummary
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnablePassthrough
from typing import TypedDict, Any, Annotated, Optional, List, Dict, Literal
import logging
import re
import json
import ipdb
import os
import glob
import traceback
from datetime import datetime

from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

from scsagent.utils.log_decorators import log_func, log_io
from scsagent.core.tool_call import (
    inspect_data_tool,
    python_repl_tool, 
    bash_tool
)
from utils.sandbox import get_docker_manager
from utils.database import Database
from scsagent.ingestion.get_doc import summary_doc


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

for t in tools:
    print(type(t))
    print("name =", t.name)
    print("description =", t.description)
    print("args_schema =", getattr(t, "args_schema", None))
    print("tool_call_schema =", getattr(t, "tool_call_schema", None))
    print("-" * 50)

tools_by_name = {tool.name: tool for tool in tools}


class CustomAgentState(AgentState):
    task: str
    tool: str
    host_data_path: str
    container_work_dir: str
    data_info: str
    docs: List[Dict[str, Any]]
    codes: list[str]
    outputs: list[str]
    done: bool


@before_model
def trim_messages(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 8:  # 2条初始消息 + 6条最近对话
        return None  # No changes needed

    # 保留前两条消息(SystemMessage + 初始HumanMessage) + 最近的6条对话
    first_two_msgs = messages[:2]
    conversation_messages = messages[2:]
    recent_messages = conversation_messages[-6:]  # 固定保留6条(3轮Human-AI对话)
    new_messages = first_two_msgs + recent_messages

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}


def debug(state: CustomAgentState):
    agent = create_agent(
        llm,
        tools=tools,
        # state_schema=CustomAgentState,
        # middleware=[trim_messages],
    )

    messages = [
        SystemMessage(
            content=f"""你是一位单细胞 RNA 测序数据(scRNA-seq)分析专家,正在完成以下任务,请执行代码或者命令来修正错误,直到任务完成(注意如果没有Error只有warnings时可以认为已经执行成功了):

【用户任务】
{state["task"]}

【输出目录】
{state["container_work_dir"]}

你可以使用以下工具:
- python_repl_tool: 执行 Python 代码
- bash_tool: 执行 Bash 命令
"""
        ),
        HumanMessage(content="请开始执行任务。"),  # Gemini 需要至少一个 HumanMessage
    ]

    # 调用 LLM 或 agent
    try:
        print("[debug] before agent.invoke", flush=True)
        final_state = agent.invoke({"messages": messages})
        print("[debug] after agent.invoke", flush=True)
    except Exception as e:
        print("[debug] agent.invoke failed:", repr(e), flush=True)
        traceback.print_exc()
        raise

    try:
        print("[debug] before extract_final_result", flush=True)
        result = extract_final_result(state["task"], final_state["messages"])
        print("[debug] after extract_final_result", flush=True)
    except Exception as e:
        print("[debug] extract_final_result failed:", repr(e), flush=True)
        traceback.print_exc()
        raise
    return {
        "done": result["task_achieved"],
        "codes": [result["final_code"]],
        "outputs": [result["final_output"]],
    }


# ==============================
# 5. 最终结果提取
# ==============================


def extract_final_result(task, messages):
    full_memory = "\n".join(getattr(m, "content", str(m)) for m in messages)

    # 构造提取 prompt
    base_prompt = f"""从以下对话中提取两项信息，并输出为纯 JSON：

要求：
- "final_code_or_command": 完整的任务代码或者命令（注意不是用于测试或者验证的片段代码，必须是完成整个任务的完整代码或者命令）
- "latest_run_output": 最新一次执行的输出（字符串）

对话内容：
{full_memory}

仅输出 JSON，不要任何其他内容。
"""

    max_retries = 3
    parsed_result = None

    for attempt in range(max_retries):
        try:
            raw_response = llm.invoke(base_prompt).content
            result = extract_json(raw_response)

            # 校验字段存在性和类型
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            if (
                "final_code_or_command" not in result
                or "latest_run_output" not in result
            ):
                raise ValueError(
                    "Missing required fields: final_code_or_command, latest_run_output"
                )
            if not isinstance(result["final_code_or_command"], str):
                raise ValueError("final_code_or_command must be a string")
            if not isinstance(result["latest_run_output"], str):
                raise ValueError("latest_run_output must be a string")

            parsed_result = result
            break  # 成功则跳出重试

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"[extract_final_result] JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(
                    "[extract_final_result] All retries failed. Using fallback."
                )
                # 使用安全默认值
                parsed_result = {
                    "final_code_or_command": "...",
                    "latest_run_output": "Failed to parse final result after retries.",
                }
            else:
                # 增强提示
                base_prompt += (
                    "\n\n注意：你上次的回复无法被解析为合法 JSON。"
                    "请严格只输出如下格式的 JSON 对象，不要任何其他内容：\n"
                    '{"final_code_or_command": "string", "latest_run_output": "string"}'
                )

    # 调用自定义完成判断
    code = parsed_result["final_code_or_command"]
    output = parsed_result["latest_run_output"]
    task_achieved = check_task_done(task, code, output)

    return {
        "task_achieved": task_achieved,
        "final_code": code,  # 存入 state
        "final_output": output,  # 存入 state
    }


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
    response = qwen3max.invoke(check_task_done_prompt).content
    logger.info("<check_task_done>" + str(response) + "</check_task_done>")
    # 解析 LLM 返回内容
    try:
        result = extract_json(response.strip())
        task_done = bool(result["task_done"])
        reason = result["reason"]
    except json.JSONDecodeError:
        logger.info("check_task_done解析失败")
        task_done = False  # 解析失败视为未完成
        reason = "check_task_done流程解析错误"
    return task_done, reason



@log_func
def inspect_data(state: CustomAgentState) -> CustomAgentState:
    prompt = f"""从以下用户任务中识别所有数据文件路径，仅输出标准 JSON 数组格式（即 ["path1", "path2"]），不要输出代码块符号```和其他文本：

任务：
{state['task']}"""

    # 调用大模型
    res = llm.invoke(prompt).content.strip()

    # 尝试解析 JSON
    try:
        file_paths = json.loads(res)
        if not isinstance(file_paths, list):
            raise ValueError("输出不是数组")
    except (json.JSONDecodeError, ValueError) as e:
        logger.info(f"解析失败或格式错误: {e}")
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
        llm_response = llm.invoke(ranking_prompt).content
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
def retrieve_docs(state: CustomAgentState, n_doc_limit: int = 1) -> CustomAgentState:
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
def generate_code(state: CustomAgentState) -> CustomAgentState:
    # 步骤1 - 提取完整代码
    EXTRACT_CODE_PROMPT = """你是代码提取专家。请从以下文档中提取完整的可运行代码：

    ## 参考文档
    ```
    {doc}
    ```

    ## 要求：
    1. 提取所有相关的代码块
    2. 将必要的解释信息写为注释
    3. 保持代码的完整性和顺序

    ## 请提取代码：
    """

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
        final_code = llm.invoke(direct_gen_prompt).content

    else:
        doc_content = "\n\n".join(d["content"] for d in state["docs"])

        # 步骤1: 提取完整代码
        extract_prompt = EXTRACT_CODE_PROMPT.format(doc=doc_content)
        extracted_code = llm.invoke(extract_prompt).content

        # 步骤2: 精简代码
        prune_prompt = PRUNE_CODE_PROMPT.format(task=task, code=extracted_code)
        pruned_code = llm.invoke(prune_prompt).content

        # 步骤3: 参数替换
        replace_prompt = REPLACE_PARAMS_PROMPT.format(
            code=pruned_code,
            data_info=data_info,
            container_work_dir=container_work_dir,
            task=task,
            code_notes=get_code_notes(),
        )
        final_code = llm.invoke(replace_prompt).content

    # 更新状态
    return {
        "codes": [final_code],
    }


def debug_workflow():
    workflow = StateGraph(CustomAgentState)

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
