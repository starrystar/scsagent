import argparse
import os
import re
import datetime
import subprocess
import logging
import tempfile
import time
import pandas as pd
import atexit
import ipdb
from pathlib import Path
from typing import Dict, Any, TypedDict, Optional, Annotated

# LangGraph & LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

# LLM

llm = ChatOpenAI(
    model="qwen3-max",
    # model="qwen3-coder-plus",
    base_url="",
    api_key="",
)


# ==================== 修改点 2: setup_logger 函数签名 ====================
def setup_logger(max_batch_size: int, log_dir: str):  # ✅ 参数名修改
    """配置模块级 logger

    Args:
        max_batch_size: 本次运行计划处理的最大工具数，用于日志文件命名区分批次
        log_dir: 日志输出目录
    """
    LOG_DIR = Path(log_dir)
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # ✅ 日志文件名中包含 batch_size 便于追溯本次运行的处理规模
    LOG_FILE = LOG_DIR / f"docker_build_batch{max_batch_size}_{timestamp}.log"

    # 清理旧 handler（防止多次调用重复添加）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


class DockerState(TypedDict):
    tool_name: str
    install_info: str
    dockerfile: Optional[str]
    build_success: bool
    build_error: Optional[str]
    retry_count: int
    max_retries: int
    image_tag: str
    failure_reason: Optional[str]


def llm_check_is_pure_python_install(install_info: str) -> bool:
    prompt = f"""请判断如下的安装指令中，是否可以仅通过 pip、conda、git、python 安装 Python 包完成安装。不必考虑可选的依赖。

## 安装说明：
{install_info}

## 示例1
输入为：
Install LIANA  
```r
if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes")

remotes::install_github('saezlab/liana')
```
输出为：
NO

## 示例2
输入为：
```
Installation
Download the code: `git clone https://github.com/epierson9/ZIFA`
Install the package: `cd ZIFA` then `python setup.py install`
```
输出为YES

请仅回答 "YES" 或 "NO"，不要输出任何其他内容、解释或标点。"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip().upper()
        result = "YES" in answer
        logger.info(
            f"🔍 [LLM Check] Install info: {repr(install_info[:100])} → {'✅ YES' if result else '❌ NO'}"
        )
        return result
    except Exception as e:
        logger.error(f"⚠️ LLM 判断是否为纯 Python 安装时出错: {e}")
        # 出错时保守返回 False，避免误构建
        return False


def strip_markdown_code_block(text: str) -> str:
    lines = text.splitlines()
    # 跳过所有以 ``` 开头的行（strip 后判断，避免空格干扰）
    filtered_lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(filtered_lines).strip()


def llm_generate_dockerfile(install_info: str, base_image: str) -> str:
    prompt = f"""你是一位 Docker 专家。请仅根据以下安装信息，生成一个最小且正确的 Dockerfile 来安装指定的软件包。

## 安装信息：
```text
{install_info}
```

## 镜像信息：
- 基础镜像名字为 `{base_image}`，基于Ubuntu 22.04构建，已预装：git、build-essential、python3-dev、wget、vim、curl、iputils-ping 等基础开发工具；
- 基础镜像中已经设置 PATH="/home/STOmics_test/.conda/envs/py310/bin:/opt/miniconda3/bin:${{PATH}}"
- 默认已激活 conda 环境 `py310`，版本为Python 3.10，实际路径为 `/home/STOmics_test/.conda/envs/py310`，且已预装 Cython、scanpy、leidenalg、umap-learn；

## 要求：
1. 必须使用基础镜像`{base_image}`（不得更改）
2. 执行权限命令必须首先通过 `USER root` 命令切换至root用户，并在执行完权限命令后切换回普通用户 `USER STOmics_test`。最后的用户必须是STOmics_test。
3. 所有命令必须以非交互式命令-y运行，例如apt、conda。
4. 如果默认的conda环境的python3.10版本不符合安装要求，请conda create新环境，有如下要求：
    conda create新环境时，必须切换到普通用户身份`USER STOmics_test`创建，安装包需要通过-p参数指定环境，新的环境会创建在/home/STOmics_test/.conda/envs目录下，参考：
        `RUN conda create -p /home/STOmics_test/.conda/envs/<replace_with_new_environment_name> -y python=3.9 && \
        conda run -p /home/STOmics_test/.conda/envs/<replace_with_new_environment_name> pip install --no-cache-dir packages`

    创建新环境后，需要更新PATH，并将新环境写入~/.bashrc，参考：
        `ENV PATH="/home/STOmics_test/.conda/envs/<replace_with_new_environment_name>/bin:${{PATH}}"`
        `RUN echo "conda activate /home/STOmics_test/.conda/envs/<replace_with_new_environment_name>" >> ~/.bashrc`
5. 所有的conda安装命令必须指定环境，例如`RUN conda install -n environment_name -y package_name`
6. 从安装信息中选择最小的安装方式，可选的包一律不要安装。
7. 不要生成 -c 参数的 conda 命令！！必须使用镜像中已经预设的channels包括 conda-forge、bioconda、pkgs/r 和 pkgs/main。

仅输出原始的 Dockerfile 内容，不得包含任何解释性文字 或 Markdown 格式```，请直接从 FROM 指令开始输出："""

    logger.info(f"🧠 [LLM] 生成初始 Dockerfile Prompt:\n{prompt}")
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        dockerfile = strip_markdown_code_block(response.content.strip())
        logger.info(f"✅ [LLM] 初始 Dockerfile:\n{dockerfile}")
        # ipdb.set_trace()
        return dockerfile
    except Exception as e:
        logger.error(f"❌ LLM 生成失败: {e}")
        raise


def llm_fix_dockerfile(
    install_info: str, current_dockerfile: str, error_log: str, base_image: str
) -> str:
    prompt = f"""# 请根据如下的报错信息修复 Dockerfile。

## 安装信息：
{install_info}

## 当前 Dockerfile：
{current_dockerfile}

## 构建错误日志：
{error_log}

## 镜像信息：
- 基础镜像名字为 `{base_image}`，基于Ubuntu 22.04构建，已预装：git、build-essential、python3-dev、wget、vim、curl、iputils-ping 等基础开发工具；
- 基础镜像中已经设置 PATH="/home/STOmics_test/.conda/envs/py310/bin:/opt/miniconda3/bin:${{PATH}}"
- 默认已激活 conda 环境 `py310`，版本为Python 3.10，实际路径为 `/home/STOmics_test/.conda/envs/py310`，且已预装 Cython、scanpy、leidenalg、umap-learn；

## 修复要求：
1. 必须使用基础镜像`{base_image}`（不得更改）
2. 执行权限命令必须首先通过 `USER root` 命令切换至root用户，并在执行完权限命令后切换回普通用户 `USER STOmics_test`。最后的用户必须是STOmics_test。
3. 所有命令必须以非交互式命令-y运行，例如apt、conda。
4. 如果默认的conda环境的python3.10版本不符合安装要求，请conda create新环境，有如下要求：
    conda create新环境时，必须切换到普通用户身份`USER STOmics_test`创建，安装包需要通过-p参数指定环境，新的环境会创建在/home/STOmics_test/.conda/envs目录下，参考：
        `RUN conda create -p /home/STOmics_test/.conda/envs/<replace_with_new_environment_name> -y python=3.9 && \
        conda run -p /home/STOmics_test/.conda/envs/<replace_with_new_environment_name> pip install --no-cache-dir packages`

    创建新环境后，需要更新PATH，并将新环境写入~/.bashrc，参考：
        `ENV PATH="/home/STOmics_test/.conda/envs/<replace_with_new_environment_name>/bin:${{PATH}}"`
        `RUN echo "conda activate /home/STOmics_test/.conda/envs/<replace_with_new_environment_name>" >> ~/.bashrc`
5. 所有的conda安装命令必须指定环境，例如`RUN conda install -n environment_name -y package_name`
6. 从安装信息中选择最小的安装方式，可选的包一律不要安装。
7. 不要生成 -c 参数的 conda 命令！！必须使用镜像中已经预设的channels包括 conda-forge、bioconda、pkgs/r 和 pkgs/main。

仅输出原始的 Dockerfile 内容，不得包含任何解释性文字 或 Markdown 格式```，请直接从 FROM 指令开始输出："""
    logger.info(f"🧠 [LLM] 修复 Dockerfile Prompt:\n{prompt}")
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        fixed_df = strip_markdown_code_block(response.content.strip())
        logger.info(f"✅ [LLM] 修复后的 Dockerfile:\n{fixed_df}")
        return fixed_df
    except Exception as e:
        logger.error(f"❌ LLM 修复失败: {e}")
        raise


def extract_docker_error(build_log: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个经验丰富的 DevOps 工程师，请从以下 Docker 构建日志中精准提取出导致构建失败的根本原因（root cause）。"
                "忽略下载、解压、缓存、元数据加载等正常过程信息，只关注：\n"
                "1. 最后一个以 `ERROR` 或 `error:` 开头的错误段落；\n"
                "2. 与 Python 包安装、依赖解析、编译失败等直接相关的错误；\n"
                "3. 明确指出失败命令、包名、错误消息和建议解决方案的部分；\n"
                "4. 如果存在多阶段构建，只关注失败的阶段（通常标记为 `> [N/M] ...` 且之后紧接 `ERROR`）。\n\n"
                "请以简洁明了的方式输出错误摘要，包括：\n"
                "- 失败的包或命令\n"
                "- 核心错误信息\n"
                "- 官方建议的修复方法（如有）\n\n"
                "不要包含日志中的时间戳、哈希值、镜像层 ID、网络传输细节或成功步骤。",
            ),
            ("user", "{log}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"log": build_log})
    error_summary = response.content if hasattr(response, "content") else str(response)
    return error_summary


def build_docker_image(dockerfile: str, tag: str) -> tuple[bool, str]:
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile)
    try:
        result = subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--builder",
                "limited-builder",
                "-t",
                tag,
                ".",
                "--load",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 900s=15min
            cwd=os.getcwd(),
        )
        logger.info(f"<build_result>\n{str(result)}\n</build_result>")
        # result = subprocess.run([
        #         "docker", "build", "-t", tag, ".",
        #     ],
        #     capture_output=True,
        #     text=True,
        #     timeout=900,
        #     cwd=os.getcwd(),
        # )
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True, ""
        else:
            zip_err = extract_docker_error(result.stderr)
            return False, zip_err
    except Exception as e:
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return False, str(e)


def generate_initial_dockerfile_node(state: DockerState) -> dict:
    df = llm_generate_dockerfile(state["install_info"], BASE_IMAGE)
    return {"dockerfile": df}


def build_image_node(state: DockerState) -> dict:
    success, error = build_docker_image(state["dockerfile"], state["image_tag"])
    return {"build_success": success, "build_error": error if not success else None}


def fix_dockerfile_node(state: DockerState) -> dict:
    logger.info(f'第{state["retry_count"]}次重试')
    fixed_df = llm_fix_dockerfile(
        install_info=state["install_info"],
        current_dockerfile=state["dockerfile"],
        error_log=state["build_error"],
        base_image=BASE_IMAGE,
    )
    return {"dockerfile": fixed_df, "retry_count": state["retry_count"] + 1}


def should_retry(state: DockerState) -> str:
    if state["build_success"]:
        return "end"
    if state["retry_count"] >= state["max_retries"]:
        return "end"
    return "fix"


def create_docker_workflow():
    workflow = StateGraph(DockerState)
    workflow.add_node("generate", generate_initial_dockerfile_node)
    workflow.add_node("build", build_image_node)
    workflow.add_node("fix", fix_dockerfile_node)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "build")
    workflow.add_conditional_edges("build", should_retry, {"end": END, "fix": "fix"})
    workflow.add_edge("fix", "build")

    return workflow.compile()


def process_tool_with_llm_workflow(tool_name: str, install_info: str) -> Dict[str, Any]:
    MAX_RETRIES = 3
    image_tag = f"wh-harbor.dcs.cloud/public-library/agent_{tool_name.lower()}:2.0"

    with tempfile.TemporaryDirectory() as tmp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)

            initial_state = {
                "tool_name": tool_name,
                "install_info": install_info,
                "dockerfile": None,
                "build_success": False,
                "build_error": None,
                "retry_count": 0,
                "max_retries": MAX_RETRIES,
                "image_tag": image_tag,
                "failure_reason": None,
            }

            app = create_docker_workflow()
            final_state = app.invoke(initial_state)

            success = final_state["build_success"]
            reason = None
            if not success:
                reason = (
                    f"FAILED_AFTER_{final_state['retry_count']}_RETRIES: {final_state['build_error']}"
                    if final_state["build_error"]
                    else "UNKNOWN"
                )

            return {
                "tool_name": tool_name,
                "final_dockerfile": final_state["dockerfile"],
                "image_tag": image_tag if success else None,
                "build_success": success,
                "attempts": final_state["retry_count"] + 1,  # 初始 + 修复次数
                "failure_reason": reason,
            }

        except Exception as e:
            logger.error(f"💥 Workflow 异常 ({tool_name}): {e}", exc_info=True)
            return {
                "tool_name": tool_name,
                "final_dockerfile": None,
                "image_tag": None,
                "build_success": False,
                "attempts": 0,
                "failure_reason": f"WORKFLOW_ERROR: {e}",
            }
        finally:
            os.chdir(original_cwd)


def update_dockerbuildresult_from_existing_csv(
    output_csv: str,
    base_image: str,
    max_batch_size: int = None,  # ✅ 原名: max_to_process
    # 用途: 限制从待重试列表中实际执行构建的工具数量，None 表示不限制（全量处理）
):
    global BASE_IMAGE
    BASE_IMAGE = base_image

    if not os.path.exists(output_csv):
        logger.error(f"❌ 输出文件不存在: {output_csv}")
        return

    try:
        df = pd.read_csv(output_csv, encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"❌ 无法读取 CSV: {e}")
        return

    def should_retry(row):
        install = row["installation_info"]
        success = row["build_success"]

        # 跳过 installation_info 为空或空白
        if pd.isna(install) or (isinstance(install, str) and not install.strip()):
            return False

        # 仅当 build_success 未设置（为空）时才重试
        return pd.isna(success)

    retry_mask = df.apply(should_retry, axis=1)
    to_retry_df = df[retry_mask]

    if max_batch_size is not None:
        to_retry_df = to_retry_df.head(max_batch_size)

    if to_retry_df.empty:
        logger.info("✅ 无需要重试的条目")
        return

    logger.info(f"🔄 将重试 {len(to_retry_df)} 个工具...")

    updated_count = 0
    for idx in to_retry_df.index:
        row = df.loc[idx]
        tool_name = row["tool_name"]
        install_info = row["installation_info"]

        logger.info(f"🔧 重试: {tool_name}")

        # ✅ 不再需要检查 install_info 是否为空，should_retry 已保证其有效
        install_lower = install_info.lower()
        if any(
            kw in install_lower for kw in ["pip", "conda", "python"]
        ) and llm_check_is_pure_python_install(install_info):
            result = process_tool_with_llm_workflow(tool_name, install_info)
        else:
            result = {
                "final_dockerfile": None,
                "image_tag": None,
                "build_success": False,
                "attempts": 0,
                "failure_reason": "NOT_PURE_PYTHON_PACKAGE_INSTALL",
            }

        # 更新原行
        for col, val in result.items():
            if col in df.columns:
                df.at[idx, col] = val

        updated_count += 1
        logger.info(f"💾 已更新: {tool_name} → build_success={result['build_success']}")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"🎉 完成，共更新 {updated_count} 行，结果已保存至 {output_csv}")


def update_dockerbuildresult_for_selected_tools(
    output_csv: str,
    base_image: str,
    tool_name_list: list,
    max_batch_size: int = None,
):
    """给定tool_name_list,构建2.0版本的docker镜像"""
    global BASE_IMAGE
    BASE_IMAGE = base_image  # 注意：这里保持与原函数一致的全局变量赋值

    if not os.path.exists(output_csv):
        logger.error(f"❌ 输出文件不存在: {output_csv}")
        return

    try:
        df = pd.read_csv(output_csv, encoding="utf-8-sig")
    except Exception as e:
        logger.error(f"❌ 无法读取 CSV: {e}")
        return

    # 构建要重试的 mask：仅限 tool_name 在给定列表中
    retry_mask = df["tool_name"].isin(tool_name_list)

    to_retry_df = df[retry_mask]

    if max_batch_size is not None:
        to_retry_df = to_retry_df.head(max_batch_size)

    if to_retry_df.empty:
        logger.info("✅ 无需要重试的指定工具")
        return

    logger.info(f"🔄 将重试 {len(to_retry_df)} 个指定工具...")

    # 确保新列存在（避免后续赋值出错）
    for col in ["final_dockerfile2", "image_tag2"]:
        if col not in df.columns:
            df[col] = None

    updated_count = 0
    for idx in to_retry_df.index:
        row = df.loc[idx]
        tool_name = row["tool_name"]
        install_info = row["installation_info"]

        logger.info(f"🔧 重试: {tool_name}")

        install_lower = install_info.lower()
        if any(
            kw in install_lower for kw in ["pip", "conda", "python"]
        ) and llm_check_is_pure_python_install(install_info):
            result = process_tool_with_llm_workflow(tool_name, install_info)
        else:
            result = {
                "final_dockerfile": None,
                "image_tag": None,
                "build_success": False,
                "attempts": 0,
                "failure_reason": "NOT_PURE_PYTHON_PACKAGE_INSTALL",
            }

        # 更新原行中的其他列（除了 final_dockerfile 和 image_tag）
        for col, val in result.items():
            if col in df.columns and col not in ["final_dockerfile", "image_tag"]:
                df.at[idx, col] = val

        # 特殊处理：写入新列
        df.at[idx, "final_dockerfile2"] = result.get("final_dockerfile")
        df.at[idx, "image_tag2"] = result.get("image_tag")

        updated_count += 1
        logger.info(f"💾 已更新: {tool_name} → build_success={result['build_success']}")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"🎉 完成，共更新 {updated_count} 行，结果已保存至 {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Docker images for Python tools using LLM-generated Dockerfiles."
    )
    parser.add_argument(
        "-n",
        "--max-batch-size", 
        type=int,
        default=5,
        help="Maximum number of tools to process in one batch (default: 5)",
    )
    parser.add_argument(
        "-i",
        "--input-excel",
        type=str,
        default="installation_info.xlsx",
        help="Path to input Excel file",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default="docker_build_results.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "-b",
        "--base-image",
        type=str,
        default="wh-harbor.dcs.cloud/public-library/ub22_co310_scanpy_stomicstest:3.0",
        help="Base Docker image to use",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        default="log_docker_build",
        help="Directory to store log files",
    )
    args = parser.parse_args()

    # 初始化日志（必须在 main 前）
    setup_logger(args.max_batch_size, args.log_dir)

    ## 1. 模式1：读取并更新文件docker_build_results.csv
    # update_dockerbuildresult_from_existing_csv(
    #     output_csv=args.output_csv,
    #     base_image=args.base_image,
    #     max_batch_size=args.max_batch_size,
    # )

    # 2. 模式2：只使用列表中指定的tools更新文件output_csv
    tools = [
        "scvi-tools",
        "BackSPIN",
        "PHATE",
        "CellTypist",
        "scGen",
        "cNMF",
        "UnionCom",
        "scFEA",
        "SCCAF",
        "CellHint",
        "Pamona",
        "scETM",
        "SCALEX",
        "CarDEC",
        "Inferelator",
        "graph-sc",
        "MultiMAP",
        "uniPort",
        "ouijaflow",
        "DendroSplit",
        "CellO",
        "HiDeF",
        "iMAP",
        "CellPath",
        "contrastiveVI",
        "spliceJAC",
        "BranchedGP",
        "DeepVelo",
        "locCSN",
        "MEBOCOST",
        "MaxFuse",
        "CeLEry",
        "ENHANCE",
    ]
    update_dockerbuildresult_for_selected_tools(
        output_csv=args.output_csv,
        base_image=args.base_image,
        tool_name_list=tools,
        max_batch_size=args.max_batch_size,
    )
