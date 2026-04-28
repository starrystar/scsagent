import scanpy as sc
from scipy.sparse import issparse
import pandas as pd
import numpy as np
import subprocess
import logging
import functools
from langchain_core.tools import tool
from scsagent.utils.log_decorators import log_func, log_io, clean_tool_output
from utils.sandbox import get_docker_manager

# Initialize logger
logger = logging.getLogger(__name__)


@tool
def python_repl_tool(code: str) -> str:
    """Execute Python code and return plain text."""
    try:
        docker_manager = get_docker_manager()
        result = docker_manager.execute_python_code(code)
        return str(result)
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"

# @tool
# @log_io
# @clean_tool_output
# def python_repl_tool(code: str):
#     """Execute arbitrary Python code in a restricted REPL-like environment.

#     This function runs the provided Python code string and captures its standard output.

#     Args:
#         code (str): pure code with no markdown ``` format.

#     Returns:
#         dict: A dictionary with two keys:
#             - 'exit_code' (int): 0 if execution succeeded, non-zero otherwise.
#             - 'content' (str): The captured stdout output from the executed code,
#                                or an error message if an exception occurred.
#     """
#     logger.info("Executing Python code")
#     try:
#         # Execute Python code in the existing Docker container
#         docker_manager = get_docker_manager()  # 👈 动态获取
#         result = docker_manager.execute_python_code(code)
#         # result = result["output"]
#         logger.info("Code execution done")
#         return result
#     except BaseException as e:
#         error_msg = f"Failed to execute. Error: {repr(e)}"
#         logger.error(error_msg)
#         return error_msg


@tool
@log_io
@clean_tool_output
def bash_tool(cmd: str):
    """Executes a shell command in a Linux environment and returns the output.
    执行权限命令必须使用sudo。
    """
    logger.info(f"Executing Bash Command")
    try:
        # Execute the command in the Docker container
        docker_manager = get_docker_manager()  # 👈 动态获取
        result = docker_manager.execute_command(cmd)
        return result
    except subprocess.CalledProcessError as e:
        # If command fails, return error information
        error_message = f"Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
        logger.error(error_message)
        return {'exit_code': e.returncode, 'output': f"Stdout: {e.stdout}\nStderr: {e.stderr}"}


@tool
@log_io
def inspect_data_tool(file_path: str) -> str:
    """Inspect and return the internal structure of an AnnData (.h5ad) file.

    This function loads a .h5ad file using scanpy and returns a detailed string report
    about its components, including expression matrix, annotations, embeddings, etc.

    Args:
        file_path (str): Path to the .h5ad file to be inspected.

    Returns:
        str: A formatted string containing all inspection output.
    """
    # Use a list to collect output lines (more efficient than repeated string concat)
    output_lines = []

    def log(msg=""):
        output_lines.append(str(msg))

    try:
        # Load the AnnData object
        adata = sc.read(file_path)

        log("=== AnnData 基本信息 ===")
        log(str(adata))

        # .X: Main expression matrix
        if hasattr(adata, "X") and adata.X is not None:
            log("\n=== .X: 主要表达矩阵（前2行，前5列） ===")
            subset = adata.X[:2, :5]
            if issparse(subset):
                subset_dense = subset.toarray()
            else:
                subset_dense = np.asarray(subset)
            log(str(subset_dense))

            # Global min/max
            try:
                X_min = adata.X.min()
                X_max = adata.X.max()
            except Exception:
                # Fallback for edge cases (e.g., empty matrix)
                X_min = X_max = "N/A"

            log(f"\n=== .X 值范围统计 ===")
            log(f"表达矩阵整体最小值: {X_min}")
            log(f"表达矩阵整体最大值: {X_max}")

            log(f"\n=== .X 类型信息 ===")
            log(f"表达矩阵是否为稀疏格式: {'是' if issparse(adata.X) else '否'}")

        # .obs
        if hasattr(adata, "obs") and not adata.obs.empty:
            log("\n=== .obs: 观测注释（样本属性，前2行，最多显示30列） ===")
            with pd.option_context('display.max_columns', 30, 'display.width', None):
                log(str(adata.obs.head(2)))

        # .var
        if hasattr(adata, "var") and not adata.var.empty:
            log("\n=== .var: 变量注释（基因信息，前2行，最多显示30列） ===")
            with pd.option_context('display.max_columns', 30, 'display.width', None):
                log(str(adata.var.head(2)))

        # .obsm
        if hasattr(adata, "obsm") and adata.obsm:
            log("\n=== .obsm: 多维观测注释 ===")
            for key in adata.obsm:
                log(f"obsm['{key}']: shape={adata.obsm[key].shape}")

        # .varm
        if hasattr(adata, "varm") and adata.varm:
            log("\n=== .varm: 多维变量注释 ===")
            for key in adata.varm:
                log(f"varm['{key}']: shape={adata.varm[key].shape}")

        # .obsp
        if hasattr(adata, "obsp") and adata.obsp:
            log("\n=== .obsp: 观测间的关系矩阵 ===")
            for key in adata.obsp:
                mat = adata.obsp[key]
                log(f"obsp['{key}']: type={type(mat).__name__}, shape={mat.shape}")

        # .varp
        if hasattr(adata, "varp") and adata.varp:
            log("\n=== .varp: 变量间的关系矩阵 ===")
            for key in adata.varp:
                mat = adata.varp[key]
                log(f"varp['{key}']: type={type(mat).__name__}, shape={mat.shape}")

        # .layers
        if hasattr(adata, "layers") and adata.layers:
            log("\n=== .layers: 表达矩阵的不同状态 ===")
            for key in adata.layers:
                layer_mat = adata.layers[key][:5, :5]
                if issparse(layer_mat):
                    disp = layer_mat.toarray()
                else:
                    disp = np.asarray(layer_mat)
                log(f"layers['{key}'] (前5x5):\n{disp}")

        # .uns
        if hasattr(adata, "uns") and adata.uns:
            log("\n=== .uns: 非结构化元数据 ===")
            for key in adata.uns:
                val = adata.uns[key]
                log(f"uns['{key}']: {type(val).__name__}")

    except Exception as e:
        error_msg = f"Error inspecting file '{file_path}': {str(e)}"
        logger.error(error_msg)
        output_lines = [error_msg]

    return "\n".join(output_lines)


@tool
def interact_with_user_tool(prompt: str) -> str:
    """在遇到下列情况时与用户进行交互，传入你想要显示的提示信息，函数将返回用户的输入：
    1. 当你需要用户提供必要信息、做出选择时，请积极使用此工具向用户请求帮助。
    2. 消息历史显示同类型的错误出现多次时，请积极使用此工具向用户请求帮助。

    使用时请尽可能给用户提供充分的信息。

    Args:
        prompt (str): 显示给用户的提示信息（问题或指令），用于引导用户输入。
    Returns:
        str: 用户输入的内容（去除前后空格）。
    Examples:
        1. 请求用户指出正确的column name
            >>> prompt = "It seems that there is no column named 'cell_type' in the `adata.obs` DataFrame. If you have a different name for the cell type annotations, please provide it. Otherwise, if the cell types are not annotated in the data, we may need to perform clustering or use other methods to assign cell types.
            Could you please confirm the correct column name for the cell types, or let me know if we need to proceed with clustering?"
            >>> input = interact_with_user(prompt)
            >>> input
            'The correct cell type name is `celltype`'
        2.
    """
    logger.info("FunctionCall: interact_with_user")
    prompt = prompt + "\nPlease enter: "
    logger.info(prompt)
    user_input = input(prompt).strip()
    logger.info("user_input:\n" + user_input)
    return user_input