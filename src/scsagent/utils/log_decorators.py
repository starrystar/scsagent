from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import functools
from langchain_openai import ChatOpenAI
from typing import Any, Callable, Type, TypeVar, Optional
import re


logger = logging.getLogger(__name__)
llm = ChatOpenAI(
    # model="qwen3-coder-plus",
    model="qwen3-max",
    base_url="",
    api_key="",
)

def log_io(
    func: Callable = None, *, log_input: bool = True, log_output: bool = True
) -> Callable:
    """
    A decorator that optionally logs input and/or output.
    Can be used as:
        @log_io              -> logs both (original behavior)
        @log_io(log_input=False) -> logs only output
    """

    # 支持两种写法：@log_io  和  @log_io(log_input=False)
    if func is None:
        return lambda f: log_io(f, log_input=log_input, log_output=log_output)


    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        logger.info(f"<{func_name}>")

        # Log input if enabled
        if log_input:
            params = ", ".join(
                [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
            )
            logger.info(f"Input parameters: {params}")

        result = func(*args, **kwargs)

        # Log output if enabled
        if log_output:
            logger.info(f"<{func_name}_return>\n{result}\n</{func_name}_return>")
            
        logger.info(f"</{func_name}>")
        return result

    return wrapper



def log_func(func: Callable) -> Callable:
    """
    A decorator that logs the input parameters and output of a tool function.

    Args:
        func: The tool function to be decorated

    Returns:
        The wrapped function with input/output logging
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Log input parameters
        func_name = func.__name__
        logger.info(f"<{func_name}>")
        # Execute the function
        result = func(*args, **kwargs)
        logger.info(f"</{func_name}>")
        return result

    return wrapper


def _rule_based_precompress(output: str, max_lines: int = 100) -> str:
    """
    仅做一件事：如果输出超过 max_lines 行，则保留开头和结尾，中间用 [...] 替代。
    不截断长行，不去重，保持原始内容完整性（除长度外）。
    """
    if not output.strip():
        return output

    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output

    keep_head = max_lines // 2
    keep_tail = max_lines - keep_head
    compressed_lines = (
        lines[:keep_head]
        + ["[... output truncated due to length ...]"]
        + lines[-keep_tail:]
    )
    return "\n".join(compressed_lines)


def _looks_like_error(text: str) -> bool:
    """简单判断是否包含异常"""
    return any(
        kw in text
        for kw in ["Traceback", "Error:", "Exception"]
    )


def hybrid_compress_python_output(
    raw_output: str,
    max_raw_chars_for_llm: int = 4000,  # 超过则先裁剪
    max_lines_pre_trim: int = 100,  # 预处理最多保留行数
    max_line_length: int = 150,
) -> str:
    """
    混合压缩 Python REPL 输出：
    1. 若输出过大，先用规则方法粗略裁剪（去重、截断、保留头尾）
    2. 再交由 LLM 进行语义感知的精炼摘要
    """
    if not raw_output or not raw_output.strip():
        return "No output from code execution."

    raw_output = raw_output.strip()

    # 快速短输出：直接返回，避免 LLM 调用开销
    if len(raw_output) <= 300 and not _looks_like_error(raw_output):
        # 简单判断是否为结构化小输出（非错误、非超长）
        if "\n" not in raw_output or len(raw_output.splitlines()) <= 3:
            return raw_output

    # 第一阶段：规则预压缩（防止 LLM 输入过长）
    pre_compressed = _rule_based_precompress(raw_output, max_lines=max_lines_pre_trim)

    # 如果预压缩后仍太大，进一步截断到 LLM 可接受范围
    if len(pre_compressed) > max_raw_chars_for_llm:
        # 保留开头和结尾，中间替换
        half = max_raw_chars_for_llm // 2 - 50
        pre_compressed = (
            pre_compressed[:half]
            + "\n\n[... MIDDLE TRUNCATED DUE TO SIZE ...]\n\n"
            + pre_compressed[-half:]
        )

    # 第二阶段：LLM 智能摘要
    prompt = f"""Compress the following Python REPL output into a minimal but informative summary for an AI reasoning agent.

Rules:
1. If it's an error (Traceback, Exception, etc.), preserve the full last 3 lines of the traceback — they contain the critical error.
2. If it's a large object (list, dict, DataFrame, array, etc.), state its type, length/shape, and show 1–2 example items.
3. If it's short (< 3 lines) or a simple value (number, bool, small string), return it unchanged.
4. NEVER add introductory phrases like "The result is..." or "Output shows...". Just give the compressed content.
5. Keep total length under 200 words.

Output to compress:
{pre_compressed}

Compressed:"""

    try:
        summary = llm.invoke(prompt).content
        return summary if summary else "<empty after LLM compression>"
    except Exception as e:
        # LLM 调用失败时回退到规则压缩结果
        fallback = _rule_based_precompress(raw_output, max_lines=20)
        return f"[LLM compression failed; fallback to rule-based]:\n{fallback}"


def clean_tool_output(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raw_result = func(*args, **kwargs)
        logger.info(f"<压缩前输出>\n{raw_result}\n</压缩前输出>\n")
        return hybrid_compress_python_output(raw_result["output"])

    return wrapper
