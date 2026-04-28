import csv
import os
import re
from langchain_openai import ChatOpenAI
import pandas as pd
import requests
import json
from typing import Union, Dict, Any, List, Tuple
import logging
from tqdm import tqdm
from datetime import datetime

from utils.database import Database
from utils.crawler import crawl_page, extract_navigation_links
from scsagent.core.debug import extract_json

#  ==================获取readthedoc文档并存入数据库==================


def _is_url_exists(db, url):
    exists_sql = "SELECT 1 FROM docs WHERE url = %s"
    return db.execute_query(exists_sql, (url,))


def fetch_and_store_readthedocs(db, tools: Union[str, List[str]], log_dir="."):
    if isinstance(tools, str):
        tools = [tools]

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"readthedocs_failed_urls_{timestamp}.csv")
    all_failed_records = []

    print(
        f"🚀 [fetch_and_store_readthedocs] 开始执行，tools={tools}，log_dir={os.path.abspath(log_dir)}"
    )

    for tool in tools:
        print(f"\n{'=' * 50}\n🔧 开始处理工具: {tool}\n{'=' * 50}")
        base_url = f"https://{tool}.readthedocs.io/en/latest/"
        print(f"📄 [{tool}] 文档首页: {base_url}")

        print(f"🕷️ [{tool}] 正在抓取首页...")
        res_base = crawl_page(base_url)
        if not res_base:
            print(f"❌ [{tool}] 首页抓取失败: {base_url}")
            all_failed_records.append(
                {
                    "tool": tool,
                    "url": base_url,
                    "failed_type": "base_url_fetch",
                    "reason": "crawl_page returned None for base_url",
                }
            )
            continue

        try:
            base_exists = _is_url_exists(db, res_base["url"])
            print(f"✅ [{tool}] 首页抓取成功: {res_base['url']}")
            print(f"💾 [{tool}] 首页是否已在数据库中: {bool(base_exists)}")
            if not base_exists:
                insert_sql = """
                    insert into docs(tool_id, url, doc, html)
                    values((select id from tools where name=%s), %s, %s, %s)
                    """
                db.execute_update(
                    insert_sql,
                    (
                        tool,
                        res_base["url"],
                        res_base["text"],
                        res_base["html"],
                    ),
                )
                print(f"✍️ [{tool}] 首页已写入 docs 表。")
            else:
                print(f"⏭️ [{tool}] 首页已存在，跳过写入。")
        except Exception as e:
            print(f"❌ [{tool}] 首页写入失败: {res_base['url']}，错误: {e}")
            all_failed_records.append(
                {
                    "tool": tool,
                    "url": res_base["url"],
                    "failed_type": "storage_error",
                    "reason": str(e),
                }
            )

        nav_links = []
        try:
            print(f"🧭 [{tool}] 正在提取导航链接...")
            nav_links = extract_navigation_links(base_url)
            print(f"🔗 [{tool}] 提取到 {len(nav_links)} 个导航链接。")
        except Exception as e:
            print(f"❌ [{tool}] 导航链接提取失败: {e}")
            all_failed_records.append(
                {
                    "tool": tool,
                    "url": base_url,
                    "failed_type": "nav_extraction",
                    "reason": f"Nav extraction failed: {e}",
                }
            )

        seen = set()
        unique_urls = []
        for url in nav_links:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        print(f"📊 [{tool}] 去重后待抓取页面数: {len(unique_urls)}")

        for idx, url in enumerate(unique_urls, start=1):
            print(f"🌐 [{tool}] ({idx}/{len(unique_urls)}) 正在抓取页面: {url}")
            res = crawl_page(url)
            if not res:
                print(f"❌ [{tool}] 页面抓取失败: {url}")
                all_failed_records.append(
                    {
                        "tool": tool,
                        "url": url,
                        "failed_type": "page_fetch",
                        "reason": "crawl_page returned None",
                    }
                )
                continue

            try:
                page_exists = _is_url_exists(db, res["url"])
                print(f"✅ [{tool}] 页面抓取成功: {res['url']}")
                print(f"💾 [{tool}] 页面是否已在数据库中: {bool(page_exists)}")
                if not page_exists:
                    print(f"✍️ [{tool}] 准备写入页面到 docs 表...")
                    insert_sql = """
                    insert into docs(tool_id, url, doc, html)
                    values((select id from tools where name=%s), %s, %s, %s)
                    """
                    db.execute_update(
                        insert_sql,
                        (
                            tool,
                            res["url"],
                            res["text"],
                            res["html"],
                        ),
                    )
                    print(f"✅ [{tool}] 页面写入成功: {res['url']}")
                else:
                    print(f"⏭️ [{tool}] 页面已存在，跳过写入: {res['url']}")
            except Exception as e:
                print(f"❌ [{tool}] 页面处理失败: {res['url']}，错误: {e}")
                all_failed_records.append(
                    {
                        "tool": tool,
                        "url": res["url"],
                        "failed_type": "storage_error",
                        "reason": str(e),
                    }
                )

    if all_failed_records:
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["tool", "url", "failed_type", "reason"]
            )
            writer.writeheader()
            writer.writerows(all_failed_records)
        print(
            f"\n⚠️ 总共有 {len(all_failed_records)} 条失败记录，涉及 {len(tools)} 个工具。"
        )
        print(f"📁 失败日志已保存到: {log_file}")
    else:
        print(f"\n🎉 全部处理完成，共成功处理 {len(tools)} 个工具。✨")


#  ===========================提取安装信息===========================


def _merge_ranges(
    ranges: List[Tuple[int, int]], min_gap: int = 50
) -> List[Tuple[int, int]]:
    """合并重叠或接近的区间（gap <= min_gap）"""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [list(sorted_ranges[0])]
    for current in sorted_ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1] + min_gap:
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    return [(s, e) for s, e in merged]


def extract_install_info(text: str, tool: str, llm) -> Dict[str, Any]:
    """
    从文档中提取工具的安装说明。
    策略：
      1. 用精确正则匹配安装命令（pip/conda/git/uv等）
      2. 提取每个匹配的前后上下文（±200字符）
      3. 合并重叠区间，生成精简上下文
      4. 交给 LLM 结构化输出
      5. LLM 失败时，用关键词 fallback
    """
    # === Step 1: 精确匹配安装命令 ===
    install_patterns = [
        r"pip\s+install",
        r"conda\s+install",
        r"uv\s+add",
        r"git\s+clone",
        r"docker\s+(pull|run)",
        r"make\s+install",
        r"python\s+setup\.py\s+install",
        r"poetry\s+add",
    ]
    combined_pattern = "|".join(f"({p})" for p in install_patterns)

    matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))
    if not matches:
        return {"success": False, "content": ""}

    # === Step 2: 提取上下文区间（前后200字符）===
    ranges: List[Tuple[int, int]] = []
    for m in matches:
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 200)
        ranges.append((start, end))

    # === Step 3: 合并重叠区间 ===
    merged_ranges = _merge_ranges(ranges, min_gap=50)
    context_parts = [text[s:e] for s, e in merged_ranges]
    condensed_context = "\n\n--- INSTALL CONTEXT ---\n\n".join(context_parts)

    # === Step 4: 用 LLM 从精简上下文中提取结构化结果 ===
    prompt = (
        f"你是一个严谨的生物信息学文档解析器。请从以下**精简上下文**中提取工具 '{tool}' 的安装说明。\n"
        "要求：\n"
        "- 如果上下文包含明确的安装命令（如 pip install, conda install, git clone 等），返回 success=true，并在 content 中返回原文相关片段（不要改写、不要总结）。\n"
        "- 如果上下文仅提到 'install' 但无具体命令（如 'see installation guide'），返回 success=false。\n"
        '- 仅输出一个合法 JSON 对象，格式：{{"success": bool, "content": string}}\n'
        "- 不要包含任何其他文字、解释或 Markdown。\n\n"
        "精简上下文如下：\n"
        f"{condensed_context}\n\n"
        "现在请返回JSON对象："
    )

    try:
        raw_output = llm.invoke(prompt).content.strip()
        result = extract_json(raw_output, ensure_type="object")
        if isinstance(result.get("success"), bool) and isinstance(
            result.get("content"), str
        ):
            return result
        else:
            raise ValueError("Invalid JSON structure")
    except Exception:
        pass  # LLM failed, proceed to fallback

    # === Step 5: Fallback — 直接返回上下文（因为已确认含安装命令）===
    return {"success": True, "content": condensed_context}


def get_install_info_from_readthedocs(tool: str, url: str, llm) -> str:
    """
    从 Read the Docs 风格的文档中提取当前页面（使用规则提取install info片段然后merge，然后llm提取）包括<nav>标签一级链接，指定生信工具的安装说明，哪个提取到信息就直接返回。
    使用 LLM 返回结构化 JSON，并通过 extract_json 安全解析。

    参数:
        tool (str): 工具名称
        url (str): 文档首页 URL
        llm: 支持 .invoke() 的大模型对象（如 LangChain ChatModel）

    返回:
        str: 成功提取的安装说明；若未找到，返回空字符串。
    """

    # Step 1: 尝试当前页面
    try:
        current_page = crawl_page(url)["text"]
        res = extract_install_info(text=current_page, tool=tool, llm=llm)
        if res["success"]:
            return res["content"].strip()
    except Exception:
        pass

    # Step 2: 获取导航链接
    try:
        links = extract_navigation_links(url)
    except Exception:
        links = []

    # Step 3: 筛选含 'install' 的链接（不区分大小写）
    install_candidates = [
        link.strip() for link in links if re.search(r"install", link, re.IGNORECASE)
    ]

    # Step 4: 依次尝试候选链接
    for candidate_url in install_candidates:
        try:
            page_text = crawl_page(candidate_url)["text"]
            res = extract_install_info(page_text, tool=tool, llm=llm)
            if res["success"]:
                return res["content"].strip()
        except Exception:
            continue

    # Step 5: 全部失败
    return ""


def batch_get_install_info_from_readthedocs(
    llm,
    urls_csv_path: str = r"D:\1Grad\Code\wkf\tests\final_urls.csv",
    tasks_excel_path: str = r"D:\1Grad\Code\wkf\tests\query\scratch\tool_tasktype_query_taskfiltered.xlsx",
    output_excel_path: str = None,
) -> None:
    """
    批量从 Read the Docs 文档中提取生信工具的安装信息。
    - 日志同时输出到控制台和文件（同目录下，带时间戳）。
    - 每个 tool_name 仅处理一次。
    - 非空 install info → status = 'READTHEDOCS'。
    """

    # === 配置日志：同时输出到控制台和文件 ===
    log_dir = os.path.dirname(tasks_excel_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_install_{timestamp}.log")

    # 创建独立 logger，避免全局污染
    logger = logging.getLogger("BatchInstall")
    logger.setLevel(logging.INFO)

    # 清除已有 handler（防止重复日志）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Starting batch installation info extraction...")

    # Step 1: 加载 URL 映射
    logger.info("Loading tool → URL mapping from CSV...")
    urls_df = pd.read_csv(urls_csv_path)
    if "name" not in urls_df.columns or "first" not in urls_df.columns:
        raise ValueError("CSV must contain 'name' and 'first' columns")

    tool_to_url: Dict[str, str] = {}
    for _, row in urls_df.iterrows():
        tool = str(row["name"]).strip()
        url = str(row["first"]).strip()
        if tool and url and "readthedocs.io" in url:
            tool_to_url[tool] = url

    logger.info(f"Loaded {len(tool_to_url)} valid ReadTheDocs URLs.")

    # Step 2: 加载任务表
    logger.info("Loading task Excel file...")
    tasks_df = pd.read_excel(tasks_excel_path)

    required_cols = {"tool_name", "can_complete", "status"}
    if not required_cols.issubset(tasks_df.columns):
        missing = required_cols - set(tasks_df.columns)
        raise ValueError(f"Missing columns in Excel: {missing}")

    if "installation_info" not in tasks_df.columns:
        tasks_df["installation_info"] = ""

    # Step 3: 筛选待处理行（排除已成功标记为 READTHEDOCS 的）
    mask = (tasks_df["can_complete"] == 1) & (tasks_df["status"] != "READTHEDOCS")
    rows_to_process = tasks_df[mask]

    seen = set()
    unique_tools = []
    for tool in rows_to_process["tool_name"].astype(str).str.strip():
        if tool and tool not in seen:
            unique_tools.append(tool)
            seen.add(tool)

    logger.info(f"Found {len(unique_tools)} unique tools to process.")

    # Step 4: 缓存安装信息（带进度条）
    tool_install_cache: Dict[str, str] = {}

    for tool in tqdm(unique_tools, desc="Processing tools"):
        if tool not in tool_to_url:
            tool_install_cache[tool] = ""
            continue

        url = tool_to_url[tool]
        try:
            install_info = get_install_info_from_readthedocs(
                tool=tool, url=url, llm=llm
            )
            tool_install_cache[tool] = install_info.strip() if install_info else ""
        except Exception as e:
            logger.error(f"Failed to process tool '{tool}' at {url}: {e}")
            tool_install_cache[tool] = ""

    # Step 5: 填充结果回原 DataFrame
    updated_count = 0
    for idx in tasks_df.index:
        if not mask[idx]:
            continue
        tool = str(tasks_df.at[idx, "tool_name"]).strip()
        if tool in tool_install_cache:
            install_info = tool_install_cache[tool]
            if install_info:
                tasks_df.at[idx, "installation_info"] = install_info
                tasks_df.at[idx, "status"] = "READTHEDOCS"
                updated_count += 1

    # Step 6: 保存结果
    output_path = output_excel_path or tasks_excel_path
    tasks_df.to_excel(output_path, index=False)

    logger.info("Batch processing completed.")
    logger.info(f"Unique tools processed: {len(unique_tools)}")
    logger.info(f"Rows updated with non-empty install info: {updated_count}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    model = ChatOpenAI(
        model="qwen3-max",
        # model="qwen3-coder-plus",
        base_url="",
        api_key="",
    )
    # global db
    db = Database()
    
    ## ==========获取readthedocs文档并存入数据库==========
    tools = ["cell2cell",] # 支持传入多个tool
    fetch_and_store_readthedocs(db, tools)

    ## ==========从readthedocs文档中获取安装信息并直接返回==========
    # r = get_install_info_from_readthedocs(
    #     tool="celltypist", url="https://celltypist.readthedocs.io/en/latest/", llm=model
    # )
    # print(r)

    ## ==========批量获取安装信息并存入excel（不推荐使用，与excel内容强耦合）==========
    # batch_get_install_info_from_readthedocs(
    #     llm=model,
    #     urls_csv_path=r"D:\Code\wkf\tests\final_urls.csv",
    #     tasks_excel_path=r"D:\Code\wkf\tests\query\scratch\tool_tasktype_query_taskfiltered.xlsx",
    # )
