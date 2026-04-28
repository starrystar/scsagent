import re
import pymysql
import requests
import time
import ast
from langchain_openai import ChatOpenAI
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup

from utils.database import Database
from utils.crawler import fetch_url, crawl_with_rec_filter

db = Database()


def store_docs(tool, url, summary, doc, html):  # url或者urls都行
    if type(url) == str:
        urls = [url]
    for url in urls:
        insert_sql = "insert into docs(tool_id, url, llm_summary, doc, html) values((select id from tools where name=%s), %s, %s, %s, %s)"
        db.execute_update(insert_sql, (tool, url, summary, doc, html))


def summary_doc(model, source: str) -> str:
    """使用 LLM 对文档内容生成摘要"""
    prompt = f"""你的任务是总结文档并输出一段简短的总结，主要关注文档完成了哪些任务、有哪些步骤等。
以下是原始文档：
<source>
{source}
</source>

你的任务是总结文档并输出一段简短的总结，主要关注文档完成了哪些任务、有哪些步骤等：
"""
    return model.invoke(prompt).content.strip()


def fetch_and_store_summary(model, tool, url, depth=1):
    # 这个是从url的递归页面中获取文档

    # db_fetch
    try:
        qsql = "select llm_summary from docs where url=%s"
        db_res = db.execute_query(qsql, (url,))
    except pymysql.err.ProgrammingError as e:
        print(f"SQL Error: {e}")
        return []
    if len(db_res) > 0 and db_res[0][0] is not None:  # 如果db已经有这个url了
        return [db_res[0][0]]

    # fetch
    domain = urlparse(url).netloc
    visited = set()
    valid = {}
    crawl_with_rec_filter(url, depth, domain, visited, valid)
    docs = []

    # store
    for url, html in valid.items():
        soup = BeautifulSoup(html, "html.parser")
        doc = soup.get_text()  # separator=" ", strip=True
        summary = summary_doc(model, doc)
        store_docs(db, tool=tool, url=url, summary=summary, doc=doc, html=html)
        docs.append(doc)
    return docs


def get_tool_doc_from_db(tool):
    # TODO  emb+reranker
    sql2 = "select doc from docs where tool_id = (select id from tools where name=%s)"
    db_res2 = db.execute_query(sql2, (tool,))
    return [doc[0] for doc in db_res2]


if __name__ == "__main__":
    # url = "https://scvi-tools.readthedocs.io/en/latest/tutorials/notebooks/scrna/scarches_scvi_tools.html"
    # url = "https://bioconductor.org/packages/release/bioc/html/AUCell.html"
    # url = 'https://bohrium.dp.tech/notebooks/86611649178'
    # url = "https://blog.csdn.net/qq_40943760/article/details/138871689"
    # tool = "scvi-tools"

    # url = "https://scanpy.readthedocs.io/en/stable/tutorials/plotting/advanced.html"
    # tool = "Scanpy"

    model = ChatOpenAI(
        # model="qwen3-max",
        model="qwen3-coder-plus",
        base_url="",
        api_key="",
    )
    tool = "DESC"
    url = "https://eleozzr.github.io/desc/tutorial.html"

    fetch_and_store_summary(model, tool, url)
