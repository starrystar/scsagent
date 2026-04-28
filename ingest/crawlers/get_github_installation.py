# ## 批量获取Github readme的安装信息

from langchain_openai import ChatOpenAI

tool = 'ASAP'

llm = ChatOpenAI(
    # model='deepseek-v3-hs',
    # model='deepseek-r1-hs',
    # model="qwen-plus",
    model="qwen-max",
    # model="qwen3-coder-plus",
    base_url="",
    api_key="",
)

# res = llm.invoke(f"请给出单细胞转录组学分析Python工具{tool}的安装命令")
# res.content

import os
import logging
import pandas as pd
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 

# ================== 配置 ==================
GITHUB_DIR = rf"D:\1Grad\Code\wkf\data\github"
REPOS_FILE = rf"{GITHUB_DIR}\repos.txt"
OUTPUT_EXCEL = rf"{GITHUB_DIR}\installation_info.xlsx"
LOG_FILE = rf"{GITHUB_DIR}\extraction.log"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

 
# 定义 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in analyzing README files of software projects."),
    ("human", """
Read the following README content and determine if it contains installation instructions.
If it does, extract ONLY the installation-related section(s) (including code blocks, commands, steps).
If it does not, respond with exactly: "NO_INSTALLATION_INFO".

README content:
{readme_content}
""")
])

chain = prompt | llm

# ================== 主逻辑 ==================
def main():
    results = []

    if not os.path.exists(REPOS_FILE):
        logger.error(f"repos.txt 文件不存在: {REPOS_FILE}")
        return

    with open(REPOS_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            logger.warning(f"跳过无效行: {line}")
            continue

        tool_name = parts[0]
        github_url = parts[1]

        repo_path = Path(GITHUB_DIR) / tool_name
        readme_path = repo_path / "README.md"

        logger.info(f"处理项目: {tool_name}")

        if not readme_path.exists(): # TODO 有的github提供的不是md文件，例如rst等需要改进
            logger.warning(f"未找到 README.md: {readme_path}")
            results.append({
                "tool_name": tool_name,
                "github_url": github_url,
                "has_readme": False,
                "installation_info": None,
                "status": "MISSING_README"
            })
            continue

        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
        except Exception as e:
            logger.error(f"读取 README 失败 ({tool_name}): {e}")
            results.append({
                "tool_name": tool_name,
                "github_url": github_url,
                "has_readme": True,
                "installation_info": None,
                "status": "READ_ERROR"
            })
            continue

        if not readme_content.strip():
            logger.warning(f"README 为空: {tool_name}")
            results.append({
                "tool_name": tool_name,
                "github_url": github_url,
                "has_readme": True,
                "installation_info": None,
                "status": "EMPTY_README"
            })
            continue

        # 调用 LLM 提取安装信息
        try:
            response = chain.invoke({"readme_content": readme_content[:12000]})  # 限制长度避免 token 超限
            install_text = response.content.strip()

            if install_text == "NO_INSTALLATION_INFO":
                logger.info(f"未检测到安装信息: {tool_name}")
                results.append({
                    "tool_name": tool_name,
                    "github_url": github_url,
                    "has_readme": True,
                    "installation_info": None,
                    "status": "NO_INSTALLATION"
                })
            else:
                logger.info(f"成功提取安装信息: {tool_name}")
                results.append({
                    "tool_name": tool_name,
                    "github_url": github_url,
                    "has_readme": True,
                    "installation_info": install_text,
                    "status": "SUCCESS"
                })
        except Exception as e:
            logger.error(f"LLM 调用失败 ({tool_name}): {e}")
            results.append({
                "tool_name": tool_name,
                "github_url": github_url,
                "has_readme": True,
                "installation_info": None,
                "status": "LLM_ERROR"
            })

    # 输出到 Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    logger.info(f"结果已保存到 {OUTPUT_EXCEL}，共处理 {len(results)} 个项目。")

if __name__ == "__main__":
    main()


