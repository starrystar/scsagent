import os
import shutil
import tempfile
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, TypedDict, Annotated, Optional
from git import Repo
from git.exc import GitCommandError
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from datetime import datetime
import nbformat
from nbconvert import MarkdownExporter


# ==================== 配置 ====================
class Config:
    """Agent 配置"""

    TOOL_NAME = None  # 工具名称，从命令行传入

    # 🔧 LLM 配置 - 自定义模型
    LLM_MODEL = "qwen3-max"
    LLM_BASE_URL = ""
    LLM_API_KEY = ""
    LLM_TEMPERATURE = 0
    LLM_TIMEOUT = 60
    LLM_MAX_RETRIES = 3

    MAX_REPO_SIZE_MB = 500  # 限制仓库大小防止滥用


# ==================== 核心处理函数（可被批量调用） ====================
def process_github_repo(github_url: str, base_local_path: str, tool_name: str):
    """
    处理单个 GitHub 仓库的核心函数

    Args:
        github_url: GitHub 仓库 URL
        base_local_path: 本地基础路径（会自动拼接 tool_name）
        tool_name: 工具名称（用于获取 tool_id 和拼接路径）

    Returns:
        dict: 处理结果，包含 success 布尔值和 identified_docs 列表
    """
    # 拼接完整的本地路径：base_local_path/tool_name
    local_path = str(Path(base_local_path) / tool_name)

    # 设置配置
    Config.TOOL_NAME = tool_name

    print("💾 使用数据库存储模式")
    print(f"\n🔧 工具名称：{Config.TOOL_NAME}")
    print(f"🌐 GitHub URL: {github_url}")
    print(f"📁 本地路径：{local_path}\n")

    # 准备输入数据
    input_data = {
        "github_url": github_url,
        "local_path": local_path,
        "default_branch": None,
        "repo_structure": None,
        "identified_docs": [],
        "error": None,
        "success": True,
    }

    # 运行 Agent
    try:
        result = agent.invoke(input_data)

        print("\n✅ Agent 执行完成")
        if result["error"]:
            print(f"❌ 错误：{result['error']}")
            return {"success": False, "error": result["error"], "identified_docs": []}
        else:
            print(f"✓ 成功识别 {len(result['identified_docs'])} 个文档/教程")
            for doc in result["identified_docs"]:
                print(f"  • {doc['path']} (置信度：{doc.get('confidence', 0.0):.2f})")
                print(f"    原因：{doc.get('reason', 'N/A')}")
            return {
                "success": True,
                "error": None,
                "identified_docs": result["identified_docs"],
            }
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        return {"success": False, "error": str(e), "identified_docs": []}


# ==================== 状态定义 ====================
class RepoAnalysisState(TypedDict):
    """Agent 状态"""

    github_url: str  # GitHub 仓库 URL
    local_path: Optional[str]  # 本地克隆路径
    default_branch: Optional[str]  # 默认分支名称
    repo_structure: Optional[str]  # 仓库目录树（文本格式）
    identified_docs: List[Dict]  # 识别出的文档/教程
    error: Optional[str]  # 错误信息
    success: bool  # 执行是否成功


# ==================== 工具函数 ====================
def get_default_branch(repo: Repo) -> str:
    """获取仓库的默认分支名称"""
    try:
        for ref in repo.refs:
            if ref.name == "HEAD":
                return ref.remote_head.split("/")[-1]

        if hasattr(repo, "remote") and repo.remotes:
            remote = repo.remotes[0]
            for ref in remote.refs:
                if ref.remote_head == "HEAD":
                    return ref.ref.name.split("/")[-1]

        common_branches = ["main", "master", "develop"]
        for branch in common_branches:
            if branch in [b.name for b in repo.branches]:
                return branch

        if repo.branches:
            return repo.branches[0].name

    except Exception as e:
        print(f"⚠️  获取默认分支失败：{e}，使用默认值 'main'")

    return "main"


def clone_github_repo(url: str, base_dir: Path) -> tuple[str, str]:
    """克隆 GitHub 仓库到本地（永久保存）"""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = base_dir / repo_name

    base_dir.mkdir(parents=True, exist_ok=True)

    if clone_path.exists():
        print(f"⚠️  仓库已存在，跳过克隆：{clone_path}")
        repo = Repo(clone_path)
        default_branch = get_default_branch(repo)
        print(f"📌 默认分支：{default_branch}")
        return str(clone_path), default_branch

    try:
        print(f"📥 正在克隆仓库到：{clone_path}")
        repo = Repo.clone_from(url, str(clone_path), depth=1)
        default_branch = get_default_branch(repo)
        print(f"✅ 仓库克隆完成：{clone_path}")
        print(f"📌 默认分支：{default_branch}")
        return str(clone_path), default_branch
    except GitCommandError as e:
        raise RuntimeError(f"克隆仓库失败：{e}")


def get_repo_structure(path: str, max_depth: int = 3) -> str:
    """生成仓库目录树（限制深度避免过大）"""
    root = Path(path)
    tree_lines = []

    def _build_tree(current_path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        try:
            items = sorted(current_path.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{item.name}")

                if item.is_dir() and not item.name.startswith((".", "__")):
                    extension = "    " if is_last else "│   "
                    _build_tree(item, prefix + extension, depth + 1)
        except PermissionError:
            pass

    tree_lines.append(f"{root.name}/")
    _build_tree(root)
    return "\n".join(tree_lines)


def is_repo_too_large(path: str, max_size_mb: int) -> bool:
    """检查仓库大小是否超过限制"""
    total_size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total_size > (max_size_mb * 1024 * 1024)


def convert_ipynb_to_markdown(input_ipynb: str, output_md: Optional[str] = None) -> str:
    """将 ipynb 文件转换为 markdown 文本（去掉代码输出结果）"""
    with open(input_ipynb, "r", encoding="utf-8") as f:
        nb_content = nbformat.read(f, as_version=4)

    md_exporter = MarkdownExporter()
    md_exporter.exclude_output = True
    (body, resources) = md_exporter.from_notebook_node(nb_content)

    if output_md:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"📝 转换成功！已生成：{output_md}")
        return body
    else:
        return body


def read_file_content(file_path: str) -> str:
    """读取文件内容，支持.md、.py、.ipynb 等格式"""
    path = Path(file_path)
    if not path.exists():
        return ""

    if path.suffix.lower() == ".ipynb":
        return convert_ipynb_to_markdown(str(path))

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def get_files_in_document_directory(repo_path: str, doc_dir_path: str) -> List[Dict]:
    """获取文档目录下的所有文件"""
    doc_path = Path(repo_path) / doc_dir_path
    files = []

    if doc_path.exists() and doc_path.is_dir():
        for file_path in doc_path.rglob("*"):
            if file_path.is_file():
                if any(
                    part.startswith(".") or part.startswith("__")
                    for part in file_path.parts
                ):
                    continue

                relative_path = str(file_path.relative_to(Path(repo_path)))
                files.append(
                    {
                        "path": relative_path,
                        "confidence": 0.9,
                        "reason": f"位于文档目录 {doc_dir_path} 下的文件",
                    }
                )
    return files


# ==================== LLM 分析器 ====================
class DocCandidate(BaseModel):
    path: str = Field(description="相对于仓库根目录的路径")
    confidence: float = Field(description="置信度 0.0-1.0", ge=0.0, le=1.0)
    reason: str = Field(description="判断为文档的原因")


class LLMAnalyzer:
    """使用 LLM 分析仓库结构并识别文档"""

    def __init__(self, model_name: str = None):
        model_name = model_name or Config.LLM_MODEL

        self.llm = ChatOpenAI(
            model=model_name,
            base_url=Config.LLM_BASE_URL,
            api_key=Config.LLM_API_KEY,
            temperature=Config.LLM_TEMPERATURE,
            request_timeout=Config.LLM_TIMEOUT,
            max_retries=Config.LLM_MAX_RETRIES,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的开源项目分析专家。请分析以下 GitHub 仓库的目录结构，识别出最可能是教程（tutorial）或文档（documentation）的目录或文件。

分析标准：
1. 教程特征：包含"tutorial"、"guide"、"learn"、"example"、"demo"等关键词
2. 文档特征：包含"docs"、"doc"、"documentation"、"api"、"reference"等；有.md/.rst/.txt等文档格式
3. 高价值文件：README.md 等
4. 文档目录：如果某个目录明显是文档相关目录 (如 docs/, tutorials/, examples/等)，请返回该目录路径，我们会自动包含其下所有文件

请只返回 JSON 格式，包含 doc_candidates 数组，每个元素包含：
- path: 相对路径
- confidence: 置信度 (0.0-1.0)
- reason: 判断为文档的原因""",
                ),
                ("human", "仓库目录结构：\n{repo_structure}"),
            ]
        )

        try:
            self.chain = self.prompt | self.llm.with_structured_output(
                schema={
                    "properties": {
                        "doc_candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["path", "confidence", "reason"],
                            },
                        }
                    },
                    "required": ["doc_candidates"],
                    "type": "object",
                }
            )
            self.use_structured_output = True
        except Exception:
            self.use_structured_output = False
            self.chain = self.prompt | self.llm

    def _extract_json_from_response(self, content: str) -> Dict:
        json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                content = content[start:end]
        return json.loads(content.strip())

    def identify_docs(
        self, repo_structure: str, repo_path: Optional[str] = None
    ) -> List[Dict]:
        all_docs = []
        print("\n📁 仓库目录结构：")
        print(repo_structure)
        print("=" * 50)

        try:
            if self.use_structured_output:
                result = self.chain.invoke({"repo_structure": repo_structure})
                llm_candidates = result.get("doc_candidates", [])
            else:
                response = self.chain.invoke({"repo_structure": repo_structure})
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                result = self._extract_json_from_response(content)
                llm_candidates = result.get("doc_candidates", [])

            llm_docs = [c for c in llm_candidates if c.get("confidence", 0) > 0.6]

            print(f"\n🤖 LLM 识别到 {len(llm_docs)} 个文档候选：")
            for doc in llm_docs:
                print(f"  - {doc.get('path')} (置信度：{doc.get('confidence'):.2f})")
                print(f"    原因：{doc.get('reason', 'N/A')}")

            if repo_path and os.path.exists(repo_path):
                expanded_docs = []
                root_path = Path(repo_path)

                for doc_candidate in llm_docs:
                    candidate_path = root_path / doc_candidate["path"]
                    if candidate_path.exists():
                        if candidate_path.is_file():
                            expanded_docs.append(doc_candidate)
                            print(f"    ✅ 添加文件：{doc_candidate['path']}")
                        elif candidate_path.is_dir():
                            dir_files = get_files_in_document_directory(
                                repo_path, doc_candidate["path"]
                            )
                            expanded_docs.extend(dir_files)
                            print(
                                f"    📂 展开目录 {doc_candidate['path']}，包含 {len(dir_files)} 个文件"
                            )
                        else:
                            print(f"    ⚠️  路径不存在：{doc_candidate['path']}")
                    else:
                        print(f"    ⚠️  路径不存在：{doc_candidate['path']}")
                        expanded_docs.append(doc_candidate)

                unique_docs = []
                seen_paths = set()
                for doc in expanded_docs:
                    path = doc.get("path", "")
                    if path not in seen_paths:
                        unique_docs.append(doc)
                        seen_paths.add(path)
                all_docs.extend(unique_docs)
                print(f"\n📊 总共识别到 {len(all_docs)} 个文档项")
            else:
                all_docs.extend(llm_docs)

        except Exception as e:
            print(f"⚠️  LLM 分析失败：{e}")
            all_docs = []

        return all_docs


# ==================== Agent 节点定义 ====================
def clone_repo_node(state: RepoAnalysisState) -> RepoAnalysisState:
    """节点 1：克隆仓库"""
    try:
        base_dir = Path(state["local_path"]).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        local_path, default_branch = clone_github_repo(state["github_url"], base_dir)

        if is_repo_too_large(local_path, Config.MAX_REPO_SIZE_MB):
            return {
                **state,
                "error": f"仓库超过大小限制 ({Config.MAX_REPO_SIZE_MB}MB)",
                "success": False,
            }

        return {
            **state,
            "local_path": local_path,
            "default_branch": default_branch,
            "error": None,
            "success": True,
        }
    except Exception as e:
        return {**state, "error": f"克隆失败：{str(e)}", "success": False}


def analyze_structure_node(state: RepoAnalysisState) -> RepoAnalysisState:
    """节点 2：分析仓库结构"""
    if not state["success"] or not state["local_path"]:
        return state
    try:
        structure = get_repo_structure(state["local_path"])
        return {**state, "repo_structure": structure, "error": None}
    except Exception as e:
        return {**state, "error": f"结构分析失败：{str(e)}", "success": False}


def identify_docs_node(state: RepoAnalysisState) -> RepoAnalysisState:
    """节点 3：LLM 识别文档 + 扫描 .ipynb 文件"""
    if not state["success"] or not state["repo_structure"]:
        return state
    try:
        analyzer = LLMAnalyzer(model_name=Config.LLM_MODEL)
        docs = analyzer.identify_docs(
            repo_structure=state["repo_structure"], repo_path=state["local_path"]
        )
        return {**state, "identified_docs": docs, "error": None}
    except Exception as e:
        return {**state, "error": f"文档识别失败：{str(e)}", "success": False}


def store_to_db_node(state: RepoAnalysisState) -> RepoAnalysisState:
    """节点 4：存储到数据库（直接调用）"""
    if not state["success"] or not state["identified_docs"]:
        return state

    try:
        # 🔥 直接调用，移除多余的 Config.DB_STORE_FUNCTION 注入
        db_store_function(
            github_url=state["github_url"],
            local_path=state["local_path"],
            docs=state["identified_docs"],
            default_branch=state.get("default_branch", "main"),
        )
        return {**state, "error": None}
    except Exception as e:
        return {**state, "error": f"数据库存储失败：{str(e)}", "success": False}


# ==================== 构建 StateGraph ====================
def create_doc_agent() -> StateGraph:
    """创建文档分析 Agent"""
    workflow = StateGraph(RepoAnalysisState)

    workflow.add_node("clone_repo", clone_repo_node)
    workflow.add_node("analyze_structure", analyze_structure_node)
    workflow.add_node("identify_docs", identify_docs_node)
    workflow.add_node("store_to_db", store_to_db_node)

    workflow.set_entry_point("clone_repo")
    workflow.add_edge("clone_repo", "analyze_structure")
    workflow.add_edge("analyze_structure", "identify_docs")
    workflow.add_edge("identify_docs", "store_to_db")
    workflow.add_edge("store_to_db", END)

    return workflow.compile()


# ==================== 数据库存储函数 ====================
def db_store_function(
    github_url: str, local_path: str, docs: List[Dict], default_branch: str = "main"
):
    """将分析结果存储到数据库的 docs 表"""
    from utils.database import Database

    if not Config.TOOL_NAME:
        print("❌ 错误：未设置 TOOL_NAME，无法获取 tool_id")
        return

    try:
        db = Database()
        tool_query = "SELECT id FROM tools WHERE name = %s"
        result = db.execute_query(tool_query, (Config.TOOL_NAME,))

        if not result or len(result) == 0:
            print(f"❌ 错误：在 tools 表中未找到工具 '{Config.TOOL_NAME}'")
            db.disconnect()
            return

        tool_id = result[0][0]
        print(f"✅ 获取 tool_id: {tool_id} (工具名：{Config.TOOL_NAME})")

        insert_sql = "INSERT INTO docs (tool_id, url, doc) VALUES (%s, %s, %s)"

        inserted_count = 0
        for doc in docs:
            file_path = doc.get("path", "")
            normalized_file_path = file_path.replace("\\", "/")
            full_github_url = (
                f"{github_url.rstrip('/')}/blob/{default_branch}/{normalized_file_path}"
            )
            full_file_path = (
                os.path.join(local_path, file_path) if local_path else file_path
            )

            doc_content = read_file_content(full_file_path)
            if not doc_content:
                print(f"⚠️  跳过空文件或读取失败的文件：{full_file_path}")
                continue

            db.execute_update(insert_sql, (tool_id, full_github_url, doc_content))
            inserted_count += 1
            print(f"  ✓ 已插入：{file_path}")
            print(f"     GitHub URL: {full_github_url}")
            print(f"     内容大小：{len(doc_content)} 字符")

        db.disconnect()
        print(f"\n📊 数据库存储完成")
        print(f"📁 仓库：{github_url}")
        print(f"🔧 工具 ID: {tool_id} ({Config.TOOL_NAME})")
        print(f"📄 成功插入 {inserted_count}/{len(docs)} 个文档")

    except Exception as e:
        print(f"❌ 数据库存储失败：{str(e)}")
        raise


# 创建 Agent（已移除 Config.DB_STORE_FUNCTION 赋值）
agent = create_doc_agent()


# ==================== 命令行参数解析 ====================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GitHub 仓库文档分析工具 - 分析 GitHub 仓库并提取文档到数据库"
    )

    parser.add_argument("--github-url", type=str, required=True, help="GitHub 仓库 URL")
    parser.add_argument("--base-local-path", type=str, required=True, help="本地基础路径")
    parser.add_argument("--tool-name", type=str, required=True, help="工具名称")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    result = process_github_repo(
        github_url=args.github_url,
        base_local_path=args.base_local_path,
        tool_name=args.tool_name,
    )

    if not result["success"]:
        exit(1)


if __name__ == "__main__":
    main()
