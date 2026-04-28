import uuid
import logging
import os
import shutil
import argparse
import logging
import tempfile
import ipdb
from langchain_openai import ChatOpenAI
from pathlib import Path
from datetime import datetime
from rapidfuzz import process

from utils.database import Database
from utils.sandbox import set_docker_manager, DockerManager
from scsagent.core.debug_react_first6 import debug_workflow
from scsagent.config.env import (
    HOST_WORK_DIR,
    HOST_DATA_PATH,
    CONTAINER_WORK_DIR,
)


llm = ChatOpenAI(
    model="qwen3-max",
    # model="qwen3-coder-plus",
    base_url="",
    api_key="",
    # model="kimi-k2-0711-preview",
    # model="kimi-k2-turbo-preview",
    # model="kimi-latest",
    # model="qwen/qwen3-coder",
    # model="qwen/qwen3-235b-a22b-2507",
)

def logger_config(log_dir, log_level=logging.INFO, enable_console=True):
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(log_level)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("[%(name)s]: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器始终开启
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(log_dir, f"{formatted_time}.log")
    file_handler = logging.FileHandler(fname, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(name)s]: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def match_tool_name(task: str, logger) -> str:
    """从任务描述中提取并匹配工具名称"""
    extract_tool_prompt = f"找到并只返回生信分析工具名称，不要任何额外文本:\n{task}"
    ori_tool = llm.invoke(extract_tool_prompt).content
    logger.info(f'[工具名提取]：{ori_tool}')

    with Database() as db:
        sql = "select name from tools"
        res = db.execute_query(sql)
        tool_list = [row[0] for row in res]

    # 内联模糊匹配逻辑
    best_match, score, idx = process.extractOne(
        ori_tool,
        tool_list,
        processor=str.lower
    )
    tool = best_match
    logger.info(f'[工具名匹配]：{tool}')
    
    return tool

def run_workflow(
    query: str,
    host_work_dir: str = HOST_WORK_DIR,
    host_data_path: str = HOST_DATA_PATH,
    docker_image: str = None,
):
    """
    执行工作流：启动Docker容器，运行debug_agent，处理任务。

    参数:
        query (str): 用户输入的任务描述，如“帮我做基因共表达分析，文件为/workspace/input/mouse.h5ad”
        host_work_dir (str): 本机工作目录，默认 HOST_WORK_DIR
        host_data_path (str): 数据目录，内容将复制到工作目录的 input 子目录
        docker_image (str): Docker 镜像名，必须指定
    """
    if docker_image is None:
        raise ValueError("docker_image 必须指定")
    
    host_work_dir = Path(host_work_dir).expanduser().resolve()   # 你的最终成果区
    
    # 生成任务ID
    with tempfile.TemporaryDirectory(prefix="stomics_agent_") as tmp:
        host_project_dir = Path(tmp).resolve()
        task_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"

        # 配置日志
        logger_config(log_dir=host_project_dir, enable_console=False)
        logger = logging.getLogger(__name__)
        logger.info(f"Starting workflow with user input: {query}")
        logger.info(f"host_project_dir: {host_project_dir.absolute()}")
        tool = match_tool_name(task=query, logger=logger)

        # 复制数据到 input 目录
        cp_to_data_path = host_project_dir / "input"
        shutil.copytree(host_data_path, cp_to_data_path, dirs_exist_ok=True)

        debug_agent = debug_workflow()
        result = None
        new_image_name = None
        success = False

        try:
            # 启动 Docker 容器
            logger.info(f"Starting Docker container for task {task_id}")
            docker_manager = DockerManager()
            set_docker_manager(docker_manager)  # 注册到当前进程上下文
            container_id = docker_manager.create_container(
                task_id, host_project_dir, docker_image
            )
            logger.info(f"Docker container started with ID: {container_id}")

             # 调用工作流
            result = debug_agent.invoke(
                {
                    "host_data_path": host_data_path,
                    "container_work_dir": CONTAINER_WORK_DIR,
                    "task": query,
                    "tool": tool,
                },
                config={"recursion_limit": 300}
            )

            # 提取文档 ID 列表（如果 docs 存在且包含 id 字段）
            # 提取文档 ID 列表并转为字符串（如 "1167,1168"）
            docs_id = ""
            if result and isinstance(result, dict) and 'docs' in result:
                ids = []
                for doc in result['docs']:
                    if isinstance(doc, dict) and 'id' in doc:
                        ids.append(str(doc['id']))  # 转为字符串，避免 int/str 混合
                docs_id = ",".join(ids)  # 空列表 → 空字符串 ""

            done = result.get('done', False) if result else False
            if done:
                success = True
                formatted_time = datetime.now().strftime("%Y%m%d%H%M%S")
                # TODO 常量放到配置文件中
                new_image_name = f'wh-harbor.dcs.cloud/public-library/agent_{tool}_{formatted_time}'.lower()
                committed_image = docker_manager.commit_container(new_image_name)
            else:
                logger.warning("Workflow finished but 'done' is False.")

        except Exception as e:
            logger.error(f"Workflow failed with error: {e}")
            success = False
            raise

        finally:
            # === 关键修复：关闭日志文件句柄，避免 Windows 文件锁 ===
            root_logger = logging.getLogger()
            handlers_to_remove = []
            for handler in root_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()  # 释放文件锁
                handlers_to_remove.append(handler)

            for handler in handlers_to_remove:
                root_logger.removeHandler(handler)
            # =======================================================

            try:
                docker_manager.remove_container()
            except Exception as cleanup_error:
                # 此时 logger 可能已无 handler，改用 print 安全输出
                print(f"[Cleanup Error] Failed to remove container: {cleanup_error}")

            # 清理 input 目录
            if cp_to_data_path.exists():
                shutil.rmtree(cp_to_data_path)

            # 复制结果到用户指定目录（此时日志文件已解锁，可安全复制/删除）
            final_dir = host_work_dir / task_id
            shutil.copytree(host_project_dir, str(final_dir))

            print("Workflow cleanup completed.")

    # 构造返回值
    done = bool(result and result.get('done'))
    # ipdb.set_trace()
    return {
        'docs_id': docs_id,
        'done': done,
        'code': result.get('codes')[-1] if done and result and result.get('codes') else '',
        'new_image_name': committed_image if success else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        help="用户任务，例如'帮我做基因共表达分析，文件名mouse.h5ad'",
        default="帮我做基因共表达分析，文件为/workspace/input/mouse.h5ad",
        required=True,
    )
    parser.add_argument("--host_work_dir", help="本机工作目录", default=HOST_WORK_DIR)
    parser.add_argument(
        "--host_data_path",
        help="数据目录，其中所有数据文件将复制到工作目录的input子文件夹中",
        default=HOST_DATA_PATH,
    )
    parser.add_argument(
        "--docker_image",
        help="运行镜像，例如 ubuntu:20.04或者image_id，必须指定，新打包的镜像将用这个repo:{time}命名",
        required=True,
    )
    args = parser.parse_args()

    # 调用封装好的函数
    result = run_workflow(
        query=args.query,
        host_work_dir=args.host_work_dir,
        host_data_path=args.host_data_path,
        docker_image=args.docker_image,
    )
