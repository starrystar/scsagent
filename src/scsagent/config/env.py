import os
from pathlib import Path
from dotenv import load_dotenv

# 获取当前文件所在目录（即 config/ 目录）
_config_dir = Path(__file__).parent.resolve()

# 可选：加载通用 .env（如果存在）
_common_env = _config_dir / ".env"
if _common_env.is_file():
    load_dotenv(_common_env, override=False)


# 平台专属配置由调用方决定何时加载（推荐显式控制）
def load_platform_env():
    """显式加载平台相关的 .env 文件（如 .env.windows 或 .env.linux）"""
    import platform

    system = platform.system().lower()
    if system == "windows":
        env_file = _config_dir / ".env.windows"
    elif system in ("linux", "darwin"):
        env_file = _config_dir / ".env.linux"
    else:
        raise OSError(f"Unsupported operating system: {system}")

    if env_file.is_file():
        load_dotenv(env_file, override=True)
        print(f"✅ 已加载平台配置: {env_file}")
    else:
        print(f"⚠️ 未找到平台配置文件: {env_file}")


load_platform_env()


BASIC_MODEL = os.getenv("BASIC_MODEL", "gpt-4o")
BASIC_BASE_URL = os.getenv("BASIC_BASE_URL")
BASIC_API_KEY = os.getenv("BASIC_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
ENCODING = os.getenv("ENCODING")

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE")
CONTAINER_WORK_DIR = os.getenv("CONTAINER_WORK_DIR")
HOST_WORK_DIR = os.getenv("HOST_WORK_DIR")
HOST_DATA_PATH = os.getenv("HOST_DATA_PATH")
