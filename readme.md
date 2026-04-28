# 🧬 SCSAgent - Single Cell Analysis Documentation Agent

> 一个用于单细胞分析工具文档爬取、管理与智能问答的系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-✅-2496ED.svg)](https://www.docker.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [✨ 项目简介](#-项目简介)
- [🚀 快速开始](#-快速开始)
- [🗄️ 数据库配置](#️-数据库配置)
- [🕷️ 文档爬取模块](#️-文档爬取模块)
- [🐳 Docker 镜像管理](#-docker-镜像管理)
- [📁 项目结构](#-项目结构)
- [🛠️ 使用指南](#️-使用指南)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

---

## ✨ 项目简介

**SCSAgent** 是一个面向单细胞分析领域的智能文档管理系统，主要功能包括：

- 🔍 自动爬取 GitHub / ReadTheDocs 上的生物信息学工具文档
- 💾 结构化存储文档内容至 MySQL 数据库
- 🐳 基于 Docker 的工具镜像构建与批量管理
- 🤖 支持自然语言查询与工具调用推荐

> ⚠️ **注意**：项目处于早期开发阶段，部分模块结构仍在优化中。

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 克隆项目
git clone https://github.com/your-org/scsagent.git
cd scsagent

# 安装依赖（必须在项目根目录执行）
pip install -e .
```

### 2️⃣ 启动 MySQL 服务

需要提前下载mysql的docker镜像。

```bash
docker run -itd --name mysql-dev \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=agent \
  -p 3306:3306 \
  docker.1ms.run/library/mysql:latest
```

### 3️⃣ 验证连接

```bash
docker exec -it mysql-dev /bin/bash 
mysql -u root -proot

# 在 MySQL 中执行
SHOW DATABASES;
USE agent;
```

---

## 🗄️ 数据库配置

### 🔐 连接信息

| 配置项 | 值 |
|--------|-----|
| Host | `localhost` |
| Port | `3306` |
| Database | `agent` |
| Username | `root` |
| Password | `root` |

### 🔍 常用查询示例

```sql
-- 查看全部文档（截断长文本便于预览）
SELECT 
    id,
    tool_id,
    LEFT(REPLACE(REPLACE(url, '\n', ' '), '\r', ' '), 20) AS url,
    LEFT(REPLACE(REPLACE(llm_summary, '\n', ' '), '\r', ' '), 20) AS llm_summary,
    LEFT(REPLACE(REPLACE(doc, '\n', ' '), '\r', ' '), 20) AS doc,
    rate
FROM docs;

-- 查询指定工具文档（例如: STREAM）
SET NAMES utf8mb4;
SELECT
    id, tool_id, url, llm_summary, doc
FROM docs
WHERE tool_id = (SELECT id FROM tools WHERE name = 'STREAM');
```

### 📤 数据导入导出

#### Linux 环境

```bash
# ===== 导出 =====
# 方法一：容器内执行
docker exec -it mysql-dev bash -c "mysqldump -u root -proot agent > /tmp/agent_backup.sql"
docker cp mysql-dev:/tmp/agent_backup.sql ./agent_backup.sql

# 方法二：宿主机直接执行（推荐）
docker exec mysql-dev \
  mysqldump -u root -proot \
    --default-character-set=utf8mb4 \
    --single-transaction \
    --hex-blob \
    --routines \
    --triggers \
    --events \
    agent > agent_backup_$(date +%Y%m%d).sql

# ===== 导入 =====
docker exec -i mysql-dev mysql -u root -proot \
  --default-character-set=utf8mb4 agent < agent_backup_20260128.sql
```

#### Windows (CMD)

```cmd
:: 导出
mysqldump -u root -proot --default-character-set=utf8mb4 ^
  --single-transaction --hex-blob --routines --triggers --events ^
  agent > agent_backup_20260127.sql

:: 导入
mysql -u root -proot --default-character-set=utf8mb4 agent < agent_backup_20260127.sql
```

> 📁 备份文件默认存储路径：`ingest/db_insert/agent_backup_*.sql`

---

## 🕷️ 文档爬取模块

### 支持的数据源

| 来源 | 状态 | 说明 |
|------|------|------|
| GitHub | ✅ 已支持 | 爬取 README、Wiki、Issues 等 |
| ReadTheDocs | ⏳ 预留 | 链接列表见 `data/doc/readthedocs_urls.xlsx` |

### 📦 数据文件

```
data/doc/
├── readthedocs_urls.xlsx    # ReadTheDocs 链接列表（备用）
└── doc_urls.csv             # 综合链接列表（含 GitHub + RTD）
```

### ▶️ 运行爬虫

需要先设置好.env文件中的数据库信息，在src/scsagent/config目录下，复制一份.env文件出来

```bash
# GitHub 文档爬取
cd scsagent && python -m ingest.crawlers.getdoc_github \
  --github-url https://github.com/LiQian-XC/sctour \
  --base-local-path D:/1Grad/Code/scsagent/tests/ingest \
  --tool-name scTour

# ReadTheDocs 文档爬取（预留接口）
cd scsagent && python -m ingest.crawlers.getdoc_readthedoc
```

---
## 🐳 Docker 镜像管理

### 🔧 构建环境依赖

构建镜像前请确保以下服务已运行：

```bash
# 必要镜像
docker images | grep -E "ub22_co310|moby/buildkit|mysql"

# 必要容器
docker ps -a | grep -E "buildx_buildkit|mysql-dev"
```

| 镜像/容器 | 用途 |
|-----------|------|
| `ub22_co310_scanpy_stomicstest:3.0` | 基础运行环境 |
| `moby/buildkit:v0.24.0` | Buildx 构建后端 |
| `mysql:latest` (容器: `mysql-dev`) | 文档数据库 |

### 🚀 启动 BuildKit 服务

```bash
cd scsagent && bash buildx.sh
```

### 🏗️ 批量构建镜像

```bash
cd scsagent/tests && python build_docker_from_scratch.py -n 10
```

| 参数 | 说明 |
|------|------|
| `-n` | 单次构建批次大小（建议 ≤10，避免内存溢出） |

📊 构建结果输出：`tests/docker_build_results.csv`

---

## 📁 项目结构

```
📦 scsagent/
├── 📄 README.md
├── 📄 setup.py / pyproject.toml
├── 📁 src/scsagent/          # 🔍 Debug 模块主目录
│   └── 📄 main.py           # 单工具执行入口
├── 📁 ingest/               # 🕷️ 文档获取模块
│   └── 📁 crawlers/
│       ├── 📄 getdoc_github.py
│       └── 📄 getdoc_readthedoc.py
├── 📁 utils/                # 🔧 公共工具函数
│   ├── 📄 web_crawler.py
│   └── 📄 docker_utils.py
├── 📁 tests/                # 🧪 测试与构建脚本
│   ├── 📄 test.py           # 批量任务测试入口
│   ├── 📄 build_docker_from_scratch.py
│   └── 📁 query/scratch/    # 临时输出目录
├── 📁 docker/               # 🐳 Docker 配置
│   └── 📄 Dockerfile.py310  # 基础镜像构建模板（使用时重命名为 Dockerfile）
├── 📁 data/doc/             # 📚 文档链接数据
│   ├── 📄 readthedocs_urls.xlsx
│   └── 📄 doc_urls.csv
└── 📁 scsagent/                  # 🔄 工作流入口（兼容旧结构）
    ├── 📄 buildx.sh
    └── 📁 ingest/           # 软链或冗余，建议统一至根目录
```

> ⚠️ 项目早期"文档获取模块"独立开发，当前结构存在冗余，后续将重构统一。

---

## 🛠️ 使用指南

### 🔎 单工具调试运行

```bash
cd ~/scsagent
python -m scsagent.main \
    --query "使用 Wishbone 进行 expression patterns 任务，数据为 /workspace/input/mouse.h5ad" \
    --host_work_dir "/stomics/ai/test/Wishbone" \
    --host_data_path "/stomics/ai/data" \
    --docker_image "wh-harbor.dcs.cloud/public-library/agent_stream:1.0"
```

| 参数 | 说明 |
|------|------|
| `--query` | 自然语言任务描述 |
| `--host_work_dir` | 宿主机结果输出目录 |
| `--host_data_path` | 宿主机数据挂载路径 |
| `--docker_image` | 指定工具运行镜像 |

### 🔄 批量任务执行

```bash
cd ~/scsagent && python tests/test.py
```

📥 输入：`tests/query/scratch/tool_tasktype_query_taskfiltered.xlsx`  
📤 输出：同目录生成执行结果

