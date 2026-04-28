# 🧬 SCSAgent - Single Cell Analysis Documentation Agent

> 单细胞分析工具文档爬取、管理与智能问答系统  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-✅-2496ED.svg)](https://www.docker.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg)](https://www.mysql.com/)

---

## 📋 使用流程总览

```
┌─────────────────────────────────────────┐
│  1️⃣ 环境准备 → 2️⃣ 数据库初始化  │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│ 3️⃣ 文档爬取 → 4️⃣ Docker镜像管理 → 5️⃣ 工具运行/调试    │
└─────────────────────────────────────────┘
```

> 🔑 **核心依赖链**：  
> 数据库(`agent`) ← 文档数据(`docs`表) ← 爬虫模块(`ingest/`) 
> 工具镜像(`Harbor仓库`) ← 运行模块(`scsagent.main`)
---

## 📁 项目结构说明

```
📦 项目根目录/
├── 📄 README.md
├── 📄 setup.py / pyproject.toml          # pip install -e . 入口
│
├── 📁 src/scsagent/                      # 🔍 Debug模块主目录
│   └── 📄 main.py                        # 单工具运行入口
│
├── 📁 ingest/                            # 🕷️ 文档获取模块（独立开发历史遗留）
│   └── 📁 crawlers/
│       ├── 📄 getdoc_github.py          # GitHub文档爬取
│       └── 📄 getdoc_readthedoc.py      # ReadTheDocs爬取（预留）
│
├── 📁 utils/                             # 🔧 公共基础依赖
│   ├── 📄 web_crawler.py                # 网页爬取工具
│   └── 📄 docker_utils.py               # Docker容器运行工具
│
├── 📁 tests/                             # 🧪 测试与构建
│   ├── 📄 test.py                       # 批量运行入口
│   ├── 📄 build_docker_from_scratch.py  # 镜像批量构建脚本
│   ├── 📄 docker_build_results.csv      # 【输出】构建结果
│   └── 📁 query/scratch/
│       └── 📄 tool_tasktype_query_taskfiltered.xlsx  # 【输入/输出】批量任务文件
│
├── 📁 docker/                            # 🐳 Docker配置
│   └── 📄 Dockerfile.py310              # 基础镜像模板（使用时去掉.py310后缀）
│
├── 📁 data/doc/                          # 📚 文档链接数据源
│   ├── 📄 readthedocs_urls.xlsx         # ReadTheDocs链接列表（备用）
│   └── 📄 doc_urls.csv                  # 综合链接（含GitHub+RTD，推荐使用）
│
├── 📁 ingest/db_insert/                  # 💾 数据库备份文件存储
│   └── 📄 agent_backup_20260128_1729.sql # 示例备份文件
│
└── 📁 scsagent/                               # 🔄 工作流兼容目录（部分脚本需在此执行）
    ├── 📄 buildx.sh                      # 启动buildkit容器
    └── 📁 ingest/                        # 兼容旧路径引用
```

---

## 🔧 前置准备（必须执行）

### 1️⃣ 安装项目依赖

```bash
# 必须在项目根目录执行
pip install -e .
```

### 2️⃣ 启动 MySQL 数据库容器

```bash
docker run -itd --name mysql-dev \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=agent \
  -p 3306:3306 \
  docker.1ms.run/library/mysql:latest
```

### 3️⃣ 导入初始数据（❗关键步骤）

> ⚠️ 数据库 `docs` 表初始为空，**必须先导入备份数据**才能进行后续查询和工具运行

```bash
# Linux 导入命令
docker exec -i mysql-dev mysql -u root -proot \
  --default-character-set=utf8mb4 agent \
  < ingest/db_insert/agent_backup_20260128_1729.sql

# Windows (CMD) 导入命令
mysql -u root -proot --default-character-set=utf8mb4 agent < agent_backup_20260127.sql
```

### 4️⃣ 验证数据库连接

```bash
docker exec -it mysql-dev /bin/bash 
mysql -u root -proot

# MySQL 内执行验证
SHOW DATABASES;      # 应看到 agent 库
USE agent;
SHOW TABLES;         # 应看到 docs, tools 等表
```

### 5 配置环境信息
复制src\scsagent\config\.env.ref为.env文件，并配置。

---

## 🗄️ 数据库管理

### 🔐 连接配置

| 参数 | 值 |
|------|-----|
| Host | `localhost` |
| Port | `3306` |
| Database | `agent` |
| Username | `root` |
| Password | `root` |

### 🔍 常用查询语句

```sql
-- 查看全部文档（列宽截断版，便于终端预览）
SELECT 
    id,
    tool_id,
    LEFT(REPLACE(REPLACE(url, '\n', ' '), '\r', ' '), 20) AS url,
    LEFT(REPLACE(REPLACE(llm_summary, '\n', ' '), '\r', ' '), 20) AS llm_summary,
    LEFT(REPLACE(REPLACE(doc, '\n', ' '), '\r', ' '), 20) AS doc,
    LEFT(REPLACE(REPLACE(html, '\n', ' '), '\r', ' '), 20) AS html,
    rate
FROM docs;

-- 查询指定工具文档（示例：查询 name='STREAM' 的工具）
SET NAMES utf8mb4;
SELECT
    id,
    tool_id,
    LEFT(REPLACE(REPLACE(url, '\n', ' '), '\r', ' '), 20) AS url,
    LEFT(REPLACE(REPLACE(llm_summary, '\n', ' '), '\r', ' '), 20) AS llm_summary,
    LEFT(REPLACE(REPLACE(doc, '\n', ' '), '\r', ' '), 20) AS doc
FROM docs
WHERE tool_id = (SELECT id FROM tools WHERE name = 'STREAM');
```

### 📤 数据导入导出

#### Linux 环境

```bash
# ===== 导出 =====
# 方法一：在docker容器内执行（简单场景）
docker exec -it mysql-dev bash -c "mysqldump -u root -proot agent > /tmp/agent_backup.sql"
docker cp mysql-dev:/tmp/agent_backup.sql ./agent_backup.sql

# 方法二：在服务器直接执行（推荐，含完整参数）
docker exec mysql-dev \
  mysqldump -u root -proot \
    --default-character-set=utf8mb4 \
    --single-transaction \
    --hex-blob \
    --routines \
    --triggers \
    --events \
    agent \
    > /home/STOmics_test/scsagent/ingest/db_insert/agent_backup_$(date +%Y%m%d).sql

# ===== 导入 =====
docker exec -i mysql-dev mysql -u root -proot \
  --default-character-set=utf8mb4 agent \
  < /home/STOmics_test/scsagent/ingest/db_insert/agent_backup_20260128_1729.sql
```

#### Windows (CMD)

```cmd
:: ===== 导出 =====
mysqldump -u root -proot ^
  --default-character-set=utf8mb4 ^
  --single-transaction ^
  --hex-blob ^
  --routines ^
  --triggers ^
  --events ^
  agent > agent_backup_20260127.sql

:: ===== 导入 =====
mysql -u root -proot --default-character-set=utf8mb4 agent < agent_backup_20260127.sql
```

> 📁 备份文件存储路径：`ingest/db_insert/agent_backup_*.sql`

---

## 🕷️ 模块一：文档爬取（数据源更新）

### 📦 链接数据文件

```
data/doc/
├── readthedocs_urls.xlsx    # ReadTheDocs 链接列表
└── doc_urls.csv             # 综合链接列表（含GitHub+ReadTheDocs，可使用）
```

### ▶️ 执行文档爬取

```bash
# GitHub 文档爬取（主用）
cd scsagent && python -m ingest.crawlers.getdoc_github \
  --github-url https://github.com/LiQian-XC/sctour \
  --base-local-path D:/1Grad/Code/scsagent/tests/ingest \
  --tool-name scTour

# ReadTheDocs 文档爬取（预留接口，当前未启用）
cd scsagent && python -m ingest.crawlers.getdoc_readthedoc
```

> 💡 爬取结果将自动存入 `agent` 数据库的 `docs` 表，可通过前述 SQL 查询验证

---

## 🐳 模块二：Docker 镜像管理

### 🔧 1. 构建环境依赖（❗构建镜像前必须检查）

> `docker build` 命令无法限制运行资源，易导致服务器内存超限告警，因此采用 `buildx` 命令构建，需确保以下服务已运行：

#### 必要镜像列表

```bash
docker images
```

| REPOSITORY | TAG | 用途 |
|-----------|-----|------|
| `wh-harbor.dcs.cloud/public-library/ub22_co310_scanpy_stomicstest` | `3.0` | 构建镜像使用的基础环境镜像 |
| `moby/buildkit` | `v0.24.0` | buildx 命令的后端构建引擎 |
| `docker.1ms.run/library/mysql` | `latest` | 文档数据库服务 |

#### 必要容器列表

```bash
docker ps -a
```

| CONTAINER | IMAGE | STATUS | 用途 |
|-----------|-------|--------|------|
| `buildx_buildkit_limited-builder0` | `moby/buildkit:v0.24.0` | Up | buildx 构建后端（必须运行） |
| `mysql-dev` | `docker.1ms.run/library/mysql:latest` | Up | MySQL 数据库服务 |

#### 启动 BuildKit 容器

```bash
cd scsagent/docker && bash buildx.sh
```

### 🏗️ 2. 批量构建镜像

```bash
cd scsagent/tests && python build_docker_from_scratch.py -n 10
```

| 参数 | 说明 | 注意事项 |
|------|------|---------|
| `-n` | 单次构建批次大小（max-batch-size） | 服务器总容量约300G，建议 ≤10，避免磁盘/内存溢出 |

📊 **输出文件**：`tests/docker_build_results.csv`（记录构建结果）

### 📄 3. 基础镜像 Dockerfile 说明

- 文件路径：`docker/Dockerfile.py310`
- 用途：构造后续运行镜像的基础环境
- 使用方法：复制并重命名为 `Dockerfile`（去掉 `.py310` 后缀）后使用
- 服务器已预置镜像：`wh-harbor.dcs.cloud/public-library/ub22_co310_scanpy_stomicstest:3.0`（3.26GB）

---

## 🚀 模块三：工具运行与调试

### 🔎 方式一：单工具独立调试

```bash
cd ~/scsagent
python -m scsagent.main \
    --query "使用Wishbone进行expression patterns任务，数据为/workspace/input/mouse.h5ad" \
    --host_work_dir "/stomics/ai/test/Wishbone" \
    --host_data_path "/stomics/ai/data" \
    --docker_image "wh-harbor.dcs.cloud/public-library/agent_stream:1.0"
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--query` | ✅ | 自然语言任务描述，用于工具匹配与参数解析 |
| `--host_work_dir` | ✅ | 宿主机结果输出目录（容器内挂载点） |
| `--host_data_path` | ✅ | 宿主机数据源目录（容器内挂载点） |
| `--docker_image` | ✅ | 指定工具运行的 Docker 镜像地址 |

> 💡 `host_work_dir` 和 `host_data_path` 需确保宿主机路径存在且有读写权限

### 🔄 方式二：批量任务执行

```bash
cd ~/scsagent && python tests/test.py
```

| 文件 | 路径 | 用途 |
|------|------|------|
| 📥 输入文件 | `tests/query/scratch/tool_tasktype_query_taskfiltered.xlsx` | 包含待执行的任务列表（工具+任务类型+查询） |
| 📤 输出文件 | 同目录生成执行结果 | 记录每个任务的运行状态与输出 |

