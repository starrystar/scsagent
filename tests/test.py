import os
import sys
import pandas as pd
import subprocess
import argparse
import ipdb
import traceback
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
from typing import Optional, List, Tuple

from tests.build_docker import push_image, remove_images_in_order
from scsagent.main import run_workflow


# ===== 全局状态（主进程管理，线程安全）=====
_base_image_total_usage = {}      # str -> int：每个 base image 总共被多少任务使用
_completed_usage = defaultdict(int)  # str -> int：已完成使用该镜像的任务数
_usage_lock = Lock()


def current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def process_single_row(
    idx: int,
    row: pd.Series,
    host_data_path: str,
) -> dict:
    """处理单行，始终返回 dict（即使出错）"""
    tool = row["tool_name"]
    query = row["query"]
    docker_image = row["docker_image"]
    host_work_dir = f"/stomics/ai/experiment/wo_rag/{tool}"

    task_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    print(
        f"[{current_time_str()}] === Processing Row {idx + 1}: {tool} (task_id={task_id}) ==="
    )

    # 初始化结果
    result = {
        "idx": idx,
        "workflow_status": "not_run",
        "code": None,
        "final_image_name": None,
        "push_status": "not_run",
        "clear_status": "not_run",
        "pull_status": "not_run",  # ← 新增
        "docs_id": None,
        "error": None,
        "base_image": docker_image,  # 用于主进程清理
    }

    try:
        # Step 1: 检查本地是否有该镜像，没有则拉取（懒加载）
        local_images = get_local_docker_images()
        if docker_image not in local_images:
            if not pull_docker_image(docker_image):
                result["pull_status"] = "failed"
                print(f"[{current_time_str()}] ❌ Pull failed for {docker_image}, skipping.")
                return result
        result["pull_status"] = "success"

        # Step 2: Run workflow
        final_image_name, code, workflow_status, docs_id = run_workflow_safely(
            docker_image, query, host_work_dir, host_data_path
        )
        result.update({
            "final_image_name": final_image_name,
            "code": code,
            "workflow_status": workflow_status,
            "docs_id": docs_id,
        })

        # Step 3: 仅推送和清理最终镜像（不碰 base image！）
        if workflow_status == "success" and final_image_name:
            # result["push_status"] = push_image_safely(final_image_name)
            result["clear_status"] = cleanup_images(final_image_name)

    except Exception as e:
        error_msg = f"Unexpected error in process_single_row: {e}"
        print(f"[{current_time_str()}] ERROR: {error_msg}")
        print(traceback.format_exc())
        result["error"] = str(e) + "\n" + traceback.format_exc()

    return result


def run_workflow_safely(docker_image, query, host_work_dir, host_data_path):
    try:
        result = run_workflow(
            query=query,
            host_work_dir=host_work_dir,
            host_data_path=host_data_path,
            docker_image=docker_image,
        )
        if result.get("done", False):
            print(f"[{current_time_str()}] ✅ Workflow succeeded!")
            return (
                result.get("new_image_name"),
                result.get("code"),
                "success",
                result.get("docs_id"),
            )
        else:
            print(f"[{current_time_str()}] ❌ Workflow finished but not done.")
            return None, None, "failed", None
    except Exception as e:
        print(f"[{current_time_str()}] 💥 Workflow error: {e}")
        return None, None, "error", None


def push_image_safely(image_name: str):
    try:
        push_image(image_name)
        print(f"[{current_time_str()}] ✅ Push succeeded!")
        return "success"
    except Exception as e:
        print(f"[{current_time_str()}] ❌ Push failed: {e}")
        return "failed"


def cleanup_images(*image_args):
    """
    安全清理一组 Docker 镜像。

    ⚠️ 重要：调用者必须确保传入的镜像顺序为 **从派生镜像到基础镜像**（即依赖链的逆序）。
    例如：[final_image, intermediate_image, base_image]

    如果顺序错误（如先删 base_image），Docker 会因“镜像被子镜像引用”而拒绝删除，
    导致清理失败或残留。

    参数:
        *image_args: 可变数量的镜像名称（str 或 None）。None 和空字符串会被自动过滤。

    返回:
        str: "success", "skipped", 或 "failed"
    """
    # 过滤掉 None 和空字符串
    images = [img for img in image_args if img]

    if not images:
        print(f"[{current_time_str()}] ℹ️ No images to clean.")
        return "skipped"

    print(f"[{current_time_str()}] 🧹 Cleaning images (in order): {images}")
    try:
        remove_images_in_order(images)
        print(f"[{current_time_str()}] ✅ Cleanup done.")
        return "success"
    except Exception as e:
        print(f"[{current_time_str()}] ❌ Cleanup failed: {e}")
        return "failed"


def get_local_docker_images() -> set:
    """获取本地所有 Docker 镜像的 {repository}:{tag} 集合"""
    try:
        # 使用 docker images --format json 获取结构化输出（Docker 20.10+ 支持）
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        images = set()
        for line in result.stdout.strip().split("\n"):
            if line and "<none>" not in line:  # 过滤掉 <none>:<none> 的 dangling 镜像
                images.add(line.strip())
        return images
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[{current_time_str()}] ⚠️ Warning: Failed to get local Docker images: {e}")
        return set()


def login_to_harbor(registry: str = "wh-harbor.dcs.cloud") -> bool:
    """登录 Harbor 镜像仓库"""
    try:
        print(f"[{current_time_str()}] 🔐 Logging into Harbor registry: {registry}...")
        subprocess.run(["docker", "login", registry], check=True)
        print(f"[{current_time_str()}] ✅ Successfully logged into Harbor.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{current_time_str()}] ⚠️ Warning: Failed to log into Harbor: {e}")
        return False


def pull_docker_image(image: str) -> bool:
    """拉取指定 Docker 镜像，成功返回 True（实时显示进度）"""
    try:
        print(f"[{current_time_str()}] 📥 Pulling image: {image}")
        # 关键：不设置 stdout/stderr，让输出直接流到终端
        subprocess.run(["docker", "pull", image], check=True)
        print(f"[{current_time_str()}] ✅ Successfully pulled: {image}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{current_time_str()}] ❌ Failed to pull image: {image} (exit code: {e.returncode})")
        return False


def prepare_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        "workflow_status",
        "code",
        "final_image_name",
        "push_status",
        "clear_status",
        "pull_status",
        "docs_id",
    ]
    for col in result_columns:
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].astype("object")
    return df


def build_task_list(
    df: pd.DataFrame, tool_list: Optional[List[str]] = None
) -> Tuple[List[Tuple[int, pd.Series]], Counter]:
    skip_stats = Counter()
    tasks = []

    for idx, row in df.iterrows():
        if pd.notna(row.get("workflow_status")):
            skip_stats["skipped_existing_status"] += 1
            continue

        tool_name = str(row.get("tool_name", "")).strip()
        docker_image = str(row.get("docker_image", "")).strip()

        if not docker_image or docker_image.lower() in ("nan", "none", ""):
            skip_stats["skipped_no_docker_image"] += 1
            continue

        if tool_list is not None and tool_name not in tool_list:
            skip_stats["skipped_not_in_tool_list"] += 1
            continue

        tasks.append((idx, row))

    return tasks, skip_stats


def print_skip_summary(skip_stats: Counter, total_tasks: int) -> None:
    """打印跳过统计信息"""
    if skip_stats:
        print("Rows skipped summary:")
        for reason, cnt in skip_stats.items():
            print(f"  {reason}: {cnt}")
    print(f"Total rows qualified for run: {total_tasks}")


# ===== 新增：任务完成后的基础镜像清理逻辑 =====
def _on_task_finished(base_image: str):
    """主进程调用：当一个任务完成对 base_image 的使用后，尝试清理"""
    global _base_image_total_usage, _completed_usage

    if not base_image or pd.isna(base_image) or str(base_image).strip().lower() in ("", "nan", "none"):
        return

    base_image = str(base_image).strip()

    with _usage_lock:
        _completed_usage[base_image] += 1
        completed = _completed_usage[base_image]
        total_needed = _base_image_total_usage.get(base_image, 0)

        if completed >= total_needed:
            # 所有需要该镜像的任务都完成了！
            print(f"[{current_time_str()}] 🧹 All {total_needed} tasks using '{base_image}' are done. Attempting cleanup...")
            try:
                local_imgs = get_local_docker_images()
                if base_image in local_imgs:
                    remove_images_in_order([base_image])
                    print(f"[{current_time_str()}] ✅ Successfully removed base image: {base_image}")
                else:
                    print(f"[{current_time_str()}] ℹ️ Base image '{base_image}' not found locally (already gone).")
            except Exception as e:
                print(f"[{current_time_str()}] ⚠️ Failed to remove base image '{base_image}': {e}")
            # 清理计数
            _completed_usage.pop(base_image, None)
        else:
            print(f"[{current_time_str()}] ℹ️ Base image '{base_image}': {completed}/{total_needed} tasks completed.")


def _run_sequential(df, tasks, host_data_path, excel_path):
    # 统计总使用量
    global _base_image_total_usage
    _base_image_total_usage = Counter(
        row["docker_image"] for _, row in tasks
        if pd.notna(row["docker_image"]) and str(row["docker_image"]).strip().lower() not in ("", "nan", "none")
    )
    print(f"[{current_time_str()}] Total base image usage plan: {_base_image_total_usage}")

    global _completed_usage
    with _usage_lock:
        _completed_usage.clear()

    for idx, row in tasks:
        try:
            print(f"[{current_time_str()}] 🔧 Processing row {idx + 1}...")
            result = process_single_row(idx, row, host_data_path)
            _update_and_save(df, idx, result, excel_path)
            print(f"[{current_time_str()}] ✅ Completed row {idx + 1}")

            # 通知主进程
            _on_task_finished(row["docker_image"])

        except Exception as e:
            print(f"[{current_time_str()}] 💥 Error in row {idx + 1}: {e}")
            df.loc[idx, "workflow_status"] = "error"
            df.loc[idx, "error"] = str(e)
            df.to_excel(excel_path, index=False, engine="openpyxl")

            # 即使失败，也算完成使用
            _on_task_finished(row["docker_image"])


def _run_parallel(df, tasks, host_data_path, excel_path, max_workers):
    # 统计每个 base image 被多少任务使用
    global _base_image_total_usage
    _base_image_total_usage = Counter(
        row["docker_image"] for _, row in tasks
        if pd.notna(row["docker_image"]) and str(row["docker_image"]).strip().lower() not in ("", "nan", "none")
    )
    print(f"[{current_time_str()}] Total base image usage plan: {_base_image_total_usage}")

    global _completed_usage
    with _usage_lock:
        _completed_usage.clear()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for idx, row in tasks:
            base_img = row["docker_image"]
            future = executor.submit(process_single_row, idx, row, host_data_path)
            future_to_info[future] = (idx, base_img)

        for future in as_completed(future_to_info):
            idx, base_img = future_to_info[future]
            try:
                result = future.result()
                _update_and_save(df, idx, result, excel_path)
                print(f"[{current_time_str()}] ✅ Completed task for row {idx + 1}")

                # 通知主进程该 base image 使用完成
                _on_task_finished(base_img)

            except Exception as e:
                print(f"[{current_time_str()}] 💥 Critical error retrieving result for row {idx + 1}: {e}")
                df.loc[idx, "workflow_status"] = "critical_error"
                df.loc[idx, "error"] = str(e)
                df.to_excel(excel_path, index=False, engine="openpyxl")

                # 即使失败，也要计入完成
                _on_task_finished(base_img)


def _update_and_save(df, idx, result, excel_path):
    df.loc[idx, "workflow_status"] = result["workflow_status"]
    df.loc[idx, "code"] = result["code"]
    df.loc[idx, "final_image_name"] = result["final_image_name"]
    df.loc[idx, "push_status"] = result["push_status"]
    df.loc[idx, "clear_status"] = result["clear_status"]
    df.loc[idx, "pull_status"] = result["pull_status"]
    df.loc[idx, "docs_id"] = result.get("docs_id")
    if result.get("error"):
        df.loc[idx, "error"] = result["error"]
    df.to_excel(excel_path, index=False, engine="openpyxl")


def main(
    excel_path: Path,
    host_data_path: str,
    max_workers: int = 4,
    debug: bool = False,
    tool_list: Optional[List[str]] = None,
):
    df = pd.read_excel(excel_path, engine="openpyxl")
    login_to_harbor("wh-harbor.dcs.cloud")
    df = prepare_result_columns(df)

    tasks, skip_stats = build_task_list(df, tool_list)
    print_skip_summary(skip_stats, len(tasks))
    if not tasks:
        print("✅ No tasks to process.")
        return

    if debug:
        print("🚀 Running in DEBUG (single-process) mode...")
        _run_sequential(df, tasks, host_data_path, excel_path)
    else:
        print(f"🚀 Running in PARALLEL mode with {max_workers} workers.")
        _run_parallel(df, tasks, host_data_path, excel_path, max_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch workflow from Excel")
    parser.add_argument(
        "--excel-path",
        type=str,
        default="/home/STOmics_test/wkf/tests/query/scratch/tool_tasktype_query_taskfiltered_sampled_wo_rag.xlsx",
        help="Path to the Excel file containing tool/query/image info",
    )
    parser.add_argument(
        "--host-data-path",
        type=str,
        default="/stomics/ai/data",
        help="Host data directory path. Default: /stomics/ai/data",
    )

    args = parser.parse_args()

    excel_path = Path(args.excel_path).resolve()
    if not excel_path.exists():
        print(f"[{current_time_str()}] ❌ ERROR: Excel file not found: {excel_path}")
        exit(1)

    print(f"[{current_time_str()}] 📁 Using Excel: {excel_path}")
    print(f"[{current_time_str()}] 📂 Using data dir: {args.host_data_path}")

    try:
        main(
            excel_path=excel_path,
            host_data_path=args.host_data_path,
            max_workers=3,  # 不要超过3，我限制每个container 4核16g
            # tool_list=["CellTypist", "scvi-tools"],
            # tool_list=["Scanpy",],
            # debug=True,
        )
    except KeyboardInterrupt:
        print(f"[{current_time_str()}] 🛑 Process interrupted by user.")
        exit(130)
    except Exception as e:
        print(f"[{current_time_str()}] 💥 Unexpected error during execution:")
        traceback.print_exc()
        exit(1)
