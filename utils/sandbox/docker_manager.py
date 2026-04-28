import shutil
import docker
import os
import logging
from pathlib import Path
from datetime import datetime

from scsagent.config.env import CONTAINER_WORK_DIR


class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None
        self.project_dir = None
        self.container_path = CONTAINER_WORK_DIR

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_container(
        self, task_id, project_dir, image, cpu_cores=4, mem_limit="16g"
    ):
        """Create a new Docker container with the project directory mounted."""
        container_name = f"coderunner-{task_id}"
        try:
            self.logger.info(f"Creating container {container_name}...")
            self.logger.info(
                f"project_dir:{project_dir} bind to container_path:{self.container_path}"
            )
            timeout_seconds = 1500
            command = f"""
                sh -c '
                cleanup() {{
                    echo "Received signal, shutting down..." >&2
                    kill $BG_PID 2>/dev/null
                    wait $BG_PID 2>/dev/null
                    exit 0
                }}
                trap cleanup TERM INT

                timeout {timeout_seconds} tail -f /dev/null &
                BG_PID=$!

                wait $BG_PID
                ec=$?
                if [ $ec -eq 124 ]; then
                    echo "Auto-stopped after {timeout_seconds} seconds (timeout)." >&2
                elif [ $ec -ne 0 ]; then
                    echo "Background process exited with code $ec." >&2
                fi
                '
                """

            self.container = self.client.containers.run(
                image=image,
                name=container_name,
                detach=True,
                volumes={
                    project_dir: {
                        "bind": self.container_path,
                        "mode": "rw",
                    }
                },
                working_dir=self.container_path,
                command=command,
                mem_limit=mem_limit,
                memswap_limit=mem_limit,  # 禁用 swap
                cpu_period=100000,  # 默认周期 100ms
                cpu_quota=int(cpu_cores * 100000),  # 4 核 → 400000
            )
            self.project_dir = project_dir
            self.logger.info(f"Container created with ID: {self.container.id}")
            return self.container.id

        except Exception as e:
            self.logger.error(f"Failed to create/start container {container_name}: {e}")
            # 尝试清理可能残留的 "Created" 容器
            try:
                orphan = self.client.containers.get(container_name)
                orphan.remove(force=True)
                self.logger.info(f"Removed orphaned container {container_name}")
            except docker.errors.NotFound:
                pass  # 不存在就不管
            except Exception as cleanup_err:
                self.logger.warning(
                    f"Failed to clean up orphan container: {cleanup_err}"
                )
            raise  # 重新抛出原异常

    def execute_command(self, command):
        if not self.container:
            raise RuntimeError("No container is running")

        try:
            self.container.reload()  # 👈 先强制刷新状态
        except docker.errors.NotFound:
            raise RuntimeError(f"Container {self.container.short_id} has been removed")

        if self.container.status != "running":
            try:
                logs = self.container.logs(tail=50).decode("utf-8", errors="replace")
            except Exception:
                logs = "<unable to fetch logs>"
            raise RuntimeError(
                f"Container {self.container.short_id} is not running (status: {self.container.status}). "
                f"Last logs:\n{logs}"
            )

        result = self.container.exec_run(command)

        # 安全地处理输出：可能是 None、bytes，或其他异常类型
        output = result.output
        output_str = ""

        if output is not None:
            try:
                if isinstance(output, bytes):
                    output_str = output.decode("utf-8", errors="replace")
                else:
                    # 非 bytes 类型（如 str 或其他），转为字符串
                    output_str = str(output)
            except Exception as e:
                self.logger.warning(f"Failed to decode command output: {e}")
                output_str = f"<Decoding failed: {e}>"

        return {
            "exit_code": result.exit_code,
            "output": output_str,
        }

    def clear_directory(self):
        directory = self.project_dir
        path = os.path.abspath(directory)
        if not os.path.exists(path):
            self.logger.info(f"路径 {directory} 不存在")
            return

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            # 跳过需要保留的文件和目录
            if os.path.isfile(item_path):
                if item.endswith((".log", ".txt", ".py")):
                    continue
            elif os.path.isdir(item_path):
                if item == "input":
                    continue

            try:
                # 尝试删除文件或链接
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    self.logger.info(f"已删除临时文件: {item_path}")

                # 尝试删除目录（非 input）
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    self.logger.info(f"已删除临时目录: {item_path}")

            except (OSError, PermissionError) as e:
                try:
                    # 先修改权限：添加写权限
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.chmod(item_path, 0o777)  # rwx for all
                        os.unlink(item_path)
                        self.logger.info(f"已删除临时文件: {item_path}")

                    elif os.path.isdir(item_path):
                        # 递归修改目录权限
                        def make_writable_and_remove(top):
                            for root, dirs, files in os.walk(top, topdown=False):
                                for name in files:
                                    filename = os.path.join(root, name)
                                    os.chmod(filename, 0o777)
                                    os.unlink(filename)
                                for name in dirs:
                                    dirname = os.path.join(root, name)
                                    os.chmod(dirname, 0o777)
                            os.rmdir(top)

                        make_writable_and_remove(item_path)
                        self.logger.info(f"已删除临时目录: {item_path}")

                except Exception as e2:
                    self.logger.info(
                        f"临时目录删除失败（PermissionError）: {item_path}"
                    )

    def execute_python_code(self, code_text):
        """Execute Python code text in the container and return the output.

        Args:
            code_text (str): Python code to execute

        Returns:
            dict: Dictionary containing exit_code and output
        """
        if not self.container:
            raise ValueError("No container is running")

        # Generate a unique filename for the temporary Python file
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        temp_filename = formatted_time + ".py"
        temp_file_path = os.path.join(self.project_dir, temp_filename)
        temp_file_output_path = os.path.join(self.project_dir, formatted_time + ".txt")
        container_file_path = (Path(self.container_path) / temp_filename).as_posix()

        try:
            # Write the code to a temporary file
            self.logger.info(f"Writing Python code to {temp_file_path}")
            cleaned_code = code_text.encode("utf-8", errors="replace").decode("utf-8")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_code)

            # Execute the Python file in the container
            self.logger.info(f"Executing Python code from {container_file_path}")
            result = self.execute_command(f"python {container_file_path}")
            with open(temp_file_output_path, "w", encoding="utf-8") as f:
                f.write(str(result))
            return result

        except Exception as e:
            self.logger.error(f"Error executing Python code: {e}")
            raise

    def commit_container(self, image_name: str, tag="latest"):
        """Commit container to image with name `image_name`, using 'latest' tag if not specified.

        Args:
            image_name (str): Base image name (e.g., 'mytool_STOmics_test'). Tag is not expected.
                            The actual committed image will be '{image_name}:latest'.
        """
        if not self.container:
            self.logger.warning("No container to commit.")
            return None

        try:
            # 你保证传入的是不带 tag 的名字，所以我们直接用它作为 repository
            repository = image_name

            full_image_name = f"{repository}:{tag}"
            self.logger.info(
                f"Committing container {self.container.id} to image: {full_image_name}"
            )

            committed_image = self.container.commit(
                repository=repository,
                tag=tag,
                message=f"Committed from container {self.container.id}",
                author="STOmics_test",
            )

            print(f"✅ 容器已提交为新镜像: {full_image_name}")
            print(f"   镜像 ID: {committed_image.id}")

            return full_image_name

        except Exception as e:
            self.logger.error(f"Error during container commit: {e}")
            raise

    def remove_container(self):
        """仅删除容器，不 commit。"""
        if not self.container:
            self.logger.warning("No container to remove.")
            return

        try:
            self.container.remove(force=True)
            self.container = None
            self.logger.info("Container removed.")
        except Exception as e:
            self.logger.error(f"Error during container removal: {e}")
            # 可选：是否 raise？根据需求决定
            raise
