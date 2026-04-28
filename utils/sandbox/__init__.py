from .docker_manager import DockerManager

# 每个进程自己的 docker_manager（初始为 None）
_current_docker_manager = None

def set_docker_manager(dm: DockerManager):
    global _current_docker_manager
    _current_docker_manager = dm

def get_docker_manager():
    global _current_docker_manager
    if _current_docker_manager is None:
        raise RuntimeError("DockerManager not initialized in this process!")
    return _current_docker_manager