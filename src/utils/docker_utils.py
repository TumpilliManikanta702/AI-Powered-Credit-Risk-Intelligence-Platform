import os

def is_running_in_docker():
    """Check if the code is running inside a Docker container."""
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )
