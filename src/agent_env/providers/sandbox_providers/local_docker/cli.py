"""The docker CLI on this machine, as the local provider and its disks run it."""

import subprocess


class DockerFailed(RuntimeError):
    """A docker command that ended with ``returncode``."""

    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.returncode = returncode


def docker(*args: str, input: str | None = None, timeout: float | None = None) -> str:
    """``docker`` with ``args``, ``input`` on its stdin: its output, stripped. A failure raises DockerFailed."""
    done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, input=input)
    if done.returncode:
        raise DockerFailed(f"docker {' '.join(args)} failed: {done.stderr.strip()}", done.returncode)
    return done.stdout.strip()
