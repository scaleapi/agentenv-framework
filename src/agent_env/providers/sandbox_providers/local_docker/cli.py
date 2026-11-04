"""The docker CLI on this machine, as the local provider and its disks run it."""

import subprocess


def docker(*args: str, input: str | None = None, timeout: float | None = None) -> str:
    """``docker`` with ``args``, ``input`` on its stdin: its output, stripped. A failure raises."""
    done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, input=input)
    if done.returncode:
        raise RuntimeError(f"docker {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()
