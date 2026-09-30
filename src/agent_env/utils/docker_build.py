"""Build a Docker image on this machine, the one way every put command and bundle writer does it."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

DEFAULT_BUILD_PLATFORM = "linux/amd64"  # the remote sandbox VMs


class DockerBuildError(RuntimeError):
    """A local ``docker build`` that failed or couldn't start; the message carries docker's output."""


def build_image(dockerfile: Path, context: Path, tag: str, *, platform: str | None,
                build_args: Mapping[str, str] | None = None) -> None:
    """Build ``context`` with ``dockerfile`` into the local image ``tag``, for ``platform``, or for this host's
    platform when it's None or empty. Raises DockerBuildError, with the build's output, when it fails."""
    argv = ["docker", "build", *(["--platform", platform] if platform else [])]
    for name, value in (build_args or {}).items():
        argv += ["--build-arg", f"{name}={value}"]
    argv += ["-f", str(dockerfile), "-t", tag, str(context)]
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    except FileNotFoundError as e:
        raise DockerBuildError(f"docker build of {tag} failed: docker is not on PATH") from e
    if result.returncode != 0:
        raise DockerBuildError(f"docker build of {tag} failed (exit {result.returncode}):\n{result.stdout.rstrip()}")
