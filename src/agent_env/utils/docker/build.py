"""Build a Docker image on this machine, the one way every put command and bundle writer does it, and check the
machine's Docker daemon answers."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BUILD_PLATFORM = "linux/amd64"  # the remote sandbox VMs
DOCKER_INFO_TIMEOUT_SECONDS = 10


# Docker's names for architectures that hosts and tools also call otherwise.
_ARCHITECTURES = {"x86_64": "amd64", "x86-64": "amd64", "aarch64": "arm64", "i386": "386"}


def normalize_platform(platform: str) -> str:
    """``platform`` as ``docker build --platform`` names it, so the same platform compares equal however it's spelled:
    lower case, the architecture by Docker's name for it, and no variant but 32-bit arm's, whose v6 and v7 differ, so an
    amd64 or arm64 image is known by its architecture alone."""
    os_name, _, rest = platform.strip().lower().partition("/")
    architecture, _, variant = rest.partition("/")
    architecture = _ARCHITECTURES.get(architecture, architecture)
    if architecture == "arm" and variant:
        return f"{os_name}/{architecture}/{variant}"
    return f"{os_name}/{architecture}" if architecture else os_name


class DockerBuildError(RuntimeError):
    """A local ``docker build`` that failed or couldn't start; the message carries docker's output."""


def build_image(dockerfile: Path, context: Path, tag: str, *, platform: str | None,
                build_args: Mapping[str, str] | None = None, labels: Mapping[str, str] | None = None) -> None:
    """Build ``context`` with ``dockerfile`` into the local image ``tag``, for ``platform``, or for this host's
    platform when it's None or empty. Raises DockerBuildError, with the build's output, when it fails."""
    argv = ["docker", "build", *(["--platform", platform] if platform else [])]
    for name, value in (build_args or {}).items():
        argv += ["--build-arg", f"{name}={value}"]
    for name, value in (labels or {}).items():
        argv += ["--label", f"{name}={value}"]
    argv += ["-f", str(dockerfile), "-t", tag, str(context)]
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    except FileNotFoundError as e:
        raise DockerBuildError(f"docker build of {tag} failed: docker is not on PATH") from e
    if result.returncode != 0:
        raise DockerBuildError(f"docker build of {tag} failed (exit {result.returncode}):\n{result.stdout.rstrip()}")


@dataclass(frozen=True)
class DockerPlugin:
    """A Docker CLI plugin a run uses, and what installs it."""

    command: str  # as ``docker <command>`` runs it
    name: str
    package: str  # Docker's own Linux package
    distribution_package: str  # a Linux distribution's, such as Ubuntu's
    formula: str  # Homebrew's


DOCKER_PLUGINS = {plugin.command: plugin for plugin in (
    DockerPlugin("compose", "Compose", "docker-compose-plugin", "docker-compose-v2", "docker-compose"),
    DockerPlugin("buildx", "Buildx", "docker-buildx-plugin", "docker-buildx", "docker-buildx"))}
# Where Homebrew puts Docker's CLI plugins, which the CLI finds only once ~/.docker/config.json lists the folder.
HOMEBREW_PLUGINS = "/opt/homebrew/lib/docker/cli-plugins"


def docker_plugin_missing(plugin: str) -> str | None:
    """Why ``docker <plugin>`` can't be run here and how to install it, or None when ``docker <plugin> version``
    answers."""
    try:
        result = subprocess.run(["docker", plugin, "version"], capture_output=True, text=True, errors="replace",
                                timeout=DOCKER_INFO_TIMEOUT_SECONDS)
    except FileNotFoundError:
        return "docker isn't on PATH"
    except subprocess.TimeoutExpired:
        return f"`docker {plugin} version` didn't answer in {DOCKER_INFO_TIMEOUT_SECONDS}s"
    if result.returncode == 0:
        return None
    known = DOCKER_PLUGINS[plugin]
    return (f"`docker {plugin}` isn't installed; install Docker's {known.name} plugin: on Linux `{known.package}`, or "
            f"`{known.distribution_package}`; on macOS `brew install {known.formula}`, with \"cliPluginsExtraDirs\": "
            f'["{HOMEBREW_PLUGINS}"] in ~/.docker/config.json')


def docker_unreachable() -> str | None:
    """Why this machine's Docker daemon can't be used, or None when ``docker info`` answers."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, errors="replace",
                                timeout=DOCKER_INFO_TIMEOUT_SECONDS)
    except FileNotFoundError:
        return "docker isn't on PATH"
    except subprocess.TimeoutExpired:
        return f"`docker info` didn't answer in {DOCKER_INFO_TIMEOUT_SECONDS}s"
    if result.returncode == 0:
        return None
    reason = next((line.strip() for line in result.stderr.splitlines() if line.strip()), "")
    return "the Docker daemon isn't reachable" + (f": {reason}" if reason else "")
