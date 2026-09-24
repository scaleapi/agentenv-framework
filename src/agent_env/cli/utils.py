import asyncio
import os
import subprocess
from pathlib import Path

import click

from agent_env.env import Env
from agent_env.store.base import NotFoundError

DEFAULT_BUILD_PLATFORM = "linux/amd64"


def build_platform_option(f):
    """Shared `--platform` option for CLI commands that run `docker build` locally.

    Defaults to linux/amd64 (the remote sandbox VMs are amd64). Pass linux/arm64 (or an
    empty string for a host-native build) to produce an image that runs under
    `env deploy --sandbox local` on Apple Silicon; such an image will NOT run on
    remote amd64 sandboxes.
    """
    return click.option(
        "--platform",
        "build_platform",
        default=DEFAULT_BUILD_PLATFORM,
        show_default=True,
        help=(
            "docker build target platform. Default linux/amd64 (the remote sandbox VMs). "
            "Use linux/arm64 for a host-native `env deploy --sandbox local` build on "
            "Apple Silicon; pass an empty string to omit --platform entirely."
        ),
    )(f)


def docker_build_platform_args(platform: str | None) -> list[str]:
    """['--platform', <platform>] when a platform is set, else [] (host-native build)."""
    return ["--platform", platform] if platform else []


def environment_name_options(f):
    """Shared `--environment-name` option.

    Not declared `required`: commands that can derive the name from the environment card
    resolve it through :func:`resolve_environment_name` with `allow_missing=True`.
    """
    return click.option(
        "--environment-name",
        "environment_name",
        default=None,
        help="Environment name (Env Card identity; used for tool prefixes, DB schema, compose naming)",
    )(f)


def resolve_environment_name(environment_name: str | None, *, allow_missing: bool = False) -> str | None:
    """Resolve `--environment-name` to a name. When it is omitted and `allow_missing` is set (the
    MCP-server / website put path), returns None so the caller can derive the name from the
    environment's card; otherwise a missing name is a UsageError.
    """
    if not environment_name:
        if allow_missing:
            return None
        raise click.UsageError("Missing option '--environment-name'")
    return environment_name


def parse_artifact_ref(ref: str) -> tuple[str, int | None]:
    """Parse 'id' or 'id:version'. Raises click.BadParameter on malformed input."""
    if ":" not in ref:
        if not ref:
            raise click.BadParameter("artifact ref must be non-empty")
        return ref, None
    id_part, _, version_part = ref.partition(":")
    if not id_part or not version_part or ":" in version_part:
        raise click.BadParameter(
            f"Expected 'id' or 'id:<positive int>', got {ref!r}"
        )
    if not version_part.isdecimal():
        raise click.BadParameter(
            f"Version must be a positive integer, got {version_part!r} in {ref!r}"
        )
    version = int(version_part)
    if version < 1:
        raise click.BadParameter(
            f"Version must be >= 1, got {version} in {ref!r}"
        )
    return id_part, version


def detect_base_metadata() -> dict[str, str]:
    """Auto-detect non-git metadata (created_by, agent_env_version, etc.)."""
    from importlib.metadata import version
    metadata: dict[str, str] = {}
    metadata["created_by"] = os.getenv("USER", "")
    try:
        metadata["agent_env_version"] = version("agentenv-framework")
    except Exception:
        pass
    return {k: v for k, v in metadata.items() if v}


def detect_env_metadata(dockerfile: Path, context: Path) -> dict[str, str]:
    """Auto-detect metadata from a Dockerfile path and its git repo."""
    metadata = detect_base_metadata()
    metadata["dockerfile_path"] = str(dockerfile.resolve())
    metadata.update(_detect_git_metadata(context))
    return {k: v for k, v in metadata.items() if v}


def _detect_git_metadata(path: Path) -> dict[str, str]:
    """Auto-detect git metadata from a path. Returns empty dict if not in a git repo."""
    directory = path if path.is_dir() else path.parent

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), *args],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    metadata: dict[str, str] = {}
    commit = _git("rev-parse", "--short", "HEAD")
    if commit:
        metadata["git_commit"] = commit
    commit_full = _git("rev-parse", "HEAD")
    if commit_full:
        metadata["git_commit_full"] = commit_full
    commit_date = _git("log", "-1", "--format=%aI")
    if commit_date:
        metadata["git_commit_date"] = commit_date
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        metadata["git_branch"] = branch
    tag = _git("describe", "--tags", "--exact-match", "HEAD")
    if tag:
        metadata["git_tag"] = tag
    remote = _git("remote", "get-url", "origin")
    if remote:
        metadata["git_repo"] = remote
    dirty = _git("status", "--porcelain")
    if dirty is not None:
        metadata["git_dirty"] = str(dirty != "").lower()
    return metadata


def deployed_env_from_instance(env_id: str | None, instance_id: str) -> Env:
    """Rehydrate the live deployed env behind ``instance_id``.

    Goes through ``Env.from_instance_id``, which reconnects the sandbox via the
    provider named by the instance's ``sandbox_type`` — so it works on every
    backend, unlike deriving an id from the MCP URL.
    """
    try:
        env = asyncio.run(Env.from_instance_id(instance_id))
    except NotFoundError:
        raise click.UsageError(f"instance '{instance_id}' not found")
    if env_id and env_id != env.id:
        raise click.UsageError(
            f"--id '{env_id}' does not match instance '{instance_id}' (env '{env.id}')")
    return env
