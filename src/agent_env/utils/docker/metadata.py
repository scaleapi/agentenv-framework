"""What a put command records about how an image was built: who built it, with which agent-env, from which
Dockerfile, and the git state of its context."""

import os
import subprocess
from importlib.metadata import version
from pathlib import Path


def detect_base_metadata() -> dict[str, str]:
    """Auto-detect non-git metadata (created_by, agent_env_version, etc.)."""
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
