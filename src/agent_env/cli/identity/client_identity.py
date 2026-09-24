"""Caller identity helpers for AgentEnv CLI task contexts."""

from __future__ import annotations

import re
import subprocess

_CLIENT_ID_PREFIX = "agent-env-cli"
_INVALID_GITHUB_USERNAME_CHARS_RE = re.compile(r"[^a-z0-9]+")


def get_agent_env_client_id() -> str | None:
    """Best-effort AgentEnv client id for this local CLI process."""
    username = _get_git_username()
    if username is None:
        return None
    return f"{_CLIENT_ID_PREFIX}/{username}"


def _get_git_username() -> str | None:
    """Return git user.name normalized as a GitHub-style username."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "user.name"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    username = _INVALID_GITHUB_USERNAME_CHARS_RE.sub(
        "-",
        result.stdout.strip().lower(),
    ).strip("-")
    return username or None
