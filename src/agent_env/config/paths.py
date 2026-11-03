"""Where agent-env keeps per-user files outside any project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR = "agent-env"


def state_root() -> Path:
    """The per-user state root: ``$XDG_STATE_HOME/agent-env`` when that is an absolute path,
    else ``%LOCALAPPDATA%\\agent-env`` on Windows, else ``~/.local/state/agent-env``.

    A path only; nothing is created. A relative ``$XDG_STATE_HOME`` is ignored, as the XDG
    Base Directory spec requires.
    """
    xdg = os.environ.get("XDG_STATE_HOME", "")
    if os.path.isabs(xdg):
        return Path(xdg) / _APP_DIR
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if sys.platform == "win32" and os.path.isabs(local_app_data):
        return Path(local_app_data) / _APP_DIR
    return Path.home() / ".local" / "state" / _APP_DIR


def cache_root() -> Path:
    """The per-user cache root, for what can be fetched again: ``$XDG_CACHE_HOME/agentenv`` when that is an absolute
    path, else ``~/.cache/agentenv``. A path only; nothing is created."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    return (Path(xdg) if os.path.isabs(xdg) else Path.home() / ".cache") / "agentenv"
