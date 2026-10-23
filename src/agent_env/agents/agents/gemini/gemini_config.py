"""The Gemini CLI's settings.

`~/.gemini/settings.json` holds what every run shares, the changelog's hook. Each run
gets settings of its own as the CLI's system settings, which override the user's: the
task's MCP servers with the turn's role header, and the turn's system prompt as a
context file. So turns running at once never share a role or a system prompt.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROLE_HEADER = "AgentEnv-Role"
# The variable naming the CLI's system settings file; a run's own replaces the host's default, which it starts from.
SYSTEM_SETTINGS_ENV = "GEMINI_CLI_SYSTEM_SETTINGS_PATH"
_DEFAULT_SYSTEM_SETTINGS = Path("/etc/gemini-cli/settings.json")
# The CLI's context file in a workspace, which a run with a system prompt still loads beside its own.
_WORKSPACE_CONTEXT_FILE = "GEMINI.md"


def gemini_home() -> Path:
    """The CLI's state: `$GEMINI_CLI_HOME/.gemini`, else `~/.gemini`."""
    return Path(os.environ.get("GEMINI_CLI_HOME") or Path.home()) / ".gemini"


def user_settings_path() -> Path:
    return gemini_home() / "settings.json"


def settings_name(name: str) -> str:
    """An MCP server's name as the CLI takes it: Gemini splits `mcp_<server>_<tool>` on the first `_`, so none."""
    return name.replace("_", "")


def _server_entry(url: str, headers: Mapping[str, str] | None, role: str | None) -> dict[str, Any]:
    """`httpUrl` (streamable HTTP, what agent-env's MCP endpoints serve) with the registration's headers; the turn's
    role, when it has one, wins over a role header the registration names."""
    merged = dict(headers or {})
    if role:
        merged = {k: v for k, v in merged.items() if k.lower() != ROLE_HEADER.lower()}
        merged[ROLE_HEADER] = role
    entry: dict[str, Any] = {"httpUrl": url}
    if merged:
        entry["headers"] = merged
    return entry


def _default_system_settings() -> dict[str, Any]:
    try:
        settings = json.loads(_DEFAULT_SYSTEM_SETTINGS.read_text())
    except (OSError, ValueError):
        return {}
    return settings if isinstance(settings, dict) else {}


@contextmanager
def run_settings(servers: Mapping[str, Mapping[str, Any]], role: str | None,
                 system_prompt: str | None) -> Iterator[Path]:
    """One run's system settings, in an owner-only file of its own that names `servers` with `role`; with a
    `system_prompt`, also a context file the CLI layers on its own system prompt (the CLI has no flag for one, and
    replacing its system prompt would lose its tool protocol). Both are removed when the run ends."""
    home = gemini_home()
    runs = home / "agentenv-runs"
    runs.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    settings = _default_system_settings()
    settings["mcpServers"] = {
        **(settings.get("mcpServers") if isinstance(settings.get("mcpServers"), dict) else {}),
        **{settings_name(name): _server_entry(info["url"], info.get("headers"), role) for name, info in servers.items()},
    }
    written: list[Path] = []
    try:
        if system_prompt:
            # The CLI's global context files live in its home, and only this run's settings name this one.
            context = home / f"agentenv-system-prompt-{token}.md"
            context.write_text(system_prompt)
            written.append(context)
            settings["context"] = {
                **(settings.get("context") if isinstance(settings.get("context"), dict) else {}),
                "fileName": [_WORKSPACE_CONTEXT_FILE, context.name],
            }
        path = runs / f"{token}.json"
        # It carries the MCP servers' headers.
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            json.dump(settings, handle)
        written.append(path)
        yield path
    finally:
        for path in written:
            path.unlink(missing_ok=True)
