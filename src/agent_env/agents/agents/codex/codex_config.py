"""Read/write helpers for ~/.codex/config.toml.

Codex has no CLI flag to pass MCP servers or model-provider config ad-hoc —
both live in config.toml. These helpers merge updates in-place under an
asyncio lock so concurrent turns don't clobber each other.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w

logger = logging.getLogger(__name__)

_write_lock = asyncio.Lock()

MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
ROLE_HEADER = "AgentEnv-Role"
_ROLE_VARIABLE = "AGENTENV_ROLE"
# Codex hands a stdio MCP server only a few of its own variables; these tell mcp-remote, a Node client, which
# certificates to trust.
_TRUST_ENV = ("NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "SSL_CERT_FILE")


def _config_path() -> Path:
    home = os.environ.get("CODEX_HOME") or str(Path.home() / ".codex")
    return Path(home) / "config.toml"


def _read() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def _write(cfg: dict[str, Any]) -> bool:
    """Replace config.toml whole, never truncating it in place: a codex CLI starting in
    another context reads it at launch. Unchanged content is left alone. Returns whether
    the file was written."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = tomli_w.dumps(cfg)
    if path.exists() and path.read_text() == text:
        return False
    # It carries MCP and model headers: keep the existing file's permissions (owner-only when new).
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with os.fdopen(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "w") as handle:
        handle.write(text)
    os.chmod(staged, mode)
    os.replace(staged, path)
    return True


def model_headers(raw: str | None) -> dict[str, str]:
    """[model.headers], from agent-env's JSON object; none when it isn't one."""
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"ignoring {MODEL_HEADERS_ENV}, which isn't JSON")
        return {}
    if not isinstance(headers, dict):
        logger.warning(f"ignoring {MODEL_HEADERS_ENV}, which isn't a JSON object")
        return {}
    return {str(name): str(value) for name, value in headers.items()}


def write_base_config(environ: Mapping[str, str]) -> None:
    """The settings every turn shares: the collaboration tools, and the model endpoint agent-env configured as a
    Responses API provider, with its headers and the run's attribution on each call."""
    cfg = _read()
    # Ships default-OFF, and the default-ON `multi_agent` is v1 scaffolding that
    # yields no collaboration tools on codex-family models.
    cfg.setdefault("features", {})["multi_agent_v2"] = True
    base_url = environ.get("LITELLM_BASE_URL") or ""
    if base_url:
        cfg["model_provider"] = "litellm"
        provider = {
            "name": "LiteLLM proxy",
            "base_url": base_url.rstrip("/") + "/openai/v1",
            "env_key": "LITELLM_API_KEY",
            "wire_api": "responses",
            # Bound codex retry/idle behavior so a hung stream fails fast
            # (~1-2 min) instead of silently reconnecting for ~25 min on the defaults
            # (stream_idle_timeout_ms=300000 x stream_max_retries=5).
            "request_max_retries": 2,
            "stream_max_retries": 2,
            "stream_idle_timeout_ms": 60000,
            # codex_bridge sets these per CLI run; codex reads each at request time and
            # sends no header for one that is unset.
            "env_http_headers": {
                "x-litellm-customer-id": "LITELLM_CUSTOMER_ID",
                "x-litellm-tags": "LITELLM_TAGS",
            },
        }
        # x-litellm-tags goes as LITELLM_TAGS instead, so a run's tags can join it.
        headers = {name: value for name, value in model_headers(environ.get(MODEL_HEADERS_ENV)).items()
                   if name.lower() != "x-litellm-tags"}
        if headers:
            provider["http_headers"] = headers
        cfg.setdefault("model_providers", {})["litellm"] = provider
    _write(cfg)


def _server_entry(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a codex `mcp_servers.<name>` config table.

    Wraps the URL in `mcp-remote` (stdio shim), a Node client that trusts the
    certificates `_TRUST_ENV` names in the agent's environment. Each header is passed
    as a `--header "Name: Value"` arg. The `AgentEnv-Role` header, which the env
    gateway filters tools by, is mcp-remote's `${AGENTENV_ROLE}`: the role the
    registration's own headers name, from the entry's env, unless a CLI run passes
    its turn's role with `role_overrides`. So the file holds no turn's role, and
    concurrent turns of different roles share it.
    """
    registered = ""
    args: list[str] = [url, "--allow-http"]
    for k, v in (headers or {}).items():
        if k.lower() == ROLE_HEADER.lower():
            registered = v
        else:
            args += ["--header", f"{k}: {v}"]
    args += ["--header", f"{ROLE_HEADER}: ${{{_ROLE_VARIABLE}}}"]
    return {
        "command": "mcp-remote",
        "args": args,
        "env": {_ROLE_VARIABLE: registered},
        "env_vars": list(_TRUST_ENV),
        "startup_timeout_sec": 60,
        # Codex silently drops non-`required` servers that miss its hard-coded
        # 1s startup grace, which mcp-remote always does — but a `required`
        # server that fails to connect aborts session creation entirely.
        "required": True,
    }


async def write_mcp_servers(servers: Mapping[str, Mapping[str, Any]]) -> None:
    """Write every registered MCP server's entry, before each turn."""
    async with _write_lock:
        cfg = _read()
        target = cfg.setdefault("mcp_servers", {})
        for name, info in servers.items():
            target[name] = _server_entry(info["url"], info.get("headers"))
        if _write(cfg):
            logger.info(f"Wrote {len(servers)} mcp_servers entries")


def role_overrides(servers: Mapping[str, Any], role: str | None) -> list[str]:
    """The `-c` overrides that give one CLI run's MCP servers its turn's role; none without one. Codex accepts only
    server names a dotted key path can name."""
    if not role:
        return []
    return [f"mcp_servers.{name}.env.{_ROLE_VARIABLE}={json.dumps(role, ensure_ascii=False)}" for name in servers]
