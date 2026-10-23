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


def _model_headers(raw: str | None) -> dict[str, str]:
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
        headers = _model_headers(environ.get(MODEL_HEADERS_ENV))
        if headers:
            provider["http_headers"] = headers
        cfg.setdefault("model_providers", {})["litellm"] = provider
    _write(cfg)


def _server_entry(url: str, role: str | None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a codex `mcp_servers.<name>` config table.

    Wraps the URL in `mcp-remote` (stdio shim), a Node client that trusts the
    certificates the agent's environment names in `_TRUST_ENV`. Each header is passed as a `--header "Name: Value"` arg; `AgentEnv-Role: <role>`
    is merged in (role wins on collision) so the env gateway can apply per-role tool
    filtering.
    """
    args: list[str] = [url, "--allow-http"]
    merged = dict(headers or {})
    if role:
        merged["AgentEnv-Role"] = role
    for k, v in merged.items():
        args += ["--header", f"{k}: {v}"]
    return {
        "command": "mcp-remote",
        "args": args,
        "env": {name: os.environ[name] for name in _TRUST_ENV if os.environ.get(name)},
        "startup_timeout_sec": 60,
        # Codex silently drops non-`required` servers that miss its hard-coded
        # 1s startup grace, which mcp-remote always does — but a `required`
        # server that fails to connect aborts session creation entirely.
        "required": True,
    }


async def rewrite_mcp_servers_role(servers: dict[str, dict], role: str | None) -> None:
    """Write every registered MCP server entry with `role` headers.

    Called before each turn, so config.toml carries the current role even though
    the role is set after individual MCP servers were registered.
    """
    async with _write_lock:
        cfg = _read()
        target = cfg.setdefault("mcp_servers", {})
        for name, info in servers.items():
            target[name] = _server_entry(info["url"], role, info.get("headers"))
        if _write(cfg):
            logger.info(
                f"Re-wrote {len(servers)} mcp_servers entries with role={role!r}"
            )
