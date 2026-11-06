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
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

_write_lock = asyncio.Lock()

MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
ROLE_HEADER = "AgentEnv-Role"
_ROLE_VARIABLE = "AGENTENV_ROLE"
# Codex hands a stdio MCP server only a few of its own variables; these tell mcp-remote, a Node client, which
# certificates to trust.
_TRUST_ENV = ("NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "SSL_CERT_FILE")

# The CLI version CodexTool names the tools of; a test holds it to package.json's @openai/codex.
TOOLS_CLI_VERSION = "0.147.0"
# The CLI's own tools, by the names its model sees, that a selection can turn off. Some only go together, as one
# setting turns them on: the shell's two, the six collaboration tools `multi_agent_v2` brings, and the three goal tools.
# `apply_patch` is turned off through the model catalog (`catalog_overrides`), the others by `_TOOL_SWITCHES`. Left out:
# the MCP tools (`tool_search`, `list_mcp_resources`, `list_mcp_resource_templates`, `read_mcp_resource`) that come
# with MCP servers.
CodexTool = Literal["apply_patch", "create_goal", "exec_command", "followup_task", "get_goal", "interrupt_agent",
                    "list_agents", "request_user_input", "send_message", "spawn_agent", "update_goal", "update_plan",
                    "view_image", "wait_agent", "web_search", "write_stdin"]
# Each setting that turns some of the CLI's tools off: those tools, and the `-c` overrides that turn them off. Turning
# off `multi_agent_v2` alone would bring back the v1 collaboration tools behind `tool_search`, so both go.
_TOOL_SWITCHES: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (frozenset({"exec_command", "write_stdin"}), ("features.shell_tool=false",)),
    (frozenset({"update_plan"}), ("tools.update_plan.enabled=false",)),
    (frozenset({"request_user_input"}), ("tools.experimental_request_user_input.enabled=false",)),
    (frozenset({"view_image"}), ("features.view_image=false",)),
    (frozenset({"followup_task", "interrupt_agent", "list_agents", "send_message", "spawn_agent", "wait_agent"}),
     ("features.multi_agent_v2=false", "features.multi_agent=false")),
    (frozenset({"create_goal", "get_goal", "update_goal"}), ("features.goals=false",)),
    (frozenset({"web_search"}), ('web_search="disabled"',)),
)


class CodexTools(BaseModel):
    """The protocol's ``BuiltinTools[CodexTool]``, which the image's protocol floor doesn't have yet: the same fields
    and checks, so the card's schema is the same. A list must name all the tools of a setting or none of them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disable: Literal["*"] | list[CodexTool] | None = None
    only: list[CodexTool] | None = None

    @model_validator(mode="after")
    def _one_of(self) -> CodexTools:
        if (self.disable is None) == (self.only is None):
            raise ValueError("builtin_tools takes exactly one of disable and only")
        named = set(self.only if self.only is not None else [] if self.disable == "*" else self.disable)
        for tools, _ in _TOOL_SWITCHES:
            if named & tools and not tools <= named:
                raise ValueError(f"codex turns {', '.join(sorted(tools))} on and off together: name all or none")
        return self


def builtin_tools_overrides(builtin_tools: CodexTools | None) -> list[str]:
    """The `-c` overrides that turn off the CLI's own tools a selection doesn't keep; none without one. The MCP
    tools agent-env gives the agent stay with any selection."""
    if builtin_tools is None:
        return []
    if builtin_tools.only is not None:
        off = [tools for tools, _ in _TOOL_SWITCHES if not tools <= set(builtin_tools.only)]
    elif builtin_tools.disable == "*":
        off = [tools for tools, _ in _TOOL_SWITCHES]
    else:
        off = [tools for tools, _ in _TOOL_SWITCHES if tools <= set(builtin_tools.disable)]
    return [override for tools, overrides in _TOOL_SWITCHES if tools in off for override in overrides]


# No setting turns off `apply_patch`: a model's entry in the CLI's model catalog offers it when its
# `apply_patch_tool_type` is set, as it is for every GPT model 0.147.0 ships with. A selection without it points the CLI,
# with `model_catalog_json`, at a copy of the catalog it would use whose models all have the field null. The copy is
# made from the installed CLI's own catalog each time, so it follows the pinned version.
CATALOG_WITHOUT_APPLY_PATCH = "model_catalog_without_apply_patch.json"
_CATALOG_TIMEOUT_S = 60
# How much of `codex debug models`' stderr a failed catalog read keeps in its error.
_CATALOG_STDERR_CHARS = 500


def removes_apply_patch(builtin_tools: CodexTools | None) -> bool:
    """Whether a selection turns `apply_patch` off; never without one."""
    if builtin_tools is None:
        return False
    if builtin_tools.only is not None:
        return "apply_patch" not in builtin_tools.only
    return builtin_tools.disable == "*" or "apply_patch" in builtin_tools.disable


async def _catalog(timeout_s: float) -> dict[str, Any]:
    """The model catalog the CLI uses: the file config.toml's `model_catalog_json` names (relative to CODEX_HOME, as
    the CLI reads it), or else the one the installed CLI ships with, from `codex debug models --bundled`."""
    configured = _read().get("model_catalog_json")
    if configured:
        return json.loads((_config_path().parent / Path(configured).expanduser()).read_text())
    process = await asyncio.create_subprocess_exec(
        "codex", "debug", "models", "--bundled", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
    except (TimeoutError, asyncio.CancelledError) as e:
        # A turn cancelled or out of time mid-read must not leave the child running.
        if process.returncode is None:
            process.kill()
            await asyncio.shield(process.wait())
        if isinstance(e, asyncio.CancelledError):
            raise
        raise RuntimeError(f"codex debug models gave no catalog within {timeout_s:.0f}s") from None
    if process.returncode != 0:
        tail = stderr.decode(errors="replace")[-_CATALOG_STDERR_CHARS:]
        raise RuntimeError(f"codex debug models exited {process.returncode}: {tail}")
    return json.loads(stdout)


async def catalog_overrides(builtin_tools: CodexTools | None, timeout_s: float = _CATALOG_TIMEOUT_S) -> list[str]:
    """The `-c` override that points the CLI at a catalog copy without `apply_patch`, when the selection turns it off;
    none otherwise. Raises when the catalog can't be read, so a turn never gets the tool its selection turned off."""
    if not removes_apply_patch(builtin_tools):
        return []
    catalog = await _catalog(min(timeout_s, _CATALOG_TIMEOUT_S))
    for model in catalog["models"]:
        model["apply_patch_tool_type"] = None
    path = _config_path().with_name(CATALOG_WITHOUT_APPLY_PATCH)
    async with _write_lock:
        _replace(path, json.dumps(catalog, ensure_ascii=False))
    return [f"model_catalog_json={json.dumps(str(path))}"]


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
    return _replace(_config_path(), tomli_w.dumps(cfg))


def _replace(path: Path, text: str) -> bool:
    """Replace a file under CODEX_HOME whole, unless its content is already ``text``; see `_write`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return False
    # config.toml carries MCP and model headers: keep the existing file's permissions (owner-only when new).
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
