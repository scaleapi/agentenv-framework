"""Legacy (pre-/v1) data-plane wire protocol for MCP server / website envs.

Standalone async helpers for the current gateway endpoints: ``POST /api/reset``
(with an MCP-tool fallback for older servers), ``POST /api/add``, and
``GET /export-state``. Mirrors the module-level function style of
``agent_env.a2a_agent.protocol`` — callers pass URLs + params, not ``self``.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from typing import TYPE_CHECKING, Optional

import httpx
from agentenv_protocol import RPC_PATH, DataPart, client as protocol_v1

if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv

logger = logging.getLogger(__name__)

DEFAULT_MCP_MAX_RETRIES = 5
EXPORT_STATE_TIMEOUT_SECONDS = 60


def environment_base_url(gateway_url: Optional[str], environment_name: str, mcp: bool = True) -> str:
    """Gateway path for a service: ``/svc/mcp-{name}`` (MCP) or ``/svc/{name}`` (website)."""
    if not gateway_url:  # an env deployed without a gateway has no gateway paths
        from agent_env.env.env import EnvNeedsGateway

        raise EnvNeedsGateway(f"Reaching '{environment_name}' here")
    prefix = "mcp-" if mcp else ""
    return f"{gateway_url}/svc/{prefix}{environment_name}"


async def v1_base_url(deployed: Optional[DeployedEnv], gateway_url: Optional[str], environment_name: str, mcp: bool = True) -> Optional[str]:
    """A child env's v1 data-plane base from the stored env card, or None for legacy; without a stored card, the live probe decides."""
    if deployed is None or not deployed.environment_card:
        base_url = environment_base_url(gateway_url, environment_name, mcp=mcp)
        return base_url if await protocol_v1.supports_v1(base_url) else None
    child = deployed.get_child_env_card(environment_name)
    return deployed.environment_url + _child_path(child) if child else None



async def child_env_card(deployed: Optional[DeployedEnv], gateway_url: Optional[str], environment_name: str, timeout: int = 10) -> tuple[str, Optional[dict]]:
    """An MCP child env's card and the base its endpoints resolve against: from the stored env card, or without one a live read (None on 404)."""
    if deployed is None or not deployed.environment_card:
        base_url = environment_base_url((gateway_url or "").rstrip("/"), environment_name)
        try:
            return base_url, await protocol_v1.get_card(base_url, timeout=timeout)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return base_url, None
            raise
    child = deployed.get_child_env_card(environment_name)
    if child is None:
        return deployed.environment_url, None
    path = _child_path(child)
    return deployed.environment_url + path, _endpoints_relative_to(path, child)


def _child_path(child: dict) -> str:
    """A child env's path under the env's address, such as ``/svc/mcp-<name>``; empty for a leaf card."""
    return child.get("url", RPC_PATH).removesuffix(RPC_PATH)


def _endpoints_relative_to(path: str, child: dict) -> dict:
    """The child card with its extension endpoints relative to the child at `path`. A composing gateway prefixes an
    extension's ``params.endpoint`` with that path only when it is at or under ``RPC_PATH``, and never a method's own."""
    card = copy.deepcopy(child)
    for ext in (card.get("capabilities") or {}).get("extensions") or []:
        params = ext.get("params")
        if isinstance(params, dict) and (params.get("endpoint") or "").startswith(f"{path}/"):
            params["endpoint"] = params["endpoint"].removeprefix(path)
    return card


async def reset_via_rest(
    base_url: str,
    mock_data_path: Optional[str] = None,
    timeout: int = 30,
    verify: bool = True,
) -> dict:
    """POST ``{base_url}/api/reset``. Sends the seed path when given. Raises on non-2xx."""
    body = {"mock_data_path": mock_data_path} if mock_data_path is not None else None
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.post(f"{base_url}/api/reset", json=body, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        logger.info(f"load_environment_artifact: REST reset succeeded: {result}")
        return result


async def add_via_rest(base_url: str, file_path: str, timeout: int = 120, verify: bool = True) -> dict:
    """POST ``{base_url}/api/add`` with a backend file path. Raises on non-2xx."""
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.post(f"{base_url}/api/add", json={"file_path": file_path}, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        logger.info(f"load_environment_artifact: add response: {result}")
        return result


async def export_state(
    gateway_url: str, environment_name: str, timeout: int = EXPORT_STATE_TIMEOUT_SECONDS, verify: bool = True
) -> dict:
    """GET ``/svc/mcp-{name}/export-state`` — the legacy state snapshot for one service."""
    return await _export_state_at(environment_base_url(gateway_url, environment_name, mcp=True), timeout, verify)


async def _export_state_at(base_url: str, timeout: int, verify: bool) -> dict:
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.get(f"{base_url}/export-state", timeout=timeout)
        response.raise_for_status()
        return response.json()


async def service_state(deployed: Optional[DeployedEnv], gateway_url: Optional[str], environment_name: str) -> dict:
    """An MCP service's state as JSON: its v1 ``data/get`` answer when that is data, else ``GET /export-state``.

    A service that exports a file from ``data/get`` (a bundle of its database, to reload from) still serves its state
    as JSON at ``/export-state``.
    """
    base_url = await v1_base_url(deployed, gateway_url, environment_name, mcp=True)
    if base_url is None:
        return await export_state(gateway_url, environment_name)
    response = await protocol_v1.get_data(base_url)
    if not response.parts:
        return {}
    if isinstance(response.parts[0], DataPart):
        return response.parts[0].data
    return await _export_state_at(base_url, timeout=EXPORT_STATE_TIMEOUT_SECONDS, verify=True)


async def reset_via_mcp_tool(
    mcp_url: str,
    environment_name: str,
    mock_data_path: str,
    max_retries: int = DEFAULT_MCP_MAX_RETRIES,
) -> None:
    """Call the ``{environment_name}_reset`` MCP tool, retrying on transient failures."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    for attempt in range(max_retries):
        reset_done = False
        try:
            async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    reset_tool_name = f"{environment_name}_reset"
                    result = await session.call_tool(reset_tool_name, {"mock_data_path": mock_data_path})
                    result_text = "".join(c.text for c in result.content if hasattr(c, "text"))
                    logger.info(f"load_environment_artifact: {reset_tool_name} returned: {result_text}")
                    reset_done = True
        except BaseExceptionGroup as eg:
            if reset_done:
                logger.info(f"load_environment_artifact: succeeded (ignoring cleanup ExceptionGroup)")
                break
            logger.info(f"load_environment_artifact attempt {attempt + 1}/{max_retries} failed: {eg}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
            else:
                raise
        except Exception as e:
            logger.info(f"load_environment_artifact attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
            else:
                raise
        else:
            break


async def reset(
    gateway_url: str,
    environment_name: str,
    mock_data_path: str,
    max_retries: int = DEFAULT_MCP_MAX_RETRIES,
) -> Optional[dict]:
    """Reset an MCP service to a seed: try REST ``/api/reset``, fall back to the MCP tool."""
    base_url = environment_base_url(gateway_url, environment_name, mcp=True)
    try:
        return await reset_via_rest(base_url, mock_data_path)
    except Exception as rest_err:
        # TODO remove MCP tool fallback once all usages of MCP servers are on REST
        logger.info(f"load_environment_artifact: REST reset failed ({type(rest_err).__name__}), falling back to MCP tool")
        await reset_via_mcp_tool(f"{gateway_url}/mcp", environment_name, mock_data_path, max_retries=max_retries)
        return None
