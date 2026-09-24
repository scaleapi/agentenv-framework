"""Legacy (pre-/v1) data-plane wire protocol for MCP server / website envs.

Standalone async helpers for the current gateway endpoints: ``POST /api/reset``
(with an MCP-tool fallback for older servers), ``POST /api/add``, and
``GET /export-state``. Mirrors the module-level function style of
``agent_env.a2a_agent.protocol`` — callers pass URLs + params, not ``self``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MCP_MAX_RETRIES = 5


def environment_base_url(gateway_url: str, environment_name: str, mcp: bool = True) -> str:
    """Gateway path for a service: ``/svc/mcp-{name}`` (MCP) or ``/svc/{name}`` (website)."""
    prefix = "mcp-" if mcp else ""
    return f"{gateway_url}/svc/{prefix}{environment_name}"


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


async def export_state(gateway_url: str, environment_name: str, timeout: int = 60, verify: bool = True) -> dict:
    """GET ``/svc/mcp-{name}/export-state`` — the legacy state snapshot for one service."""
    base_url = environment_base_url(gateway_url, environment_name, mcp=True)
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.get(f"{base_url}/export-state", timeout=timeout)
        response.raise_for_status()
        return response.json()


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
