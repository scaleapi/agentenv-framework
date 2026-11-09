"""Legacy (pre-/v1) data-plane wire protocol for MCP server / website envs.

Standalone async helpers for the current gateway endpoints: ``POST /api/reset``
(with an MCP-tool fallback for older servers), ``POST /api/add``, and
``GET /export-state``. Mirrors the module-level function style of
``agent_env.a2a_agent.protocol`` — callers pass URLs + params, not ``self``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import httpx
from agentenv_protocol import METHOD_GET, DataPart, client as protocol_v1
from mcp import ClientSession
from mcp.types import CallToolResult
from mcp.client.streamable_http import streamable_http_client

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


@dataclass(frozen=True)
class BackingServer:
    """A server behind an env's address, by its environment_name: an MCP server or, with ``mcp=False``, a website.

    One the env's card lists (``card_servers``) carries that ``child`` and the ``address`` its paths resolve against.
    One the card lists none for, a server serving no card, carries the ``address`` alone: it has no card, and none is
    read for it. One named by the env definition of a record whose card can't be read carries neither, so its card is
    read live at its gateway path, ``/svc/mcp-{name}`` or a website's ``/svc/{name}`` (``server_card``). One on a
    sandbox of its own, a multi env's computer or one of its apps, carries the ``record`` of the env it runs in, whose
    card lists it and whose control token its routes take, which the env's own doesn't."""

    environment_name: str
    mcp: bool = True
    child: Optional[protocol_v1.ChildEnv] = None
    address: Optional[str] = None
    # Not compared: a record isn't hashable, and a server's address already tells it apart.
    record: Optional[DeployedEnv] = dataclasses.field(default=None, compare=False)

    @property
    def card(self) -> Optional[dict]:
        """The server's card as the env's card lists it, or None."""
        return self.child.card if self.child is not None else None

    def owner(self, deployed: DeployedEnv) -> DeployedEnv:
        """The record whose control token a call to the server takes: its own ``record``, a multi env's computer's or
        one of its apps', else ``deployed``, the record of the env it was listed for."""
        return self.record or deployed


def definition_servers(env: Any) -> list[BackingServer]:
    """The servers an env definition names: a MultiEnv's MCP servers then its websites, a WebsiteEnv's website, or
    else the env's one MCP server; empty for an env with none. Read only for what the env's card can't say
    (``backing_servers``)."""
    from agent_env.env.envs.website import WebsiteEnv

    if isinstance(env, WebsiteEnv):
        return [BackingServer(env.environment_name, mcp=False)]
    servers = [BackingServer(e.environment_name) for e in getattr(env, "mcp_server_envs", None) or []]
    servers += [BackingServer(e.environment_name, mcp=False) for e in getattr(env, "website_envs", None) or []]
    if not servers and getattr(env, "environment_name", None):
        servers = [BackingServer(env.environment_name)]
    return servers


def card_servers(card: dict, address: str) -> list[BackingServer]:
    """The servers ``card``, read at ``address``, lists, each with its own card: a leaf card's env itself; else each
    child the composed card nests, after the env itself when its own card serves ``data/get``, as a computer's driver
    fronting its apps does. A child is a website when its card carries ``website/v1`` or its url is at a website's
    gateway route, ``/svc/{name}`` (an MCP server's is ``/svc/mcp-{name}``), so a website and an MCP server sharing a
    name are two servers."""
    own = protocol_v1.ChildEnv(name=card.get("name"), url=card.get("url"), card=card)
    if not isinstance(card.get("children_environments"), list):
        return [_listed(own, address)]
    servers = [_listed(own, address)] if protocol_v1.serves(card, METHOD_GET) else []
    return servers + [_listed(child, address) for child in protocol_v1.children(card)]


def _listed(child: protocol_v1.ChildEnv, address: str) -> BackingServer:
    from agent_env.env.env import website_child

    return BackingServer(child.known_as, mcp=not website_child(child), child=child, address=address)


async def backing_servers(deployed: Optional[DeployedEnv], gateway_url: Optional[str], env: Any,
                          timeout: int = 10) -> list[BackingServer]:
    """The servers behind an env as its card lists them (``card_servers``), then, behind a gateway, each server its
    definition ``env`` names that the card lists none of the same name and kind for: one serving no card, which the
    gateway leaves off its card, so it has none.

    The card is ``deployed``'s stored one when ``gateway_url`` is its own gateway or None, else read live at
    ``gateway_url`` (an override, or a record that kept no card). A record with no card URL, an older one, or no
    record, or a live read that fails, leaves the definition alone: its servers, each read live at its gateway path,
    logged once."""
    from agent_env.env.env import gateway_url_of

    gateway = gateway_url.rstrip("/") if gateway_url else None
    card, address, why = None, None, "it has no record"
    if deployed is not None and deployed.environment_card and (gateway is None or (gateway_url_of(deployed) or "").rstrip("/") == gateway):
        card, address = deployed.environment_card, deployed.environment_url
    elif deployed is not None and deployed.environment_card_url and gateway:
        try:
            card, address = await protocol_v1.get_card(gateway, timeout=timeout), gateway
        except httpx.HTTPError as e:
            why = f"its card at {gateway} could not be read ({type(e).__name__})"
    elif deployed is not None:
        why = "its record has no card URL"
    named = definition_servers(env)
    # A record's computers are its own deployment's: none when the gateway asked is another's.
    own = ((getattr(deployed, "child_envs", None) or {})
           if deployed is not None and (gateway is None or (gateway_url_of(deployed) or "").rstrip("/") == gateway) else {})
    if not isinstance(card, dict):
        logger.info(f"env {getattr(env, 'id', None)!r}: {why}, so its servers are the ones its definition names, "
                    f"each read live at its gateway path: {[(s.environment_name, s.mcp) for s in named]}")
        return _on_own_sandboxes(named, own)
    servers = card_servers(card, address)
    if not gateway:
        return _on_own_sandboxes(servers, own)
    listed = {(s.environment_name, s.mcp) for s in servers}
    return _on_own_sandboxes(servers + [dataclasses.replace(s, address=address) for s in named
                                        if (s.environment_name, s.mcp) not in listed], own)


def _on_own_sandboxes(servers: list[BackingServer], child_envs: dict[str, dict]) -> list[BackingServer]:
    """``servers``, each child env on a sandbox of its own (``DeployedGatewayEnv.child_envs``, a multi env's computers)
    listed from its own record instead of as the gateway fronts it: its card's servers at its address, each carrying the
    record, whose control token the gateway doesn't send. One whose record keeps no card is left out, and logged."""
    from agent_env.env.env import DeployedEnv

    own = []
    for name, data in child_envs.items():
        record = DeployedEnv.from_dict(data)
        if not record.environment_card or not record.environment_url:
            logger.warning(f"child env {name!r}: its record keeps no card, so none of its servers is listed")
            continue
        own += [dataclasses.replace(server, record=record)
                for server in card_servers(record.environment_card, record.environment_url)]
    return [server for server in servers if server.environment_name not in child_envs] + own


async def server_card(server: BackingServer, gateway_url: Optional[str], timeout: int = 10) -> tuple[Optional[str], Optional[dict]]:
    """A server's card and the base its endpoints resolve against: as the env's card lists it, None for a server the
    card lists none for, else read live at its gateway path (``child_env_card``)."""
    if server.address is not None:
        return server.address, server.card
    return await child_env_card(None, gateway_url, server.environment_name, timeout=timeout, mcp=server.mcp)


async def server_v1_base_url(server: BackingServer, gateway_url: Optional[str]) -> Optional[str]:
    """A server's v1 data-plane base: its card's ``url`` resolved against the env's address, None for a server the
    env's card lists none for, else as the live probe at its gateway path finds it (``v1_base_url``)."""
    if server.address is not None:
        return server.child.base_url(server.address) if server.child is not None else None
    return await v1_base_url(None, gateway_url, server.environment_name, mcp=server.mcp)


async def v1_base_url(deployed: Optional[DeployedEnv], gateway_url: Optional[str], environment_name: str, mcp: bool = True) -> Optional[str]:
    """A child env's v1 data-plane base from the stored env card, the MCP server's or with ``mcp=False`` the website's
    of that name, or None for legacy; without a stored card, the live probe
    decides. A website's probe is a card read that is None only on a 404 and raises on any other failure, so a website whose
    card could not be read is never taken for one serving no data plane."""
    if deployed is None or not deployed.environment_card:
        base_url = environment_base_url(gateway_url, environment_name, mcp=mcp)
        if not mcp:
            _, card = await child_env_card(None, gateway_url, environment_name, mcp=False)
            return base_url if card is not None else None
        return base_url if await protocol_v1.supports_v1(base_url) else None
    child = deployed.child_env(environment_name, mcp=mcp)
    return child.base_url(deployed.environment_url) if child is not None else None



async def child_env_card(deployed: Optional[DeployedEnv], gateway_url: Optional[str], environment_name: str, timeout: int = 10,
                         mcp: bool = True) -> tuple[str, Optional[dict]]:
    """A child env's card and the base its endpoints resolve against: from the stored env card, the MCP server's or
    with ``mcp=False`` the website's of that name, or without one a live read
    at the MCP server's gateway path, or with ``mcp=False`` the website's (None on 404)."""
    if deployed is None or not deployed.environment_card:
        base_url = environment_base_url((gateway_url or "").rstrip("/"), environment_name, mcp=mcp)
        try:
            return base_url, await protocol_v1.get_card(base_url, timeout=timeout)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return base_url, None
            raise
    return deployed.environment_url, deployed.get_child_env_card(environment_name, mcp=mcp)

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
    """An MCP service's state as JSON: its v1 ``data/get`` answer when that is data, asked with the deployment's
    control token, else ``GET /export-state``.

    A service that exports a file from ``data/get`` (a bundle of its database, to reload from) still serves its state
    as JSON at ``/export-state``.
    """
    base_url = await v1_base_url(deployed, gateway_url, environment_name, mcp=True)
    if base_url is None:
        return await export_state(gateway_url, environment_name)
    response = await protocol_v1.get_data(base_url,
                                          headers=deployed.control_headers_for(base_url) if deployed is not None else None)
    if not response.parts:
        return {}
    if isinstance(response.parts[0], DataPart):
        return response.parts[0].data
    return await _export_state_at(base_url, timeout=EXPORT_STATE_TIMEOUT_SECONDS, verify=True)


async def offers_tool(mcp_url: str, name: str) -> bool:
    """Whether the MCP endpoint at ``mcp_url`` lists the tool ``name``, on any page of its tools."""
    async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            cursor = None
            while True:
                page = await session.list_tools(cursor)
                if any(tool.name == name for tool in page.tools):
                    return True
                if not (cursor := page.nextCursor):
                    return False


async def reset_via_mcp_tool(
    mcp_url: str,
    environment_name: str,
    mock_data_path: Optional[str] = None,
    max_retries: int = DEFAULT_MCP_MAX_RETRIES,
) -> CallToolResult:
    """Call the ``{environment_name}_reset`` MCP tool, retrying on transient failures; what it answered. With
    ``mock_data_path``, the seed it loads; with none, an older server's tool puts back the data it boots with."""
    arguments = {} if mock_data_path is None else {"mock_data_path": mock_data_path}
    for attempt in range(max_retries):
        reset_done = False
        try:
            async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    reset_tool_name = f"{environment_name}_reset"
                    result = await session.call_tool(reset_tool_name, arguments)
                    result_text = "".join(c.text for c in result.content if hasattr(c, "text"))
                    logger.info(f"load_environment_artifact: {reset_tool_name} returned: {result_text}")
                    reset_done = True
        except BaseExceptionGroup as eg:
            if reset_done:
                logger.info("load_environment_artifact: succeeded (ignoring cleanup ExceptionGroup)")
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
    return result


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
