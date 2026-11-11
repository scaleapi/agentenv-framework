"""Entry point for the Gateway container."""
import asyncio
import logging
import os
import time
from urllib.parse import urlsplit

import httpx

from . import AGENT_ENV_GATEWAY_MCP_PORT, GatewayMode, InternalMCPServer
from .constants import (CONTROL_TOKEN_ENV, DEFAULT_MCP_SERVER_NAME, GATEWAY_PROVIDER_HEADERS_ENV, MCP_TRANSPORT,
                        WELL_KNOWN_PATH)
from .gateway import Gateway, card_url

logger = logging.getLogger(__name__)


def parse_internal_servers(spec: str, port: int) -> list[InternalMCPServer]:
    servers: list[InternalMCPServer] = []
    seen_names: set[str] = set()
    if not spec.strip():
        return servers
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, url = entry.split("=", 1)
            name = name.strip()
            url = url.strip()
            if not name or not url:
                raise ValueError(f"Invalid INTERNAL_MCP_SERVERS entry: '{entry}' (expected 'name=url' or 'name')")
        else:
            name = entry
            url = f"http://{name}:{port}/mcp"
        if name in seen_names:
            raise ValueError(f"Duplicate internal server name: '{name}'")
        seen_names.add(name)
        servers.append(InternalMCPServer(name=name, mcp_url=url))
    return servers


# How long to wait for each internal MCP server to become ready before giving
# up. Generous + configurable so a slow-starting server (image pull, DB
# migration, warmup) doesn't trip a premature failure that crash-loops the
# gateway. In Modal/container mode this is the *only* readiness gate (there's no
# docker-compose `depends_on`), so the old 30s ceiling was far too short.
_INTERNAL_SERVER_TIMEOUT_S = int(os.environ.get("GATEWAY_INTERNAL_SERVER_TIMEOUT_S", "180"))
_INTERNAL_SERVER_POLL_S = 2
_UNREAD = object()


def _ready_url(mcp_url: str) -> str:
    """The server's /ready URL (root sibling of /mcp)."""
    base = mcp_url.rstrip("/")
    if base.endswith("/mcp"):
        base = base[: -len("/mcp")]
    return f"{base}/ready"


async def _wait_for_server(name: str, mcp_url: str, timeout_s: int = _INTERNAL_SERVER_TIMEOUT_S) -> bool:
    """Wait until an internal MCP server is ready.

    Prefer the server's ``/ready`` endpoint (HTTP 200 = fully initialised). If
    the server has no ``/ready`` (older image → 404), fall back to the ``/mcp``
    endpoint merely responding (<500). Poll up to ``timeout_s`` so a slow server
    doesn't make the gateway give up and crash-loop.
    """
    ready_url = _ready_url(mcp_url)
    deadline = time.monotonic() + timeout_s
    last = "no response"
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(ready_url, timeout=5)
                if r.status_code == 200:
                    logger.info(f"  {name} ready via /ready (200)")
                    return True
                # No usable /ready: 404 = no such route; some MCP servers (e.g.
                # @playwright/mcp, used by the website-browser companion) answer
                # 400 to a plain GET /ready. In every non-200 case fall back to
                # /mcp liveness and keep waiting.
                m = await client.get(mcp_url, timeout=5)
                if m.status_code < 500:
                    logger.info(f"  {name} ready via /mcp fallback (/ready={r.status_code}, /mcp={m.status_code})")
                    return True
                last = f"/ready={r.status_code} /mcp={m.status_code}"
            except Exception as e:
                last = type(e).__name__
            logger.info(f"  {name} not ready yet ({last}); retrying...")
            await asyncio.sleep(_INTERNAL_SERVER_POLL_S)
    logger.error(f"{name} not ready after {timeout_s}s (last: {last})")
    return False


async def wait_for_internal_servers(servers: list[InternalMCPServer]) -> None:
    """Wait for all internal servers to be ready. Raises if any never come up."""
    t0 = time.monotonic()
    for server in servers:
        logger.info(f"Waiting for {server.name} at {server.mcp_url} (timeout {_INTERNAL_SERVER_TIMEOUT_S}s)...")
    results = await asyncio.gather(*[_wait_for_server(s.name, s.mcp_url) for s in servers])
    for server, ready in zip(servers, results):
        if not ready:
            raise RuntimeError(f"Internal server '{server.name}' not ready at {server.mcp_url}")
        logger.info(f"Internal MCP server {server.name} is ready")
    logger.info(f"All {len(servers)} internal MCP server(s) ready in {time.monotonic() - t0:.1f}s")


async def website_mcp_servers(sites: dict[str, str], taken: frozenset[str] = frozenset(),
                              timeout_s: int = _INTERNAL_SERVER_TIMEOUT_S) -> list[InternalMCPServer]:
    """The sites, by name at the address their server answers, that are MCP servers too: each as an internal server at
    the MCP endpoint its card declares, so the gateway serves its tools as it does an MCP env's. A site whose card
    declares no MCP interface, or that answers no card, is left out, as is one named as an internal server already
    (``taken``). Every site is asked at once, and one whose server can't be reached yet is asked again until
    ``timeout_s``; raises, naming them, when a site's card still can't be read then, so the gateway doesn't start
    without a site, as with an MCP server that never comes up."""
    asked = {name: url.rstrip("/") for name, url in sites.items() if name not in taken}
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as client:
        cards = await asyncio.gather(*(_site_card(client, base, deadline) for base in asked.values()))
    unread = [name for name, card in zip(asked, cards) if card is _UNREAD]
    if unread:
        raise RuntimeError(f"Website card(s) not readable after {timeout_s}s: {', '.join(sorted(unread))}")
    found: list[InternalMCPServer] = []
    for (name, base), card in zip(asked.items(), cards):
        declared = next((i.get("url") for i in (card or {}).get("additionalInterfaces") or []
                         if isinstance(i, dict) and i.get("transport") == MCP_TRANSPORT and i.get("url")), None)
        if isinstance(declared, str) and (mcp_url := _site_mcp_url(base, declared)) is not None:
            logger.info(f"Site {name} is an MCP server at {mcp_url}")
            found.append(InternalMCPServer(name=name, mcp_url=mcp_url))
        elif declared is not None:
            logger.warning(f"Site {name}'s card declares its MCP endpoint at {declared!r}, not on its own server; its "
                           "tools aren't served")
    return found


def _site_mcp_url(base: str, declared: str) -> str | None:
    """Where the site at ``base`` serves the MCP endpoint its card declares at ``declared`` (``card_url``), on the
    address's scheme and host as the address spells them, which the gateway's /svc routes use; None for another
    server."""
    resolved = card_url(base, declared)
    if resolved is None or not resolved.own_server:
        return None
    own = urlsplit(base.rstrip("/"))
    return f"{own.scheme}://{own.netloc}{resolved.path}" + (f"?{resolved.query}" if resolved.query else "")


async def _site_card(client: httpx.AsyncClient, base: str, deadline: float) -> dict | None | object:
    """The card the site at ``base`` serves, None when it serves none; asked again while its server can't be reached
    or answers 5xx, until ``deadline``, then ``_UNREAD``."""
    while True:
        try:
            answer = await client.get(base + WELL_KNOWN_PATH, timeout=5)
        except httpx.HTTPError:
            answer = None
        if answer is not None and answer.status_code < 500:
            try:
                card = answer.json() if answer.status_code == 200 else None
            except ValueError:
                return None
            return card if isinstance(card, dict) else None
        if time.monotonic() >= deadline:
            return _UNREAD
        await asyncio.sleep(_INTERNAL_SERVER_POLL_S)


def parse_name_url_map(env_str: str) -> dict[str, str]:
    # Expected format: 'slack=http://slack-website-backend:8000,email=http://email-website-backend:8000'
    urls: dict[str, str] = {}
    if not env_str.strip():
        return urls
    for entry in env_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid REST_PROXY_URLS entry: '{entry}' (expected 'name=url')")
        name, url = entry.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name:
            raise ValueError(f"Empty service name in REST_PROXY_URLS entry: '{entry}'")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL in REST_PROXY_URLS: '{url}' (must start with http:// or https://)")
        if name in urls:
            raise ValueError(f"Duplicate service name in REST_PROXY_URLS: '{name}'")
        urls[name] = url
    return urls


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    internal_mcp_servers = parse_internal_servers(
        spec=os.environ.get("INTERNAL_MCP_SERVERS", ""),
        port=int(os.environ.get("INTERNAL_MCP_PORT", str(AGENT_ENV_GATEWAY_MCP_PORT))),
    )
    website_urls = parse_name_url_map(os.environ.get("WEBSITE_URLS", ""))
    rest_proxy_urls = parse_name_url_map(os.environ.get("REST_PROXY_URLS", ""))
    internal_mcp_servers += asyncio.run(website_mcp_servers(
        {name: rest_proxy_urls.get(name, url) for name, url in website_urls.items()},
        taken=frozenset(server.name for server in internal_mcp_servers)))
    asyncio.run(wait_for_internal_servers(internal_mcp_servers))

    gateway_mode = GatewayMode(os.environ.get("GATEWAY_MODE", GatewayMode.PERFORMANCE.value))
    service_db_url = os.environ.get("SERVICE_DB_URL")
    gateway = Gateway(
        host=os.environ.get("MCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("MCP_PORT", str(AGENT_ENV_GATEWAY_MCP_PORT))),
        server_name=os.environ.get("MCP_SERVER_NAME") or DEFAULT_MCP_SERVER_NAME,
        internal_mcp_servers=internal_mcp_servers,
        website_urls=website_urls,
        rest_proxy_urls=rest_proxy_urls,
        gateway_mode=gateway_mode,
        service_db_url=service_db_url,
        control_token=os.environ.get(CONTROL_TOKEN_ENV),
        provider_headers=os.environ.get(GATEWAY_PROVIDER_HEADERS_ENV, "").split(","),
    )
    gateway.run()
