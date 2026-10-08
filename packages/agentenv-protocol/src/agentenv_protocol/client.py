"""Data-plane wire protocol client (JSON-RPC) for servers implementing the AgentEnv protocol."""
from __future__ import annotations

import logging

import httpx

from .manifest import INTERFACE_MANIFEST_PATH
from .transfers import WriteNamespaceGrant
from .types import INTAKE_EXTENSION_URI, MCP_PATH, MCP_TRANSPORT, METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH, AddDataResponse, GetDataResponse, Part, ResetDataResponse

logger = logging.getLogger(__name__)


async def _rpc(base_url: str, method: str, params: dict, timeout: int, verify: bool) -> dict:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.post(f"{base_url}{RPC_PATH}", json=request, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    if "error" in body:
        error = body["error"]
        raise RuntimeError(f"{method} failed: {error.get('message')} ({error.get('data') or error.get('code')})")
    return body["result"]


async def reset_data(base_url: str, timeout: int = 30, verify: bool = True) -> ResetDataResponse:
    return ResetDataResponse.model_validate(await _rpc(base_url, METHOD_RESET, {}, timeout, verify))


async def add_data(base_url: str, parts: list[Part], timeout: int = 120, verify: bool = True) -> AddDataResponse:
    params = {"parts": [p.model_dump(mode="json", exclude_none=True) for p in parts]}
    return AddDataResponse.model_validate(await _rpc(base_url, METHOD_ADD, params, timeout, verify))


async def get_data(
    base_url: str, timeout: int = 30, verify: bool = True, *, write_namespace: WriteNamespaceGrant | None = None
) -> GetDataResponse:
    """`data/get`. A server advertising `DATA_OBJECTS_EXTENSION_URI` may be handed `write_namespace` to upload
    its export under; `uploaded_object_path` reads from the answer where it did."""
    params = {} if write_namespace is None else {"write_namespace": write_namespace.model_dump(mode="json", exclude_none=True)}
    return GetDataResponse.model_validate(await _rpc(base_url, METHOD_GET, params, timeout, verify))


async def get_card(base_url: str, timeout: int = 10, verify: bool = True) -> dict:
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.get(f"{base_url}{WELL_KNOWN_PATH}", timeout=timeout)
        response.raise_for_status()
        return response.json()


async def supports_v1(base_url: str, verify: bool = True) -> bool:
    try:
        await get_card(base_url, verify=verify)
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            logger.warning(f"v1 probe {base_url}: unexpected {e.response.status_code}, treating as legacy")
        return False
    except Exception as e:
        logger.warning(f"v1 probe {base_url}: {type(e).__name__}, treating as legacy")
        return False


def find_extension(card: dict, uri: str) -> dict | None:
    """Return the EnvironmentCard extension advertised under `uri`, or None.

    Extensions live under `capabilities.extensions` (mirroring A2A's AgentCard). Tolerates a
    missing/null `capabilities` or `extensions` so legacy and no-extensions cards return None
    rather than raising.
    """
    for ext in (card.get("capabilities") or {}).get("extensions") or []:
        if ext.get("uri") == uri:
            return ext
    return None


def extension_params(card: dict, uri: str) -> dict:
    """Return the `params` of the extension at `uri`, or {} if absent or paramless."""
    ext = find_extension(card, uri)
    return (ext.get("params") or {}) if ext else {}


def find_extension_method(card: dict, uri: str, method: str) -> dict | None:
    """Return the method named `method` that the extension at `uri` advertises, or None.

    `method` names an entry of the extension's `params.methods`. The returned entry's own `method`
    key is its HTTP verb (POST unless advertised), and its `endpoint` is the method's own, else the
    extension's.
    """
    advertised = extension_params(card, uri)
    entry = (advertised.get("methods") or {}).get(method)
    if entry is None:
        return None
    return {**entry, "endpoint": entry.get("endpoint") or advertised.get("endpoint"), "method": (entry.get("method") or "POST").upper()}


def mcp_path(card: dict) -> str:
    """Return the path of the MCP endpoint a card declares, or `MCP_PATH` by convention.

    Paths are relative to the address the card was fetched from.
    """
    for interface in card.get("additionalInterfaces") or []:
        if interface.get("transport") == MCP_TRANSPORT and interface.get("url"):
            return interface["url"]
    return MCP_PATH


def intake_declaration(card: dict) -> dict | None:
    """Return the intake declaration advertised under `INTAKE_EXTENSION_URI`, or None.

    Distinct from `extension_params`: absence of the extension (or its `params`) returns None —
    "no claim", so consumers skip fit-checks — rather than `{}`, which would read as "declares an
    empty intake". Mirrors the `supports_v1` 404->legacy tolerance.
    """
    ext = find_extension(card, INTAKE_EXTENSION_URI)
    if not ext:
        return None
    params = ext.get("params")
    return params if params else None


def find_tool(card: dict, name: str) -> dict | None:
    """Return the MCP tool advertised under `name`, or None.

    `capabilities.tools` is advertisement-only (invocation stays MCP). Tolerates a missing/null
    `capabilities` or `tools` so legacy and no-tools cards return None rather than raising.
    """
    for entry in (card.get("capabilities") or {}).get("tools") or []:
        if entry.get("name") == name:
            return entry
    return None


def find_child(card: dict, name: str) -> dict | None:
    """Return the nested child EnvironmentCard named `name`, or None.

    A composed card (e.g. a gateway fronting one or more MCP servers) nests each backing
    environment's card under `children_environments`. agent-env's gateway prefixes the child's
    `url` and each extension's `params.endpoint` with the child's path on the gateway
    (`/svc/<key>`) when they are `RPC_PATH` or under it, and passes every other endpoint through.
    An operation the composed card's own extensions also offer (same extension, HTTP verb and
    endpoint, and the same method unless the child's extension lists none) is one of the gateway's
    routes; the child serves every other one, a method's own endpoint included, under its path.
    Tolerates a missing/null `children_environments` so single/leaf cards return None rather than
    raising.
    """
    for child in card.get("children_environments") or []:
        if child.get("name") == name:
            return child
    return None


async def get_interface_manifest(base_url: str, interface: str, endpoint: str = INTERFACE_MANIFEST_PATH, timeout: int = 30, verify: bool = True):
    """Fetch one interface's manifest from a server's manifest index endpoint.

    ``endpoint`` is the index path — pass the one advertised on the card under
    ``GET_INTERFACES_EXTENSION_URI`` (already gateway-rewritten) or leave the
    default when talking to a server directly. Mechanical by design: raises
    ``httpx.HTTPStatusError`` on a non-2xx response and ``ValueError`` on a
    non-JSON body; treating a 404 as "server opted out" is caller policy.
    """
    url = f"{base_url.rstrip('/')}{endpoint}/{interface}"
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()


async def invoke_extension(base_url: str, card: dict, uri: str, params: dict | None = None, timeout: int = 30, verify: bool = True, *, method: str | None = None):
    """Invoke a card extension via its advertised REST endpoint.

    Reads the endpoint and HTTP verb of the extension method named `method` from the card, so the
    card alone is enough to invoke; without `method`, the first method listed. Returns the parsed
    JSON result; raises on a non-2xx response.
    """
    ext = find_extension(card, uri)
    if ext is None:
        raise ValueError(f"extension not advertised on card: {uri}")
    advertised = ext.get("params") or {}
    name = method if method is not None else next(iter(advertised.get("methods") or {}), None)
    op = find_extension_method(card, uri, name) if name is not None else {"endpoint": advertised.get("endpoint"), "method": "POST"}
    if op is None:
        offered = ", ".join(advertised.get("methods") or {}) or "none"
        raise ValueError(f"extension {uri} does not advertise method {method!r} (advertises: {offered})")
    if not op["endpoint"]:
        raise ValueError(f"extension {uri} has no endpoint in params")
    url = f"{base_url}{op['endpoint']}"
    async with httpx.AsyncClient(verify=verify) as client:
        if op["method"] == "GET":
            response = await client.get(url, params=params or {}, timeout=timeout)
        else:
            response = await client.request(op["method"], url, json=params or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()
