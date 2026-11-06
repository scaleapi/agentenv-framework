"""Traffic capture, ``urn:agentenv:traffic/v1``: an env that records the packets its containers send and receive serves
them as capture files through its card. ``list`` (``TRAFFIC_LIST_OP``) answers a ``TrafficCaptureListing``, each file a
``TrafficCapture`` named for the container it holds, oldest first; ``get`` (``TRAFFIC_GET_OP``) streams the file a
``list`` entry names, given as its ``name`` param, as ``PCAP_MEDIA_TYPE``. Both are framework-only: they take the env's
control token (``client.control_headers``). ``serves_traffic`` reads whether a card advertises it,
``list_traffic_captures`` and ``fetch_traffic_capture`` call it as a card advertises it."""
from __future__ import annotations

from datetime import datetime

import httpx
from pydantic import BaseModel, ConfigDict

from .client import ExtensionCallError, find_extension_method, invoke_extension, request_headers

TRAFFIC_EXTENSION_URI = "urn:agentenv:traffic/v1"
TRAFFIC_LIST_OP = "list"
TRAFFIC_GET_OP = "get"
TRAFFIC_FORMAT = "pcap"
PCAP_MEDIA_TYPE = "application/vnd.tcpdump.pcap"
LIST_TIMEOUT_S = 30.0
FETCH_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class TrafficCapture(BaseModel):
    """One capture file a ``list`` names: its ``name``, which ``get`` takes, the ``container`` whose packets it holds,
    its size in ``bytes`` and when it was last written. Fields it doesn't name are ignored, so a reader keeps reading
    an env that says more."""

    model_config = ConfigDict(extra="ignore")

    name: str
    container: str
    bytes: int
    modified_at: datetime


class TrafficCaptureListing(BaseModel):
    """``list``'s answer: the files' ``format`` and each ``TrafficCapture``, oldest first within a container, so its
    files read in the order their packets were captured."""

    model_config = ConfigDict(extra="ignore")

    format: str = TRAFFIC_FORMAT
    captures: list[TrafficCapture] = []


def serves_traffic(card: dict) -> bool:
    """Whether ``card`` advertises traffic/v1's ``list``; no network."""
    return isinstance(card, dict) and find_extension_method(card, TRAFFIC_EXTENSION_URI, TRAFFIC_LIST_OP) is not None


async def list_traffic_captures(base_url: str, card: dict, *, timeout: float = LIST_TIMEOUT_S, verify: bool = True,
                                headers: dict[str, str] | None = None) -> TrafficCaptureListing:
    """The captures the env whose ``card`` is read from ``base_url`` serves, through its ``list``. Raises
    ``ValueError`` when the card doesn't advertise it or the answer isn't a ``TrafficCaptureListing``, httpx's error
    when it can't be asked, and ``ExtensionCallError`` for a non-2xx answer."""
    answer = await invoke_extension(base_url, card, TRAFFIC_EXTENSION_URI, timeout=timeout, verify=verify,
                                    method=TRAFFIC_LIST_OP, headers=headers)
    return TrafficCaptureListing.model_validate(answer)


async def fetch_traffic_capture(base_url: str, card: dict, name: str, path: str, *,
                                timeout: httpx.Timeout | float = FETCH_TIMEOUT, verify: bool = True,
                                headers: dict[str, str] | None = None) -> int:
    """Stream the capture ``name``, as a ``list`` entry names it, from the env whose ``card`` is read from
    ``base_url`` through its ``get`` to the file ``path``, and return its size. Raises ``ValueError`` when the card
    doesn't advertise ``get``, httpx's error when it can't be asked, and ``ExtensionCallError`` for a non-2xx answer,
    with nothing written."""
    op = find_extension_method(card, TRAFFIC_EXTENSION_URI, TRAFFIC_GET_OP)
    if op is None or not op["endpoint"]:
        raise ValueError(f"card does not advertise {TRAFFIC_EXTENSION_URI} {TRAFFIC_GET_OP}")
    url = f"{base_url}{op['endpoint']}"
    params = {"name": name}
    request = {"params": params} if op["method"] == "GET" else {"json": params}
    size = 0
    async with httpx.AsyncClient(verify=verify, timeout=timeout) as client:
        async with client.stream(op["method"], url, headers=request_headers(url, headers), **request) as response:
            if response.status_code >= 400:
                await response.aread()
                raise ExtensionCallError(response)
            with open(path, "wb") as out:
                async for chunk in response.aiter_bytes():
                    out.write(chunk)
                    size += len(chunk)
    return size
