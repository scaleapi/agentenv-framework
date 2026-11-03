"""traffic/v1's client helpers: a card says whether an env serves it, ``list_traffic_captures`` reads its listing as
typed captures and ``fetch_traffic_capture`` streams one to a file, each at the endpoint and verb the card advertises,
with the headers given or granted."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.traffic import (
    PCAP_MEDIA_TYPE,
    TRAFFIC_EXTENSION_URI,
    TrafficCapture,
    fetch_traffic_capture,
    list_traffic_captures,
    serves_traffic,
)

_RealAsyncClient = httpx.AsyncClient
CARD = {"capabilities": {"extensions": [{"uri": TRAFFIC_EXTENSION_URI, "params": {"methods": {
    "list": {"method": "GET", "endpoint": "/svc/x/traffic/captures"},
    "get": {"method": "GET", "endpoint": "/svc/x/traffic/capture"}}}}]}}
LISTING = {"format": "pcap", "extra": 1, "captures": [
    {"name": "site-backend.pcap0", "container": "site-backend", "bytes": 24, "modified_at": "1970-01-01T00:16:40+00:00",
     "more": "ignored"}]}


@pytest.fixture
def seen(monkeypatch):
    requests: list[httpx.Request] = []

    def env(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers.get("authorization") != "Bearer tok":
            return httpx.Response(401, json={"ok": False, "error": {"code": "unauthorized", "message": "no token"}})
        if request.url.path.endswith("/captures"):
            return httpx.Response(200, json=LISTING)
        return httpx.Response(200, content=b"pcap-bytes", headers={"content-type": PCAP_MEDIA_TYPE})

    transport = httpx.MockTransport(env)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    return requests


def test_a_card_serves_traffic_only_when_it_advertises_list():
    assert serves_traffic(CARD)
    assert not serves_traffic({"capabilities": {"extensions": []}})
    assert not serves_traffic(None)


@pytest.mark.asyncio
async def test_list_reads_the_advertised_endpoint_as_typed_captures(seen):
    listing = await list_traffic_captures("http://env", CARD, headers=protocol_v1.control_headers("tok"))
    assert listing.format == "pcap"
    assert listing.captures == [TrafficCapture(name="site-backend.pcap0", container="site-backend", bytes=24,
                                               modified_at=datetime(1970, 1, 1, 0, 16, 40, tzinfo=timezone.utc))]
    assert [str(request.url) for request in seen] == ["http://env/svc/x/traffic/captures"]


@pytest.mark.asyncio
async def test_fetch_streams_the_named_capture_to_a_file_with_the_granted_headers(seen, tmp_path):
    path = tmp_path / "out.pcap"
    with protocol_v1.authorized({"http://env": protocol_v1.control_headers("tok")}):
        size = await fetch_traffic_capture("http://env", CARD, "site-backend.pcap0", str(path))
    assert (size, path.read_bytes()) == (10, b"pcap-bytes")
    assert str(seen[0].url) == "http://env/svc/x/traffic/capture?name=site-backend.pcap0"


@pytest.mark.asyncio
async def test_a_refusal_raises_an_extension_call_error_and_writes_nothing(seen, tmp_path):
    path = tmp_path / "out.pcap"
    with pytest.raises(protocol_v1.ExtensionCallError) as refused:
        await fetch_traffic_capture("http://env", CARD, "site-backend.pcap0", str(path), headers={})
    with pytest.raises(protocol_v1.ExtensionCallError):
        await list_traffic_captures("http://env", CARD, headers={})
    assert refused.value.code == "unauthorized" and not path.exists()


@pytest.mark.asyncio
async def test_a_card_without_traffic_is_refused_before_any_call(seen, tmp_path):
    with pytest.raises(ValueError):
        await list_traffic_captures("http://env", {}, headers={})
    with pytest.raises(ValueError):
        await fetch_traffic_capture("http://env", {}, "x.pcap0", str(tmp_path / "x"), headers={})
    assert seen == []
