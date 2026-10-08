"""Unit tests for SyncEnvClockTaskStep — round-trip + the per-server sync_time fan-out (mocked)."""
import json

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import sync_env_clock as mod
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep

T0 = "2026-01-01T00:00:00Z"
CLOCK_URL = "http://gateway:9000/clock/time"


def _step(**kw):
    return SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time=T0, depends_on=[], **kw)


def test_round_trip():
    s = _step(virtual_seconds_per_real_second=3600, tolerate_missing_sync_time=True)
    d = s.to_dict()
    s2 = SyncEnvClockTaskStep.from_dict(d)
    assert d["type"] == "sync_env_clock"
    assert s2.virtual_time == T0 and s2.virtual_seconds_per_real_second == 3600 and s2.tolerate_missing_sync_time is True


def test_defaults():
    s = SyncEnvClockTaskStep(id="s", version=1, env_id="e", virtual_time=T0)
    assert s.virtual_seconds_per_real_second == 1.0 and s.tolerate_missing_sync_time is True


@pytest.mark.asyncio
async def test_sync_one_skips_unadvertised(monkeypatch):
    async def fake_card(base_url, **kw):
        return {"capabilities": {"extensions": []}}
    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    out = await _step()._sync_one(None, "http://gw", "email", CLOCK_URL)
    assert out == {"synced": False, "reason": "not_advertised"}


@pytest.mark.asyncio
async def test_sync_one_invokes_advertised(monkeypatch):
    card = {"capabilities": {"extensions": [
        {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time",
                                          "methods": {"sync_time": {"method": "POST"}}}}]}}
    calls = {}

    async def fake_card(base_url, **kw):
        return card

    async def fake_invoke(base_url, c, uri, params=None, **kw):
        calls.update(base_url=base_url, uri=uri, params=params)
        return {"ok": True}

    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    monkeypatch.setattr(mod.protocol_v1, "invoke_extension", fake_invoke)
    out = await _step()._sync_one(None, "http://gw", "email", CLOCK_URL)
    assert out["synced"] is True
    assert calls["uri"] == EXT_CLOCK_URI and calls["params"] == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_unadvertised_strict_raises(monkeypatch):
    async def fake_card(base_url, **kw):
        return {"capabilities": {"extensions": []}}
    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    with pytest.raises(RuntimeError):
        await _step(tolerate_missing_sync_time=False)._sync_one(None, "http://gw", "email", CLOCK_URL)


@pytest.mark.asyncio
async def test_sync_one_invokes_at_the_stored_child_card_with_no_card_read(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))
    out = await _step()._sync_one(record, "http://gw", "email", CLOCK_URL)

    assert out["synced"] is True
    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time")]
    assert json.loads(sent[0].content) == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_resolves_a_child_endpoint_outside_the_rpc_path_against_the_child(monkeypatch):
    sent = _mock_http(monkeypatch)
    ext = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/ext/clock/sync-time", "methods": {"sync_time": {"method": "POST"}}}}
    record = _carded(await _composed_card({"mcp-email": [ext]}))

    assert (await _step()._sync_one(record, "http://gw", "email", CLOCK_URL))["synced"] is True
    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/ext/clock/sync-time")]


@pytest.mark.asyncio
async def test_sync_one_resolves_a_method_endpoint_against_the_child(monkeypatch):
    sent = _mock_http(monkeypatch)
    ext = {"uri": EXT_CLOCK_URI, "params": {"methods": {"sync_time": {"method": "POST", "endpoint": "/agentenv/ext/sync_time"}}}}
    record = _carded(await _composed_card({"mcp-email": [ext]}))

    await _step()._sync_one(record, "http://gw", "email", CLOCK_URL)
    assert [str(r.url) for r in sent] == [f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time"]


@pytest.mark.asyncio
async def test_sync_one_posts_sync_time_to_the_child_at_a_path_where_the_gateway_serves_only_get_time(monkeypatch):
    sent = _mock_http(monkeypatch)
    ext = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/clock/time", "methods": {"sync_time": {"method": "POST"}}}}
    record = _carded(await _composed_card({"mcp-email": [ext]}))

    assert (await _step()._sync_one(record, "http://gw", "email", CLOCK_URL))["synced"] is True
    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/clock/time")]


@pytest.mark.asyncio
async def test_sync_one_skips_or_raises_for_a_child_env_missing_from_the_stored_card(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded(await _composed_card({"mcp-slack": [_CLOCK_EXT]}))

    assert await _step()._sync_one(record, "http://gw", "email", CLOCK_URL) == {"synced": False, "reason": "no_env_card"}
    with pytest.raises(RuntimeError, match="email has no env card"):
        await _step(tolerate_missing_sync_time=False)._sync_one(record, "http://gw", "email", CLOCK_URL)
    assert sent == []


@pytest.mark.asyncio
async def test_execute_syncs_each_child_env_through_the_stored_card(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr(SyncEnvClockTaskStep, "_service_names", lambda self, deployed: ["email"])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))])

    await _step().execute(context)

    assert [(r.method, str(r.url)) for r in sent] == [
        ("PUT", f"{_CARD}/clock/set-time"), ("GET", f"{_CARD}/clock/state"),
        ("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time"),
    ]
    assert json.loads(sent[0].content) == {"virtual_time": T0, "virtual_seconds_per_real_second": 1.0}
    assert [s["service"] for s in context.metadata["clock_configurations"][0]["synced"]] == ["email"]


@pytest.mark.asyncio
async def test_execute_raises_before_any_request_when_the_card_lacks_set_time(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded({"name": "gw", "capabilities": {"extensions": []}, "children_environments": []})
    with pytest.raises(EnvCapabilityUnsupported, match="does not offer 'set_time' on urn:agentenv:clock/v1"):
        await _step().execute(TaskStepContext(deployed_envs=[record]))
    assert sent == []


@pytest.mark.asyncio
async def test_execute_keeps_the_arm_failure_message(monkeypatch):
    _mock_http(monkeypatch, set_time_status=409)
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))])
    with pytest.raises(RuntimeError, match=r"clock arm failed \(HTTP 409\): "):
        await _step().execute(context)


_CLOCK_EXT = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time", "methods": {"sync_time": {"method": "POST"}}}}
# The card's address differs from the gateway URL, so a call built from the stored card is told apart from one built from /svc/... itself.
_CARD = "https://sandbox.example/sb-1"


async def _composed_card(children: dict[str, list]) -> dict:
    """The real gateway's composed card over child envs at the given gateway keys, each advertising the given extensions."""
    from agent_env.env.gateway.gateway import Gateway

    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={key: f"http://{key}:18765" for key in children})

    async def fetch(client, key, base_url):
        card = {"name": key.removeprefix("mcp-"), "url": "/agentenv", "capabilities": {"extensions": children[key]}}
        return gw._rewrite_child_card(key, card)

    gw._fetch_child_card = fetch
    return json.loads((await gw._serve_env_card()).body)


def _carded(card: dict) -> DeployedEnv:
    from agent_env.env.gateway.constants import WELL_KNOWN_PATH

    return DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None,
                       sandbox_id="sb-1", environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}", environment_card=card)


def _mock_http(monkeypatch, state: dict | None = None, set_time_status: int = 200):
    """Record every request: `/clock/state` answers `state`, `/clock/set-time` `set_time_status`, anything else an empty object."""
    import httpx

    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/clock/set-time"):
            return httpx.Response(set_time_status, json={})
        return httpx.Response(200, json=state if request.url.path.endswith("/clock/state") else {})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
