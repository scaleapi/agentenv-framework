"""Unit tests for SyncEnvClockTaskStep — round-trip + the per-server sync_time fan-out (mocked)."""
import json
from types import SimpleNamespace

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import sync_env_clock as mod
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep, resolve_virtual_time

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
    out = await _step()._sync_one(_cardless(), "http://gw", "email", CLOCK_URL)
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
    out = await _step()._sync_one(_cardless(), "http://gw", "email", CLOCK_URL)
    assert out["synced"] is True
    assert calls["uri"] == EXT_CLOCK_URI and calls["params"] == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_unadvertised_strict_raises(monkeypatch):
    async def fake_card(base_url, **kw):
        return {"capabilities": {"extensions": []}}
    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    with pytest.raises(RuntimeError):
        await _step(tolerate_missing_sync_time=False)._sync_one(_cardless(), "http://gw", "email", CLOCK_URL)


@pytest.mark.asyncio
async def test_sync_one_invokes_at_the_stored_child_card_with_no_card_read(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))
    out = await _step()._sync_one(record, "http://gw", "email", CLOCK_URL)

    assert out["synced"] is True
    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time")]
    assert json.loads(sent[0].content) == {"env_get_time_url": CLOCK_URL}


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
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a, **k: SimpleNamespace(environment_name="email"))
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


@pytest.mark.asyncio
async def test_execute_syncs_a_multi_envs_websites_alongside_its_mcp_servers(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name="email")],
                          website_envs=[SimpleNamespace(environment_name="shop"), SimpleNamespace(environment_name="blog")])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT], "shop": [_CLOCK_EXT], "blog": []}))])

    with patch("agent_env.env.env.Env.get", return_value=env):
        await _step().execute(context)

    assert [(r.method, str(r.url)) for r in sent][2:] == [
        ("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time"), ("POST", f"{_CARD}/svc/shop/agentenv/ext/sync_time"),
    ]
    record = context.metadata["clock_configurations"][0]
    assert [(s["service"], s.get("website")) for s in record["synced"]] == [("email", None), ("shop", True)]
    assert [(s["service"], s["website"], s["reason"]) for s in record["skipped"]] == [("blog", True, "not_advertised")]


@pytest.mark.asyncio
async def test_execute_syncs_a_single_website_env(monkeypatch):
    from unittest.mock import patch

    from agent_env.env.envs.website import WebsiteEnv

    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"shop": [_CLOCK_EXT]}))])

    with patch("agent_env.env.env.Env.get", return_value=WebsiteEnv(id="e1", version=1, environment_name="shop")):
        await _step().execute(context)

    assert [(r.method, str(r.url)) for r in sent][2:] == [("POST", f"{_CARD}/svc/shop/agentenv/ext/sync_time")]
    assert json.loads(sent[2].content) == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_reads_a_websites_card_live_at_its_own_gateway_path(monkeypatch):
    read = []

    async def fake_card(base_url, **kw):
        read.append(base_url)
        return {"capabilities": {"extensions": []}}

    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    assert await _step()._sync_one(None, "http://gw", "shop", CLOCK_URL, mcp=False) == {"synced": False, "reason": "not_advertised"}
    assert read == ["http://gw/svc/shop"]


@pytest.mark.asyncio
async def test_execute_syncs_a_website_sharing_an_mcp_servers_name_at_its_own_route(monkeypatch):
    """The card lists the two as children at distinct urls, so each is synced by its own card, with no card read."""
    from unittest.mock import patch

    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name="email")],
                          website_envs=[SimpleNamespace(environment_name="email")])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT], "email": [_CLOCK_EXT]}))])

    with patch("agent_env.env.env.Env.get", return_value=env):
        await _step().execute(context)

    assert [(r.method, str(r.url)) for r in sent][2:] == [
        ("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time"), ("POST", f"{_CARD}/svc/email/agentenv/ext/sync_time"),
    ]
    record = context.metadata["clock_configurations"][0]
    assert [(s["service"], s.get("website")) for s in record["synced"]] == [("email", None), ("email", True)]


@pytest.mark.asyncio
async def test_execute_without_a_card_url_syncs_the_servers_the_definition_names_read_live(monkeypatch):
    """An older record kept no card URL, so its servers come from its env definition and each card is read live at
    its gateway path."""
    from unittest.mock import patch

    import httpx

    sent, real = [], httpx.AsyncClient
    cards = {"/svc/mcp-email/.well-known/agent-env.json": {"name": "email", "capabilities": {"extensions": [_CLOCK_EXT]}},
             "/svc/shop/.well-known/agent-env.json": {"name": "shop", "capabilities": {"extensions": []}}}

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path == "/clock/state":
            return httpx.Response(200, json={"env_get_time_url": CLOCK_URL})
        return httpx.Response(200, json=cards.get(request.url.path, {}))

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    record = _cardless()

    async def invoke(uri, method, params=None, **kw):
        return {"env_get_time_url": CLOCK_URL} if method == "state" else {}

    record.invoke = invoke
    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name="email")],
                          website_envs=[SimpleNamespace(environment_name="shop")])
    context = TaskStepContext(deployed_envs=[record])

    with patch("agent_env.env.env.Env.get", return_value=env):
        await _step().execute(context)

    assert [(r.method, str(r.url)) for r in sent] == [
        ("GET", "http://gw/svc/mcp-email/.well-known/agent-env.json"), ("POST", "http://gw/svc/mcp-email/agentenv/ext/sync_time"),
        ("GET", "http://gw/svc/shop/.well-known/agent-env.json"),
    ]
    cfg = context.metadata["clock_configurations"][0]
    assert [s["service"] for s in cfg["synced"]] == ["email"]
    assert [(s["service"], s["reason"]) for s in cfg["skipped"]] == [("shop", "not_advertised")]


@pytest.mark.asyncio
async def test_strict_execute_skips_websites_but_not_mcp_servers_that_do_not_follow_the_clock(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name="email")],
                          website_envs=[SimpleNamespace(environment_name="shop"), SimpleNamespace(environment_name="legacy")])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT], "shop": []}))])

    with patch("agent_env.env.env.Env.get", return_value=env):
        await _step(tolerate_missing_sync_time=False).execute(context)

    record = context.metadata["clock_configurations"][0]
    assert [s["service"] for s in record["synced"]] == ["email"]
    assert [(s["service"], s["reason"]) for s in record["skipped"]] == [("shop", "not_advertised"), ("legacy", "no_env_card")]

    unsynced = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [], "shop": [_CLOCK_EXT]}))])
    with patch("agent_env.env.env.Env.get", return_value=env), pytest.raises(RuntimeError, match="email does not advertise"):
        await _step(tolerate_missing_sync_time=False).execute(unsynced)


_CLOCK_EXT = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time", "methods": {"sync_time": {"method": "POST"}}}}
_ZONE_EXT = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time", "methods": {"sync_time": {"method": "POST", "request": {
    "type": "object", "properties": {"env_get_time_url": {"type": "string"}, "timezone": {"type": "string"}}, "required": ["env_get_time_url"]}}}}}
_OLD_EXT = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time", "methods": {"sync_time": {"method": "POST", "request": {
    "type": "object", "properties": {"env_get_time_url": {"type": "string"}}, "required": ["env_get_time_url"]}}}}}
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


def _cardless() -> DeployedEnv:
    """A record that kept no card, which a sync reads live from the gateway."""
    return DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None,
                              sandbox_id="sb-1")


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


def _zoned(virtual_time="2026-01-01T12:00:00", timezone="America/New_York", **kw):
    return SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time=virtual_time, timezone=timezone, depends_on=[], **kw)


def test_timezone_round_trips_and_is_absent_from_an_unzoned_step():
    d = _zoned().to_dict()
    assert d["timezone"] == "America/New_York"
    assert SyncEnvClockTaskStep.from_dict(d).timezone == "America/New_York"
    assert "timezone" not in _step().to_dict()
    assert SyncEnvClockTaskStep.from_dict(_step().to_dict()).timezone is None


@pytest.mark.parametrize("name", ["", "  ", "Mars/Olympus", 7, "Europe", "x" * 5000, "../x"])
def test_an_invalid_zone_is_refused_when_the_step_is_built(name):
    with pytest.raises(ValueError, match="IANA"):
        _zoned(timezone=name)


@pytest.mark.parametrize("virtual_time, zone, armed", [
    ("2026-01-01T12:00:00", "America/New_York", "2026-01-01T17:00:00Z"),
    ("2026-07-01T12:00:00", "America/New_York", "2026-07-01T16:00:00Z"),
    ("2026-01-01T12:00:00", "UTC", "2026-01-01T12:00:00Z"),
    ("2026-01-01T12:00:00-05:00", "America/New_York", "2026-01-01T17:00:00Z"),
    ("2026-01-01T17:00:00Z", "UTC", "2026-01-01T17:00:00Z"),
    ("2026-11-01T01:30:00", "America/New_York", "2026-11-01T05:30:00Z"),
])
def test_a_wall_clock_time_is_read_in_the_zone(virtual_time, zone, armed):
    assert _zoned(virtual_time, zone).armed_virtual_time == armed


@pytest.mark.parametrize("virtual_time, zone, match", [
    ("2026-01-01T12:00:00Z", "America/New_York", "carries offset"),
    ("2026-07-01T12:00:00-05:00", "America/New_York", "carries offset"),
    ("2026-03-08T02:30:00", "America/New_York", "does not exist"),
    ("not a time", "America/New_York", "not an ISO"),
])
def test_a_time_that_disagrees_with_the_zone_is_refused(virtual_time, zone, match):
    with pytest.raises(ValueError, match=match):
        _zoned(virtual_time, zone)


def test_without_a_zone_virtual_time_is_passed_through_untouched():
    assert _step().armed_virtual_time == T0
    assert resolve_virtual_time("2026-01-01T12:00:00", None) == "2026-01-01T12:00:00"


@pytest.mark.asyncio
async def test_execute_arms_the_gateway_at_the_local_instant_and_tells_every_server_the_zone(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a, **k: SimpleNamespace(environment_name="email"))
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_ZONE_EXT]}))])

    await _zoned().execute(context)

    assert json.loads(sent[0].content)["virtual_time"] == "2026-01-01T17:00:00Z"
    assert json.loads(sent[2].content) == {"env_get_time_url": CLOCK_URL, "timezone": "America/New_York"}
    cfg = context.metadata["clock_configurations"][0]
    assert (cfg["timezone"], cfg["virtual_time"], cfg["armed_at"]) == ("America/New_York", "2026-01-01T12:00:00", "2026-01-01T17:00:00Z")


@pytest.mark.asyncio
@pytest.mark.parametrize("tolerate", [True, False])
async def test_a_server_that_cannot_take_the_zone_fails_the_step_before_any_sync_is_sent(monkeypatch, tolerate):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a, **k: SimpleNamespace(environment_name="email"))
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_OLD_EXT]}))])

    with pytest.raises(RuntimeError, match="takes no timezone"):
        await _zoned(tolerate_missing_sync_time=tolerate).execute(context)
    assert not any(str(r.url).endswith("/sync_time") for r in sent)


@pytest.mark.asyncio
async def test_an_unzoned_step_sends_the_same_body_to_an_old_and_a_new_server(monkeypatch):
    sent = _mock_http(monkeypatch)
    for ext in (_OLD_EXT, _ZONE_EXT, _CLOCK_EXT):
        await _step()._sync_one(_carded(await _composed_card({"mcp-email": [ext]})), "http://gw", "email", CLOCK_URL)
    assert [json.loads(r.content) for r in sent] == [{"env_get_time_url": CLOCK_URL}] * 3


@pytest.mark.asyncio
async def test_a_server_that_does_not_advertise_clock_is_still_skipped_when_a_zone_is_set(monkeypatch):
    _mock_http(monkeypatch)
    out = await _zoned()._sync_one(_carded(await _composed_card({"mcp-email": []})), "http://gw", "email", CLOCK_URL)
    assert out == {"synced": False, "reason": "not_advertised"}


@pytest.mark.asyncio
async def test_a_zoned_step_tells_a_website_that_takes_the_zone_and_fails_on_one_that_cannot(monkeypatch):
    """A website is tolerated for not following the clock, but one that follows it without taking the zone fails the
    step, as an MCP server does."""
    from types import SimpleNamespace
    from unittest.mock import patch

    env = SimpleNamespace(mcp_server_envs=[], website_envs=[SimpleNamespace(environment_name="shop")])
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"shop": [_ZONE_EXT]}))])
    with patch("agent_env.env.env.Env.get", return_value=env):
        await _zoned().execute(context)
    assert json.loads(sent[2].content) == {"env_get_time_url": CLOCK_URL, "timezone": "America/New_York"}
    assert context.metadata["clock_configurations"][0]["synced"][0]["website"] is True

    _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    older = TaskStepContext(deployed_envs=[_carded(await _composed_card({"shop": [_OLD_EXT]}))])
    with patch("agent_env.env.env.Env.get", return_value=env), pytest.raises(RuntimeError, match="shop advertises .* takes no timezone"):
        await _zoned().execute(older)


@pytest.mark.asyncio
async def test_a_website_sharing_an_mcp_servers_name_is_synced_with_the_envs_control_headers(monkeypatch):
    import dataclasses
    from unittest.mock import patch

    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name="email")],
                          website_envs=[SimpleNamespace(environment_name="email")])
    record = dataclasses.replace(_carded(await _composed_card({"mcp-email": [_CLOCK_EXT], "email": [_CLOCK_EXT]})), control_token="ctl")
    context = TaskStepContext(deployed_envs=[record])

    with patch("agent_env.env.env.Env.get", return_value=env):
        await _step().execute(context)

    website_sync = [r for r in sent if str(r.url) == f"{_CARD}/svc/email/agentenv/ext/sync_time"]
    assert len(website_sync) == 1
    assert {k: website_sync[0].headers.get(k) for k in record.control_headers} == record.control_headers
    assert [s["service"] for s in context.metadata["clock_configurations"][0]["synced"]] == ["email", "email"]


@pytest.mark.asyncio
async def test_a_multi_envs_computer_is_synced_at_its_own_address_with_its_own_control_token(monkeypatch):
    """Not through the gateway's /svc/desktop/, which carries the gateway's token, not the computer's."""
    import dataclasses
    from unittest.mock import patch

    from agent_env.env.env import DeployedSandboxEnv
    from agent_env.env.gateway.constants import WELL_KNOWN_PATH

    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    computer = DeployedSandboxEnv(env_id="desk", env_version=1, sandbox_id="sb-desk", control_token="desk-token",
                                  environment_card_url=f"https://desk.example{WELL_KNOWN_PATH}",
                                  environment_card={"name": "desktop", "url": "/agentenv",
                                                    "capabilities": {"extensions": [_CLOCK_EXT]}})
    record = dataclasses.replace(_carded(await _composed_card({"mcp-email": [_CLOCK_EXT], "desktop": [_CLOCK_EXT]})),
                                 control_token="gateway-token", child_envs={"desktop": dataclasses.asdict(computer)})
    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name="email")], website_envs=[])

    with patch("agent_env.env.env.Env.get", return_value=env):
        await _step().execute(TaskStepContext(deployed_envs=[record]))

    syncs = {str(r.url): r.headers for r in sent if r.url.path.endswith("/sync_time")}
    assert set(syncs) == {f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time", "https://desk.example/agentenv/ext/sync_time"}
    assert {k: syncs["https://desk.example/agentenv/ext/sync_time"].get(k) for k in computer.control_headers} == \
        computer.control_headers


def test_a_padded_zone_is_stored_and_sent_stripped():
    s = _zoned(timezone=" America/New_York ")
    assert s.timezone == "America/New_York" and s.to_dict()["timezone"] == "America/New_York"
