"""Unit tests for SyncEnvClockTaskStep — round-trip + the per-server sync_time fan-out (mocked)."""
import json
import logging

import pytest

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.environment_universe import VirtualClockReading
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


def test_existing_positional_arguments_keep_their_meaning():
    s = SyncEnvClockTaskStep("sync", 1, "e", T0, 2.0, False, [], False, 45)
    assert (s.virtual_seconds_per_real_second, s.tolerate_missing_sync_time, s.depends_on) == (2.0, False, [])
    assert (s.fail_task_on_error, s.timeout_seconds, s.resume_from_loaded_universe) == (False, 45, False)


def test_defaults():
    s = SyncEnvClockTaskStep(id="s", version=1, env_id="e", virtual_time=T0)
    assert s.virtual_seconds_per_real_second == 1.0 and s.tolerate_missing_sync_time is True


_LEGACY_KEYS = {"env_id", "virtual_time", "virtual_seconds_per_real_second", "tolerate_missing_sync_time", "timeout_seconds"}


def test_a_step_that_does_not_resume_serializes_as_before():
    d = _step().to_dict()
    assert _LEGACY_KEYS <= d.keys() and "resume_from_loaded_universe" not in d
    assert d["virtual_seconds_per_real_second"] == 1.0
    legacy = {k: v for k, v in d.items() if k != "virtual_seconds_per_real_second"}
    assert SyncEnvClockTaskStep.from_dict(legacy).virtual_seconds_per_real_second == 1.0


def test_round_trip_resuming_from_the_loaded_universe():
    s = SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", resume_from_loaded_universe=True)
    d = s.to_dict()
    s2 = SyncEnvClockTaskStep.from_dict(d)
    assert d["resume_from_loaded_universe"] is True
    assert s2.resume_from_loaded_universe and s2.virtual_time is None and s2.virtual_seconds_per_real_second is None
    assert s2.to_dict() == d


def test_virtual_time_is_required_unless_resuming():
    with pytest.raises(ValueError, match="omit it only with resume_from_loaded_universe"):
        SyncEnvClockTaskStep(id="s", version=1, env_id="e")


_RECORDED = VirtualClockReading(virtual_time="2026-01-01T05:30:00Z", virtual_seconds_per_real_second=60.0)


class _InstanceStore:
    """The env-instance store's loaded-universe record: `loaded` per instance id, or raising `error`."""

    def __init__(self, loaded=None, error=None):
        self.loaded, self.error = loaded or {}, error

    def get_environment_universe(self, instance_id):
        if self.error:
            raise self.error
        return self.loaded.get(instance_id)


def _loaded(monkeypatch, virtual_clock=None, *, instance_store=None):
    """Put a universe recording `virtual_clock` in the local artifact store and record it as loaded into "inst-1"."""
    from agent_env.artifact.ref import ArtifactRef
    from agent_env.artifact.store import get_artifact_store

    universe = get_artifact_store().put_document(EnvironmentUniverseArtifact(
        id="snap-u", environment_artifact_refs=[ArtifactRef(id="snap-u__gmail", version=1)], virtual_clock=virtual_clock))
    store = instance_store or _InstanceStore({"inst-1": {"id": universe.id, "version": universe.version}})
    monkeypatch.setattr("agent_env.env.store.get_env_instance_store", lambda: store)


async def _execute(monkeypatch, step, instance_id="inst-1"):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr(SyncEnvClockTaskStep, "_service_names", lambda self, deployed: [])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({}), instance_id=instance_id)])
    await step.execute(context)
    return sent, context


@pytest.mark.parametrize("kw, virtual_clock, armed", [
    ({}, _RECORDED, ("2026-01-01T05:30:00Z", 60.0)),
    ({"virtual_seconds_per_real_second": 2.0}, _RECORDED, ("2026-01-01T05:30:00Z", 2.0)),
    ({"virtual_time": T0}, _RECORDED, ("2026-01-01T05:30:00Z", 60.0)),
    ({"virtual_time": T0}, None, (T0, 1.0)),
    ({"virtual_time": T0, "virtual_seconds_per_real_second": 3.0}, None, (T0, 3.0)),
], ids=["recorded", "explicit-rate-wins", "recorded-over-fallback", "fallback", "fallback-with-rate"])
@pytest.mark.asyncio
async def test_execute_arms_at_the_loaded_universes_clock_else_the_fallback(local_stores, monkeypatch, kw, virtual_clock, armed):
    _loaded(monkeypatch, virtual_clock)

    sent, context = await _execute(monkeypatch, SyncEnvClockTaskStep(
        id="sync", version=1, env_id="e1", resume_from_loaded_universe=True, **kw))

    virtual_time, rate = armed
    assert json.loads(sent[0].content) == {"virtual_time": virtual_time, "virtual_seconds_per_real_second": rate}
    entry = context.metadata["clock_configurations"][-1]
    assert (entry["virtual_time"], entry["virtual_seconds_per_real_second"]) == armed


@pytest.mark.asyncio
async def test_a_step_that_does_not_resume_ignores_the_loaded_universes_clock(local_stores, monkeypatch):
    _loaded(monkeypatch, _RECORDED)

    sent, _ = await _execute(monkeypatch, _step())

    assert json.loads(sent[0].content) == {"virtual_time": T0, "virtual_seconds_per_real_second": 1.0}


@pytest.mark.parametrize("setup, instance_id, why", [
    (lambda mp: _loaded(mp, None), "inst-1", "universe snap-u v1 recorded no virtual clock"),
    (lambda mp: _loaded(mp, _RECORDED, instance_store=_InstanceStore()), "inst-1", "no universe was loaded into it"),
    (lambda mp: _loaded(mp, _RECORDED, instance_store=_InstanceStore(error=RuntimeError("store down"))), "inst-1",
     "its loaded universe could not be read (RuntimeError: store down)"),
    (lambda mp: _loaded(mp, _RECORDED), None, "the env has no instance record"),
], ids=["no-clock", "nothing-loaded", "store-error", "no-instance"])
@pytest.mark.asyncio
async def test_execute_warns_naming_why_it_arms_at_the_fallback(local_stores, monkeypatch, caplog, setup, instance_id, why):
    setup(monkeypatch)
    step = SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time=T0, resume_from_loaded_universe=True)

    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        sent, _ = await _execute(monkeypatch, step, instance_id=instance_id)

    assert json.loads(sent[0].content)["virtual_time"] == T0
    assert any(r.levelno == logging.WARNING and f"sync_env_clock 'sync': {why}; arming at the fallback" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_execute_raises_before_any_request_with_no_recorded_clock_and_no_fallback(local_stores, monkeypatch):
    _loaded(monkeypatch, None)
    sent = _mock_http(monkeypatch)
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({}), instance_id="inst-1")])
    step = SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", resume_from_loaded_universe=True)
    with pytest.raises(RuntimeError, match="cannot resume env 'e1' from its loaded universe: universe snap-u v1 "
                                           "recorded no virtual clock, and no virtual_time fallback is set"):
        await step.execute(context)
    assert sent == []


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


def _carded(card: dict, instance_id: str | None = None) -> DeployedEnv:
    from agent_env.env.gateway.constants import WELL_KNOWN_PATH

    return DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None,
                       sandbox_id="sb-1", environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}", environment_card=card,
                       instance_id=instance_id)


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
