"""Unit tests for SyncEnvClockTaskStep — round-trip + the per-server sync_time fan-out (mocked)."""
import pytest

from agent_env.env.gateway.constants import EXT_CLOCK_URI
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
    out = await _step()._sync_one("http://gw/svc/mcp-email", "email", CLOCK_URL)
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
    out = await _step()._sync_one("http://gw/svc/mcp-email", "email", CLOCK_URL)
    assert out["synced"] is True
    assert calls["uri"] == EXT_CLOCK_URI and calls["params"] == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_unadvertised_strict_raises(monkeypatch):
    async def fake_card(base_url, **kw):
        return {"capabilities": {"extensions": []}}
    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    with pytest.raises(RuntimeError):
        await _step(tolerate_missing_sync_time=False)._sync_one("http://gw/svc/mcp-email", "email", CLOCK_URL)
