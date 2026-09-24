"""Unit tests for the typed-stakeholder helpers added to PromptAgentTaskStep:
_read_env_triggers (env-trigger snapshot; fail-closed; 404->{}) and _decide (the /decide call).
Only the HTTP boundary is mocked; the loop routing itself is covered by the dev e2e."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.env.gateway.constants import TRIGGER_IN_FLIGHT_STATUSES, TRIGGER_STATUSES
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _step(timeout: float = 30):
    # The two helpers only read self.user_agent_timeout_seconds; skip the heavy __init__.
    s = PromptAgentTaskStep.__new__(PromptAgentTaskStep)
    s.user_agent_timeout_seconds = timeout
    return s


def _env(env_id: str, gateway_url: str):
    return DeployedEnv(env_id=env_id, env_version=1, gateway_url=gateway_url, mcp_url="",
                       db_web_url=None, sandbox_id="sb")


def _ctx(*envs):
    return TaskStepContext(deployed_envs=list(envs), deployed_agents=[], metadata={})


def _resp(status: int, body: dict, method: str = "GET") -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request(method, "http://x"))


# ---- _read_env_triggers -----------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_shape_multi_env():
    async def fake_get(self, url, timeout):
        return _resp(200, {"triggers": [{"id": "v6-rate", "status": "fired"},
                                        {"id": "v6-accept", "status": "armed"}]})

    ctx = _ctx(_env("env-a", "http://gw-a"), _env("env-b", "http://gw-b"))
    with patch.object(httpx.AsyncClient, "get", fake_get):
        snap = await _step()._read_env_triggers(ctx)

    assert snap == {"env-a": {"v6-rate": "fired", "v6-accept": "armed"},
                    "env-b": {"v6-rate": "fired", "v6-accept": "armed"}}


@pytest.mark.asyncio
async def test_404_maps_to_empty():
    async def fake_get(self, url, timeout):
        return _resp(404, {})

    ctx = _ctx(_env("env-x", "http://gw-x"))
    with patch.object(httpx.AsyncClient, "get", fake_get):
        snap = await _step()._read_env_triggers(ctx)

    assert snap == {"env-x": {}}


@pytest.mark.asyncio
async def test_fail_closed_raises_on_persistent_error():
    async def fake_get(self, url, timeout):
        raise httpx.ConnectError("gateway down")

    ctx = _ctx(_env("env-x", "http://gw-x"))
    with patch.object(httpx.AsyncClient, "get", fake_get), \
         patch("agent_env.task_step.task_steps.prompt_agent.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(RuntimeError, match="fail-closed: could not read /triggers/state for env 'env-x'"):
            await _step()._read_env_triggers(ctx)


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient")
        return _resp(200, {"triggers": [{"id": "t", "status": "fired"}]})

    ctx = _ctx(_env("env-x", "http://gw-x"))
    with patch.object(httpx.AsyncClient, "get", fake_get), \
         patch("agent_env.task_step.task_steps.prompt_agent.asyncio.sleep", new=AsyncMock()):
        snap = await _step()._read_env_triggers(ctx)

    assert snap == {"env-x": {"t": "fired"}} and calls["n"] == 2


# ---- _decide ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_decide_request_body_and_parse():
    captured = {}

    async def fake_post(self, url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _resp(200, {"parts": [{"kind": "text", "text": "[Julia]: re-sweep"}], "done": False, "fired": ["re-sweep"]}, "POST")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        out = await _step()._decide("http://sk/ext/triggers/decide", 3, "solver said X",
                                    "conv-1", {"env-a": {"v6-rate": "fired"}})

    assert captured["url"] == "http://sk/ext/triggers/decide"
    assert captured["json"] == {"turn": 3, "solver_message": "solver said X",
                                "context_id": "conv-1", "env_triggers": {"env-a": {"v6-rate": "fired"}}}
    assert out == {"parts": [{"kind": "text", "text": "[Julia]: re-sweep"}], "done": False, "fired": ["re-sweep"]}


@pytest.mark.asyncio
async def test_decide_4xx_raises():
    async def fake_post(self, url, json, timeout):
        return _resp(400, {"error": "bad turn"}, "POST")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(RuntimeError, match="agent-trigger /decide failed"):
            await _step()._decide("http://sk/ext/triggers/decide", 1, "", "c", {})


# ---- _persist_env_trigger_state ---------------------------------------------

_STATE = {
    "config": {"watch_roles": ["default"], "executor_configured": True},
    "triggers": [{"id": "v6-rate", "status": "fired", "detected_at": "t0", "fired_at": "t1"}],
    "events": [
        {"seq": 1, "ts": "t", "kind": "added", "trigger_id": "v6-rate"},
        {"seq": 2, "ts": "t", "kind": "detected", "trigger_id": "v6-rate",
         "provoking": {"tool": "gdocs_create_document", "role": "default"}},
        {"seq": 3, "ts": "t", "kind": "fired", "trigger_id": "v6-rate"},
    ],
}

_CLOCK = {"armed": True, "t0": "2026-06-01T00:00:00Z",
          "virtual_seconds_per_real_second": 86400, "virtual_time": "2026-06-02T21:36:00Z"}

# Observability fields; the recurrence sits at "armed" between arrivals.
_CLOCK_STATE = {
    "config": {"watch_roles": ["default"], "executor_configured": False},
    "triggers": [
        {"id": "beat", "type": "time", "when": {"type": "time", "every": "PT12H"}, "status": "armed",
         "detected_at": "t0", "fired_at": "t1", "fire_count": 3, "next_mark": "2026-06-03T00:00:00Z"},
        {"id": "v6-grant", "type": "action", "when": {"type": "action", "tool": "gdocs_create_document"},
         "status": "fired", "detected_at": "t0", "fired_at": "t1", "fire_count": 1, "next_mark": None},
    ],
    "events": [{"seq": 1, "ts": "t", "virtual_time": "2026-06-01T12:00:00Z", "kind": "fired",
                "trigger_id": "beat", "fire_count": 1, "mark": "2026-06-01T12:00:00Z"}],
    "events_dropped": 0,
}


class _FakeStore:
    def __init__(self):
        self.puts = []

    def put(self, key, data, content_type="application/octet-stream", allow_overwrite=False):
        self.puts.append({"key": key, "data": data, "content_type": content_type,
                          "allow_overwrite": allow_overwrite})
        return f"s3://test-bucket/{key}"


def _capture_step():
    s = _step()
    s.id = "prompt-1"
    return s


def _patched_store(store=None):
    from types import SimpleNamespace
    store = store if store is not None else _FakeStore()
    cfg = SimpleNamespace(get_object_store=lambda: store)
    return store, patch("agent_env.task_step.task_steps.prompt_agent.get_config", return_value=cfg)


@pytest.mark.asyncio
async def test_capture_uploads_state_and_records_summary():
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        if url.endswith("/clock/state"):
            return _resp(200, _CLOCK)
        calls["n"] += 1
        return _resp(200, _STATE)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a"), _env("env-b", "http://gw-b")],
        deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["v6-rate"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch:
        await _capture_step()._persist_env_trigger_state(ctx)

    assert len(store.puts) == 1
    put = store.puts[0]
    assert put["key"].startswith("env_trigger_state/instance_id=inst-1/env-a-")
    assert put["key"].endswith(".json")
    assert put["allow_overwrite"] is False and put["content_type"] == "application/json"
    import json as _json
    envelope = _json.loads(put["data"])
    assert envelope["instance_id"] == "inst-1" and envelope["env_id"] == "env-a"
    assert envelope["capture_source"] == "prompt_agent" and envelope["state"] == _STATE

    entry = ctx.metadata["env_trigger_state"]["env-a"]
    assert entry["object_url"] == f"s3://test-bucket/{put['key']}"
    assert entry["statuses"] == {"v6-rate": "fired"} and entry["event_count"] == 3
    assert entry["capture_step_id"] == "prompt-1" and "captured_at_utc" in entry
    assert "env-b" not in ctx.metadata["env_trigger_state"]
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_capture_records_clock_context_and_per_trigger_detail():
    """A recurrence reads "armed" forever, so only fire_count shows it fired."""
    async def fake_get(self, url, timeout):
        return _resp(200, _CLOCK if url.endswith("/clock/state") else _CLOCK_STATE)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["beat"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch:
        await _capture_step()._persist_env_trigger_state(ctx)

    import json as _json
    envelope = _json.loads(store.puts[0]["data"])
    assert envelope["clock"] == _CLOCK
    assert envelope["capture_is_final"] is False  # `beat` is still armed and will keep firing

    entry = ctx.metadata["env_trigger_state"]["env-a"]
    assert entry["triggers"]["beat"] == {"type": "time", "status": "armed", "fire_count": 3,
                                         "next_mark": "2026-06-03T00:00:00Z",
                                         "failure_count": None, "last_failure_at": None}
    assert entry["triggers"]["v6-grant"]["fire_count"] == 1
    assert entry["statuses"] == {"beat": "armed", "v6-grant": "fired"}  # back-compat preserved
    assert entry["capture_is_final"] is False


@pytest.mark.asyncio
async def test_capture_distinguishes_a_failed_coupling_from_one_that_never_fired():
    """A failed action re-arms, so `armed` + fire_count alone cannot show the failure."""
    failed = {**_CLOCK_STATE, "triggers": [
        {"id": "never", "type": "action", "when": {"type": "action", "tool": "gdocs_create_document"},
         "status": "armed", "fire_count": 0, "next_mark": None,
         "failure_count": 0, "last_failure_at": None},
        {"id": "tried-and-failed", "type": "action",
         "when": {"type": "action", "tool": "gsheets_values_update"},
         "status": "armed", "fire_count": 0, "next_mark": None,
         "failure_count": 2, "last_failure_at": "2026-06-01T12:00:00Z"},
    ]}

    async def fake_get(self, url, timeout):
        return _resp(200, _CLOCK if url.endswith("/clock/state") else failed)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["never", "tried-and-failed"],
             "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch:
        await _capture_step()._persist_env_trigger_state(ctx)

    rows = ctx.metadata["env_trigger_state"]["env-a"]["triggers"]
    # Identical on status and fire_count; only the failure fields tell them apart.
    assert rows["never"]["status"] == rows["tried-and-failed"]["status"] == "armed"
    assert rows["never"]["fire_count"] == rows["tried-and-failed"]["fire_count"] == 0
    assert rows["never"]["failure_count"] == 0
    assert rows["tried-and-failed"]["failure_count"] == 2
    assert rows["tried-and-failed"]["last_failure_at"] == "2026-06-01T12:00:00Z"


@pytest.mark.asyncio
async def test_capture_does_not_wait_on_a_firing_recurring_clock_trigger():
    """An unbounded recurrence re-enters `firing` every arrival; waiting on it burns the budget."""
    firing = {**_CLOCK_STATE, "triggers": [
        {"id": "beat", "type": "time", "when": {"type": "time", "every": "PT12H"},
         "status": "firing", "fire_count": 3, "next_mark": "2026-06-03T00:00:00Z"}]}
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        if url.endswith("/clock/state"):
            return _resp(200, _CLOCK)
        calls["n"] += 1
        return _resp(200, firing)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["beat"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch, \
         patch("agent_env.task_step.task_steps.prompt_agent.asyncio.sleep", new=AsyncMock()):
        await _capture_step()._persist_env_trigger_state(ctx)

    assert calls["n"] == 1, "a recurring clock trigger must not trigger the settle poll"
    assert store.puts, "the capture must still upload"


@pytest.mark.asyncio
async def test_capture_tolerates_a_gateway_without_clock_state():
    """No clock/v1: capture still succeeds, clock recorded as None."""
    async def fake_get(self, url, timeout):
        if url.endswith("/clock/state"):
            raise httpx.ConnectError("no clock route")
        return _resp(200, _STATE)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["v6-rate"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch:
        await _capture_step()._persist_env_trigger_state(ctx)

    import json as _json
    envelope = _json.loads(store.puts[0]["data"])
    assert envelope["clock"] is None
    entry = ctx.metadata["env_trigger_state"]["env-a"]
    assert "object_url" in entry and "error" not in entry
    # old-shape rows degrade to None fields rather than raising
    assert entry["triggers"]["v6-rate"] == {"type": None, "status": "fired",
                                            "fire_count": None, "next_mark": None,
                                            "failure_count": None, "last_failure_at": None}
    # ...and finality is UNKNOWN, never a false "complete" claim (no `type` to reason from)
    assert entry["capture_is_final"] is None


# ---- back-compat: a new worker routinely talks to an env on an older gateway image ------------

_OLD_LIVE_RECURRING = {  # older row shape, without the observability fields; recurrence still going
    "config": {"watch_roles": ["default"], "executor_configured": False},
    "triggers": [{"id": "beat", "status": "armed", "detected_at": "t0", "fired_at": "t1",
                  "fire_count": 7, "next_mark": "2026-06-03T00:00:00Z"}],
    "events": [],
}


def test_capture_is_final_is_unknown_not_true_on_an_old_gateway():
    """Without `type`, True would assert completeness while the driver is still firing."""
    assert PromptAgentTaskStep._capture_is_final(_OLD_LIVE_RECURRING) is None


def test_capture_is_final_flags_a_non_time_trigger_still_firing():
    """The settle poll can time out with an nl action mid-flight."""
    state = {"triggers": [{"id": "v6", "type": "action", "when": {"type": "action", "tool": "t"},
                           "status": "firing"}]}
    assert PromptAgentTaskStep._capture_is_final(state) is False


def test_settle_poll_skips_a_live_recurrence_on_an_old_gateway():
    """Without `when`, next_mark is the only tell that a firing trigger is a live recurrence."""
    firing = {"triggers": [{"id": "beat", "status": "firing", "next_mark": "2026-06-03T00:00:00Z"}]}
    assert PromptAgentTaskStep._is_settling(firing) is False
    one_shot = {"triggers": [{"id": "corr", "status": "firing", "next_mark": None}]}
    assert PromptAgentTaskStep._is_settling(one_shot) is True


def test_settle_poll_still_waits_for_a_bounded_recurrence():
    """A bounded recurrence terminates, so its final action is worth waiting for."""
    bounded = {"triggers": [{"id": "beat", "type": "time", "status": "firing",
                             "when": {"type": "time", "every": "PT1H", "count": 3}}]}
    assert PromptAgentTaskStep._is_settling(bounded) is True
    unbounded = {"triggers": [{"id": "beat", "type": "time", "status": "firing",
                               "when": {"type": "time", "every": "PT1H"}}]}
    assert PromptAgentTaskStep._is_settling(unbounded) is False


@pytest.mark.asyncio
async def test_capture_noop_without_registrations():
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        calls["n"] += 1
        return _resp(200, _STATE)

    ctx = _ctx(_env("env-a", "http://gw-a"))
    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch:
        await _capture_step()._persist_env_trigger_state(ctx)

    assert calls["n"] == 0 and store.puts == [] and "env_trigger_state" not in ctx.metadata


@pytest.mark.asyncio
async def test_capture_fail_open_records_error():
    async def fake_get(self, url, timeout):
        raise httpx.ConnectError("gateway gone")

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["v6-rate"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch, \
         patch("agent_env.task_step.task_steps.prompt_agent.asyncio.sleep", new=AsyncMock()):
        await _capture_step()._persist_env_trigger_state(ctx)

    entry = ctx.metadata["env_trigger_state"]["env-a"]
    assert "error" in entry and "object_url" not in entry
    assert store.puts == []


@pytest.mark.asyncio
async def test_capture_grace_polls_until_settled():
    firing = {**_STATE, "triggers": [{"id": "v6-rate", "status": "firing"}]}
    responses = [firing, _STATE]
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        if url.endswith("/clock/state"):
            return _resp(200, _CLOCK)
        body = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return _resp(200, body)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["v6-rate"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch, \
         patch("agent_env.task_step.task_steps.prompt_agent.asyncio.sleep", new=AsyncMock()):
        await _capture_step()._persist_env_trigger_state(ctx)

    assert calls["n"] == 2
    assert ctx.metadata["env_trigger_state"]["env-a"]["statuses"] == {"v6-rate": "fired"}


@pytest.mark.asyncio
async def test_capture_budget_bounds_total_time():
    async def hanging_get(self, url, timeout):
        await asyncio.sleep(999)

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a"), _env("env-b", "http://gw-b")],
        deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["v6-rate"], "executor_agent_name": None},
            {"step_id": "reg-2", "env_id": "env-b", "added": ["t2"], "executor_agent_name": None}]})

    store, cfg_patch = _patched_store()
    with patch.object(httpx.AsyncClient, "get", hanging_get), cfg_patch, \
         patch.object(PromptAgentTaskStep, "_CAPTURE_BUDGET_SECONDS", 0.05):
        await _capture_step()._persist_env_trigger_state(ctx)

    assert store.puts == []
    for env_id in ("env-a", "env-b"):
        entry = ctx.metadata["env_trigger_state"][env_id]
        assert "budget" in entry["error"] and entry["capture_step_id"] == "prompt-1"


@pytest.mark.asyncio
async def test_capture_put_failure_keeps_inline_summary():
    async def fake_get(self, url, timeout):
        return _resp(200, _STATE)

    class _FailingStore(_FakeStore):
        def put(self, *args, **kwargs):
            raise RuntimeError("s3 down")

    ctx = TaskStepContext(
        deployed_envs=[_env("env-a", "http://gw-a")], deployed_agents=[], instance_id="inst-1",
        metadata={"env_trigger_registrations": [
            {"step_id": "reg-1", "env_id": "env-a", "added": ["v6-rate"], "executor_agent_name": None}]})

    _, cfg_patch = _patched_store(_FailingStore())
    with patch.object(httpx.AsyncClient, "get", fake_get), cfg_patch:
        await _capture_step()._persist_env_trigger_state(ctx)

    entry = ctx.metadata["env_trigger_state"]["env-a"]
    assert entry["statuses"] == {"v6-rate": "fired"} and entry["event_count"] == 3
    assert "error" in entry and "object_url" not in entry


# The other half of the partition. It lives here rather than in the engine or the step because it is
# a ledger for the test below, not something either of them needs at runtime.
_KNOWN_SETTLED_STATUSES = ("armed", "fired", "failed")


def test_the_capture_path_classifies_every_status_the_engine_can_emit():
    """A status the engine can report must have a decided meaning for the capture path.

    `queued` shipped unclassified: the engine grew a status, the capture path enumerated only the
    ones it knew, and a status it had never heard of defaulted to "settled" -- the unsafe direction.
    Driving this off the engine's own vocabulary means the next addition fails here until someone
    decides what it means."""
    classified = set(TRIGGER_IN_FLIGHT_STATUSES) | set(_KNOWN_SETTLED_STATUSES)
    assert classified == set(TRIGGER_STATUSES), (
        "the engine's trigger status vocabulary changed. Decide whether the new status means the "
        "capture path must keep waiting, then add it to TRIGGER_IN_FLIGHT_STATUSES in "
        "gateway/constants.py or to _KNOWN_SETTLED_STATUSES here"
    )
    assert not set(TRIGGER_IN_FLIGHT_STATUSES) & set(_KNOWN_SETTLED_STATUSES)


def test_queued_work_is_unsettled_even_under_a_status_the_list_never_learned():
    """The clause that does not depend on knowing the vocabulary: `pending` is the queue itself.

    A status outside `TRIGGER_IN_FLIGHT_STATUSES` is what the `queued` regression looked like from
    the consumer's side. Keying on the queue length too means outstanding work still reads as
    outstanding while nobody has taught the list about the new name."""
    state = {"triggers": [{"id": "t", "type": "action", "when": {"type": "action", "tool": "x"},
                           "status": "a_status_from_the_future", "pending": 2}]}
    assert PromptAgentTaskStep._is_settling(state) is True
    assert PromptAgentTaskStep._capture_is_final(state) is False
