"""Behavior tests for RegisterEnvTriggersStep: executor-name resolution, the POST body/audit, 4xx propagation, and
the optional await of trigger statuses. Only the gateway HTTP boundary is mocked; the step logic runs real."""

from __future__ import annotations

import copy
import json
import time

import httpx
import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep

_GATEWAY = "http://gw:18765"


def _context(*, with_executor: bool = False):
    env = _record()
    agents = []
    if with_executor:
        agents.append(DeployedAgent(agent_name="exec-1", api_url="http://exec:9000",
                                    a2a_url="http://exec:9000/a2a-base", role=with_executor
                                    if isinstance(with_executor, str) else "executor"))
    return TaskStepContext(deployed_envs=[env], deployed_agents=agents, metadata={})


_TRIGGERS = [{"id": "t1", "when": {"type": "action", "tool": "gdocs_create_document"},
              "actions": [{"type": "permission", "action": "enable", "role": "default",
                           "tools": ["snowflake_submit_query"]}]}]


def _step(**overrides):
    base = {"id": "reg-1", "version": None, "env_id": "env-x", "triggers": _TRIGGERS}
    base.update(overrides)
    return RegisterEnvTriggersStep(**base)


def test_round_trips_through_to_dict_from_dict():
    step = _step(watch_roles=["default"], executor_agent_name="exec-1", executor_timeout_seconds=90,
                 executor_hours={"exec-1": _HOURS})
    rebuilt = RegisterEnvTriggersStep.from_dict(step.to_dict())
    assert rebuilt.env_id == "env-x"
    assert rebuilt.triggers == _TRIGGERS
    assert rebuilt.watch_roles == ["default"]
    assert rebuilt.executor_agent_name == "exec-1"
    assert rebuilt.executor_timeout_seconds == 90
    assert rebuilt.executor_hours == {"exec-1": _HOURS}
    assert (rebuilt.await_fired, rebuilt.await_timeout_seconds) == (None, 120)

    awaiting = RegisterEnvTriggersStep.from_dict(_step(await_fired=["t1", "t1"], await_timeout_seconds=7).to_dict())
    assert (awaiting.await_fired, awaiting.await_timeout_seconds) == (["t1"], 7.0)
    stored_before_await = {k: v for k, v in _step().to_dict().items() if k not in ("await_fired", "await_timeout_seconds")}
    assert RegisterEnvTriggersStep.from_dict(stored_before_await).await_fired is None


def test_bad_triggers_rejected_at_construction():
    with pytest.raises(ValueError, match="triggers must be a list"):
        _step(triggers="nope")


@pytest.mark.asyncio
async def test_registers_and_audits(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    out = await _step(watch_roles=["default"]).execute(_context())

    [request] = sent
    assert (request.method, str(request.url)) == ("POST", f"{_GATEWAY}/triggers/register")
    assert json.loads(request.content) == {"triggers": _TRIGGERS, "watch_roles": ["default"]}
    audit = out.metadata["env_trigger_registrations"]
    assert len(audit) == 1
    assert audit[0]["env_id"] == "env-x"
    assert audit[0]["added"] == ["t1"]
    assert audit[0]["executor_agent_names"] == []


@pytest.mark.asyncio
async def test_resolves_executor_agent_to_a2a_url(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    await _step(executor_agent_name="exec-1", executor_timeout_seconds=45,
                watch_roles=["default"]).execute(_context(with_executor=True))

    assert json.loads(sent[0].content)["executor"] == {
        "a2a_url": "http://exec:9000/a2a-base", "timeout_seconds": 45, "role": "executor"}


@pytest.mark.asyncio
@pytest.mark.parametrize("in_our_sandbox, expected", [
    (True, "http://host.docker.internal:41234"), (False, "http://127.0.0.1:41234"),
], ids=["local-gateway", "env-outside-our-sandboxes"])
async def test_a_local_gateway_reaches_a_local_executor_through_the_host(monkeypatch, in_our_sandbox, expected):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    env = _record()
    if in_our_sandbox:
        env.sandbox_type = "local"
    else:
        env = DeployedEnv(env_id=env.env_id, env_version=1, mcp_url="", environment_card_url=env.environment_card_url,
                          environment_card=env.environment_card)
    executor = DeployedAgent(agent_name="exec-1", api_url="http://127.0.0.1:41234", a2a_url="http://127.0.0.1:41234",
                             sandbox_id="local-a", sandbox_type="local", role="executor")
    context = TaskStepContext(deployed_envs=[env], deployed_agents=[executor], metadata={})

    await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(context)

    assert json.loads(sent[0].content)["executor"]["a2a_url"] == expected


@pytest.mark.asyncio
async def test_missing_executor_agent_raises():
    ctx = _context()  # no agents deployed
    with pytest.raises(RuntimeError, match="Executor agent 'exec-1' not found"):
        await _step(executor_agent_name="exec-1").execute(ctx)


@pytest.mark.asyncio
async def test_executor_without_role_raises():
    ctx = TaskStepContext(
        deployed_envs=[_record()],
        deployed_agents=[DeployedAgent(agent_name="exec-1", api_url="http://e", role=None)],
        metadata={})
    with pytest.raises(RuntimeError, match="deployed without an explicit role"):
        await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(ctx)


@pytest.mark.asyncio
async def test_executor_role_in_watch_roles_raises():
    ctx = _context(with_executor="default")  # executor deployed under the watched role
    with pytest.raises(RuntimeError, match="is in watch_roles"):
        await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(ctx)


@pytest.mark.asyncio
async def test_missing_env_raises():
    ctx = _context()
    with pytest.raises(RuntimeError, match="Env 'other-env' not found"):
        await _step(env_id="other-env").execute(ctx)


@pytest.mark.asyncio
async def test_gateway_4xx_raises(monkeypatch):
    _mock_gateway(monkeypatch, {"ok": False, "error": "trigger 't1': when.tool: bad"}, status=400)
    with pytest.raises(RuntimeError, match=r"trigger registration failed \(HTTP 400\): .*when.tool: bad"):
        await _step().execute(_context())


@pytest.mark.asyncio
async def test_a_card_without_triggers_raises_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    ctx = TaskStepContext(deployed_envs=[_record(extensions=[])], deployed_agents=[], metadata={})
    with pytest.raises(EnvCapabilityUnsupported, match="does not offer 'register' on urn:agentenv:triggers/v1"):
        await _step().execute(ctx)
    assert sent == []


@pytest.mark.asyncio
async def test_resolves_each_named_executor_alongside_the_default(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["a", "b", "d"], "all": ["a", "b", "d"]})
    context = TaskStepContext(deployed_envs=[_record()], deployed_agents=[
        DeployedAgent(agent_name="exec-1", api_url="http://exec:9000", role="executor"),
        DeployedAgent(agent_name="alice", api_url="http://alice:9000", a2a_url="http://alice:9000/a2a-base",
                      role="alice@example.com"),
        DeployedAgent(agent_name="bob", api_url="http://bob:9000", role="bob@example.com"),
    ], metadata={})

    out = await _step(triggers=_NAMED_TRIGGERS, watch_roles=["default"], executor_agent_name="exec-1",
                      executor_timeout_seconds=30).execute(context)

    body = json.loads(sent[0].content)
    assert body["executor"] == {"a2a_url": "http://exec:9000", "timeout_seconds": 30, "role": "executor"}
    assert body["executors"] == {
        "alice": {"a2a_url": "http://alice:9000/a2a-base", "timeout_seconds": 30, "role": "alice@example.com"},
        "bob": {"a2a_url": "http://bob:9000", "timeout_seconds": 30, "role": "bob@example.com"}}
    assert out.metadata["env_trigger_registrations"][0]["executor_agent_names"] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_a_missing_named_executor_raises_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    with pytest.raises(RuntimeError, match="Executor agent 'alice' not found"):
        await _step(triggers=_NAMED_TRIGGERS).execute(_context())
    assert sent == []


@pytest.mark.asyncio
async def test_a_named_executor_in_watch_roles_raises(monkeypatch):
    _mock_gateway(monkeypatch, {})
    context = TaskStepContext(deployed_envs=[_record()], deployed_agents=[
        DeployedAgent(agent_name="alice", api_url="http://alice", role="alice@example.com"),
        DeployedAgent(agent_name="bob", api_url="http://bob", role="default"),
    ], metadata={})
    with pytest.raises(RuntimeError, match="'default' is in watch_roles"):
        await _step(triggers=_NAMED_TRIGGERS, watch_roles=["default"]).execute(context)


@pytest.mark.asyncio
async def test_a_gateway_without_named_executors_refuses_them_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    context = TaskStepContext(deployed_envs=[_record(_with_features())], deployed_agents=[
        DeployedAgent(agent_name="alice", api_url="http://alice", role="alice@example.com"),
        DeployedAgent(agent_name="bob", api_url="http://bob", role="bob@example.com"),
    ], metadata={})
    with pytest.raises(RuntimeError, match=r"does not declare \['named_executors'\]"):
        await _step(triggers=_NAMED_TRIGGERS).execute(context)
    assert sent == []


@pytest.mark.asyncio
async def test_a_gateway_without_named_executors_still_takes_the_default(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    context = _context(with_executor=True)
    context.deployed_envs = [_record(_with_features())]
    await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(context)
    assert "executors" not in json.loads(sent[0].content)


@pytest.mark.asyncio
async def test_a_gateway_without_provoking_roles_refuses_a_trigger_that_lists_roles(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    triggers = [{"id": "hears-denise", "when": {"type": "action", "tool": "slack_send_message",
                                                "roles": ["denise@example.com"]}, "actions": []}]
    context = TaskStepContext(deployed_envs=[_record(_with_features("named_executors"))], deployed_agents=[],
                              metadata={})
    with pytest.raises(RuntimeError, match=r"does not declare \['provoking_roles'\]"):
        await _step(triggers=triggers).execute(context)
    assert sent == []


_HOURS = {"start": "09:00", "end": "17:00", "utc_offset": "-07:00"}


@pytest.mark.asyncio
async def test_executor_hours_ride_on_the_executors_they_name(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["a", "b", "a2", "d"], "all": ["a", "b", "a2", "d"]})
    context = TaskStepContext(deployed_envs=[_record()], deployed_agents=[
        DeployedAgent(agent_name="exec-1", api_url="http://exec:9000", role="executor"),
        DeployedAgent(agent_name="alice", api_url="http://alice:9000", role="alice@example.com"),
        DeployedAgent(agent_name="bob", api_url="http://bob:9000", role="bob@example.com"),
    ], metadata={})

    await _step(triggers=_NAMED_TRIGGERS, executor_agent_name="exec-1", executor_hours={"bob": _HOURS}).execute(context)

    body = json.loads(sent[0].content)
    assert "hours" not in body["executor"] and "hours" not in body["executors"]["alice"]
    assert body["executors"]["bob"]["hours"] == _HOURS


@pytest.mark.asyncio
async def test_executor_hours_for_an_agent_no_trigger_uses_raise(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    with pytest.raises(RuntimeError, match=r"executor_hours names \['carol'\]"):
        await _step(executor_hours={"carol": _HOURS}).execute(_context())
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("triggers, hours, missing", [
    ([{"id": "x", "when": {"type": "action", "tool": "t"},
       "actions": [{"type": "permission", "action": "enable", "role": "r", "tools": ["t"], "delay": "PT5M"}]}],
     None, "action_delay"),
    ([{"id": "x", "when": {"type": "time", "after": "y", "offset": "PT5M", "unless": {"tool": "t"}}, "actions": []}],
     None, "unless"),
    ([{"id": "x", "when": {"type": "action", "tool": "t"}, "actions": [{"type": "nl", "instruction": "x"}]}],
     {"exec-1": _HOURS}, "executor_hours"),
], ids=["delay", "unless", "hours"])
async def test_a_gateway_without_a_feature_refuses_the_triggers_that_use_it(monkeypatch, triggers, hours, missing):
    sent = _mock_gateway(monkeypatch, {})
    context = _context(with_executor=True)
    context.deployed_envs = [_record(_with_features("named_executors", "provoking_roles"))]
    with pytest.raises(RuntimeError, match=rf"does not declare \['{missing}'\]"):
        await _step(triggers=triggers, executor_agent_name="exec-1", executor_hours=hours).execute(context)
    assert sent == []


_NAMED_TRIGGERS = [
    {"id": tid, "when": {"type": "action", "tool": "slack_send_message"},
     "actions": [{"type": "nl", "instruction": f"reply as {tid}", **named}]}
    for tid, named in (("a", {"executor": "alice"}), ("b", {"executor": "bob"}), ("a2", {"executor": "alice"}),
                       ("d", {}))
]


def _with_features(*features: str) -> list:
    """The gateway's own extensions as a gateway declaring only ``features`` advertises them (none: one that predates
    the list)."""
    extensions = copy.deepcopy(GATEWAY_EXTENSIONS)
    params = next(e for e in extensions if e["uri"] == EXT_TRIGGERS_URI)["params"]
    if features:
        params["features"] = list(features)
    else:
        del params["features"]
    return extensions


def _record(extensions: list = GATEWAY_EXTENSIONS) -> DeployedEnv:
    """A gateway deployment whose stored card advertises `extensions` (the gateway's own, by default)."""
    return DeployedGatewayEnv(env_id="env-x", env_version=1, gateway_url=_GATEWAY, mcp_url="", db_web_url=None, sandbox_id="sb",
                       environment_card_url=f"{_GATEWAY}{WELL_KNOWN_PATH}",
                       environment_card={"name": "gw", "capabilities": {"extensions": extensions}})


def _mock_gateway(monkeypatch, body: dict, status: int = 200) -> list[httpx.Request]:
    """Answer every request the protocol client sends with `body`; return the requests."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


def test_bad_await_config_rejected_at_construction():
    with pytest.raises(ValueError, match="await_fired must be a list of trigger ids"):
        _step(await_fired="t1")
    with pytest.raises(ValueError, match="await_timeout_seconds must be a positive number"):
        _step(await_fired=["t1"], await_timeout_seconds=0)
    with pytest.raises(ValueError, match="await_timeout_seconds must be a positive number"):
        _step(await_fired=["t1"], await_timeout_seconds=True)


def test_preflight_names_an_awaited_trigger_this_step_does_not_register():
    assert _step(await_fired=["t1"]).preflight() == []
    [problem] = _step(await_fired=["t1", "nope"]).preflight()
    assert "await_fired names ['nope']" in problem and "reg-1" in problem


@pytest.mark.asyncio
async def test_an_unregistered_awaited_trigger_raises_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    with pytest.raises(RuntimeError, match=r"register_env_triggers 'reg-1': await_fired names \['nope'\]"):
        await _step(await_fired=["nope"]).execute(_context())
    assert sent == []


@pytest.mark.asyncio
async def test_a_card_without_triggers_state_refuses_the_await_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    ctx = TaskStepContext(deployed_envs=[_record(extensions=_extensions_without_state())], deployed_agents=[], metadata={})
    with pytest.raises(RuntimeError, match="does not offer 'state', which await_fired needs"):
        await _step(await_fired=["t1"]).execute(ctx)
    assert sent == []


@pytest.mark.asyncio
async def test_await_returns_once_each_named_trigger_reads_fired(monkeypatch):
    _no_sleep(monkeypatch)
    sent = _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1", "t2"], "all": ["t1", "t2"]}],
        "/triggers/state": [_state({"t1": "fired", "t2": "armed"}), _state({"t1": "fired", "t2": "firing"}),
                            _state({"t1": "fired", "t2": "fired"})],
    })
    out = await _step(triggers=_TWO, await_fired=["t1", "t2"]).execute(_context())

    assert [str(r.url.path) for r in sent] == ["/triggers/register"] + ["/triggers/state"] * 3
    [audit] = out.metadata["env_trigger_registrations"]
    assert audit["added"] == ["t1", "t2"]
    assert audit["awaited"] == {"t1": "fired", "t2": "fired"}


@pytest.mark.asyncio
async def test_no_await_means_no_state_read_and_no_awaited_key(monkeypatch):
    sent = _mock_gateway_by_path(monkeypatch, {"/triggers/register": [{"ok": True, "added": ["t1"], "all": ["t1"]}]})
    out = await _step().execute(_context())
    assert [str(r.url.path) for r in sent] == ["/triggers/register"]
    assert "awaited" not in out.metadata["env_trigger_registrations"][0]


@pytest.mark.asyncio
async def test_await_times_out_naming_statuses_and_failure_events(monkeypatch):
    _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1", "t2"], "all": ["t1", "t2"]}],
        "/triggers/state": [_state({"t1": "fired", "t2": "armed"},
                                   events=[{"seq": 3, "ts": "x", "kind": "action_failed", "trigger_id": "t2",
                                            "error": "tool 'slack_send_message' refused"},
                                           {"seq": 4, "ts": "y", "kind": "fired", "trigger_id": "t1"}])],
    })
    started = time.monotonic()
    with pytest.raises(RuntimeError) as err:
        await _step(triggers=_TWO, await_fired=["t1", "t2"], await_timeout_seconds=1.2).execute(_context())
    assert time.monotonic() - started < 1.2
    message = str(err.value)
    assert "did not all fire within 1.2s" in message
    assert "{'t1': 'fired', 't2': 'armed'}" in message
    assert "action_failed" in message and "slack_send_message" in message
    assert "'seq'" not in message and "'kind': 'fired'" not in message


@pytest.mark.asyncio
async def test_a_failed_wait_still_records_the_registration_with_the_last_statuses(monkeypatch):
    _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1", "t2"], "all": ["t1", "t2"]}],
        "/triggers/state": [_state({"t1": "fired", "t2": "armed"})],
    })
    ctx = _context()
    with pytest.raises(RuntimeError):
        await _step(triggers=_TWO, await_fired=["t1", "t2"], await_timeout_seconds=1.2).execute(ctx)
    [audit] = ctx.metadata["env_trigger_registrations"]
    assert audit["added"] == ["t1", "t2"]
    assert audit["awaited"] == {"t1": "fired", "t2": "armed"}


@pytest.mark.asyncio
async def test_slow_state_reads_cannot_outlive_the_await_budget(monkeypatch):
    """Each read's timeout is cut to the time left and no read starts inside the last minimum window, so a stalled
    gateway cannot hold the step past the deadline."""
    import asyncio as real_asyncio

    _mock_gateway_by_path(monkeypatch, {"/triggers/register": [{"ok": True, "added": ["t1"], "all": ["t1"]}]})
    timeouts: list[float] = []

    async def stalled(self, uri, method, params=None, *, environment_name=None, timeout=30):
        if method == "register":
            return {"ok": True, "added": ["t1"], "all": ["t1"]}
        timeouts.append(timeout)
        await real_asyncio.sleep(min(timeout, 0.05))
        raise httpx.ReadTimeout("gateway stalled")

    monkeypatch.setattr(DeployedGatewayEnv, "invoke", stalled)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"could not read /triggers/state within the 2.5s await budget: gateway stalled; statuses \{'t1': None\}"):
        await _step(await_fired=["t1"], await_timeout_seconds=2.5).execute(_context())
    assert time.monotonic() - started < 2.5
    assert timeouts and all(t <= 2.5 for t in timeouts) and timeouts == sorted(timeouts, reverse=True), timeouts


@pytest.mark.asyncio
async def test_await_stops_early_on_a_failed_trigger(monkeypatch):
    _no_sleep(monkeypatch)
    sent = _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1", "t2"], "all": ["t1", "t2"]}],
        "/triggers/state": [_state({"t1": "armed", "t2": "failed"},
                                   events=[{"seq": 1, "ts": "x", "kind": "failed", "trigger_id": "t2", "error": "boom"}])],
    })
    with pytest.raises(RuntimeError, match=r"reached a terminal status before firing: \['t2'\].*boom"):
        await _step(triggers=_TWO, await_fired=["t1", "t2"], await_timeout_seconds=60).execute(_context())
    assert [str(r.url.path) for r in sent] == ["/triggers/register", "/triggers/state"]


@pytest.mark.asyncio
async def test_a_trigger_missing_from_state_keeps_waiting_until_the_deadline(monkeypatch):
    sent = _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1"], "all": ["t1"]}],
        "/triggers/state": [_state({})],
    })
    with pytest.raises(RuntimeError, match=r"within 1.2s; statuses \{'t1': None\}"):
        await _step(await_fired=["t1"], await_timeout_seconds=1.2).execute(_context())
    assert [str(r.url.path) for r in sent].count("/triggers/state") == 1  # one read, one poll interval, the deadline


@pytest.mark.asyncio
async def test_a_budget_shorter_than_one_read_fails_without_reading(monkeypatch):
    sent = _mock_gateway_by_path(monkeypatch, {"/triggers/register": [{"ok": True, "added": ["t1"], "all": ["t1"]}]})
    ctx = _context()
    with pytest.raises(RuntimeError, match=r"within 0.5s; statuses \{'t1': None\}"):
        await _step(await_fired=["t1"], await_timeout_seconds=0.5).execute(ctx)
    assert [str(r.url.path) for r in sent] == ["/triggers/register"]
    assert ctx.metadata["env_trigger_registrations"][0]["awaited"] == {"t1": None}


@pytest.mark.asyncio
async def test_a_failed_state_read_is_retried_on_the_next_poll(monkeypatch):
    _no_sleep(monkeypatch)
    sent = _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1"], "all": ["t1"]}],
        "/triggers/state": [(503, {"error": "busy"}), _state({"t1": "fired"})],
    })
    out = await _step(await_fired=["t1"]).execute(_context())
    assert out.metadata["env_trigger_registrations"][0]["awaited"] == {"t1": "fired"}
    assert [str(r.url.path) for r in sent].count("/triggers/state") == 2


@pytest.mark.asyncio
async def test_a_state_read_that_keeps_failing_is_reported_with_its_cause_at_the_deadline(monkeypatch):
    sent = _mock_gateway_by_path(monkeypatch, {
        "/triggers/register": [{"ok": True, "added": ["t1"], "all": ["t1"]}],
        "/triggers/state": [(503, {"error": "busy"})],
    })
    with pytest.raises(RuntimeError, match=r"(?s)could not read /triggers/state within the 2.5s await budget: .*503.*; statuses \{'t1': None\}"):
        await _step(await_fired=["t1"], await_timeout_seconds=2.5).execute(_context())
    assert [str(r.url.path) for r in sent].count("/triggers/state") == 2  # reads at 0 s and 1 s; 2 s leaves no full window


_TWO = [{"id": "t1", "when": {"type": "time", "at": "2026-07-01T17:00:02-05:00"},
         "actions": [{"type": "tool", "tool": "slack_send_message", "args": {"channel_id": "C1", "message": "hi"}}]},
        {"id": "t2", "when": {"type": "time", "at": "2026-07-01T17:00:04-05:00"},
         "actions": [{"type": "tool", "tool": "slack_send_message", "args": {"channel_id": "C1", "message": "hi"}}]}]


def _state(statuses: dict[str, str], events: list | None = None) -> dict:
    return {"config": {}, "triggers": [{"id": tid, "type": "time", "status": status, "fire_count": int(status == "fired")}
                                       for tid, status in statuses.items()],
            "events": events or [], "events_dropped": 0}


def _extensions_without_state() -> list:
    extensions = copy.deepcopy(GATEWAY_EXTENSIONS)
    del next(e for e in extensions if e["uri"] == EXT_TRIGGERS_URI)["params"]["methods"]["state"]
    return extensions


def _no_sleep(monkeypatch) -> None:
    """Polls and retries run back to back; the deadline still comes from the monotonic clock."""
    async def instant(_seconds):
        return None
    monkeypatch.setattr("agent_env.task_step.task_steps.register_env_triggers.asyncio.sleep", instant)


def _mock_gateway_by_path(monkeypatch, scripted: dict[str, list]) -> list[httpx.Request]:
    """Answer each path from its scripted list in order, repeating the last entry; an entry may be `(status, body)`."""
    sent, real, served = [], httpx.AsyncClient, {path: 0 for path in scripted}

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        answers = scripted[request.url.path]
        entry = answers[min(served[request.url.path], len(answers) - 1)]
        served[request.url.path] += 1
        status, body = entry if isinstance(entry, tuple) else (200, entry)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
