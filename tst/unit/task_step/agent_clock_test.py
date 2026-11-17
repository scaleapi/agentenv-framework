"""Agents on the env's clock: once sync_env_clock has armed it, every agent registered with the env whose card serves
clock/v1 sync_time is declared at the gateway as a follower with a token of its own and handed the follower's config;
one that doesn't is reported and skipped, even when the env's servers must follow (tolerate_missing_sync_time false),
or fails the step under require_agent_clock; and an agent
deploy_agent deploys while the clock is armed follows the same way."""
import json
from types import SimpleNamespace

import pytest

from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import DeployedAgent, StepContextError, TaskStepContext
from agent_env.task_step.task_steps import agent_clock
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.unit.task_step.sync_env_clock_test import (CLOCK_URL, T0, _CARD, _CLOCK_EXT, _ZONE_EXT, _carded, _composed_card, _mock_http,
                                                    _step)

AGENT_URL = "http://solver:8000"
SYNC_TIME = {"method": "POST", "endpoint": "/ext/clock/sync-time",
             "request": {"required": ["env_get_time_url"], "optional": ["timezone", "watch_url", "name", "read_token"]}}


def _agent(name="solver", sync_time=SYNC_TIME, env_ids=("e1",)) -> DeployedAgent:
    extensions = [{"uri": EXT_CLOCK_URI, "params": {"endpoint": "/ext/clock", "methods": {"sync_time": sync_time}}}]
    return DeployedAgent(agent_name=name, api_url=AGENT_URL, a2a_url=AGENT_URL, env_ids=list(env_ids),
                         a2a_card={"capabilities": {"extensions": extensions if sync_time else []}})


async def _context(*agents) -> TaskStepContext:
    """An env whose one MCP server follows the clock, and ``agents``."""
    return TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))],
                           deployed_agents=list(agents))


def _no_servers(monkeypatch):
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a, **k: SimpleNamespace(environment_name="email"))


@pytest.mark.asyncio
async def test_an_agent_is_declared_a_follower_then_handed_its_config_with_the_same_token(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    _no_servers(monkeypatch)
    context = await _context(_agent())
    context.deployed_envs = [_carded(await _composed_card({"mcp-email": [_ZONE_EXT]}))]

    await SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time="2026-01-01T06:00:00",
                               timezone="America/Chicago").execute(context)

    declared, synced = sent[3], sent[4]
    assert (declared.method, str(declared.url)) == ("PUT", f"{_CARD}/clock/followers")
    assert (synced.method, str(synced.url)) == ("POST", f"{AGENT_URL}/ext/clock/sync-time")
    declaration, config = json.loads(declared.content), json.loads(synced.content)
    assert declaration == {"name": "agent-solver", "read_token": config["read_token"]} and len(config["read_token"]) > 30
    assert config == {"env_get_time_url": CLOCK_URL, "watch_url": "http://gw/clock/watch", "name": "agent-solver",
                      "read_token": declaration["read_token"], "timezone": "America/Chicago"}
    [agent] = context.metadata["clock_configurations"][0]["agents"]
    assert (agent["agent"], agent["synced"], agent["follower"]) == ("solver", True, "agent-solver")


@pytest.mark.asyncio
async def test_only_the_agents_registered_with_the_env_follow_its_clock(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    _no_servers(monkeypatch)

    await _step().execute(await _context(_agent("elsewhere", env_ids=("e2",))))

    assert [r.url.path for r in sent][3:] == []


@pytest.mark.parametrize("sync_time, reason", [
    (None, "not_advertised"),
    ({**SYNC_TIME, "request": {"required": ["env_get_time_url"], "optional": ["timezone"]}}, "no_follower_fields"),
])
@pytest.mark.asyncio
async def test_an_agent_that_cant_follow_is_reported_and_skipped_or_fails_a_strict_step(monkeypatch, sync_time, reason):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    _no_servers(monkeypatch)
    context = await _context(_agent(sync_time=sync_time))

    await _step().execute(context)
    assert context.metadata["clock_configurations"][0]["agents"] == [{"agent": "solver", "synced": False,
                                                                      "reason": reason}]
    assert [r.url.path for r in sent][3:] == []
    strict = await _context(_agent(sync_time=sync_time))
    await _step(tolerate_missing_sync_time=False).execute(strict)
    assert strict.metadata["clock_configurations"][0]["agents"][0]["synced"] is False
    with pytest.raises(StepContextError, match=f"agent 'solver' can't follow the clock of env 'e1': {reason}"):
        await _step(require_agent_clock=True).execute(await _context(_agent(sync_time=sync_time)))


@pytest.mark.asyncio
async def test_a_gateway_from_before_followers_after_start_is_reported_and_the_agent_skipped(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    _no_servers(monkeypatch)
    context = await _context(_agent())
    methods = next(e for e in context.deployed_envs[0].environment_card["capabilities"]["extensions"]
                   if e["uri"] == EXT_CLOCK_URI)["params"]["methods"]
    del methods["declare_follower"]

    await _step().execute(context)

    assert context.metadata["clock_configurations"][0]["agents"][0]["reason"] == "gateway_takes_no_followers"
    assert [r.url.path for r in sent][3:] == []


@pytest.mark.asyncio
async def test_an_agent_that_serves_sync_time_and_fails_fails_the_step_saying_why(monkeypatch):
    import httpx

    real = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ext/clock/sync-time":
            return httpx.Response(404, json={"detail": "target_unavailable: this agent's clock follower isn't following"})
        return httpx.Response(200, json={"env_get_time_url": CLOCK_URL} if request.url.path.endswith("/state") else {})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    _no_servers(monkeypatch)

    with pytest.raises(RuntimeError, match="agent 'solver' doesn't follow the clock of env 'e1' \\(HTTP 404\\): .*"
                                           "isn't following"):
        await _step().execute(await _context(_agent()))


@pytest.mark.asyncio
async def test_an_agent_deployed_while_the_clock_is_armed_follows_it(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    context = await _context()
    context.metadata["clock_configurations"] = [
        {"step_id": "sync", "env_id": "e1", "virtual_time": T0, "env_get_time_url": CLOCK_URL, "timezone": "Asia/Tokyo",
         "require_agent_clock": False},
        {"step_id": "rate", "env_id": "e1", "virtual_seconds_per_real_second": 60, "virtual_time": T0},
    ]
    late = _agent("late")

    outcomes = await agent_clock.follow_armed_clocks(context, late, timeout=5)

    assert [(r.method, str(r.url)) for r in sent] == [("PUT", f"{_CARD}/clock/followers"),
                                                      ("POST", f"{AGENT_URL}/ext/clock/sync-time")]
    assert json.loads(sent[1].content)["timezone"] == "Asia/Tokyo"
    assert [(o["env_id"], o["agent"], o["synced"]) for o in outcomes] == [("e1", "late", True)]


@pytest.mark.asyncio
async def test_an_agent_deployed_before_any_clock_is_armed_is_left_alone(monkeypatch):
    sent = _mock_http(monkeypatch)
    assert await agent_clock.follow_armed_clocks(await _context(), _agent(), timeout=5) == []
    assert sent == []


@pytest.mark.asyncio
async def test_a_late_agent_that_cant_follow_fails_a_deploy_only_under_require_agent_clock(monkeypatch):
    _mock_http(monkeypatch)
    context = await _context()
    context.metadata["clock_configurations"] = [{"env_id": "e1", "env_get_time_url": CLOCK_URL}]
    [skipped] = await agent_clock.follow_armed_clocks(context, _agent("late", sync_time=None), timeout=5)
    assert (skipped["synced"], skipped["reason"]) == (False, "not_advertised")
    context.metadata["clock_configurations"][0]["require_agent_clock"] = True
    with pytest.raises(StepContextError, match="agent 'late' can't follow the clock of env 'e1': not_advertised"):
        await agent_clock.follow_armed_clocks(context, _agent("late", sync_time=None), timeout=5)


def test_require_agent_clock_round_trips_and_is_left_out_when_off():
    assert "require_agent_clock" not in _step().to_dict()
    strict = _step(require_agent_clock=True).to_dict()
    assert strict["require_agent_clock"] is True
    assert SyncEnvClockTaskStep.from_dict(strict).require_agent_clock is True


@pytest.mark.asyncio
async def test_an_agent_follows_one_envs_clock_and_another_envs_is_reported_and_skipped(monkeypatch):
    sent = _mock_http(monkeypatch)
    context = await _context()
    context.metadata["clock_configurations"] = [{"env_id": "e1", "env_get_time_url": CLOCK_URL}]
    context.metadata[agent_clock.FOLLOWING_KEY] = {"late": {"agent": "late", "env_id": "e0", "follower": "agent-late"}}

    [outcome] = await agent_clock.follow_armed_clocks(context, _agent("late", env_ids=("e0", "e1")), timeout=5)

    assert (outcome["env_id"], outcome["synced"], outcome["reason"], outcome["follows"]) == (
        "e1", False, "follows_another_env", "e0")
    assert sent == []  # e0 isn't deployed here; e1's gateway is never asked to take the agent
    assert context.metadata[agent_clock.FOLLOWING_KEY] == {"late": {"agent": "late", "env_id": "e0",
                                                                    "follower": "agent-late"}}


@pytest.mark.asyncio
async def test_a_teardown_retires_the_agents_follower_at_each_gateway_it_followed(monkeypatch):
    from agent_env.task_step.task_steps.teardown_sandboxes import TeardownSandboxesTaskStep

    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    _no_servers(monkeypatch)
    context = await _context(_agent(), _agent("bystander"))
    await _step().execute(context)
    del sent[:]

    async def nothing(*a, **k):
        return SimpleNamespace(terminated=[], failed=[])

    monkeypatch.setattr("agent_env.task_step.task_steps.teardown_sandboxes.before_teardown", nothing)
    monkeypatch.setattr("agent_env.task_step.task_steps.teardown_sandboxes.tear_down", nothing)
    await TeardownSandboxesTaskStep(id="down", version=1, agent_names=["solver"]).execute(context)

    assert [(r.method, str(r.url), json.loads(r.content)) for r in sent] == [
        ("DELETE", f"{_CARD}/clock/followers", {"name": "agent-solver"})]
    assert context.metadata[agent_clock.FOLLOWING_KEY] == {
        "bystander": {"agent": "bystander", "env_id": "e1", "follower": "agent-bystander"}}
    assert agent_clock.followed_env(context, _agent()) is None


@pytest.mark.asyncio
async def test_a_gateway_that_cant_retire_a_follower_doesnt_fail_the_teardown(monkeypatch):
    _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    _no_servers(monkeypatch)
    context = await _context(_agent())
    await _step().execute(context)
    methods = next(e for e in context.deployed_envs[0].environment_card["capabilities"]["extensions"]
                   if e["uri"] == EXT_CLOCK_URI)["params"]["methods"]
    del methods["retire_follower"]

    assert await agent_clock.retire_followers(context, ["solver"]) == []
    assert agent_clock.followed_env(context, _agent()) == "e1"


def test_a_log_of_an_outcome_leaves_out_the_agents_answer():
    assert agent_clock.summary({"synced": True, "follower": "agent-solver", "result": {"big": "x" * 10_000}}) == {
        "synced": True, "follower": "agent-solver"}


@pytest.mark.asyncio
async def test_late_agents_deployed_at_once_each_save_only_their_own_record(monkeypatch):
    import copy

    from agent_env.task_step.context_ops import build_context_update_ops

    _mock_http(monkeypatch)
    pre = await _context()
    pre.metadata["clock_configurations"] = [{"env_id": "e1", "env_get_time_url": CLOCK_URL}]
    saved = {}
    for name in ("first", "second"):  # each from the same context, as two deploy_agent steps at once
        post = copy.deepcopy(pre)
        await agent_clock.follow_armed_clocks(post, _agent(name), timeout=5)
        ops = build_context_update_ops(pre, post)
        saved[name] = {*ops.sets, *ops.add_to_sets, *ops.unsets}

    for name, paths in saved.items():
        assert paths and all(p.startswith(f"context.metadata.{agent_clock.FOLLOWING_KEY}.{name}") for p in paths), paths


def test_an_agent_name_with_a_dot_keeps_a_key_of_its_own():
    assert agent_clock._slot("team.solver") == "team_solver" and agent_clock._slot("$x") == "_x"
