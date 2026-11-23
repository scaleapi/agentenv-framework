from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.validator import CLOCK_PROBE_TIMES, CLOCK_PROBE_ZONE, A2AAgentValidator, clock_reading
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import DeployedAgent, PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep

CLOCK_CARD = {"capabilities": {"extensions": [{"uri": EXT_CLOCK_URI, "params": {"endpoint": "/ext/clock", "methods": {
    "sync_time": {"method": "POST", "endpoint": "/ext/clock/sync-time"}}}}]}}


@pytest.mark.parametrize(("reply", "time", "zone"), [
    pytest.param("2031-03-03T23:06:07-0600", True, True, id="env-time-in-the-tasks-zone"),
    pytest.param("2031-03-04T05:06:07+0000", True, False, id="env-time-without-the-zone"),
    pytest.param("2026-11-17T11:20:00-0600", False, True, id="real-time"),
    pytest.param("I can't run shell commands here.", False, False, id="no-reading"),
])
def test_a_reading_is_the_frozen_instant_in_the_tasks_zone(reply, time, zone):
    reading = clock_reading(f"Output:\n{reply}\n", CLOCK_PROBE_TIMES[0])
    assert (reading["expected"], reading["time"], reading["zone"]) == ("2031-03-03T23:06:07-0600", time, zone)


class _Env:
    env_id = A2AAgent.VALIDATION_ENV_ID

    def __init__(self, write_status: int = 200):
        self.write_status = write_status
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, uri, method, params=None, *, timeout=None):
        self.calls.append((method, params))
        if self.write_status >= 400:
            request = httpx.Request("PUT", "http://gateway/clock/advance")
            response = httpx.Response(self.write_status, json={"ok": False, "stale": ["agent-agent"]}, request=request)
            raise httpx.HTTPStatusError("not acknowledged", request=request, response=response)
        return {"ok": True}


@pytest.fixture
def persisted(monkeypatch) -> SimpleNamespace:
    agent = SimpleNamespace(metadata={})
    agent.update_metadata = lambda metadata: setattr(agent, "metadata", metadata)
    monkeypatch.setattr(A2AAgent, "get", classmethod(lambda cls, agent_id, version=None: agent))
    return agent


@pytest.fixture
def synced(monkeypatch) -> list[SyncEnvClockTaskStep]:
    armed: list[SyncEnvClockTaskStep] = []

    async def execute(self, context):
        armed.append(self)
        return context

    monkeypatch.setattr(SyncEnvClockTaskStep, "execute", execute)
    return armed


def _env_time(at: str) -> str:
    return clock_reading("", at)["expected"]


def _real_time(at: str) -> str:
    return "2026-11-17T11:20:00-0600"


def _shows(monkeypatch, reply_at) -> None:
    times = iter(CLOCK_PROBE_TIMES)

    async def execute(self, context):
        context.prompt_responses.append(PromptResponse(prompt_id=self.prompt_id, response=reply_at(next(times))))
        return context

    monkeypatch.setattr(PromptAgentTaskStep, "execute", execute)


async def _validate(card: dict, env: _Env) -> dict:
    agent = SimpleNamespace(id="agent-under-test", VALIDATION_ENV_ID=A2AAgent.VALIDATION_ENV_ID)
    context = TaskStepContext(deployed_envs=[env], deployed_agents=[DeployedAgent(
        agent_name=TaskStep.DEFAULT_AGENT_NAME, api_url="http://agent", a2a_card=card, env_ids=[env.env_id])])
    await A2AAgentValidator._validate_clock(agent, context, task_id="validate")
    return context.metadata["verifications"]["a2a_clock"]


@pytest.mark.asyncio
async def test_an_agent_whose_card_serves_no_sync_time_is_skipped_without_arming(persisted, synced):
    result = await _validate({"capabilities": {"extensions": []}}, _Env())
    assert (result["supported"], result["advertised"], synced) == (False, False, [])
    assert result["skipped_reason"] == "clock/v1 sync_time not advertised on live agent card"
    assert persisted.metadata["validated_a2a_extensions"][EXT_CLOCK_URI] == result


@pytest.mark.asyncio
async def test_an_agent_that_follows_shows_each_time_the_clock_is_at(monkeypatch, persisted, synced):
    _shows(monkeypatch, _env_time)
    env = _Env()
    result = await _validate(CLOCK_CARD, env)
    assert (result["supported"], result["synced"], result["follows_writes"]) == (True, True, True)
    assert [r["shown"] for r in result["readings"]] == ["2031-03-03T23:06:07-0600", "2031-03-04T23:06:07-0600"]
    assert [(s.virtual_time, s.virtual_seconds_per_real_second, s.timezone, s.require_agent_clock) for s in synced] == [
        (CLOCK_PROBE_TIMES[0], 0, CLOCK_PROBE_ZONE, True)]
    assert env.calls == [("advance", {"virtual_time": CLOCK_PROBE_TIMES[1], "wait_followers_s": 10.0})]
    assert persisted.metadata["validated_a2a_extensions"][EXT_CLOCK_URI]["supported"] is True


@pytest.mark.asyncio
async def test_an_agent_whose_commands_read_real_time_fails(monkeypatch, persisted, synced):
    _shows(monkeypatch, _real_time)
    result = await _validate(CLOCK_CARD, _Env())
    assert (result["supported"], result["synced"], result["follows_writes"]) == (False, True, True)
    assert [(r["time"], r["zone"]) for r in result["readings"]] == [(False, True), (False, True)]


@pytest.mark.asyncio
async def test_an_agent_that_doesnt_acknowledge_a_write_fails_naming_it(monkeypatch, persisted, synced):
    _shows(monkeypatch, _env_time)
    result = await _validate(CLOCK_CARD, _Env(write_status=504))
    assert (result["supported"], result["synced"], result["follows_writes"], len(result["readings"])) == (False, True, False, 1)
    assert result["error"].startswith("HTTP 504") and "agent-agent" in result["error"]


@pytest.mark.asyncio
async def test_an_agent_that_cant_follow_fails_with_why(monkeypatch, persisted):
    async def refuse(self, context):
        raise StepContextError("agent 'agent' can't follow the clock of env 'mail': no_follower_fields")

    monkeypatch.setattr(SyncEnvClockTaskStep, "execute", refuse)
    result = await _validate(CLOCK_CARD, _Env())
    assert (result["supported"], result["synced"], result["readings"]) == (False, False, [])
    assert result["error"] == "StepContextError: agent 'agent' can't follow the clock of env 'mail': no_follower_fields"
