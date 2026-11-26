"""The ``stop_agent`` step: where it ends an agent's processes, what it records and when it fails. Its script is run
against real containers in ``tst/integration/task_step/test_stop_agent_local.py``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM
from agent_env.task_step.context import DeployedAgent, StepContextError, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import stop_agent
from agent_env.task_step.task_steps.stop_agent import StopAgentTaskStep

ENDED = "SERVER\t7\tuvicorn\nTERM\t20\tsh\nTERM\t21\tsleep\nKILL\t21\tsleep\n"


class _Sandbox:
    """A sandbox that answers ``docker ps`` with its containers and the script with what it's given, recording every
    command."""

    def __init__(self, mode=SANDBOX_MODE_VM, ended=(0, ENDED, "")):
        self.mode, self.container_name, self.owns_container = mode, "agent-api", True
        self.ended, self.calls = ended, []

    async def exec_with_output(self, *args):
        self.calls.append(args)
        if args[:3] == ("sudo", "docker", "ps"):
            return 0, "agent-api\nother\n", ""
        return self.ended


@pytest.fixture
def stopping(monkeypatch):
    """A sandbox every provider lookup gets."""
    state = SimpleNamespace(sandbox=_Sandbox())
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=state.sandbox))
    monkeypatch.setattr(stop_agent, "build_sandbox_provider", lambda name: provider)
    monkeypatch.setattr(stop_agent, "get_agent_sandbox_provider", lambda: provider)
    return state


def _context(**agent) -> TaskStepContext:
    agent = {"agent_name": "coder", "api_url": "http://coder", "sandbox_id": "sb-1", "sandbox_type": "modal_vm",
             **agent}
    return TaskStepContext(deployed_agents=[DeployedAgent(**agent)])


def _step(**fields) -> StopAgentTaskStep:
    return StopAgentTaskStep(id="stop", version=1, **{"agent_name": "coder", **fields})


def test_registered_and_round_trips():
    assert get_task_step_registry().get("stop_agent") is StopAgentTaskStep
    step = StopAgentTaskStep.from_dict(_step(grace_s=2).to_dict())
    assert (step.agent_name, step.grace_s, step.fail_task_on_error) == ("coder", 2, True)
    assert StopAgentTaskStep.from_dict({"id": "stop"}).agent_name == "default-agent"


@pytest.mark.parametrize("grace_s", [-1, 301, True, "5"])
def test_a_grace_a_run_cant_use_is_a_config_problem(grace_s):
    assert _step(grace_s=grace_s).config_problems() == [
        f"grace_s must be a number of seconds from 0 to 300, not {grace_s!r}"]
    assert _step(grace_s=0).config_problems() == []


def test_an_installed_agent_is_stopped_in_its_container_as_root_and_reported(stopping):
    context = _context(installed_container="repo-sb-1", a2a_port=8123)

    asyncio.run(_step(grace_s=3).execute(context))

    [ended] = stopping.sandbox.calls
    assert ended == ("sudo", "docker", "exec", "-u", "0", "-w", "/", "repo-sb-1", "/bin/sh", "-c",
                     stop_agent._END_PROCESSES, "stop_agent", "8123", "30", "0.1", "5", "0.2")
    report = context.metadata["stop_agent"]["stop"]
    assert report.pop("stopped_at")
    assert report == {
        "agent_name": "coder", "container": "repo-sb-1", "server": {"pid": 7, "name": "uvicorn"},
        "processes": [{"pid": 20, "name": "sh", "signals": ["TERM"]},
                      {"pid": 21, "name": "sleep", "signals": ["TERM", "KILL"]}],
        "processes_ended": 2, "processes_left": [],
    }


def test_a_deployed_agent_is_stopped_in_the_container_it_runs_in_on_its_port(stopping):
    asyncio.run(_step().execute(_context()))

    ps, ended = stopping.sandbox.calls
    assert ps[:3] == ("sudo", "docker", "ps") and ended[7] == "agent-api" and ended[-5:-3] == ("8000", "50")


def test_the_other_agents_in_its_container_are_named_so_their_servers_are_spared(stopping):
    context = _context(installed_container="repo", a2a_port=8000)
    context.deployed_agents += [
        DeployedAgent(agent_name="helper", api_url="http://helper", sandbox_id="sb-1", installed_container="repo",
                      a2a_port=8001),
        DeployedAgent(agent_name="elsewhere", api_url="http://e", sandbox_id="sb-1", installed_container="other",
                      a2a_port=8002),
        DeployedAgent(agent_name="judge", api_url="http://j", sandbox_id="sb-2", installed_container="repo",
                      a2a_port=8003),
    ]

    asyncio.run(_step().execute(context))

    [ended] = stopping.sandbox.calls
    assert ended[-6:] == ("8000", "50", "0.1", "5", "0.2", "8001")


def test_an_agent_whose_sandbox_is_its_container_is_stopped_in_the_sandbox(stopping):
    stopping.sandbox = _Sandbox(mode="container")

    asyncio.run(_step().execute(_context(sandbox_type="modal")))

    [ended] = stopping.sandbox.calls
    assert ended[:3] == ("/bin/sh", "-c", stop_agent._END_PROCESSES)


def test_a_process_that_outlives_sigkill_fails_the_step_once_the_report_is_written(stopping):
    stopping.sandbox = _Sandbox(ended=(0, ENDED + "LEFT\t30\tsleep\n", ""))
    context = _context(installed_container="repo")

    with pytest.raises(RuntimeError, match=r"1 process\(es\) still ran after SIGKILL: 30 \(sleep\)"):
        asyncio.run(_step().execute(context))

    assert context.metadata["stop_agent"]["stop"]["processes_left"] == [{"pid": 30, "name": "sleep"}]


def test_a_server_that_isnt_running_fails_the_step_saying_so(stopping):
    stopping.sandbox = _Sandbox(ended=(3, "", "nothing listens on port 8000, so the agent's server isn't running\n"))

    with pytest.raises(RuntimeError, match="exit 3.*nothing listens on port 8000"):
        asyncio.run(_step().execute(_context(installed_container="repo")))


@pytest.mark.parametrize(("agent", "error", "message"), [
    ({"on_host": True}, RuntimeError, "installed on its sandbox's host"),
    ({"sandbox_id": None}, RuntimeError, "runs outside agent-env's sandboxes"),
    ({"agent_name": "helper"}, StepContextError, "Agent 'coder' not found"),
], ids=["on-host", "no-sandbox", "not-deployed"])
def test_an_agent_it_cant_stop_is_refused_before_anything_runs(stopping, agent, error, message):
    with pytest.raises(error, match=message):
        asyncio.run(_step().execute(_context(**agent)))

    assert stopping.sandbox.calls == []
