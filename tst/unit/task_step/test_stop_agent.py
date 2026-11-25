"""The ``stop_agent`` step: where it ends an agent's processes, what it cancels, what it records and when it fails. Its
scripts are run against real containers in ``tst/integration/task_step/stop_agent_test.py``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM
from agent_env.task_step.context import DeployedAgent, PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import stop_agent
from agent_env.task_step.task_steps.stop_agent import StopAgentTaskStep

SERVER = "SERVER\t7\t/opt/agent/venv/bin/uvicorn a2a_server:app --port 8123 \n"
ENDED = (SERVER + "TERM\t20\tsh -c while :; do echo 0 > /logs/verifier/reward.txt; done \n"
         "TERM\t21\tsh -c trap '' TERM; sleep 1000 \nKILL\t21\tsh -c trap '' TERM; sleep 1000 \n")
CLEARED = (f"file\t1\t2\t1795600000\t{'a' * 64}\t\nlink\t1\t\t\t\t/etc/shadow \nmissing\t\t\t\t\t\n")


class _Sandbox:
    """A sandbox that answers ``docker ps`` with its containers and each script with what it's given, recording every
    command."""

    def __init__(self, mode=SANDBOX_MODE_VM, ended=(0, ENDED, ""), cleared=(0, CLEARED, "")):
        self.mode, self.container_name, self.owns_container = mode, "agent-api", True
        self.ended, self.cleared, self.calls = ended, cleared, []

    async def exec_with_output(self, *args):
        self.calls.append(args)
        if args[:3] == ("sudo", "docker", "ps"):
            return 0, "agent-api\nother\n", ""
        return self.ended if stop_agent._END_PROCESSES in args else self.cleared


@pytest.fixture
def stopping(monkeypatch):
    """A sandbox every provider lookup gets, and the A2A tasks cancelled, each answered as ``answers`` says."""
    state = SimpleNamespace(sandbox=_Sandbox(), cancelled=[], answers={})

    async def cancel(url, task_id):
        state.cancelled.append((url, task_id))
        return state.answers.get(task_id)

    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=state.sandbox))
    monkeypatch.setattr(stop_agent, "build_sandbox_provider", lambda name: provider)
    monkeypatch.setattr(stop_agent, "get_agent_sandbox_provider", lambda: provider)
    monkeypatch.setattr(stop_agent.protocol, "cancel_a2a_task", cancel)
    return state


def _context(**agent) -> TaskStepContext:
    agent = {"agent_name": "coder", "api_url": "http://coder", "sandbox_id": "sb-1", "sandbox_type": "modal_vm",
             **agent}
    return TaskStepContext(deployed_agents=[DeployedAgent(**agent)], prompt_responses=[
        PromptResponse(prompt_id="ask", response="done", agent_name="coder", agent_session_id="task-1"),
        PromptResponse(prompt_id="ask", response="done", agent_name="coder", agent_session_id="task-2"),
        PromptResponse(prompt_id="judge", response="ok", agent_name="judge", agent_session_id="task-3"),
        PromptResponse(prompt_id="again", response="done", agent_name="coder", agent_session_id="task-2"),
    ])


def _step(**fields) -> StopAgentTaskStep:
    return StopAgentTaskStep(id="stop", version=1, **{"agent_name": "coder", **fields})


def test_registered_and_round_trips():
    assert get_task_step_registry().get("stop_agent") is StopAgentTaskStep
    step = StopAgentTaskStep.from_dict(_step(clear_paths=["/logs/verifier/reward.txt"], grace_s=2).to_dict())
    assert (step.agent_name, step.clear_paths, step.grace_s, step.fail_task_on_error) == (
        "coder", ["/logs/verifier/reward.txt"], 2, True)
    assert StopAgentTaskStep.from_dict({"id": "stop"}).agent_name == "default-agent"


@pytest.mark.parametrize(("fields", "problem"), [
    ({"clear_paths": ["logs/reward.txt"]}, "clear_paths entry 'logs/reward.txt' isn't an absolute path"),
    ({"clear_paths": ["/logs/.."]}, "clear_paths entry '/logs/..' would remove the container's whole filesystem"),
    ({"grace_s": -1}, "grace_s must be a number of seconds from 0 to 300, not -1"),
    ({"grace_s": True}, "grace_s must be a number of seconds from 0 to 300, not True"),
])
def test_a_path_or_grace_a_run_cant_use_is_a_config_problem(fields, problem):
    assert _step(**fields).config_problems() == [problem]
    assert _step(clear_paths=["/logs/verifier/reward.txt"], grace_s=0).config_problems() == []


def test_an_installed_agent_is_stopped_in_its_container_as_root_and_reported(stopping):
    context = _context(installed_container="repo-sb-1", a2a_port=8123)

    asyncio.run(_step(clear_paths=["/logs/verifier/reward.txt", "/logs/verifier/link.txt", "/tmp/none"],
                      grace_s=3).execute(context))

    ended, cleared = stopping.sandbox.calls
    in_container = ("sudo", "docker", "exec", "-u", "0", "-w", "/", "repo-sb-1", "sh", "-c")
    assert ended == (*in_container, stop_agent._END_PROCESSES, "stop_agent", "8123", "30")
    assert cleared == (*in_container, stop_agent._CLEAR_PATHS, "stop_agent", "/logs/verifier/reward.txt",
                       "/logs/verifier/link.txt", "/tmp/none")
    assert stopping.cancelled == [("http://coder", "task-1"), ("http://coder", "task-2")]
    report = context.metadata["stop_agent"]["stop"]
    assert report.pop("stopped_at")
    assert report == {
        "agent_name": "coder", "container": "repo-sb-1",
        "cancelled": [{"task_id": "task-1", "cancelled": True}, {"task_id": "task-2", "cancelled": True}],
        "server": {"pid": 7, "command": "/opt/agent/venv/bin/uvicorn a2a_server:app --port 8123"},
        "processes": [
            {"pid": 20, "command": "sh -c while :; do echo 0 > /logs/verifier/reward.txt; done", "signals": ["TERM"]},
            {"pid": 21, "command": "sh -c trap '' TERM; sleep 1000", "signals": ["TERM", "KILL"]},
        ],
        "processes_ended": 2, "processes_left": [],
        "cleared": [
            {"path": "/logs/verifier/reward.txt", "kind": "file", "removed": True, "size": 2, "sha256": "a" * 64,
             "mtime": "2026-11-25T09:46:40+00:00"},
            {"path": "/logs/verifier/link.txt", "kind": "link", "removed": True, "target": "/etc/shadow"},
            {"path": "/tmp/none", "kind": "missing", "removed": False},
        ],
    }


def test_a_deployed_agent_is_stopped_in_the_container_it_runs_in_on_its_port(stopping):
    context = _context()

    asyncio.run(_step().execute(context))

    ps, ended = stopping.sandbox.calls
    assert ps[:3] == ("sudo", "docker", "ps") and ended[7] == "agent-api" and ended[-2:] == ("8000", "50")
    assert "cleared" not in context.metadata["stop_agent"]["stop"]


def test_an_agent_whose_sandbox_is_its_container_is_stopped_in_the_sandbox(stopping):
    stopping.sandbox = _Sandbox(mode="container")

    asyncio.run(_step().execute(_context(sandbox_type="modal")))

    [ended] = stopping.sandbox.calls
    assert ended[:3] == ("sh", "-c", stop_agent._END_PROCESSES)


def test_a_turn_past_cancelling_is_recorded_with_the_agents_answer(stopping):
    stopping.answers["task-1"] = "Task cannot be canceled"
    context = _context(installed_container="repo")

    asyncio.run(_step().execute(context))

    assert context.metadata["stop_agent"]["stop"]["cancelled"] == [
        {"task_id": "task-1", "cancelled": False, "why_not": "Task cannot be canceled"},
        {"task_id": "task-2", "cancelled": True}]


def test_a_process_that_outlives_sigkill_fails_the_step_once_the_report_is_written(stopping):
    stopping.sandbox = _Sandbox(ended=(0, ENDED + "LEFT\t30\tsleep 1000 \n", ""))
    context = _context(installed_container="repo")

    with pytest.raises(RuntimeError, match=r"1 process\(es\) still ran after SIGKILL: 30 \(sleep 1000\)"):
        asyncio.run(_step(clear_paths=["/r"]).execute(context))

    assert context.metadata["stop_agent"]["stop"]["processes_left"] == [{"pid": 30, "command": "sleep 1000"}]
    assert context.metadata["stop_agent"]["stop"]["cleared"]


def test_a_path_left_in_place_fails_the_step(stopping):
    stopping.sandbox = _Sandbox(cleared=(0, "directory\t\t\t\t\t\n", ""))
    context = _context(installed_container="repo")

    with pytest.raises(RuntimeError, match=r"couldn't remove \['/logs'\]"):
        asyncio.run(_step(clear_paths=["/logs"]).execute(context))

    assert context.metadata["stop_agent"]["stop"]["cleared"] == [{"path": "/logs", "kind": "directory",
                                                                  "removed": False}]


def test_a_server_that_isnt_running_fails_the_step_saying_so(stopping):
    stopping.sandbox = _Sandbox(ended=(3, "", "nothing listens on port 8000, so the agent's server isn't running\n"))

    with pytest.raises(RuntimeError, match="exit 3.*nothing listens on port 8000"):
        asyncio.run(_step().execute(_context(installed_container="repo")))


@pytest.mark.parametrize(("agent", "error", "message"), [
    ({"on_host": True}, RuntimeError, "installed on its sandbox's host"),
    ({"sandbox_id": None}, RuntimeError, "runs outside agent-env's sandboxes"),
    ({"agent_name": "helper"}, StepContextError, "Agent 'coder' not found"),
], ids=["on-host", "no-sandbox", "not-deployed"])
def test_an_agent_it_cant_stop_is_refused_before_anything_is_cancelled(stopping, agent, error, message):
    with pytest.raises(error, match=message):
        asyncio.run(_step().execute(_context(**agent)))

    assert stopping.cancelled == [] and stopping.sandbox.calls == []
