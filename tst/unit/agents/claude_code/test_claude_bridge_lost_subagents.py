"""A turn whose background subagents the CLI stopped fails, though the CLI exits 0 with a successful `result`.

In print mode the CLI waits a while after the last turn for background subagents, then stops those still running,
says so on stderr and exits 0. The event shapes below are copied from a capture of the pinned CLI, with ids and text
replaced by fakes.
"""
import asyncio
import json

import pytest

from tst.unit.agents.claude_code.agent import claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI

_STOPPING = (b"Background tasks still running 10m after the last turn (1 task); stopping them. "
             b"Set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 to wait indefinitely.\n")


def _stdout(*events):
    lines = [{"type": "system", "subtype": "init", "session_id": "sess-1"}, *events]
    return "".join(json.dumps(e) + "\n" for e in lines).encode()


def _agent_call(tool_use_id, description):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": tool_use_id, "name": "Agent",
         "input": {"description": description, "subagent_type": "general-purpose", "prompt": "p"}}]}}


def _task_stop(task_id):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "toolu_stop", "name": "TaskStop", "input": {"task_id": task_id}}]}}


def _started(task_id, description, task_type="local_agent", backgrounded=True):
    return {"type": "system", "subtype": "task_started", "task_id": task_id, "tool_use_id": f"toolu_{task_id}",
            "description": description, "subagent_type": "general-purpose", "is_backgrounded": backgrounded,
            "task_type": task_type, "session_id": "sess-1"}


def _ended(task_id, status):
    patch_status = "killed" if status == "stopped" else status
    return [
        {"type": "system", "subtype": "task_updated", "task_id": task_id, "patch": {"status": patch_status},
         "session_id": "sess-1"},
        {"type": "system", "subtype": "task_notification", "task_id": task_id, "tool_use_id": f"toolu_{task_id}",
         "status": status, "summary": "s", "session_id": "sess-1"},
    ]


def _result(text):
    return {"type": "result", "subtype": "success", "is_error": False, "terminal_reason": "completed",
            "session_id": "sess-1", "result": text}


def _run(monkeypatch, stdout, stderr=b""):
    async def _exec(*cmd, **kwargs):
        return FakeCLI(stdout, 0, stderr)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    return asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))


def test_a_turn_whose_subagent_the_cli_stopped_fails(monkeypatch):
    stdout = _stdout(
        _agent_call("toolu_a1", "Summarize the invoices"), _started("a1", "Summarize the invoices"),
        _agent_call("toolu_a2", "Collect the receipts"), _started("a2", "Collect the receipts"),
        _result("Two background agents are still working; I'll report when they finish."),
        *_ended("a1", "completed"), _result("The invoices are summarized."),
        *_ended("a2", "stopped"),
    )

    result = _run(monkeypatch, stdout, _STOPPING)

    assert result.error == "claude CLI stopped background subagents before they finished: Collect the receipts"


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_subagent_that_ran_to_its_end_is_not_lost(monkeypatch, status):
    stdout = _stdout(_started("a1", "Summarize the invoices"), _result("Working on it."), *_ended("a1", status),
                     _result("Done."))

    result = _run(monkeypatch, stdout)

    assert result.error is None
    assert result.response == "Done."


def test_a_subagent_the_agent_stopped_itself_is_not_lost(monkeypatch):
    stdout = _stdout(_started("a1", "Summarize the invoices"), _task_stop("a1"), *_ended("a1", "stopped"),
                     _result("Stopped it; not needed."))

    assert _run(monkeypatch, stdout).error is None


@pytest.mark.parametrize("task_type", ["local_bash", "local_workflow", "remote_agent"])
def test_background_tasks_other_than_subagents_are_not_counted(monkeypatch, task_type):
    stdout = _stdout(_started("b1", "npm run dev", task_type=task_type), _result("Started the server."),
                     *_ended("b1", "stopped"))

    assert _run(monkeypatch, stdout, _STOPPING).error is None


def test_a_foreground_subagent_is_not_counted(monkeypatch):
    stdout = _stdout(_started("a1", "Summarize the invoices", backgrounded=False), _result("Done."))

    assert _run(monkeypatch, stdout).error is None
