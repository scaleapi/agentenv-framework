"""A turn's reply holds everything the model told the user across the CLI's wakes.

The CLI ends each wake with a `result` and wakes the model again when a background task it started finishes, so one
turn can emit several: an answer then a remark on the finished task, or a holding note then the answer. The event shapes
are trimmed from a capture of Claude Code 2.1.322 run against a fake model.
"""
import asyncio
import json

import pytest

from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.agent import claude_bridge

ANSWER = "INVOICE: INV-1042 $2,740.00; GET_TIME: 2027-02-01"
REMARK = 'Task 2 (the 15-second sleep) has now completed successfully with output "waited".'
HOLDING = "Waiting on the build; I'll report as soon as it finishes."


def _line(**event) -> str:
    return json.dumps(event) + "\n"


def _result(text, *, woken=False, is_error=False, **extra) -> str:
    origin = {"origin": {"kind": "task-notification", "producer": "session-task", "runId": "r1"}} if woken else {}
    return _line(type="result", subtype="success", is_error=is_error, result=text, session_id="sess-1",
                 terminal_reason="completed", **origin, **extra)


INIT = _line(type="system", subtype="init", session_id="sess-1")
BACKGROUNDED = (
    _line(type="assistant", message={"content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                  "input": {"command": "sleep 15", "run_in_background": True}}]})
    + _line(type="system", subtype="task_started", task_id="b1", tool_use_id="t1", is_backgrounded=True,
            task_type="local_bash")
)
FINISHED = (
    _line(type="system", subtype="task_updated", task_id="b1", patch={"status": "completed"})
    + _line(type="system", subtype="task_notification", task_id="b1", tool_use_id="t1", status="completed")
    + _line(type="system", subtype="background_tasks_changed", tasks=[])
)


@pytest.fixture
def turn(monkeypatch, tmp_path):
    """Run one turn against scripted `claude -p` invocations of (stdout, returncode)."""
    queue: list[tuple[str, int]] = []

    async def _fake_exec(*cmd, **kwargs):
        stdout, returncode = queue.pop(0)
        return FakeCLI(stdout.encode(), returncode)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.delenv("CLAUDE_API_ERROR_RETRIES", raising=False)

    def run(*invocations: tuple[str, int], **kwargs):
        queue.extend(invocations)
        return asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, **kwargs))

    run.session_dir = tmp_path
    return run


def test_an_answer_then_a_remark_on_the_finished_task_keeps_the_answer(turn):
    result = turn((INIT + BACKGROUNDED + _result(ANSWER) + FINISHED + _result(REMARK, woken=True), 0))
    assert result.error is None
    assert result.response == f"{ANSWER}\n\n{REMARK}"


def test_a_holding_note_then_the_answer_keeps_the_answer(turn):
    result = turn((INIT + BACKGROUNDED + _result(HOLDING) + FINISHED + _result(ANSWER, woken=True), 0))
    assert result.response == f"{HOLDING}\n\n{ANSWER}"


def test_a_turn_with_one_result_replies_with_it_as_it_was(turn):
    text = "  The answer, with its own spacing.\n"
    assert turn((INIT + _result(text), 0)).response == text


def test_an_error_result_stays_out_of_a_reply_that_has_an_answer(turn):
    refused = "API Error: the safeguards flagged this message."
    result = turn((INIT + _result(refused, is_error=True) + BACKGROUNDED + FINISHED + _result(ANSWER, woken=True), 0))
    assert result.response == ANSWER


def test_a_text_repeating_the_one_before_is_said_once(turn):
    answer = '{"findings_count": 2}'
    result = turn((INIT + BACKGROUNDED + _result(answer) + FINISHED + _result(answer, woken=True), 0))
    assert result.response == answer


def test_a_blank_result_adds_nothing(turn):
    result = turn((INIT + BACKGROUNDED + _result(ANSWER) + FINISHED + _result(" \n", woken=True), 0))
    assert result.response == ANSWER


def test_a_failed_turn_with_no_answer_replies_with_its_error(turn):
    result = turn((INIT + _result("API Error: 401 authentication_error", is_error=True), 1))
    assert result.error is not None
    assert result.response == "API Error: 401 authentication_error"


def test_a_requested_schema_answers_with_its_structured_output(turn):
    first, last = {"verdict": "pending"}, {"verdict": "pass"}
    result = turn((INIT + BACKGROUNDED + _result("", structured_output=first) + FINISHED
                   + _result("", woken=True, structured_output=last), 0))
    assert json.loads(result.response) == last


def test_a_resumed_turn_keeps_the_answer_given_before_the_wake_that_failed(turn):
    (turn.session_dir / "sess-1.jsonl").write_text('{"type":"assistant"}\n')
    cut_short = INIT + BACKGROUNDED + _result(ANSWER) + FINISHED + _result(
        "API Error: The operation timed out.", is_error=True)
    result = turn((cut_short, 1), (INIT + _result("Continued after the interruption.", woken=True), 0))
    assert result.attempts == 2
    assert result.error is None
    assert result.response == f"{ANSWER}\n\nContinued after the interruption."
    assert not any(e.get("is_error") for e in result.trajectory)
