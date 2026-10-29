"""A transient upstream failure resumes the CLI session instead of losing the turn.

When the model call stalls, the CLI exits 1 with a terminal `result` event carrying is_error and its own message
("API Error: The operation timed out."). The session transcript survives on disk, so the turn is resumable. These pin
which failures resume, which don't, and that resumes never outlive the caller's timeout budget.

"""
import asyncio
import json

import pytest

from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.agent import claude_bridge


def _event(**kw):
    return json.dumps(kw) + "\n"


def _ok_stdout(text="done", session_id="sess-1"):
    return (_event(type="result", subtype="success", result=text, session_id=session_id)).encode()


def _api_error_stdout(message, session_id="sess-1", tool_uses=0, **verdict):
    lines = _event(type="system", subtype="init", session_id=session_id)
    for _ in range(tool_uses):
        lines += _event(
            type="assistant",
            message={"content": [{"type": "tool_use", "name": "Bash", "input": {}}]},
        )
    lines += _event(
        type="result", subtype="success", is_error=True, result=message,
        session_id=session_id, **verdict
    )
    return lines.encode()


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """Drive run_claude with a queue of (stdout, returncode) invocations."""
    calls: list[list[str]] = []
    procs: list[FakeCLI] = []
    queue: list[tuple[bytes, int]] = []

    async def _fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        stdout, rc = queue.pop(0)
        procs.append(FakeCLI(stdout, rc))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    # The resume gate requires a real session transcript on disk.
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.delenv("CLAUDE_API_ERROR_RETRIES", raising=False)
    return {"calls": calls, "procs": procs, "queue": queue, "session_dir": tmp_path}


def _write_session(scripted, session_id="sess-1"):
    (scripted["session_dir"] / f"{session_id}.jsonl").write_text('{"type":"assistant"}\n')


def test_success_makes_exactly_one_invocation(scripted):
    scripted["queue"].append((_ok_stdout(), 0))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 1
    assert result.error is None
    assert result.response == "done"
    assert len(scripted["calls"]) == 1


def test_stalled_turn_resumes_the_session_and_recovers(scripted):
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out.", tool_uses=3), 1))
    scripted["queue"].append((_ok_stdout("finished after resume"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 2
    assert result.error is None
    assert result.response == "finished after resume"
    # First invocation opens the session, second resumes it.
    assert "--session-id" in scripted["calls"][0]
    assert scripted["calls"][1][scripted["calls"][1].index("--resume") + 1] == "sess-1"
    assert "--session-id" not in scripted["calls"][1]
    # The resume carries a continuation nudge, never the original prompt again.
    assert scripted["procs"][0].stdin.data == b"p"
    assert scripted["procs"][1].stdin.data == claude_bridge._RESUME_PROMPT.encode()
    # Tool calls accumulate across attempts.
    assert result.tool_call_count == 3


def test_abandoned_attempt_leaves_its_work_but_not_its_error(scripted):
    """Downstream readers take the FIRST result event as the turn's answer."""
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out.", tool_uses=3), 1))
    scripted["queue"].append((_ok_stdout("finished after resume"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    results = [e for e in result.trajectory if e.get("type") == "result"]
    assert len(results) == 1
    assert results[0]["result"] == "finished after resume"
    assert not any(e.get("is_error") for e in result.trajectory)
    # The abandoned attempt's real work is still there: init + 3 tool_use turns.
    assert len([e for e in result.trajectory if e.get("type") == "assistant"]) == 3
    assert len(result.trajectory) == 5


def test_non_resumable_api_error_is_not_retried(scripted):
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: 401 authentication_error"), 1))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 1
    assert result.error is not None
    assert len(scripted["calls"]) == 1


@pytest.mark.parametrize(
    "message",
    [
        # What the CLI says once a sandbox's egress is cut. Each emits a terminal result event, so only the
        # message allowlist keeps them from being retried.
        "API Error: Unable to connect to API (FailedToOpenSocket)",
        "API Error: Unable to connect to API (ConnectionRefused)",
    ],
)
def test_unreachable_endpoint_is_never_retried(scripted, message):
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout(message), 1))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 1
    assert len(scripted["calls"]) == 1


def test_cli_death_without_a_result_event_is_not_retried(scripted):
    _write_session(scripted)
    scripted["queue"].append((b"", 1))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 1
    assert len(scripted["calls"]) == 1


def test_missing_session_transcript_blocks_the_resume(scripted):
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 1
    assert result.error is not None


def test_retries_are_bounded(scripted, monkeypatch):
    _write_session(scripted)
    monkeypatch.setenv("CLAUDE_API_ERROR_RETRIES", "1")
    for _ in range(3):
        scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 2
    assert result.error is not None


def test_retries_can_be_disabled(scripted, monkeypatch):
    _write_session(scripted)
    monkeypatch.setenv("CLAUDE_API_ERROR_RETRIES", "0")
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 1


def test_exhausted_time_budget_blocks_the_resume(scripted):
    """A retry must never outlive the deadline the caller sized."""
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))
    result = asyncio.run(
        claude_bridge.run_claude(
            prompt="p", session_id="sess-1", resume=False,
            timeout_s=claude_bridge._MIN_RETRY_BUDGET_S - 1,
        )
    )
    assert result.attempts == 1
    assert result.error is not None


def test_structured_output_survives_a_resume(scripted):
    _write_session(scripted)
    schema = {"type": "json_schema", "schema": {"type": "object"}}
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))
    scripted["queue"].append((
        (_event(type="result", subtype="success", result="{}", session_id="sess-1",
                structured_output={"verdict": "pass"})).encode(), 0))

    result = asyncio.run(
        claude_bridge.run_claude(
            prompt="p", session_id="sess-1", resume=False, output_format=schema
        )
    )

    assert result.structured_output == {"verdict": "pass"}
    assert result.response == json.dumps({"verdict": "pass"})
    # The schema is re-applied on the resumed invocation, not dropped.
    assert "--json-schema" in scripted["calls"][1]


@pytest.mark.parametrize(
    "message",
    [
        # Real CLI messages; the first is the stall this exists for.
        "API Error: The operation timed out.",
        "API Error: The socket connection was closed unexpectedly. For more "
        "information, pass `verbose: true` in the second argument to fetch()",
        'API Error: 529 {"type":"error","error":{"type":"overloaded_error"}}',
        'API Error: 500 {"type":"error","error":{"type":"api_error","message":'
        '"Internal server error"}}',
    ],
)
def test_resumable_markers(scripted, message):
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout(message), 1))
    scripted["queue"].append((_ok_stdout("recovered"), 0))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))
    assert result.attempts == 2, f"expected {message!r} to be resumable"
    assert result.error is None


# Captured from claude-code 2.1.177 driven against a scripted upstream: every
# field below is what the CLI actually emitted, not a guess at its schema.
_CAPTURED = {
    "stalled_stream": {
        "message": "API Error: API returned an empty or malformed response (HTTP 200) "
                   "\u2014 check for a proxy or gateway intercepting the request",
        "verdict": {"terminal_reason": "completed", "api_error_status": None},
        "resumable": True,
    },
    "overloaded": {
        "message": "API Error: Repeated 529 Overloaded errors. The API is at capacity",
        "verdict": {"terminal_reason": "completed", "api_error_status": None},
        "resumable": True,
    },
    # 2.1.232 moved upstream failures out of "completed" into a new "api_error";
    # the Dockerfile pins the CLI and bumping it is routine, so both must resume.
    "stalled_stream_2_1_232": {
        "message": "API Error: API returned an empty or malformed response (HTTP 200)",
        "verdict": {"terminal_reason": "api_error", "api_error_status": None},
        "resumable": True,
    },
    "overloaded_2_1_232": {
        "message": "API Error: Repeated 529 Overloaded errors. The API is at capacity",
        "verdict": {"terminal_reason": "api_error", "api_error_status": 529},
        "resumable": True,
    },
    "prompt_too_long": {
        "message": "Prompt is too long \u00b7 the request is ~300000 tokens (limit 200000)",
        "verdict": {"terminal_reason": "prompt_too_long", "api_error_status": 400},
        "resumable": False,
    },
}


@pytest.mark.parametrize("name", sorted(_CAPTURED))
def test_captured_cli_verdicts(scripted, name):
    """The CLI's own `terminal_reason` decides, so a reworded message cannot
    silently turn a resumable failure into an unrecoverable one."""
    case = _CAPTURED[name]
    _write_session(scripted)
    scripted["queue"].append(
        (_api_error_stdout(case["message"], **case["verdict"]), 1)
    )
    scripted["queue"].append((_ok_stdout("recovered"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == (2 if case["resumable"] else 1)


@pytest.mark.parametrize(
    "api_error_status, expected_attempts",
    [
        # `terminal_reason: completed` is not sufficient on its own: a 4xx is the
        # request's own fault and reproduces byte-for-byte on a resume.
        (400, 1),
        (503, 2),
    ],
)
def test_api_error_status_gates_a_completed_turn(scripted, api_error_status, expected_attempts):
    _write_session(scripted)
    scripted["queue"].append(
        (_api_error_stdout(f"API Error: {api_error_status}",
                           terminal_reason="completed",
                           api_error_status=api_error_status), 1)
    )
    scripted["queue"].append((_ok_stdout("recovered"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == expected_attempts


def test_message_markers_still_apply_without_a_terminal_reason(scripted):
    """Builds that do not report `terminal_reason` keep the pre-existing
    behaviour exactly: the marker list is a fallback, not a replacement."""
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))
    scripted["queue"].append((_ok_stdout("recovered"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 2


def test_unknown_message_without_a_terminal_reason_is_not_resumed(scripted):
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: something brand new"), 1))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 1


@pytest.mark.parametrize(
    "api_error_status",
    # The CLI types this as a bare JS number today; the veto must not depend on
    # that staying true, and must beat a marker the prose happens to contain.
    [400, "400", "429", "400 Bad Request", 400.0],
)
def test_a_client_error_vetoes_even_when_the_prose_matches_a_marker(scripted, api_error_status):
    _write_session(scripted)
    scripted["queue"].append(
        (_api_error_stdout('API Error: 400 {"error":{"message":"Internal server error"}}',
                           terminal_reason="completed",
                           api_error_status=api_error_status), 1)
    )

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 1


def test_a_permanent_reason_does_not_veto_a_marker_it_cannot_explain(scripted):
    """An unrecognised `terminal_reason` must fall through to the markers rather
    than refuse: the enum is open and grew 12 -> 19 values across two builds, so
    reading anything unknown as permanent would disable this on a CLI bump."""
    _write_session(scripted)
    scripted["queue"].append(
        (_api_error_stdout("API Error: The operation timed out.",
                           terminal_reason="some_reason_added_next_release"), 1)
    )
    scripted["queue"].append((_ok_stdout("recovered"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 2


def test_the_original_failure_survives_into_the_final_error(scripted):
    """`_without_terminal_result` drops the event carrying the original text, and
    `cli_attempts` does not cross the A2A boundary — so the error string is the
    only field that can still name what actually went wrong."""
    _write_session(scripted)
    scripted["queue"].append(
        (_api_error_stdout("API Error: The operation timed out."), 1)
    )
    scripted["queue"].append((_api_error_stdout("API Error: something permanent"), 1))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 2
    assert result.error == (
        "claude CLI exited with code 1 on attempt 2: API Error: something permanent "
        "(attempt 1: API Error: The operation timed out.)"
    )


def test_a_resume_that_fails_the_same_way_names_it_once(scripted):
    _write_session(scripted)
    for _ in range(3):
        scripted["queue"].append((_api_error_stdout("API Error: The operation timed out."), 1))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 3
    assert result.error == "claude CLI exited with code 1 on attempt 3: API Error: The operation timed out."


def test_an_unresumed_turn_does_not_gain_an_attribution_suffix(scripted):
    _write_session(scripted)
    scripted["queue"].append((_api_error_stdout("API Error: something permanent"), 1))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 1
    assert result.error == "claude CLI exited with code 1: API Error: something permanent"


@pytest.mark.parametrize(
    "api_error_status, expected_attempts",
    [
        # litellm mints 408 for its own Timeout, so a proxy timeout — the failure
        # this retry exists for — arrives as a 4xx and must not be vetoed.
        (408, 2),
        (425, 2),
        # 429 stays vetoed: the CLI has already backed off on it.
        (429, 1),
        (400, 1),
        (401, 1),
    ],
)
def test_the_retryable_4xx_survive_the_veto(scripted, api_error_status, expected_attempts):
    _write_session(scripted)
    scripted["queue"].append(
        (_api_error_stdout(f"API Error: {api_error_status} Request timed out.",
                           terminal_reason="api_error",
                           api_error_status=api_error_status), 1)
    )
    scripted["queue"].append((_ok_stdout("recovered"), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == expected_attempts
