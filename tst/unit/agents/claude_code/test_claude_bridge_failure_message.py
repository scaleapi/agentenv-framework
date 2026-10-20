"""A failed turn's error names the CLI's own cause, ahead of anything a caller may cut.

The CLI reports an API failure as a terminal `result` event with `is_error` set
(subtype "success" on 2.1.177) and exits 1 with nothing on stderr, so the cause is
in the stream or nowhere. The event shapes below are copied from captures of the
pinned CLI, with keys replaced by fakes.

"""
import asyncio
import json
import logging

import pytest

from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.agent import claude_bridge

# What agent-env keeps of an agent's error_message, and a2a_server's own cap.
_AGENT_ENV_KEEPS = 500
_A2A_SERVER_CAP = 1000

_REFUSAL = (
    "API Error: Claude Code is unable to respond to this request, which appears to violate "
    "our Usage Policy (https://www.anthropic.com/legal/aup). API integrators: you can reduce "
    "refusals for your users by configuring a fallback model — see "
    "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback. Try "
    "rephrasing the request in a new session or change your model."
)
_FAKE_KEY = "sk-FAKEfake0000fake0000fak"


def _stdout(*events):
    lines = [{"type": "system", "subtype": "init", "session_id": "sess-1"}, *events]
    return "".join(json.dumps(e) + "\n" for e in lines).encode()


def _api_error(text, **fields):
    return {
        "type": "result", "subtype": "success", "is_error": True, "session_id": "sess-1",
        "terminal_reason": "completed", "result": text, **fields,
    }


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run one turn against scripted `claude -p` invocations of (stdout, returncode[, stderr])."""
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.delenv("CLAUDE_API_ERROR_RETRIES", raising=False)
    (tmp_path / "sess-1.jsonl").write_text('{"type":"assistant"}\n')

    def _run(*invocations, **kwargs):
        queue = list(invocations)

        async def _exec(*cmd, **kwargs):
            return FakeCLI(*queue.pop(0))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
        return asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, **kwargs))

    return _run


@pytest.mark.parametrize(
    "event",
    [
        _api_error("API Error: 400 Budget has been exceeded! Key=pub-xxxx Spend=1 Budget=1",
                   api_error_status=400),
        _api_error("Failed to authenticate. API Error: 401 Authentication Error, Invalid proxy "
                   "server token passed. Received API Key = sk-...0000, Key Hash (Token) =deadbeef.",
                   api_error_status=401),
        _api_error("API Error: 400 {'error': '/v1/messages: Invalid model name passed in "
                   "model=claude-nope-9. Call `/v1/models` to view available models for your key.'}",
                   api_error_status=400),
        _api_error("API Error: Request rejected (429) · 429: Rate limit exceeded for api_key: "
                   "pub-xxxx. Limit type: tokens.", api_error_status=429),
        _api_error("Prompt is too long", api_error_status=400, terminal_reason="prompt_too_long"),
    ],
    ids=["budget", "auth", "invalid-model", "rate-limit", "prompt-too-long"],
)
def test_the_cli_cause_follows_the_exit_code(run, event):
    result = run((_stdout(event), 1))

    assert result.attempts == 1
    assert result.error == f"claude CLI exited with code 1: {event['result']}"


def test_an_errors_only_result_names_its_errors(run):
    max_turns = {
        "type": "result", "subtype": "error_max_turns", "is_error": True,
        "terminal_reason": "max_turns", "errors": ["Reached maximum number of turns (1)"],
    }

    result = run((_stdout(max_turns), 1))

    assert result.error == "claude CLI exited with code 1: Reached maximum number of turns (1)"


def test_a_structured_answer_keeps_its_response_and_still_reports_the_failure(run):
    event = _api_error("API Error: Request rejected (429) · 429: Rate limit exceeded",
                       api_error_status=429, structured_output={"verdict": "pass"})

    result = run((_stdout(event), 1))

    assert result.response == json.dumps({"verdict": "pass"})
    assert result.error == "claude CLI exited with code 1: API Error: Request rejected (429) · 429: Rate limit exceeded"


def test_a_killed_cli_names_the_signal(run):
    tool_use = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "pkill -9 -f gimp"}},
    ]}}

    result = run((_stdout(tool_use), -9))

    assert result.error == "claude CLI was killed by signal 9"


def test_nothing_to_report_keeps_the_exit_code_alone(run):
    assert run((b"", 1)).error == "claude CLI exited with code 1"


def test_a_finished_turn_that_exits_nonzero_keeps_its_answer_out_of_the_error(run):
    answer = {"type": "result", "subtype": "success", "is_error": False, "result": "the answer"}

    result = run((_stdout(answer), 1))

    assert result.response == "the answer"
    assert result.error == "claude CLI exited with code 1"


def test_an_error_event_on_a_zero_exit_stays_a_success(run):
    result = run((_stdout(_api_error("API Error: 400 bad", api_error_status=400)), 0))

    assert result.error is None


@pytest.mark.parametrize("errors", [[{"message": "boom"}], "boom"])
def test_an_unexpected_errors_shape_falls_back_to_the_exit_code(run, errors):
    event = {"type": "result", "subtype": "error_during_execution", "is_error": True, "errors": errors}

    assert run((_stdout(event), 1)).error == "claude CLI exited with code 1"


def test_stderr_explains_a_cli_that_emitted_no_result(run):
    result = run((b"", 1, b"No conversation found with session ID: abc\n"))

    assert result.error == "claude CLI exited with code 1: No conversation found with session ID: abc"


def test_the_cli_result_wins_over_stderr(run):
    event = _api_error("API Error: 400 Budget has been exceeded!", api_error_status=400)

    result = run((_stdout(event), 1, b"(node:12) Warning: something noisy\n"))

    assert result.error == "claude CLI exited with code 1: API Error: 400 Budget has been exceeded!"


def test_a_long_cause_on_a_resumed_turn_keeps_the_attempt_in_view(run):
    refusal = (_stdout(_api_error(_REFUSAL + " " + "x" * 200)), 1)

    result = run(refusal, refusal, refusal)

    assert result.attempts == 3
    assert "claude CLI exited with code 1 on attempt 3: API Error: Claude Code is unable" in result.error[:_AGENT_ENV_KEEPS]
    assert len(result.error) <= _A2A_SERVER_CAP


def test_verbose_stderr_after_a_resume_stays_under_the_cap(run):
    timed_out = (_stdout(_api_error("API Error: The operation timed out.")), 1)
    died = (b"", 1, b"x" * 1500 + b"\nfinal stderr line\n")

    result = run(timed_out, died)

    assert result.attempts == 2
    assert result.error.startswith("claude CLI exited with code 1 on attempt 2: ")
    assert result.error.endswith(" | final stderr line (attempt 1: API Error: The operation timed out.)")
    assert len(result.error) <= _A2A_SERVER_CAP


def test_a_crash_keeps_its_error_line_ahead_of_its_stack(run):
    crash = b"TypeError: Cannot read properties of undefined (reading 'map')\n" + (
        b"    at fn (file:///usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js:1003:2003)\n" * 10
    ) + b"\nNode.js v20.18.1\n"

    result = run((b"", 1, crash))

    assert result.error == (
        "claude CLI exited with code 1: TypeError: Cannot read properties of undefined "
        "(reading 'map') | Node.js v20.18.1"
    )


def test_a_key_cut_by_the_stderr_tail_is_still_redacted(run):
    stderr = "x" * 100 + " " + _FAKE_KEY + " " + "y" * 573

    assert _FAKE_KEY[3:] not in run((b"", 1, stderr.encode())).error


def test_a_key_in_the_cli_error_is_redacted_from_error_response_and_logs(run, caplog):
    # The shape of a real incident: a multi-line key value echoed in an HTTP header error.
    leak = (_stdout(_api_error(
        f"API Error: Header '14' has invalid value: 'Bearer Name: PUB_FAKE\nValue: {_FAKE_KEY}'"
    )), 1)

    with caplog.at_level(logging.INFO, logger=claude_bridge.__name__):
        result = run(leak, leak, leak)

    assert result.attempts == 3
    assert "[redacted]" in result.error
    for text in (result.error, result.response, caplog.text):
        assert _FAKE_KEY not in text


def test_a_key_a_failed_call_echoes_is_redacted_from_the_events_streamed_and_kept_and_from_the_transcript(run, tmp_path):
    text = f"API Error: 400 Budget has been exceeded! Key={_FAKE_KEY}"
    # The pinned CLI's failure, captured: an error `assistant` event, the error `result`, and the transcript's record.
    failed = {"type": "assistant", "is_api_error_message": True, "error": "unknown", "session_id": "sess-1",
              "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text(json.dumps({"type": "assistant", "isApiErrorMessage": True, "message": failed["message"]}) + "\n")
    streamed: list[dict] = []

    result = run((_stdout(failed, _api_error(text, api_error_status=400)), 1), event_callback=streamed.append)

    kept = json.dumps([streamed, result.trajectory, claude_bridge.read_partial_session_trajectory("sess-1")[0]])
    assert _FAKE_KEY not in kept and _FAKE_KEY not in transcript.read_text()
    assert "[redacted]" in kept and "[redacted]" in transcript.read_text()


def test_redaction_leaves_masked_keys_and_ids_alone():
    text = ("Received API Key = sk-...0000 for task-abcdefghijkl; raw sk-abcd1234efgh; "
            "google AIzaFAKEfake0000fake0000fake0000fake000; "
            'Bearer tok.en; {"auth":"Bearer abc123","code":"401"}; '
            "Header 'x-goog-api-key' has invalid value: 'owner's note\nopaque-secret'")

    assert claude_bridge._redact(text) == (
        "Received API Key = sk-...0000 for task-abcdefghijkl; raw [redacted]; google [redacted]; "
        '[redacted]; {"auth":"[redacted]","code":"401"}; '
        "Header 'x-goog-api-key' has invalid value: '[redacted]'"
    )
