"""A failed turn's error names codex's own cause, ahead of anything a caller may cut.

codex 0.147 reports an API failure in its `turn.failed` event and exits 1 with
nothing on stderr, so the cause is in the stream or nowhere. The streams below are
copied from captures of the pinned CLI (`exec --json`, the bridge's argv) against a
LiteLLM-shaped upstream, with keys replaced by fakes.
"""
import asyncio
import json
import logging

import pytest

from tst.unit.agents.codex.agent import codex_bridge

# What agent-env keeps of an agent's error_message.
_AGENT_ENV_KEEPS = 500

THREAD_ID = "01a0e9c4-31e3-79f1-a8b4-cc1d1cb064fe"
THREAD_STARTED = {"type": "thread.started", "thread_id": THREAD_ID}
TURN_STARTED = {"type": "turn.started"}
ADVISORY = {"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": "Model metadata for `gpt-5.3-codex` not found. Defaulting to fallback metadata; this can degrade performance and cause issues."}}

AUTH_401 = "unexpected status 401 Unauthorized: Authentication Error, Invalid proxy server token passed. Received API Key = sk-...1234, Key Hash (Token) =deadbeefdeadbeef. Unable to find token in cache or `LiteLLM_VerificationTokenTable`, url: http://127.0.0.1:18080/openai/v1/responses, request id: req_mock_3"
BUDGET_400 = "{\"error\": {\"message\": \"Budget has been exceeded! Key=pub-xxxx Spend=1 Budget=1\", \"type\": \"budget_exceeded\", \"param\": null, \"code\": \"400\"}}"
BUDGET_CAUSE = "Budget has been exceeded! Key=pub-xxxx Spend=1 Budget=1 [400, budget_exceeded]"
BAD_MODEL_400 = "{\"error\": {\"message\": \"{'error': '/openai/v1/responses: Invalid model name passed in model=gpt-nope-9. Call `/v1/models` to view available models for your key.'}\", \"type\": \"None\", \"param\": \"None\", \"code\": \"400\"}}"
CONTEXT_WINDOW = "Codex ran out of room in the model's context window. Start a new thread or clear earlier history before retrying."
RATE_LIMIT_429 = "exceeded retry limit, last status: 429 Too Many Requests, request id: req_mock_{}"
HIGH_DEMAND_500 = "We're currently experiencing high demand, which may cause temporary errors."
UNAVAILABLE_503 = "unexpected status 503 Service Unavailable: litellm.ServiceUnavailableError: OpenAIException - Service Unavailable, url: http://127.0.0.1:18080/openai/v1/responses, request id: req_mock_9"
STREAM_RATE_LIMIT = "stream disconnected before completion: Your requests to gpt-5.6-luna for gpt-5.6-luna in westus have exceeded token rate limit."
TOOL_ROUTER_STDERR = b"2026-09-28T20:47:28.641556Z ERROR codex_core::tools::router: error=unsupported call: apply_patch\n"
MISSING_THREAD_STDERR = b"Error: thread/resume: thread/resume failed: no rollout found for thread id 019f0000-0000-7000-8000-000000000000 (code -32600)\n"
_FAKE_KEY = "sk-FAKEleakFAKEleak9999"
LEAKY_401 = f"unexpected status 401 Unauthorized: Authentication Error: upstream rejected Authorization: Bearer {_FAKE_KEY} for key {_FAKE_KEY}, url: http://127.0.0.1:18080/openai/v1/responses, request id: req_mock_3"


def failed(message, reconnects=()):
    """One attempt as codex prints it: reconnect notices, the error, then turn.failed."""
    notices = [{"type": "error", "message": f"Reconnecting... {n}/2 ({text})"}
               for n, text in enumerate(reconnects, 1)]
    return [THREAD_STARTED, ADVISORY, TURN_STARTED, *notices,
            {"type": "error", "message": message},
            {"type": "turn.failed", "error": {"message": message}}]


def ndjson(events):
    return "".join(json.dumps(e) + "\n" for e in events).encode()


class _Stream:
    def __init__(self, data: bytes):
        self.data = data

    async def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            data, self.data = self.data, b""
            return data
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk


class _Stdin:
    def write(self, data): pass
    async def drain(self): pass
    def close(self): pass


class _Proc:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self.stdin, self.stdout, self.stderr = _Stdin(), _Stream(stdout), _Stream(stderr)
        self.returncode = returncode
        self.pid = None

    async def wait(self):
        return self.returncode

    def kill(self):
        pass


@pytest.fixture
def attempts(monkeypatch, tmp_path):
    """Queue (events, stderr, returncode) per codex invocation."""
    queue: list[tuple[list[dict], bytes, int]] = []

    async def fake_exec(*cmd, **kwargs):
        events, stderr, rc = queue.pop(0)
        return _Proc(ndjson(events), stderr, rc)

    rollout = tmp_path / f"rollout-{THREAD_ID}.jsonl"
    rollout.write_text("{}\n")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: rollout)
    monkeypatch.setattr(codex_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.delenv("CODEX_API_ERROR_RETRIES", raising=False)
    monkeypatch.chdir(tmp_path)
    return queue


def run(**kwargs):
    return asyncio.run(codex_bridge.run_codex(prompt="Say hi.", thread_id=None, **kwargs))


@pytest.mark.parametrize("attempt, expected", [
    (failed(AUTH_401, [AUTH_401.replace("req_mock_3", "req_mock_1"),
                       AUTH_401.replace("req_mock_3", "req_mock_2")]),
     "codex CLI exited with code 1: " + AUTH_401),
    (failed(BUDGET_400), "codex CLI exited with code 1: " + BUDGET_CAUSE),
    (failed(BAD_MODEL_400),
     ("codex CLI exited with code 1: {'error': '/openai/v1/responses: Invalid model name passed in "
      "model=gpt-nope-9. Call `/v1/models` to view available models for your key.'} [400]")),
    (failed(CONTEXT_WINDOW), "codex CLI exited with code 1: " + CONTEXT_WINDOW),
], ids=["auth", "budget", "invalid-model", "context-window"])
def test_the_fatal_event_is_the_cause(attempts, attempt, expected):
    attempts.append((attempt, b"", 1))

    result = run()

    assert (result.error, result.attempts) == (expected, 1)


def test_tracing_on_stderr_does_not_displace_the_fatal_event(attempts):
    attempts.append((failed(BUDGET_400), TOOL_ROUTER_STDERR, 1))

    assert run().error == "codex CLI exited with code 1: " + BUDGET_CAUSE


def test_stderr_explains_a_cli_that_emitted_no_event(attempts):
    attempts.append(([], MISSING_THREAD_STDERR, 1))

    assert run().error == "codex CLI exited with code 1: " + MISSING_THREAD_STDERR.decode().strip()


def test_a_killed_cli_names_the_signal_not_an_advisory(attempts):
    attempts.append(([THREAD_STARTED, ADVISORY, TURN_STARTED], b"", -9))

    assert run().error == "codex CLI was killed by signal 9"


def test_a_turn_failed_on_a_zero_exit_says_so(attempts):
    attempts.append((failed(AUTH_401), b"", 0))

    assert run().error == "codex CLI turn failed: " + AUTH_401


def test_a_turn_failed_that_the_watchdog_stopped_is_not_called_a_kill(attempts):
    attempts.append((failed(BUDGET_400), b"", -9))

    assert run().error == "codex CLI turn failed: " + BUDGET_CAUSE


def test_a_kill_during_a_reconnect_names_the_reconnect_not_tracing(attempts):
    notice = "Reconnecting... 2/5 (stream disconnected before completion: stream closed before response.completed)"
    attempts.append(([THREAD_STARTED, ADVISORY, TURN_STARTED, {"type": "error", "message": notice}],
                     TOOL_ROUTER_STDERR, -9))

    assert run().error == "codex CLI was killed by signal 9: " + notice


def test_the_cause_is_bounded(attempts):
    attempts.append((failed("x" * 5000), b"", 1))

    assert run().error == "codex CLI exited with code 1: " + "x" * 600


def test_a_key_in_the_cli_error_is_redacted_from_error_response_warnings_and_logs(attempts, caplog):
    attempt = failed(LEAKY_401, [LEAKY_401.replace("req_mock_3", "req_mock_1"),
                                 LEAKY_401.replace("req_mock_3", "req_mock_2")])
    attempt.insert(3, {"type": "item.completed", "item": {
        "id": "item_1", "type": "agent_message", "text": f"partial, key {_FAKE_KEY}"}})
    attempts.append((attempt, b"", 1))

    with caplog.at_level(logging.INFO, logger=codex_bridge.__name__):
        result = run()

    assert "Authorization: [redacted] for key [redacted], url:" in result.error
    for text in (result.error, result.response, *result.warnings, caplog.text):
        assert _FAKE_KEY not in text
    # The notices stay JSON, their keys intact.
    assert [json.loads(w)["type"] for w in result.warnings] == ["error"] * 4


def test_a_key_cut_by_the_stderr_tail_is_still_redacted(attempts):
    key = "sk-FAKEfake0000fake0000fakeZZ"
    stderr = "x" * 100 + " " + key + " " + "y" * 573
    attempts.append(([], stderr.encode(), 1))

    assert key[3:] not in run().error


def test_redaction_leaves_masked_keys_and_ids_alone():
    text = ("Received API Key = sk-...0000 for task-abcdefghijkl; raw sk-abcd1234efgh; "
            "google AIzaFAKEfake0000fake0000fake0000fake000; "
            'Bearer tok.en; {"auth":"Bearer abc123","code":"401"}; '
            "Header 'x-goog-api-key' has invalid value: 'owner's note\nopaque-secret'")

    assert codex_bridge._redact(text) == (
        "Received API Key = sk-...0000 for task-abcdefghijkl; raw [redacted]; google [redacted]; "
        '[redacted]; {"auth":"[redacted]","code":"401"}; '
        "Header 'x-goog-api-key' has invalid value: '[redacted]'"
    )


def test_a_resumed_turn_names_the_final_cause_and_how_it_began(attempts):
    attempts.extend([(failed(UNAVAILABLE_503), b"", 1), (failed(BUDGET_400), b"", 1)])

    result = run()

    assert result.attempts == 2
    assert result.error == (
        f"codex CLI exited with code 1 on attempt 2: {BUDGET_CAUSE} (attempt 1: {UNAVAILABLE_503})"
    )


def test_a_resume_that_fails_the_same_way_names_it_once(attempts):
    attempts.extend([(failed(RATE_LIMIT_429.format(n)), b"", 1) for n in (1, 2, 3)])

    result = run()

    assert (result.attempts, result.error) == (
        3, "codex CLI exited with code 1 on attempt 3: " + RATE_LIMIT_429.format(3))


@pytest.mark.parametrize("suffix", ["request id: req_mock_{}", "cf-ray: a3ed0000{}-PDX"])
def test_per_request_ids_do_not_make_a_failure_different(suffix):
    first, last = (f"exceeded retry limit, {suffix.format(n)}" for n in (1, 3))

    assert codex_bridge._failure_message("head", last, 3, first) == f"head on attempt 3: {last}"


def test_a_long_first_failure_keeps_the_final_cause_in_view(attempts):
    first = STREAM_RATE_LIMIT + " " + "y" * 600
    attempts.extend([(failed(first), b"", 1), (failed(BUDGET_400), b"", 1)])

    error = run().error

    assert f"on attempt 2: {BUDGET_CAUSE}" in error[:_AGENT_ENV_KEEPS]
    assert error.endswith(f"(attempt 1: {first[:200]})")


def test_repeated_high_demand_retries_and_reports_the_last_attempt(attempts):
    attempts.extend([(failed(HIGH_DEMAND_500, [HIGH_DEMAND_500] * 2), b"", 1)] * 3)

    result = run()

    assert (result.attempts, result.error) == (
        3, "codex CLI exited with code 1 on attempt 3: " + HIGH_DEMAND_500)


def test_a_stall_during_a_reconnect_names_the_reconnect(monkeypatch, tmp_path):
    notice = ("Reconnecting... 2/12 (stream disconnected before completion: Your requests to "
              "gpt-5.6-luna for gpt-5.6-luna in westus have exceeded rate limit.)")

    class Silent(_Stream):
        async def read(self, n=-1):
            if self.data:
                return await super().read(n)
            await asyncio.sleep(3600)

    proc = _Proc(b"", b"", 0)
    proc.stdout = Silent(ndjson([THREAD_STARTED, TURN_STARTED, {"type": "error", "message": notice}]))

    async def fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: None)
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(codex_bridge, "_kill_process_group", lambda _: None)
    monkeypatch.chdir(tmp_path)

    assert run(timeout_s=30).error == (
        "codex CLI produced no output for 0.05s "
        "(stalled — likely a hung model call through the proxy): " + notice
    )


def test_a_timeout_names_its_budget_in_whole_seconds(monkeypatch, tmp_path):
    async def fake_exec(*cmd, **kwargs):
        return _Proc(b"", b"", -9)

    async def deadline_passed(*args, **kwargs):
        return [THREAD_STARTED, TURN_STARTED], b"", True, False

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(codex_bridge, "_read_with_watchdog", deadline_passed)
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: None)
    monkeypatch.chdir(tmp_path)

    assert run(timeout_s=90).error == "codex CLI timed out after 90s"
