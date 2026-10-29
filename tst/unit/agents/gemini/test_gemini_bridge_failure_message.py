"""A failed turn's error names the CLI's own cause, and reaches agent-env.

gemini-cli 0.38.2 reports an API failure, after its own retries, as a terminal
stream-json `result` event with status "error". A startup failure (bad settings,
no auth, an unknown resume id) exits before the stream starts and says why on
stderr only, behind the yolo banner and Node's TLS warning that open every run.
The streams and stderr below are trimmed from captures of the pinned CLI behind
the wrapper's own proxy, with keys replaced by fakes; the turn-limit result follows
the CLI's own handler for exceeding its session turns.

Run from this directory: pytest test_gemini_bridge_failure_message.py -q
"""
import asyncio
import json
import logging

import pytest

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge
from tst.unit.agents.gemini.fake_cli import FakeCLI
from tst.unit.agents.gemini.support import send, terminal

_FAKE_KEY = "sk-FAKEfake0000fake0000fak"

_BANNER = (
    "YOLO mode is enabled. All tool calls will be automatically approved.\n"
    "YOLO mode is enabled. All tool calls will be automatically approved.\n"
    "(node:26) Warning: Setting the NODE_TLS_REJECT_UNAUTHORIZED environment variable to '0' makes "
    "TLS connections and HTTPS requests insecure by disabling certificate verification.\n"
    "(Use `node --trace-warnings ...` to show where the warning was created)\n"
)
_AUTH_401 = (
    '[API Error: {"error":{"message":"Authentication Error, Invalid proxy server token passed. '
    "Received API Key = sk-...0000, Key Hash (Token) =deadbeef. Unable to find token in cache or "
    '`LiteLLM_VerificationTokenTable`","type":"token_not_found_in_db","param":"key","code":"401"}}]'
)
_BUDGET_400 = (
    '[API Error: {"error":{"message":"Budget has been exceeded! Key=pub-xxxx Spend=1 Budget=1",'
    '"type":"budget_exceeded","param":null,"code":"400"}}]'
)
_RATE_LIMIT_429 = (
    '[API Error: {"error":{"message":"LiteLLM Rate Limit Handler for rate limit type = key. Crossed '
    "TPM / RPM / Max Parallel Request Limit. current rpm: 1, rpm limit: 1, current tpm: 0, tpm limit: "
    "9223372036854775807, current max_parallel_requests: 0, max_parallel_requests: "
    '9223372036854775807","type":"None","param":"None","code":"429"}}]'
)
_RATE_LIMIT_ADVICE = (
    "\nPlease wait and try again later. To increase your limits, request a quota increase through "
    "AI Studio, or switch to another /auth method"
)
# LiteLLM's passthrough wraps Google's body as the repr of its bytes.
_UNAVAILABLE_503 = (
    r'''[API Error: {"error":{"message":"b'{\\n  \"error\": {\\n    \"code\": 503,\\n    \"message\": '''
    r'''\"This model is currently experiencing high demand. Spikes in demand are usually temporary. '''
    r'''Please try again later.\",\\n    \"status\": \"UNAVAILABLE\"\\n  }\\n}\\n'","type":"None",'''
    r'''"param":"None","code":"503"}}]'''
)
_PROXY_REFUSED = '[API Error: {"detail":"gemini upstream request failed: All connection attempts failed"}]'
_TURN_LIMIT = (
    "Reached max session turns for this session. Increase the number of turns by specifying "
    "maxSessionTurns in settings.json."
)
_STACK = (
    "    at throwErrorIfNotOK (file:///usr/local/lib/node_modules/@google/gemini-cli/bundle/"
    "chunk-ETUADTWF.js:36178:24)\n"
    "    at async Turn.run (file:///usr/local/lib/node_modules/@google/gemini-cli/bundle/"
    "chunk-IWSCP2GY.js:311329:24) {\n"
)


def _stdout(*events):
    init = {"type": "init", "session_id": "69dfa81e-475e-47af-848c-c1d7bec3c71e", "model": "gemini-2.5-flash"}
    prompt = {"type": "message", "role": "user", "content": "Say hi."}
    return "".join(json.dumps(e) + "\n" for e in (init, prompt, *events)).encode()


def _api_error(message, error_type="unknown"):
    return {"type": "result", "status": "error", "error": {"type": error_type, "message": message},
            "stats": {"tool_calls": 0}}


def _said(text):
    return {"type": "message", "role": "assistant", "content": text, "delta": True}


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Script the next `gemini` run as (stdout, stderr, returncode); returncode None never exits."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini_bridge, "GEMINI_PROJECTS_JSON", tmp_path / "projects.json")

    def _script(stdout, stderr, returncode):
        async def _exec(*cmd, **kwargs):
            return FakeCLI(stdout, stderr.encode(), returncode, hang=returncode is None)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)

    return _script


@pytest.fixture
def run(cli):
    def _run(stdout=b"", stderr="", returncode=1, **kwargs):
        cli(stdout, stderr, returncode)
        return asyncio.run(gemini_bridge.run_gemini(prompt="Say hi.", session_id=None, **kwargs))

    return _run


@pytest.mark.parametrize(
    "message, error_type, returncode, cause",
    [
        (_AUTH_401, "unknown", 145, _AUTH_401),
        (_BUDGET_400, "unknown", 144, _BUDGET_400),
        (_RATE_LIMIT_429 + _RATE_LIMIT_ADVICE, "unknown", 173, _RATE_LIMIT_429),
        (_UNAVAILABLE_503, "unknown", 247, _UNAVAILABLE_503),
        (_PROXY_REFUSED, "unknown", 246, _PROXY_REFUSED),
        (_TURN_LIMIT, "FatalTurnLimitedError", 53, f"FatalTurnLimitedError: {_TURN_LIMIT}"),
    ],
    ids=["auth", "budget", "rate-limit", "unavailable", "proxy-refused", "turn-limit"],
)
def test_the_cli_cause_follows_the_exit_code(run, message, error_type, returncode, cause):
    result = run(_stdout(_api_error(message, error_type)), _BANNER, returncode)

    assert result.error == f"gemini CLI exited with code {returncode}: {cause}"


def test_the_stream_cause_wins_over_the_logged_error(run):
    logged = (
        "Error when talking to Gemini API Full report available at: "
        "/tmp/gemini-client-error-Turn.run-sendMessageStream-2026-09-28T20-47-14-104Z.json _ApiError: "
        '{"error":{"message":"Budget has been exceeded! Key=pub-xxxx Spend=1 Budget=1",'
        '"type":"budget_exceeded","param":null,"code":"400"}}\n'
        f"{_STACK}  status: 400\n}}\n"
    )

    result = run(_stdout(_api_error(_BUDGET_400)), _BANNER + logged, 144)

    assert result.error == f"gemini CLI exited with code 144: {_BUDGET_400}"


@pytest.mark.parametrize(
    "stderr, returncode, cause",
    [
        (
            _BANNER
            + 'Error resuming session: Invalid session identifier "11111111-0000-4000-8000-000000000000".\n'
            "  Searched for sessions in /home/agent/.gemini/tmp/workspace/chats.\n"
            "  Use --list-sessions to see available sessions, then use --resume {number}, --resume {uuid}, "
            "or --resume latest.\n",
            42,
            'Error resuming session: Invalid session identifier "11111111-0000-4000-8000-000000000000". | '
            "Searched for sessions in /home/agent/.gemini/tmp/workspace/chats. | Use --list-sessions to see "
            "available sessions, then use --resume {number}, --resume {uuid}, or --resume latest.",
        ),
        (
            "YOLO mode is enabled. All tool calls will be automatically approved.\n"
            "Please set an Auth method in your /home/agent/.gemini/settings.json or specify one of the "
            "following environment variables before running: GEMINI_API_KEY, GOOGLE_GENAI_USE_VERTEXAI, "
            "GOOGLE_GENAI_USE_GCA\n",
            41,
            "Please set an Auth method in your /home/agent/.gemini/settings.json or specify one of the "
            "following environment variables before running: GEMINI_API_KEY, GOOGLE_GENAI_USE_VERTEXAI, "
            "GOOGLE_GENAI_USE_GCA",
        ),
        (
            "\x1b[31mError in /home/agent/.gemini/settings.json: Unexpected end of JSON input\n"
            "Please fix the configuration file(s) and try again.\x1b[0m\n",
            52,
            "Error in /home/agent/.gemini/settings.json: Unexpected end of JSON input | "
            "Please fix the configuration file(s) and try again.",
        ),
    ],
    ids=["unknown-resume-id", "no-auth", "bad-settings"],
)
def test_a_cli_that_died_before_its_stream_is_explained_by_stderr(run, stderr, returncode, cause):
    assert run(b"", stderr, returncode).error == f"gemini CLI exited with code {returncode}: {cause}"


def test_stderr_noise_alone_leaves_the_exit_code_alone(run):
    assert run(_stdout(), _BANNER, 1).error == "gemini CLI exited with code 1"


def test_a_killed_cli_names_the_signal_and_its_last_logged_error(run):
    retrying = (
        "Attempt 1 failed with status 502. Retrying with backoff... _ApiError: "
        '{"detail":"gemini upstream request failed: All connection attempts failed"}\n'
        f"{_STACK}  status: 502\n}}\n"
    )

    result = run(_stdout(), _BANNER + retrying, -9)

    assert result.error == (
        "gemini CLI was killed by signal 9: Attempt 1 failed with status 502. Retrying with backoff... "
        '_ApiError: {"detail":"gemini upstream request failed: All connection attempts failed"}'
    )


def test_a_long_cause_is_cut_to_the_limit(run):
    result = run(_stdout(_api_error("x" * 5000)), _BANNER, 1)

    assert result.error == "gemini CLI exited with code 1: " + "x" * gemini_bridge._ERROR_DETAIL_LIMIT


def test_a_long_stderr_keeps_its_end(run):
    result = run(b"", _BANNER + "x" * 5000 + "\nfinal stderr line\n", 1)

    assert result.error.endswith("x | final stderr line")
    assert len(result.error) == len("gemini CLI exited with code 1: ") + gemini_bridge._ERROR_DETAIL_LIMIT


def test_a_key_cut_by_the_stderr_tail_is_still_redacted(run):
    stderr = "x" * 100 + " " + _FAKE_KEY + " " + "y" * 573

    assert _FAKE_KEY[3:] not in run(b"", stderr, 1).error


def test_a_key_in_the_cli_error_is_redacted_from_error_response_and_logs(run, caplog):
    leak = _api_error(
        f"[API Error: Header 'x-goog-api-key' has invalid value: 'Bearer {_FAKE_KEY}' ({_FAKE_KEY})]"
    )

    with caplog.at_level(logging.ERROR, logger=gemini_bridge.__name__):
        result = run(_stdout(_said(f"Calling with {_FAKE_KEY}"), leak), _BANNER, 1)

    assert result.error == (
        "gemini CLI exited with code 1: [API Error: Header 'x-goog-api-key' has invalid value: "
        "'[redacted]' ([redacted])]"
    )
    assert result.response == "Calling with [redacted]"
    assert _FAKE_KEY not in caplog.text


def test_a_google_key_or_an_echoed_header_value_is_redacted(run):
    google_key = "AIzaFAKEfake0000fake0000fake0000fake000"
    leak = _api_error(
        f"[API Error: key {google_key} rejected; Header 'x-goog-api-key' has invalid value: 'plain-key\nsecond-line']"
    )

    result = run(_stdout(leak), _BANNER, 1)

    assert result.error == (
        "gemini CLI exited with code 1: [API Error: key [redacted] rejected; "
        "Header 'x-goog-api-key' has invalid value: '[redacted]']"
    )


def test_redaction_leaves_masked_keys_and_ids_alone():
    text = ("Received API Key = sk-...0000 for task-abcdefghijkl; raw sk-abcd1234efgh; "
            "google AIzaFAKEfake0000fake0000fake0000fake000; "
            'Bearer tok.en; {"auth":"Bearer abc123","code":"401"}; '
            "Header 'x-goog-api-key' has invalid value: 'owner's note\nopaque-secret'")

    assert gemini_bridge._redact(text) == (
        "Received API Key = sk-...0000 for task-abcdefghijkl; raw [redacted]; google [redacted]; "
        '[redacted]; {"auth":"[redacted]","code":"401"}; '
        "Header 'x-goog-api-key' has invalid value: '[redacted]'"
    )


def test_a_timeout_is_reported_as_before_and_keeps_the_session_the_cli_started(run):
    result = run(_stdout(), returncode=None, timeout_s=0.05)

    assert (result.error, result.session_id) == (
        "gemini CLI timed out after 0.05s", "69dfa81e-475e-47af-848c-c1d7bec3c71e")


# --- what agent-env reads off the wire --------------------------------------------


def test_a_failed_turn_puts_the_cause_on_the_wire(cli, client):
    cli(_stdout(_api_error(_BUDGET_400)), _BANNER, 144)

    state, text, data = terminal(send(client, "Say hi."))

    error = f"gemini CLI exited with code 144: {_BUDGET_400}"
    assert (state, text) == ("failed", error)
    assert data == [
        {"tool_call_count": 0},
        {"usage": {"tool_call_count": 0}, "error_type": "agent_error", "error_code": "cli_error", "error_message": error},
    ]


def test_a_failed_turn_keeps_its_partial_answer_and_bounds_the_message(monkeypatch, client):
    async def fake_run_gemini(**kwargs):
        return gemini_bridge.GeminiResult(response="partial answer", tool_call_count=2, session_id="s-1",
                                          error="x" * 1500)

    monkeypatch.setattr(a2a_server, "run_gemini", fake_run_gemini)

    state, text, data = terminal(send(client, "Say hi."))

    assert (state, text, data[-1]["error_message"]) == ("failed", "partial answer", "x" * 1000)


def test_a_finished_turn_is_unchanged(cli, agent, client):
    done = {"type": "result", "status": "success", "stats": {"tool_calls": 0}}
    cli(_stdout(_said("hello"), done), _BANNER, 0)

    assert terminal(send(client, "Say hi.")) == (
        "completed", "hello", [{"tool_call_count": 0}, {"usage": {"tool_call_count": 0}}])
    assert agent.session_ref_for_context("ctx-1") == "69dfa81e-475e-47af-848c-c1d7bec3c71e"
