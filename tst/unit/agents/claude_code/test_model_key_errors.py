"""A turn the model endpoint refused for the run's key fails as model_key_budget or model_key_auth, not agent_error.

The CLI exits 1 and copies the endpoint's status and error body `type` into its terminal `result` event. The events
below are trimmed from captures of CLI 2.1.325 against a local endpoint answering with each body.
"""
import asyncio
import json

import pytest

from tst.unit.agents.claude_code.agent import claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI, send, terminal


def _result(status: int, kind: str, text: str, terminal_reason: str = "api_error") -> dict:
    return {
        "type": "result", "subtype": "success", "is_error": True, "session_id": "sess-1", "num_turns": 1,
        "terminal_reason": terminal_reason, "api_error_status": status, "api_error_type": kind, "result": text,
    }


@pytest.fixture
def cli(monkeypatch):
    """Every `claude -p` prints the init event and ``event``, then exits 1."""
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))

    def _script(event: dict) -> None:
        stdout = "".join(json.dumps(e) + "\n" for e in (
            {"type": "system", "subtype": "init", "session_id": "sess-1"}, event)).encode()

        async def _exec(*cmd, **kwargs):
            return FakeCLI(stdout, 1)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)

    return _script


@pytest.mark.parametrize(("event", "error_type"), [
    (_result(400, "budget_exceeded", "API Error: 400 Budget has been exceeded! Current cost: 10.1, Max budget: 10.0"),
     "model_key_budget"),
    (_result(401, "auth_error",
             "Failed to authenticate. API Error: 401 Authentication Error, Invalid proxy server token passed."),
     "model_key_auth"),
    (_result(401, "authentication_error", "Invalid API key · Fix external API key"), "model_key_auth"),
    (_result(403, "permission_error", "Failed to authenticate. API Error: 403 forbidden"), "model_key_auth"),
    (_result(429, "rate_limit_error", "API Error: Request rejected (429) · slow down"), "agent_error"),
    (_result(400, "invalid_request_error", "Prompt is too long", terminal_reason="prompt_too_long"), "agent_error"),
], ids=["litellm-budget", "litellm-auth", "anthropic-auth", "anthropic-permission", "rate-limit", "prompt-too-long"])
def test_a_failed_turn_says_whether_the_endpoint_refused_the_key(cli, client, event, error_type):
    cli(event)

    state, _, data = terminal(send(client, "hello"))

    assert state == "failed"
    assert (data[-1]["error_type"], data[-1]["error_code"]) == (error_type, "cli_error")
    assert event["result"] in data[-1]["error_message"]
