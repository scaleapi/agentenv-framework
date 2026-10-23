"""Regression tests for OpenCode's JSONL terminal/error event handling."""

import json

import pytest

from tst.unit.agents.opencode.agent import opencode_bridge
from tst.unit.agents.opencode.fake_cli import FakeCLI

_parse_events = opencode_bridge._parse_events


def _jsonl(*events: dict) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


def test_structured_stdout_error_is_returned_with_status_code():
    event = {
        "type": "error",
        "sessionID": "ses_test",
        "error": {
            "name": "APIError",
            "data": {
                "message": (
                    "litellm.BadRequestError: reasoning_effort 'max' is not "
                    "supported; use xhigh"
                ),
                "statusCode": 400,
                "responseHeaders": {"authorization": "must-not-be-rendered"},
            },
        },
    }

    result = _parse_events(_jsonl(event), b"", 1)

    assert result.error == (
        "APIError: litellm.BadRequestError: reasoning_effort 'max' is not "
        "supported; use xhigh"
    )
    assert result.error_code == "400"
    assert result.session_id == "ses_test"
    assert "authorization" not in result.error
    assert result.trajectory == [event]


def test_stderr_remains_the_fallback_when_no_structured_error_exists():
    result = _parse_events(b"", b"configuration failed\n", 1)

    assert result.error == "configuration failed"
    assert result.error_code is None


def test_completed_turn_does_not_fail_for_an_earlier_error_event():
    result = _parse_events(
        _jsonl(
            {
                "type": "error",
                "error": {"name": "RetryError", "data": {"message": "retrying"}},
            },
            {"type": "step_finish", "part": {"type": "step-finish", "reason": "stop"}},
        ),
        b"",
        0,
    )

    assert result.error is None
    assert result.error_code is None


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected_error", [
    ('{"answer":"done"}', None),
    ('{"other":"done"}', "invalid_output_format"),
    ("not JSON", "invalid_output_format"),
])
async def test_formatted_turn_requires_valid_json_schema_output(monkeypatch, answer, expected_error):
    async def fake_subprocess(*_args, **_kwargs):
        return FakeCLI(_jsonl(
            {"type": "text", "part": {"type": "text", "text": answer}},
            {"type": "step_finish", "part": {"type": "step-finish", "reason": "stop"}},
        ))

    monkeypatch.setattr(opencode_bridge.asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})
    result = await opencode_bridge.run_opencode(
        prompt="answer", context_id="format-test",
        output_format={"type": "json_schema", "schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }},
    )
    assert result.error_code == expected_error
    if expected_error is None:
        assert result.structured_output == {"answer": "done"}
    else:
        assert result.structured_output is None
        assert len(result.trajectory) == 2
