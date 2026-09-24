"""Unit tests for TerminalResponse.from_message.

The terminal A2A status message carries the agent's text reply plus a DataPart with
telemetry and — when the agent ran with output_format — the typed structured_output.
"""

from agent_env.a2a_agent.protocol import TerminalResponse, extract_terminal_response
from agentenv_protocol.a2a_agent import TaskResult, Usage
from agentenv_protocol.a2a_agent.framework import _to_a2a_parts


def _sdk_terminal_message(result: TaskResult) -> dict:
    """Serialize the terminal parts exactly as the A2A SDK sends them."""
    return {
        "parts": [
            part.model_dump(mode="json", by_alias=True, exclude_none=True)
            for part in _to_a2a_parts(result)
        ]
    }


def test_sdk_success_round_trips_through_platform_parser():
    result = (
        TaskResult.builder()
        .succeeded()
        .add_text("done")
        .add_structured_output({"answer": 42})
        .usage(Usage(tool_call_count=3, input_tokens=10))
        .build()
    )

    parsed = TerminalResponse.from_message(_sdk_terminal_message(result))

    assert parsed.response_text == "done"
    assert parsed.structured_output == {"answer": 42}
    assert parsed.tool_call_count == 3


def test_sdk_failure_round_trips_through_platform_parser():
    result = TaskResult.failure(
        "provider_rate_limited",
        "The provider is temporarily unavailable",
        error_type="infra_error",
    )

    parsed = TerminalResponse.from_message(_sdk_terminal_message(result))

    assert parsed.response_text == "The provider is temporarily unavailable"
    assert parsed.error_type == "infra_error"
    assert parsed.error_code == "provider_rate_limited"
    assert parsed.error_message == "The provider is temporarily unavailable"


def test_extracts_text_telemetry_and_structured_output():
    msg = {
        "parts": [
            {"kind": "text", "text": '{"artifacts": {"zip": "/app/x.zip"}}'},
            {"kind": "data", "data": {
                "tool_call_count": 4,
                "structured_output": {"artifacts": {"zip": "/app/x.zip"}},
            }},
        ]
    }
    tr = TerminalResponse.from_message(msg)
    assert isinstance(tr, TerminalResponse)
    assert tr.response_text == '{"artifacts": {"zip": "/app/x.zip"}}'
    assert tr.tool_call_count == 4
    assert tr.structured_output == {"artifacts": {"zip": "/app/x.zip"}}


def test_extracts_tool_count_from_typed_usage_with_legacy_fallback():
    typed = TerminalResponse.from_message(
        {
            "parts": [
                {
                    "kind": "data",
                    "data": {
                        "tool_call_count": 1,
                        "usage": {"tool_call_count": 4},
                    },
                }
            ]
        }
    )
    assert typed.tool_call_count == 4


def test_structured_output_none_when_absent():
    # Free-text reply / no output_format -> no structured_output on the DataPart.
    tr = TerminalResponse.from_message({"parts": [{"kind": "text", "text": "done"}]})
    assert tr.response_text == "done"
    assert tr.structured_output is None
    assert tr.tool_call_count is None


def test_extracts_error_fields_from_data_part():
    msg = {"parts": [
        {"kind": "text", "text": "boom"},
        {"kind": "data", "data": {
            "error_type": "agent_error", "error_code": "provider_rate_limited",
            "error_class": "ValueError", "error_message": "bad",
        }},
    ]}
    tr = TerminalResponse.from_message(msg)
    assert (tr.error_type, tr.error_class, tr.error_message) == ("agent_error", "ValueError", "bad")
    assert tr.error_code == "provider_rate_limited"
    assert tr.structured_output is None


def test_extract_terminal_response_shim_returns_legacy_5_tuple():
    msg = {"parts": [
        {"kind": "text", "text": "hi"},
        {"kind": "data", "data": {"tool_call_count": 2, "error_type": "E"}},
    ]}
    assert extract_terminal_response(msg) == ("hi", 2, "E", None, None)
