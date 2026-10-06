"""Unit tests for the AgentEnv-side A2A protocol helpers.

The terminal A2A status message carries the agent's text reply plus a DataPart with
telemetry and — when the agent ran with output_format — the typed structured_output.
"""

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from agent_env.a2a_agent import protocol
from agent_env.a2a_agent.protocol import (
    UNREACHABLE_AFTER_SECONDS,
    AgentUnreachableError,
    TerminalResponse,
    extract_terminal_response,
    poll_a2a_task,
    raise_for_extension_status,
)
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


def test_extension_http_error_names_the_sdk_error_code_only():
    response = httpx.Response(
        410,
        json={
            "error": {
                "code": "grant_expired",
                "message": "The transfer grant has expired.",
                "retryable": False,
            }
        },
        request=httpx.Request("POST", "https://agent.test/ext/snapshot"),
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        raise_for_extension_status(response, operation="snapshot save")

    assert str(raised.value) == "snapshot save failed with HTTP 410: grant_expired"
    assert raised.value.response is response


def test_extension_http_error_keeps_only_a_code_shaped_error_code():
    response = httpx.Response(
        502,
        json={"error": {"code": "https://bucket.test/key?X-Amz-Signature=abc"}},
        request=httpx.Request("POST", "https://agent.test/ext/snapshot"),
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        raise_for_extension_status(response, operation="snapshot save")

    assert str(raised.value) == "snapshot save failed with HTTP 502"


def test_extension_http_error_without_structured_body_omits_response_detail():
    response = httpx.Response(
        500,
        text="upstream failed",
        request=httpx.Request("POST", "https://agent.test/ext/snapshot"),
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        raise_for_extension_status(response, operation="snapshot save")

    assert str(raised.value) == "snapshot save failed with HTTP 500"


def test_extension_http_error_for_a_request_without_grants_keeps_the_body():
    response = httpx.Response(
        409,
        json={"detail": "cannot load a workspace while an agent turn is running"},
        request=httpx.Request("PUT", "https://agent.test/ext/snapshot"),
    )

    with pytest.raises(httpx.HTTPStatusError) as raised:
        raise_for_extension_status(response, operation="snapshot load", include_body=True)

    assert str(raised.value) == (
        'snapshot load failed with HTTP 409: '
        '{"detail":"cannot load a workspace while an agent turn is running"}'
    )


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


class _Agent:
    """An agent behind a mock transport, on a fake clock: ``answer(method, request, agent)`` replies to each
    JSON-RPC call, and sleeping only moves the clock."""

    def __init__(self, monkeypatch, answer):
        self.now = 0.0
        self.calls: list[str] = []
        self._answer = answer
        transport = httpx.MockTransport(self._handle)
        client = httpx.AsyncClient
        monkeypatch.setattr(protocol.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
        monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic=lambda: self.now))
        monkeypatch.setattr(protocol, "asyncio", SimpleNamespace(sleep=self._sleep))

    async def _sleep(self, seconds):
        self.now += seconds

    def _handle(self, request):
        method = json.loads(request.content)["method"]
        self.calls.append(method)
        return self._answer(method, request, self)


def _task(state):
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": "poll", "result": {"id": "t-1", "status": {"state": state}}})


def _disconnected(request, agent):
    raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)


def _refused(request, agent):
    raise httpx.ConnectError("Connection refused", request=request)


def _hung(request, agent):
    agent.now += 30
    raise httpx.ReadTimeout("timed out", request=request)


def _bad_gateway(request, agent):
    return httpx.Response(502)


def _working_then(failure):
    def answer(method, request, agent):
        return _task("working") if agent.calls.count("tasks/get") == 1 else failure(request, agent)
    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, last", [
    (_disconnected, "RemoteProtocolError"), (_refused, "ConnectError"), (_hung, "ReadTimeout"), (_bad_gateway, "HTTP 502"),
])
async def test_an_agent_whose_sandbox_died_is_given_up_on_naming_it(monkeypatch, failure, last):
    agent = _Agent(monkeypatch, _working_then(failure))

    with pytest.raises(AgentUnreachableError, match=rf"The agent on sandbox sb-1 stopped answering: .*\(last: {last}") as raised:
        await poll_a2a_task("http://agent", "t-1", 1200, 2, sandbox_id="sb-1")

    assert isinstance(raised.value, TimeoutError)
    assert agent.now <= 2 * UNREACHABLE_AFTER_SECONDS
    assert "tasks/cancel" not in agent.calls


@pytest.mark.asyncio
async def test_a_blip_shorter_than_the_window_is_ridden_out(monkeypatch):
    def answer(method, request, agent):
        if 10 < agent.now < 50:
            _disconnected(request, agent)
        return _task("completed" if agent.now > 50 else "working")

    agent = _Agent(monkeypatch, answer)

    result = await poll_a2a_task("http://agent", "t-1", 1200, 2, sandbox_id="sb-1")

    assert result["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_it_takes_three_unanswered_polls_however_far_apart(monkeypatch):
    agent = _Agent(monkeypatch, _working_then(_refused))

    with pytest.raises(AgentUnreachableError):
        await poll_a2a_task("http://agent", "t-1", 1200, 60, sandbox_id="sb-1")

    assert agent.calls.count("tasks/get") == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    httpx.Response(500),
    httpx.Response(200, json={"jsonrpc": "2.0", "id": "poll", "error": {"code": -32603, "message": "boom"}}),
], ids=["http-500", "json-rpc-error"])
async def test_an_agent_answering_with_errors_is_waited_on_until_the_timeout(monkeypatch, reply):
    agent = _Agent(monkeypatch, lambda method, request, agent: reply if method == "tasks/get" else _task("canceled"))

    with pytest.raises(TimeoutError, match="did not complete within 600s") as raised:
        await poll_a2a_task("http://agent", "t-1", 600, 2, sandbox_id="sb-1")

    assert not isinstance(raised.value, AgentUnreachableError)
    assert agent.now >= 600


@pytest.mark.asyncio
async def test_without_a_sandbox_a_silent_agent_is_waited_on_until_the_timeout(monkeypatch):
    agent = _Agent(monkeypatch, lambda method, request, agent: _refused(request, agent))

    with pytest.raises(TimeoutError, match="did not complete within 600s") as raised:
        await poll_a2a_task("http://agent", "t-1", 600, 2)

    assert type(raised.value) is TimeoutError
    assert agent.now >= 600


@pytest.mark.asyncio
async def test_running_out_of_time_cancels_the_task(monkeypatch):
    cancelled = []

    def answer(method, request, agent):
        if method == "tasks/cancel":
            cancelled.append(json.loads(request.content)["params"])
            return _task("canceled")
        return _task("working")

    agent = _Agent(monkeypatch, answer)

    with pytest.raises(TimeoutError, match="did not complete within 60s"):
        await poll_a2a_task("http://agent", "t-1", 60, 2)

    assert cancelled == [{"id": "t-1"}]
    assert agent.calls[-1] == "tasks/cancel"


def _unsupported(request, agent):
    return httpx.Response(200, json={
        "jsonrpc": "2.0", "id": "cancel", "error": {"code": -32004, "message": "This operation is not supported"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal, logged", [
    (_unsupported, "The agent didn't cancel A2A task t-1"),
    (lambda request, agent: httpx.Response(500), "Couldn't cancel A2A task t-1: HTTPStatusError"),
    (_refused, "Couldn't cancel A2A task t-1: ConnectError"),
], ids=["unsupported", "http-500", "unreachable"])
async def test_a_cancel_that_fails_still_reports_the_timeout(monkeypatch, caplog, refusal, logged):
    _Agent(monkeypatch, lambda method, request, agent: refusal(request, agent) if method == "tasks/cancel" else _task("working"))

    with caplog.at_level(logging.WARNING, logger=protocol.__name__):
        with pytest.raises(TimeoutError, match="did not complete within 60s"):
            await poll_a2a_task("http://agent", "t-1", 60, 2)

    assert logged in caplog.text
