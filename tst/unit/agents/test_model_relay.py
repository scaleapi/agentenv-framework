"""The model relay an agent's CLI calls in place of the endpoint: it forwards each call as the endpoint answered it and
counts the call from the answer's own usage. Driven on loopback against a stand-in endpoint."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import agent_env
import httpx
import pytest
from aiohttp import web

_PATH = Path(agent_env.__file__).parent / "utils" / "model_relay" / "model_relay.py"
_spec = importlib.util.spec_from_file_location("shared_model_relay", _PATH)
model_relay = importlib.util.module_from_spec(_spec)
# Registered first: its dataclass resolves its annotations through the module's entry.
sys.modules[_spec.name] = model_relay
_spec.loader.exec_module(model_relay)

# The relay and the stand-in endpoint listen on loopback.
pytestmark = pytest.mark.usefixtures("socket_enabled")

# A Claude answer through Scale's LiteLLM proxy, streamed: its usage on the last chunk, the cache counted in
# prompt_tokens and named beside it.
CLAUDE_STREAM = (
    'data: {"choices": [{"delta": {"content": "4"}}]}\n\n'
    'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
    'data: {"choices": [], "usage": {"prompt_tokens": 20948, "completion_tokens": 43, "total_tokens": 20991, '
    '"prompt_tokens_details": {"cached_tokens": 10363}, "cache_read_input_tokens": 10363, '
    '"cache_creation_input_tokens": 199}}\n\n'
    "data: [DONE]\n\n"
)


class Endpoint:
    """The stand-in endpoint: answers each request with the next of ``answers`` (status, body, content type), and
    keeps what each request carried."""

    def __init__(self) -> None:
        self.answers: list[tuple[int, str, str]] = []
        self.requests: list[dict] = []
        self.hold: asyncio.Event | None = None
        self.hold_headers: asyncio.Event | None = None
        # Each answer written this many bytes at a time, when set, rather than an event at a time.
        self.write_bytes: int | None = None

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append({"method": request.method, "path": request.path_qs, "body": await request.read(),
                              "headers": {name.lower(): value for name, value in request.headers.items()}})
        status, body, kind = self.answers.pop(0)
        if self.hold_headers is not None:
            await self.hold_headers.wait()
        response = web.StreamResponse(status=status, headers={"content-type": kind})
        await response.prepare(request)
        if self.write_bytes is not None:
            encoded = body.encode()
            for start in range(0, len(encoded), self.write_bytes):
                await response.write(encoded[start:start + self.write_bytes])
                await asyncio.sleep(0)
            await response.write_eof()
            return response
        chunks = body.split("\n\n")
        for index, chunk in enumerate(chunks):
            await response.write((chunk + ("\n\n" if index < len(chunks) - 1 else "")).encode())
            if self.hold is not None and index == 0:
                await self.hold.wait()
        await response.write_eof()
        return response


async def _serve(endpoint: Endpoint) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", endpoint.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def _through_relay(answers, exercise, **relay_options):
    """Run ``exercise(client, relay_root)`` against a relay to a stand-in endpoint answering ``answers``; returns what
    it returned, the relay's calls and the endpoint."""
    endpoint, calls = Endpoint(), model_relay.Calls()
    endpoint.answers = list(answers)

    async def run():
        runner, upstream = await _serve(endpoint)
        try:
            async with model_relay.model_relay(upstream, calls, **relay_options) as root:
                async with httpx.AsyncClient(timeout=10) as client:
                    return await exercise(client, root, endpoint)
        finally:
            await runner.cleanup()

    return asyncio.run(run()), calls, endpoint


def _chat(client, root, **headers):
    return client.post(f"{root}/v1/chat/completions", json={"model": "m", "stream": True}, headers=headers)


def test_a_streamed_answer_is_forwarded_as_sent_and_counted_from_its_usage():
    async def exercise(client, root, endpoint):
        return (await _chat(client, root, authorization="Bearer sk-cli")).text

    text, calls, endpoint = _through_relay([(200, CLAUDE_STREAM, "text/event-stream")], exercise)

    assert text == CLAUDE_STREAM
    assert endpoint.requests[0]["path"] == "/v1/chat/completions"
    assert endpoint.requests[0]["headers"]["authorization"] == "Bearer sk-cli"
    assert endpoint.requests[0]["headers"]["accept-encoding"] == "identity"
    (usage,) = calls.answered
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens,
            usage.input_includes_cache) == (20948, 43, 10363, 199, True)


def test_each_shape_of_answer_is_read():
    responses = ('data: {"type": "response.completed", "response": {"usage": {"input_tokens": 900, '
                 '"input_tokens_details": {"cached_tokens": 512}, "output_tokens": 7}}}\n\n')
    messages = ('data: {"type": "message_start", "message": {"usage": {"input_tokens": 3, '
                '"cache_read_input_tokens": 800, "cache_creation_input_tokens": 0, "output_tokens": 1}}}\n\n'
                'data: {"type": "message_delta", "usage": {"output_tokens": 12}}\n\n')
    body = json.dumps({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}})

    async def exercise(client, root, endpoint):
        await client.post(f"{root}/v1/responses", json={})
        await client.post(f"{root}/v1/messages", json={})
        await client.post(f"{root}/v1/chat/completions", json={})

    _, calls, _ = _through_relay([(200, responses, "text/event-stream"), (200, messages, "text/event-stream"),
                                  (200, body, "application/json")], exercise)

    assert [(usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) for usage in calls.answered] == [
        (900, 7, 512), (803, 12, 800), (50, 5, None)]


def test_a_refused_call_counts_nothing_and_one_answered_without_usage_is_unknown():
    no_usage = 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
    zeros = json.dumps({"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    async def exercise(client, root, endpoint):
        return [(await _chat(client, root)).status_code for _ in range(3)]

    statuses, calls, _ = _through_relay([(429, '{"error": "slow down"}', "application/json"),
                                         (200, no_usage, "text/event-stream"), (200, zeros, "application/json")],
                                        exercise)

    assert statuses == [429, 200, 200]
    assert (calls.answered, calls.unreported, calls.refused) == ([], 2, 1)
    assert calls.usage(3).to_dict() == {"tool_call_count": 3, "provider_details": {"model_calls": 2,
                                                                                    "unreported_calls": 2}}


def test_a_call_whose_cli_hung_up_is_read_to_its_end_and_counted():
    async def exercise(client, root, endpoint):
        endpoint.hold = asyncio.Event()
        async with client.stream("POST", f"{root}/v1/chat/completions", json={}) as answer:
            async for _ in answer.aiter_bytes():
                break
        endpoint.hold.set()

    _, calls, _ = _through_relay([(200, CLAUDE_STREAM, "text/event-stream")], exercise)

    assert [usage.input_tokens for usage in calls.answered] == [20948]


def test_a_call_whose_cli_hung_up_before_the_answer_began_is_still_counted():
    async def exercise(client, root, endpoint):
        endpoint.hold_headers = asyncio.Event()
        with pytest.raises(httpx.ReadTimeout):
            await client.post(f"{root}/v1/chat/completions", json={}, timeout=0.3)
        endpoint.hold_headers.set()
        await asyncio.sleep(0.3)

    _, calls, _ = _through_relay([(200, CLAUDE_STREAM, "text/event-stream")], exercise)

    assert [usage.input_tokens for usage in calls.answered] == [20948]


def test_a_long_streams_usage_is_read_however_long_it_ran():
    """The usage comes last, so it's read as the stream passes, not from a kept prefix of it."""
    chunk = 'data: {"choices": [{"delta": {"content": "%s"}}]}\n\n' % ("x" * 4000)
    reader = model_relay.AnswerReader(stream=True)
    for _ in range(20_000):
        reader.feed(chunk.encode())
    reader.feed(CLAUDE_STREAM.encode())

    assert reader.usage().input_tokens == 20948


def test_a_streams_usage_split_across_reads_is_read_whole():
    """Read 7 bytes at a time, the first read ``data: {``, and 1 byte at a time, ``d`` first."""
    encoded = CLAUDE_STREAM.encode()
    for size in (7, 1):
        reader = model_relay.AnswerReader(stream=True)
        for start in range(0, len(encoded), size):
            reader.feed(encoded[start:start + size])

        assert (reader.usage().input_tokens, reader.usage().cache_write_tokens) == (20948, 199)


def test_a_relayed_stream_written_a_byte_at_a_time_is_counted():
    """The relay judges an answer by its content type, so a stream whose first read is ``d`` is still a stream."""
    async def exercise(client, root, endpoint):
        endpoint.write_bytes = 1
        return (await _chat(client, root)).text

    text, calls, _ = _through_relay([(200, CLAUDE_STREAM, "text/event-stream")], exercise)

    assert text == CLAUDE_STREAM
    assert [usage.input_tokens for usage in calls.answered] == [20948]


def test_another_turns_path_reaches_nothing_and_other_routes_are_not_calls():
    async def exercise(client, root, endpoint):
        stale = await client.post(root.rsplit("/", 1)[0] + "/0123/v1/chat/completions", json={})
        models = await client.get(f"{root}/v1/models")
        return stale.status_code, models.json()

    (stale, models), calls, endpoint = _through_relay([], exercise, models=["m"])

    assert stale == 404 and models == {"object": "list", "data": [{"id": "m", "object": "model"}]}
    assert endpoint.requests == [] and (calls.answered, calls.unreported, calls.refused) == ([], 0, 0)


def test_the_relays_headers_replace_the_clis():
    async def exercise(client, root, endpoint):
        await _chat(client, root, authorization="Bearer placeholder", **{"x-cli": "1"})

    _, _, endpoint = _through_relay([(200, CLAUDE_STREAM, "text/event-stream")], exercise,
                                    headers={"authorization": "Bearer sk-relay", "x-litellm-tags": "taskId:t1"})

    sent = endpoint.requests[0]["headers"]
    assert (sent["authorization"], sent["x-litellm-tags"], "x-cli" in sent) == ("Bearer sk-relay", "taskId:t1", False)
    assert sent["content-type"] == "application/json"


def test_an_unreachable_endpoint_is_a_502_and_counts_nothing():
    calls = model_relay.Calls()

    async def run():
        async with model_relay.model_relay("http://127.0.0.1:9", calls) as root:
            async with httpx.AsyncClient(timeout=10) as client:
                return (await _chat(client, root)).status_code

    assert asyncio.run(run()) == 502
    assert (calls.answered, calls.unreported, calls.refused) == ([], 0, 1)


def test_the_calls_usage_sums_them_with_the_cache_in_the_input():
    calls = model_relay.Calls()
    calls.add(200, model_relay.answer_usage(CLAUDE_STREAM.encode()))
    calls.add(200, model_relay.answer_usage(json.dumps({"usage": {"prompt_tokens": 1000, "completion_tokens": 10,
                                                                  "prompt_tokens_details": {"cached_tokens": 900}}})
                                            .encode()))
    calls.add(200, model_relay.answer_usage(b'data: {"choices": []}\n\n'))

    assert calls.usage(2).to_dict() == {
        "tool_call_count": 2, "input_tokens": 21948, "output_tokens": 53, "total_tokens": 22001,
        "cache_read_tokens": 11263, "cache_write_tokens": 199, "input_includes_cache": True,
        "provider_details": {"model_calls": 3, "unreported_calls": 1}}
