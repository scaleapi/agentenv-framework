"""The key holder (utils/key_holder/key_holder.py), imported by its own name as the agent images copy it: it forwards
only model calls that carry a context's token, with the sealed key and headers and the context's attribution in place
of the client's, pinned to the context's models, and scrubs every sealed value from what comes back. Every case runs
against a stub endpoint that records what reaches it, with a sentinel key."""

from __future__ import annotations

import asyncio
import gzip
import importlib
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest

import agent_env

sys.path.insert(0, str(Path(agent_env.__file__).parent / "utils" / "key_holder"))
key_holder = importlib.import_module("key_holder")

KEY = "sk-sentinel-key-holder-0123456789"
BILL_TO = "team-alpha-bill-to-secret"
SHORT = "abc123"
UPSTREAM = "https://litellm.test"
MODEL, OTHER = "anthropic/claude-run", "anthropic/claude-other"
TOKEN = key_holder.TOKEN_PREFIX + "context-one-token"
SECOND = key_holder.TOKEN_PREFIX + "context-two-token"


@dataclass
class Answer:
    status: int
    headers: dict[str, str]
    chunks: list[bytes] = field(default_factory=list)

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)

    def json(self) -> dict:
        return json.loads(self.body)


class Upstream:
    """The stub endpoint: records each request and answers with ``reply`` (a response, or a callable making one)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply = lambda request: httpx.Response(200, json={"ok": True})

    async def handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        reply = self.reply(request)
        return await reply if asyncio.iscoroutine(reply) else reply

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self) -> dict:
        return json.loads(self.last.content)


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


def _holder(upstream: Upstream, *, headers: dict[str, str] | None = None, models: tuple[str, ...] = (MODEL,),
            project_id: str | None = "p-1", task_id: str | None = "t-1", **kwargs) -> key_holder.KeyHolder:
    route = key_holder.ModelRoute(UPSTREAM, KEY, {"X-Bill-To": BILL_TO, "X-Short": SHORT} if headers is None
                                  else headers)
    tokens = key_holder.Tokens()
    tokens.add(TOKEN, key_holder.Context("ctx-1", models, project_id, task_id))
    return key_holder.KeyHolder(key_holder.Sealed({key_holder.DEFAULT_ROUTE: route}), tokens,
                                transport=httpx.MockTransport(upstream.handle), **kwargs)


@pytest.fixture
def holder(upstream: Upstream) -> key_holder.KeyHolder:
    return _holder(upstream)


async def call(holder, method: str, target: str, *, headers: dict[str, str] | None = None, body: bytes | dict = b"",
               token: str | None = TOKEN, on_chunk=None) -> Answer:
    """``target`` (a raw path and query) sent to the key holder's ASGI app as uvicorn would, the path decoded;
    each body chunk the app sends is kept apart, and ``on_chunk`` sees each as it is sent."""
    raw, _, query = target.partition("?")
    content = json.dumps(body).encode() if isinstance(body, dict) else body
    sent_headers = {"content-type": "application/json", **({"authorization": f"Bearer {token}"} if token else {}),
                    **(headers or {})}
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
        "path": unquote(raw), "raw_path": raw.encode(), "query_string": query.encode(), "root_path": "",
        "headers": [(name.lower().encode(), value.encode()) for name, value in sent_headers.items()],
        "client": ("127.0.0.1", 40000), "server": ("127.0.0.1", key_holder.PORT),
    }
    done, delivered = asyncio.Event(), False
    answer = Answer(0, {})

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": content, "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            answer.status = message["status"]
            answer.headers = {name.decode(): value.decode() for name, value in message["headers"]}
        elif message["type"] == "http.response.body":
            if message.get("body"):
                answer.chunks.append(message["body"])
                if on_chunk:
                    await on_chunk(message["body"])
            if not message.get("more_body"):
                done.set()

    await holder.app(scope, receive, send)
    return answer


def _messages(model: str = MODEL, **extra) -> dict:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], **extra}


# Routing


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, kind", [
    ("POST", "/v1/messages", "anthropic"),
    ("POST", "/v1/messages/count_tokens", "anthropic"),
    ("POST", "/v1/responses", "openai"),
    ("POST", "/openai/v1/responses", "openai"),
    ("POST", "/v1/chat/completions", "openai"),
    ("GET", "/v1/responses/resp_abc-1.2=", "openai_object"),
    ("POST", "/openai/v1/responses/resp_abc/cancel", "openai_object"),
    ("GET", "/v1/models", "models"),
    ("POST", "/v1beta/models/gemini-2.5-pro:streamGenerateContent", "gemini"),
    ("GET", "/gemini/v1beta/models", "models"),
])
async def test_the_model_routes_are_forwarded(method, path, kind):
    assert key_holder.model_call_kind(method, path) == kind


@pytest.mark.asyncio
@pytest.mark.parametrize("method, target", [
    ("GET", "/key/info"), ("POST", "/key/generate"), ("POST", "/v1/batches"), ("GET", "/spend/logs"),
    ("POST", "/v1/messages/batches"), ("DELETE", "/v1/responses/resp_abc"), ("PUT", "/v1/messages"),
    ("HEAD", "/v1/models"), ("POST", "/v1/responses/../batches"), ("POST", "/v1/responses/%2e%2e/batches"),
    ("POST", "/openai/v1/responses/../batches"), ("POST", "/./v1/messages"), ("POST", "//chat/completions"),
    ("POST", "/v1//messages"), ("POST", "/v1/messages/"), ("POST", "/v1/messages%3Fx=1"),
    ("POST", "/v1/messages%23frag"), ("POST", "/v1/responses/resp%252e%252e"), ("POST", "/v1\\messages"),
    ("POST", "/v1beta/models/gem%20ini:generateContent"),
])
async def test_anything_else_is_refused_before_it_reaches_the_endpoint(holder, upstream, method, target):
    answer = await call(holder, method, target, body=_messages())

    assert answer.status == 403
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_a_percent_encoded_query_or_fragment_never_opens_one_upstream(holder, upstream):
    for target in ("/v1/messages%3Fkey=other", "/v1/messages%23x", "/v1/responses/resp_a%3F/x"):
        assert (await call(holder, "POST", target, body=_messages())).status == 403
    answer = await call(holder, "POST", "/v1/messages?beta=true&key=client", body=_messages())

    assert answer.status == 200
    assert str(upstream.last.url) == f"{UPSTREAM}/v1/messages?beta=true"


@pytest.mark.parametrize("base", ["https://litellm.test", "https://litellm.test/", "https://litellm.test/v1",
                                  "https://litellm.test/v1/", " https://litellm.test/v1 "])
def test_the_endpoint_is_sealed_without_its_own_v1_so_calls_name_one(base):
    sealed = key_holder.sealed_from_env({"LITELLM_API_KEY": KEY, "LITELLM_BASE_URL": base})

    assert sealed.model_routes["default"].upstream == "https://litellm.test"


@pytest.mark.asyncio
@pytest.mark.parametrize("base, url", [
    ("https://litellm.test/v1", "https://litellm.test/v1/messages"),
    ("https://proxy.test/prefix/", "https://proxy.test/prefix/v1/messages"),
    ("https://proxy.test/x?team=a", "https://proxy.test/x/v1/messages?team=a"),
])
async def test_calls_go_to_the_sealed_endpoint_with_one_v1(upstream, base, url):
    sealed = key_holder.sealed_from_env({"LITELLM_API_KEY": KEY, "LITELLM_BASE_URL": base})
    tokens = key_holder.Tokens()
    tokens.add(TOKEN, key_holder.Context("ctx-1", (MODEL,)))
    holder = key_holder.KeyHolder(sealed, tokens, transport=httpx.MockTransport(upstream.handle))

    assert (await call(holder, "POST", "/v1/messages", body=_messages())).status == 200
    assert str(upstream.last.url) == url


# The sealed route table


def test_a_version_1_file_loads():
    sealed = key_holder.Sealed({"default": key_holder.ModelRoute(UPSTREAM, KEY, {"X-Bill-To": BILL_TO})})

    assert key_holder.Sealed.from_json(sealed.to_json()) == sealed
    assert json.loads(sealed.to_json())["upstream_headers"] == {}


@pytest.mark.parametrize("raw", [
    json.dumps({"version": 2, "model_routes": {}}), json.dumps({"model_routes": {}}), "[]", "not json",
    json.dumps({"version": 1, "model_routes": {"default": {"upstream": UPSTREAM}}}),
    json.dumps({"version": 1, "model_routes": {"default": {"upstream": "file:///etc/passwd", "key": KEY}}}),
    json.dumps({"version": 1, "model_routes": {"default": {"upstream": "https://u:p@litellm.test", "key": KEY}}}),
    json.dumps({"version": 1, "model_routes": {"default": {"upstream": UPSTREAM, "key": KEY, "headers": {"a": 1}}}}),
])
def test_a_file_it_doesnt_know_refuses_to_load(raw):
    with pytest.raises(key_holder.SealError):
        key_holder.Sealed.from_json(raw)


@pytest.mark.asyncio
async def test_without_a_model_route_every_model_call_answers_503(upstream):
    tokens = key_holder.Tokens()
    tokens.add(TOKEN, key_holder.Context("ctx-1", (MODEL,)))
    holder = key_holder.KeyHolder(key_holder.Sealed({}), tokens, transport=httpx.MockTransport(upstream.handle))

    for method, target in (("POST", "/v1/messages"), ("GET", "/v1/models"), ("POST", "/v1/responses")):
        answer = await call(holder, method, target, body=_messages())
        assert (answer.status, answer.json()["error"]["message"]) == (503, "this box has no model endpoint")
    assert upstream.requests == []


# Tokens and pinning


@pytest.mark.asyncio
@pytest.mark.parametrize("headers, token", [
    ({}, None),
    ({}, key_holder.TOKEN_PREFIX + "unknown"),
    ({}, KEY),
    ({"x-api-key": "anything"}, None),
])
async def test_a_call_without_a_token_the_key_holder_holds_is_refused(holder, upstream, headers, token):
    answer = await call(holder, "POST", "/v1/messages", headers=headers, body=_messages(), token=token)

    assert answer.status == 401
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{"x-api-key": TOKEN}, {"x-goog-api-key": TOKEN},
                                     {"authorization": f"bearer {TOKEN}"}])
async def test_the_token_may_come_as_an_api_key_too(holder, headers):
    answer = await call(holder, "POST", "/v1/messages", headers=headers, body=_messages(), token=None)

    assert answer.status == 200


@pytest.mark.asyncio
async def test_a_dropped_token_stops_working_and_another_contexts_keeps_its_own(holder):
    holder.tokens.add(SECOND, key_holder.Context("ctx-2", (OTHER,)))
    holder.tokens.drop(TOKEN)

    assert (await call(holder, "POST", "/v1/messages", body=_messages())).status == 401
    assert (await call(holder, "POST", "/v1/messages", body=_messages(OTHER), token=SECOND)).status == 200
    holder.tokens.drop_all()
    assert (await call(holder, "POST", "/v1/messages", body=_messages(OTHER), token=SECOND)).status == 401


@pytest.mark.asyncio
async def test_two_contexts_tokens_on_one_call_are_refused(holder):
    holder.tokens.add(SECOND, key_holder.Context("ctx-2", (OTHER,)))

    answer = await call(holder, "POST", "/v1/messages", headers={"x-api-key": SECOND}, body=_messages())

    assert answer.status == 401


def test_a_token_is_a_context_token():
    with pytest.raises(ValueError):
        key_holder.Tokens().add("sk-not-a-context-token", key_holder.Context("ctx", (MODEL,)))
    assert key_holder.new_token().startswith(key_holder.TOKEN_PREFIX)
    assert key_holder.new_token() != key_holder.new_token()


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/json", "text/plain", None,
                                          "application/x-www-form-urlencoded"])
async def test_another_model_is_pinned_to_the_contexts_whatever_the_content_type(holder, upstream, content_type):
    headers = {"content-type": content_type} if content_type else {"content-type": ""}

    answer = await call(holder, "POST", "/v1/messages", headers=headers, body=_messages(OTHER))

    assert answer.status == 200
    assert upstream.body()["model"] == MODEL
    assert upstream.last.headers["content-type"] == "application/json"
    assert holder.stats["pinned"] == 1


@pytest.mark.asyncio
async def test_a_call_naming_no_model_gets_the_contexts(holder, upstream):
    body = _messages()
    del body["model"]

    assert (await call(holder, "POST", "/v1/responses", body=body)).status == 200
    assert upstream.body()["model"] == MODEL


@pytest.mark.asyncio
async def test_a_model_the_context_names_goes_unchanged(upstream):
    holder = _holder(upstream, models=(MODEL, OTHER))
    raw = json.dumps(_messages(OTHER), indent=1).encode()

    assert (await call(holder, "POST", "/v1/messages", body=raw)).status == 200
    assert upstream.last.content == raw
    assert holder.stats["pinned"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path, pinned", [
    ("/v1beta/models/gemini-other:generateContent", "/v1beta/models/gemini-run:generateContent"),
    ("/gemini/v1beta/models/gemini-other:streamGenerateContent", "/gemini/v1beta/models/gemini-run:"
                                                                 "streamGenerateContent"),
    ("/v1beta/models/gemini-run:countTokens", "/v1beta/models/gemini-run:countTokens"),
])
async def test_a_gemini_path_is_pinned_to_the_contexts_model(upstream, path, pinned):
    holder = _holder(upstream, models=("gemini/gemini-run",))

    answer = await call(holder, "POST", path + "?alt=sse", body={"contents": []})

    assert answer.status == 200
    assert upstream.last.url.path == pinned
    assert upstream.last.url.params["alt"] == "sse"
    assert upstream.last.headers["x-goog-api-key"] == KEY


@pytest.mark.asyncio
async def test_with_no_model_a_post_answers_503_and_nothing_goes_upstream(upstream):
    holder = _holder(upstream, models=())

    answer = await call(holder, "POST", "/v1/messages", body=_messages())

    assert (answer.status, answer.json()["error"]["message"]) == (503, "this run names no model")
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_the_model_list_is_the_contexts_alone(upstream):
    holder = _holder(upstream, models=(MODEL, "gemini/gemini-run"))

    openai = await call(holder, "GET", "/v1/models")
    gemini = await call(holder, "GET", "/v1beta/models")

    assert [model["id"] for model in openai.json()["data"]] == [MODEL, "gemini/gemini-run"]
    assert [model["name"] for model in gemini.json()["models"]] == ["models/claude-run", "models/gemini-run"]
    assert upstream.requests == []


# Bodies and headers


@pytest.mark.asyncio
async def test_a_compressed_body_is_refused(holder, upstream):
    answer = await call(holder, "POST", "/v1/messages", headers={"content-encoding": "gzip"},
                        body=gzip.compress(json.dumps(_messages(OTHER)).encode()))

    assert answer.status == 415
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"[]", b'"text"', b"not json", b"", b"\xff\xfe", json.dumps(_messages(7)).encode(),
                                  json.dumps(_messages(None) | {"model": ["a"]}).encode()])
async def test_a_body_that_isnt_an_object_naming_its_model_as_a_string_is_refused(holder, upstream, body):
    answer = await call(holder, "POST", "/v1/messages", body=body)

    assert answer.status == 400
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("param", ["api_key", "api_base", "base_url"])
async def test_a_body_naming_its_own_endpoint_or_key_is_refused(holder, upstream, param):
    answer = await call(holder, "POST", "/v1/chat/completions", body=_messages(**{param: "https://elsewhere"}))

    assert answer.status == 400
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_a_body_over_the_cap_is_refused(upstream):
    holder = _holder(upstream, max_body_bytes=1024)

    small = await call(holder, "POST", "/v1/messages", body=_messages())
    large = await call(holder, "POST", "/v1/messages", body=_messages(padding="x" * 2048))

    assert (small.status, large.status) == (200, 413)
    assert len(upstream.requests) == 1


@pytest.mark.asyncio
async def test_the_clients_credentials_and_attribution_are_replaced_by_the_sealed_ones(holder, upstream):
    client = {"x-api-key": TOKEN, "x-goog-api-key": "client-goog", "api-key": "client-azure",
              "x-litellm-api-key": "client-litellm", "proxy-authorization": "Basic Y2xpZW50",
              "x-litellm-customer-id": "client-project", "x-litellm-tags": "projectId:client",
              "x-http-method-override": "DELETE", "x-bill-to": "client-bill", "anthropic-beta": "beta-1,beta-2",
              "anthropic-version": "2023-06-01", "user-agent": "claude-cli/2"}

    await call(holder, "POST", "/v1/messages?key=client-query", headers=client, body=_messages())

    sent = upstream.last.headers
    assert sent["authorization"] == f"Bearer {KEY}"
    assert sent["x-api-key"] == KEY
    assert sent.get_list("x-bill-to") == [BILL_TO]
    assert sent["x-short"] == SHORT
    assert sent["x-litellm-customer-id"] == "p-1"
    assert sent["x-litellm-tags"] == "projectId:p-1,taskId:t-1"
    assert (sent["anthropic-beta"], sent["anthropic-version"], sent["user-agent"]) == ("beta-1,beta-2", "2023-06-01",
                                                                                       "claude-cli/2")
    for name in ("x-goog-api-key", "api-key", "x-litellm-api-key", "proxy-authorization", "x-http-method-override"):
        assert name not in sent
    assert "key" not in upstream.last.url.params
    assert TOKEN not in str(sent.raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured, expected", [
    (None, "projectId:p-1,taskId:t-1"),
    ("useCase:evals", "useCase:evals,projectId:p-1,taskId:t-1"),
    ("taskId:own, useCase:evals", "taskId:own,useCase:evals,projectId:p-1"),
    ("projectId:mine,taskId:mine", "projectId:mine,taskId:mine"),
    (" , useCase:evals,", "useCase:evals,projectId:p-1,taskId:t-1"),
])
async def test_the_runs_tags_join_the_configured_ones_after_them(upstream, configured, expected):
    holder = _holder(upstream, headers={"X-LiteLLM-Tags": configured} if configured else {})

    await call(holder, "POST", "/v1/messages", headers={"x-litellm-tags": "projectId:forged"}, body=_messages())

    assert upstream.last.headers.get_list("x-litellm-tags") == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("project_id, task_id, expected", [
    (None, "t-1", {"x-litellm-tags": "taskId:t-1"}),
    (None, None, {}),
])
async def test_attribution_follows_what_the_context_names(upstream, project_id, task_id, expected):
    holder = _holder(upstream, headers={}, project_id=project_id, task_id=task_id)

    await call(holder, "POST", "/v1/messages", headers={"x-litellm-customer-id": "forged"}, body=_messages())

    sent = upstream.last.headers
    assert {name: sent[name] for name in ("x-litellm-customer-id", "x-litellm-tags") if name in sent} == expected


@pytest.mark.asyncio
async def test_a_redirect_is_relayed_not_followed(holder, upstream):
    upstream.reply = lambda request: httpx.Response(302, headers={"location": "https://elsewhere.test/steal"})

    answer = await call(holder, "POST", "/v1/messages", body=_messages())

    assert (answer.status, answer.headers["location"]) == (302, "https://elsewhere.test/steal")
    assert len(upstream.requests) == 1


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_answers_502_naming_only_the_error_type(holder, upstream):
    def unreachable(request):
        raise httpx.ConnectError(f"can't reach {KEY}")
    upstream.reply = unreachable

    answer = await call(holder, "POST", "/v1/messages", body=_messages())

    assert answer.status == 502
    assert answer.json()["error"]["message"] == "the model endpoint is unreachable: ConnectError"


@pytest.mark.asyncio
async def test_a_stream_passes_chunk_by_chunk(holder, upstream):
    release = asyncio.Event()
    first = b"event: message_start\ndata: {}\n\n"

    async def stream():
        yield first
        await release.wait()
        yield b"event: message_stop\ndata: {}\n\n"

    upstream.reply = lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"},
                                                    content=stream())
    seen = []

    async def on_chunk(chunk):
        seen.append(chunk)
        release.set()

    answer = await asyncio.wait_for(call(holder, "POST", "/v1/messages", body=_messages(stream=True),
                                         on_chunk=on_chunk), timeout=5)

    assert seen[0] == first
    assert answer.chunks == [first, b"event: message_stop\ndata: {}\n\n"]
    assert answer.headers["content-type"] == "text/event-stream"


@pytest.mark.asyncio
async def test_a_compressed_answer_is_decoded_so_its_scrubbed(holder, upstream):
    upstream.reply = lambda request: httpx.Response(401, headers={"content-encoding": "gzip"},
                                                    content=gzip.compress(f"bad key {KEY}".encode()))

    answer = await call(holder, "POST", "/v1/messages", body=_messages())

    assert answer.body == b"bad key " + key_holder.REDACTED
    assert "content-encoding" not in answer.headers


@pytest.mark.asyncio
async def test_an_answer_in_an_encoding_it_cant_decode_is_refused(holder, upstream):
    upstream.reply = lambda request: httpx.Response(200, headers={"content-encoding": "br"}, content=b"\x00\x01")

    assert (await call(holder, "POST", "/v1/messages", body=_messages())).status == 502


# Tools are left alone (N3)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [
    {"web_search_options": {"search_context_size": "low"}},
    {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
    {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
    {"tools": [{"name": "WebSearch", "input_schema": {"type": "object"}}]},
    {"tools": [{"type": "web_search"}]},
])
async def test_web_tools_reach_the_endpoint_unchanged(holder, upstream, extra):
    raw = json.dumps(_messages(**extra)).encode()

    assert (await call(holder, "POST", "/v1/messages", body=raw)).status == 200
    assert upstream.last.content == raw


# Scrubbing


@pytest.mark.parametrize("parts", [2, 3])
@pytest.mark.parametrize("secret", [KEY, BILL_TO, json.dumps('quo"te-secret-value')[1:-1]])
def test_a_sealed_value_split_across_chunks_is_scrubbed(parts, secret):
    values = key_holder._scrub_forms([KEY, BILL_TO, SHORT, 'quo"te-secret-value'])
    text = f'{{"error": "{secret}", "x": "{SHORT}"}}'.encode()
    start = text.index(secret.encode())
    cuts = [start + 1 + index * (len(secret) // parts) for index in range(parts - 1)]
    chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
    scrubber = key_holder._Scrubber(values)

    out = b"".join([*(scrubber.feed(chunk) for chunk in chunks), scrubber.flush()])

    assert out == text.replace(secret.encode(), key_holder.REDACTED)
    assert SHORT.encode() in out


@pytest.mark.asyncio
async def test_no_sealed_value_comes_back_in_an_error_body_header_or_the_log(holder, upstream, capsys):
    upstream.reply = lambda request: httpx.Response(
        401, headers={"x-echo": f"{KEY} {BILL_TO}"},
        json={"error": {"message": f"invalid key {KEY} for {BILL_TO}", "short": SHORT}})
    capsys.readouterr()

    answer = await call(holder, "POST", "/v1/messages", body=_messages(OTHER))
    for status in (500, 429):
        upstream.reply = lambda request, status=status: httpx.Response(status, text=f"{KEY}{BILL_TO}")
        await call(holder, "POST", "/v1/messages", body=_messages())

    log = capsys.readouterr().out
    assert answer.status == 401
    for secret in (KEY, BILL_TO, TOKEN):
        assert secret not in answer.body.decode() and secret not in json.dumps(answer.headers) and secret not in log
    assert SHORT in answer.body.decode()
    lines = [json.loads(line) for line in log.splitlines()]
    assert [(line["status"], line["model"], line.get("pinned_to")) for line in lines] == [
        (401, OTHER, MODEL), (500, MODEL, None), (429, MODEL, None)]
    assert all(line["context"] == key_holder._digest("ctx-1") for line in lines)


# Sealing


@pytest.fixture
def sealed_dir(tmp_path) -> Path:
    directory = tmp_path / "agentenv-keyholder"
    directory.mkdir(mode=0o700)
    return directory


def test_seal_writes_a_file_only_the_key_holder_reads(sealed_dir, monkeypatch):
    chowned = []
    monkeypatch.setattr(key_holder.os, "fchown", lambda fd, uid, gid: chowned.append((uid, gid)))
    sealed = key_holder.sealed_from_env({"LITELLM_API_KEY": KEY, "LITELLM_BASE_URL": UPSTREAM + "/v1",
                                         "LITELLM_EXTRA_HEADERS": json.dumps({"X-Bill-To": BILL_TO})})

    path = key_holder.write_sealed(sealed, sealed_dir, uid=os.getuid(), gid=os.getgid())

    assert chowned == [(os.getuid(), os.getgid())]
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert sorted(child.name for child in sealed_dir.iterdir()) == ["sealed.json"]
    assert key_holder.load_sealed(path) == sealed
    assert sealed.model_routes["default"] == key_holder.ModelRoute(UPSTREAM, KEY, {"X-Bill-To": BILL_TO})


@pytest.mark.parametrize("environ, route", [
    ({"LITELLM_API_KEY": KEY}, None),
    ({"LITELLM_BASE_URL": UPSTREAM}, None),
    ({}, None),
    ({"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": UPSTREAM + "/v1"}, key_holder.ModelRoute(UPSTREAM, KEY)),
    ({"LITELLM_API_KEY": KEY, "LITELLM_BASE_URL": UPSTREAM, "LITELLM_EXTRA_HEADERS": "not json"},
     key_holder.ModelRoute(UPSTREAM, KEY)),
    ({"LITELLM_API_KEY": "sail-injected-model-key", "LITELLM_BASE_URL": UPSTREAM},
     key_holder.ModelRoute(UPSTREAM, "sail-injected-model-key")),
])
def test_seal_takes_the_endpoint_from_its_environment(environ, route):
    assert key_holder.sealed_from_env(environ).model_routes.get("default") == route


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o701])
def test_seal_refuses_a_directory_anyone_else_can_enter(sealed_dir, mode):
    sealed_dir.chmod(mode)

    with pytest.raises(key_holder.SealError):
        key_holder.write_sealed(key_holder.Sealed({}), sealed_dir, uid=os.getuid(), gid=os.getgid())


def test_seal_refuses_a_directory_another_user_owns(sealed_dir):
    with pytest.raises(key_holder.SealError):
        key_holder.write_sealed(key_holder.Sealed({}), sealed_dir, uid=os.getuid() + 1, gid=os.getgid())


@pytest.mark.parametrize("mode", [0o444, 0o440, 0o404, 0o600, 0o640])
def test_serve_refuses_a_file_that_isnt_0400(sealed_dir, mode):
    path = key_holder.write_sealed(key_holder.Sealed({}), sealed_dir, uid=os.getuid(), gid=os.getgid())
    path.chmod(mode)

    with pytest.raises(key_holder.SealError):
        key_holder.load_sealed(path)


def test_serve_refuses_a_file_another_user_owns(sealed_dir, monkeypatch):
    path = key_holder.write_sealed(key_holder.Sealed({}), sealed_dir, uid=os.getuid(), gid=os.getgid())
    monkeypatch.setattr(key_holder.os, "geteuid", lambda: os.getuid() + 1)

    with pytest.raises(key_holder.SealError):
        key_holder.load_sealed(path)


def test_serve_refuses_a_link_to_the_file(sealed_dir, tmp_path):
    path = key_holder.write_sealed(key_holder.Sealed({}), sealed_dir, uid=os.getuid(), gid=os.getgid())
    link = tmp_path / "link.json"
    link.symlink_to(path)

    with pytest.raises(key_holder.SealError):
        key_holder.load_sealed(link)


@pytest.fixture
def as_seal(sealed_dir, monkeypatch):
    monkeypatch.setattr(key_holder, "SEALED_DIR", sealed_dir)
    monkeypatch.setattr(key_holder, "SEALED_FILE", sealed_dir / "sealed.json")
    monkeypatch.setattr(key_holder.pwd, "getpwnam",
                        lambda name: SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid()))
    for name in (*key_holder.KEY_NAMES, *key_holder.BASE_URL_NAMES, key_holder.MODEL_HEADERS_ENV):
        monkeypatch.delenv(name, raising=False)
    return sealed_dir


def test_the_seal_command_reads_its_environment_never_its_arguments(as_seal, monkeypatch, capsys):
    monkeypatch.setenv("LITELLM_API_KEY", KEY)
    monkeypatch.setenv("LITELLM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps({"X-Bill-To": BILL_TO}))

    with pytest.raises(SystemExit):
        key_holder.main(["seal", "sk-on-argv"])
    assert list(as_seal.iterdir()) == []
    assert key_holder.main(["seal"]) == 0

    assert key_holder.load_sealed(as_seal / "sealed.json").model_routes["default"].key == KEY
    out = capsys.readouterr().out
    assert KEY not in out and BILL_TO not in out
    assert json.loads(out.splitlines()[-1])["routes"] == {"default": {"upstream": UPSTREAM, "headers": ["X-Bill-To"]}}


def test_the_seal_command_without_an_endpoint_seals_no_route(as_seal):
    assert key_holder.main(["seal"]) == 0

    assert key_holder.load_sealed(as_seal / "sealed.json").model_routes == {}


def test_the_serve_command_refuses_to_run_as_root(as_seal, monkeypatch):
    assert key_holder.main(["seal"]) == 0
    monkeypatch.setattr(key_holder.os, "geteuid", lambda: 0)

    assert key_holder.main(["serve"]) == 1


def test_the_serve_command_refuses_a_missing_file(as_seal, capsys):
    assert key_holder.main(["serve"]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "refused"


def test_the_port_is_the_one_clients_are_given():
    assert key_holder.PORT == 18431


@pytest.mark.asyncio
async def test_health_answers_without_a_token_or_naming_the_endpoint(holder):
    answer = await call(holder, "GET", "/healthz", token=None)

    assert (answer.status, answer.json()) == (200, {"status": "ok"})
