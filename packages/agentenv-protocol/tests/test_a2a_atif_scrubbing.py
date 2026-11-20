"""An agent's atif/1 reads leave out every secret its SDK holds or is handed, and every one it registers, each written
``[redacted:<length>]`` wherever it occurs: in the document, the aux log and the turn context, inline, uploaded and in
cursor pages. Native reads keep them, byte for byte."""
from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import quote

import pytest
from starlette.testclient import TestClient
from test_a2a_atif_serving import _ended, _send, _serving

from agentenv_protocol.a2a_agent import (
    CDP_CONFIG_V1,
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    TrajectoryState,
    Uploaded,
    WriteOnly,
    a2a_agent,
    enable,
    framework,
)
from agentenv_protocol.a2a_agent._atif_copy import AtifCopy, TurnStart
from agentenv_protocol.a2a_agent import _secrets
from agentenv_protocol.a2a_agent._secrets import Secrets
from agentenv_protocol.a2a_agent.extensions import ClockSyncTimeRequest
from agentenv_protocol.atif import FORMAT, Adapter, atif_adapter, aux, errors_in, native, validate
from agentenv_protocol.transfers import HttpGetGrant, HttpPostPolicyGrant, HttpPutGrant, ReadObject
from agentenv_protocol.types import CONTROL_TOKEN_ENV, CONTROL_TOKEN_HEADER

NATIVE = "echo-secrets/1"
SECRET = "sk-registered-0123456789"


@atif_adapter(native_format=NATIVE, version="1", input_includes_cache=False)
class Echoes(Adapter):
    @native(type="say")
    def say(self, record: dict[str, Any], turn: Any) -> None:
        turn.message(step="said", text=record["text"])
        turn.step(key="said", llm_call_count=1, model_name="toy-model")

    @aux("notes", type="note")
    def note(self, record: dict[str, Any], turn: Any) -> None:
        turn.message(step="said", text=record["text"])


class ToyConfig(AgentConfig):
    api_token: WriteOnly[str | None] = None
    model_params: WriteOnly[dict[str, Any] | None] = None


def _echoing(*, register: str | None = None, atif: bool = True) -> AgentEnvAgent:
    """An agent whose turn says its prompt, notes it in an aux record and answers it, so a secret in the prompt is in
    every string an atif/1 read serves: the user step, a message, the aux log, the final response, the turn context."""
    trajectory = enable(TRAJECTORY_V1, live=True, native=NATIVE, **({"atif": Echoes} if atif else {}))

    @a2a_agent(identity=AgentIdentity(name="echoes", description="test", version="1"), config=ToyConfig,
               extensions=(trajectory, MCP_CONFIG_V1, CDP_CONFIG_V1))
    class Echoing(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            if register is not None:
                self.register_secret(register)
            prompt = request.parts[0].text
            request.trajectory.set_format(NATIVE)
            request.trajectory.append({"type": "say", "text": prompt})
            if atif:
                request.trajectory.aux("notes", {"type": "note", "text": f"noted {prompt}"})
            return TaskResult.text(f"answer: {prompt}")

    return Echoing()


def _grant(name: str) -> dict[str, Any]:
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {"media_type": "application/json", "max_bytes": 1_000_000,
            "write": {"kind": "http-put", "url": f"https://objects.example.test/{name}", "expires_at": expires}}


def _reads(client: TestClient, task_id: str, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every atif/1 read of the task and its context: inline, uploaded and paged."""
    uploads: dict[str, bytes] = {}

    async def upload(target, body: bytes) -> Uploaded:
        uploads[target.write.url.rsplit("/", 1)[-1]] = body
        return Uploaded(size_bytes=len(body))

    monkeypatch.setattr(framework, "upload", upload)
    reads = [
        client.post("/ext/trajectory", json={"task_id": task_id, "format": FORMAT}).json(),
        client.post("/ext/trajectory", json={"context_id": "ctx-1", "format": FORMAT}).json(),
        client.post("/ext/trajectory", json={"task_id": task_id, "format": FORMAT, "after": 0}).json(),
        client.post("/ext/trajectory", json={"context_id": "ctx-1", "format": FORMAT, "after": 0}).json(),
    ]
    client.post("/ext/trajectory", json={"task_id": task_id, "format": FORMAT, "objects": {
        name: _grant(name) for name in ("trajectory", "aux", "turn_context")}})
    return [*reads, *(json.loads(body) for body in uploads.values())]


def _native(client: TestClient, task_id: str) -> Any:
    return client.post("/ext/trajectory", json={"task_id": task_id}).json()["trajectory"]


def _map_strings(value: Any, change: Callable[[str], str]) -> Any:
    """``value`` with ``change`` applied to every string it holds, mapping keys included."""
    if isinstance(value, str):
        return change(value)
    if isinstance(value, dict):
        return {_map_strings(key, change): _map_strings(item, change) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, change) for item in value]
    return value


def _strings(value: Any) -> list[str]:
    found: list[str] = []
    _map_strings(value, lambda text: found.append(text) or text)
    return found


def _turn(client: TestClient, prompt: str) -> str:
    task_id = _send(client, prompt, "m-1")
    _ended(client, task_id)
    return task_id


def _assert_redacted(reads: list[Any], secret: str) -> None:
    served = [s for read in reads for s in _strings(read)]
    assert not [s for s in served if secret in s]
    assert any(f"[redacted:{len(secret)}]" in s for s in served)


def test_a_secret_the_agent_registers_is_redacted_in_every_atif_read_and_kept_in_the_native_log(
        monkeypatch: pytest.MonkeyPatch) -> None:
    with _serving(_echoing(register=SECRET)) as client:
        task_id = _turn(client, f"the key is {SECRET}")
        reads = _reads(client, task_id, monkeypatch)
        native_read = _native(client, task_id)
    _assert_redacted(reads, SECRET)
    document = reads[0]["trajectory"]
    assert errors_in(validate(document)) == []
    assert [step["message"] for step in document["steps"]][0] == f"the key is [redacted:{len(SECRET)}]"
    assert reads[0]["aux"][0]["record"]["text"] == f"noted the key is [redacted:{len(SECRET)}]"
    assert native_read == [{"type": "say", "text": f"the key is {SECRET}"}]


def _model_key(monkeypatch: pytest.MonkeyPatch, client: Callable[[], TestClient]) -> tuple[TestClient, str]:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-model-key-0123456789")
    return client(), "sk-model-key-0123456789"


def _model_headers(monkeypatch: pytest.MonkeyPatch, client: Callable[[], TestClient]) -> tuple[TestClient, str]:
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps({"X-Gateway-Key": "gw-header-0123456789",
                                                            "Content-Type": "application/json"}))
    return client(), "gw-header-0123456789"


def _control_token(monkeypatch: pytest.MonkeyPatch, client: Callable[[], TestClient]) -> tuple[TestClient, str]:
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "control-token-0123456789")
    return client(headers={CONTROL_TOKEN_HEADER: "control-token-0123456789"}), "control-token-0123456789"


def _write_only_config(monkeypatch: pytest.MonkeyPatch, client: Callable[[], TestClient]) -> tuple[TestClient, str]:
    opened = client()
    assert opened.post("/ext/agent-config", json={"api_token": "config-token-0123456789"}).status_code == 200
    return opened, "config-token-0123456789"


def _mcp_headers(monkeypatch: pytest.MonkeyPatch, client: Callable[[], TestClient]) -> tuple[TestClient, str]:
    opened = client()
    added = opened.post("/ext/mcp-config", json={"url": "https://gateway.example.test/mcp", "name": "env",
                                                 "headers": {"Authorization": "Bearer mcp-token-0123456789"}})
    assert added.status_code == 200, added.text
    return opened, "Bearer mcp-token-0123456789"


def _cdp_url(monkeypatch: pytest.MonkeyPatch, client: Callable[[], TestClient]) -> tuple[TestClient, str]:
    opened = client()
    ws_url = "wss://gateway.example.test/browser/web/cdp/cdp-token-0123456789/devtools/browser/1"
    added = opened.post("/ext/cdp-config", json={"name": "web", "http_url": "https://gateway.example.test",
                                                 "ws_url": ws_url})
    assert added.status_code == 200, added.text
    return opened, "cdp-token-0123456789"


@pytest.mark.parametrize("handed", [_model_key, _model_headers, _control_token, _write_only_config, _mcp_headers,
                                    _cdp_url],
                         ids=["model-key", "model-headers", "control-token", "write-only-config", "mcp-headers",
                              "cdp-url"])
def test_a_secret_the_sdk_holds_or_is_handed_is_redacted(monkeypatch: pytest.MonkeyPatch, handed) -> None:
    app = None

    def client(**kwargs: Any) -> TestClient:
        nonlocal app
        agent = _echoing()
        opened = _serving(agent, **kwargs)
        client = opened.__enter__()
        app = client.app
        return client

    opened, secret = handed(monkeypatch, client)
    try:
        task_id = _turn(opened, f"echo: {secret}")
        reads = _reads(opened, task_id, monkeypatch)
        assert _native(opened, task_id) == [{"type": "say", "text": f"echo: {secret}"}]
    finally:
        opened.__exit__(None, None, None)
    _assert_redacted(reads, secret)


def test_a_presigned_grant_is_redacted_by_its_signature_wherever_it_is_printed() -> None:
    secrets = Secrets()
    url = ("https://bucket.example.test/runs/1/trajectory.json?X-Amz-Algorithm=AWS4-HMAC-SHA256"
           "&X-Amz-Date=20261119T204500Z&X-Amz-Signature=abcdef0123456789&X-Amz-Expires=900")
    secrets.add_request(ReadObject(media_type="application/json", max_bytes=10, read=HttpGetGrant(
        kind="http-get", url=url, expires_at=datetime.now(timezone.utc) + timedelta(hours=1))))
    target = url.removeprefix("https://bucket.example.test")
    plain = "signed with AWS4-HMAC-SHA256 at 20261119T204500Z"
    printed = [url, f"GET {target} HTTP/1.1", "sig abcdef0123456789", plain, f"wrapped {quote(url, safe='')}"]
    assert secrets.scrub(printed) == [text.replace("abcdef0123456789", "[redacted:16]") for text in printed]


def test_a_tokened_url_is_redacted_by_its_token_and_plain_paths_and_values_are_left_unwarned(
        caplog: pytest.LogCaptureFixture) -> None:
    secrets = Secrets()
    token = "eyJvcCI6ImdldCJ9.c2lnbmF0dXJlLXRoYXQtaXMtbG9uZw"
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    with caplog.at_level(logging.WARNING):
        secrets.add_request(HttpGetGrant(kind="http-get", url=f"https://host.example.test:8443/v1/grants/{token}",
                                         expires_at=expires))
        secrets.add_request(HttpPutGrant(kind="http-put", headers={"Content-Type": "application/json"},
                                         url="https://objects.example.test/runs/trajectory.json?X-Amz-Expires=900",
                                         expires_at=expires))
    text = f"token {token} under /v1/grants and /runs/trajectory.json, as application/json for 900 s"
    assert secrets.scrub(text) == text.replace(token, f"[redacted:{len(token)}]")
    assert not caplog.records


def test_a_write_only_mapping_is_redacted_by_its_credentials_alone(monkeypatch: pytest.MonkeyPatch,
                                                                   caplog: pytest.LogCaptureFixture) -> None:
    params = {"service_tier": "priority", "thinking": {"type": "enabled"},
              "extra_headers": {"Authorization": "Bearer params-0123456789", "X-Title": "toy agent app"},
              "auth": {"user": "bob-the-builder"}}
    with caplog.at_level(logging.WARNING), _serving(_echoing()) as client:
        assert client.post("/ext/agent-config", json={"model_params": params}).status_code == 200
        task_id = _turn(client, "priority enabled toy agent app Bearer params-0123456789 bob-the-builder")
        document = client.post("/ext/trajectory", json={"task_id": task_id, "format": FORMAT}).json()["trajectory"]
    assert document["steps"][0]["message"] == "priority enabled toy agent app [redacted:24] [redacted:15]"
    assert not [record for record in caplog.records if "shorter than" in record.message]


def test_a_path_token_beside_a_plain_query_and_a_signed_url_encoded_again_are_redacted() -> None:
    secrets = Secrets()
    token = "eyJvcCI6ImdldCJ9.c2lnbmF0dXJlLXRoYXQtaXMtbG9uZw"
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    secrets.add_request(HttpGetGrant(kind="http-get", url=f"https://host.example.test/v1/grants/{token}?download=1",
                                     expires_at=expires))
    signed = "https://store.example.test/runs/1.json?X-Amz-Security-Token=abc%2Fdef%2Bghi%3D123456789&X-Amz-Expires=900"
    secrets.add_request(HttpGetGrant(kind="http-get", url=signed, expires_at=expires))
    nested = f"https://viewer.example.test/open?src={quote(signed, safe='')}"
    assert "abc%252Fdef%252Bghi%253D123456789" in nested
    scrubbed = secrets.scrub([f"GET /v1/grants/{token}?download=1", signed, nested])
    assert not [text for text in scrubbed if token in text or "abc" in text]
    assert all("[redacted:" in text for text in scrubbed)


def test_a_roles_grants_add_their_signatures_alone() -> None:
    """Every grant a role signs carries the same session token and access key: each is one secret, so a grant's share of
    the match is its signature, however long its URL."""
    secrets = Secrets()
    token = "IQoJb3JpZ2luX2VjEJr" + "x" * 800
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    for n in range(200):
        secrets.add_request(HttpPutGrant(kind="http-put", expires_at=expires, url=(
            f"https://store.s3.ca-central-1.amazonaws.com/runs/{n}/trajectory.json?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=ASIAEXAMPLE0123%2F20261119%2Fca-central-1%2Fs3%2Faws4_request"
            f"&X-Amz-Security-Token={token}&X-Amz-Signature={n:064x}")))
    assert sum(map(len, secrets._forms)) < 200 * 64 + 5 * (len(token) + 80)


def test_a_signed_upload_form_is_redacted_by_its_signing_fields_alone() -> None:
    secrets = Secrets()
    form = HttpPostPolicyGrant(kind="http-post-policy", url="https://bucket.example.test/", path_field="key",
                               file_field="file", headers={"Content-Type": "application/json"}, fields={
                                   "key": "runs/1/${filename}", "x-amz-algorithm": "AWS4-HMAC-SHA256",
                                   "x-amz-credential": "AKIAEXAMPLE/20261119/us-east-1/s3", "policy": "eyJwb2xpY3kiOnt9",
                                   "x-amz-signature": "0123456789abcdef"})
    secrets.add_request(form)
    text = "runs/1/${filename} AWS4-HMAC-SHA256 application/json AKIAEXAMPLE/20261119/us-east-1/s3 eyJwb2xpY3kiOnt9"
    assert secrets.scrub(text) == "runs/1/${filename} AWS4-HMAC-SHA256 application/json [redacted:33] [redacted:16]"


def test_a_clock_followers_read_token_is_redacted_and_a_bare_cdp_origin_is_not() -> None:
    secrets = Secrets()
    secrets.add_request(ClockSyncTimeRequest(env_get_time_url="https://gw.example.test/clock",
                                             read_token="clock-token-0123456789"))
    assert secrets.scrub("token clock-token-0123456789 at https://gw.example.test/clock") == (
        "token [redacted:22] at https://gw.example.test/clock")


def test_the_longest_secret_wins_and_an_escaped_one_is_redacted_too() -> None:
    secrets = Secrets()
    secrets.add("abcdefgh", source="test")
    secrets.add("abcdefgh-ijkl", source="test")
    secrets.add('pa"ss-wörd-0123', source="test")
    assert secrets.scrub({"a": ["abcdefgh-ijkl and abcdefgh"], "b": json.dumps({"p": 'pa"ss-wörd-0123'})}) == {
        "a": ["[redacted:13] and [redacted:8]"], "b": '{"p": "[redacted:21]"}'}


def test_a_secret_is_redacted_in_mapping_keys_too() -> None:
    secrets = Secrets()
    secrets.add(SECRET, source="test")
    assert secrets.scrub({SECRET: 1, "nested": [{f"key {SECRET}": SECRET}]}) == {
        f"[redacted:{len(SECRET)}]": 1, "nested": [{f"key [redacted:{len(SECRET)}]": f"[redacted:{len(SECRET)}]"}]}


def test_a_non_ascii_secret_in_an_inbound_data_part_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = 'pä"ss-wörd-0123'
    with _serving(_echoing(register=secret)) as client:
        task_id = _send(client, "hello", "m-1", extra_parts=[{"kind": "data", "data": {"password": secret}}])
        _ended(client, task_id)
        reads = _reads(client, task_id, monkeypatch)
    served = [text for read in reads for text in _strings(read)]
    assert not [text for text in served if "ss-wörd-0123" in text or "ss-w\\u00f6rd" in text]
    assert any('{"password":"[redacted:16]"}' in text for text in served)


def test_the_trie_matches_what_a_longest_first_alternation_does() -> None:
    rng = random.Random(7)
    alphabet = "ab/?=-ü"
    forms = {"".join(rng.choice(alphabet) for _ in range(rng.randint(8, 14))) for _ in range(60)}
    forms |= {form[:9] for form in list(forms)[:20]}
    naive = re.compile("|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True)))
    trie = re.compile(_secrets._trie(forms))
    for _ in range(300):
        text = "".join(rng.choice(alphabet + "xyz ") for _ in range(rng.randint(0, 80)))
        assert trie.sub("#", text) == naive.sub("#", text)


def test_many_grants_compile_once_when_a_read_needs_them(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = []
    trie = _secrets._trie
    monkeypatch.setattr(_secrets, "_trie", lambda forms: compiled.append(len(forms)) or trie(forms))
    secrets = Secrets()
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    urls = [f"https://bucket.example.test/skills/s/file-{n}.md?X-Amz-Signature={n:024d}" for n in range(3000)]
    secrets.add_request(ReadObject(media_type="text/markdown", max_bytes=10, read=HttpGetGrant(
        kind="http-get", url=urls[0], expires_at=expires)))
    for url in urls[1:]:
        secrets.add_request(HttpGetGrant(kind="http-get", url=url, expires_at=expires))
    assert compiled == []
    text = " ".join(["fetched", urls[1234], "and", "x" * 50_000])
    assert secrets.scrub(text).startswith(f"fetched {urls[1234].replace(f'{1234:024d}', '[redacted:24]')} and")
    assert secrets.scrub("again") == "again" and len(compiled) == 1


def test_a_short_secret_is_left_with_one_warning_per_source(caplog: pytest.LogCaptureFixture) -> None:
    secrets = Secrets()
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            secrets.add("short", source="the agent")
    assert secrets.scrub("a short note") == "a short note"
    assert sum("shorter than 8 characters" in record.message for record in caplog.records) == 1
    assert "short" not in [record.args for record in caplog.records]


def _three_steps(**scrub: Any) -> AtifCopy:
    copy = AtifCopy(Echoes, "ctx-1", **scrub)
    copy.start("t1", 0, TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy", agent_version="1",
                                  agent_model=None))
    for text in ("one", "two", "three"):
        copy.native("t1", {"type": "say", "text": text})
    copy.end("t1", state=TrajectoryState.COMPLETED.value, final=(TextPart(text="done"),))
    return copy


def test_a_cursor_page_counts_its_bytes_as_it_serves_them() -> None:
    plain = _three_steps()
    budget = sum(len(json.dumps(step)) for step in plain.page(0, 10, 10_000).steps[:2])
    assert len(plain.page(0, 10, budget).steps) == 2
    # A scrub that lengthens every string, as a redaction longer than its secret does.
    padded = _three_steps(scrub=lambda value: _map_strings(value, lambda text: text + "x" * 200)).page(0, 10, budget)
    assert len(padded.steps) == 1 and padded.has_more


def test_an_agent_that_serves_no_atif_registers_nothing_and_registering_before_its_app_is_refused() -> None:
    with _serving(_echoing(register=SECRET, atif=False)) as client:
        task_id = _turn(client, f"the key is {SECRET}")
        assert _native(client, task_id) == [{"type": "say", "text": f"the key is {SECRET}"}]
    with pytest.raises(RuntimeError, match="after create_app"):
        _echoing().register_secret(SECRET)
