import asyncio
import base64
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
from agentenv_protocol.a2a_agent import Usage
from starlette.testclient import TestClient

from tst.unit.agents.pydantic_ai.agent import AGENT_DIR, pydantic_ai_agent

PydanticAIAgent = pydantic_ai_agent.PydanticAIAgent
Endpoint = pydantic_ai_agent.Endpoint
Outcome = pydantic_ai_agent.Outcome

# What a run cancelled midway had spent by then.
SPENT: Final[Usage] = Usage(tool_call_count=1, input_tokens=90, output_tokens=10, total_tokens=100)
GRANT: Final[dict[str, str]] = {"kind": "http-put", "url": "https://objects.test/put", "expires_at": "2099-01-01T00:00:00Z"}
READ: Final[dict[str, str]] = {"kind": "http-get", "url": "https://objects.test/get", "expires_at": "2099-01-01T00:00:00Z"}
ENDPOINT: Final[Endpoint] = Endpoint("http://litellm.test", "sk-test", {"x-project": "p1"})


@dataclass(slots=True)
class FakeRuntime:
    """Keeps each conversation's messages in its directory, as the real runtime does once a run succeeds, and records
    the turns it ran. With ``gate`` set, a run waits for it to open."""

    turns: list[Any] = field(default_factory=list)
    outcome: Outcome | None = None
    hang: bool = False
    cancelled: bool = False
    gate: threading.Event | None = None
    active: Counter = field(default_factory=Counter)
    stopped: bool = False

    async def stop(self) -> None:
        self.stopped = True

    def check_messages(self, data: bytes) -> None:
        messages = json.loads(data)
        if not isinstance(messages, list) or not all(isinstance(message, str) for message in messages):
            raise ValueError("not this runtime's messages")

    async def run(self, turn: Any) -> Outcome:
        self.turns.append(turn)
        turn.on_event({"kind": "request", "parts": [{"part_kind": "user-prompt", "content": turn.text}]})
        turn.on_event({"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "shell"}]})
        self.active[turn.conversation_dir] += 1
        try:
            while self.gate is not None and not self.gate.is_set():
                await asyncio.sleep(0.01)
            if self.hang:
                # Bounded, so an agent that applies no timeout fails the test instead of hanging it.
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            turn.on_usage(SPENT)
            raise
        finally:
            self.active[turn.conversation_dir] -= 1
        messages = history(turn.conversation_dir)
        turn.conversation_dir.mkdir(parents=True, exist_ok=True)
        (turn.conversation_dir / "messages.json").write_text(json.dumps([*messages, turn.text]))
        return self.outcome or Outcome("Done.", Usage(tool_call_count=1, input_tokens=200, output_tokens=40,
                                                      total_tokens=240))


def history(conversation_dir: Path) -> list[str]:
    path = conversation_dir / "messages.json"
    return json.loads(path.read_text()) if path.exists() else []


def conversation(home: Path, context_id: str) -> Path:
    return home / "conversations" / pydantic_ai_agent.conversation_id(context_id)


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def agent(tmp_path: Path, runtime: FakeRuntime) -> PydanticAIAgent:
    return PydanticAIAgent(runtime, endpoint=ENDPOINT, home=tmp_path / "home", workspace=tmp_path / "ws")


@pytest.fixture
def client(agent: PydanticAIAgent):
    with TestClient(agent.create_app(), client=("127.0.0.1", 50000)) as client:
        client.post("/ext/agent-config", json={"model": "vllm:qwen3.8-27b", "effort": "high"})
        yield client


def _send(client: TestClient, parts: list[dict[str, Any]], context_id: str = "ctx-1") -> dict[str, Any]:
    response = client.post("/a2a", json={
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user",
                        "contextId": context_id, "parts": parts},
            "configuration": {"blocking": True},
        },
    })
    return response.json()["result"]


def _reply(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    parts = task["status"]["message"]["parts"]
    text = "".join(part.get("text", "") for part in parts if part["kind"] == "text")
    data = next((part["data"] for part in parts if part["kind"] == "data"), {})
    return text, data


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def _send_in_background(client: TestClient, text: str, context_id: str) -> threading.Thread:
    thread = threading.Thread(target=_send, args=(client, [{"kind": "text", "text": text}], context_id))
    thread.start()
    return thread


def test_run_returns_the_reply_usage_and_live_trajectory(client: TestClient, runtime: FakeRuntime) -> None:
    task = _send(client, [{"kind": "text", "text": "fix the bug"}, {"kind": "data", "data": {"k": 1}}])

    text, data = _reply(task)
    assert task["status"]["state"] == "completed"
    assert text == "Done."
    assert data["usage"] == {"tool_call_count": 1, "input_tokens": 200, "output_tokens": 40, "total_tokens": 240}
    turn = runtime.turns[0]
    assert turn.text == 'fix the bug\n\n{\n  "k": 1\n}'
    assert (turn.config.model, turn.config.effort) == ("vllm:qwen3.8-27b", "high")
    assert turn.endpoint == ENDPOINT
    trajectory = client.post("/ext/trajectory", json={"task_id": task["id"]}).json()["trajectory"]
    assert [event["kind"] for event in trajectory] == ["request", "response"]


def test_a_context_keeps_one_conversation(client: TestClient, runtime: FakeRuntime, tmp_path: Path) -> None:
    _send(client, [{"kind": "text", "text": "one"}], context_id="ctx/a b")
    _send(client, [{"kind": "text", "text": "two"}], context_id="ctx/a b")
    _send(client, [{"kind": "text", "text": "three"}], context_id="ctx-2")

    first, second, other = (turn.conversation_dir for turn in runtime.turns)
    assert first == second == conversation(tmp_path / "home", "ctx/a b")
    assert other != first
    assert history(first) == ["one", "two"]


def test_mcp_servers_and_the_peer_tools_reach_the_runtime(client: TestClient, runtime: FakeRuntime) -> None:
    client.post("/ext/mcp-config", json={"name": "files", "url": "http://mcp.test/mcp",
                                         "headers": {"Authorization": "Bearer t"}})

    _send(client, [{"kind": "text", "text": "go"}])

    servers = runtime.turns[0].mcp_servers
    assert servers["files"] == {"url": "http://mcp.test/mcp", "headers": {"Authorization": "Bearer t"}}
    assert servers["peer_agents"]["url"].endswith("/ext/peer-agents/mcp")


def test_the_configured_role_is_sent_to_every_mcp_server(client: TestClient, runtime: FakeRuntime) -> None:
    client.post("/ext/mcp-config", json={"name": "files", "url": "http://mcp.test/mcp",
                                         "headers": {"Authorization": "Bearer t", "AgentEnv-Role": "stale"}})
    client.post("/ext/agent-config", json={"role": "buyer"})

    _send(client, [{"kind": "text", "text": "go"}])

    servers = runtime.turns[0].mcp_servers
    assert servers["files"]["headers"] == {"Authorization": "Bearer t", "AgentEnv-Role": "buyer"}
    assert servers["peer_agents"]["headers"] == {"AgentEnv-Role": "buyer"}


def test_images_are_attached_and_every_file_is_left_in_the_conversation(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path
) -> None:
    png = base64.b64encode(b"\x89PNG").decode()
    mp3 = base64.b64encode(b"ID3").decode()
    task = _send(client, [{"kind": "text", "text": "look"},
                          {"kind": "file", "file": {"name": "../red.png", "mimeType": "image/png", "bytes": png}},
                          {"kind": "file", "file": {"name": "clip.mp3", "mimeType": "audio/mpeg", "bytes": mp3}}])

    turn = runtime.turns[0]
    saved = [_attached(line) for line in turn.text.split("\n\n")[1:]]
    assert saved == [turn.conversation_dir / "inputs" / task["id"] / name for name in ("1-red.png", "2-clip.mp3")]
    assert [(image.path, image.media_type) for image in turn.images] == [(saved[0], "image/png")]
    assert saved[1].read_bytes() == b"ID3"


def test_inline_and_bundle_skills_are_installed_for_the_runtime(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"from {source.read.url}")

    monkeypatch.setattr(pydantic_ai_agent, "download", fake_download)
    grant = {"media_type": "text/markdown", "max_bytes": 64, "size_bytes": 8, "read": READ}
    client.post("/ext/skill-config", json={"name": "greet", "description": "Greet", "skill_md": "# Greet\nSay hi."})
    client.post("/ext/skill-config", json={
        "name": "lint", "description": "Lint",
        "skill_bundle": {"max_total_bytes": 128, "files": [
            {"path": "SKILL.md", "object": grant}, {"path": "scripts/run.sh", "object": grant}]},
    })

    _send(client, [{"kind": "text", "text": "go"}])

    skills = tmp_path / "home" / "skills"
    assert (skills / "greet" / "SKILL.md").read_text() == "# Greet\nSay hi."
    assert (skills / "lint" / "scripts" / "run.sh").read_text() == f"from {READ['url']}"
    assert sorted(path.name for path in skills.iterdir()) == ["greet", "lint"]
    assert runtime.turns[0].skills_dir == skills
    assert sorted(runtime.turns[0].skill_names) == ["greet", "lint"]


def test_a_bundle_skill_cannot_write_outside_its_directory(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("payload")

    monkeypatch.setattr(pydantic_ai_agent, "download", fake_download)
    grant = {"media_type": "text/plain", "max_bytes": 64, "size_bytes": 8, "read": READ}
    responses = [client.post("/ext/skill-config", json={
        "name": "evil", "description": "Evil",
        "skill_bundle": {"max_total_bytes": 128, "files": [
            {"path": "SKILL.md", "object": grant}, {"path": path, "object": grant}]},
    }) for path in ("../../escaped.sh", "/tmp/escaped.sh", "scripts/../../escaped.sh")]

    assert [response.status_code for response in responses] == [400, 400, 400]
    assert not (tmp_path / "home" / "escaped.sh").exists()
    assert not (tmp_path / "home" / "skills").exists()


def test_a_runtime_failure_keeps_its_code_and_type(client: TestClient, runtime: FakeRuntime) -> None:
    runtime.outcome = Outcome("The model endpoint answered 429: slow down", Usage(), "pydantic_ai.model_http_429",
                              "infra_error")

    task = _send(client, [{"kind": "text", "text": "go"}])

    text, data = _reply(task)
    assert task["status"]["state"] == "failed"
    assert text == "The model endpoint answered 429: slow down"
    assert (data["error_code"], data["error_type"]) == ("pydantic_ai.model_http_429", "infra_error")


def test_a_timed_out_task_cancels_the_runtime_and_reports_what_it_had_spent(
    client: TestClient, runtime: FakeRuntime
) -> None:
    runtime.hang = True
    client.post("/ext/agent-config", json={"timeout_seconds": 1})

    task = _send(client, [{"kind": "text", "text": "go"}])

    _, data = _reply(task)
    assert task["status"]["state"] == "failed"
    assert data["error_code"] == "pydantic_ai.timeout"
    assert data["usage"] == {"tool_call_count": 1, "input_tokens": 90, "output_tokens": 10, "total_tokens": 100}
    assert runtime.cancelled


def test_reading_a_file_part_counts_toward_the_task_timeout(
    client: TestClient, runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_client = httpx.AsyncClient

    async def stall(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, content=b"late")

    monkeypatch.setattr(pydantic_ai_agent.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(stall), **kwargs))
    client.post("/ext/agent-config", json={"timeout_seconds": 1})
    started = time.monotonic()

    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "slow.txt", "uri": "https://files.test/slow.txt"}}])

    assert _reply(task)[1]["error_code"] == "pydantic_ai.timeout"
    assert time.monotonic() - started < 10
    assert runtime.turns == []


def test_a_cancelled_task_cancels_the_runtime(client: TestClient, runtime: FakeRuntime) -> None:
    runtime.hang = True
    sent = client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
        "message": {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user", "contextId": "ctx",
                    "parts": [{"kind": "text", "text": "go"}]},
        "configuration": {"blocking": False}}}).json()["result"]
    _wait_until(lambda: sum(runtime.active.values()) == 1)

    client.post("/a2a", json={"jsonrpc": "2.0", "id": "2", "method": "tasks/cancel", "params": {"id": sent["id"]}})
    _wait_until(lambda: runtime.cancelled)

    task = client.post("/a2a", json={"jsonrpc": "2.0", "id": "3", "method": "tasks/get",
                                     "params": {"id": sent["id"]}}).json()["result"]
    assert task["status"]["state"] == "canceled"


def test_the_server_shutting_down_stops_the_runtime(agent: PydanticAIAgent, runtime: FakeRuntime) -> None:
    with TestClient(agent.create_app()):
        assert not runtime.stopped

    assert runtime.stopped


def test_missing_model_fails_without_running(agent: PydanticAIAgent, runtime: FakeRuntime) -> None:
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    assert _reply(task)[1]["error_code"] == "pydantic_ai.no_model"
    assert runtime.turns == []


def test_the_default_model_is_the_fallback(tmp_path: Path, runtime: FakeRuntime) -> None:
    agent = PydanticAIAgent(runtime, endpoint=ENDPOINT, default_model="fallback-model", home=tmp_path / "home",
                            workspace=tmp_path / "ws")
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    assert task["status"]["state"] == "completed"
    assert runtime.turns[0].config.model == "fallback-model"


def test_the_environment_names_the_fallback_model_else_the_default() -> None:
    assert pydantic_ai_agent.model_from_environ({}) == "anthropic/claude-sonnet-4-6"
    assert pydantic_ai_agent.model_from_environ({"PYDANTIC_AI_A2A_MODEL": "vllm:qwen3.8-27b"}) == "vllm:qwen3.8-27b"


def test_missing_endpoint_is_an_infra_failure(tmp_path: Path, runtime: FakeRuntime) -> None:
    agent = PydanticAIAgent(runtime, endpoint=None, default_model="m", home=tmp_path / "home", workspace=tmp_path / "ws")
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    _, data = _reply(task)
    assert (data["error_code"], data["error_type"]) == ("pydantic_ai.no_endpoint", "infra_error")
    assert runtime.turns == []


def test_the_endpoint_is_read_from_the_litellm_variables() -> None:
    environ = {"LITELLM_BASE_URL": "http://litellm.test", "LITELLM_API_KEY": "sk-test",
               "LITELLM_EXTRA_HEADERS": '{"x-project": "p1"}'}
    without_headers = {"LITELLM_BASE_URL": "http://litellm.test", "LITELLM_EXTRA_HEADERS": ""}

    assert Endpoint.from_environ(environ) == ENDPOINT
    assert Endpoint.from_environ(without_headers) == Endpoint("http://litellm.test")
    assert Endpoint.from_environ({"LITELLM_API_KEY": "sk-test"}) is None


@pytest.mark.parametrize("headers", ["{not json", '["x-project"]'])
def test_malformed_extra_headers_are_refused_at_startup(headers: str) -> None:
    with pytest.raises(ValueError, match="LITELLM_EXTRA_HEADERS"):
        Endpoint.from_environ({"LITELLM_BASE_URL": "http://litellm.test", "LITELLM_EXTRA_HEADERS": headers})


def test_only_http_file_uris_are_fetched(client: TestClient, runtime: FakeRuntime) -> None:
    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "passwd", "uri": "file:///etc/passwd"}}])

    text, data = _reply(task)
    assert task["status"]["state"] == "failed"
    assert data["error_code"] == "pydantic_ai.attachment"
    assert "unsupported file URI scheme" in text
    assert runtime.turns == []


def test_an_oversized_attachment_is_refused(
    client: TestClient, runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(pydantic_ai_agent, "MAX_ATTACHMENT_BYTES", 8)
    monkeypatch.setattr(pydantic_ai_agent.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 64)), **kwargs))

    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "big.txt", "uri": "https://files.test/big.txt"}}])

    assert _reply(task)[1]["error_code"] == "pydantic_ai.attachment"
    assert runtime.turns == []


def test_an_oversized_inline_attachment_is_refused(
    client: TestClient, runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pydantic_ai_agent, "MAX_ATTACHMENT_BYTES", 8)
    big = base64.b64encode(b"x" * 9).decode()

    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "big.txt", "mimeType": "text/plain", "bytes": big}}])

    assert _reply(task)[1]["error_code"] == "pydantic_ai.attachment"
    assert runtime.turns == []


def _write() -> dict[str, Any]:
    return {"media_type": "application/octet-stream", "max_bytes": 1 << 20, "write": GRANT}


def _read(name: str) -> dict[str, Any]:
    return {"media_type": "application/octet-stream", "max_bytes": 1 << 20,
            "read": {**READ, "url": f"https://objects.test/{name}"}}


def _attached(text: str) -> Path:
    return Path(text.split(" saved at ")[1].rstrip("]"))


@pytest.fixture
def objects(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """An in-memory object store behind the transfer helpers the agent calls."""
    stored: dict[str, bytes] = {}
    names = iter(["trajectory", "workspace"])

    class FakeUploaded(dict):
        def model_dump(self, mode: str = "json") -> dict:
            return dict(self)

    async def fake_upload(target, source: Path):
        body = source.read_bytes()
        stored[next(names)] = body
        return FakeUploaded(size_bytes=len(body))

    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(stored[source.read.url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(pydantic_ai_agent, "upload", fake_upload)
    monkeypatch.setattr(pydantic_ai_agent, "download", fake_download)
    return stored


def test_snapshot_round_trip_restores_the_conversation_and_workspace_into_the_target_context(
    client: TestClient, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    _send(client, [{"kind": "text", "text": "remember TOKEN-1"}], context_id="source")
    (tmp_path / "ws" / "notes.txt").write_text("remember TOKEN-1")
    saved = client.post("/ext/snapshot", json={"context_id": "source", "objects": {
        "trajectory": _write(), "workspace": _write()}})
    assert saved.status_code == 200, saved.text
    (tmp_path / "ws" / "notes.txt").unlink()
    (tmp_path / "ws" / "after.txt").write_text("written after the snapshot")

    loaded = client.put("/ext/snapshot", json={"objects": {
        "trajectory": _read("trajectory"), "workspace": _read("workspace")}, "target_context_id": "target"})
    task = _send(client, [{"kind": "text", "text": "what was the token?"}], context_id="target")

    target = conversation(tmp_path / "home", "target")
    assert loaded.json() == {"context_id": "target"}
    assert (tmp_path / "ws" / "notes.txt").read_text() == "remember TOKEN-1"
    assert not (tmp_path / "ws" / "after.txt").exists()
    assert task["status"]["state"] == "completed"
    assert history(target) == ["remember TOKEN-1", "what was the token?"]
    assert sorted(path.name for path in target.iterdir()) == ["messages.json"]


def test_a_snapshot_loaded_on_an_agent_with_another_home_names_its_attached_files_there(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    notes = base64.b64encode(b"TOKEN-8").decode()
    _send(client, [{"kind": "text", "text": "keep this"},
                   {"kind": "file", "file": {"name": "notes.txt", "mimeType": "text/plain", "bytes": notes}}],
          context_id="source")
    attached = _attached(runtime.turns[0].text)
    client.post("/ext/snapshot", json={"context_id": "source", "objects": {"trajectory": _write()}})
    attached.write_bytes(b"changed after the snapshot")
    other = PydanticAIAgent(runtime, endpoint=ENDPOINT, home=tmp_path / "other", workspace=tmp_path / "other-ws")

    with TestClient(other.create_app()) as other_client:
        loaded = other_client.put("/ext/snapshot", json={"objects": {"trajectory": _read("trajectory")},
                                                         "target_context_id": "target"})

    target = conversation(tmp_path / "other", "target")
    (message,) = history(target)
    assert loaded.status_code == 200
    assert _attached(message) == target / attached.relative_to(conversation(tmp_path / "home", "source"))
    assert _attached(message).read_bytes() == b"TOKEN-8"


def test_a_workspace_holding_the_agents_home_is_restored_around_it(
    runtime: FakeRuntime, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    """An install in a home directory works there, so the agent's home is inside its workspace."""
    workspace = tmp_path / "home-dir"
    agent = PydanticAIAgent(runtime, endpoint=ENDPOINT, default_model="m", home=workspace / ".pydantic-ai-a2a",
                            workspace=workspace)
    with TestClient(agent.create_app(), client=("127.0.0.1", 50000)) as client:
        _send(client, [{"kind": "text", "text": "remember TOKEN-1"}], context_id="source")
        _send(client, [{"kind": "text", "text": "another conversation"}], context_id="other")
        (workspace / "notes.txt").write_text("remember TOKEN-1")
        saved = client.post("/ext/snapshot", json={"context_id": "source", "objects": {
            "trajectory": _write(), "workspace": _write()}})
        (workspace / "notes.txt").unlink()
        (workspace / "after.txt").write_text("written after the snapshot")

        loaded = client.put("/ext/snapshot", json={"objects": {
            "trajectory": _read("trajectory"), "workspace": _read("workspace")}, "target_context_id": "target"})

    with tarfile.open(fileobj=io.BytesIO(objects["workspace"])) as tar:
        assert tar.getnames() == ["notes.txt"]
    assert saved.status_code == 200 and loaded.status_code == 200, loaded.text
    assert (workspace / "notes.txt").read_text() == "remember TOKEN-1"
    assert not (workspace / "after.txt").exists()
    assert history(conversation(workspace / ".pydantic-ai-a2a", "target")) == ["remember TOKEN-1"]
    assert history(conversation(workspace / ".pydantic-ai-a2a", "other")) == ["another conversation"]


@pytest.mark.parametrize("member, content", [
    ("README", b""),
    ("messages.json", b"{not json"),
    ("messages.json", b'{"kind": "request"}'),
    ("messages.json", b"[{}]"),
], ids=["no-messages", "unparseable", "not-a-list", "not-the-runtimes-messages"])
def test_a_snapshot_that_holds_no_conversation_changes_nothing(
    client: TestClient, tmp_path: Path, objects: dict[str, bytes], member: str, content: bytes
) -> None:
    _send(client, [{"kind": "text", "text": "remember TOKEN-1"}], context_id="source")
    client.post("/ext/snapshot", json={"context_id": "source", "objects": {"trajectory": _write(), "workspace": _write()}})
    (tmp_path / "ws" / "notes.txt").write_text("current")
    not_a_conversation = io.BytesIO()
    with tarfile.open(fileobj=not_a_conversation, mode="w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    objects["trajectory"] = not_a_conversation.getvalue()
    source = conversation(tmp_path / "home", "source")

    loaded = client.put("/ext/snapshot", json={"objects": {
        "trajectory": _read("trajectory"), "workspace": _read("workspace")}, "target_context_id": "source"})

    assert loaded.status_code == 400
    assert "holds no Pydantic AI conversation" in loaded.json()["detail"]
    assert [path.name for path in (tmp_path / "ws").iterdir()] == ["notes.txt"]
    assert [path.name for path in source.parent.iterdir()] == [source.name]
    assert history(source) == ["remember TOKEN-1"]


def test_a_snapshot_of_an_unknown_context_is_a_json_404(client: TestClient, objects: dict[str, bytes]) -> None:
    response = client.post("/ext/snapshot", json={"context_id": "nope", "objects": {"trajectory": _write()}})

    assert response.status_code == 404
    assert response.json() == {"detail": "No conversation for context 'nope'"}


def test_a_snapshot_load_is_refused_while_a_task_runs(
    client: TestClient, runtime: FakeRuntime, objects: dict[str, bytes]
) -> None:
    _send(client, [{"kind": "text", "text": "remember TOKEN-1"}], context_id="source")
    client.post("/ext/snapshot", json={"context_id": "source", "objects": {"trajectory": _write()}})
    runtime.gate = threading.Event()
    running = _send_in_background(client, "keep working", "busy")
    _wait_until(lambda: sum(runtime.active.values()) == 1)

    busy = client.put("/ext/snapshot", json={"objects": {"trajectory": _read("trajectory")}, "target_context_id": "t"})
    runtime.gate.set()
    running.join()
    idle = client.put("/ext/snapshot", json={"objects": {"trajectory": _read("trajectory")}, "target_context_id": "t"})

    assert (busy.status_code, idle.status_code) == (409, 200)


def test_a_snapshot_save_waits_for_its_context_running_task(
    client: TestClient, runtime: FakeRuntime, objects: dict[str, bytes]
) -> None:
    _send(client, [{"kind": "text", "text": "plant"}], context_id="ctx-a")
    runtime.gate = threading.Event()
    running = _send_in_background(client, "more", "ctx-a")
    _wait_until(lambda: sum(runtime.active.values()) == 1)
    saved: dict[str, httpx.Response] = {}
    saver = threading.Thread(target=lambda: saved.update(response=client.post(
        "/ext/snapshot", json={"context_id": "ctx-a", "objects": {"trajectory": _write()}})))
    saver.start()
    time.sleep(0.3)
    finished_while_running = "response" in saved
    runtime.gate.set()
    running.join()
    saver.join()

    with tarfile.open(fileobj=io.BytesIO(objects["trajectory"])) as tar:
        messages = json.load(tar.extractfile("messages.json"))
    assert not finished_while_running
    assert saved["response"].status_code == 200
    assert messages == ["plant", "more"]


def test_conversation_locks_are_dropped_once_released(
    agent: PydanticAIAgent, client: TestClient, objects: dict[str, bytes]
) -> None:
    for context in ("one", "two", "three"):
        _send(client, [{"kind": "text", "text": "go"}], context_id=context)
    client.post("/ext/snapshot", json={"context_id": "one", "objects": {"trajectory": _write()}})

    assert agent._conversation_locks == {}


def test_the_card_advertises_the_extensions_agent_env_validates(client: TestClient) -> None:
    card = client.get("/.well-known/agent.json").json()
    uris = {extension["uri"] for extension in card["capabilities"]["extensions"]}
    config = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:agent-config/v1")

    assert {f"urn:agentenv:{name}/v1" for name in (
        "agent-config", "mcp-config", "skill-config", "trajectory", "snapshot", "peer-agents", "install", "triggers",
    )} <= uris
    assert {"model", "system_prompt", "role", "timeout_seconds"} <= set(
        config["params"]["methods"]["set"]["request"]["supported"])
    assert card["defaultInputModes"] == ["text", "image/png", "image/jpeg", "image/gif", "image/webp"]


def test_install_extension_is_declared_on_the_card(client: TestClient) -> None:
    card = client.get("/.well-known/agent.json").json()
    install = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:install/v1")["params"]

    assert install["a2a_port"] == 8000
    assert set(install["required_params"]) == {"container", "agent_ctx_tar", "a2a_port", "litellm_api_key",
                                               "litellm_base_url", "litellm_extra_headers", "trust_env"}
    formatted = [command.format(container="c", agent_ctx_tar="/t.tar", a2a_port="8000", litellm_api_key="'sk-secret'",
                                litellm_base_url="u", litellm_extra_headers="'{}'", trust_env="''")
                 for command in install["install_commands"]]
    assert formatted[-1].endswith("c sh /opt/pydantic-ai-a2a/start.sh")
    assert all("sk-secret" not in command.splitlines()[0] for command in formatted)
    assert "LITELLM_API_KEY='sk-secret'" in formatted[-2].splitlines()


def test_the_installed_agent_starts_with_the_endpoint_and_trust_lines_install_wrote(tmp_path: Path) -> None:
    here = tmp_path / "pydantic-ai-a2a"
    here.mkdir()
    (here / "start.sh").write_text((AGENT_DIR / "start.sh").read_text().replace("/tmp/", f"{tmp_path}/"))
    trust = "SSL_CERT_FILE='/etc/agentenv/ca-bundle.pem'\nREQUESTS_CA_BUNDLE='/etc/agentenv/ca-bundle.pem'\n"
    (here / "agent.env").write_text(f"LITELLM_API_KEY={shlex.quote('sk x')}\nAGENT_TRUST_ENV={shlex.quote(trust)}\n")
    # In place of the server: what it would be started with, written beside it.
    (here / "a2a_server.py").write_text(
        "import json, os, pathlib\npathlib.Path(__file__).with_name('seen').write_text(json.dumps("
        "[os.environ.get(name) for name in ('LITELLM_API_KEY', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE')]))\n")
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.symlink_to(sys.executable)

    subprocess.run(["sh", str(here / "start.sh")], cwd="/", env={"PATH": f"{python.parent}{os.pathsep}/usr/bin:/bin"},
                   check=True, timeout=60)

    assert json.loads((here / "seen").read_text()) == ["sk x", "/etc/agentenv/ca-bundle.pem",
                                                       "/etc/agentenv/ca-bundle.pem"]


def test_the_card_version_is_the_pinned_pydantic_ai() -> None:
    requirements = (AGENT_DIR / "requirements.txt").read_text().splitlines()
    version = pydantic_ai_agent.PYDANTIC_AI_VERSION

    assert f"pydantic-ai-slim[mcp,openai]=={version}" in requirements
    assert any(line.startswith("pydantic-ai-harness[") for line in requirements)


def test_the_image_copies_every_module_the_agent_imports() -> None:
    copied = next(line for line in (AGENT_DIR / "Dockerfile").read_text().splitlines() if line.startswith("COPY "))
    sources = {path.name for path in AGENT_DIR.iterdir() if path.suffix in {".py", ".sh", ".txt"}}

    assert set(copied.split()[1:-1]) == sources


def test_a_threaded_write_cancelled_midway_finishes_before_the_cancel_reaches_its_caller() -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def write() -> None:
        started.set()
        release.wait(5)
        finished.set()

    async def cancel_midway() -> bool:
        task = asyncio.ensure_future(pydantic_ai_agent.to_completion(write))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.1)
        still_waiting = not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return still_waiting

    assert asyncio.run(cancel_midway())
    assert finished.is_set()
