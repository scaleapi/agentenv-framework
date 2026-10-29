import asyncio
import base64
import io
import json
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

from tst.unit.agents.openhands.agent import openhands_agent

OpenHandsAgent = openhands_agent.OpenHandsAgent
Endpoint = openhands_agent.Endpoint
Outcome = openhands_agent.Outcome

# What a run cancelled midway had spent by then.
SPENT: Final[Usage] = Usage(tool_call_count=1, input_tokens=90, output_tokens=10, total_tokens=100, cost_usd=0.1)
GRANT: Final[dict[str, str]] = {"kind": "http-put", "url": "https://objects.test/put", "expires_at": "2099-01-01T00:00:00Z"}
READ: Final[dict[str, str]] = {"kind": "http-get", "url": "https://objects.test/get", "expires_at": "2099-01-01T00:00:00Z"}


@dataclass(slots=True)
class FakeRuntime:
    """Persists each conversation as the SDK does, a directory named by its id holding ``base_state.json``, which it
    resumes only under the id and directory the state records, and records the turns it ran. With ``gate`` set, a run
    waits for it to open."""

    turns: list[Any] = field(default_factory=list)
    outcome: Outcome | None = None
    hang: bool = False
    cancelled: bool = False
    gate: threading.Event | None = None
    active: Counter = field(default_factory=Counter)
    stopped: bool = False

    def stop(self) -> None:
        self.stopped = True

    async def run(self, turn: Any) -> Outcome:
        self.turns.append(turn)
        directory = turn.conversations_dir / turn.conversation_id
        state_path = directory / "base_state.json"
        expected = {"id": str(uuid.UUID(turn.conversation_id)), "persistence_dir": str(directory)}
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if {key: state[key] for key in expected} != expected:
                raise ValueError(f"Conversation ID mismatch: provided {expected}, persisted {state}")
        else:
            directory.mkdir(parents=True)
            state = {**expected, "agent": {"kind": "Agent"}, "history": []}
        state["history"].append(turn.text)
        state_path.write_text(json.dumps(state))
        turn.on_event({"kind": "MessageEvent", "source": "user"})
        turn.on_event({"kind": "ActionEvent", "tool_name": "terminal"})
        self.active[turn.conversation_id] += 1
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
            self.active[turn.conversation_id] -= 1
        return self.outcome or Outcome("Done.", Usage(tool_call_count=1, input_tokens=200, output_tokens=40,
                                                      total_tokens=240, cost_usd=0.5))

    def history(self, conversations_dir: Path, conversation_id: str) -> dict[str, Any]:
        return json.loads((conversations_dir / conversation_id / "base_state.json").read_text())


ENDPOINT: Final[Endpoint] = Endpoint("http://litellm.test", "sk-test", {"x-project": "p1"})


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def agent(tmp_path: Path, runtime: FakeRuntime) -> OpenHandsAgent:
    return OpenHandsAgent(runtime, endpoint=ENDPOINT, home=tmp_path / "home", workspace=tmp_path / "ws")


@pytest.fixture
def client(agent: OpenHandsAgent):
    with TestClient(agent.create_app(), client=("127.0.0.1", 50000)) as client:
        client.post("/ext/agent-config", json={"model": "claude-sonnet", "effort": "high"})
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


def test_run_returns_the_reply_usage_and_live_trajectory(client: TestClient, runtime: FakeRuntime) -> None:
    task = _send(client, [{"kind": "text", "text": "fix the bug"}, {"kind": "data", "data": {"k": 1}}])

    text, data = _reply(task)
    assert task["status"]["state"] == "completed"
    assert text == "Done."
    assert data["usage"] == {"tool_call_count": 1, "input_tokens": 200, "output_tokens": 40, "total_tokens": 240,
                             "cost_usd": 0.5}
    turn = runtime.turns[0]
    assert turn.text == 'fix the bug\n\n{\n  "k": 1\n}'
    assert turn.config.model == "claude-sonnet"
    assert turn.config.effort == "high"
    assert turn.endpoint == ENDPOINT
    trajectory = client.post("/ext/trajectory", json={"task_id": task["id"]}).json()["trajectory"]
    assert [event["kind"] for event in trajectory] == ["MessageEvent", "ActionEvent"]


def test_a_context_keeps_one_conversation(client: TestClient, runtime: FakeRuntime, tmp_path: Path) -> None:
    _send(client, [{"kind": "text", "text": "one"}], context_id="ctx/a b")
    _send(client, [{"kind": "text", "text": "two"}], context_id="ctx/a b")
    _send(client, [{"kind": "text", "text": "three"}], context_id="ctx-2")

    first, second, other = (turn.conversation_id for turn in runtime.turns)
    assert first == second == openhands_agent.conversation_id("ctx/a b")
    assert other != first
    assert runtime.history(tmp_path / "home" / "conversations", first)["history"] == ["one", "two"]


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


def test_images_are_attached_and_every_file_is_left_on_disk(client: TestClient, runtime: FakeRuntime) -> None:
    png = base64.b64encode(b"\x89PNG").decode()
    mp3 = base64.b64encode(b"ID3").decode()
    _send(client, [{"kind": "text", "text": "look"},
                   {"kind": "file", "file": {"name": "../red.png", "mimeType": "image/png", "bytes": png}},
                   {"kind": "file", "file": {"name": "clip.mp3", "mimeType": "audio/mpeg", "bytes": mp3}}])

    turn = runtime.turns[0]
    assert turn.images == (f"data:image/png;base64,{png}",)
    saved = [line.split(" saved at ")[1].rstrip("]") for line in turn.text.split("\n\n")[1:]]
    assert [Path(path).name for path in saved] == ["1-red.png", "2-clip.mp3"]
    assert Path(saved[1]).read_bytes() == b"ID3"


def test_inline_and_bundle_skills_are_installed_for_the_runtime(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"from {source.read.url}")

    monkeypatch.setattr(openhands_agent, "download", fake_download)
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


def test_a_runtime_failure_keeps_its_code_and_type(client: TestClient, runtime: FakeRuntime) -> None:
    runtime.outcome = Outcome("OpenHands failed: rate limited", Usage(), "openhands.RateLimitError", "infra_error")

    task = _send(client, [{"kind": "text", "text": "go"}])

    text, data = _reply(task)
    assert task["status"]["state"] == "failed"
    assert text == "OpenHands failed: rate limited"
    assert (data["error_code"], data["error_type"]) == ("openhands.RateLimitError", "infra_error")


def test_timeout_cancels_the_runtime(client: TestClient, runtime: FakeRuntime) -> None:
    runtime.hang = True
    client.post("/ext/agent-config", json={"timeout_seconds": 1})

    task = _send(client, [{"kind": "text", "text": "go"}])

    assert task["status"]["state"] == "failed"
    assert _reply(task)[1]["error_code"] == "openhands.timeout"
    assert runtime.cancelled


def test_a_timed_out_task_reports_what_its_run_had_spent(client: TestClient, runtime: FakeRuntime) -> None:
    runtime.hang = True
    client.post("/ext/agent-config", json={"timeout_seconds": 1})

    task = _send(client, [{"kind": "text", "text": "go"}])

    _, data = _reply(task)
    assert data["error_code"] == "openhands.timeout"
    assert data["usage"] == {"tool_call_count": 1, "input_tokens": 90, "output_tokens": 10, "total_tokens": 100,
                             "cost_usd": 0.1}


def test_reading_a_file_part_counts_toward_the_task_timeout(
    client: TestClient, runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_client = httpx.AsyncClient

    async def stall(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, content=b"late")

    monkeypatch.setattr(openhands_agent.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(stall), **kwargs))
    client.post("/ext/agent-config", json={"timeout_seconds": 1})
    started = time.monotonic()

    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "slow.txt", "uri": "https://files.test/slow.txt"}}])

    assert _reply(task)[1]["error_code"] == "openhands.timeout"
    assert time.monotonic() - started < 10
    assert runtime.turns == []


def test_a_cancelled_task_interrupts_the_runtime(client: TestClient, runtime: FakeRuntime) -> None:
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


def test_the_server_shutting_down_stops_the_runtime(agent: OpenHandsAgent, runtime: FakeRuntime) -> None:
    with TestClient(agent.create_app()):
        assert not runtime.stopped

    assert runtime.stopped


def test_errors_are_answered_as_json_details(client: TestClient, objects: dict[str, bytes]) -> None:
    response = client.post("/ext/snapshot", json={"context_id": "nope", "objects": {"trajectory": _write()}})

    assert response.status_code == 404
    assert response.json() == {"detail": "No conversation for context 'nope'"}


def test_missing_model_fails_without_running(agent: OpenHandsAgent, runtime: FakeRuntime) -> None:
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    assert _reply(task)[1]["error_code"] == "openhands.no_model"
    assert runtime.turns == []


def test_the_default_model_is_the_fallback(tmp_path: Path, runtime: FakeRuntime) -> None:
    agent = OpenHandsAgent(runtime, endpoint=ENDPOINT, default_model="fallback-model", home=tmp_path / "home",
                           workspace=tmp_path / "ws")
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    assert task["status"]["state"] == "completed"
    assert runtime.turns[0].config.model == "fallback-model"


def test_missing_endpoint_is_an_infra_failure(tmp_path: Path, runtime: FakeRuntime) -> None:
    agent = OpenHandsAgent(runtime, endpoint=None, default_model="m", home=tmp_path / "home", workspace=tmp_path / "ws")
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    _, data = _reply(task)
    assert (data["error_code"], data["error_type"]) == ("openhands.no_endpoint", "infra_error")
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


def _write() -> dict[str, Any]:
    return {"media_type": "application/octet-stream", "max_bytes": 1 << 20, "write": GRANT}


def _read(name: str) -> dict[str, Any]:
    return {"media_type": "application/octet-stream", "max_bytes": 1 << 20,
            "read": {**READ, "url": f"https://objects.test/{name}"}}


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

    monkeypatch.setattr(openhands_agent, "upload", fake_upload)
    monkeypatch.setattr(openhands_agent, "download", fake_download)
    return stored


def test_snapshot_round_trip_rebinds_the_conversation_to_the_target_context(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, objects: dict[str, bytes]
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
    target = openhands_agent.conversation_id("target")
    conversations = tmp_path / "home" / "conversations"
    restored = runtime.history(conversations, target)
    task = _send(client, [{"kind": "text", "text": "what was the token?"}], context_id="target")

    assert loaded.json() == {"context_id": "target"}
    assert (tmp_path / "ws" / "notes.txt").read_text() == "remember TOKEN-1"
    assert not (tmp_path / "ws" / "after.txt").exists()
    assert (restored["id"], restored["persistence_dir"]) == (str(uuid.UUID(target)), str(conversations / target))
    assert task["status"]["state"] == "completed"
    assert runtime.history(conversations, target)["history"] == ["remember TOKEN-1", "what was the token?"]


def test_an_empty_workspace_restores_as_empty(
    client: TestClient, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    _send(client, [{"kind": "text", "text": "remember TOKEN-1"}], context_id="source")
    client.post("/ext/snapshot", json={"context_id": "source", "objects": {
        "trajectory": _write(), "workspace": _write()}})
    (tmp_path / "ws" / "after.txt").write_text("written after the snapshot")

    loaded = client.put("/ext/snapshot", json={"objects": {
        "trajectory": _read("trajectory"), "workspace": _read("workspace")}, "target_context_id": "target"})

    assert loaded.status_code == 200, loaded.text
    assert list((tmp_path / "ws").iterdir()) == []


def _attached(text: str) -> Path:
    return Path(text.split(" saved at ")[1].rstrip("]"))


def test_a_snapshot_restores_the_files_attached_in_its_context_as_they_were_saved(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    notes = base64.b64encode(b"TOKEN-7").decode()
    _send(client, [{"kind": "text", "text": "keep this"},
                   {"kind": "file", "file": {"name": "notes.txt", "mimeType": "text/plain", "bytes": notes}}],
          context_id="source")
    attached = _attached(runtime.turns[0].text)
    saved = client.post("/ext/snapshot", json={"context_id": "source", "objects": {"trajectory": _write()}})
    attached.write_bytes(b"changed after the snapshot")

    loaded = client.put("/ext/snapshot", json={"objects": {"trajectory": _read("trajectory")},
                                               "target_context_id": "target"})

    inputs, conversations = tmp_path / "home" / "inputs", tmp_path / "home" / "conversations"
    source, target = (openhands_agent.conversation_id(context) for context in ("source", "target"))
    (message,) = runtime.history(conversations, target)["history"]
    assert (saved.status_code, loaded.status_code) == (200, 200)
    assert _attached(message) == inputs / target / attached.relative_to(inputs / source)
    assert _attached(message).read_bytes() == b"TOKEN-7"
    assert attached.read_bytes() == b"changed after the snapshot"
    assert sorted(path.name for path in (conversations / target).iterdir()) == ["base_state.json"]


def test_a_snapshot_loaded_on_an_agent_with_another_home_names_its_files_there(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    notes = base64.b64encode(b"TOKEN-8").decode()
    _send(client, [{"kind": "text", "text": "keep this"},
                   {"kind": "file", "file": {"name": "notes.txt", "mimeType": "text/plain", "bytes": notes}}],
          context_id="source")
    client.post("/ext/snapshot", json={"context_id": "source", "objects": {"trajectory": _write()}})
    other = OpenHandsAgent(runtime, endpoint=ENDPOINT, home=tmp_path / "other", workspace=tmp_path / "other-ws")

    with TestClient(other.create_app()) as other_client:
        loaded = other_client.put("/ext/snapshot", json={"objects": {"trajectory": _read("trajectory")},
                                                         "target_context_id": "target"})

    target = openhands_agent.conversation_id("target")
    (message,) = runtime.history(tmp_path / "other" / "conversations", target)["history"]
    assert loaded.status_code == 200
    assert _attached(message).is_relative_to(tmp_path / "other" / "inputs" / target)
    assert _attached(message).read_bytes() == b"TOKEN-8"


@pytest.mark.parametrize("member, content", [
    ("README", b""),
    ("base_state.json", b"{}"),
    ("base_state.json", b'{"id": "not-a-uuid", "agent": {}}'),
    ("base_state.json", json.dumps({"id": str(uuid.uuid4())}).encode()),
], ids=["no-state", "empty-state", "bad-id", "no-agent"])
def test_a_snapshot_that_holds_no_conversation_changes_nothing(
    client: TestClient, runtime: FakeRuntime, tmp_path: Path, objects: dict[str, bytes], member: str, content: bytes
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
    conversations = tmp_path / "home" / "conversations"
    source = openhands_agent.conversation_id("source")

    loaded = client.put("/ext/snapshot", json={"objects": {
        "trajectory": _read("trajectory"), "workspace": _read("workspace")}, "target_context_id": "source"})

    assert loaded.status_code == 400
    assert "holds no OpenHands conversation" in loaded.json()["detail"]
    assert [path.name for path in (tmp_path / "ws").iterdir()] == ["notes.txt"]
    assert (tmp_path / "ws" / "notes.txt").read_text() == "current"
    assert [path.name for path in conversations.iterdir()] == [source]
    assert runtime.history(conversations, source)["history"] == ["remember TOKEN-1"]


def test_snapshot_of_an_unknown_context_is_404(client: TestClient, objects: dict[str, bytes]) -> None:
    response = client.post("/ext/snapshot", json={"context_id": "nope", "objects": {"trajectory": _write()}})

    assert response.status_code == 404


def _send_in_background(client: TestClient, text: str, context_id: str) -> threading.Thread:
    thread = threading.Thread(target=_send, args=(client, [{"kind": "text", "text": text}], context_id))
    thread.start()
    return thread


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


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
        state = json.load(tar.extractfile("base_state.json"))
    assert not finished_while_running
    assert saved["response"].status_code == 200
    assert state["history"] == ["plant", "more"]


def test_a_workspace_snapshot_waits_for_the_tasks_running_in_other_contexts(
    client: TestClient, runtime: FakeRuntime, objects: dict[str, bytes]
) -> None:
    _send(client, [{"kind": "text", "text": "plant"}], context_id="ctx-a")
    runtime.gate = threading.Event()
    running = _send_in_background(client, "write files", "ctx-b")
    _wait_until(lambda: sum(runtime.active.values()) == 1)
    saved: dict[str, httpx.Response] = {}
    saver = threading.Thread(target=lambda: saved.update(response=client.post(
        "/ext/snapshot", json={"context_id": "ctx-a", "objects": {"trajectory": _write(), "workspace": _write()}})))
    saver.start()
    time.sleep(0.3)
    finished_while_running = "response" in saved
    runtime.gate.set()
    running.join()
    saver.join()

    assert not finished_while_running
    assert saved["response"].status_code == 200


def test_conversation_locks_are_dropped_once_released(
    agent: OpenHandsAgent, client: TestClient, objects: dict[str, bytes]
) -> None:
    for context in ("one", "two", "three"):
        _send(client, [{"kind": "text", "text": "go"}], context_id=context)
    client.post("/ext/snapshot", json={"context_id": "one", "objects": {"trajectory": _write()}})

    assert agent._conversation_locks == {}


def test_a_bundle_skill_cannot_write_outside_its_directory(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("payload")

    monkeypatch.setattr(openhands_agent, "download", fake_download)
    grant = {"media_type": "text/plain", "max_bytes": 64, "size_bytes": 8, "read": READ}
    responses = [client.post("/ext/skill-config", json={
        "name": "evil", "description": "Evil",
        "skill_bundle": {"max_total_bytes": 128, "files": [
            {"path": "SKILL.md", "object": grant}, {"path": path, "object": grant}]},
    }) for path in ("../../escaped.sh", "/tmp/escaped.sh", "scripts/../../escaped.sh")]

    assert [response.status_code for response in responses] == [400, 400, 400]
    assert not (tmp_path / "home" / "escaped.sh").exists()
    assert not (tmp_path / "home" / "skills").exists()


def test_an_oversized_attachment_is_refused(
    client: TestClient, runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(openhands_agent, "MAX_ATTACHMENT_BYTES", 8)
    monkeypatch.setattr(openhands_agent.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 64)), **kwargs))

    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "big.txt", "uri": "https://files.test/big.txt"}}])

    assert task["status"]["state"] == "failed"
    assert _reply(task)[1]["error_code"] == "openhands.attachment"
    assert runtime.turns == []


def test_only_http_file_uris_are_fetched(client: TestClient, runtime: FakeRuntime) -> None:
    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "passwd", "uri": "file:///etc/passwd"}}])

    text, data = _reply(task)
    assert task["status"]["state"] == "failed"
    assert data["error_code"] == "openhands.attachment"
    assert "unsupported file URI scheme" in text
    assert runtime.turns == []


def test_a_malformed_file_uri_fails_as_an_attachment(client: TestClient, runtime: FakeRuntime) -> None:
    task = _send(client, [{"kind": "text", "text": "read it"},
                          {"kind": "file", "file": {"name": "x.txt", "uri": "https://[::1/x.txt"}}])

    assert _reply(task)[1]["error_code"] == "openhands.attachment"
    assert runtime.turns == []


def test_install_extension_is_declared_on_the_card(client: TestClient) -> None:
    card = client.get("/.well-known/agent.json").json()
    install = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:install/v1")["params"]

    assert install["a2a_port"] == 8000
    assert set(install["required_params"]) == {"container", "agent_ctx_tar", "a2a_port", "litellm_api_key",
                                               "litellm_base_url", "litellm_extra_headers"}
    formatted = [command.format(container="c", agent_ctx_tar="/t.tar", a2a_port="8000", litellm_api_key="'sk-secret'",
                                litellm_base_url="u", litellm_extra_headers="'{}'")
                 for command in install["install_commands"]]
    assert formatted[-1].endswith("c sh /opt/openhands-a2a/start.sh")
    assert all("sk-secret" not in command.splitlines()[0] for command in formatted)
    assert "LITELLM_API_KEY='sk-secret'" in formatted[-2].splitlines()


def test_the_card_advertises_the_extensions_agent_env_validates(client: TestClient) -> None:
    card = client.get("/.well-known/agent.json").json()
    uris = {extension["uri"] for extension in card["capabilities"]["extensions"]}

    assert {f"urn:agentenv:{name}/v1" for name in (
        "agent-config", "mcp-config", "skill-config", "trajectory", "snapshot", "peer-agents", "install", "triggers",
    )} <= uris


def test_the_card_version_is_the_pinned_sdk() -> None:
    requirements = (openhands_agent.Path(openhands_agent.__file__).parent / "requirements.txt").read_text().splitlines()

    assert f"openhands-sdk=={openhands_agent.OPENHANDS_VERSION}" in requirements
    assert f"openhands-tools=={openhands_agent.OPENHANDS_VERSION}" in requirements
