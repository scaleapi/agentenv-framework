import base64
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

import changelog
import peers
import pi_agent
from pi_agent import PiAgent

FAKE_PI = Path(__file__).with_name("fake_pi.py")


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(path))
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.test/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    return path


@pytest.fixture
def client(tmp_path: Path, record: Path):
    agent = PiAgent(home=tmp_path / "home", workspace=tmp_path / "ws", pi_command=(sys.executable, str(FAKE_PI)))
    with TestClient(agent.create_app()) as client:
        client.post("/ext/agent-config", json={"model": "anthropic/claude-sonnet", "effort": "high"})
        yield client


def _send(client: TestClient, parts: list[dict[str, Any]], context_id: str = "ctx-1") -> dict[str, Any]:
    response = client.post("/a2a", json={
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {"kind": "message", "messageId": "m-" + context_id, "role": "user",
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


def test_run_returns_final_text_usage_and_trajectory(client: TestClient, record: Path) -> None:
    task = _send(client, [{"kind": "text", "text": "fix the bug"}, {"kind": "data", "data": {"k": 1}}])

    text, data = _reply(task)
    assert task["status"]["state"] == "completed"
    assert text == "Done."
    assert data["usage"] == {
        "tool_call_count": 1, "input_tokens": 200, "output_tokens": 40, "total_tokens": 252, "cost_usd": 0.5,
        "provider_details": {"cache_read_tokens": 10, "cache_write_tokens": 2},
    }
    invocation = json.loads(record.read_text())
    assert invocation["stdin"] == 'fix the bug\n\n{\n  "k": 1\n}'
    argv = invocation["argv"]
    assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet"
    assert argv[argv.index("--provider") + 1] == "agentenv"
    assert argv[argv.index("--thinking") + 1] == "high"
    assert "--no-approve" in argv
    provider = invocation["models"]["providers"]["agentenv"]
    assert provider["baseUrl"] == "http://litellm.test/v1"
    assert provider["apiKey"] == "${LITELLM_API_KEY}"
    assert provider["models"][0]["id"] == "anthropic/claude-sonnet"
    assert provider["models"][0]["reasoning"] is True
    assert "samplingParams" not in provider["models"][0]

    trajectory = client.post("/ext/trajectory", json={"task_id": task["id"]}).json()["trajectory"]
    kinds = [event["type"] for event in trajectory]
    assert kinds[0] == "session" and "tool_execution_end" in kinds
    assert "message_update" not in kinds


def test_a_context_keeps_one_pi_session(client: TestClient, record: Path) -> None:
    _send(client, [{"kind": "text", "text": "one"}], context_id="ctx/a b")
    first = json.loads(record.read_text())["argv"]
    _send(client, [{"kind": "text", "text": "two"}], context_id="ctx/a b")
    second = json.loads(record.read_text())["argv"]

    assert first[first.index("--session-id") + 1] == "ctx-a-b"
    assert second[second.index("--session-id") + 1] == "ctx-a-b"


def test_mcp_servers_are_direct_and_headers_stay_off_disk(client: TestClient, record: Path) -> None:
    client.post("/ext/mcp-config", json={"name": "files", "url": "http://mcp.test/mcp",
                                         "headers": {"Authorization": "!rm -rf /"}})

    _send(client, [{"kind": "text", "text": "go"}])

    invocation = json.loads(record.read_text())
    assert invocation["mcp"] == {"mcpServers": {"files": {
        "type": "http", "url": "http://mcp.test/mcp", "exposure": "direct", "timeout": 1800,
        "headers": {"Authorization": "${AGENTENV_MCP_0_HEADER_0}"},
    }}}
    assert invocation["env"]["AGENTENV_MCP_0_HEADER_0"] == "!rm -rf /"


def test_file_parts_are_attached(client: TestClient, record: Path) -> None:
    encoded = base64.b64encode(b"col\n1\n").decode()
    _send(client, [{"kind": "text", "text": "read it"},
                   {"kind": "file", "file": {"name": "../data.csv", "mimeType": "text/csv", "bytes": encoded}}])

    attachments = json.loads(record.read_text())["attachments"]
    assert list(attachments.values()) == ["col\n1\n"]
    assert all(Path(name).name == "1-data.csv" for name in attachments)


def test_inline_skill_is_passed_to_pi(client: TestClient, record: Path, tmp_path: Path) -> None:
    added = client.post("/ext/skill-config", json={"name": "review", "description": "Review code",
                                                   "skill_md": "---\nname: review\n---\nReview."})
    assert added.json() == {"name": "review"}

    _send(client, [{"kind": "text", "text": "go"}])

    argv = json.loads(record.read_text())["argv"]
    skill_dir = Path(argv[argv.index("--skill") + 1])
    assert skill_dir == tmp_path / "home" / "skills" / "review"
    assert (skill_dir / "SKILL.md").read_text().endswith("Review.")


def test_bundle_skill_is_downloaded(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"from {source.read.url}")

    monkeypatch.setattr(pi_agent, "download", fake_download)
    grant = {"media_type": "text/markdown", "max_bytes": 64, "size_bytes": 8,
             "read": {"kind": "http-get", "url": "https://objects.test/x", "expires_at": "2099-01-01T00:00:00Z"}}
    response = client.post("/ext/skill-config", json={
        "name": "lint", "description": "Lint",
        "skill_bundle": {"max_total_bytes": 128, "files": [
            {"path": "SKILL.md", "object": grant}, {"path": "scripts/run.sh", "object": grant},
        ]},
    })

    assert response.json() == {"name": "lint"}
    skill_dir = tmp_path / "home" / "skills" / "lint"
    assert (skill_dir / "scripts" / "run.sh").read_text() == "from https://objects.test/x"
    assert [path.name for path in skill_dir.parent.iterdir()] == ["lint"]


def test_model_error_is_an_infra_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_PI_SCENARIO", "error")

    task = _send(client, [{"kind": "text", "text": "go"}])

    text, data = _reply(task)
    assert task["status"]["state"] == "failed"
    assert text == "429 rate limited"
    assert (data["error_type"], data["error_code"]) == ("infra_error", "pi.model_error")
    assert data["usage"]["tool_call_count"] == 1


def test_crash_reports_stderr(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_PI_SCENARIO", "crash")

    text, data = _reply(_send(client, [{"kind": "text", "text": "go"}]))

    assert text == "boom: provider not configured"
    assert data["error_code"] == "pi.exited"


def test_timeout_kills_pi(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_PI_SCENARIO", "hang")
    client.post("/ext/agent-config", json={"timeout_seconds": 1})

    _, data = _reply(_send(client, [{"kind": "text", "text": "go"}]))

    assert data["error_code"] == "pi.timeout"


def test_missing_model_fails_without_running_pi(tmp_path: Path, record: Path) -> None:
    agent = PiAgent(home=tmp_path / "home", workspace=tmp_path / "ws", pi_command=(sys.executable, str(FAKE_PI)))
    with TestClient(agent.create_app()) as client:
        _, data = _reply(_send(client, [{"kind": "text", "text": "go"}]))

    assert data["error_code"] == "pi.no_model"
    assert not record.exists()


def test_model_params_reach_the_request_body(client: TestClient, record: Path) -> None:
    client.post("/ext/agent-config", json={"model_params": {"user": "project-1", "metadata": {"tags": ["t"]}}})

    _send(client, [{"kind": "text", "text": "go"}])

    model = json.loads(record.read_text())["models"]["providers"]["agentenv"]["models"][0]
    assert model["samplingParams"] == {"user": "project-1", "metadata": {"tags": ["t"]}}
    assert client.get("/ext/agent-config").json()["config"]["model_params"] == "***"


def test_registration_params_are_defaults_under_config_params(
    client: TestClient, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_A2A_MODEL_PARAMS", '{"user": "project-1", "temperature": 0.2}')
    client.post("/ext/agent-config", json={"model_params": {"temperature": 0.7}})

    _send(client, [{"kind": "text", "text": "go"}])

    model = json.loads(record.read_text())["models"]["providers"]["agentenv"]["models"][0]
    assert model["samplingParams"] == {"user": "project-1", "temperature": 0.7}


GRANT = {"kind": "http-put", "url": "https://objects.test/put", "expires_at": "2099-01-01T00:00:00Z"}
READ = {"kind": "http-get", "url": "https://objects.test/get", "expires_at": "2099-01-01T00:00:00Z"}


def _write(name: str) -> dict[str, Any]:
    return {"media_type": "application/octet-stream", "max_bytes": 1 << 20, "write": GRANT}


def _read(name: str) -> dict[str, Any]:
    return {"media_type": "application/octet-stream", "max_bytes": 1 << 20,
            "read": {**READ, "url": f"https://objects.test/{name}"}}


@pytest.fixture
def objects(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """An in-memory object store behind the transfer helpers the agent calls."""
    stored: dict[str, bytes] = {}
    names = iter(["trajectory", "workspace"])

    async def fake_upload(target, source):
        body = source.read_bytes() if isinstance(source, Path) else source
        stored[next(names)] = body
        return {"size_bytes": len(body)}

    async def fake_download(source, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(stored[source.read.url.rsplit("/", 1)[-1]])

    class FakeUploaded(dict):
        def model_dump(self, mode: str = "json") -> dict:
            return dict(self)

    async def uploaded(target, source):
        return FakeUploaded(await fake_upload(target, source))

    monkeypatch.setattr(pi_agent, "upload", uploaded)
    monkeypatch.setattr(pi_agent, "download", fake_download)
    return stored


def test_snapshot_round_trip_restores_conversation_and_workspace(
    client: TestClient, record: Path, tmp_path: Path, objects: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PI_WRITE", "notes.txt=remember TOKEN-1")
    _send(client, [{"kind": "text", "text": "remember TOKEN-1"}], context_id="source")
    saved = client.post("/ext/snapshot", json={"context_id": "source", "objects": {
        "trajectory": _write("trajectory"), "workspace": _write("workspace")}})
    assert saved.status_code == 200, saved.text
    assert saved.json()["context_id"] == "source"
    (tmp_path / "ws" / "notes.txt").unlink()
    monkeypatch.delenv("FAKE_PI_WRITE")

    loaded = client.put("/ext/snapshot", json={"objects": {
        "trajectory": _read("trajectory"), "workspace": _read("workspace")}, "target_context_id": "target"})
    _send(client, [{"kind": "text", "text": "what was the token?"}], context_id="target")

    assert loaded.json() == {"context_id": "target"}
    assert (tmp_path / "ws" / "notes.txt").read_text() == "remember TOKEN-1"
    invocation = json.loads(record.read_text())
    assert invocation["argv"][invocation["argv"].index("--session-id") + 1] == "target"
    assert any("remember TOKEN-1" in line for line in invocation["history"])


def test_snapshot_of_an_unknown_context_is_404(client: TestClient, objects: dict[str, bytes]) -> None:
    response = client.post("/ext/snapshot", json={"context_id": "nope", "objects": {"trajectory": _write("t")}})

    assert response.status_code == 404


class FakeNamespace:
    uploads: dict[str, bytes] = {}

    def __init__(self, grant) -> None:
        pass

    async def upload(self, relative_path: str, source: bytes) -> None:
        FakeNamespace.uploads[relative_path] = source


NAMESPACE = {"root_path": "agent_changelog/x", "expires_at": "2099-01-01T00:00:00Z", "max_objects": 10,
             "max_object_bytes": 1 << 20, "max_total_bytes": 1 << 24,
             "write": {"kind": "http-post-policy", "url": "https://objects.test/post", "fields": {},
                       "path_field": "key", "file_field": "file"}}


def test_changelog_captures_each_tool_call_and_replays_it(
    client: TestClient, record: Path, tmp_path: Path, objects: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeNamespace.uploads = {}
    monkeypatch.setattr(changelog, "NamespaceUploader", FakeNamespace)
    workspace = tmp_path / "ws"
    enabled = client.post("/ext/snapshot/changelog", json={"write_namespace": NAMESPACE, "roots": [str(workspace)]})
    monkeypatch.setenv("FAKE_PI_WRITE", "marker.txt=CHANGELOG-TOKEN")

    _send(client, [{"kind": "text", "text": "write the marker"}], context_id="capture")

    assert enabled.json() == {"roots": [str(workspace)]}
    assert list(FakeNamespace.uploads) == ["000000.tar"]
    with tarfile.open(fileobj=io.BytesIO(FakeNamespace.uploads["000000.tar"])) as archive:
        names = archive.getnames()
    assert "files/" + str(workspace / "marker.txt").lstrip("/") in names
    assert "session.jsonl" in names

    (workspace / "marker.txt").unlink()
    objects["000000"] = FakeNamespace.uploads["000000.tar"]
    monkeypatch.delenv("FAKE_PI_WRITE")
    applied = client.put("/ext/snapshot/changelog", json={
        "increments": [{"sequence": 0, "object": _read("000000")}],
        "resume_conversation": True, "target_context_id": "resumed"})
    _send(client, [{"kind": "text", "text": "continue"}], context_id="resumed")

    assert applied.json() == {"count": 1, "context_id": "resumed"}
    assert (workspace / "marker.txt").read_text() == "CHANGELOG-TOKEN"
    assert any("write the marker" in line for line in json.loads(record.read_text())["history"])


def test_changelog_increment_cannot_write_outside_its_roots(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, body in (("meta.json", json.dumps({"roots": [str(tmp_path / "ws")], "deleted": []}).encode()),
                           ("files/etc/passwd", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    (tmp_path / "evil.tar").write_bytes(buffer.getvalue())

    with pytest.raises(ValueError, match="outside its roots"):
        changelog.apply(tmp_path / "evil.tar")


def test_peers_register_the_loopback_mcp_server_and_relay_messages(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []

    def peer(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            raise httpx.ConnectError("refused", request=request)
        sent.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {
            "kind": "task", "status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": "PEER-OK"}]}}}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(peers.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(peer), **kwargs))
    client.post("/ext/peer-agents", json={"peers": [
        {"name": "helper", "url": "http://127.0.0.1:9100", "card": {"url": "/a2a"}, "description": "Helps"}]})

    listed = client.get("/ext/peer-agents").json()
    servers = client.get("/ext/mcp-config").json()["mcp_servers"]
    init = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                     "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    notified = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).json()["result"]["tools"]
    called = client.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "peer_send_message", "arguments": {"peer_name": "helper", "message": "say PEER-OK"}}}).json()
    again = client.post("/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
        "name": "peer_send_message", "arguments": {"peer_name": "helper", "message": "again"}}}).json()

    assert listed == {"peers": [{"name": "helper", "url": "http://127.0.0.1:9100", "description": "Helps"}]}
    assert servers["peers"]["url"] == "http://127.0.0.1:8000/mcp"
    assert init.json()["result"]["protocolVersion"] == "2025-06-18"
    assert notified.status_code == 202
    assert client.get("/mcp").status_code == 405
    assert [tool["name"] for tool in tools] == ["peer_list", "peer_send_message"]
    assert called["result"] == {"content": [{"type": "text", "text": "PEER-OK"}], "isError": False}
    assert sent[0]["url"] == "http://host.docker.internal:9100/a2a"
    first, second = (item["body"]["params"]["message"]["contextId"] for item in sent)
    assert first == second
    assert again["result"]["isError"] is False


def test_unknown_peer_is_a_tool_error(client: TestClient) -> None:
    called = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "peer_send_message", "arguments": {"peer_name": "ghost", "message": "hi"}}}).json()

    assert called["result"]["isError"] is True
    assert "unknown peer" in called["result"]["content"][0]["text"]


def test_install_extension_is_declared_on_the_card(client: TestClient) -> None:
    card = client.get("/.well-known/agent.json").json()
    install = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:install/v1")["params"]

    assert install["a2a_port"] == 8000
    assert set(install["required_params"]) == {"container", "agent_ctx_tar", "a2a_port", "litellm_api_key",
                                               "litellm_base_url"}
    formatted = [command.format(container="c", agent_ctx_tar="/t.tar", a2a_port="8000", litellm_api_key="k",
                                litellm_base_url="u") for command in install["install_commands"]]
    assert formatted[-1].endswith("c sh /opt/pi-a2a/start.sh")
    assert all(command.startswith("$(sudo -n true 2>/dev/null && echo sudo) docker ") for command in formatted)


def test_unattachable_files_are_left_on_disk_for_tools(client: TestClient, record: Path) -> None:
    encoded = base64.b64encode(b"ID3...").decode()
    _send(client, [{"kind": "text", "text": "transcribe"},
                   {"kind": "file", "file": {"name": "clip.mp3", "mimeType": "audio/mpeg", "bytes": encoded}}])

    invocation = json.loads(record.read_text())
    assert invocation["attachments"] == {}
    assert "[Attached file (audio/mpeg) saved at " in invocation["stdin"]


def test_a2a_server_exposes_the_card_install_reads() -> None:
    import a2a_server

    assert a2a_server.AGENT_CARD.name == "pi"
    assert any(e.uri == "urn:agentenv:install/v1" for e in a2a_server.AGENT_CARD.capabilities.extensions)


def test_registration_model_is_the_fallback(
    tmp_path: Path, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_A2A_MODEL", "fallback-model")
    agent = PiAgent(home=tmp_path / "home", workspace=tmp_path / "ws", pi_command=(sys.executable, str(FAKE_PI)))
    with TestClient(agent.create_app()) as client:
        task = _send(client, [{"kind": "text", "text": "go"}])

    argv = json.loads(record.read_text())["argv"]
    assert task["status"]["state"] == "completed"
    assert argv[argv.index("--model") + 1] == "fallback-model"
