import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

import agent_env

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "browser-use"
# By its path: another agent's a2a_server, imported by the same name, may be loaded already.
_spec = importlib.util.spec_from_file_location("browser_use_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = a2a_server
_spec.loader.exec_module(a2a_server)

FAKE_AGENT = Path(__file__).with_name("fake_agent.py")


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(a2a_server, "HOME", tmp_path / "home")
    monkeypatch.setenv("FAKE_TURN_RECORD", str(tmp_path / "turn.json"))
    return tmp_path / "turn.json"


@pytest.fixture
def client(record: Path):
    agent = a2a_server.BrowserUseAgent(command=(sys.executable, str(FAKE_AGENT)))
    with TestClient(agent.create_app()) as client:
        client.post("/ext/agent-config", json={"model": "gpt-test", "effort": "low", "system_prompt": "Be brief."})
        yield client


def _send(client: TestClient, text: str, context_id: str = "ctx-1") -> dict[str, Any]:
    response = client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
        "message": {"kind": "message", "messageId": f"m-{text}", "role": "user", "contextId": context_id,
                    "parts": [{"kind": "text", "text": text}]},
        "configuration": {"blocking": True},
    }})
    return response.json()["result"]


def _reply(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    parts = task["status"]["message"]["parts"]
    return "".join(p.get("text", "") for p in parts if p["kind"] == "text"), next(p["data"] for p in parts if p["kind"] == "data")


def test_a_turn_returns_the_answer_and_a_context_keeps_one_conversation(client: TestClient, record: Path) -> None:
    task = _send(client, "top story?")
    text, data = _reply(task)
    assert (task["status"]["state"], text) == ("completed", "The top story.")
    assert data["usage"]["tool_call_count"] == 1 and data["usage"]["total_tokens"] == 120
    turn = json.loads(record.read_text())
    assert (turn["prompt"], turn["model"], turn["effort"], turn["systemPrompt"]) == ("top story?", "gpt-test", "low", "Be brief.")
    events = client.post("/ext/trajectory", json={"task_id": task["id"], "after": 0}).json()["events"]
    assert [e["type"] for e in events] == ["agent_start", "tool_execution_end", "agent_end"]

    _send(client, "and the second?")
    assert json.loads(record.read_text())["historyFile"] == turn["historyFile"]
    _send(client, "other context", context_id="ctx-2")
    assert json.loads(record.read_text())["historyFile"] != turn["historyFile"]


def test_a_run_that_stops_early_fails_with_its_reason(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "max_steps")
    task = _send(client, "too long")
    assert task["status"]["state"] == "failed"
    assert _reply(task)[1]["error_code"] == "browser_use.max_steps"


def test_a_crash_is_an_infra_error_with_its_stderr(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "crash")
    data = _reply(_send(client, "hi"))[1]
    assert (data["error_type"], data["error_code"]) == ("infra_error", "browser_use.crashed")
    assert "401" in data["error_message"]


def test_a_turn_past_its_deadline_is_killed(client: TestClient, record: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "hang")
    client.post("/ext/agent-config", json={"timeout_seconds": 1})
    assert _reply(_send(client, "hi"))[1]["error_code"] == "browser_use.timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(json.loads(record.read_text())["pid"], 0)
