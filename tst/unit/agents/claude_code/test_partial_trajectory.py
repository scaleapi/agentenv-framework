from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx
import pytest

from tst.unit.agents.claude_code.support import extension_methods, send
from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge

TRAJECTORY_PATH = a2a_server.TRAJECTORY_V1.endpoint


def _write_jsonl(path: Path, records: list[dict], trailing: bytes = b"") -> None:
    body = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    path.write_bytes(body + trailing)


def test_agent_card_advertises_one_get_trajectory_method(app):
    assert extension_methods(app, a2a_server.TRAJECTORY_V1.uri) == {
        "get": {
            "method": "POST",
            "request": {
                "oneOf": [
                    {"required": ["task_id"]},
                    {"required": ["task_id", "objects"]},
                    {"required": ["context_id"]},
                    {"required": ["context_id", "objects"]},
                    {"required": ["task_id", "after"], "optional": ["limit"]},
                    {"required": ["context_id", "after"], "optional": ["limit"]},
                ],
            },
        }
    }


def test_partial_trajectory_matches_stream_message_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    session_id = "session-1"
    path = tmp_path / f"{session_id}.jsonl"
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}],
    }
    tool_result_message = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}
        ],
    }
    _write_jsonl(
        path,
        [
            {"type": "queue-operation", "operation": "enqueue"},
            {"type": "user", "message": {"role": "user", "content": "prompt"}},
            {
                "type": "assistant",
                "message": assistant_message,
                "sessionId": session_id,
                "requestId": "request-1",
                "uuid": "assistant-1",
            },
            {
                "type": "user",
                "message": tool_result_message,
                "sessionId": session_id,
                "toolUseResult": {"stdout": "ok", "stderr": ""},
                "uuid": "result-1",
            },
        ],
    )

    trajectory, session_lines = claude_bridge.read_partial_session_trajectory(session_id)

    assert session_lines == 4
    assert trajectory == [
        {
            "type": "assistant",
            "message": assistant_message,
            "session_id": session_id,
            "request_id": "request-1",
            "uuid": "assistant-1",
        },
        {
            "type": "user",
            "message": tool_result_message,
            "session_id": session_id,
            "tool_use_result": {"stdout": "ok", "stderr": ""},
            "uuid": "result-1",
        },
    ]


def test_compaction_and_transcript_only_records_and_incomplete_tail_are_excluded(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    session_id = "session-2"
    path = tmp_path / f"{session_id}.jsonl"
    before = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "before"}]}}
    after = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "after"}]}}
    _write_jsonl(
        path,
        [
            before,
            {"type": "system", "subtype": "compact_boundary", "compactMetadata": {}},
            {
                "type": "assistant",
                "isCompactSummary": True,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "compaction summary"}],
                },
            },
            {
                "type": "assistant",
                "isVisibleInTranscriptOnly": True,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "UI-only message"}],
                },
            },
            after,
        ],
        trailing=b'{"type":"assistant","message":',
    )

    trajectory, session_lines = claude_bridge.read_partial_session_trajectory(session_id)

    assert session_lines == 5
    assert [event["message"] for event in trajectory] == [before["message"], after["message"]]


def test_context_trajectory_endpoint_reports_complete_live_session(monkeypatch, tmp_path, agent, client):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    agent.set_session_ref_for_context("ctx-1", "session-1")
    agent._active_contexts.add("ctx-1")
    previous_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "previous turn"}],
    }
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "working"}],
    }
    _write_jsonl(
        tmp_path / "session-1.jsonl",
        [
            {
                "type": "assistant",
                "message": previous_message,
                "sessionId": "session-1",
            },
            {"type": "assistant", "message": message, "sessionId": "session-1"},
        ],
    )

    result = client.post(TRAJECTORY_PATH, json={"context_id": "ctx-1"}).json()

    assert result == {
        "trajectory": [
            {
                "type": "assistant",
                "message": previous_message,
                "session_id": "session-1",
            },
            {
                "type": "assistant",
                "message": message,
                "session_id": "session-1",
            }
        ],
        "session_lines": 2,
        "is_live": True,
        "session_id": "session-1",
        "harness": "claude-code",
    }


def test_completed_task_returns_finalized_stream_trajectory(monkeypatch, client):
    trajectory = [{"type": "result", "result": "done"}]

    async def _run_claude(**kwargs):
        for event in trajectory:
            kwargs["event_callback"](event)
        return claude_bridge.ClaudeResult(response="done", tool_call_count=0, session_id=kwargs["session_id"], trajectory=trajectory)

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    task = send(client, "go")

    result = client.post(TRAJECTORY_PATH, json={"task_id": task["id"]}).json()
    live = client.post(TRAJECTORY_PATH, json={"task_id": task["id"], "after": 0}).json()

    assert result == {"trajectory": trajectory}
    assert (live["format"], live["state"], live["events"]) == (
        a2a_server.TRAJECTORY_FORMAT, "completed", trajectory,
    )


@pytest.mark.parametrize(
    "payload",
    [{}, {"task_id": "task-1", "context_id": "ctx-1"}],
)
def test_trajectory_endpoint_requires_exactly_one_identifier(client, payload):
    response = client.post(TRAJECTORY_PATH, json=payload)

    assert response.status_code == 400
    assert response.json()["detail"].startswith("request must match exactly one of variants")


def test_task_trajectory_is_unavailable_until_turn_completes(client):
    response = client.post(TRAJECTORY_PATH, json={"task_id": "task-1"})

    assert response.status_code == 404
    assert response.json() == {"detail": "No completed trajectory available for task 'task-1'"}


def test_partial_trajectory_endpoint_maps_missing_jsonl_to_first_tick_404(
    monkeypatch, tmp_path, agent, client
):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    agent.set_session_ref_for_context("ctx-1", "session-1")

    response = client.post(TRAJECTORY_PATH, json={"context_id": "ctx-1"})

    assert response.status_code == 404
    assert response.json() == {"detail": "No transcript yet for context_id 'ctx-1'"}


def test_a_context_without_a_session_is_a_404(client):
    response = client.post(TRAJECTORY_PATH, json={"context_id": "ctx-unknown"})

    assert response.status_code == 404
    assert response.json() == {"detail": "No session available for context 'ctx-unknown'"}


_INIT = {"type": "system", "subtype": "init", "session_id": "sess-1"}
_TOOL = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}}
_DONE = {"type": "result", "subtype": "success", "result": "done", "session_id": "sess-1"}


def _lines(*events: dict) -> bytes:
    return b"".join(json.dumps(event).encode() + b"\n" for event in events)


def test_a_turn_is_readable_while_the_cli_runs_and_its_log_is_its_trajectory(app, monkeypatch, tmp_path):
    """Driven over HTTP, through the real bridge: the CLI prints one event, then blocks until released."""
    started: list[FakeCLI] = []

    async def spawn(*cmd, **kwargs):
        started.append(FakeCLI(_lines(_INIT), hang=True))
        return started[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://claude") as client:
            message = {
                "kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
                "parts": [{"kind": "text", "text": "go"}],
            }
            sent = await client.post("/a2a", json={
                "jsonrpc": "2.0", "id": 1, "method": "message/send",
                "params": {"message": message, "configuration": {"blocking": False}},
            })
            task_id = sent.json()["result"]["id"]

            async def read(after: int) -> dict:
                return (await client.post(TRAJECTORY_PATH, json={"task_id": task_id, "after": after})).json()

            async with asyncio.timeout(5):
                while (running := await read(0))["next"] < 1:
                    await asyncio.sleep(0.01)
            cli = started[0]
            cli.stdout.feed_data(_lines(_TOOL, _DONE))
            cli.stdout.feed_eof()
            cli.stderr.feed_eof()
            async with asyncio.timeout(5):
                while (ended := await read(running["next"]))["state"] != "completed":
                    await asyncio.sleep(0.01)
            final = (await client.post(TRAJECTORY_PATH, json={"task_id": task_id})).json()
            return running, ended, final

    running, ended, final = asyncio.run(exercise())

    assert (running["state"], running["format"], running["events"]) == (
        "running", a2a_server.TRAJECTORY_FORMAT, [_INIT],
    )
    assert (ended["events"], ended["next"], ended["has_more"]) == ([_TOOL, _DONE], 3, False)
    assert final == {"trajectory": [_INIT, _TOOL, _DONE]}
