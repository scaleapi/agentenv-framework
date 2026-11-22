"""What the bridge feeds Claude Code's ATIF adapter beside stdout: each turn's new transcript records, its sub-agents'
included, trimmed to what the adapter takes and redacted where an API error could echo a key, and each resumed
attempt's error `result`; only for a task that builds an atif/1 copy."""
from __future__ import annotations

import asyncio
import json

import pytest

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import send, terminal
from tst.unit.agents.claude_code.test_claude_bridge_retry import _api_error_stdout, _event, scripted  # noqa: F401

KEY = "sk-" + "Q" * 24


def _assistant(uuid, text="hi", msg="msg-1", **extra):
    return {"type": "assistant", "uuid": uuid, "timestamp": "2026-11-22T02:01:00Z", "sessionId": "sess-1",
            "parentUuid": None, "requestId": "req-1", "cwd": "/app", "version": "2.1.322",
            "message": {"id": msg, "model": "m", "role": "assistant", "content": [{"type": "text", "text": text}],
                        "usage": {"input_tokens": 2, "output_tokens": 7}, "stop_reason": "end_turn"}, **extra}


def _shape(digest="h1", **extra):
    return {"type": "api-request-shape", "shapeHash": digest, "timestamp": "2026-11-22T02:01:00Z",
            "shape": {"system": [{"type": "text", "text": "the system prompt"}], "tools": [{"name": "Bash"}]}, **extra}


def _lines(*records) -> bytes:
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def test_a_message_stdout_carried_keeps_its_ids_time_and_usage_without_its_content():
    name, fed = claude_bridge.session_aux(_assistant("u1"), None, {"u1"}, set())
    assert name == "session" and fed == {
        "type": "assistant", "uuid": "u1", "timestamp": "2026-11-22T02:01:00Z",
        "message": {"id": "msg-1", "model": "m", "usage": {"input_tokens": 2, "output_tokens": 7},
                    "stop_reason": "end_turn"}}


def test_a_message_stdout_hasnt_carried_keeps_its_content_and_a_sub_agents_record_names_its_agent():
    _, fed = claude_bridge.session_aux(_assistant("u1", isSidechain=True), "a7f9", set(), set())
    assert fed["message"]["content"] == [{"type": "text", "text": "hi"}]
    assert (fed["agentId"], fed["isSidechain"]) == ("a7f9", True)


def test_a_tool_result_record_is_fed_without_its_content_or_its_second_copy():
    record = {"type": "user", "uuid": "u2", "timestamp": "2026-11-22T02:01:01Z", "toolUseResult": {"stdout": "x"},
              "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "x"}]}}
    assert claude_bridge.session_aux(record, None, set(), set()) == ("session", {
        "type": "user", "uuid": "u2", "timestamp": "2026-11-22T02:01:01Z", "message": {}})


def test_an_api_error_record_is_redacted_before_it_is_fed():
    record = _assistant("u3", text=f"API Error: 401 key {KEY} invalid", msg="m-err", isApiErrorMessage=True)
    _, fed = claude_bridge.session_aux(record, None, set(), set())
    assert KEY not in json.dumps(fed) and fed["isApiErrorMessage"] is True


def test_a_request_shape_is_fed_once_per_agent_with_its_tools_alone():
    shapes: set = set()
    first = claude_bridge.session_aux(_shape(), None, set(), shapes)
    again = claude_bridge.session_aux(_shape(), None, set(), shapes)
    sub = claude_bridge.session_aux(_shape(), "a7f9", set(), shapes)
    assert first == ("tools", {"type": "api-request-shape", "shapeHash": "h1", "tools": [{"name": "Bash"}]})
    assert again is None and sub[1]["agentId"] == "a7f9"


@pytest.mark.parametrize("kind", ["attachment", "api-request", "api-request-blob", "queue-operation", "cost-state"])
def test_bookkeeping_records_are_not_fed(kind):
    assert claude_bridge.session_aux({"type": kind, "uuid": "u"}, None, set(), set()) is None


def test_the_tail_starts_where_the_files_end_and_waits_for_a_whole_line(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_bytes(_lines(_assistant("old")))
    tail = claude_bridge.SessionTail("sess-1", set(), set())
    with transcript.open("ab") as file:
        file.write(_lines(_assistant("new")) + b'{"type": "assist')
    records, files, shapes = tail.read()
    assert [record["uuid"] for _, record in records] == ["new"]
    assert tail.read()[0] == records
    tail.advance(files, shapes)
    with transcript.open("ab") as file:
        file.write(b'ant", "uuid": "torn"}\n')
    assert [record["uuid"] for _, record in tail.read()[0]] == ["torn"]


def test_the_tail_takes_sub_agent_transcripts_new_or_old_by_their_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    (tmp_path / "sess-1.jsonl").write_bytes(b"")
    subagents = tmp_path / "sess-1" / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-old.jsonl").write_bytes(_lines(_assistant("before")))
    (subagents / "agent-old.meta.json").write_text("{}")
    tail = claude_bridge.SessionTail("sess-1", set(), set())
    with (subagents / "agent-old.jsonl").open("ab") as file:
        file.write(_lines(_assistant("later")))
    (subagents / "agent-new.jsonl").write_bytes(_lines(_shape("h2"), _assistant("first")))
    fed = tail.read()[0]
    assert [(name, record.get("uuid"), record.get("agentId")) for name, record in fed] == [
        ("tools", None, "new"), ("session", "first", "new"), ("session", "later", "old")]


def test_a_transcript_the_scrub_replaced_is_followed_by_its_line_count(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    transcript = tmp_path / "sess-1.jsonl"
    leaked = _assistant("err", text=f"API Error: {KEY}" + "!" * 200, isApiErrorMessage=True)
    transcript.write_bytes(_lines(leaked))
    tail = claude_bridge.SessionTail("sess-1", set(), set())
    claude_bridge.scrub_session_transcript("sess-1")
    with transcript.open("ab") as file:
        file.write(_lines(_assistant("after")))
    assert [record["uuid"] for _, record in tail.read()[0]] == ["after"]


def _transcript_writing(scripted, per_spawn):  # noqa: F811
    """Each scripted CLI, as it starts, appends its records to the session transcript, as the real one does."""
    original = asyncio.create_subprocess_exec
    spawns = iter(per_spawn)

    async def _exec(*cmd, **kwargs):
        with (scripted["session_dir"] / "sess-1.jsonl").open("ab") as file:
            file.write(_lines(*next(spawns)))
        return await original(*cmd, **kwargs)

    return _exec


def test_run_claude_feeds_a_turns_transcript_records_and_each_resumed_attempts_error(scripted, monkeypatch):  # noqa: F811
    (scripted["session_dir"] / "sess-1.jsonl").write_bytes(_lines(_assistant("earlier-turn")))
    failure = _api_error_stdout("API Error: The operation timed out.")
    scripted["queue"].extend([(failure, 1), (_event(type="assistant", uuid="u2", message={
        "id": "msg-2", "content": []}).encode() + _event(type="result", subtype="success", result="done").encode(), 0)])
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        _transcript_writing(scripted, [[_assistant("u1"), _shape()], [_assistant("u2", msg="msg-2")]]))
    fed: list = []
    events: list = []
    result = asyncio.run(claude_bridge.run_claude(
        prompt="p", session_id="sess-1", resume=True, event_callback=lambda e: events.append(("stdout", e)),
        aux_callback=lambda name, record: events.append((name, record)), tool_shapes=(shapes := set())))
    fed = [(name, record.get("uuid") or record.get("subtype")) for name, record in events if name != "stdout"]
    assert result.attempts == 2
    assert fed == [("session", "u1"), ("tools", None), ("attempt", "success"), ("session", "u2")]
    assert next(record for name, record in events if name == "attempt")["is_error"] is True
    assert shapes == {(None, "h1")}
    assert [name for name, _ in events][-1] == "stdout"


def test_run_claude_without_an_aux_callback_reads_no_transcript(scripted, monkeypatch):  # noqa: F811
    scripted["queue"].append((_event(type="result", subtype="success", result="done").encode(), 0))
    monkeypatch.setattr(claude_bridge, "SessionTail", None)
    assert asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False)).response == "done"


@pytest.fixture
def turns(monkeypatch):
    seen: list[dict] = []

    async def _run_claude(**kwargs):
        seen.append(kwargs)
        return claude_bridge.ClaudeResult(response="done", tool_call_count=0, session_id=kwargs["session_id"],
                                          run_usage=[{"input_tokens": 1, "output_tokens": 1}],
                                          model_usage={"m": {"inputTokens": 1, "outputTokens": 1, "costUSD": 0.25}},
                                          session_base={})

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    return seen


def test_a_task_that_builds_no_atif_copy_feeds_nothing_beside_stdout(turns, client):
    client.post("/ext/trajectory/prescribe", json={"formats": []}).raise_for_status()
    assert terminal(send(client, "hello"))[0] == "completed"
    assert "aux_callback" not in turns[0] and "tool_shapes" not in turns[0]


def test_a_task_that_builds_an_atif_copy_is_fed_its_records_and_the_turns_cost(turns, client):
    client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]}).raise_for_status()
    task = send(client, "hello", context_id="ctx-a")
    assert terminal(task)[0] == "completed"
    assert callable(turns[0]["aux_callback"]) and turns[0]["tool_shapes"] == set()
    read = client.post("/ext/trajectory", json={"task_id": task["id"], "format": "atif/1"}).json()
    assert {"name": "turn", "record": {"type": "turn", "cost_usd": 0.25}} in [
        {key: entry[key] for key in ("name", "record")} for entry in read["aux"]]
    assert read["trajectory"]["final_metrics"]["total_cost_usd"] == 0.25


def test_a_transcript_rewritten_under_the_same_inode_is_followed_by_its_line_count(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    transcript = tmp_path / "sess-1.jsonl"
    leaked = _assistant("err", text=f"API Error: {KEY}" + "!" * 200, isApiErrorMessage=True)
    transcript.write_bytes(_lines(leaked, _assistant("seen")))
    tail = claude_bridge.SessionTail("sess-1", set(), set())
    inode = transcript.stat().st_ino
    with transcript.open("r+b") as file:
        file.write(_lines(_assistant("err", text="API Error: [redacted]", isApiErrorMessage=True),
                          _assistant("seen"), _assistant("after")))
        file.truncate()
    assert transcript.stat().st_ino == inode
    assert [record["uuid"] for _, record in tail.read()[0]] == ["after"]


def test_a_new_document_is_fed_the_latest_shape_its_transcript_already_held(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    (tmp_path / "sess-1.jsonl").write_bytes(_lines(_shape("h1"), _assistant("old"), _shape("h2")))
    shapes = {(None, "h2")}
    plain = claude_bridge.SessionTail("sess-1", set(), set(shapes))
    fresh = claude_bridge.SessionTail("sess-1", set(), set(), new_document=True)
    records, files, seen = fresh.read()
    assert plain.read()[0] == []
    assert [(name, record["shapeHash"]) for name, record in records] == [("tools", "h2")]
    fresh.advance(files, seen)
    assert fresh.read()[0] == []


def test_a_task_that_opens_a_new_document_feeds_its_tools_again(turns, client, monkeypatch):
    client.post("/ext/trajectory/prescribe", json={"formats": ["atif/1"]}).raise_for_status()
    send(client, "one", context_id="ctx-b")
    turns[0]["tool_shapes"].add((None, "h1"))
    send(client, "two", context_id="ctx-b")
    agent = client.app.state.agentenv_a2a.services.task_trajectories
    agent.restored("ctx-b", 2)
    send(client, "three", context_id="ctx-b")
    assert [call["new_document"] for call in turns] == [True, False, True]
    assert turns[2]["tool_shapes"] == set()
