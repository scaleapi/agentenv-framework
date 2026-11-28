"""The rollout tail that feeds Codex's atif/1 adapter (``codex_rollout``): each turn's new rollout records, its
thread's and its sub-agents', as the CLI appends them, and ``run_codex`` feeding them while the CLI runs and the rest
once it has exited."""
import asyncio
import json
import os

import pytest

from tst.unit.agents.codex.agent import codex_bridge, codex_rollout

MAIN = "01a21385-2388-7b60-adb8-331c882b4e9a"
SUB = "01a21385-2cea-7b83-9a89-8cd36b7ffe19"
DEEPER = "01a21385-3fff-7b83-9a89-8cd36b7ffe20"


@pytest.fixture
def sessions(monkeypatch, tmp_path):
    folder = tmp_path / "sessions" / "2026" / "11" / "25"
    folder.mkdir(parents=True)
    monkeypatch.setattr(codex_bridge, "SESSIONS_DIR", tmp_path / "sessions")
    return folder


def tail_for(thread):
    """The tail a turn on ``thread`` (None for a new one) follows, over the rollouts the fixture puts in place."""
    return codex_rollout.RolloutTail(thread, codex_bridge.SESSIONS_DIR, codex_bridge.find_rollout_path)


def meta(thread, parent=None):
    return {"timestamp": "2026-11-25T16:23:42.479Z", "type": "session_meta",
            "payload": {"id": thread, "parent_thread_id": parent, "agent_nickname": "Mendel" if parent else None,
                        "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent, "agent_path": "/root/wc"}}}
                        if parent else "exec", "base_instructions": {"text": "You are Codex."}}}


def item(text, at="2026-11-25T16:23:43.000Z"):
    return {"timestamp": at, "type": "response_item",
            "payload": {"type": "message", "id": f"msg_{text}", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}


def write(path, *records, partial=""):
    with path.open("a") as file:
        file.writelines(json.dumps(record) + "\n" for record in records)
        file.write(partial)


def texts(records):
    return [record["payload"]["content"][0]["text"] for record in records if record["type"] == "response_item"]


def rollout(folder, thread):
    return folder / f"rollout-2026-11-25T16-23-42-{thread}.jsonl"


def test_a_new_threads_rollout_is_read_from_its_start_and_a_resumed_ones_from_where_it_ended(sessions):
    path = rollout(sessions, MAIN)
    write(path, meta(MAIN), item("one"))
    tail = tail_for(None)
    tail.bind(MAIN)

    first = tail.read()
    assert [record["type"] for record in first] == ["session_meta", "response_item"]
    assert all(record["thread"] == MAIN and "parent" not in record for record in first)

    resumed = tail_for(MAIN)
    assert resumed.read() == []
    write(path, item("two"))
    assert texts(resumed.read()) == ["two"]


def test_a_line_the_cli_is_still_writing_waits_for_its_end(sessions):
    path = rollout(sessions, MAIN)
    write(path, meta(MAIN), partial=json.dumps(item("half"))[:30])
    tail = tail_for(None)
    tail.bind(MAIN)

    assert texts(tail.read()) == []
    write(path, partial=json.dumps(item("half"))[30:] + "\n")
    assert texts(tail.read()) == ["half"]


def test_sub_agents_rollouts_are_found_at_any_depth_and_marked_with_their_parent(sessions):
    write(rollout(sessions, MAIN), meta(MAIN))
    tail = tail_for(None)
    tail.bind(MAIN)
    tail.read()
    write(rollout(sessions, SUB), meta(SUB, MAIN), item("sub"))
    write(rollout(sessions, DEEPER), meta(DEEPER, SUB), item("deeper"))
    write(rollout(sessions, "01a21385-ffff-7b83-9a89-000000000000"), meta("other"), item("someone else's"))

    records = tail.read(final=True)
    assert sorted(texts(records)) == ["deeper", "sub"]
    assert {(record["thread"], record.get("parent")) for record in records} == {(SUB, MAIN), (DEEPER, SUB)}
    (sub_meta,) = [record["payload"] for record in records if record["type"] == "session_meta"
                   and record["thread"] == SUB]
    assert sub_meta == {"id": SUB, "parent_thread_id": MAIN, "agent_path": "/root/wc", "agent_nickname": "Mendel"}


def test_a_sub_agents_rollout_seen_before_its_parents_is_followed_once_its_parent_is(sessions):
    write(rollout(sessions, MAIN), meta(MAIN))
    tail = tail_for(None)
    tail.bind(MAIN)
    write(rollout(sessions, DEEPER), meta(DEEPER, SUB), item("deeper"))
    assert texts(tail.read(final=True)) == []

    write(rollout(sessions, SUB), meta(SUB, MAIN), item("sub"))
    assert sorted(texts(tail.read(final=True))) == ["deeper", "sub"]


def test_a_forked_sub_agents_rollout_leaves_out_what_it_copied_from_its_parent(sessions):
    def started(turn):
        return {"timestamp": "2026-11-25T16:23:42.500Z", "type": "event_msg",
                "payload": {"type": "task_started", "turn_id": turn}}

    def call(call_id, turn):
        return {"timestamp": "2026-11-25T16:23:43.000Z", "type": "response_item",
                "payload": {"type": "function_call", "id": f"fc_{call_id}", "name": "exec_command", "call_id": call_id,
                            "arguments": "{}", "internal_chat_message_metadata_passthrough": {"turn_id": turn}}}

    reply = {**item("one"), "payload": {**item("one")["payload"],
                                        "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"}}}
    write(rollout(sessions, MAIN), meta(MAIN), started("turn-1"), call("call_1", "turn-1"), reply, started("turn-2"))
    tail = tail_for(None)
    tail.bind(MAIN)
    tail.read()
    forked = {**meta(SUB, MAIN), "payload": {**meta(SUB, MAIN)["payload"], "forked_from_id": MAIN}}
    write(rollout(sessions, SUB), forked, meta(MAIN), started("turn-1"), call("call_1", "turn-1"), reply,
          started("turn-2"), started("turn-s"), call("call_s", "turn-s"))

    records = tail.read(final=True)
    assert [record["type"] for record in records] == ["session_meta", "event_msg", "response_item"]
    assert records[0]["payload"]["id"] == SUB
    assert [record["payload"].get("call_id") for record in records] == [None, None, "call_s"]


def test_a_replaced_rollout_is_read_on_from_its_line_count(sessions):
    path = rollout(sessions, MAIN)
    write(path, meta(MAIN), item("one"))
    tail = tail_for(MAIN)
    replacement = path.with_suffix(".tmp")
    write(replacement, meta(MAIN), item("one"), item("two"))
    os.replace(replacement, path)

    assert texts(tail.read()) == ["two"]


@pytest.mark.parametrize("record", [
    {"type": "world_state", "payload": {"full": True}},
    {"type": "response_item", "payload": {"type": "message", "role": "developer", "content": []}},
    {"type": "response_item", "payload": {"type": "agent_message", "author": "/root/wc", "content": []}},
    {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
    {"type": "event_msg", "payload": {"type": "item_completed"}},
])
def test_records_the_adapter_doesnt_read_are_left_out(record):
    assert codex_rollout.rollout_aux({"timestamp": "2026-11-25T16:23:43.000Z", **record}, MAIN) is None


def test_a_usage_record_keeps_its_response_and_its_counts_alone():
    usage = {"input_tokens": 13350, "cached_input_tokens": 13184, "output_tokens": 90}
    fed = codex_rollout.rollout_aux({"timestamp": "2026-11-25T16:23:44.676Z", "type": "token_usage_record",
                                    "payload": {"thread_id": MAIN, "response_id": "resp_1", "usage": usage,
                                                "turn_token_usage": usage, "thread_token_usage": usage}}, MAIN)

    assert fed == {"type": "token_usage_record", "timestamp": "2026-11-25T16:23:44.676Z", "thread": MAIN,
                   "payload": {"response_id": "resp_1", "usage": usage}}


class _Stream:
    def __init__(self, lines, before_each=None):
        self.lines, self.before_each = list(lines), before_each

    async def read(self, n=-1):
        if not self.lines:
            return b""
        line = self.lines.pop(0)
        if self.before_each is not None:
            await self.before_each(line)
        return line


class _Proc:
    def __init__(self, stdout):
        self.stdin = type("Stdin", (), {"write": lambda self, data: None, "drain": lambda self: asyncio.sleep(0),
                                        "close": lambda self: None})()
        self.stdout, self.stderr, self.returncode = stdout, _Stream([]), 0

    async def wait(self):
        return 0

    def kill(self):
        self.returncode = -9


def test_run_codex_feeds_the_turns_rollout_while_the_cli_runs_and_the_rest_once_it_exits(sessions, monkeypatch):
    path = rollout(sessions, MAIN)
    fed: list[tuple[str, dict]] = []
    seen_before_exit: list[int] = []

    async def written(line):
        event = json.loads(line)
        if event["type"] == "thread.started":
            write(path, meta(MAIN), item("one"))
        elif event["type"] == "turn.completed":
            await asyncio.sleep(codex_rollout.ROLLOUT_POLL_S * 4)
            seen_before_exit.append(len(fed))
            write(path, item("two"))

    events = [{"type": "thread.started", "thread_id": MAIN}, {"type": "turn.started"},
              {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "two"}},
              {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}]

    async def exec_(*cmd, **kwargs):
        return _Proc(_Stream([(json.dumps(event) + "\n").encode() for event in events], written))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_)
    monkeypatch.chdir(sessions)
    result = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None,
                                                aux_callback=lambda name, record: fed.append((name, record))))

    assert result.error is None and result.response == "two"
    assert {name for name, _ in fed} == {"rollout"}
    assert texts([record for _, record in fed]) == ["one", "two"]
    assert seen_before_exit == [2]
