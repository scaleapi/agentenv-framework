"""The session tail that feeds Gemini's atif/1 adapter (``gemini_transcript``): each turn's new model messages from the transcript
the CLI rewrites whole, each once a later one follows it and the last at the turn's end, and ``run_gemini`` feeding
them while the CLI runs and the rest once it has exited."""
import asyncio
import functools
import json
import os

import pytest

from tst.unit.agents.gemini.agent import gemini_bridge, gemini_transcript
from tst.unit.agents.gemini.fake_cli import FakeCLI
from tst.unit.agents.gemini.support import chats

SESSION = "97b4a411-6d52-4c55-9e61-1b2c3d4e5f60"


def tail_for(session):
    """The tail a turn on ``session`` (None for a new one) follows, over the transcripts the home fixture puts in place."""
    return gemini_transcript.SessionTail(session, functools.partial(gemini_bridge.find_session_file,
                                                                    cwd=os.environ["GEMINI_CWD"]))


def user(message_id):
    return {"id": message_id, "timestamp": "2026-11-25T16:24:25.449Z", "type": "user", "content": [{"text": "go"}]}


def model(message_id, text=""):
    return {"id": message_id, "timestamp": "2026-11-25T16:24:27.358Z", "type": "gemini", "content": text,
            "tokens": {"input": 7338, "output": 77}, "model": "gemini-2.5-flash",
            "toolCalls": [{"id": "write_1", "name": "write_file", "args": {"file_path": "hello.py"}, "status": "success",
                           "timestamp": "2026-11-25T16:24:27.367Z",
                           "result": [{"functionResponse": {"id": "write_1", "response": {"output": "ok"}}}],
                           "resultDisplay": {"fileDiff": "+print(1)"}, "displayName": "WriteFile",
                           "description": "hello.py", "renderOutputAsMarkdown": True}]}


def write(home, *messages, raw=None):
    path = chats(home) / f"session-2026-11-25T16-24-{SESSION[:8]}.json"
    path.write_text(raw if raw is not None else json.dumps({"sessionId": SESSION, "messages": list(messages)}))
    return path


def ids(records):
    return [record["id"] for record in records]


def test_a_message_is_fed_once_a_later_one_follows_it_and_the_last_at_the_turns_end(home):
    tail = tail_for(None)
    tail.bind(SESSION)
    write(home, user("u-1"), model("g-1"))
    assert tail.read() == []

    write(home, user("u-1"), model("g-1"), model("g-2", "5050"))
    assert ids(tail.read()) == ["g-1"]
    assert ids(tail.read(final=True)) == ["g-2"]
    assert tail.read(final=True) == []


def test_a_resumed_sessions_earlier_messages_are_not_fed_again(home):
    write(home, user("u-1"), model("g-1"), model("g-2"))
    tail = tail_for(SESSION)
    write(home, user("u-1"), model("g-1"), model("g-2"), user("u-2"), model("g-3"))

    assert ids(tail.read(final=True)) == ["g-3"]


def test_a_transcript_the_cli_is_still_writing_is_read_at_the_next_poll(home):
    tail = tail_for(None)
    tail.bind(SESSION)
    whole = json.dumps({"sessionId": SESSION, "messages": [user("u-1"), model("g-1")]})
    write(home, raw=whole[:len(whole) // 2])
    assert tail.read(final=True) == []

    write(home, raw=whole)
    assert ids(tail.read(final=True)) == ["g-1"]


def test_a_model_message_keeps_what_the_adapter_reads_and_drops_how_the_cli_displays_it():
    record = gemini_transcript.session_aux(model("g-1", "done"))

    assert record == {"type": "gemini", "id": "g-1", "timestamp": "2026-11-25T16:24:27.358Z", "content": "done",
                      "tokens": {"input": 7338, "output": 77}, "model": "gemini-2.5-flash",
                      "toolCalls": [{"id": "write_1", "name": "write_file", "args": {"file_path": "hello.py"},
                                     "status": "success", "timestamp": "2026-11-25T16:24:27.367Z",
                                     "result": [{"functionResponse": {"id": "write_1",
                                                                      "response": {"output": "ok"}}}]}]}
    assert gemini_transcript.session_aux(user("u-1")) is None


@pytest.fixture
def cli(monkeypatch):
    queue: list = []

    async def _exec(*cmd, **kwargs):
        return queue.pop(0)()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    return queue


def test_run_gemini_feeds_the_turns_messages_while_the_cli_runs_and_the_last_once_it_exits(home, cli, monkeypatch):
    monkeypatch.setattr(gemini_bridge, "SESSION_POLL_S", 0.01)
    fed: list[tuple[str, dict]] = []

    async def turn():
        proc = FakeCLI(hang=True)
        cli.append(lambda: proc)
        running = asyncio.create_task(gemini_bridge.run_gemini(
            prompt="go", session_id=None, cwd=os.environ["GEMINI_CWD"],
            aux_callback=lambda name, record: fed.append((name, record))))
        proc.stdout.feed_data((json.dumps({"type": "init", "session_id": SESSION}) + "\n").encode())
        write(home, user("u-1"), model("g-1"), model("g-2", "5050"))
        async with asyncio.timeout(5):
            while not fed:
                await asyncio.sleep(0.01)
        seen_while_running = ids(record for _, record in fed)
        proc.stdout.feed_eof()
        proc.stderr.feed_eof()
        await running
        return seen_while_running

    seen_while_running = asyncio.run(turn())

    assert seen_while_running == ["g-1"]
    assert fed and {name for name, _ in fed} == {"session"}
    assert ids(record for _, record in fed) == ["g-1", "g-2"]


def write_jsonl(home, *messages):
    path = chats(home) / f"session-2026-11-25T16-24-{SESSION[:8]}.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in ({"sessionId": SESSION, "projectHash": "p"},
                                                                  *messages)))
    return path


def test_a_restored_jsonl_sessions_new_messages_are_fed_as_their_latest_copy_and_its_earlier_ones_not_again(home):
    path = write_jsonl(home, user("u-1"), model("g-1"))
    tail = tail_for(SESSION)
    first = {**model("g-2"), "toolCalls": []}
    write_jsonl(home, user("u-1"), model("g-1"), user("u-2"), first)
    assert tail.read() == []

    with path.open("a") as file:
        file.write(json.dumps(model("g-2")) + "\n" + json.dumps(model("g-3", "done")) + "\n" + '{"id": "g-4", "ty')
    assert ids(tail.read()) == ["g-2"]
    assert tail.read(final=True)[0]["id"] == "g-3"


def test_a_settled_message_written_again_in_a_jsonl_transcript_is_fed_as_its_latest_copy(home):
    tail = tail_for(None)
    tail.bind(SESSION)
    write_jsonl(home, user("u-1"), {**model("g-1"), "toolCalls": []}, model("g-1"), model("g-2", "5050"))

    (fed,) = tail.read()
    assert fed["id"] == "g-1" and fed["toolCalls"][0]["result"]


def test_a_transcript_moved_from_json_to_jsonl_is_found_again_and_nothing_is_fed_twice(home):
    tail = tail_for(None)
    tail.bind(SESSION)
    legacy = write(home, user("u-1"), model("g-1"), model("g-2"))
    assert ids(tail.read()) == ["g-1"]

    legacy.unlink()
    write_jsonl(home, user("u-1"), model("g-1"), model("g-2"), model("g-3"))
    assert ids(tail.read(final=True)) == ["g-2", "g-3"]
