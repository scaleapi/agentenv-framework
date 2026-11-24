"""A prompt step of a run that prescribes atif/1 follows each turn's atif/1 copy live: each poll's new closed steps a
page, the document's header beside the page where it changed, the media the steps name pulled as they come, and the
pages a prefix of the turn's stored copy; a turn the agent won't serve atif/1 is followed natively, and one it stops
serving partway ends with the reason. Against a live SDK agent served in-process."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from agentenv_protocol.a2a_agent import TrajectoryState
from agentenv_protocol.atif import FORMAT as ATIF, media_paths, tasks

from agent_env.a2a_agent.trajectory_follower import TrajectoryBatch, follow_trajectory, live_reads_take_cursors
from agent_env.task_step.snapshot_utils import atif_copies
from agent_env.task_step.snapshot_utils.live_trajectory import LiveTrajectoryChunks
from agent_env.task_step.task_steps import prompt_agent as pa
from tst.unit.task_step.test_prompt_agent_atif_copies import _PREFIX, NATIVE, _atifs, _Run
from tst.util.granting_object_store import GrantingObjectStore

pytestmark = pytest.mark.usefixtures("local_stores")


def _step(n: int, text: str) -> dict:
    return {"step_id": n, "source": "agent", "message": text}


def _live(run: _Run) -> dict[str, dict[str, bytes]]:
    """Each turn's live folder, by turn id: its files by name under ``live/``."""
    turns: dict[str, dict[str, bytes]] = {}
    for key in run.store.list(_PREFIX):
        rest = key[len(_PREFIX):]
        if "/live/" in rest:
            turn, name = rest.split("/live/", 1)
            turns.setdefault(turn, {})[name] = run.store.read(key)
    return turns


def _reassembled(files: dict[str, bytes]) -> tuple[dict, list[dict]]:
    """A turn's live document as a reader rebuilds it: the latest root header and every page's steps, in order."""
    headers = sorted(name for name in files if "/" not in name and name.endswith(".header.json"))
    pages = sorted(name for name in files if "/" not in name and name.endswith(".jsonl"))
    steps = [json.loads(line) for page in pages for line in files[page].splitlines() if line]
    return json.loads(files[headers[-1]]), steps


@pytest.mark.asyncio
async def test_a_prescribed_turn_is_followed_as_atif_and_its_pages_are_a_prefix_of_its_stored_copy(monkeypatch,
                                                                                                    tmp_path):
    run = _Run(monkeypatch, tmp_path, live=True)
    context = run.context()

    response = await run.execute(context, live_trajectory=True)

    final = run.read(_atifs(response)[-1]["url"])
    turns = _live(run)
    assert len(turns) == 2
    for task, (turn, files) in zip(tasks(final), sorted(turns.items(), key=lambda kv: json.loads(
            kv[1]["meta.json"])["trajectory_id"] + kv[0])):
        meta = json.loads(files["meta.json"])
        assert meta["format"] == ATIF and meta["trajectory_id"] == final["trajectory_id"]
        assert meta["session_id"] == final["session_id"] and meta["media_base"] == atif_copies.atif_folder(
            context, "solver")
        header, steps = _reassembled(files)
        assert header["trajectory_id"] == final["trajectory_id"] and header["agent"] == final["agent"]
        assert json.loads(files["end.json"]) == {"state": "completed", "next": steps[-1]["step_id"]}
    live_steps = [step for files in turns.values() for step in _reassembled(files)[1]]
    assert sorted(step["step_id"] for step in live_steps) == [step["step_id"] for step in final["steps"]]
    by_id = {step["step_id"]: step for step in final["steps"]}
    assert all(step == by_id[step["step_id"]] for step in live_steps)


@pytest.mark.asyncio
async def test_a_turns_header_is_stored_once_per_change_not_once_a_page(monkeypatch, tmp_path):
    stored = asyncio.Event()
    store_atif = LiveTrajectoryChunks._store_atif

    async def storing(self, batch):
        await store_atif(self, batch)
        if batch.events:
            stored.set()

    monkeypatch.setattr(LiveTrajectoryChunks, "_store_atif", storing)
    run = _Run(monkeypatch, tmp_path, live=True, gate=stored)

    await run.execute(run.context(), live_trajectory=True, max_conversation_turns=1)

    [files] = _live(run).values()
    headers = [name for name in files if name.endswith(".header.json")]
    pages = [name for name in files if name.endswith(".jsonl")]
    assert json.loads(files["meta.json"])["format"] == ATIF and len(pages) >= 2
    assert 1 <= len({files[name] for name in headers}) == len(headers) < len(pages) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cursors", [True, False], ids=["card-takes-cursors", "older-card"])
async def test_an_agent_whose_card_takes_cursors_sends_the_header_only_when_it_changed(monkeypatch, tmp_path, cursors):
    """An older agent's card doesn't take them, and it's read as before #890's follower did: the header on every read,
    the stored pages the same either way. The turn holds until three reads were answered, so some find it unchanged."""
    answered = asyncio.Event()
    if not cursors:
        monkeypatch.setattr(pa, "live_reads_take_cursors", lambda card: False)
    run = _Run(monkeypatch, tmp_path, live=True, gate=answered)
    reads: list[tuple[dict, dict]] = []

    def seen(request, path, sent, response):
        if path == "/ext/trajectory" and sent.get("format") == ATIF and "after" in sent and response.status_code == 200:
            reads.append((sent, json.loads(response.content)))
            if len(reads) >= 3:
                answered.set()

    run.wire.answer = seen
    context = run.context()
    response = await run.execute(context, live_trajectory=True, max_conversation_turns=1)

    [files] = _live(run).values()
    header, steps = _reassembled(files)
    final = run.read(_atifs(response)[-1]["url"])
    assert steps == final["steps"][:len(steps)] and header["agent"] == final["agent"]
    served = [answer["header"] for _, answer in reads if "header" in answer]
    if cursors:
        assert all("header_digest" in sent and sent["subagent_cursors"] == {} for sent, _ in reads)
        assert len(served) < len(reads) and all(a != b for a, b in zip(served, served[1:]))
    else:
        assert not any("header_digest" in sent or "subagent_cursors" in sent for sent, _ in reads)
        assert len(served) == len(reads)


def test_only_a_card_declaring_the_digest_and_cursors_takes_them(monkeypatch, tmp_path):
    card = _Run(monkeypatch, tmp_path, live=True).card
    older = copy.deepcopy(card)
    [trajectory] = [ext for ext in older["capabilities"]["extensions"] if ext["uri"].endswith("trajectory/v1")]
    for branch in trajectory["params"]["methods"]["get"]["request"]["oneOf"]:
        branch["optional"] = [name for name in branch.get("optional", [])
                              if "digest" not in name and "cursor" not in name]
    assert live_reads_take_cursors(card) and not live_reads_take_cursors(older)


@pytest.mark.asyncio
async def test_a_follow_with_cursors_stores_a_running_sub_agents_steps_before_the_call_that_started_it_closes(tmp_path):
    """A read names what the follower holds; the agent sends what's new: a sub-agent's steps while the root has none."""
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True)
    sink = LiveTrajectoryChunks(store, "p/", "turn-1")
    sub_head = {"trajectory_id": "toolu_1", "agent": {"name": "explorer"}}
    answers = [
        ({"header_digest": "", "subagent_cursors": {}}, "running", {
            "header": {"session_id": "c", "trajectory_id": "d"}, "header_digest": "sha256:h1",
            "events": [_step(1, "r1")], "next": 1, "subagents": [{**sub_head, "steps": [_step(1, "s1")]}],
            "subagent_header_digests": {"toolu_1": "sha256:s"}}),
        ({"header_digest": "sha256:h1", "subagent_cursors": {"toolu_1": {"after": 1, "header_digest": "sha256:s"}}},
         "running", {"header_digest": "sha256:h1", "events": [], "next": 1,
                     "subagents": [{"trajectory_id": "toolu_1", "steps": [_step(2, "s2")]}],
                     "subagent_header_digests": {"toolu_1": "sha256:s"}}),
        ({"header_digest": "sha256:h1", "subagent_cursors": {"toolu_1": {"after": 2, "header_digest": "sha256:s"}}},
         "completed", {"header": {"session_id": "c", "trajectory_id": "d", "final_metrics": {"total_steps": 2}},
                       "header_digest": "sha256:h2", "events": [_step(2, "r2")], "next": 2, "subagents": [],
                       "subagent_header_digests": {"toolu_1": "sha256:s"}}),
    ]
    stored_before_root_closed: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        expected, state, page = answers.pop(0)
        assert {key: sent[key] for key in expected} == expected
        if state == "completed":
            stored_before_root_closed.extend(store.list("p/turn-1/live/sub/"))
        return httpx.Response(200, json={"context_id": "c", "task_id": "t", "state": state, "format": ATIF,
                                         "schema_version": "ATIF-v1.8", "profile_version": "1.0", "has_more": False,
                                         **page})

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        await follow_trajectory("http://agent.test/ext/trajectory", "t", sink, poll_interval_seconds=0,
                                stop=asyncio.Event(), client=client, format=ATIF, fallback_format=NATIVE, cursors=True)

    names = sorted(key[len("p/turn-1/live/"):] for key in store.list("p/turn-1/live/"))
    assert sorted(key[len("p/turn-1/live/"):] for key in stored_before_root_closed) == [
        "sub/toolu_1/00000000-00000001.0001.header.json", "sub/toolu_1/00000000-00000001.jsonl",
        "sub/toolu_1/00000001-00000002.jsonl"]
    assert [n for n in names if n.endswith(".header.json") and "/" not in n] == [
        "00000000-00000001.0001.header.json", "00000001-00000002.0002.header.json"]
    assert json.loads(store.read("p/turn-1/live/sub/toolu_1/00000000-00000001.0001.header.json")) == sub_head
    assert not answers


@pytest.mark.asyncio
async def test_an_agent_whose_card_declares_cursors_but_whose_answers_dont_carry_them_is_followed_as_before(tmp_path):
    """Its own handler answers in today's shape, whole sub-agents and no digests: each sub-agent is still stored."""
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True)
    sink = LiveTrajectoryChunks(store, "p/", "turn-1")
    sub = {"trajectory_id": "toolu_1", "agent": {"name": "explorer"}}
    answers = [("running", [_step(1, "r1")], 1, [_step(1, "s1")]), ("completed", [_step(2, "r2")], 2,
                                                                    [_step(1, "s1"), _step(2, "s2")])]
    named = []

    def answer(request: httpx.Request) -> httpx.Response:
        named.append("subagent_cursors" in json.loads(request.content))
        state, events, nxt, sub_steps = answers.pop(0)
        return httpx.Response(200, json={"context_id": "c", "task_id": "t", "state": state, "format": ATIF,
                                         "schema_version": "ATIF-v1.8", "profile_version": "1.0", "has_more": False,
                                         "header": {"session_id": "c", "trajectory_id": "d"}, "events": events,
                                         "next": nxt, "subagents": [{**sub, "steps": sub_steps}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        await follow_trajectory("http://agent.test/ext/trajectory", "t", sink, poll_interval_seconds=0,
                                stop=asyncio.Event(), client=client, format=ATIF, fallback_format=NATIVE, cursors=True)

    names = sorted(key[len("p/turn-1/live/"):] for key in store.list("p/turn-1/live/") if "/sub/" in key)
    assert named == [True, True]
    assert names == ["sub/toolu_1/00000000-00000001.0001.header.json", "sub/toolu_1/00000000-00000001.jsonl",
                     "sub/toolu_1/00000001-00000002.jsonl"]
    assert json.loads(store.read("p/turn-1/live/sub/toolu_1/00000000-00000001.0001.header.json")) == sub


@pytest.mark.asyncio
async def test_a_turn_whose_atif_read_is_refused_is_followed_natively(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, live=True)
    run.wire.unreachable.add("/ext/trajectory/prescribe")

    response = await run.execute(run.context(), live_trajectory=True, max_conversation_turns=1)

    assert [atif["reason"] for atif in _atifs(response)] == ["format_not_prescribed"]
    [files] = _live(run).values()
    assert json.loads(files["meta.json"]) == {"format": NATIVE}
    assert not [name for name in files if name.endswith(".header.json")]


@pytest.mark.asyncio
async def test_a_live_pages_media_are_pulled_while_the_turn_runs(monkeypatch, tmp_path):
    """The agent holds its turn open until a page's media were pulled, so the pull can only have come mid-turn."""
    held = asyncio.Event()
    run = _Run(monkeypatch, tmp_path, live=True, gate=held)
    pulled: list[list[str]] = []
    pull = atif_copies.AtifCopies.pull_live_media

    async def spy(self, document, session_id):
        pulled.append(sorted(media_paths(document)))
        await pull(self, document, session_id)
        if media_paths(document):
            held.set()

    monkeypatch.setattr(atif_copies.AtifCopies, "pull_live_media", spy)
    context = run.context()

    response = await run.execute(context, live_trajectory=True, max_conversation_turns=1)

    [atif] = _atifs(response)
    first_shot = media_paths({"steps": tasks(run.read(atif["url"]))[0].steps[:2]})
    assert first_shot and any(set(first_shot) <= set(names) for names in pulled)
    assert atif["status"] == "valid"
    held = {path.name for path in (run.folder(context) / "media").glob("*")}
    assert {Path(name).name for names in pulled for name in names} <= held


@pytest.mark.asyncio
async def test_an_atif_follow_the_agent_refuses_after_its_pages_were_stored_ends_naming_why(tmp_path):
    """Its pages never mix formats: the follow stops and end.json says why, the turn's final copy unaffected."""
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True)
    reads = iter([
        httpx.Response(200, json={"context_id": "c", "task_id": "t", "state": "running", "format": ATIF,
                                  "schema_version": "ATIF-v1.8", "profile_version": "1.0",
                                  "header": {"session_id": "c", "trajectory_id": "d"},
                                  "events": [{"step_id": 1, "source": "user", "message": "go"}], "next": 1,
                                  "has_more": False}),
        httpx.Response(409, json={"error": {"code": "atif_unavailable", "message": "the adapter failed"}}),
    ])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(reads)))
    sink = LiveTrajectoryChunks(store, "p/", "turn-1")

    await follow_trajectory("http://agent.test/ext/trajectory", "t", sink, poll_interval_seconds=0,
                            stop=asyncio.Event(), client=client, format=ATIF, fallback_format=NATIVE)
    await sink.end(TrajectoryState.FAILED)

    assert json.loads(store.read("p/turn-1/live/end.json")) == {"state": "failed", "next": 1,
                                                                "stopped": "atif_unavailable"}
    assert json.loads(store.read("p/turn-1/live/meta.json"))["format"] == ATIF
    assert store.read("p/turn-1/live/00000000-00000001.jsonl").count(b"\n") == 1


@pytest.mark.asyncio
async def test_a_sub_agents_new_steps_and_changed_header_are_stored_under_its_folder_as_they_come(tmp_path):
    """A page carries each sub-agent as far as it's served; only what wasn't stored yet is written, a nested one too."""
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True)
    sink = LiveTrajectoryChunks(store, "p/", "turn-1")

    def sub(n: int, *, nested: int = 0) -> dict:
        inner = [{"trajectory_id": "inner", "agent": {"name": "helper"},
                  "steps": [{"step_id": k, "source": "agent", "message": f"i{k}"} for k in range(1, nested + 1)]}]
        return {"trajectory_id": "toolu_1", "agent": {"name": "explorer"},
                "steps": [{"step_id": k, "source": "agent", "message": f"s{k}"} for k in range(1, n + 1)],
                **({"subagent_trajectories": inner} if nested else {})}

    def batch(after: int, nxt: int, subagent: dict) -> TrajectoryBatch:
        return TrajectoryBatch(after=after, next=nxt, events=[{"step_id": nxt, "source": "agent", "message": "r"}],
                               format=ATIF, state=TrajectoryState.RUNNING, last=False, header={"session_id": "c"},
                               subagents=[subagent], schema_version="ATIF-v1.8", profile_version="1.0")

    await sink(batch(0, 1, sub(2)))
    await sink(batch(1, 2, sub(3, nested=1)))
    await sink(batch(2, 3, sub(3, nested=1)))

    names = sorted(key[len("p/turn-1/live/"):] for key in store.list("p/turn-1/live/"))
    assert [n for n in names if n.startswith("sub/toolu_1/") and n.endswith(".jsonl") and "/sub/" not in n[4:]] == [
        "sub/toolu_1/00000000-00000002.jsonl", "sub/toolu_1/00000002-00000003.jsonl"]
    assert [n for n in names if n.startswith("sub/toolu_1/") and n.endswith(".header.json") and "/sub/" not in n[4:]] \
        == ["sub/toolu_1/00000000-00000002.0001.header.json"]
    assert "sub/toolu_1/sub/inner/00000000-00000001.jsonl" in names
    assert [n for n in names if n.endswith(".header.json") and "/" not in n] == ["00000000-00000001.0001.header.json"]


def _page(after: int, steps: int, state: str = "running") -> httpx.Response:
    events = [{"step_id": n, "source": "agent", "message": f"m{n}"} for n in range(after + 1, after + steps + 1)]
    return httpx.Response(200, json={"context_id": "c", "task_id": "t", "state": state, "format": ATIF,
                                     "schema_version": "ATIF-v1.8", "profile_version": "1.0",
                                     "header": {"session_id": "c", "trajectory_id": "d"}, "events": events,
                                     "next": after + steps, "has_more": False})


@pytest.mark.asyncio
async def test_an_atif_follow_read_again_until_its_copy_starts_stores_every_page(tmp_path):
    """A conversation's first task has no copy for a moment after the agent takes it: its 404 isn't the task gone."""
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True)
    reads = iter([httpx.Response(404), httpx.Response(404), _page(0, 2), _page(2, 1, state="completed")])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(reads)))
    sink = LiveTrajectoryChunks(store, "p/", "turn-1")

    await follow_trajectory("http://agent.test/ext/trajectory", "t", sink, poll_interval_seconds=0,
                            stop=asyncio.Event(), client=client, format=ATIF, fallback_format=NATIVE)

    assert json.loads(store.read("p/turn-1/live/meta.json"))["format"] == ATIF
    assert json.loads(store.read("p/turn-1/live/end.json")) == {"state": "completed", "next": 3}


@pytest.mark.asyncio
async def test_an_atif_follow_whose_copy_never_starts_ends_with_its_turn(tmp_path):
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    sink = LiveTrajectoryChunks(store, "p/", "turn-1")
    stop = asyncio.Event()
    follower = asyncio.create_task(follow_trajectory(
        "http://agent.test/ext/trajectory", "t", sink, poll_interval_seconds=0.01, stop=stop, client=client,
        format=ATIF, fallback_format=NATIVE))
    await asyncio.sleep(0.1)
    assert not follower.done()

    stop.set()

    await asyncio.wait_for(follower, 5)
    assert store.list("p/turn-1/live/") == []
