"""A prompt step of a run that prescribes atif/1 follows each turn's atif/1 copy live: each poll's new closed steps a
page, the document's header beside the page where it changed, the media the steps name pulled as they come, and the
pages a prefix of the turn's stored copy; a turn the agent won't serve atif/1 is followed natively, and one it stops
serving partway ends with the reason. Against a live SDK agent served in-process."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from agentenv_protocol.a2a_agent import TrajectoryState
from agentenv_protocol.atif import FORMAT as ATIF, media_paths, tasks

from agent_env.a2a_agent.trajectory_follower import TrajectoryBatch, follow_trajectory
from agent_env.task_step.snapshot_utils import atif_copies
from agent_env.task_step.snapshot_utils.live_trajectory import LiveTrajectoryChunks
from tst.unit.task_step.test_prompt_agent_atif_copies import _PREFIX, NATIVE, _atifs, _Run
from tst.util.granting_object_store import GrantingObjectStore

pytestmark = pytest.mark.usefixtures("local_stores")


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
    run = _Run(monkeypatch, tmp_path, live=True, pause=0.3)

    await run.execute(run.context(), live_trajectory=True, max_conversation_turns=1)

    [files] = _live(run).values()
    headers = [name for name in files if name.endswith(".header.json")]
    pages = [name for name in files if name.endswith(".jsonl")]
    assert json.loads(files["meta.json"])["format"] == ATIF and len(pages) >= 2
    assert 1 <= len({files[name] for name in headers}) == len(headers) < len(pages) + 1


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
    run = _Run(monkeypatch, tmp_path, live=True, pause=0.5)
    pulled: list[list[str]] = []
    pull = atif_copies.AtifCopies.pull_live_media

    async def spy(self, document, session_id):
        pulled.append(sorted(media_paths(document)))
        await pull(self, document, session_id)

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
