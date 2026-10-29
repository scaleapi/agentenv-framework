"""What a turn hands Hermes and when it runs: turns of different contexts at once and a context's own one at a time, the
session a context is bound to when its turn starts, the task's config, files and MCP servers, a failed turn, and a
snapshot of a context's conversation and workspace restored into another context."""

from __future__ import annotations

import asyncio
import shutil

import pytest
from agentenv_protocol.a2a_agent import Uploaded

from tst.unit.agents.hermes.agent import a2a_server, hermes_bridge, hermes_kit
from tst.unit.agents.hermes.conftest import result


class Turns:
    """Stand in for run_hermes: each call's arguments, held until ``release`` when it is set."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.release: asyncio.Event | None = None
        self.answer = lambda kwargs: result(session_id=kwargs["session_id"] or f"s-{kwargs['context_id']}")

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.release is not None:
            await self.release.wait()
        return self.answer(kwargs)


@pytest.fixture
def turns(monkeypatch) -> Turns:
    stand_in = Turns()
    monkeypatch.setattr(a2a_server, "run_hermes", stand_in)
    return stand_in


async def _until(condition, seconds: float = 5) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_turns_of_different_contexts_run_at_once(hermes, turns):
    turns.release = asyncio.Event()
    async with hermes.client() as client:
        sent = [asyncio.create_task(hermes.send(client, "go", context_id)) for context_id in ("ctx-a", "ctx-b")]
        await _until(lambda: len(turns.calls) == 2)
        turns.release.set()
        tasks = await asyncio.gather(*sent)

    assert {call["context_id"] for call in turns.calls} == {"ctx-a", "ctx-b"}
    assert [task["status"]["state"] for task in tasks] == ["completed", "completed"]


@pytest.mark.asyncio
async def test_a_contexts_turns_run_one_at_a_time_each_resuming_the_last(hermes, turns):
    turns.release = asyncio.Event()
    async with hermes.client() as client:
        first = asyncio.create_task(hermes.send(client, "one"))
        await _until(lambda: len(turns.calls) == 1)
        second = asyncio.create_task(hermes.send(client, "two"))
        await asyncio.sleep(0.1)
        running_together = len(turns.calls)
        turns.release.set()
        await asyncio.gather(first, second)

    assert running_together == 1
    assert [call["session_id"] for call in turns.calls] == [None, "s-ctx-1"]


@pytest.mark.asyncio
async def test_the_turns_config_servers_and_workspace_reach_hermes(monkeypatch, hermes, turns, folders):
    monkeypatch.setenv("A2A_PORT", "8000")
    config = {"model": "provider/model", "effort": "high", "system_prompt": "be brief", "project_id": "p-1",
              "task_id": "t-1", "role": "solver", "timeout_seconds": 42}
    async with hermes.client() as client:
        assert (await client.post("/ext/agent-config", json=config)).status_code == 200
        await client.post("/ext/mcp-config", json={"name": "env", "url": "https://mcp.example/mcp",
                                                   "headers": {"X-Key": "k"}})
        task = await hermes.send(client, "go")

    (call,) = turns.calls
    assert {key: call[key] for key in ("prompt", "model", "effort", "system_prompt", "project_id", "task_id", "role",
                                       "timeout_s", "workspace", "skills_dir")} == {
        "prompt": "go", "model": "provider/model", "effort": "high", "system_prompt": "be brief",
        "project_id": "p-1", "task_id": "t-1", "role": "solver", "timeout_s": 42, "workspace": str(folders),
        "skills_dir": a2a_server.SKILLS_DIR}
    assert call["mcp_servers"]["env"] == {"url": "https://mcp.example/mcp", "headers": {"X-Key": "k"}}
    assert call["mcp_servers"]["peer_agents"]["url"] == "http://127.0.0.1:8000/ext/peer-agents/mcp"
    assert task["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_a_file_part_is_written_under_the_task_and_named_in_the_prompt(hermes, turns, folders):
    parts = [{"kind": "text", "text": "describe"},
             {"kind": "file", "file": {"name": "../a b.png", "mimeType": "image/png", "bytes": "cG5n"}}]
    async with hermes.client() as client:
        task = await hermes.send(client, parts)

    attached = folders / "inputs" / task["id"] / "a_b.png"
    assert attached.read_bytes() == b"png"
    (call,) = turns.calls
    assert call["prompt"].startswith("describe\n\nAttached files")
    assert f"- {attached}" in call["prompt"]


@pytest.mark.asyncio
async def test_a_failed_turn_is_a_failed_task_that_keeps_its_session(hermes, turns):
    turns.answer = lambda kwargs: (
        result(response="", error="HTTP 401: invalid key", error_code="401", session_id="s-started")
        if not kwargs["session_id"] else result(session_id=kwargs["session_id"]))
    async with hermes.client() as client:
        failed = await hermes.send(client, "one")
        await hermes.send(client, "two")

    assert failed["status"]["state"] == "failed"
    assert "HTTP 401: invalid key" in str(failed["status"]["message"])
    assert [call["session_id"] for call in turns.calls] == [None, "s-started"]


@pytest.mark.asyncio
async def test_a_first_turn_cancelled_after_hermes_named_its_session_keeps_it_for_the_next(monkeypatch, hermes):
    calls: list[dict] = []
    started = asyncio.Event()

    async def run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["event_callback"]({"type": "on_session_start", "session_id": "s-started"})
            started.set()
            await asyncio.Event().wait()
        return result(session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_hermes", run)
    async with hermes.client() as client:
        task = await hermes.send(client, "one", blocking=False)
        await asyncio.wait_for(started.wait(), 5)
        await client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": task["id"]}})
        await _until(lambda: hermes.agent.session_ref_for_context("ctx-1") == "s-started")
        await hermes.send(client, "two")

    assert [call["session_id"] for call in calls] == [None, "s-started"]


def test_session_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.HermesAgent, hermes_kit.DurableSessionRefs)


_GRANT = {"expires_at": "2099-01-01T00:00:00Z", "url": "https://objects.example/o"}
_WRITE = {"media_type": "application/octet-stream", "max_bytes": 2**20, "write": {"kind": "http-put", **_GRANT}}


@pytest.mark.asyncio
async def test_a_snapshot_restores_the_conversation_and_workspace_into_another_context(
        monkeypatch, hermes, turns, folders, tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()

    async def upload(target, source):
        name = "workspace" if "workspace" in str(source) else "trajectory"
        shutil.copyfile(source, objects / name)
        return Uploaded(size_bytes=(objects / name).stat().st_size)

    async def download(source, destination):
        shutil.copyfile(objects / ("workspace" if "workspace" in str(destination) else "trajectory"), destination)

    monkeypatch.setattr(a2a_server, "upload", upload)
    monkeypatch.setattr(a2a_server, "download", download)
    async with hermes.client() as client:
        await hermes.send(client, "go", "ctx-src")
        source = hermes_bridge.profile_dir("ctx-src")
        source.mkdir(parents=True)
        (source / "state.db").write_bytes(b"the conversation")
        (source / "memories").mkdir()
        (source / "memories" / "MEMORY.md").write_text("a memory")
        (source / "config.yaml").write_text("rewritten each turn")
        (folders / "result.txt").write_text("42")
        saved = await client.post("/ext/snapshot", json={"context_id": "ctx-src", "objects": {
            "trajectory": _WRITE, "workspace": _WRITE}})
        (folders / "result.txt").write_text("changed since")
        read = {"media_type": "application/octet-stream", "max_bytes": 2**20, "size_bytes": 1,
                "read": {"kind": "http-get", **_GRANT}}
        loaded = await client.put("/ext/snapshot", json={"target_context_id": "ctx-dst", "objects": {
            "trajectory": read, "workspace": read}})
        await hermes.send(client, "go on", "ctx-dst")

    assert saved.status_code == 200, saved.text
    assert loaded.status_code == 200, loaded.text
    target = hermes_bridge.profile_dir("ctx-dst")
    assert (target / "state.db").read_bytes() == b"the conversation"
    assert (target / "memories" / "MEMORY.md").read_text() == "a memory"
    assert not (target / "config.yaml").exists()
    assert (folders / "result.txt").read_text() == "42"
    assert turns.calls[-1]["session_id"] == "s-ctx-src"
