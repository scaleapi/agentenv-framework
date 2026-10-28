"""What a turn hands the CLI and when it runs: turns of different contexts at once and a context's own one at a time,
a snapshot alone, the session a context is bound to when its turn starts, and the SDK's peer tools among its MCP
servers."""

from __future__ import annotations

import asyncio

import pytest
from agentenv_protocol.a2a_agent import Uploaded

from tst.unit.agents.opencode.agent import a2a_server, opencode_kit
from tst.unit.agents.opencode.conftest import Harness, result


class Turns:
    """Stand in for run_opencode: each call's arguments, held until ``release`` when it is set."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.release: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.answer = lambda kwargs: result(session_id=kwargs["session_id"] or f"ses_{kwargs['context_id']}")

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        return self.answer(kwargs)


@pytest.fixture
def turns(monkeypatch) -> Turns:
    stand_in = Turns()
    monkeypatch.setattr(a2a_server, "run_opencode", stand_in)
    return stand_in


async def _until(condition, seconds: float = 5) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_turns_of_different_contexts_run_at_once(oc, turns):
    turns.release = asyncio.Event()
    async with oc.client() as client:
        sent = [asyncio.create_task(oc.send(client, "go", context_id)) for context_id in ("ctx-a", "ctx-b")]
        await _until(lambda: len(turns.calls) == 2)
        turns.release.set()
        tasks = await asyncio.gather(*sent)

    assert {call["context_id"] for call in turns.calls} == {"ctx-a", "ctx-b"}
    assert [task["status"]["state"] for task in tasks] == ["completed", "completed"]


@pytest.mark.asyncio
async def test_a_contexts_turns_run_one_at_a_time_each_resuming_the_last(oc, turns):
    turns.release = asyncio.Event()
    async with oc.client() as client:
        first = asyncio.create_task(oc.send(client, "one"))
        await _until(lambda: len(turns.calls) == 1)
        second = asyncio.create_task(oc.send(client, "two"))
        await asyncio.sleep(0.1)
        running_together = len(turns.calls)
        turns.release.set()
        await asyncio.gather(first, second)

    assert running_together == 1
    assert [call["session_id"] for call in turns.calls] == [None, "ses_ctx-1"]


@pytest.mark.asyncio
async def test_a_snapshot_waits_for_running_turns_and_holds_new_ones_back(monkeypatch, oc, turns):
    oc.agent.set_session_ref_for_context("ctx-a", "ses_a")
    exported = asyncio.Event()

    async def export(_context, _session):
        exported.set()
        return b"{}"

    monkeypatch.setattr(a2a_server, "export_session", export)
    monkeypatch.setattr(a2a_server, "upload", _upload)
    turns.release = asyncio.Event()
    async with oc.client() as client:
        running = asyncio.create_task(oc.send(client, "go", "ctx-b"))
        await _until(lambda: len(turns.calls) == 1)
        save = asyncio.create_task(client.post("/ext/snapshot", json={"context_id": "ctx-a", "objects": {
            "trajectory": _WRITE}}))
        await asyncio.sleep(0.1)
        held = asyncio.create_task(oc.send(client, "go", "ctx-c"))
        await asyncio.sleep(0.1)
        while_running = (exported.is_set(), len(turns.calls))
        turns.release.set()
        saved = await save
        await asyncio.gather(running, held)

    assert while_running == (False, 1)
    assert saved.status_code == 200, saved.text
    assert [call["context_id"] for call in turns.calls] == ["ctx-b", "ctx-c"]


@pytest.mark.asyncio
async def test_a_turn_resumes_the_session_a_snapshot_load_bound_while_it_waited(monkeypatch, oc, turns):
    async def download(_source, destination):
        destination.write_bytes(b"{}")

    async def imported(_context, _path):
        return "ses_imported"

    monkeypatch.setattr(a2a_server, "download", download)
    monkeypatch.setattr(a2a_server, "import_session", imported)
    turns.release = asyncio.Event()
    async with oc.client() as client:
        first = asyncio.create_task(oc.send(client, "one", "ctx-x"))
        await _until(lambda: len(turns.calls) == 1)
        load = asyncio.create_task(client.put("/ext/snapshot", json={"target_context_id": "ctx-x", "objects": {
            "trajectory": _READ}}))
        await asyncio.sleep(0.1)
        second = asyncio.create_task(oc.send(client, "two", "ctx-x"))
        await asyncio.sleep(0.1)
        turns.release.set()
        loaded = await load
        await asyncio.gather(first, second)

    assert loaded.status_code == 200, loaded.text
    assert [call["session_id"] for call in turns.calls] == [None, "ses_imported"]


@pytest.mark.asyncio
async def test_turns_past_the_cap_wait_for_one_to_finish(monkeypatch, turns):
    """Each `opencode run` peaks near 900 MB, and a VM that ran every concurrent turn at once had its kernel kill one."""
    monkeypatch.setenv("OPENCODE_MAX_CONCURRENT_TURNS", "2")
    oc = Harness(a2a_server.OpenCodeChangelogAgent())
    turns.release = asyncio.Event()
    async with oc.client() as client:
        sent = [asyncio.create_task(oc.send(client, "go", f"ctx-{n}")) for n in range(3)]
        await _until(lambda: len(turns.calls) == 2)
        await asyncio.sleep(0.1)
        assert len(turns.calls) == 2
        turns.release.set()
        tasks = await asyncio.gather(*sent)

    assert [task["status"]["state"] for task in tasks] == ["completed"] * 3


def test_the_cap_is_what_memory_fits_past_what_the_server_needs(monkeypatch):
    monkeypatch.setattr(a2a_server, "memory_mb", lambda: 8192)
    assert a2a_server.max_concurrent_turns({}) == 7
    monkeypatch.setattr(a2a_server, "memory_mb", lambda: 1500)
    assert a2a_server.max_concurrent_turns({}) == 1
    assert a2a_server.max_concurrent_turns({"OPENCODE_MAX_CONCURRENT_TURNS": "3"}) == 3
    assert a2a_server.max_concurrent_turns({"OPENCODE_MAX_CONCURRENT_TURNS": "0"}) == 1


def test_a_cgroup_limit_below_the_machines_memory_is_the_memory(tmp_path):
    machine = a2a_server.memory_mb(tmp_path / "no-cgroup")
    limit = tmp_path / "memory.max"
    limit.write_text("max\n")
    assert a2a_server.memory_mb(limit) == machine
    limit.write_text(f"{512 * 2**20}\n")
    assert a2a_server.memory_mb(limit) == min(512, machine)


@pytest.mark.asyncio
async def test_a_turn_gets_the_sdks_peer_tools_with_its_registered_servers(monkeypatch, oc, turns):
    monkeypatch.setenv("A2A_PORT", "8000")
    async with oc.client() as client:
        added = await client.post("/ext/mcp-config", json={"name": "env", "url": "https://mcp.example/mcp",
                                                           "headers": {"X-Key": "k"}})
        await oc.send(client, "go")

    assert added.status_code == 200, added.text
    (call,) = turns.calls
    assert call["mcp_servers"]["env"] == {"url": "https://mcp.example/mcp", "headers": {"X-Key": "k"}}
    assert call["mcp_servers"]["peer_agents"] == {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": None}


@pytest.mark.asyncio
async def test_the_turns_config_and_workspace_reach_the_cli(oc, turns, folders):
    config = {"model": "provider/model", "effort": "high", "system_prompt": "be brief", "project_id": "p-1",
              "task_id": "t-1", "role": "solver", "timeout_seconds": 42}
    async with oc.client() as client:
        assert (await client.post("/ext/agent-config", json=config)).status_code == 200
        await oc.send(client, "go")

    (call,) = turns.calls
    assert {key: call[key] for key in ("prompt", "model", "effort", "system_prompt", "project_id", "task_id", "role",
                                       "timeout_s", "workspace")} == {
        "prompt": "go", "model": "provider/model", "effort": "high", "system_prompt": "be brief",
        "project_id": "p-1", "task_id": "t-1", "role": "solver", "timeout_s": 42, "workspace": str(folders)}


@pytest.mark.asyncio
async def test_a_timed_out_turn_keeps_the_session_it_started_for_the_next(oc, turns):
    turns.answer = lambda kwargs: (
        result(response="", error="opencode timed out after 1s", session_id="ses_started") if not kwargs["session_id"]
        else result(session_id=kwargs["session_id"]))
    async with oc.client() as client:
        timed_out = await oc.send(client, "one")
        await oc.send(client, "two")

    assert timed_out["status"]["state"] == "failed"
    assert [call["session_id"] for call in turns.calls] == [None, "ses_started"]


def test_session_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.OpenCodeAgent, opencode_kit.DurableSessionRefs)


_GRANT = {"expires_at": "2099-01-01T00:00:00Z", "url": "https://objects.example/o"}
_WRITE = {"media_type": "application/octet-stream", "max_bytes": 1024, "write": {"kind": "http-put", **_GRANT}}
_READ = {"media_type": "application/octet-stream", "max_bytes": 1024, "size_bytes": 2,
         "read": {"kind": "http-get", **_GRANT}}


async def _upload(_target, source):
    return Uploaded(size_bytes=len(source))


@pytest.mark.asyncio
async def test_a_file_part_is_written_under_the_task_and_attached(oc, turns, folders):
    parts = [{"kind": "text", "text": "describe"},
             {"kind": "file", "file": {"name": "../a b.png", "mimeType": "image/png", "bytes": "cG5n"}}]
    async with oc.client() as client:
        task = await oc.send(client, parts)

    (call,) = turns.calls
    (attached,) = call["attachment_paths"]
    assert attached == str(folders / "inputs" / task["id"] / "a_b.png")
    assert open(attached, "rb").read() == b"png"


@pytest.mark.asyncio
async def test_an_unreadable_file_part_fails_the_turn_before_the_cli_runs(oc, turns):
    parts = [{"kind": "file", "file": {"name": "a.png", "mimeType": "image/png", "bytes": "not base64!"}}]
    async with oc.client() as client:
        task = await oc.send(client, parts)

    assert task["status"]["state"] == "failed" and turns.calls == []


@pytest.mark.asyncio
async def test_a_snapshot_asked_for_during_a_contexts_first_turn_waits_for_its_session(monkeypatch, oc, turns):
    exported = []

    async def export(_context, session):
        exported.append(session)
        return b"{}"

    monkeypatch.setattr(a2a_server, "export_session", export)
    monkeypatch.setattr(a2a_server, "upload", _upload)
    turns.release = asyncio.Event()
    async with oc.client() as client:
        first = asyncio.create_task(oc.send(client, "go", "ctx-new"))
        await _until(lambda: len(turns.calls) == 1)
        save = asyncio.create_task(client.post("/ext/snapshot", json={"context_id": "ctx-new", "objects": {
            "trajectory": _WRITE}}))
        await asyncio.sleep(0.1)
        turns.release.set()
        saved = await save
        await first

    assert saved.status_code == 200, saved.text
    assert exported == ["ses_ctx-new"]


@pytest.mark.asyncio
async def test_a_first_turn_cancelled_after_the_cli_named_its_session_keeps_it_for_the_next(monkeypatch, oc):
    calls: list[dict] = []
    started = asyncio.Event()

    async def run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["event_callback"]({"type": "step_start", "sessionID": "ses_started", "part": {"type": "step-start"}})
            started.set()
            await asyncio.Event().wait()
        return result(session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_opencode", run)
    async with oc.client() as client:
        task = await oc.send(client, "one", blocking=False)
        await asyncio.wait_for(started.wait(), 5)
        await client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": task["id"]}})
        await _until(lambda: oc.agent.session_ref_for_context("ctx-1") == "ses_started")
        await oc.send(client, "two")

    assert [call["session_id"] for call in calls] == [None, "ses_started"]
