"""Work a turn or snapshot starts ends before it lets go of what it holds: a cancelled save keeps the workspace until its
archive is written, a cancelled turn's PDF render is killed and reaped, attachments wait for a turn slot, and a load
whose workspace swap fails leaves the context's previous conversation and session in place."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import threading

import pytest
from agentenv_protocol.a2a_agent import Uploaded

from tst.unit.agents.hermes.agent import a2a_server, hermes_bridge
from tst.unit.agents.hermes.conftest import Harness
from tst.unit.agents.hermes.test_message_contract import _GRANT, _WRITE, Turns, _until

_READ = {"media_type": "application/octet-stream", "max_bytes": 2**20, "size_bytes": 1,
         "read": {"kind": "http-get", **_GRANT}}
_PDF = {"kind": "file", "file": {"name": "report.pdf", "mimeType": "application/pdf", "bytes": "JVBERi0="}}


@pytest.fixture
def turns(monkeypatch) -> Turns:
    stand_in = Turns()
    monkeypatch.setattr(a2a_server, "run_hermes", stand_in)
    return stand_in


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.asyncio
async def test_a_cancelled_save_holds_the_workspace_until_its_archive_ends(monkeypatch, hermes, turns):
    started, finish = threading.Event(), threading.Event()

    def archive(home, session_id, target):
        started.set()
        finish.wait(10)
        target.write_bytes(b"archive")

    monkeypatch.setattr(a2a_server, "_archive_profile", archive)
    async with hermes.client() as client:
        await hermes.send(client, "go")
        save = asyncio.create_task(
            client.post("/ext/snapshot", json={"context_id": "ctx-1", "objects": {"trajectory": _WRITE}}))
        await _until(started.is_set)
        save.cancel()
        turn = asyncio.create_task(hermes.send(client, "next"))
        await asyncio.sleep(0.3)
        turned_during_archive = len(turns.calls) > 1
        finish.set()
        await asyncio.gather(save, return_exceptions=True)
        await turn

    assert not turned_during_archive
    assert len(turns.calls) == 2


@pytest.mark.asyncio
async def test_a_cancelled_turns_pdf_render_is_killed_and_reaped(monkeypatch, tmp_path, hermes, turns):
    pids = tmp_path / "pdftoppm.pid"
    fake = tmp_path / "pdfbin" / "pdftoppm"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho $$ > "{pids}"\nexec sleep 600\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    async with hermes.client() as client:
        task = await hermes.send(client, [{"kind": "text", "text": "read it"}, _PDF], blocking=False)
        await _until(lambda: pids.exists() and pids.read_text().strip())
        pid = int(pids.read_text())
        await client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel",
                                        "params": {"id": task["id"]}})
        try:
            await _until(lambda: not _alive(pid), seconds=5)
        finally:
            if _alive(pid):
                os.kill(pid, 9)

    assert turns.calls == []


@pytest.mark.asyncio
async def test_attachments_wait_for_a_turn_slot(monkeypatch, folders, turns):
    monkeypatch.setenv("HERMES_MAX_CONCURRENT_TURNS", "1")
    hermes = Harness(a2a_server.HermesAgent())
    turns.release = asyncio.Event()
    part = {"kind": "file", "file": {"name": "notes.txt", "mimeType": "text/plain", "bytes": "bm90ZXM="}}
    async with hermes.client() as client:
        first = asyncio.create_task(hermes.send(client, [{"kind": "text", "text": "one"}], "ctx-a"))
        await _until(lambda: len(turns.calls) == 1)
        second = asyncio.create_task(hermes.send(client, [{"kind": "text", "text": "two"}, part], "ctx-b"))
        await asyncio.sleep(0.3)
        written_while_waiting = list((folders / "inputs").rglob("notes.txt")) if (folders / "inputs").exists() else []
        turns.release.set()
        await asyncio.gather(first, second)

    assert written_while_waiting == []
    assert len(turns.calls) == 2


@pytest.mark.asyncio
async def test_a_load_whose_workspace_swap_fails_keeps_the_previous_conversation(
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
        (source / "state.db").write_bytes(b"the snapshot's conversation")
        (folders / "result.txt").write_text("snapshot")
        saved = await client.post("/ext/snapshot", json={"context_id": "ctx-src", "objects": {
            "trajectory": _WRITE, "workspace": _WRITE}})
        assert saved.status_code == 200, saved.text
        await hermes.send(client, "go", "ctx-dst")
        target = hermes_bridge.profile_dir("ctx-dst")
        target.mkdir(parents=True, exist_ok=True)
        (target / "state.db").write_bytes(b"the previous conversation")
        (folders / "result.txt").write_text("previous")

        def full_disk(source, destination):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(a2a_server.shutil, "move", full_disk)
        failed = await client.put("/ext/snapshot", json={"target_context_id": "ctx-dst", "objects": {
            "trajectory": _READ, "workspace": _READ}})
        monkeypatch.undo()

    assert failed.status_code == 500
    assert (target / "state.db").read_bytes() == b"the previous conversation"
    assert (folders / "result.txt").read_text() == "previous"
    assert hermes.agent.session_ref_for_context("ctx-dst") == "s-ctx-dst"
