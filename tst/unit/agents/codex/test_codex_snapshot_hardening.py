"""Workspace/session snapshot hardening for the Codex A2A wrapper."""
import asyncio
import io
import json
import tarfile
import threading
from pathlib import Path

import pytest

from tst.unit.agents.codex.agent import a2a_server, codex_bridge


def test_complete_jsonl_reader_ignores_torn_final_record(tmp_path):
    path = tmp_path / "rollout.jsonl"
    first = {"type": "session_meta", "payload": {"id": "thread-1"}}
    second = {"type": "event_msg", "payload": {"type": "task_started"}}
    path.write_bytes(
        (json.dumps(first) + "\n" + json.dumps(second) + "\n{\"type\":").encode()
    )

    assert a2a_server._read_complete_jsonl(path) == (
        json.dumps(first) + "\n" + json.dumps(second) + "\n"
    ).encode()


def test_complete_jsonl_reader_rejects_malformed_complete_record(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(b'{"type":"session_meta"}\n{"type":\n')
    with pytest.raises(ValueError, match="malformed complete record"):
        a2a_server._read_complete_jsonl(path)


def test_snapshot_build_limits_are_enforced(monkeypatch, tmp_path):
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text('{"type":"session_meta"}\n')
    monkeypatch.setattr(a2a_server, "_MAX_TRANSCRIPT_BYTES", 4)
    with pytest.raises(ValueError, match="transcript exceeds"):
        a2a_server._read_complete_jsonl(transcript)

    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "large.txt").write_text("x" * 20)
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "_MAX_SNAPSHOT_UNCOMPRESSED_BYTES", 10)
    with pytest.raises(ValueError, match="uncompressed size"):
        a2a_server._build_workspace_archive(tmp_path / "workspace.tar.gz")


def test_workspace_archive_round_trip(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "result.txt").write_text("finished")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "data.json").write_text('{"ok": true}')
    archive = tmp_path / "workspace.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)

    a2a_server._build_workspace_archive(archive)
    (workspace / "result.txt").write_text("changed")
    (workspace / "stale.txt").write_text("remove me")
    a2a_server._restore_workspace_archive(archive)

    assert (workspace / "result.txt").read_text() == "finished"
    assert (workspace / "nested" / "data.json").read_text() == '{"ok": true}'
    assert not (workspace / "stale.txt").exists()


def test_workspace_restore_does_not_require_writable_parent(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "restored.txt").write_text("restored")
    archive = tmp_path / "workspace.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", source)
    a2a_server._build_workspace_archive(archive)

    readonly_parent = tmp_path / "readonly-parent"
    readonly_parent.mkdir()
    workspace = readonly_parent / "app"
    workspace.mkdir()
    (workspace / "old.txt").write_text("old")
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    readonly_parent.chmod(0o555)
    try:
        a2a_server._restore_workspace_archive(archive)
    finally:
        readonly_parent.chmod(0o755)

    assert (workspace / "restored.txt").read_text() == "restored"
    assert not (workspace / "old.txt").exists()


def test_workspace_archive_preserves_safe_symlink(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "target.txt").write_text("target")
    (workspace / "link.txt").symlink_to("target.txt")
    archive = tmp_path / "workspace.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)

    a2a_server._build_workspace_archive(archive)
    (workspace / "link.txt").unlink()
    a2a_server._restore_workspace_archive(archive)

    assert (workspace / "link.txt").is_symlink()
    assert (workspace / "link.txt").readlink() == Path("target.txt")


def test_workspace_archive_rejects_unsafe_symlink(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "link.txt").symlink_to("/etc/passwd")
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)

    with pytest.raises(ValueError, match="unsafe link"):
        a2a_server._build_workspace_archive(
            tmp_path / "workspace.tar.gz"
        )


def test_workspace_restore_rolls_back_failed_swap(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "old.txt").write_text("old")
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new")
    archive = tmp_path / "workspace.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", source)
    a2a_server._build_workspace_archive(archive)
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)

    replace = a2a_server.os.replace
    calls = 0

    def fail_new_workspace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected swap failure")
        return replace(src, dst)

    monkeypatch.setattr(a2a_server.os, "replace", fail_new_workspace)
    with pytest.raises(OSError, match="injected"):
        a2a_server._restore_workspace_archive(archive)
    assert (workspace / "old.txt").read_text() == "old"
    assert not (workspace / "new.txt").exists()


def test_workspace_restore_rejects_large_tar_metadata(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "app"
    workspace.mkdir()
    archive = tmp_path / "pax.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "_MAX_TAR_METADATA_BYTES", 512)
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for index in range(4):
            info = tarfile.TarInfo(f"file-{index}.txt")
            info.size = 1
            info.pax_headers = {"comment": "x" * 200}
            tar.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="metadata is too large"):
        a2a_server._restore_workspace_archive(archive)


def test_workspace_restore_checks_expanded_size_before_extract(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "app"
    workspace.mkdir()
    archive = tmp_path / "large.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(
        a2a_server, "_MAX_SNAPSHOT_UNCOMPRESSED_BYTES", 10
    )
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("large.txt")
        info.size = 20
        tar.addfile(info, io.BytesIO(b"x" * 20))

    with pytest.raises(ValueError, match="uncompressed size"):
        a2a_server._restore_workspace_archive(archive)


def test_workspace_restore_checks_free_space(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    archive = tmp_path / "workspace.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("file.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    monkeypatch.setattr(
        a2a_server.shutil,
        "disk_usage",
        lambda _: type("Usage", (), {"free": 0})(),
    )

    with pytest.raises(ValueError, match="insufficient disk space"):
        a2a_server._restore_workspace_archive(archive)


def test_cancel_waits_for_restore_thread(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()

    def restore(_):
        started.set()
        release.wait()

    monkeypatch.setattr(a2a_server, "_restore_workspace_archive", restore)

    async def exercise():
        task = asyncio.create_task(
            a2a_server._restore_workspace_before_cancelling(
                tmp_path / "workspace.tar.gz"
            )
        )
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_workspace_restore_rejects_path_escape(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    archive = tmp_path / "hostile.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"bad"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="escapes"):
        a2a_server._restore_workspace_archive(archive)
    assert not (tmp_path / "escape.txt").exists()


def test_workspace_restore_rejects_links(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    archive = tmp_path / "link.tar.gz"
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError, match="unsafe link"):
        a2a_server._restore_workspace_archive(archive)


def test_agent_card_does_not_claim_unsupported_allowed_tools(codex):
    extension = next(
        ext
        for ext in codex.card["capabilities"]["extensions"]
        if ext["uri"] == "urn:agentenv:agent-config/v1"
    )
    supported = extension["params"]["methods"]["set"]["request"]["supported"]
    assert "allowed_tools" not in supported

    async def exercise():
        async with codex.client() as client:
            return await client.post("/ext/agent-config", json={"allowed_tools": ["Bash"]})

    response = asyncio.run(exercise())
    assert response.status_code == 400 and "allowed_tools" in response.text


def _trajectory(codex, body: dict):
    async def exercise():
        async with codex.client() as client:
            return await client.post("/ext/trajectory", json=body)

    return asyncio.run(exercise())


THREAD_STARTED = {"type": "thread.started", "thread_id": "thread-1"}


def test_live_trajectory_is_available_by_context(codex, monkeypatch):
    async def exercise():
        gate = asyncio.Event()
        codex.answer_turns(monkeypatch, [THREAD_STARTED], gate=gate)
        async with codex.client() as client:
            await codex.send(client, "go", context_id="context-1", blocking=False)
            async with asyncio.timeout(5):
                while (response := await client.post("/ext/trajectory", json={"context_id": "context-1"})).status_code == 404:
                    await asyncio.sleep(0.01)
            gate.set()
            return response.json()

    assert asyncio.run(exercise()) == {
        "trajectory": [THREAD_STARTED],
        "is_live": True,
        "session_id": "thread-1",
        "harness": "codex-cli",
        "session_lines": 1,
    }


def test_a_live_trajectory_that_has_not_started_is_404(codex, monkeypatch):
    async def exercise():
        gate = asyncio.Event()
        codex.answer_turns(monkeypatch, [], gate=gate)
        async with codex.client() as client:
            await codex.send(client, "go", context_id="context-1", blocking=False)
            started = await client.post("/ext/trajectory", json={"context_id": "context-1"})
            unknown = await client.post("/ext/trajectory", json={"context_id": "unknown"})
            gate.set()
            return started, unknown

    started, unknown = asyncio.run(exercise())
    assert (started.status_code, started.json()) == (
        404, {"detail": "Trajectory has not started for context 'context-1'"})
    assert (unknown.status_code, unknown.json()) == (404, {"detail": "Unknown context 'unknown'"})


def test_completed_context_trajectory_is_cumulative(codex, monkeypatch):
    # The first turn alone is more than one framework read, so the answer spans several pages.
    first = [THREAD_STARTED] + [{"type": "item.completed", "n": n} for n in range(1_200)]
    second = [{"type": "turn.completed"}]
    codex.answer_turns(monkeypatch, first, second)

    async def exercise():
        async with codex.client() as client:
            await codex.send(client, "one", context_id="completed-context")
            await codex.send(client, "two", context_id="completed-context")
            return (await client.post("/ext/trajectory", json={"context_id": "completed-context"})).json()

    response = asyncio.run(exercise())
    assert response["trajectory"] == first + second
    assert response["session_lines"] == len(first + second)
    assert response["is_live"] is False
    assert response["session_id"] == "thread-1"


def test_task_trajectory_keeps_legacy_response_shape(codex, monkeypatch):
    events = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.completed"}]

    async def fake_run_codex(**kwargs):
        for event in events:
            kwargs["event_callback"](event)
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t", trajectory=events)

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            task = await codex.send(client, "go", context_id="ctx-task")
            return await client.post("/ext/trajectory", json={"task_id": task["id"]})

    assert asyncio.run(exercise()).json() == {"trajectory": events}


def test_errors_keep_the_json_detail_body(codex):
    response = _trajectory(codex, {"task_id": "nope"})
    assert (response.status_code, response.json()) == (
        404, {"detail": "No completed trajectory available for task 'nope'"})
