"""Skills, trajectories, snapshots and the changelog through agent-env's object-transfer grants."""
import asyncio
import json
import os
import shutil
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from agentenv_protocol.a2a_agent import TaskRequest, TextPart, Uploaded, card_request_accepts

from tst.unit.agents.opencode.agent import a2a_server as server
from tst.unit.agents.opencode.agent import opencode_kit


def _grant(kind):
    return {
        "kind": kind,
        "url": "https://objects.example.test/object?secret=redacted",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    }


def _read(media_type="application/octet-stream"):
    return {"media_type": media_type, "max_bytes": 1024, "size_bytes": 3, "read": _grant("http-get")}


def _write(media_type="application/octet-stream"):
    return {"media_type": media_type, "max_bytes": 1024, "write": _grant("http-put")}


def _request(oc, extension, method):
    for item in oc.app.state.agentenv_a2a.card.capabilities.extensions:
        if item.uri.endswith(extension):
            return item.params["methods"][method]["request"]
    raise AssertionError(extension)


@pytest.mark.asyncio
async def test_framework_card_and_standard_routes(oc):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        card = (await client.get("/.well-known/agent.json")).json()
        assert (await client.get("/health")).json() == {"status": "ok"}
    extensions = {entry["uri"] for entry in card["capabilities"]["extensions"]}
    assert "urn:agentenv:attribution-probe/v1" in extensions
    assert "urn:agentenv:install/v1" in extensions
    assert card_request_accepts(_request(oc, "skill-config/v1", "add"), {"name", "description", "skill_bundle"})
    assert card_request_accepts(_request(oc, "trajectory/v1", "get"), {"task_id", "objects"})
    assert card_request_accepts(_request(oc, "snapshot/v1", "save"), {"context_id", "objects"})
    assert card_request_accepts(_request(oc, "snapshot/v1", "load"), {"objects"})
    assert card_request_accepts(_request(oc, "snapshot/v1", "enable-changelog"), {"write_namespace"})
    assert card_request_accepts(_request(oc, "snapshot/v1", "apply-changelog"), {"increments"})


@pytest.mark.asyncio
async def test_skill_bundle_route_downloads_files(monkeypatch, tmp_path, oc):
    monkeypatch.setattr(server, "SKILLS_DIR", tmp_path)

    async def fake_download(_source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"abc")

    monkeypatch.setattr(opencode_kit, "download", fake_download)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        answer = await client.post("/ext/skill-config", json={
            "name": "portable-probe", "description": "probe", "skill_bundle": {
                "max_total_bytes": 1024,
                "files": [{"path": "SKILL.md", "object": _read()}],
            },
        })
        listed = await client.get("/ext/skill-config")
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"name": "portable-probe"}
    assert listed.json()["skills"]["portable-probe"]["description"] == "probe"
    assert (tmp_path / "portable-probe" / "SKILL.md").read_bytes() == b"abc"


@pytest.mark.asyncio
async def test_a2a_run_and_framework_trajectory_upload(monkeypatch, oc):
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)
        kwargs["event_callback"]({"type": "event"})
        return SimpleNamespace(response="OK", error=None, error_code=None,
                               session_id="ses_framework", trajectory=[{"type": "event"}],
                               tool_call_count=2)

    async def fake_upload(_target, body):
        assert json.loads(body) == [{"type": "event"}]
        return Uploaded(size_bytes=len(body))

    monkeypatch.setattr(server, "run_opencode", fake_run)
    monkeypatch.setattr("agentenv_protocol.a2a_agent.framework.upload", fake_upload)
    message = {
        "jsonrpc": "2.0", "id": "r1", "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": "m1", "role": "user",
                              "contextId": "framework-context", "parts": [{"kind": "text", "text": "hello"}]},
                   "configuration": {"blocking": True}},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        response = await client.post("/a2a", json=message)
        task = response.json()["result"]
        trajectory = await client.post("/ext/trajectory", json={
            "task_id": task["id"], "objects": {"trajectory": _write("application/json")},
        })
    assert response.status_code == 200, response.text
    assert task["status"]["state"] == "completed"
    assert task["status"]["message"]["parts"][1]["data"]["usage"]["tool_call_count"] == 2
    assert called["prompt"] == "hello"
    assert trajectory.status_code == 200, trajectory.text
    assert trajectory.json()["objects"]["trajectory"]["size_bytes"] > 0
    assert oc.agent.session_ref_for_context("framework-context") == "ses_framework"


@pytest.mark.asyncio
async def test_output_format_reaches_opencode(monkeypatch, oc):
    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(response='{"message":"done","done":true}', error=None,
                               error_code=None, session_id="ses_format", trajectory=[],
                               tool_call_count=0,
                               structured_output={"message": "done", "done": True})

    monkeypatch.setattr(server, "run_opencode", fake_run)
    schema = {"type": "json_schema", "schema": {"type": "object", "properties":
              {"message": {"type": "string"}, "done": {"type": "boolean"}},
              "required": ["message", "done"]}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        set_response = await client.post("/ext/agent-config", json={"output_format": schema})
        assert set_response.status_code == 200, set_response.text
        response = await client.post("/a2a", json={
            "jsonrpc": "2.0", "id": "fmt", "method": "message/send",
            "params": {"message": {"kind": "message", "messageId": "format-test",
                                  "role": "user", "contextId": "format-context",
                                  "parts": [{"kind": "text", "text": "hello"}]},
                       "configuration": {"blocking": True}},
        })
    assert response.status_code == 200, response.text
    assert called["output_format"] == schema
    parts = response.json()["result"]["status"]["message"]["parts"]
    assert any(part.get("data", {}).get("structured_output") ==
               {"message": "done", "done": True} for part in parts)


@pytest.mark.asyncio
async def test_bad_output_format_is_rejected_at_configuration_time(oc):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        for output_format in (
            {"type": "unsupported", "schema": {"type": "object"}},
            {"type": "json_schema", "schema": {"type": "impossible"}},
        ):
            response = await client.post("/ext/agent-config", json={"output_format": output_format})
            assert response.status_code == 400, response.text


@pytest.mark.asyncio
async def test_json_null_is_returned_as_structured_output(monkeypatch, oc):
    async def fake_run(**_kwargs):
        return SimpleNamespace(response="null", error=None, error_code=None,
                               session_id="ses_null", trajectory=[], tool_call_count=0,
                               structured_output=None)

    monkeypatch.setattr(server, "run_opencode", fake_run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        configured = await client.post("/ext/agent-config", json={
            "output_format": {"type": "json_schema", "schema": {"type": "null"}},
        })
        assert configured.status_code == 200, configured.text
        response = await client.post("/a2a", json={
            "jsonrpc": "2.0", "id": "null", "method": "message/send",
            "params": {"message": {"kind": "message", "messageId": "null-answer",
                                  "role": "user", "contextId": "null-answer",
                                  "parts": [{"kind": "text", "text": "reply null"}]},
                       "configuration": {"blocking": True}},
        })
    assert response.status_code == 200, response.text
    parts = response.json()["result"]["status"]["message"]["parts"]
    assert any(part.get("data", {}).get("structured_output", "missing") is None for part in parts)


@pytest.mark.asyncio
async def test_failed_a2a_run_keeps_trajectory(monkeypatch, oc):
    async def fake_run(**kwargs):
        kwargs["event_callback"]({"type": "error", "message": "model call failed"})
        return SimpleNamespace(
            response="", error="model call failed", error_code="model_error",
            session_id="ses_failed", trajectory=[{"type": "error", "message": "model call failed"}],
            tool_call_count=1,
        )

    monkeypatch.setattr(server, "run_opencode", fake_run)
    message = {
        "jsonrpc": "2.0", "id": "failed-request", "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": "failed-message", "role": "user",
                              "contextId": "failed-context", "parts": [{"kind": "text", "text": "go"}]},
                   "configuration": {"blocking": True}},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        task = (await client.post("/a2a", json=message)).json()["result"]
        trajectory = await client.post("/ext/trajectory", json={"task_id": task["id"]})
    assert task["status"]["state"] == "failed"
    assert trajectory.status_code == 200
    assert trajectory.json()["trajectory"] == [{"type": "error", "message": "model call failed"}]


@pytest.mark.asyncio
async def test_snapshot_routes_save_and_load_objects(monkeypatch, tmp_path, oc):
    workspace_dir = tmp_path / "app"
    workspace_dir.mkdir()
    (workspace_dir / "stale.txt").write_text("stale")
    monkeypatch.setattr(server, "WORKSPACE_DIR", workspace_dir)
    oc.agent.set_session_ref_for_context("snapshot-source", "ses_source")

    async def fake_export(_context, _session):
        return b'{"session":"source"}'

    async def fake_upload(_target, source):
        if isinstance(source, bytes):
            assert b"session" in source
        else:
            assert source.name == "workspace.tgz"
        return Uploaded(size_bytes=3)

    async def fake_download(_source, destination):
        destination.write_bytes(b'{"session":"source"}' if destination.name == "session.json" else b"tar")

    async def fake_import(_context, path):
        assert b"session" in Path(path).read_bytes()
        assert (workspace_dir / "stale.txt").exists()
        return "ses_restored"

    def fake_run(args, **_kwargs):
        if "-czf" in args:
            Path(args[args.index("-czf") + 1]).write_bytes(b"tar")
        if "-xzf" in args:
            (Path(args[args.index("-C") + 1]) / "fresh.txt").write_text("fresh")

    monkeypatch.setattr(server, "export_session", fake_export)
    monkeypatch.setattr(server, "upload", fake_upload)
    monkeypatch.setattr(server, "download", fake_download)
    monkeypatch.setattr(server, "import_session", fake_import)
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        saved = await client.post("/ext/snapshot", json={
            "context_id": "snapshot-source",
            "objects": {"trajectory": _write(), "workspace": _write()},
        })
        loaded = await client.put("/ext/snapshot", json={
            "target_context_id": "snapshot-restored",
            "objects": {"trajectory": _read(), "workspace": _read()},
        })
    assert saved.status_code == 200, saved.text
    assert set(saved.json()["objects"]) == {"trajectory", "workspace"}
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["context_id"] == "snapshot-restored"
    assert oc.agent.session_ref_for_context("snapshot-restored") == "ses_restored"
    assert not (workspace_dir / "stale.txt").exists()
    assert (workspace_dir / "fresh.txt").read_text() == "fresh"
    assert sorted(path.name for path in workspace_dir.iterdir()) == ["fresh.txt"]


@pytest.mark.asyncio
async def test_failed_session_import_keeps_live_workspace(monkeypatch, tmp_path, oc):
    workspace_dir = tmp_path / "app"
    workspace_dir.mkdir()
    (workspace_dir / "live.txt").write_text("live")
    monkeypatch.setattr(server, "WORKSPACE_DIR", workspace_dir)

    async def fake_download(_source, destination):
        destination.write_bytes(b"tar")

    def fake_run(args, **_kwargs):
        if "-xzf" in args:
            (Path(args[args.index("-C") + 1]) / "restored.txt").write_text("restored")

    async def failed_import(_context, _path):
        raise RuntimeError("invalid session")

    monkeypatch.setattr(server, "download", fake_download)
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server, "import_session", failed_import)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        response = await client.put("/ext/snapshot", json={
            "target_context_id": "bad-restore",
            "objects": {"trajectory": _read(), "workspace": _read()},
        })
    assert response.status_code == 500
    assert (workspace_dir / "live.txt").read_text() == "live"
    assert not (workspace_dir / "restored.txt").exists()


def test_failed_workspace_backup_keeps_originals(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "app"
    staged = tmp_path / "staged"
    workspace_dir.mkdir()
    staged.mkdir()
    (workspace_dir / "first.txt").write_text("first")
    (workspace_dir / "second.txt").write_text("second")
    monkeypatch.setattr(server, "WORKSPACE_DIR", workspace_dir)
    original_rename = Path.rename

    def failed_rename(source, destination):
        if source == workspace_dir / "second.txt":
            raise OSError("cross-device move refused")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", failed_rename)
    with pytest.raises(OSError, match="cross-device move refused"):
        server._replace_workspace(staged)
    assert (workspace_dir / "first.txt").read_text() == "first"
    assert (workspace_dir / "second.txt").read_text() == "second"
    assert sorted(path.name for path in workspace_dir.iterdir()) == ["first.txt", "second.txt"]


def test_failed_workspace_copy_preserves_live_files(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "app"
    staged = tmp_path / "staged"
    workspace_dir.mkdir()
    staged.mkdir()
    (workspace_dir / "live.txt").write_text("intact")
    (staged / "fresh.txt").write_text("replacement")
    monkeypatch.setattr(server, "WORKSPACE_DIR", workspace_dir)

    def failed_copy(source, destination):
        Path(destination).write_text("partial")
        raise OSError("copy failed")

    monkeypatch.setattr(server.shutil, "move", failed_copy)
    with pytest.raises(OSError, match="copy failed"):
        server._replace_workspace(staged)
    assert (workspace_dir / "live.txt").read_text() == "intact"
    assert not (workspace_dir / "fresh.txt").exists()
    assert sorted(path.name for path in workspace_dir.iterdir()) == ["live.txt"]


@pytest.mark.asyncio
async def test_backup_cleanup_failure_keeps_restored_session(monkeypatch, tmp_path, oc):
    workspace_dir = tmp_path / "app"
    workspace_dir.mkdir()
    (workspace_dir / "old.txt").write_text("old")
    monkeypatch.setattr(server, "WORKSPACE_DIR", workspace_dir)

    async def fake_download(_source, destination):
        destination.write_bytes(b"session" if destination.name == "session.json" else b"tar")

    async def fake_import(_context, _path):
        return "ses_new"

    def fake_run(args, **_kwargs):
        if "-xzf" in args:
            (Path(args[args.index("-C") + 1]) / "new.txt").write_text("new")

    original_rmtree = server.shutil.rmtree

    def failed_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".opencode-backup-"):
            raise PermissionError(f"cannot remove {path}")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(server, "download", fake_download)
    monkeypatch.setattr(server, "import_session", fake_import)
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.shutil, "rmtree", failed_cleanup)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        response = await client.put("/ext/snapshot", json={
            "target_context_id": "cleanup-failure",
            "objects": {"trajectory": _read(), "workspace": _read()},
        })
    assert response.status_code == 200, response.text
    assert oc.agent.session_ref_for_context("cleanup-failure") == "ses_new"
    assert (workspace_dir / "new.txt").read_text() == "new"
    assert not (workspace_dir / "old.txt").exists()


@pytest.mark.asyncio
async def test_snapshot_omits_backup_left_by_failed_cleanup(monkeypatch, tmp_path, oc):
    workspace_dir = tmp_path / "app"
    workspace_dir.mkdir()
    (workspace_dir / "current.txt").write_text("current")
    (workspace_dir / "report.opencode-backup-final.txt").write_text("user file")
    backup = workspace_dir / ".opencode-backup-leftover"
    backup.mkdir()
    (backup / "saved").mkdir()
    (backup / "saved" / "old.txt").write_text("old")
    monkeypatch.setattr(server, "WORKSPACE_DIR", workspace_dir)
    oc.agent.set_session_ref_for_context("backup-exclusion", "ses_current")
    archive_names = []

    async def fake_export(_context, _session):
        return b'{"session":"current"}'

    async def fake_upload(_target, source):
        if isinstance(source, Path):
            with tarfile.open(source, "r:gz") as archive:
                archive_names.extend(archive.getnames())
        return Uploaded(size_bytes=3)

    monkeypatch.setattr(server, "export_session", fake_export)
    monkeypatch.setattr(server, "upload", fake_upload)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        response = await client.post("/ext/snapshot", json={
            "context_id": "backup-exclusion",
            "objects": {"trajectory": _write(), "workspace": _write()},
        })
    assert response.status_code == 200, response.text
    assert any(name.endswith("current.txt") for name in archive_names)
    assert any(name.endswith("report.opencode-backup-final.txt") for name in archive_names)
    assert not any(name.startswith("./.opencode-backup-leftover") for name in archive_names)


def _snar(args) -> str:
    """The incremental state file a tar command line names."""
    return next(a for a in args if a.startswith("--listed-incremental=")).partition("=")[2]


@pytest.mark.asyncio
async def test_changelog_uses_portable_namespace_and_ordered_increments(monkeypatch, tmp_path, oc):
    captures = []

    class FakeUploader:
        def __init__(self, grant):
            assert grant.root_path == "captures"

        async def upload(self, path, source):
            captures.append((path, source.read_bytes()))
            return Uploaded(size_bytes=3)

    def fake_run(args, **_kwargs):
        if "-czf" in args:
            assert "--ignore-failed-read" in args
            Path(_snar(args)).write_text("advanced")
            Path(args[args.index("-czf") + 1]).write_bytes(b"tar")
        if "-xzf" in args:
            captures.append(("applied", Path(args[args.index("-xzf") + 1]).read_bytes()))
        return SimpleNamespace(stdout=b"")

    async def fake_download(_source, destination):
        destination.write_bytes(b"tar")

    monkeypatch.setattr(server, "NamespaceUploader", FakeUploader)
    def fake_baseline(_roots):
        server._CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
        server._CHANGELOG_SNAR.write_text("baseline")

    monkeypatch.setattr(server, "_baseline_changelog", fake_baseline)
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server, "download", fake_download)
    monkeypatch.setattr(server, "_CHANGELOG_DIR", tmp_path / "changelog")
    monkeypatch.setattr(server, "_CHANGELOG_SNAR", tmp_path / "changelog" / "state.snar")
    namespace = {
        "root_path": "captures", "expires_at": _grant("http-put")["expires_at"],
        "max_objects": 10, "max_object_bytes": 1024, "max_total_bytes": 10240,
        "write": {"kind": "http-post-policy", "url": "https://objects.example.test/upload",
                  "fields": {"key": "captures/"}, "path_field": "key", "file_field": "file"},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        enabled = await client.post("/ext/snapshot/changelog", json={
            "write_namespace": namespace, "roots": [str(tmp_path)],
        })
        await oc.agent._capture_changelog()
        applied = await client.put("/ext/snapshot/changelog", json={
            "increments": [{"sequence": 0, "object": _read()}],
        })
    assert enabled.status_code == 200, enabled.text
    assert captures == [("000000.tgz", b"tar"), ("applied", b"tar")]
    assert server._CHANGELOG_SNAR.read_text() == "advanced"
    assert applied.status_code == 200, applied.text
    assert applied.json()["count"] == 1


@pytest.mark.asyncio
async def test_failed_changelog_upload_preserves_incremental_baseline(monkeypatch, tmp_path, oc):
    directory = tmp_path / "changelog"
    directory.mkdir()
    baseline = directory / "state.snar"
    baseline.write_text("baseline")
    monkeypatch.setattr(server, "_CHANGELOG_DIR", directory)
    monkeypatch.setattr(server, "_CHANGELOG_SNAR", baseline)
    starting_states = []

    def fake_run(args, **_kwargs):
        pending = Path(_snar(args))
        starting_states.append(pending.read_text())
        pending.write_text("advanced")
        Path(args[args.index("-czf") + 1]).write_bytes(b"tar")

    class FlakyUploader:
        attempts = 0

        async def upload(self, _path, _source):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("upload failed")
            return Uploaded(size_bytes=3)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    config = {"roots": [str(tmp_path)], "present": [str(tmp_path)], "sequence": 0, "uploader": FlakyUploader()}
    monkeypatch.setattr(oc.agent, "_changelog", config)
    await oc.agent._capture_changelog()
    assert baseline.read_text() == "baseline"
    assert config["sequence"] == 0
    await oc.agent._capture_changelog()
    assert starting_states == ["baseline", "baseline"]
    assert baseline.read_text() == "advanced"
    assert config["sequence"] == 1


@pytest.mark.asyncio
async def test_snapshot_waits_for_active_workspace_turn(monkeypatch, oc):
    started = asyncio.Event()
    release = asyncio.Event()
    exported = asyncio.Event()
    oc.agent._changelog = None
    oc.agent.set_session_ref_for_context("locked-context", "ses_locked")

    async def fake_run(**_kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(response="OK", error=None, error_code=None,
                               session_id="ses_locked", trajectory=[], tool_call_count=0)

    async def fake_export(_context, _session):
        exported.set()
        return b"session"

    async def fake_upload(_target, _source):
        return Uploaded(size_bytes=7)

    monkeypatch.setattr(server, "run_opencode", fake_run)
    monkeypatch.setattr(server, "export_session", fake_export)
    monkeypatch.setattr(server, "upload", fake_upload)
    turn = asyncio.create_task(oc.agent.run(TaskRequest(
        task_id="locked-task", context_id="locked-context", parts=(TextPart(text="go"),),
        config=server.OpenCodeConfig(),
    )))
    await started.wait()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=oc.app), base_url="http://test") as client:
        save = asyncio.create_task(client.post("/ext/snapshot", json={
            "context_id": "locked-context", "objects": {"trajectory": _write()},
        }))
        await asyncio.sleep(0)
        assert not exported.is_set()
        release.set()
        assert (await save).status_code == 200
    await turn
    assert exported.is_set()


class _KeptUploads:
    """A namespace uploader that keeps each increment it is handed."""

    def __init__(self, _grant=None) -> None:
        self.objects: dict[str, bytes] = {}

    async def upload(self, path, source):
        self.objects[path] = source.read_bytes()
        return Uploaded(size_bytes=len(self.objects[path]))


@pytest.mark.skipif(not server.gnu_tar(), reason="the changelog's incremental archives need GNU tar")
@pytest.mark.asyncio
async def test_a_changelog_replays_each_turns_changes_deletions_included(monkeypatch, oc, folders):
    uploads = _KeptUploads()
    monkeypatch.setattr(server, "NamespaceUploader", lambda _grant: uploads)
    (folders / "kept.txt").write_text("before")
    (folders / "removed.txt").write_text("going")
    second = folders.parent / "second-root"
    second.mkdir()
    (second / "inside.txt").write_text("x")
    namespace = {
        "root_path": "captures", "expires_at": _grant("http-put")["expires_at"],
        "max_objects": 10, "max_object_bytes": 1 << 20, "max_total_bytes": 1 << 22,
        "write": {"kind": "http-post-policy", "url": "https://objects.example.test/upload",
                  "fields": {"key": "captures/"}, "path_field": "key", "file_field": "file"},
    }
    async with oc.client() as client:
        enabled = await client.post("/ext/snapshot/changelog", json={"write_namespace": namespace,
                                                                     "roots": [str(folders), str(second)]})
        (folders / "kept.txt").write_text("after")
        # A file's mtime comes from a coarser clock than the baseline's time, so a rewrite this soon after it can read
        # as no later; a turn's writes come well after.
        later = time.time() + 2
        os.utime(folders / "kept.txt", (later, later))
        (folders / "removed.txt").unlink()
        (folders / "added.txt").write_text("new")
        shutil.rmtree(second)
        await oc.agent._capture_changelog()

    assert enabled.status_code == 200, enabled.text
    assert list(uploads.objects) == ["000000.tgz"]
    (folders / "kept.txt").write_text("before")
    (folders / "removed.txt").write_text("going")
    (folders / "added.txt").unlink()
    second.mkdir()
    (second / "inside.txt").write_text("x")

    async def from_uploads(source, destination):
        destination.write_bytes(uploads.objects["000000.tgz"])

    monkeypatch.setattr(server, "download", from_uploads)
    async with oc.client() as client:
        applied = await client.put("/ext/snapshot/changelog", json={"increments": [{"sequence": 0, "object": _read()}]})

    assert applied.status_code == 200, applied.text
    assert sorted(path.name for path in folders.iterdir()) == ["added.txt", "kept.txt"]
    assert (folders / "kept.txt").read_text() == "after"
    assert not second.exists()
    assert not server._CHANGELOG_META_DIR.exists() or not any(server._CHANGELOG_META_DIR.iterdir())


@pytest.mark.skipif(not server.gnu_tar(), reason="the changelog's incremental archives need GNU tar")
@pytest.mark.asyncio
async def test_a_turn_that_removes_every_root_still_uploads_their_removal(monkeypatch, oc, folders):
    uploads = _KeptUploads()
    monkeypatch.setattr(server, "NamespaceUploader", lambda _grant: uploads)
    root = folders / "only-root"
    root.mkdir()
    (root / "a.txt").write_text("a")
    namespace = {
        "root_path": "captures", "expires_at": _grant("http-put")["expires_at"],
        "max_objects": 10, "max_object_bytes": 1 << 20, "max_total_bytes": 1 << 22,
        "write": {"kind": "http-post-policy", "url": "https://objects.example.test/upload",
                  "fields": {"key": "captures/"}, "path_field": "key", "file_field": "file"},
    }
    async with oc.client() as client:
        await client.post("/ext/snapshot/changelog", json={"write_namespace": namespace, "roots": [str(root)]})
        shutil.rmtree(root)
        await oc.agent._capture_changelog()
        await oc.agent._capture_changelog()

    assert list(uploads.objects) == ["000000.tgz"]
    root.mkdir()

    async def from_uploads(source, destination):
        destination.write_bytes(uploads.objects["000000.tgz"])

    monkeypatch.setattr(server, "download", from_uploads)
    async with oc.client() as client:
        applied = await client.put("/ext/snapshot/changelog", json={"increments": [{"sequence": 0, "object": _read()}]})

    assert applied.status_code == 200, applied.text
    assert not root.exists()


@pytest.mark.parametrize("path", ["/", "/etc", "relative/path", "/tmp/../etc", 7])
def test_a_record_never_removes_the_root_of_the_system(path):
    assert not server._removable(path)


@pytest.mark.parametrize("member, record", [
    ("tmp/opencode-changelog-8000.changelog-meta/000003/opencode-changelog-meta.json", True),
    ("var/tmp/opencode-changelog-default.changelog-meta/000000/opencode-changelog-meta.json", True),
    ("app/opencode-changelog-meta.json", False),
    ("app/notes.changelog-meta/opencode-changelog-meta.json", False),
    ("app/x.changelog-meta/000001/opencode-changelog-meta.json.bak", False),
])
def test_a_record_is_its_reserved_directory_and_name_not_any_file_of_that_name(member, record):
    assert bool(server._META_MEMBER.search(member)) is record


@pytest.mark.skipif(not server.gnu_tar(), reason="the changelog's incremental archives need GNU tar")
@pytest.mark.asyncio
async def test_a_workspace_file_named_like_a_record_is_restored_and_removes_nothing(monkeypatch, oc, folders):
    uploads = _KeptUploads()
    monkeypatch.setattr(server, "NamespaceUploader", lambda _grant: uploads)
    bystander = folders.parent / "bystander"
    bystander.mkdir()
    namespace = {
        "root_path": "captures", "expires_at": _grant("http-put")["expires_at"],
        "max_objects": 10, "max_object_bytes": 1 << 20, "max_total_bytes": 1 << 22,
        "write": {"kind": "http-post-policy", "url": "https://objects.example.test/upload",
                  "fields": {"key": "captures/"}, "path_field": "key", "file_field": "file"},
    }
    async with oc.client() as client:
        await client.post("/ext/snapshot/changelog", json={"write_namespace": namespace, "roots": [str(folders)]})
        planted = folders / "opencode-changelog-meta.json"
        planted.write_text(json.dumps({"removed": [str(bystander)]}))
        later = time.time() + 2
        os.utime(planted, (later, later))
        await oc.agent._capture_changelog()
    planted.unlink()

    async def from_uploads(source, destination):
        destination.write_bytes(uploads.objects["000000.tgz"])

    monkeypatch.setattr(server, "download", from_uploads)
    async with oc.client() as client:
        applied = await client.put("/ext/snapshot/changelog", json={"increments": [{"sequence": 0, "object": _read()}]})

    assert applied.status_code == 200, applied.text
    assert json.loads(planted.read_text()) == {"removed": [str(bystander)]}
    assert bystander.exists()
