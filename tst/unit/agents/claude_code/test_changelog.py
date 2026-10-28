"""snapshot/v1's changelog: the capture hook's increments, their replay onto a fresh agent, and the agent that serves
it only where the host's tar is GNU tar."""

from __future__ import annotations

import asyncio
import email.parser
import email.policy
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agentenv_protocol.a2a_agent import NamespaceChangelogEnableRequest, ObjectChangelogApplyRequest
from agentenv_protocol.transfers import TransferError
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge, tar_changelog
from tst.unit.agents.claude_code.support import expiry, extension_methods, read_grant

GNU_TAR = shutil.which("tar") is not None and tar_changelog.gnu_tar()
needs_gnu_tar = pytest.mark.skipif(not GNU_TAR, reason="the changelog is a GNU incremental tar")
STAGING_PATH = "/staging/changelog"
ROOT_PATH = "agent_changelog/run-1/solver"


def namespace_grant(**limits) -> dict:
    """A namespace grant on the agent's own staging, so its uploads reach a local server over loopback."""
    return {
        "root_path": ROOT_PATH,
        "expires_at": expiry(),
        "max_objects": limits.get("max_objects", 10),
        "max_object_bytes": limits.get("max_object_bytes", 1 << 20),
        "max_total_bytes": limits.get("max_total_bytes", 1 << 22),
        "write": {
            "kind": "http-post-policy",
            "url": f"https://bucket.example.com{STAGING_PATH}",
            "fields": {"policy": "p", "signature": "s"},
            "path_field": "key",
            "file_field": "file",
            "headers": {"AgentEnv-Staging-Path": STAGING_PATH},
        },
    }


class Staging:
    """A local server taking the namespace's multipart POSTs: each object's bytes, by the path it was posted to."""

    def __init__(self, status: int = 204) -> None:
        self.objects: dict[str, bytes] = {}
        self.status = status
        staging = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["content-length"]))
                message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    f"content-type: {self.headers['content-type']}\r\n\r\n".encode() + body)
                fields = {part.get_param("name", header="content-disposition"): part.get_content()
                          for part in message.iter_parts()}
                if staging.status < 300:
                    staging.objects[fields["key"]] = fields["file"]
                self.send_response(staging.status)
                self.send_header("content-length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self.server.server_port

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def staging(socket_enabled):
    """The hook uploads to a server on loopback, as it reaches the agent's own staging."""
    server = Staging()
    yield server
    server.close()


def _changelog(tmp_path, roots=None):
    installed: list[str] = []
    after: list[bool] = []
    resumed: list[tuple[str | None, str | None]] = []

    async def resume(target_context_id, session):
        resumed.append((target_context_id, session))
        return target_context_id or "minted"

    log = tar_changelog.Changelog(
        default_roots=roots or [str(tmp_path / "default-root")],
        install_hook=installed.append,
        resume=resume,
        directory=tmp_path / "changelog",
        after_apply=lambda: after.append(True),
    )
    return log, installed, after, resumed


def _enable(log, roots=None, **limits):
    body = {"write_namespace": namespace_grant(**limits), **({"roots": roots} if roots is not None else {})}
    return asyncio.run(log.enable(NamespaceChangelogEnableRequest.model_validate(body)))


def _capture(log, port: int, tool: str = "Bash", session: str = "session-1", path: str | None = None
             ) -> subprocess.CompletedProcess:
    """One run of the post-tool hook, as the CLI starts it: the agent's environment, the hook's JSON on stdin."""
    env = {**os.environ, "A2A_PORT": str(port), **({"PATH": path} if path else {})}
    return subprocess.run(
        [sys.executable, log.capture_path], input=json.dumps({"tool_name": tool, "session_id": session}),
        capture_output=True, text=True, env=env, timeout=120,
    )


def _members(archive: bytes) -> list[str]:
    listing = subprocess.run(["tar", "--incremental", "-tzf", "-"], input=archive, capture_output=True, check=True)
    return [line for line in listing.stdout.decode().splitlines() if not line.endswith("/")]


def _tgz(*paths) -> bytes:
    return subprocess.run(["tar", "-czf", "-", *map(str, paths)], capture_output=True, check=True).stdout


def test_the_hook_waits_long_enough_for_its_turn_and_the_script_reads_its_own_config(tmp_path):
    log, _, _, _ = _changelog(tmp_path)
    assert tar_changelog.HOOK_TIMEOUT_S == 600
    assert f"CFG = {json.dumps(str(tmp_path / 'changelog' / 'config.json'))}" in log.capture_script()
    assert log.hook_command.endswith(f"{tmp_path / 'changelog' / 'capture.py'}")
    assert log.hook_command.split()[0] == sys.executable


def test_the_changelog_directory_is_one_per_port():
    assert tar_changelog.changelog_dir({"A2A_PORT": "8123"}).name == "claude-code-changelog-8123"
    assert tar_changelog.changelog_dir({"A2A_PORT": "8123"}) != tar_changelog.changelog_dir({"A2A_PORT": "8124"})
    assert tar_changelog.changelog_dir({}).name == "claude-code-changelog-default"


def test_outermost_keeps_each_path_once_and_none_inside_another():
    assert tar_changelog.outermost(["/home/agent/.claude", "/app", "/home/agent", "/app/", "/etc"]) == [
        "/app", "/home/agent", "/etc"]


def test_merge_adds_the_hook_once_and_keeps_the_rest(tmp_path):
    path = tmp_path / "cli" / "settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"model": "m", "hooks": {"Other": [{"matcher": "x", "hooks": []}]}}))

    for _ in range(2):
        tar_changelog.merge_settings_hook(path, event="PostToolUse", matcher="", command="py /c.py", timeout=600)

    assert json.loads(path.read_text()) == {
        "model": "m",
        "hooks": {
            "Other": [{"matcher": "x", "hooks": []}],
            "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "py /c.py", "timeout": 600}]}],
        },
    }


def test_merge_replaces_an_unreadable_settings_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json")

    tar_changelog.merge_settings_hook(path, event="PostToolUse", matcher="", command="c", timeout=1)

    assert json.loads(path.read_text()) == {
        "hooks": {"PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "c", "timeout": 1}]}]}
    }


def test_remove_drops_only_the_hooks_whose_command_matches(tmp_path):
    path = tmp_path / "settings.json"
    keep = {"matcher": "", "hooks": [{"type": "command", "command": "other"}]}
    drop = {"matcher": "", "hooks": [{"type": "command", "command": "py /tmp/claude-code-changelog-8000/capture.py"}]}
    path.write_text(json.dumps({"hooks": {"PostToolUse": [keep, drop]}}))

    assert tar_changelog.remove_settings_hooks(path, event="PostToolUse", matches=tar_changelog.is_capture_command) is True
    assert json.loads(path.read_text()) == {"hooks": {"PostToolUse": [keep]}}
    assert tar_changelog.remove_settings_hooks(path, event="PostToolUse", matches=tar_changelog.is_capture_command) is False
    assert tar_changelog.remove_settings_hooks(tmp_path / "absent.json", event="PostToolUse",
                                           matches=tar_changelog.is_capture_command) is False


@needs_gnu_tar
def test_enable_writes_the_script_and_config_baselines_and_installs_the_hook(tmp_path):
    log, installed, _, _ = _changelog(tmp_path)
    (tmp_path / "default-root").mkdir()
    directory = tmp_path / "changelog"
    directory.mkdir()
    for stale in ("seq", "turn", "namespace.json", "held.3", "state.snar.2"):
        (directory / stale).write_text("0")

    response = _enable(log)

    assert response.roots == [str(tmp_path / "default-root")]
    assert installed == [log.hook_command]
    assert (directory / "capture.py").read_text() == log.capture_script()
    config = json.loads((directory / "config.json").read_text())
    assert set(config) == {"dir", "snar", "roots", "epoch", "write_namespace", "meta", "meta_explicit"}
    assert (config["dir"], config["snar"]) == (str(directory), str(directory / "state.snar"))
    assert config["write_namespace"]["root_path"] == ROOT_PATH
    assert (directory / "state.snar").is_file()
    assert not any((directory / stale).exists() for stale in ("seq", "turn", "namespace.json", "held.3", "state.snar.2"))
    # The config holds the grant, so no other user reads the directory.
    assert directory.stat().st_mode & 0o777 == 0o700


def test_enable_refuses_an_empty_root_list(tmp_path):
    log, installed, _, _ = _changelog(tmp_path)

    with pytest.raises(HTTPException) as raised:
        _enable(log, roots=[])

    assert (raised.value.status_code, raised.value.detail) == (
        400, "roots must be a non-empty list of strings (omit to use defaults)")
    assert installed == []


@needs_gnu_tar
def test_each_enable_starts_a_new_epoch(tmp_path):
    log, _, _, _ = _changelog(tmp_path)
    config = tmp_path / "changelog" / "config.json"

    _enable(log, roots=[str(tmp_path)])
    first = json.loads(config.read_text())["epoch"]
    _enable(log, roots=[str(tmp_path)])

    assert json.loads(config.read_text())["epoch"] != first


@needs_gnu_tar
def test_each_tool_call_uploads_its_own_writes_and_the_increments_replay_them(tmp_path, staging, store):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "before.txt").write_text("there before the capture")
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log)

    (root / "first.txt").write_text("one")
    first = _capture(log, staging.port)
    (root / "second.txt").write_text("two")
    (root / "first.txt").unlink()
    second = _capture(log, staging.port, tool="Write")

    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    assert "captured seq=0 tool=Bash" in first.stderr and "captured seq=1 tool=Write" in second.stderr
    assert sorted(staging.objects) == [f"{ROOT_PATH}/000000.tgz", f"{ROOT_PATH}/000001.tgz"]
    meta = str(log.meta_path).lstrip("/")
    assert sorted(_members(staging.objects[f"{ROOT_PATH}/000000.tgz"])) == sorted([str(root / "first.txt").lstrip("/"), meta])
    assert sorted(_members(staging.objects[f"{ROOT_PATH}/000001.tgz"])) == sorted([str(root / "second.txt").lstrip("/"), meta])
    assert not list((tmp_path / "changelog").glob("*.tgz"))

    # Replayed onto the state before the capture, the increments reproduce the state after the last tool call.
    (root / "second.txt").unlink()
    (root / "first.txt").write_text("stale")
    increments = []
    for sequence in (0, 1):
        key = f"ns/{sequence:06d}.tgz"
        store.objects[key] = staging.objects[f"{ROOT_PATH}/{sequence:06d}.tgz"]
        increments.append({"sequence": sequence, "object": read_grant(key, len(store.objects[key]))})
    log.meta_path.unlink()
    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert answer.count == 2
    assert sorted(p.name for p in root.iterdir()) == ["before.txt", "second.txt"]
    assert (root / "second.txt").read_text() == "two"
    assert not log.meta_path.exists()


@needs_gnu_tar
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_file_the_agent_cant_read_is_skipped_and_the_rest_still_captured(tmp_path, staging):
    """As /etc/ssl/private is to a non-root agent: tar can't read it, and the increment still uploads the rest."""
    root = tmp_path / "workspace"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (locked / "key").write_text("secret")
    locked.chmod(0)
    try:
        log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
        _enable(log)
        (root / "marker.txt").write_text("token")
        result = _capture(log, staging.port)
    finally:
        locked.chmod(0o700)

    assert result.returncode == 0 and "captured seq=0" in result.stderr, result.stderr
    assert str(root / "marker.txt").lstrip("/") in _members(staging.objects[f"{ROOT_PATH}/000000.tgz"])


@needs_gnu_tar
def test_a_capture_past_the_grants_object_limit_is_skipped_across_hook_processes(tmp_path, staging):
    root = tmp_path / "workspace"
    root.mkdir()
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log, max_objects=1)

    (root / "first.txt").write_text("one")
    first = _capture(log, staging.port)
    (root / "second.txt").write_text("two")
    second = _capture(log, staging.port)

    assert first.returncode == second.returncode == 0
    assert sorted(staging.objects) == [f"{ROOT_PATH}/000000.tgz"]
    assert "skipped seq=1: past the grant's limits" in second.stderr
    assert not list((tmp_path / "changelog").glob("*.tgz"))
    assert not list((tmp_path / "changelog").glob("state.snar.*"))


@needs_gnu_tar
def test_a_refused_upload_gives_back_its_reservation_and_exits_zero(tmp_path, socket_enabled):
    root = tmp_path / "workspace"
    root.mkdir()
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log, max_objects=1)
    refusing = Staging(status=403)

    (root / "first.txt").write_text("one")
    try:
        refused = _capture(log, refusing.port)
    finally:
        refusing.close()

    assert refused.returncode == 0 and "upload of seq=0 failed; the next capture carries its writes" in refused.stderr
    assert json.loads((tmp_path / "changelog" / "namespace.json").read_text()) == {}
    assert not list((tmp_path / "changelog").glob("*.tgz"))
    assert not list((tmp_path / "changelog").glob("state.snar.*"))


@needs_gnu_tar
def test_the_writes_of_a_capture_that_wasnt_stored_go_with_the_next_one(tmp_path, staging):
    root = tmp_path / "workspace"
    root.mkdir()
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log)
    baseline = (tmp_path / "changelog" / "state.snar").read_bytes()
    refusing = Staging(status=403)

    (root / "first.txt").write_text("one")
    try:
        _capture(log, refusing.port)
    finally:
        refusing.close()
    unchanged = (tmp_path / "changelog" / "state.snar").read_bytes()
    (root / "second.txt").write_text("two")
    stored = _capture(log, staging.port)

    assert unchanged == baseline
    assert stored.returncode == 0 and "captured seq=1" in stored.stderr
    assert sorted(staging.objects) == [f"{ROOT_PATH}/000001.tgz"]
    assert sorted(m for m in _members(staging.objects[f"{ROOT_PATH}/000001.tgz"])
                  if not m.endswith(tar_changelog.META_NAME)) == [
        str(root / "first.txt").lstrip("/"), str(root / "second.txt").lstrip("/")]


@needs_gnu_tar
def test_apply_extracts_in_order_then_runs_after_apply_and_resumes_only_when_asked(tmp_path, store):
    target = tmp_path / "restored" / "file.txt"
    target.parent.mkdir()
    target.write_text("final")
    store.objects["ns/000003.tgz"] = _tgz(target)
    target.unlink()
    log, _, after, resumed = _changelog(tmp_path)
    increments = [{"sequence": 3, "object": read_grant("ns/000003.tgz", len(store.objects["ns/000003.tgz"]))}]

    plain = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))
    resumed_answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(
        {"increments": [], "resume_conversation": True, "target_context_id": "ctx"})))

    assert plain.model_dump(exclude_none=True) == {"count": 1}
    assert resumed_answer.model_dump(exclude_none=True) == {"count": 0, "context_id": "ctx"}
    assert target.read_text() == "final"
    assert after == [True, True] and resumed == [("ctx", None)]
    assert not any((tmp_path / "changelog" / "apply").iterdir())


def test_a_failed_increment_download_is_the_frameworks_typed_error(tmp_path, store):
    log, _, after, _ = _changelog(tmp_path)

    with pytest.raises(TransferError) as raised:
        asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(
            {"increments": [{"sequence": 0, "object": read_grant("ns/missing.tgz", 1)}]})))

    assert raised.value.code == "transfer_rejected" and after == []


@needs_gnu_tar
def test_an_increment_that_restores_only_partly_still_counts_and_is_logged(tmp_path, store, caplog):
    blocked = tmp_path / "restored" / "blocked"
    blocked.parent.mkdir()
    blocked.write_text("from the capture")
    store.objects["ns/000000.tgz"] = _tgz(blocked)
    blocked.unlink()
    blocked.mkdir()
    (blocked / "keeps-the-directory").write_text("")
    log, _, _, _ = _changelog(tmp_path)
    increments = [{"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))}]

    with caplog.at_level("WARNING", logger="tar_changelog"):
        answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert answer.model_dump(exclude_none=True) == {"count": 1}
    warning = next(record for record in caplog.records if record.levelname == "WARNING")
    assert "increment 000000: tar exited" in warning.getMessage()
    assert not any((tmp_path / "changelog" / "apply").iterdir())


@needs_gnu_tar
def test_an_unreadable_increment_fails_the_apply(tmp_path, store):
    first = tmp_path / "restored" / "first.txt"
    first.parent.mkdir()
    first.write_text("from increment 0")
    store.objects["ns/000000.tgz"] = _tgz(first)
    first.unlink()
    store.objects["ns/000001.tgz"] = b"not a gzip archive"
    log, _, after, _ = _changelog(tmp_path)
    increments = [
        {"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))},
        {"sequence": 1, "object": read_grant("ns/000001.tgz", len(store.objects["ns/000001.tgz"]))},
    ]

    with pytest.raises(HTTPException) as raised:
        asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert raised.value.status_code == 422
    assert raised.value.detail.startswith("changelog increment 000001 is not a readable archive: ")
    assert after == []
    assert not any((tmp_path / "changelog" / "apply").iterdir())


@needs_gnu_tar
def test_each_increment_is_removed_before_the_next_is_fetched(tmp_path, store, monkeypatch):
    restored = [tmp_path / "restored" / f"file-{n}.txt" for n in range(3)]
    restored[0].parent.mkdir()
    for n, path in enumerate(restored):
        path.write_text(str(n))
        store.objects[f"ns/{n:06d}.tgz"] = _tgz(path)
        path.unlink()
    apply_dir = tmp_path / "changelog" / "apply"
    on_disk_at_each_fetch: list[list[str]] = []
    serve = type(store).__call__

    def observing(fake, request):
        on_disk_at_each_fetch.append(sorted(p.name for p in apply_dir.iterdir()))
        return serve(fake, request)

    monkeypatch.setattr(type(store), "__call__", observing)
    log, _, _, _ = _changelog(tmp_path)
    increments = [
        {"sequence": n, "object": read_grant(f"ns/{n:06d}.tgz", len(store.objects[f"ns/{n:06d}.tgz"]))}
        for n in range(3)
    ]

    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert answer.model_dump(exclude_none=True) == {"count": 3}
    for n, names in enumerate(on_disk_at_each_fetch):
        assert all(name.startswith(f".{n:06d}.tgz.") and name.endswith(".part") for name in names), names
    assert [path.read_text() for path in restored] == ["0", "1", "2"]


@pytest.mark.parametrize(("gnu", "expected"), [(True, "ClaudeCodeChangelogAgent"), (False, "ClaudeCodeAgent")])
def test_the_changelog_is_served_only_where_tar_is_gnu_tar(monkeypatch, gnu, expected):
    monkeypatch.setattr(a2a_server, "gnu_tar", lambda: gnu)

    assert type(a2a_server.agent_for_this_host()).__name__ == expected


def test_the_changelog_agent_advertises_both_methods_and_the_base_agent_neither(app):
    changelog_app = a2a_server.build_app(a2a_server.ClaudeCodeChangelogAgent())

    assert extension_methods(changelog_app, "urn:agentenv:snapshot/v1")["enable-changelog"] == {
        "method": "POST", "endpoint": "/ext/snapshot/changelog",
        "request": {"required": ["write_namespace"], "optional": ["roots"]}, "response": {"required": ["roots"]},
    }
    assert extension_methods(changelog_app, "urn:agentenv:snapshot/v1")["apply-changelog"] == {
        "method": "PUT", "endpoint": "/ext/snapshot/changelog",
        "request": {"required": ["increments"], "optional": ["resume_conversation", "target_context_id"]},
        "response": {"required": ["count"], "optional": ["context_id"]},
    }
    assert sorted(extension_methods(app, "urn:agentenv:snapshot/v1")) == ["load", "save"]


def test_the_default_roots_hold_the_workspace_and_the_clis_state(agent_dirs):
    roots = a2a_server._changelog_roots()

    assert str(a2a_server.WORKSPACE_DIR) in roots
    assert any(str(agent_dirs) == root or str(agent_dirs).startswith(root + "/") for root in roots)


def _changelog_client(monkeypatch, tmp_path) -> tuple[TestClient, a2a_server.ClaudeCodeChangelogAgent]:
    monkeypatch.setenv("A2A_PORT", "8000")
    monkeypatch.setattr(a2a_server, "changelog_dir", lambda environ: tmp_path / "changelog")
    agent = a2a_server.ClaudeCodeChangelogAgent()
    return TestClient(a2a_server.build_app(agent)), agent


def test_apply_resumes_the_restored_transcript_and_drops_the_restored_capture_hook(monkeypatch, tmp_path, agent_dirs):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    sessions = a2a_server.SESSION_JSONL_DIR
    sessions.mkdir(parents=True)
    (sessions / "captured.jsonl").write_text(
        json.dumps({"sessionId": "old", "type": "user"}) + "\n" + json.dumps({"sessionId": "old", "type": "assistant"}))
    restored_hook = f"python3 {tmp_path / 'claude-code-changelog-9' / 'capture.py'}"
    (agent_dirs / "settings.json").write_text(json.dumps(
        {"hooks": {"PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": restored_hook}]}]}}))

    response = client.put("/ext/snapshot/changelog",
                          json={"increments": [], "resume_conversation": True, "target_context_id": "ctx-r"})

    assert response.status_code == 200, response.text
    assert response.json() == {"count": 0, "context_id": "ctx-r"}
    session = agent.session_ref_for_context("ctx-r")
    assert session and session != "old"
    assert [json.loads(line)["sessionId"] for line in (sessions / f"{session}.jsonl").read_text().splitlines()] == [
        session, session]
    assert json.loads((agent_dirs / "settings.json").read_text()) == {"hooks": {"PostToolUse": []}}


def test_apply_keeps_a_capture_hook_whose_script_this_agent_has(monkeypatch, tmp_path, agent_dirs):
    client, _ = _changelog_client(monkeypatch, tmp_path)
    own = tmp_path / "claude-code-changelog-8000" / "capture.py"
    own.parent.mkdir()
    own.write_text("")
    settings = {"hooks": {"PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": f"python3 {own}"}]}]}}
    agent_dirs.mkdir(parents=True)
    (agent_dirs / "settings.json").write_text(json.dumps(settings))

    response = client.put("/ext/snapshot/changelog", json={"increments": []})

    assert response.status_code == 200
    assert json.loads((agent_dirs / "settings.json").read_text()) == settings


def test_apply_without_a_restored_transcript_resumes_nothing(monkeypatch, tmp_path):
    client, _ = _changelog_client(monkeypatch, tmp_path)

    response = client.put("/ext/snapshot/changelog", json={"increments": [], "resume_conversation": True})

    assert response.status_code == 200 and response.json() == {"count": 0}


def test_apply_refuses_while_a_turn_runs_and_an_empty_target(monkeypatch, tmp_path):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    agent._active_contexts.add("busy")
    running = client.put("/ext/snapshot/changelog", json={"increments": []})
    agent._active_contexts.clear()
    empty = client.put("/ext/snapshot/changelog", json={"increments": [], "target_context_id": ""})

    assert (running.status_code, running.json()["detail"]) == (
        409, "cannot apply a changelog while an agent turn is running")
    assert empty.status_code == 400


def test_apply_rebinds_a_target_context_already_bound(monkeypatch, tmp_path):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    _restored_transcript({"sessionId": "captured"})
    agent.set_session_ref_for_context("rewound", "session-before")

    response = client.put("/ext/snapshot/changelog",
                          json={"increments": [], "resume_conversation": True, "target_context_id": "rewound"})

    assert response.status_code == 200 and response.json() == {"count": 0, "context_id": "rewound"}
    assert agent.session_ref_for_context("rewound") not in (None, "session-before")


def _restored_transcript(*records: dict, name: str = "captured.jsonl", extra: str = "") -> None:
    sessions = a2a_server.SESSION_JSONL_DIR
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / name).write_text("".join(json.dumps(record) + "\n" for record in records) + extra)


def test_apply_resumes_the_largest_restored_transcript_with_its_session_ids_rewritten(monkeypatch, tmp_path):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    _restored_transcript({"sessionId": "small"}, name="small.jsonl")
    _restored_transcript({"sessionId": "large", "message": "x" * 100}, name="large.jsonl", extra="\nnot json\n")

    result = client.put("/ext/snapshot/changelog",
                        json={"increments": [], "resume_conversation": True, "target_context_id": "rewound"}).json()

    assert result == {"count": 0, "context_id": "rewound"}
    session_id = agent.session_ref_for_context("rewound")
    lines = (a2a_server.SESSION_JSONL_DIR / f"{session_id}.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"sessionId": session_id, "message": "x" * 100} and lines[1] == "not json"


def test_apply_without_a_target_mints_the_context_it_resumes(monkeypatch, tmp_path):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    _restored_transcript({"sessionId": "captured"})

    result = client.put("/ext/snapshot/changelog", json={"increments": [], "resume_conversation": True}).json()

    assert result["count"] == 0 and result["context_id"]
    assert agent.session_ref_for_context(result["context_id"]) is not None


def _hooks(command: str) -> dict:
    return {"hooks": {event: [{"matcher": "", "hooks": [{"type": "command", "command": command}]}]
                      for event in ("PostToolUse", "PostToolUseFailure")}}


def test_enable_installs_the_capture_after_both_a_tool_that_succeeded_and_one_that_failed(
        monkeypatch, tmp_path, agent_dirs):
    client, agent = _changelog_client(monkeypatch, tmp_path)

    response = client.post("/ext/snapshot/changelog", json={"write_namespace": namespace_grant()})

    assert response.status_code == 200, response.text
    settings = json.loads((agent_dirs / "settings.json").read_text())
    for event in ("PostToolUse", "PostToolUseFailure"):
        assert settings["hooks"][event] == [{"matcher": "", "hooks": [
            {"type": "command", "command": agent._changelog.hook_command, "timeout": tar_changelog.HOOK_TIMEOUT_S}]}]


def test_apply_drops_the_restored_capture_hooks_of_both_events(monkeypatch, tmp_path, agent_dirs):
    client, _ = _changelog_client(monkeypatch, tmp_path)
    agent_dirs.mkdir(parents=True)
    (agent_dirs / "settings.json").write_text(json.dumps(_hooks(f"python3 {tmp_path / 'claude-code-changelog-9' / 'capture.py'}")))

    response = client.put("/ext/snapshot/changelog", json={"increments": []})

    assert response.status_code == 200
    assert json.loads((agent_dirs / "settings.json").read_text()) == {
        "hooks": {"PostToolUse": [], "PostToolUseFailure": []}}


def test_a_snapshot_load_drops_the_capture_hooks_it_restored(monkeypatch, tmp_path, agent, client, store):
    source_workspace, source_state = tmp_path / "source" / "app", tmp_path / "source" / "home" / ".claude"
    (source_state / "projects" / "-app").mkdir(parents=True)
    source_workspace.mkdir(parents=True)
    kept = {"matcher": "", "hooks": [{"type": "command", "command": "echo kept"}]}
    settings = _hooks(f"python3 {tmp_path / 'claude-code-changelog-9' / 'capture.py'}")
    settings["hooks"]["PostToolUse"].append(kept)
    (source_state / "settings.json").write_text(json.dumps(settings))
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", source_workspace)
    monkeypatch.setattr(a2a_server, "CLAUDE_STATE_DIR", source_state)
    archive = tmp_path / "workspace.tar.gz"
    a2a_server._build_workspace_archive(archive)
    destination_state = tmp_path / "destination" / "home" / ".claude"
    (tmp_path / "destination" / "app").mkdir(parents=True)
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", tmp_path / "destination" / "app")
    monkeypatch.setattr(a2a_server, "CLAUDE_STATE_DIR", destination_state)
    conversation = json.dumps({"type": "user", "sessionId": "s"}).encode() + b"\n"
    store.objects.update({"snap/workspace": archive.read_bytes(), "snap/trajectory": conversation})

    response = client.put(a2a_server.SNAPSHOT_V1.endpoint, json={"objects": {
        "trajectory": read_grant("snap/trajectory", len(conversation)),
        "workspace": read_grant("snap/workspace", archive.stat().st_size)}})

    assert response.status_code == 200, response.text
    assert json.loads((destination_state / "settings.json").read_text()) == {
        "hooks": {"PostToolUse": [kept], "PostToolUseFailure": []}}


def _turns(monkeypatch, agent) -> list[str]:
    """Run two contexts' turns at once; the order their CLIs started and ended in."""
    events: list[str] = []

    async def _run_claude(**kwargs):
        events.append(f"start {kwargs['session_id']}")
        await asyncio.sleep(0.05)
        events.append(f"end {kwargs['session_id']}")
        return claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    for context in ("a", "b"):
        agent.set_session_ref_for_context(context, context)

    def request(context: str):
        return a2a_server.TaskRequest(task_id=f"t-{context}", context_id=context,
                                      parts=(a2a_server.TextPart(text="go"),), config=a2a_server.ClaudeCodeConfig())

    async def both():
        await asyncio.gather(agent.run(request("a")), agent.run(request("b")))

    asyncio.run(both())
    return events


def test_while_a_changelog_is_captured_turns_take_turns(monkeypatch, tmp_path):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    assert _turns(monkeypatch, agent)[:2] == ["start a", "start b"]

    client.post("/ext/snapshot/changelog", json={"write_namespace": namespace_grant()})

    assert _turns(monkeypatch, agent) == ["start a", "end a", "start b", "end b"]


@pytest.mark.parametrize("make", ["symlink", "file"])
def test_enable_and_apply_refuse_a_changelog_directory_that_isnt_one_this_user_owns(tmp_path, make):
    directory = tmp_path / "changelog"
    if make == "symlink":
        (tmp_path / "elsewhere").mkdir()
        directory.symlink_to(tmp_path / "elsewhere")
    else:
        directory.write_text("")
    log, installed, after, _ = _changelog(tmp_path)

    with pytest.raises(HTTPException) as enabled:
        _enable(log, roots=[str(tmp_path)])
    with pytest.raises(HTTPException) as applied:
        asyncio.run(log.apply(ObjectChangelogApplyRequest(increments=[])))

    for raised in (enabled, applied):
        assert raised.value.status_code == 500 and "must be a directory, not a link" in raised.value.detail
    assert installed == [] and after == []
    assert not (tmp_path / "elsewhere" / "config.json").exists()


def _rig(tmp_path, *more):
    """A capture whose custom roots hold a workspace and a root a tool may remove, and leave out the CLI's state."""
    workspace, removable_root = tmp_path / "workspace", tmp_path / "data"
    for root in (workspace, removable_root, *more):
        root.mkdir()
    (removable_root / "kept.txt").write_text("in the root")
    log, _, _, resumed = _changelog(tmp_path, roots=[str(workspace), str(removable_root), *map(str, more)])
    _enable(log)
    return log, workspace, removable_root, resumed


def _replay(log, store, staging, sequences, **request) -> int:
    increments = []
    for sequence in sequences:
        key = f"ns/{sequence:06d}.tgz"
        store.objects[key] = staging.objects[f"{ROOT_PATH}/{sequence:06d}.tgz"]
        increments.append({"sequence": sequence, "object": read_grant(key, len(store.objects[key]))})
    body = {"increments": increments, **request}
    return asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(body))).count


@needs_gnu_tar
def test_every_capture_archives_its_metadata_with_its_position_and_session(tmp_path, staging):
    log, workspace, _, _ = _rig(tmp_path)

    for position in range(3):
        (workspace / f"{position}.txt").write_text(str(position))
        _capture(log, staging.port, session=f"session-{position}")

    assert json.loads(log.meta_path.read_text()) == {"position": 2, "session_id": "session-2", "removed": []}
    for position in range(3):
        assert str(log.meta_path).lstrip("/") in _members(staging.objects[f"{ROOT_PATH}/{position:06d}.tgz"])


def test_the_config_always_names_the_metadata_and_whether_a_root_already_holds_it(tmp_path):
    for roots, explicit in (([str(tmp_path / "workspace")], True), ([str(tmp_path)], False)):
        log, _, _, _ = _changelog(tmp_path, roots=roots)
        _enable(log)
        config = json.loads((tmp_path / "changelog" / "config.json").read_text())
        assert (config["meta"], config["meta_explicit"]) == (str(log.meta_path), explicit)
        assert log.meta_path.parent.stat().st_mode & 0o777 == 0o700
        assert not tar_changelog._under(str(log.meta_path), [str(tmp_path / "changelog")])


@needs_gnu_tar
@pytest.mark.parametrize("removed_at", [0, 1])
def test_a_root_removed_at_any_position_is_removed_on_replay(tmp_path, staging, store, removed_at):
    log, workspace, removable_root, _ = _rig(tmp_path)
    for position in range(3):
        (workspace / f"{position}.txt").write_text(str(position))
        if position == removed_at:
            shutil.rmtree(removable_root)
        _capture(log, staging.port)

    removable_root.mkdir()
    (removable_root / "kept.txt").write_text("the target's copy")
    log.meta_path.unlink()
    applied = _replay(log, store, staging, [0, 1, 2])

    assert applied == 3 and not removable_root.exists()
    assert sorted(p.name for p in workspace.iterdir()) == ["0.txt", "1.txt", "2.txt"]
    assert not log.meta_path.exists()


@needs_gnu_tar
def test_custom_roots_without_the_clis_state_still_replay_the_last_tool_calls_session(tmp_path, staging, store):
    log, workspace, _, resumed = _rig(tmp_path)
    for position, session in enumerate(("first", "second")):
        (workspace / f"{position}.txt").write_text(str(position))
        _capture(log, staging.port, session=session)

    _replay(log, store, staging, [0, 1], resume_conversation=True, target_context_id="rewound")

    assert resumed == [("rewound", "second")]


@needs_gnu_tar
def test_deleting_the_clis_state_doesnt_stop_the_capture_of_the_other_roots(tmp_path, staging, store):
    state = tmp_path / "state"
    log, workspace, _, _ = _rig(tmp_path, state)
    (state / "settings.json").write_text("{}")

    shutil.rmtree(state)
    (workspace / "a.txt").write_text("a")
    first = _capture(log, staging.port)
    (workspace / "b.txt").write_text("b")
    second = _capture(log, staging.port)

    assert "captured seq=0" in first.stderr and "captured seq=1" in second.stderr
    state.mkdir()
    assert _replay(log, store, staging, [0, 1]) == 2 and not state.exists()


@needs_gnu_tar
def test_a_metadata_write_that_fails_still_captures(tmp_path, staging):
    log, workspace, _, _ = _rig(tmp_path)
    log.meta_path.mkdir()

    (workspace / "a.txt").write_text("a")
    captured = _capture(log, staging.port)

    assert "metadata write failed seq=0; capturing without it" in captured.stderr and "captured seq=0" in captured.stderr
    members = _members(staging.objects[f"{ROOT_PATH}/000000.tgz"])
    assert str(workspace / "a.txt").lstrip("/") in members
    assert not any(member.endswith(tar_changelog.META_NAME) for member in members)


@needs_gnu_tar
def test_a_removal_a_capture_that_wasnt_stored_saw_goes_with_the_next_one(tmp_path, staging, store):
    log, workspace, removable_root, _ = _rig(tmp_path)
    refusing = Staging(status=403)
    shutil.rmtree(removable_root)
    try:
        _capture(log, refusing.port)
    finally:
        refusing.close()
    (workspace / "after.txt").write_text("x")
    _capture(log, staging.port)

    removable_root.mkdir()
    assert _replay(log, store, staging, [1]) == 1 and not removable_root.exists()


@needs_gnu_tar
@pytest.mark.parametrize("metadata", ["missing", "mismatched"])
def test_an_increment_without_its_own_metadata_removes_nothing_and_resumes_the_largest(tmp_path, store, metadata):
    log, _, _, resumed = _changelog(tmp_path)
    target = tmp_path / "restored" / "file.txt"
    target.parent.mkdir()
    target.write_text("restored")
    members = [target]
    elsewhere = tmp_path / "another-agent.changelog-meta" / tar_changelog.META_NAME
    if metadata == "mismatched":
        elsewhere.parent.mkdir()
        elsewhere.write_text(json.dumps({"position": 5, "session_id": "wrong", "removed": [str(target.parent)]}))
        members.append(elsewhere)
    store.objects["ns/000000.tgz"] = _tgz(*members)
    target.unlink()
    elsewhere.unlink(missing_ok=True)
    increments = [{"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))}]

    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(
        {"increments": increments, "resume_conversation": True, "target_context_id": "ctx"})))

    assert answer.count == 1 and target.read_text() == "restored"
    assert resumed == [("ctx", None)] and not elsewhere.exists()


@needs_gnu_tar
def test_a_user_file_named_like_the_metadata_is_restored(tmp_path, store):
    log, _, _, resumed = _changelog(tmp_path)
    user = tmp_path / "app" / tar_changelog.META_NAME
    user.parent.mkdir()
    user.write_text("the user's own")
    meta = tmp_path / "t" / "claude-code-changelog-8000.changelog-meta" / tar_changelog.META_NAME
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"position": 0, "session_id": "the-session", "removed": []}))
    store.objects["ns/000000.tgz"] = _tgz(user, meta)
    user.unlink()
    meta.unlink()
    increments = [{"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))}]

    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(
        {"increments": increments, "resume_conversation": True, "target_context_id": "ctx"})))

    assert answer.count == 1 and user.read_text() == "the user's own" and not meta.exists()
    assert resumed == [("ctx", "the-session")]


@pytest.mark.parametrize(("member", "is_meta"), [
    (f"tmp/claude-code-changelog-8000.changelog-meta/{tar_changelog.META_NAME}", True),
    (f"var/tmp/x/changelog.changelog-meta/{tar_changelog.META_NAME}", True),
    (f"app/{tar_changelog.META_NAME}", False),
    (f"tmp/claude-code-changelog-8000/{tar_changelog.META_NAME}", False),
    (f"tmp/claude-code-changelog-8000.changelog-meta/{tar_changelog.META_NAME}.bak", False),
    (f"tmp/claude-code-changelog-8000.changelog-meta/nested/{tar_changelog.META_NAME}", False),
    (f"tmp/changelog-meta/{tar_changelog.META_NAME}", False),
])
def test_only_the_reserved_metadata_path_is_the_captures_metadata(member, is_meta):
    assert tar_changelog.is_meta_member(member) is is_meta


@pytest.mark.parametrize(("path", "allowed"), [
    ("/tmp/data", True), ("/app", True), ("/", False), ("/etc", False), ("/usr", False), ("relative/path", False),
    ("/tmp/data/../etc", False), ("/tmp/data/", False), (None, False), ("/does/not/exist", False),
])
def test_only_an_absolute_normalized_existing_path_outside_the_system_dirs_is_removed(monkeypatch, path, allowed):
    monkeypatch.setattr(tar_changelog.os.path, "lexists", lambda p: p != "/does/not/exist")

    assert tar_changelog.removable(path) is allowed


@needs_gnu_tar
def test_apply_resumes_the_session_of_the_last_applied_tool_call_not_the_largest(monkeypatch, tmp_path, store):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    sessions = a2a_server.SESSION_JSONL_DIR
    sessions.mkdir(parents=True)
    other = sessions / "other.jsonl"
    other.write_text(json.dumps({"sessionId": "other", "message": "x" * 500}) + "\n")
    mine = sessions / "mine.jsonl"
    mine.write_text(json.dumps({"sessionId": "mine", "message": "the rewound one"}) + "\n")
    # Where another agent kept it: a different temp directory and port than this agent's.
    meta = tmp_path / "elsewhere" / "claude-code-changelog-8123.changelog-meta" / tar_changelog.META_NAME
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"position": 0, "session_id": "mine", "removed": []}))
    store.objects["ns/000000.tgz"] = _tgz(other, mine, meta)
    for path in (other, mine, meta):
        path.unlink()

    response = client.put("/ext/snapshot/changelog", json={
        "increments": [{"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))}],
        "resume_conversation": True, "target_context_id": "rewound"})

    assert response.status_code == 200, response.text
    bound = sessions / f"{agent.session_ref_for_context('rewound')}.jsonl"
    assert json.loads(bound.read_text().splitlines()[0])["message"] == "the rewound one"
    assert not meta.exists() and not agent._changelog.meta_path.exists()


def _fake_tar(tmp_path, rc: int) -> str:
    """A PATH whose tar exits ``rc``: 1 after making the archive (a file changed as tar read it), else before."""
    fake = tmp_path / f"tar-{rc}"
    fake.mkdir()
    real = shutil.which("tar")
    body = (f'"{real}" "$@"\necho "tar: a.txt: file changed as we read it" >&2\nexit 1\n' if rc == 1
            else f'echo "tar: cannot open: Permission denied" >&2\nexit {rc}\n')
    (fake / "tar").write_text("#!/bin/sh\n" + body)
    (fake / "tar").chmod(0o755)
    return f"{fake}:{os.environ['PATH']}"


@needs_gnu_tar
def test_a_tar_that_warns_a_file_changed_is_still_stored(tmp_path, staging):
    log, workspace, _, _ = _rig(tmp_path)

    (workspace / "a.txt").write_text("a")
    captured = _capture(log, staging.port, path=_fake_tar(tmp_path, 1))

    assert "tar warned seq=0:" in captured.stderr and "a.txt: file changed as we read it" in captured.stderr
    assert sorted(staging.objects) == [f"{ROOT_PATH}/000000.tgz"]


@needs_gnu_tar
@pytest.mark.parametrize("rc", [2, 3])
def test_a_tar_that_fails_stores_nothing_keeps_the_state_and_lets_the_next_position_carry_its_writes(
        tmp_path, staging, rc):
    log, workspace, _, _ = _rig(tmp_path)
    snar, present = tmp_path / "changelog" / "state.snar", tmp_path / "changelog" / "present.json"
    before = (snar.read_bytes(), present.read_text())

    (workspace / "a.txt").write_text("a")
    failed = _capture(log, staging.port, path=_fake_tar(tmp_path, rc))
    after = (snar.read_bytes(), present.read_text())
    (workspace / "b.txt").write_text("b")
    next_one = _capture(log, staging.port)

    assert f"tar failed seq=0 rc={rc}: tar: cannot open: Permission denied" in failed.stderr
    assert after == before and not list((tmp_path / "changelog").glob("*.tgz"))
    assert "captured seq=1" in next_one.stderr
    assert {str(workspace / "a.txt").lstrip("/"), str(workspace / "b.txt").lstrip("/")} <= set(
        _members(staging.objects[f"{ROOT_PATH}/000001.tgz"]))


def _gated_turn(monkeypatch):
    """A fake CLI that holds its turn open until released; the order turns started and ended in."""
    events: list[str] = []
    release = asyncio.Event()

    async def _run_claude(**kwargs):
        events.append(f"start {kwargs['session_id']}")
        await release.wait()
        events.append(f"end {kwargs['session_id']}")
        return claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    return events, release


def _request(context: str):
    return a2a_server.TaskRequest(task_id=f"t-{context}", context_id=context, parts=(a2a_server.TextPart(text="go"),),
                                  config=a2a_server.ClaudeCodeConfig())


def test_enable_refuses_while_a_turn_runs(monkeypatch, tmp_path):
    _, agent = _changelog_client(monkeypatch, tmp_path)
    events, release = _gated_turn(monkeypatch)
    agent.set_session_ref_for_context("a", "a")

    async def scenario():
        turn = asyncio.create_task(agent.run(_request("a")))
        while not events:
            await asyncio.sleep(0.01)
        with pytest.raises(HTTPException) as refused:
            await agent.enable_changelog(NamespaceChangelogEnableRequest.model_validate(
                {"write_namespace": namespace_grant()}))
        release.set()
        await turn
        return refused.value

    refused = asyncio.run(scenario())

    assert (refused.status_code, refused.detail) == (409, "cannot enable a changelog while an agent turn is running")
    assert not agent._capturing


def test_a_turn_that_arrives_during_enable_is_captured_and_takes_its_turn(monkeypatch, tmp_path):
    _, agent = _changelog_client(monkeypatch, tmp_path)
    events, release = _gated_turn(monkeypatch)
    for context in ("a", "b"):
        agent.set_session_ref_for_context(context, context)
    enabling = asyncio.Event()
    real_enable = agent._changelog.enable

    async def slow_enable(request):
        enabling.set()
        await asyncio.sleep(0.05)
        events.append("enabled")
        return await real_enable(request)

    monkeypatch.setattr(agent._changelog, "enable", slow_enable)

    async def scenario():
        enable = asyncio.create_task(agent.enable_changelog(NamespaceChangelogEnableRequest.model_validate(
            {"write_namespace": namespace_grant()})))
        await enabling.wait()
        turns = [asyncio.create_task(agent.run(_request(context))) for context in ("a", "b")]
        await enable
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(*turns)

    asyncio.run(scenario())

    assert events == ["enabled", "start a", "end a", "start b", "end b"]
