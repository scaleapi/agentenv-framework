"""The object forms of skill-config, trajectory, snapshot and changelog: agent-env's grants
drive the transfer against a fake object store.

Run from this directory with agentenv-protocol installed and the shared packages on PYTHONPATH:
PYTHONPATH=../shared/agentenv_agent_kit/src:../shared/agent_object_transfer/src:../peer_agents_mcp/src:. pytest -q
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge
from tst.unit.agents.gemini.support import (
    SESSION_ID,
    card,
    chats,
    extension_methods,
    namespace_grant,
    read_grant,
    send,
    settings,
    transcript,
    write_grant,
)

HERE = Path(__file__).resolve().parent
SKILL_CONFIG_PATH = a2a_server.SKILL_CONFIG_V1.endpoint
TRAJECTORY_PATH = a2a_server.TRAJECTORY_V1.endpoint
SNAPSHOT_PATH = a2a_server.SNAPSHOT_V1.endpoint
CHANGELOG_PATH = "/ext/snapshot/changelog"


def _plant_session(home: Path, agent, fmt: str) -> bytes:
    data = transcript(fmt)
    (chats(home) / f"session-2026-09-27T10-00-{SESSION_ID[:8]}.{fmt}").write_bytes(data)
    agent.set_session_ref_for_context("ctx-1", SESSION_ID)
    return data


# ── card and install paths ───────────────────────────────────────────────────────


def test_card_offers_every_object_form_and_no_s3_form(app):
    assert extension_methods(app, a2a_server.SKILL_CONFIG_V1.uri)["add"]["request"] == {
        "required": ["name", "description"],
        "oneOf": [{"required": ["skill_md"]}, {"required": ["skill_bundle"]}],
    }
    assert extension_methods(app, a2a_server.TRAJECTORY_V1.uri)["get"]["request"] == {
        "oneOf": [
            {"required": ["task_id"]},
            {"required": ["task_id", "objects"]},
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
            {"required": ["task_id", "after"], "optional": ["limit"]},
            {"required": ["context_id", "after"], "optional": ["limit"]},
        ],
    }
    snapshot = extension_methods(app, a2a_server.SNAPSHOT_V1.uri)
    assert snapshot["save"]["request"] == {"required": ["context_id", "objects"]}
    assert snapshot["load"]["request"] == {"required": ["objects"], "optional": ["target_context_id"]}
    assert snapshot["enable-changelog"]["request"] == {"required": ["write_namespace"], "optional": ["roots"]}
    assert snapshot["apply-changelog"]["request"] == {
        "required": ["increments"],
        "optional": ["resume_conversation", "target_context_id"],
    }
    requests = json.dumps([
        method.get("request")
        for ext in card(app)["capabilities"]["extensions"]
        for method in ext["params"].get("methods", {}).values()
    ])
    for s3_form in ("skill_s3_url", "trajectory_s3_prefix", "s3_prefix", "presigned_post", "up_to_tool_call_exclusive"):
        assert s3_form not in requests


# ── skills ───────────────────────────────────────────────────────────────────────


def test_a_skill_bundle_is_installed_through_its_read_grants(home, client, store):
    store.objects.update({"skills/demo/SKILL.md": b"# Demo\n", "skills/demo/scripts/run.sh": b"echo hi\n"})
    bundle = {
        "max_total_bytes": 16,
        "files": [
            {"path": "SKILL.md", "object": {**read_grant("skills/demo/SKILL.md", 7), "sha256": None}},
            {"path": "scripts/run.sh", "object": read_grant("skills/demo/scripts/run.sh", 8)},
        ],
    }

    result = client.post(SKILL_CONFIG_PATH, json={"name": "demo", "description": "d", "skill_bundle": bundle})

    assert result.json() == {"name": "demo"}
    assert (a2a_server.SKILLS_DIR / "demo" / "SKILL.md").read_bytes() == b"# Demo\n"
    assert (a2a_server.SKILLS_DIR / "demo" / "scripts" / "run.sh").read_bytes() == b"echo hi\n"
    assert "demo" in client.get(SKILL_CONFIG_PATH).json()["skills"]


def test_a_skill_bundle_that_fails_part_way_leaves_no_skill_behind(home, client, store):
    store.objects["skills/gone/SKILL.md"] = b"# Gone\n"
    bundle = {
        "max_total_bytes": 14,
        "files": [
            {"path": "SKILL.md", "object": read_grant("skills/gone/SKILL.md", 7)},
            {"path": "scripts/run.sh", "object": read_grant("skills/gone/scripts/run.sh", 7)},
        ],
    }

    response = client.post(SKILL_CONFIG_PATH, json={"name": "gone", "description": "d", "skill_bundle": bundle})

    assert response.status_code == 502 and response.json()["error"]["code"] == "transfer_rejected"
    assert store.gets() == ["skills/gone/SKILL.md", "skills/gone/scripts/run.sh"]
    assert not (a2a_server.SKILLS_DIR / "gone").exists()
    assert "gone" not in client.get(SKILL_CONFIG_PATH).json()["skills"]


def test_a_skill_takes_exactly_one_source(home, client, store):
    bundle = {"max_total_bytes": 1, "files": [{"path": "SKILL.md", "object": read_grant("s/SKILL.md", 1)}]}

    response = client.post(
        SKILL_CONFIG_PATH, json={"name": "two", "description": "d", "skill_md": "# Two", "skill_bundle": bundle}
    )

    assert response.status_code == 400 and store.requests == []
    assert not (a2a_server.SKILLS_DIR / "two").exists()


# ── trajectory ───────────────────────────────────────────────────────────────────


def test_the_validator_reads_one_task_inline_then_through_its_grant(monkeypatch, client, store):
    trajectory = [{"type": "message", "role": "assistant", "content": "hi"}, {"type": "result", "stats": {}}]

    async def _run_gemini(**kwargs):
        for event in trajectory:
            kwargs["event_callback"](event)
        return gemini_bridge.GeminiResult(response="hi", tool_call_count=0, session_id="s-1", trajectory=trajectory)

    monkeypatch.setattr(a2a_server, "run_gemini", _run_gemini)
    task_id = send(client, "go")["id"]

    assert client.post(TRAJECTORY_PATH, json={"task_id": task_id}).json() == {"trajectory": trajectory}

    granted = client.post(
        TRAJECTORY_PATH,
        json={"task_id": task_id, "objects": {"trajectory": write_grant("traj/trajectory-t-1.json", "application/json")}},
    ).json()
    body = store.objects["traj/trajectory-t-1.json"]
    assert body == json.dumps(trajectory, separators=(",", ":")).encode()
    assert granted == {"objects": {"trajectory": {"size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}}}
    assert store.requests[-1].headers["content-type"] == "application/json"
    # The object form does not consume the trajectory: a later read of the same task still answers.
    assert client.post(TRAJECTORY_PATH, json={"task_id": task_id}).json() == {"trajectory": trajectory}


# ── snapshot ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", ["json", "jsonl"])
def test_a_snapshot_saves_and_loads_through_grants(home, agent, client, store, fmt):
    data = _plant_session(home, agent, fmt)

    saved = client.post(
        SNAPSHOT_PATH,
        json={"context_id": "ctx-1", "objects": {"trajectory": write_grant("snap/trajectory"), "workspace": write_grant("snap/workspace")}},
    ).json()

    assert store.objects == {"snap/trajectory": data, "snap/workspace": b""}
    assert saved == {
        "context_id": "ctx-1",
        "objects": {
            "trajectory": {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
            "workspace": {"size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        },
    }
    assert [r.headers["content-length"] for r in store.requests] == [str(len(data)), "0"]

    loaded = client.put(
        SNAPSHOT_PATH,
        json={
            "objects": {"trajectory": read_grant("snap/trajectory", len(data)), "workspace": read_grant("snap/workspace", 0)},
            "target_context_id": "restored",
        },
    ).json()

    assert loaded == {"context_id": "restored"}
    assert store.gets() == ["snap/trajectory"]
    new_session = agent.session_ref_for_context("restored")
    (planted,) = chats(home).glob(f"session-*-{new_session[:8]}.*")
    assert planted.suffix == f".{fmt}"
    assert planted.read_bytes() == gemini_bridge.rewrite_session_id(data, new_session)
    assert gemini_bridge.find_session_file(new_session, cwd=a2a_server.GEMINI_CWD) == planted


def test_a_snapshot_load_takes_the_trajectory_object_alone(home, agent, client, store):
    data = transcript("jsonl", "old-session")
    store.objects["snap/trajectory"] = data

    loaded = client.put(SNAPSHOT_PATH, json={"objects": {"trajectory": read_grant("snap/trajectory", len(data))}}).json()

    assert set(loaded) == {"context_id"} and agent.session_ref_for_context(loaded["context_id"]) is not None


def test_a_null_optional_field_is_absent(home, agent, client, store, changelog_dir):
    """An optional field sent as null is the field left out: a save without a workspace
    object writes the transcript alone, and a load without a target mints a context."""
    data = _plant_session(home, agent, "jsonl")

    saved = client.post(
        SNAPSHOT_PATH, json={"context_id": "ctx-1", "objects": {"trajectory": write_grant("snap/t"), "workspace": None}}
    ).json()
    loaded = client.put(
        SNAPSHOT_PATH,
        json={"objects": {"trajectory": read_grant("snap/t", len(data)), "workspace": None}, "target_context_id": None},
    ).json()
    applied = client.put(CHANGELOG_PATH, json={"increments": [], "target_context_id": None}).json()

    assert set(saved["objects"]) == {"trajectory"} and list(store.objects) == ["snap/t"]
    assert agent.session_ref_for_context(loaded["context_id"]) is not None
    assert applied == {"count": 0}


def test_a_snapshot_save_refuses_a_turn_in_flight_and_an_unknown_context(home, agent, client, store):
    _plant_session(home, agent, "jsonl")
    busy = asyncio.Lock()
    asyncio.run(busy.acquire())
    agent._context_locks["ctx-1"] = busy
    body = {"context_id": "ctx-1", "objects": {"trajectory": write_grant("snap/trajectory")}}

    mid_turn = client.post(SNAPSHOT_PATH, json=body)
    unknown = client.post(SNAPSHOT_PATH, json={**body, "context_id": "nope"})

    assert (mid_turn.status_code, mid_turn.json()) == (
        409, {"detail": "session is mid-turn; retry after current turn completes"})
    assert (unknown.status_code, unknown.json()) == (404, {"detail": "No active session for context_id 'nope'"})
    assert store.requests == []


def test_a_snapshot_save_without_a_transcript_answers_with_its_diagnostics(home, agent, client, store):
    agent.set_session_ref_for_context("ctx-1", SESSION_ID)
    (chats(home) / "session-2026-09-27T10-00-ffffffff.json").write_bytes(transcript("json", "other"))

    response = client.post(SNAPSHOT_PATH, json={"context_id": "ctx-1", "objects": {"trajectory": write_grant("snap/t")}})

    assert response.status_code == 500
    head, _, diag = response.json()["detail"].partition(" — diag: ")
    assert head == f"no session JSONL found for session_id={SESSION_ID}"
    assert json.loads(diag) == {
        "session_id": SESSION_ID,
        "cwd": a2a_server.GEMINI_CWD,
        "slug": "workspace",
        "gemini_home": str(home / ".gemini"),
        "gemini_home_exists": True,
        "projects_json_exists": True,
        "chats_dir": str(chats(home)),
        "chats_dir_exists": True,
        "projects_json_content": {"projects": {a2a_server.GEMINI_CWD: "workspace"}},
        "chats_listing": ["session-2026-09-27T10-00-ffffffff.json"],
        "gemini_tmp_listing": ["workspace"],
        "slug_dir_listing": ["chats"],
    }
    assert store.requests == []


def test_a_snapshot_load_refuses_a_bound_context_and_waits_for_a_slug(home, agent, client, store):
    data = transcript("jsonl", "old")
    store.objects["snap/trajectory"] = data
    body = {"objects": {"trajectory": read_grant("snap/trajectory", len(data))}, "target_context_id": "bound"}
    agent.set_session_ref_for_context("bound", "s-0")

    bound = client.put(SNAPSHOT_PATH, json=body)
    (home / ".gemini" / "projects.json").unlink()
    no_slug = client.put(SNAPSHOT_PATH, json={**body, "target_context_id": "fresh"})

    assert (bound.status_code, bound.json()) == (
        409, {"detail": "context_id 'bound' is already bound — retry with a different target_context_id"})
    assert agent.session_ref_for_context("bound") == "s-0" and list(chats(home).iterdir()) == []
    assert no_slug.status_code == 503
    assert no_slug.json()["detail"].startswith("gemini project slug not yet allocated for cwd=")
    assert agent.session_ref_for_context("fresh") is None


def test_a_snapshot_save_and_load_keep_their_400s(home, client, store):
    store.objects["snap/bad"] = b'{"no": "session id"}\n'
    save = client.post(SNAPSHOT_PATH, json={"context_id": "", "objects": {"trajectory": write_grant("snap/trajectory")}})
    load = client.put(
        SNAPSHOT_PATH, json={"objects": {"trajectory": read_grant("snap/trajectory", 1)}, "target_context_id": ""}
    )
    unreadable = client.put(SNAPSHOT_PATH, json={"objects": {"trajectory": read_grant("snap/bad", 21)}})

    assert (save.status_code, save.json()) == (400, {"detail": "context_id is required"})
    assert (load.status_code, load.json()) == (400, {"detail": "target_context_id must be a non-empty string"})
    assert (unreadable.status_code, unreadable.json()) == (
        400, {"detail": "failed to rewrite sessionId: first JSONL line missing 'sessionId' — unexpected gemini chat format"})
    assert list(chats(home).iterdir()) == []


# ── changelog ────────────────────────────────────────────────────────────────────


def _hook_commands(home: Path) -> list[str]:
    return [h["command"] for e in settings(home).get("hooks", {}).get("AfterTool", []) for h in e["hooks"]]


GNU_TAR = shutil.which("tar") is not None and b"GNU tar" in subprocess.run(
    ["tar", "--version"], capture_output=True
).stdout
needs_gnu_tar = pytest.mark.skipif(not GNU_TAR, reason="the changelog is a GNU incremental tar")


def test_the_default_roots_add_the_workspace(home):
    assert a2a_server._changelog_roots() == [
        str(home), a2a_server.GEMINI_CWD, "/app", "/tmp/data", "/usr/local", "/etc",
    ]


def test_changelog_enable_refuses_an_empty_root_list(home, client):
    response = client.post(CHANGELOG_PATH, json={"write_namespace": namespace_grant(), "roots": []})

    assert response.status_code == 400
    assert response.json() == {"detail": "roots must be a non-empty list of strings (omit to use defaults)"}


def test_enable_installs_the_capture_as_an_after_tool_hook_run_by_the_agents_python(home, changelog_dir, client):
    client.post(CHANGELOG_PATH, json={"write_namespace": namespace_grant(), "roots": [str(home)]})

    (entry,) = settings(home)["hooks"]["AfterTool"]
    assert entry == {"matcher": ".*", "hooks": [{
        "type": "command", "command": f"{sys.executable} {changelog_dir / 'capture.py'}", "timeout": 600000}]}


def test_enable_refuses_while_a_turn_runs(home, changelog_dir, agent, client):
    agent._active_contexts.add("ctx-running")

    response = client.post(CHANGELOG_PATH, json={"write_namespace": namespace_grant(), "roots": [str(home)]})

    assert (response.status_code, response.json()) == (
        409, {"detail": "cannot enable a changelog while an agent turn is running"})


@needs_gnu_tar
def test_a_granted_changelog_rewinds_a_fresh_agent_and_resumes_its_conversation(
    home, changelog_dir, agent, client, store, staging
):
    """Each tool call's writes are captured by the AfterTool hook the CLI runs, uploaded through the namespace grant;
    applied in order, they restore the files and the conversation, which the target context resumes."""
    client.post(CHANGELOG_PATH, json={"write_namespace": staging.grant(), "roots": [str(home)]})
    data = transcript("jsonl")
    session_file = chats(home) / f"session-2026-09-27T10-00-{SESSION_ID[:8]}.jsonl"
    session_file.write_bytes(data)
    notes = home / "notes.txt"
    notes.write_text("first")
    staging.capture(changelog_dir, "write_file", SESSION_ID)
    notes.write_text("second")
    staging.capture(changelog_dir, "replace", SESSION_ID)

    # As on a fresh agent: no transcript, file, settings or capture script of the agent that captured.
    session_file.unlink()
    notes.unlink()
    (home / ".gemini" / "settings.json").unlink()
    shutil.rmtree(changelog_dir)
    increments = []
    for seq in (0, 1):
        key = f"{staging.ROOT_PATH}/{seq:06d}.tgz"
        store.objects[key] = staging.objects[key]
        increments.append({"sequence": seq, "object": read_grant(key, len(staging.objects[key]))})

    result = client.put(
        CHANGELOG_PATH, json={"increments": increments, "resume_conversation": True, "target_context_id": "cont"}
    ).json()

    assert result == {"count": 2, "context_id": "cont"}
    assert notes.read_text() == "second" and session_file.read_bytes() == data
    assert _hook_commands(home) == []
    resumed = agent.session_ref_for_context("cont")
    assert resumed not in (None, SESSION_ID)
    assert gemini_bridge.find_session_file(resumed, cwd=a2a_server.GEMINI_CWD) is not None


@needs_gnu_tar
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_file_the_agent_cant_read_is_skipped_and_the_rest_still_captured(home, changelog_dir, client, staging):
    """As /etc/ssl/private is to a non-root agent: tar can't read it, and the increment still uploads the rest."""
    locked = home / "locked"
    locked.mkdir()
    (locked / "key").write_text("secret")
    locked.chmod(0)
    try:
        client.post(CHANGELOG_PATH, json={"write_namespace": staging.grant(), "roots": [str(home)]})
        (home / "marker.txt").write_text("token")
        result = staging.capture(changelog_dir, "write_file", SESSION_ID)
    finally:
        locked.chmod(0o700)

    assert "captured seq=0" in result.stderr, result.stderr
    assert f"{staging.ROOT_PATH}/000000.tgz" in staging.objects


def test_changelog_apply_accepts_the_empty_baseline(home, changelog_dir, client, store):
    assert client.put(CHANGELOG_PATH, json={"increments": [], "resume_conversation": False}).json() == {"count": 0}
    assert store.requests == []


def test_changelog_resume_picks_the_newest_transcript(home, changelog_dir, agent, client, store):
    older = chats(home) / "session-2026-09-27T10-00-aaaaaaaa.jsonl"
    older.write_bytes(transcript("jsonl", "aaaaaaaa-old"))
    os.utime(older, (1, 1))
    other_slug = home / ".gemini" / "tmp" / "elsewhere" / "chats"
    other_slug.mkdir(parents=True)
    (other_slug / "session-2026-09-27T10-05-bbbbbbbb.json").write_bytes(transcript("json", "bbbbbbbb-new"))

    result = client.put(CHANGELOG_PATH, json={"increments": [], "resume_conversation": True}).json()

    session_id = agent.session_ref_for_context(result["context_id"])
    planted = gemini_bridge.find_session_file(session_id, cwd=a2a_server.GEMINI_CWD)
    assert planted.parent == chats(home) and planted.suffix == ".json"
    assert json.loads(planted.read_text())["messages"] == json.loads(transcript("json"))["messages"]


@pytest.mark.parametrize("newest", [None, b"{not json"])
def test_changelog_resume_never_raises(home, changelog_dir, client, store, newest):
    if newest is not None:
        (chats(home) / "session-2026-09-27T10-00-cccccccc.jsonl").write_bytes(newest)

    response = client.put(CHANGELOG_PATH, json={"increments": [], "resume_conversation": True, "target_context_id": "c"})

    assert (response.status_code, response.json()) == (200, {"count": 0})


@pytest.mark.parametrize("sequences", [[1, 1], [2, 1], [-1]])
def test_changelog_apply_refuses_unordered_sequences(home, changelog_dir, client, store, sequences):
    increments = [{"sequence": s, "object": read_grant(f"ns/{abs(s):06d}.tgz", 1)} for s in sequences]

    response = client.put(CHANGELOG_PATH, json={"increments": increments})

    assert response.status_code == 400 and store.requests == []


# ── typed errors ─────────────────────────────────────────────────────────────────


def test_a_malformed_grant_is_a_validation_400(home, client):
    response = client.post(CHANGELOG_PATH, json={"write_namespace": {"root_path": "../x"}})

    assert response.status_code == 400
    errors = json.loads(response.json()["detail"])
    assert {tuple(error["loc"]) for error in errors} >= {("write_namespace", "root_path"), ("write_namespace", "write")}


def test_a_transfer_error_answers_with_the_typed_body(home, changelog_dir, client, store):
    """A malformed grant is refused by validation; an I/O failure behind a well-formed one is
    agent-env's typed error body, never the grant."""
    response = client.put(
        CHANGELOG_PATH, json={"increments": [{"sequence": 0, "object": read_grant("ns/000000.tgz", 1)}]}
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {"code": "transfer_rejected", "message": "The object store rejected the transfer.", "retryable": False}
    }
    assert "sig=r" not in response.text


def test_a_refused_upload_answers_without_the_grant(monkeypatch, client, store):
    async def _run_gemini(**kwargs):
        kwargs["event_callback"]({"type": "message"})
        return gemini_bridge.GeminiResult(response="hi", tool_call_count=0, session_id="s-1", trajectory=[{"type": "message"}])

    monkeypatch.setattr(a2a_server, "run_gemini", _run_gemini)
    task_id = send(client, "go")["id"]
    store.refuse.add("traj/trajectory-t-1.json")

    response = client.post(
        TRAJECTORY_PATH,
        json={"task_id": task_id, "objects": {"trajectory": write_grant("traj/trajectory-t-1.json", "application/json")}},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "transfer_rejected"
    assert "sig=" not in response.text and "trajectory-t-1" not in response.text
    assert client.post(TRAJECTORY_PATH, json={"task_id": task_id}).json() == {"trajectory": [{"type": "message"}]}
