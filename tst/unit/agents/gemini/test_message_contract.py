"""What a turn sends the CLI and what agent-env reads back: the terminal message's parts for a
success and for each failure, and the session threading across turns."""

from __future__ import annotations

import asyncio
import base64
import json
import os

import pytest
from agentenv_protocol.a2a_agent import AGENT_CONFIG_V1

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge, gemini_kit
from tst.unit.agents.gemini.fake_cli import FakeCLI
from tst.unit.agents.gemini.support import SESSION_ID, chats, send, terminal, transcript

AGENT_CONFIG_PATH = AGENT_CONFIG_V1.endpoint
TRAJECTORY_PATH = a2a_server.TRAJECTORY_V1.endpoint


class Calls(list):
    """Every run_gemini call's arguments; each is answered with ``result_for(kwargs)``."""

    def __init__(self) -> None:
        super().__init__()
        self.result_for = lambda kwargs: gemini_bridge.GeminiResult(
            response="done", tool_call_count=2, session_id=kwargs["session_id"] or "s-new",
            trajectory=[{"type": "result"}],
        )


@pytest.fixture
def calls(monkeypatch) -> Calls:
    recorded = Calls()

    async def _run_gemini(**kwargs):
        # The bridge reports each event to the trajectory log as the CLI prints it.
        event_callback = kwargs.pop("event_callback")
        kwargs.pop("env")
        recorded.append(kwargs)
        result = recorded.result_for(kwargs)
        for event in result.trajectory:
            event_callback(event)
        return result

    monkeypatch.setattr(a2a_server, "run_gemini", _run_gemini)
    return recorded


def test_a_successful_turn(calls, client):
    state, text, data = terminal(send(client, "hello"))

    assert (state, text) == ("completed", "done")
    assert data == [{"tool_call_count": 2}, {"usage": {"tool_call_count": 2}}]


def test_a_cli_error_is_a_failed_task_with_the_capped_message(calls, client):
    error = "gemini CLI exited with code 1: " + "x" * 2000
    calls.result_for = lambda kwargs: gemini_bridge.GeminiResult(response="", tool_call_count=4, session_id="s-1", error=error)

    state, text, data = terminal(send(client, "hello"))

    assert (state, text) == ("failed", error)
    assert data == [
        {"tool_call_count": 4},
        {
            "usage": {"tool_call_count": 4},
            "error_type": "agent_error",
            "error_code": "cli_error",
            "error_message": error[:1000],
        },
    ]


def test_a_cli_error_keeps_the_response_text_when_there_is_one(calls, client):
    calls.result_for = lambda kwargs: gemini_bridge.GeminiResult(response="partial answer", tool_call_count=0, session_id="s-1", error="boom")

    state, text, _ = terminal(send(client, "hello"))

    assert (state, text) == ("failed", "partial answer")


def test_a_bridge_exception_is_an_infra_error(monkeypatch, client):
    async def _raise(**_kwargs):
        raise OSError(7, "Argument list too long")

    monkeypatch.setattr(a2a_server, "run_gemini", _raise)

    task = send(client, "hello")
    state, text, data = terminal(task)

    assert (state, text) == ("failed", "[Errno 7] Argument list too long")
    assert data == [{"error_type": "infra_error", "error_code": "infra_error", "error_message": text}]
    assert client.post(TRAJECTORY_PATH, json={"task_id": task["id"]}).status_code == 404


def test_an_unreadable_file_part_is_a_filepart_error(calls, client):
    parts = [{"kind": "file", "file": {"name": "a.png", "mimeType": "image/png", "bytes": "not base64!"}}]

    state, text, data = terminal(send(client, parts))

    assert state == "failed" and text.startswith("malformed base64 in FilePart 'a.png'")
    assert data == [{"error_type": "agent_error", "error_code": "filepart_error", "error_message": text}]
    assert calls == []


def test_file_parts_are_materialized_under_the_task_and_attached(calls, home, client):
    image = base64.b64encode(b"png-bytes").decode()
    parts = [
        {"kind": "text", "text": "describe"},
        {"kind": "file", "file": {"name": " ../../etc/a b.png ", "mimeType": "image/png", "bytes": image}},
        {"kind": "file", "file": {"name": "a b.png", "mimeType": "image/png", "bytes": image}},
        {"kind": "file", "file": {"mimeType": "audio/wav", "bytes": base64.b64encode(b"wav").decode()}},
        {"kind": "data", "data": {"ignored": True}},
        {"kind": "text", "text": "carefully"},
    ]

    task = send(client, parts)

    (call,) = calls
    inputs = a2a_server.INPUTS_DIR / task["id"]
    first, second, nameless = call["attachment_paths"]
    assert call["prompt"] == "describe\ncarefully"
    assert first == str(inputs / "a_b.png")
    assert os.path.dirname(second) == str(inputs) and os.path.basename(second).endswith("-a_b.png")
    assert os.path.dirname(nameless) == str(inputs) and nameless.endswith(".bin")
    assert [open(path, "rb").read() for path in call["attachment_paths"]] == [b"png-bytes", b"png-bytes", b"wav"]


def test_a_context_resumes_its_session_and_a_new_context_starts_one(calls, client):
    calls.result_for = lambda kwargs: gemini_bridge.GeminiResult(
        response="ok", tool_call_count=0, session_id=kwargs["session_id"] or f"s-{len(calls)}",
    )

    send(client, "one", context_id="ctx-a")
    send(client, "two", context_id="ctx-a")
    send(client, "three", context_id="ctx-b")

    assert [call["session_id"] for call in calls] == [None, "s-1", None]


def test_a_session_a_turn_started_stays_claimed_once_its_context_moves_on(calls, agent, client):
    calls.result_for = lambda kwargs: gemini_bridge.GeminiResult(
        response="ok", tool_call_count=0, session_id="s-other", trajectory=[{"type": "init", "session_id": "s-other"}],
    )
    send(client, "one", context_id="ctx-other")
    agent.set_session_ref_for_context("ctx-other", "s-loaded")

    send(client, "two")

    assert calls[1]["other_sessions"]() == {"s-other", "s-loaded"}


def test_a_turn_that_reports_no_session_binds_none(calls, agent, client):
    calls.result_for = lambda kwargs: gemini_bridge.GeminiResult(response="ok", tool_call_count=0, session_id=None)

    send(client, "one")
    send(client, "two")

    assert [call["session_id"] for call in calls] == [None, None]
    assert agent.session_ref_for_context("ctx-1") is None


def test_agent_config_reaches_the_cli(calls, client):
    config = {
        "model": "vertex_ai/global/gemini-3.1-pro-preview",
        "allowed_tools": ["run_shell_command"],
        "system_prompt": "be brief",
        "timeout_seconds": 42,
        "project_id": "p-1",
        "task_id": "t-1",
        "role": "solver",
    }
    assert client.post(AGENT_CONFIG_PATH, json=config).json() == {"status": "updated"}

    send(client, "go")

    (call,) = calls
    assert callable(call.pop("other_sessions"))
    assert call == {
        "prompt": "go",
        "session_id": None,
        "model": "gemini-3.1-pro-preview",
        "attachment_paths": None,
        "timeout_s": 42,
        "cwd": a2a_server.GEMINI_CWD,
    }


def test_the_default_timeout_is_ten_minutes(calls, client):
    send(client, "go")

    assert (calls[0]["timeout_s"], calls[0]["model"]) == (600, None)


def test_a_turn_keeps_its_trajectory_failed_or_not(calls, client):
    trajectory = [{"type": "init", "session_id": "s-1"}, {"type": "result", "status": "error"}]
    calls.result_for = lambda kwargs: gemini_bridge.GeminiResult(
        response="", tool_call_count=0, session_id="s-1", trajectory=trajectory, error="boom",
    )

    task = send(client, "go")

    assert client.post(TRAJECTORY_PATH, json={"task_id": task["id"]}).json() == {"trajectory": trajectory}


# ── the CLI invocation itself ────────────────────────────────────────────────────


@pytest.fixture
def spawned(monkeypatch, home):
    """The real bridge against a scripted `gemini`: the argv of every spawn, answered with
    ``spawned.stdout``, and the cwd it ran in."""
    os.makedirs(a2a_server.GEMINI_CWD, exist_ok=True)

    class _Argvs(list):
        stdout = b""
        cwds: list = []
        on_spawn = staticmethod(lambda: None)

    argvs = _Argvs()

    async def _exec(*cmd, **kwargs):
        argvs.append(list(cmd))
        argvs.cwds.append(kwargs.get("cwd"))
        argvs.on_spawn()
        return FakeCLI(argvs.stdout)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    return argvs


def _stream(*events) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


def test_the_prompt_and_attachments_go_on_argv(spawned, client):
    spawned.stdout = _stream(
        {"type": "init", "session_id": "s-init"},
        {"type": "message", "role": "assistant", "content": "seen"},
        {"type": "result", "status": "success", "stats": {"tool_calls": 0}},
    )
    client.post(AGENT_CONFIG_PATH, json={"model": "gemini-2.5-flash"})
    parts = [
        {"kind": "text", "text": "describe"},
        {"kind": "file", "file": {"name": "a.png", "mimeType": "image/png", "bytes": base64.b64encode(b"p").decode()}},
    ]

    task = send(client, parts)
    send(client, "again")

    image = a2a_server.INPUTS_DIR / task["id"] / "a.png"
    assert spawned[0] == [
        "gemini", "-p", f"describe\n\n@{image}", "--output-format", "stream-json", "--approval-mode", "yolo",
        "--model", "gemini-2.5-flash",
    ]
    assert spawned[1][:3] == ["gemini", "-p", "again"] and spawned[1][7:9] == ["-r", "s-init"]
    assert spawned.cwds == [a2a_server.GEMINI_CWD] * 2
    assert terminal(task)[:2] == ("completed", "seen")


def test_without_an_init_event_the_session_is_the_one_the_run_wrote_and_no_other_contexts(
        spawned, home, agent, client):
    """A CLI that died before its init event still wrote its transcript: the run takes the one written since it
    started, never an older one or one another context is bound to by the time the run ends."""
    older = chats(home) / "session-2026-09-27T10-00-aaaaaaaa.json"
    older.write_bytes(transcript("json", "aaaaaaaa-old"))
    os.utime(older, (1, 1))

    def another_context_starts_a_session_meanwhile():
        (chats(home) / f"session-2026-09-27T10-01-{SESSION_ID[:8]}.jsonl").write_bytes(transcript("jsonl"))
        (chats(home) / "session-2026-09-27T10-02-bbbbbbbb.jsonl").write_bytes(transcript("jsonl", "bbbbbbbb-other"))
        agent.set_session_ref_for_context("ctx-other", "bbbbbbbb-other")

    spawned.on_spawn = another_context_starts_a_session_meanwhile
    spawned.stdout = _stream({"type": "result", "status": "success", "stats": {"tool_calls": 0}})

    send(client, "hello")

    assert agent.session_ref_for_context("ctx-1") == SESSION_ID


def test_without_an_init_event_no_session_is_taken_when_the_run_could_have_written_either_of_two(
        spawned, home, agent, client):
    def two_unclaimed_transcripts():
        (chats(home) / f"session-2026-09-27T10-01-{SESSION_ID[:8]}.jsonl").write_bytes(transcript("jsonl"))
        (chats(home) / "session-2026-09-27T10-02-bbbbbbbb.jsonl").write_bytes(transcript("jsonl", "bbbbbbbb-other"))

    spawned.on_spawn = two_unclaimed_transcripts
    spawned.stdout = _stream({"type": "result", "status": "success", "stats": {"tool_calls": 0}})

    send(client, "hello")

    assert agent.session_ref_for_context("ctx-1") is None


def test_an_init_event_behind_a_notice_the_cli_printed_binds_its_session(spawned, agent, client):
    """The CLI prints some notices, such as an MCP server that failed to connect, with no newline before its
    events."""
    spawned.stdout = b"MCP issues detected. Run /mcp list for status." + _stream(
        {"type": "init", "session_id": "s-init"},
        {"type": "result", "status": "success", "stats": {"tool_calls": 0}},
    )

    send(client, "hello")

    assert agent.session_ref_for_context("ctx-1") == "s-init"


def test_session_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.GeminiAgent, gemini_kit.DurableSessionRefs)
