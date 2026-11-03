"""The prompt goes to `opencode run` on stdin, never argv: Linux caps one argument at 128 KiB (MAX_ARG_STRLEN), and a
longer prompt fails exec with E2BIG before the CLI starts. opencode-ai 1.17.18 reads stdin when it isn't a TTY and,
with no positional message, sends it to the model unchanged."""

import asyncio
import json
import shlex

import pytest

from tst.unit.agents.opencode.agent import opencode_bridge
from tst.unit.agents.opencode.fake_cli import FakeCLI, FakeStdin

MAX_ARG_STRLEN = 131072  # Linux single-argument limit (PAGE_SIZE * 32)

_STOP = {
    "type": "step_finish",
    "sessionID": "ses_1",
    "part": {"type": "step-finish", "reason": "stop"},
}
_TEXT = {
    "type": "text",
    "sessionID": "ses_1",
    "part": {"type": "text", "text": '{"ok": true}'},
}


def _lines(*events: dict) -> bytes:
    return b"".join(json.dumps(event).encode() + b"\n" for event in events)


@pytest.fixture
def spawned(monkeypatch):
    """Every spawn's argv, keyword arguments and scripted CLI, each CLI finishing its turn."""
    calls: list[dict] = []

    async def _exec(*cmd, **kwargs):
        proc = FakeCLI(_lines(_TEXT, _STOP))
        calls.append({"cmd": list(cmd), "kwargs": kwargs, "proc": proc})
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})
    return calls


def test_a_first_turn_sends_its_prompt_on_stdin_and_not_in_argv(spawned):
    result = asyncio.run(
        opencode_bridge.run_opencode(
            prompt="summarise the filing", context_id="ctx", model="m", workspace="/w"
        )
    )

    (call,) = spawned
    assert call["cmd"] == [
        "opencode",
        "run",
        "--format",
        "json",
        "--model",
        "litellm/m",
        "--dir",
        "/w",
    ]
    assert call["kwargs"]["stdin"] is asyncio.subprocess.PIPE
    assert (
        call["proc"].stdin.data == b"summarise the filing" and call["proc"].stdin.closed
    )
    assert result.error is None and result.session_id == "ses_1"


def test_a_resumed_turn_names_its_session_and_sends_its_prompt_on_stdin(spawned):
    asyncio.run(
        opencode_bridge.run_opencode(
            prompt="and the next one",
            context_id="ctx",
            session_id="ses_0",
            model="m",
            attachment_paths=["/w/a.png"],
            workspace="/w",
        )
    )

    (call,) = spawned
    assert call["cmd"] == [
        "opencode",
        "run",
        "--format",
        "json",
        "--model",
        "litellm/m",
        "--dir",
        "/w",
        "-s",
        "ses_0",
        "--file",
        "/w/a.png",
    ]
    assert call["proc"].stdin.data == b"and the next one"


def test_a_one_mib_prompt_and_its_output_format_reach_stdin_byte_for_byte(spawned):
    prompt = "  médical record line\n" * (1024 * 1024 // 22) + "\n\n"
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    result = asyncio.run(
        opencode_bridge.run_opencode(
            prompt=prompt,
            context_id="ctx",
            session_id="ses_0",
            output_format={"type": "json_schema", "schema": schema},
        )
    )

    (call,) = spawned
    assert max(len(arg.encode()) for arg in call["cmd"]) < 4096
    suffix = (
        "\n\nReturn only a JSON object matching this schema, with no Markdown or surrounding text: "
        + json.dumps(schema, separators=(",", ":"))
    )
    assert call["proc"].stdin.data == (prompt + suffix).encode()
    assert result.structured_output == {"ok": True}


def test_a_real_cli_reads_a_one_mib_prompt_whole_from_stdin(monkeypatch, tmp_path):
    # Exec'd for real: on main the prompt is one argument, over Linux's per-argument limit and macOS's 1 MiB total.
    received = tmp_path / "stdin"
    stop = shlex.quote(_lines(_STOP).decode())
    fake_opencode = tmp_path / "opencode"
    fake_opencode.write_text(
        f"#!/bin/sh\ncat > {shlex.quote(str(received))}\nprintf '%s' {stop}\n"
    )
    fake_opencode.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})
    prompt = "x" * (1024 * 1024 + 7)

    result = asyncio.run(
        opencode_bridge.run_opencode(
            prompt=prompt, context_id="ctx", workspace=str(tmp_path)
        )
    )

    assert result.error is None, result.error
    assert received.read_bytes() == prompt.encode()


def test_a_cli_that_exits_before_reading_its_prompt_still_ends_the_turn(
    monkeypatch, tmp_path
):
    # A CLI that rejects its config exits without reading stdin; the bridge's write then hits a closed pipe.
    spawn = asyncio.create_subprocess_exec

    async def cli_that_never_reads(*_cmd, **kwargs):
        return await spawn("sh", "-c", "echo 'Error: bad config' >&2; exit 1", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cli_that_never_reads)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})

    result = asyncio.run(
        opencode_bridge.run_opencode(
            prompt="x" * (4 * 1024 * 1024), context_id="ctx", timeout_s=30
        )
    )

    assert result.error == "Error: bad config"


def test_a_real_cli_that_prints_before_reading_its_prompt_does_not_deadlock(
    monkeypatch, tmp_path
):
    # The CLI fills its stdout and stderr pipes before it reads stdin. Were the prompt written to completion before
    # stdout and stderr were read, both sides would block on a full pipe until the turn timed out.
    stop = shlex.quote(_lines(_STOP).decode())
    fake_opencode = tmp_path / "opencode"
    fake_opencode.write_text(
        "#!/bin/sh\n"
        "head -c 300000 /dev/zero | tr '\\0' a; echo\n"
        "head -c 300000 /dev/zero | tr '\\0' b >&2\n"
        "cat > /dev/null\n"
        f"printf '%s' {stop}\n"
    )
    fake_opencode.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})

    result = asyncio.run(
        opencode_bridge.run_opencode(
            prompt="x" * (1024 * 1024),
            context_id="ctx",
            workspace=str(tmp_path),
            timeout_s=20,
        )
    )

    assert result.error is None, result.error


def test_a_failed_prompt_write_fails_the_turn(spawned, monkeypatch):
    # Only a CLI that has closed stdin (broken pipe, connection reset) is tolerated; any other write error means the
    # CLI may have got part of the prompt, so the turn must not report success.
    async def drain_fails(_self):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(FakeStdin, "drain", drain_fails)

    with pytest.raises(OSError, match="Input/output error"):
        asyncio.run(opencode_bridge.run_opencode(prompt="summarise", context_id="ctx"))

    (call,) = spawned
    assert call["proc"].stdin.closed
