"""The CLIs a crashed server left running. In the image, supervisord restarts the A2A server in place, and the CLIs it
was running, each leading a session of its own, outlive it: the restarted server kills them, process group and all,
before it takes a turn, so none goes on writing beside its turns. Elsewhere (install/v1) a crash ends the agent and
nothing is looked for."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pytest

from tst.unit.agents.codex.agent import a2a_server, agent_egress, codex_bridge

LAUNCHER = [b"node", b"/usr/local/bin/codex", b"exec", b"--cd", b"/app", b"-", b"--json"]
BINARY = [b"/opt/agent/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex", b"exec",
          b"--cd", b"/app", b"-", b"--json"]


def _process(proc: Path, pid: int, argv: list[bytes], group: int, session: int) -> None:
    folder = proc / str(pid)
    folder.mkdir()
    (folder / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
    (folder / "stat").write_text(f"{pid} ({argv[0][-15:].decode()}) S 1 {group} {session} 0 -1 4194560 0 0\n")


@pytest.fixture
def proc(tmp_path, monkeypatch) -> Path:
    """A /proc of an image whose server crashed mid-turn: its CLI's launcher, leading the session the server started
    it in, and the binary under it; an MCP server and a `codex debug models` leading sessions of their own; and this
    process, as if it were a CLI too."""
    root = tmp_path / "proc"
    root.mkdir()
    _process(root, 101, LAUNCHER, 101, 101)
    _process(root, 102, BINARY, 101, 101)
    _process(root, 103, [b"/usr/local/bin/node", b"/opt/agent/node_modules/mcp-remote/dist/proxy.js"], 103, 103)
    _process(root, 104, [b"node", b"/usr/local/bin/codex", b"debug", b"models"], 104, 104)
    _process(root, os.getpid(), LAUNCHER, os.getpid(), os.getpid())
    (root / "self").mkdir()
    monkeypatch.setenv(agent_egress.EGRESS_PROXY_ENV, "http://127.0.0.1:3128")
    return root


@pytest.fixture
def killed(monkeypatch) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(codex_bridge.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    return calls


def test_the_image_kills_each_orphaned_cli_by_its_process_group(proc, killed):
    assert codex_bridge.kill_orphaned_clis(proc) == 1
    assert killed == [(101, signal.SIGKILL)]


def test_another_users_cli_is_left(proc, killed, monkeypatch):
    other = os.getuid() + 1
    monkeypatch.setattr(codex_bridge.os, "getuid", lambda: other)

    assert codex_bridge.kill_orphaned_clis(proc) == 0 and killed == []


def test_a_server_the_image_doesnt_start_looks_for_none(proc, killed, monkeypatch):
    monkeypatch.delenv(agent_egress.EGRESS_PROXY_ENV)

    assert codex_bridge.kill_orphaned_clis(proc) == 0 and killed == []


def test_the_server_kills_them_as_it_starts(codex, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(a2a_server, "kill_orphaned_clis", lambda: calls.append("killed") or 1)

    async def start():
        async with codex.app.router.lifespan_context(codex.app):
            assert calls == ["killed"]

    asyncio.run(start())
