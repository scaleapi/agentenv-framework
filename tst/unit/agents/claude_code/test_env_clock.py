"""The env's clock reaches the model: while an MCP server lists get_time (the gateway does only while the env's clock is
armed), each prompt carries the env's time, and the CLI runs a PostToolUse hook that ends each tool call's result with
a fresh reading. Unarmed, nothing changes. Read here from real MCP servers over streamable HTTP."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from tst.unit.agents.claude_code.agent import AGENT_DIR, a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI

env_clock = importlib.import_module("env_clock")
env_clock_hook = importlib.import_module("env_clock_hook")

NOW = "2026-05-01T09:00:00Z"


def _serve(server: FastMCP) -> tuple[str, uvicorn.Server]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="warning"))
    threading.Thread(target=lambda: asyncio.run(running.serve(sockets=[sock])), daemon=True).start()
    deadline = time.monotonic() + 10
    while not running.started:
        assert time.monotonic() < deadline, "the MCP server didn't start"
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}/mcp", running


@pytest.fixture(scope="module")
def servers():  # every test using it asks for socket_enabled: they read it over this machine's loopback
    """A gateway-like server listing get_time, and one without it."""
    gateway = FastMCP("env", stateless_http=True, json_response=True)

    @gateway.tool()
    def get_time() -> dict:
        return {"current_time": NOW}

    @gateway.tool()
    def list_notes() -> list[str]:
        return []

    other = FastMCP("peers", stateless_http=True, json_response=True)

    @other.tool()
    def peer_send_message(text: str) -> str:
        return text

    started = [_serve(gateway), _serve(other)]
    yield {"env": started[0][0], "peers": started[1][0]}
    for _, running in started:
        running.should_exit = True


def _entries(**urls: str) -> dict:
    return {name: env_clock.ClockServer(url=url) for name, url in urls.items()}


def test_the_server_listing_get_time_is_found_and_read(socket_enabled, servers):
    found = asyncio.run(env_clock.find_env_clock(_entries(peers=servers["peers"], env=servers["env"])))

    assert found is not None
    assert found[0].url == servers["env"] and found[1] == NOW


def test_unarmed_no_server_lists_get_time_and_nothing_is_found(socket_enabled, servers):
    assert asyncio.run(env_clock.find_env_clock(_entries(peers=servers["peers"]))) is None


def test_a_server_that_fails_is_skipped_for_the_next(socket_enabled, servers):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    closed = f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    sock.close()

    found = asyncio.run(env_clock.find_env_clock(_entries(gone=closed, env=servers["env"])))

    assert found is not None and found[1] == NOW


def test_a_slow_clock_leaves_the_prompt_without_its_note(monkeypatch):
    async def _hangs(server, *, find):
        await asyncio.sleep(60)

    monkeypatch.setattr(env_clock, "_read", _hangs)
    monkeypatch.setattr(env_clock, "PROMPT_READ_TIMEOUT_S", 0.05)

    assert asyncio.run(env_clock.find_env_clock(_entries(env="http://unused"))) is None


def test_the_mcp_servers_entries_carry_the_role_the_cli_sends():
    entries = claude_bridge.mcp_server_entries({"env": {"url": "http://gw/mcp", "headers": {"X": "1"}}}, "agent")

    assert entries == {"env": env_clock.ClockServer(url="http://gw/mcp", headers={"X": "1", "AgentEnv-Role": "agent"})}


def _turn(monkeypatch, agent, found) -> dict:
    """One turn with ``find_env_clock`` answering ``found``; what run_claude was called with."""
    called: dict = {}

    async def _find(servers):
        called["servers"] = servers
        return found

    async def _run_claude(**kwargs):
        called.update(kwargs)
        return claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "find_env_clock", _find)
    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    request = a2a_server.TaskRequest(task_id="t", context_id="c", parts=(a2a_server.TextPart(text="book it"),),
                                     config=a2a_server.ClaudeCodeConfig(role="agent"),
                                     mcp_servers={"env": {"url": "http://gw/mcp"}})
    asyncio.run(agent.run(request))
    return called


def test_an_armed_clock_puts_the_envs_time_in_the_prompt_and_the_hook_on_the_cli(monkeypatch, agent):
    server = env_clock.ClockServer(url="http://gw/mcp", headers={"AgentEnv-Role": "agent"})
    called = _turn(monkeypatch, agent, (server, NOW))

    assert called["servers"] == {"env": server}
    assert called["prompt"].startswith("book it\n\n<ENV_CLOCK>")
    assert NOW in called["prompt"] and called["env_clock"] == server


def test_an_unarmed_clock_leaves_the_prompt_and_the_cli_alone(monkeypatch, agent):
    called = _turn(monkeypatch, agent, None)

    assert called["prompt"] == "book it" and called["env_clock"] is None


def _command(env_clock_on: bool) -> list[str]:
    return claude_bridge._command("s", False, model=None, allowed_tools=None, mcp_servers=None, system_prompt=None,
                                  output_format=None, role=None, effort=None, max_turns=None, env_clock=env_clock_on)


def test_the_hook_is_on_the_command_line_only_while_armed():
    assert "--settings" not in _command(False)
    command = _command(True)
    hooks = json.loads(command[command.index("--settings") + 1])["hooks"]

    # After a tool call that failed too: a long command that fails takes as much of the clock.
    assert set(hooks) == {"PostToolUse", "PostToolUseFailure"}
    for entries in hooks.values():
        [hook] = entries[0]["hooks"]
        assert hook["command"].endswith("env_clock_hook.py") and hook["timeout"] == env_clock.HOOK_TIMEOUT_S


def test_a_resume_after_an_api_failure_carries_the_envs_time_now(monkeypatch, tmp_path):
    procs: list[FakeCLI] = []
    answers = [
        {"type": "result", "subtype": "success", "is_error": True, "result": "API Error: The operation timed out.",
         "session_id": "s"},
        {"type": "result", "subtype": "success", "result": "done", "session_id": "s"},
    ]

    async def _exec(*cmd, **kwargs):
        procs.append(FakeCLI((json.dumps(answers.pop(0)) + "\n").encode(), 1 if answers else 0))
        return procs[-1]

    async def _read(server, timeout_s=env_clock.HOOK_READ_TIMEOUT_S):
        return "2026-05-02T10:00:00Z"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    (tmp_path / "s.jsonl").write_text('{"type":"assistant"}\n')
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.setattr(claude_bridge, "read_env_clock", _read)
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False,
                                                  env_clock=env_clock.ClockServer(url="http://gw/mcp")))

    prompts = [proc.stdin.data.decode() for proc in procs]
    assert result.attempts == 2 and prompts[0] == "p"
    assert prompts[1].startswith(claude_bridge._RESUME_PROMPT)
    assert prompts[1].endswith(env_clock.prompt_note("2026-05-02T10:00:00Z"))


def test_the_cli_gets_the_clocks_server_in_its_environment_only_while_armed(monkeypatch):
    envs = []

    async def _exec(*cmd, **kwargs):
        envs.append(kwargs["env"])
        return FakeCLI((json.dumps({"type": "result", "subtype": "success", "result": "ok",
                                    "session_id": "s"}) + "\n").encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    server = env_clock.ClockServer(url="http://gw/mcp", headers={"AgentEnv-Role": "agent"})
    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False))
    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False, env_clock=server))

    assert env_clock.ENV_CLOCK_SERVER_ENV not in envs[0]
    assert env_clock.ClockServer.from_env(envs[1][env_clock.ENV_CLOCK_SERVER_ENV]) == server


def _hook(stdin: dict, environ: dict) -> subprocess.CompletedProcess:
    """The hook script as the CLI runs it: its own process, the hook input on stdin."""
    return subprocess.run([sys.executable, str(AGENT_DIR / "env_clock_hook.py")], input=json.dumps(stdin),
                          capture_output=True, text=True, timeout=30,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), **environ})


@pytest.mark.parametrize("event", ["PostToolUse", "PostToolUseFailure"])
def test_the_hook_ends_a_tool_calls_result_with_the_envs_time(socket_enabled, servers, event):
    ran = _hook({"hook_event_name": event, "tool_name": "Bash"},
                {env_clock.ENV_CLOCK_SERVER_ENV: env_clock.ClockServer(url=servers["env"]).to_env()})

    assert ran.returncode == 0, ran.stderr
    output = json.loads(ran.stdout)["hookSpecificOutput"]
    assert output == {"hookEventName": event, "additionalContext": env_clock.turn_note(NOW)}


@pytest.mark.parametrize("stdin, environ", [
    ({"tool_name": "Bash"}, {}),  # unarmed: no server configured
    ({"tool_name": "mcp__env__get_time"}, {"configured": True}),  # its own result is the reading
    ({"tool_name": "Bash"}, {"unreachable": True}),  # a clock blip prints nothing
])
def test_the_hook_prints_nothing_unarmed_after_get_time_or_on_a_failed_read(socket_enabled, servers, stdin, environ):
    if environ.get("configured"):
        environ = {env_clock.ENV_CLOCK_SERVER_ENV: env_clock.ClockServer(url=servers["env"]).to_env()}
    elif environ.get("unreachable"):
        environ = {env_clock.ENV_CLOCK_SERVER_ENV: env_clock.ClockServer(url="http://127.0.0.1:9/mcp").to_env()}

    ran = _hook(stdin, environ)

    assert ran.returncode == 0 and ran.stdout == "", (ran.stdout, ran.stderr)


def test_the_hook_reads_its_input_as_the_cli_gives_it():
    assert env_clock_hook.main("not json", {}) == ""
