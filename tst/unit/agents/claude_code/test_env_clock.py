"""Claude Code on the task env's clock. The framework ends each prompt with the env's time and gives the turn
``TaskRequest.env_clock`` (``agentenv_protocol.a2a_agent.env_clock``); this agent has the CLI run a PostToolUse and
PostToolUseFailure hook that ends each tool call's result with a fresh reading, and a resume after an API failure read
the clock again. Unarmed, nothing changes. The hook is run here as the CLI runs it, against a real MCP server."""

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
from agentenv_protocol.a2a_agent.env_clock import ENV_CLOCK_SERVER_ENV, ClockServer, EnvClock, prompt_note, turn_note
from mcp.server.fastmcp import FastMCP

from tst.unit.agents.claude_code.agent import AGENT_DIR, a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI

env_clock_hook = importlib.import_module("env_clock_hook")

NOW = "2026-05-01T09:00:00Z"
CLOCK = EnvClock(server=ClockServer(url="http://gw/mcp", headers={"AgentEnv-Role": "agent"}), time=NOW)


@pytest.fixture(scope="module")
def gateway():  # every test using it asks for socket_enabled: it's read over this machine's loopback
    """A gateway-like MCP server listing get_time."""
    server = FastMCP("env", stateless_http=True, json_response=True)

    @server.tool()
    def get_time() -> dict:
        return {"current_time": NOW}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    running = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="warning"))
    threading.Thread(target=lambda: asyncio.run(running.serve(sockets=[sock])), daemon=True).start()
    deadline = time.monotonic() + 10
    while not running.started:
        assert time.monotonic() < deadline, "the MCP server didn't start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    running.should_exit = True


def _turn(monkeypatch, agent, clock: EnvClock | None) -> dict:
    """One turn whose request carries ``clock``; what run_claude was called with."""
    called: dict = {}

    async def _run_claude(**kwargs):
        called.update(kwargs)
        return claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    request = a2a_server.TaskRequest(task_id="t", context_id="c", parts=(a2a_server.TextPart(text="book it"),),
                                     config=a2a_server.ClaudeCodeConfig(), env_clock=clock)
    asyncio.run(agent.run(request))
    return called


def test_an_armed_clock_reaches_the_cli(monkeypatch, agent):
    assert _turn(monkeypatch, agent, CLOCK)["env_clock"] == CLOCK


def test_an_unarmed_clock_leaves_the_cli_alone(monkeypatch, agent):
    assert _turn(monkeypatch, agent, None)["env_clock"] is None


def test_the_agent_leaves_the_prompt_note_to_the_framework():
    # The framework ends the request's parts with the env's time; the agent adds none of its own.
    assert a2a_server.ClaudeCodeAgent.env_clock_prompt_note is True


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
        assert hook["command"].endswith("env_clock_hook.py") and hook["timeout"] == env_clock_hook.HOOK_TIMEOUT_S


def test_the_cli_gets_the_clocks_server_in_its_environment_only_while_armed(monkeypatch):
    envs = []

    async def _exec(*cmd, **kwargs):
        envs.append(kwargs["env"])
        return FakeCLI((json.dumps({"type": "result", "subtype": "success", "result": "ok",
                                    "session_id": "s"}) + "\n").encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False))
    asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False, env_clock=CLOCK))

    assert ENV_CLOCK_SERVER_ENV not in envs[0]
    assert ClockServer.from_env(envs[1][ENV_CLOCK_SERVER_ENV]) == CLOCK.server


class _Clock(EnvClock):
    async def read(self, timeout_s: float = 5.0) -> str:
        return "2026-05-02T10:00:00Z"


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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(claude_bridge, "SESSION_JSONL_DIR", tmp_path)
    (tmp_path / "s.jsonl").write_text('{"type":"assistant"}\n')
    monkeypatch.setattr(claude_bridge, "_RETRY_BACKOFF_S", (0, 0))
    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="s", resume=False,
                                                  env_clock=_Clock(server=CLOCK.server, time=NOW)))

    prompts = [proc.stdin.data.decode() for proc in procs]
    assert result.attempts == 2 and prompts[0] == "p"
    assert prompts[1] == f"{claude_bridge._RESUME_PROMPT}\n\n{prompt_note('2026-05-02T10:00:00Z')}"


def _hook(stdin: dict, environ: dict) -> subprocess.CompletedProcess:
    """The hook script as the CLI runs it: its own process, the hook input on stdin."""
    return subprocess.run([sys.executable, str(AGENT_DIR / "env_clock_hook.py")], input=json.dumps(stdin),
                          capture_output=True, text=True, timeout=30,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), **environ})


@pytest.mark.parametrize("event", ["PostToolUse", "PostToolUseFailure"])
def test_the_hook_ends_a_tool_calls_result_with_the_envs_time(socket_enabled, gateway, event):
    ran = _hook({"hook_event_name": event, "tool_name": "Bash"},
                {ENV_CLOCK_SERVER_ENV: ClockServer(url=gateway).to_env()})

    assert ran.returncode == 0, ran.stderr
    output = json.loads(ran.stdout)["hookSpecificOutput"]
    assert output == {"hookEventName": event, "additionalContext": turn_note(NOW)}


@pytest.mark.parametrize("stdin, configured", [
    ({"tool_name": "Bash"}, None),  # unarmed: no server configured
    ({"tool_name": "mcp__env__get_time"}, "gateway"),  # its own result is the reading
    ({"tool_name": "Bash"}, "http://127.0.0.1:9/mcp"),  # a clock blip prints nothing
])
def test_the_hook_prints_nothing_unarmed_after_get_time_or_on_a_failed_read(socket_enabled, gateway, stdin,
                                                                           configured):
    url = gateway if configured == "gateway" else configured
    ran = _hook(stdin, {ENV_CLOCK_SERVER_ENV: ClockServer(url=url).to_env()} if url else {})

    assert ran.returncode == 0 and ran.stdout == "", (ran.stdout, ran.stderr)


def test_the_hook_reads_its_input_as_the_cli_gives_it():
    assert env_clock_hook.main("not json", {}) == ""
