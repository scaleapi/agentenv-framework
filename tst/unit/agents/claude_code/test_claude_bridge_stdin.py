"""The prompt goes to the CLI on stdin, never argv: a judge prompt (the agent's prompt, its whole response and the
criteria) can pass Linux's per-argument limit (MAX_ARG_STRLEN, 128 KiB), and exec fails with E2BIG before the CLI
starts. The CLI's environment is checked here too.

"""
import asyncio
import json

import pytest

from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.agent import claude_bridge

MAX_ARG_STRLEN = 131072  # Linux single-argument limit (PAGE_SIZE * 32)


_RESULT = {"type": "result", "subtype": "success", "result": "judge verdict", "session_id": "sess-1"}


@pytest.fixture
def spawn_capture(monkeypatch):
    capture: dict = {}

    async def _fake_exec(*cmd, **kwargs):
        capture["cmd"] = list(cmd)
        capture["kwargs"] = kwargs
        capture["proc"] = FakeCLI((json.dumps(_RESULT) + "\n").encode())
        return capture["proc"]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return capture


@pytest.fixture
def spawn_capture_hang(monkeypatch):
    capture: dict = {}

    async def _fake_exec(*cmd, **kwargs):
        capture["cmd"] = list(cmd)
        capture["kwargs"] = kwargs
        # Never finishes within the test's timeout — forces the
        # asyncio.TimeoutError path in run_claude (proc.kill/proc.wait).
        capture["proc"] = FakeCLI(hang=True)
        return capture["proc"]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return capture


def test_prompt_goes_to_stdin_not_argv(spawn_capture):
    prompt = "evaluate this trajectory against the criteria"
    result = asyncio.run(claude_bridge.run_claude(prompt=prompt, session_id="sess-1", resume=False))
    assert not any(prompt in arg for arg in spawn_capture["cmd"])
    assert spawn_capture["kwargs"].get("stdin") is asyncio.subprocess.PIPE
    assert spawn_capture["proc"].stdin.data == prompt.encode()
    assert result.response == "judge verdict"
    assert result.error is None


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5", "gpt-5.6-sol", "gemini/gemini-3.1-pro-preview"],
)
def test_disables_dynamic_attribution_block_for_every_model(spawn_capture, monkeypatch, model):
    monkeypatch.setenv("CLAUDE_CODE_ATTRIBUTION_HEADER", "1")

    asyncio.run(
        claude_bridge.run_claude(
            prompt="p",
            session_id="sess-1",
            resume=False,
            model=model,
        )
    )

    assert spawn_capture["kwargs"]["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert spawn_capture["cmd"][spawn_capture["cmd"].index("--model") + 1] == model


def test_cache_fix_preserves_litellm_cost_attribution(spawn_capture):
    asyncio.run(
        claude_bridge.run_claude(
            prompt="p",
            session_id="sess-1",
            resume=False,
            project_id="project-1",
            task_id="task-1",
        )
    )

    env = spawn_capture["kwargs"]["env"]
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == (
        "x-litellm-customer-id: project-1\n"
        "x-litellm-tags: projectId:project-1,taskId:task-1"
    )


def test_oversized_judge_prompt_never_lands_on_argv(spawn_capture):
    # ~360 KB, well past MAX_ARG_STRLEN.
    prompt = "criteria " * 45_000
    asyncio.run(claude_bridge.run_claude(prompt=prompt, session_id="sess-1", resume=False))
    assert all(len(arg) < MAX_ARG_STRLEN for arg in spawn_capture["cmd"])
    assert spawn_capture["proc"].stdin.data == prompt.encode()


def test_timeout_kills_process_and_returns_error(spawn_capture_hang):
    # A hung CLI must hit the asyncio.TimeoutError path, which kills the
    # process and awaits proc.wait() — exercises those _FakeProc stubs.
    result = asyncio.run(
        claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, timeout_s=0)
    )
    assert result.error is not None
    assert "timed out" in result.error
    assert result.response == ""
    assert spawn_capture_hang["proc"].killed is True


def test_a_timed_out_turn_keeps_the_tool_calls_it_made(monkeypatch):
    tool_use = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}}

    async def _exec(*cmd, **kwargs):
        return FakeCLI((json.dumps(tool_use) + "\n").encode(), hang=True)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, timeout_s=1))

    assert "timed out" in result.error and result.tool_call_count == 1
