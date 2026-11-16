"""A turn's ``max_turns`` bounds its tool calls: the CLI is killed once that many have finished, as codex's is."""
import asyncio
import json
import os
import shlex

import pytest

from agent_env.a2a_agent import A2AAgent
from tst.unit.agents.opencode.agent import opencode_bridge
from tst.unit.agents.opencode.fake_cli import FakeCLI

SESSION = "ses_limit"


def _jsonl(*events: dict) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


def tool_use(call_id: str, status: str = "completed") -> dict:
    return {"type": "tool_use", "sessionID": SESSION,
            "part": {"type": "tool", "callID": call_id, "tool": "bash", "state": {"status": status}}}


def step_finish(reason: str) -> dict:
    return {"type": "step_finish", "sessionID": SESSION, "part": {"type": "step-finish", "reason": reason}}


def text(answer: str) -> dict:
    return {"type": "text", "sessionID": SESSION, "part": {"type": "text", "text": answer}}


def test_prompt_agents_max_turns_survives_negotiation_with_the_card(oc):
    negotiated = A2AAgent.negotiate_agent_config(oc.card, {"model": "provider/model", "max_turns": 200})

    assert negotiated.fields["max_turns"] == 200


@pytest.fixture
def cli(monkeypatch):
    """Runs a turn against a FakeCLI printing ``events``; the CLI is the run's second value."""
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})

    async def run(events, *, hang=False, **kwargs):
        proc = FakeCLI(_jsonl(*events), hang=hang)

        async def spawn(*_cmd, **_kwargs):
            return proc

        monkeypatch.setattr(opencode_bridge.asyncio, "create_subprocess_exec", spawn)
        result = await asyncio.wait_for(
            opencode_bridge.run_opencode(prompt="p", context_id="limit-test", timeout_s=60, **kwargs), 10)
        return result, proc

    return run


@pytest.mark.asyncio
async def test_a_turn_limit_stops_the_cli_once_that_many_tool_calls_finish_in_claude_codes_words(cli):
    result, proc = await cli([tool_use("c1"), step_finish("tool-calls"), tool_use("c2")], hang=True, max_turns=2)

    assert proc.killed
    assert (result.error, result.error_code, result.tool_call_count) == ("Reached maximum number of turns (2)", None, 2)
    assert result.session_id == SESSION
    assert len(result.trajectory) == 3


@pytest.mark.asyncio
async def test_a_turn_that_completes_past_its_limit_in_one_read_fails(cli):
    result, _ = await cli([tool_use("c1"), tool_use("c2"), text("done"), step_finish("stop")], max_turns=1)

    assert (result.error, result.tool_call_count) == ("Reached maximum number of turns (1)", 2)


@pytest.mark.asyncio
async def test_a_turn_that_ends_within_its_limit_completes(cli):
    result, proc = await cli([tool_use("c1"), step_finish("tool-calls"), tool_use("c2"), step_finish("tool-calls"),
                              text("done"), step_finish("stop")], max_turns=2)

    assert not proc.killed
    assert (result.error, result.response, result.tool_call_count) == (None, "done", 2)


@pytest.mark.asyncio
async def test_a_tool_call_seen_in_several_states_counts_once_toward_the_limit(cli):
    result, proc = await cli([tool_use("c1", "error"), tool_use("c1"), text("done"), step_finish("stop")],
                             max_turns=1)

    assert not proc.killed
    assert (result.error, result.tool_call_count) == (None, 1)


@pytest.mark.asyncio
async def test_without_a_limit_the_turn_runs_every_tool_call(cli):
    events = [event for n in range(5) for event in (tool_use(f"c{n}"), step_finish("tool-calls"))]
    result, proc = await cli([*events, text("done"), step_finish("stop")])

    assert not proc.killed
    assert (result.error, result.tool_call_count) == (None, 5)


@pytest.mark.asyncio
async def test_a_turn_stopped_at_its_limit_skips_the_output_format_check(cli):
    result, _ = await cli([tool_use("c1")], hang=True, max_turns=1,
                          output_format={"type": "json_schema", "schema": {"type": "object"}})

    assert (result.error, result.structured_output) == ("Reached maximum number of turns (1)", None)


def test_a_turn_limit_kills_a_real_cli_and_returns_without_waiting_for_it(monkeypatch, tmp_path):
    pid_file = tmp_path / "pid"
    lines = _jsonl(tool_use("c1"), step_finish("tool-calls"), tool_use("c2")).decode()
    script = f"echo $$ > {pid_file}; printf %s {shlex.quote(lines)}; exec sleep 600"
    spawn = asyncio.create_subprocess_exec

    async def cli_that_keeps_running(*_cmd, **kwargs):
        return await spawn("sh", "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cli_that_keeps_running)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})

    result = asyncio.run(asyncio.wait_for(
        opencode_bridge.run_opencode(prompt="p", context_id="limit-test", timeout_s=60, max_turns=2), 10))

    assert (result.error, result.tool_call_count) == ("Reached maximum number of turns (2)", 2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
