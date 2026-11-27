"""A turn's ``max_turns`` bounds its steps, OpenCode's model requests: once the last step it allows asks for another,
the CLI is killed and the turn fails in Claude Code's words."""
import asyncio
import itertools
import json
import os
import shlex

import pytest

from agent_env.a2a_agent import A2AAgent
from tst.unit.agents.opencode.agent import opencode_bridge
from tst.unit.agents.opencode.fake_cli import FakeCLI

SESSION = "ses_limit"
LIMIT_ERROR = "Reached maximum number of turns ({})"
_part_ids = itertools.count()


def _jsonl(*events: dict) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


def _event(etype: str, ptype: str, **part) -> dict:
    return {"type": etype, "sessionID": SESSION,
            "part": {"id": f"prt_{next(_part_ids)}", "sessionID": SESSION, "type": ptype, **part}}


def step(*tool_calls: str, answer: str | None = None, reason: str | None = None) -> list[dict]:
    """One step's events as `opencode run` prints them: its tools' results, then its end, which asks for another step
    when it made tool calls."""
    return [
        _event("step_start", "step-start"),
        *(_event("tool_use", "tool", callID=call, tool="bash", state={"status": "completed"}) for call in tool_calls),
        *([_event("text", "text", text=answer)] if answer else []),
        _event("step_finish", "step-finish", reason=reason or ("tool-calls" if tool_calls else "stop")),
    ]


class _Reads:
    """A CLI's stdout handing out ``chunks`` one read at a time, then ending, or staying open as a CLI still running."""

    def __init__(self, chunks: list[bytes], *, hang: bool) -> None:
        self.chunks = list(chunks)
        self.hang = hang

    async def read(self, _n: int = -1) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        if self.hang:
            await asyncio.Event().wait()
        return b""


# Ways the CLI's output can arrive. The final step is its last three events: its start, its answer and its end.
SPLITS = {
    "in-one-read": lambda events: [_jsonl(*events)],
    "a-read-per-event": lambda events: [_jsonl(event) for event in events],
    "final-step-in-its-own-read": lambda events: [_jsonl(*events[:-3]), _jsonl(*events[-3:])],
    "lines-split-across-reads": lambda events: [
        data[start:start + 7] for data in [_jsonl(*events)] for start in range(0, len(data), 7)],
}


@pytest.fixture
def cli(monkeypatch):
    """Runs a turn against a FakeCLI whose stdout hands out ``chunks``; the CLI is the run's second value."""
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})

    async def run(chunks, *, hang=False, **kwargs):
        proc = FakeCLI(hang=hang)
        proc.stdout = _Reads(chunks, hang=hang)

        async def spawn(*_cmd, **_kwargs):
            return proc

        monkeypatch.setattr(opencode_bridge.asyncio, "create_subprocess_exec", spawn)
        result = await asyncio.wait_for(
            opencode_bridge.run_opencode(prompt="p", context_id="limit-test", timeout_s=60,
                                         **{"web_tools": True, **kwargs}), 10)
        return result, proc

    return run


def test_prompt_agents_max_turns_survives_negotiation_with_the_card(oc):
    negotiated = A2AAgent.negotiate_agent_config(oc.card, {"model": "provider/model", "max_turns": 200})

    assert negotiated.fields["max_turns"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["tool-calls", "unknown"])
async def test_the_cli_is_stopped_once_its_last_allowed_step_asks_for_another_in_claude_codes_words(cli, reason):
    events = [*step("c1"), *step("c2", reason=reason)]

    result, proc = await cli([_jsonl(*events)], hang=True, max_turns=2)

    assert proc.killed
    assert (result.error, result.error_code) == (LIMIT_ERROR.format(2), None)
    assert (result.tool_call_count, result.session_id, result.trajectory) == (2, SESSION, events)


@pytest.mark.asyncio
@pytest.mark.parametrize("split", SPLITS.values(), ids=SPLITS.keys())
async def test_a_turn_that_answers_on_its_last_allowed_step_completes_however_its_output_is_read(cli, split):
    result, proc = await cli(split([*step("c1"), *step("c2"), *step(answer="done")]), max_turns=3)

    assert not proc.killed
    assert (result.error, result.response, result.tool_call_count) == (None, "done", 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("split", SPLITS.values(), ids=SPLITS.keys())
async def test_a_turn_that_needs_a_step_past_its_limit_fails_however_its_output_is_read(cli, split):
    result, _ = await cli(split([*step("c1"), *step("c2"), *step(answer="done")]), max_turns=2)

    assert result.error == LIMIT_ERROR.format(2)


@pytest.mark.asyncio
async def test_a_steps_parallel_tool_calls_count_as_one_turn(cli):
    result, proc = await cli([_jsonl(*step("c1", "c2", "c3"), *step(answer="done"))], max_turns=2)

    assert not proc.killed
    assert (result.error, result.tool_call_count) == (None, 3)


@pytest.mark.asyncio
async def test_a_step_reported_twice_counts_once(cli):
    first = step("c1")
    result, _ = await cli([_jsonl(*first, first[-1], *step(answer="done"))], max_turns=2)

    assert result.error is None


@pytest.mark.asyncio
async def test_a_last_allowed_step_that_ends_the_turn_another_way_is_not_the_limit(cli):
    result, _ = await cli([_jsonl(*step(answer="cut off", reason="length"))], max_turns=1)

    assert result.error is not None
    assert result.error != LIMIT_ERROR.format(1)


@pytest.mark.asyncio
async def test_without_a_limit_the_turn_runs_every_step(cli):
    events = [event for n in range(5) for event in step(f"c{n}")]
    result, proc = await cli([_jsonl(*events, *step(answer="done"))])

    assert not proc.killed
    assert (result.error, result.tool_call_count) == (None, 5)


@pytest.mark.asyncio
async def test_a_turn_stopped_at_its_limit_skips_the_output_format_check(cli):
    result, _ = await cli([_jsonl(*step("c1"))], hang=True, max_turns=1,
                          output_format={"type": "json_schema", "schema": {"type": "object"}})

    assert (result.error, result.structured_output) == (LIMIT_ERROR.format(1), None)


def test_a_turn_limit_kills_a_real_cli_and_returns_without_waiting_for_it(monkeypatch, tmp_path):
    pid_file = tmp_path / "pid"
    lines = _jsonl(*step("c1"), *step("c2")).decode()
    script = f"echo $$ > {pid_file}; printf %s {shlex.quote(lines)}; exec sleep 600"
    spawn = asyncio.create_subprocess_exec

    async def cli_that_keeps_running(*_cmd, **kwargs):
        return await spawn("sh", "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cli_that_keeps_running)
    monkeypatch.setattr(opencode_bridge, "_context_xdg", lambda _context: {})

    result = asyncio.run(asyncio.wait_for(
        opencode_bridge.run_opencode(prompt="p", context_id="limit-test", timeout_s=60, max_turns=2,
                                     web_tools=True), 10))

    assert (result.error, result.tool_call_count) == (LIMIT_ERROR.format(2), 2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
