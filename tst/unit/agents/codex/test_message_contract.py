"""What a turn sends the CLI and what agent-env reads back: the terminal message's parts for a
success and for each failure, and the session threading across turns."""

import asyncio

import pytest

from tst.unit.agents.codex.agent import a2a_server, codex_bridge, codex_kit

AGENT_CONFIG_PATH = a2a_server.AGENT_CONFIG_V1.endpoint
TRAJECTORY_PATH = a2a_server.TRAJECTORY_V1.endpoint


class Calls(list):
    """Every run_codex call's arguments; each is answered with ``result_for(kwargs)``."""

    def __init__(self) -> None:
        super().__init__()
        self.result_for = lambda kwargs: codex_bridge.CodexResult(response="answer", tool_call_count=7, session_id="tid-1")


@pytest.fixture
def calls(monkeypatch) -> Calls:
    recorded = Calls()

    async def _run_codex(**kwargs):
        recorded.append(kwargs)
        return recorded.result_for(kwargs)

    monkeypatch.setattr(a2a_server, "run_codex", _run_codex)
    return recorded


def run_turns(codex, *turns: tuple[str, str | None]) -> list[dict]:
    """message/send for each ``(text, context_id)`` in order, over one client; the tasks."""

    async def exercise():
        async with codex.client() as client:
            return [await codex.send(client, text, context_id=context_id) for text, context_id in turns]

    return asyncio.run(exercise())


def test_a_successful_turn(codex, calls):
    """The agent's own data part keeps every legacy key; the framework's carries usage."""
    calls.result_for = lambda kwargs: codex_bridge.CodexResult(
        response="answer", tool_call_count=7, session_id="tid-1", attempts=2,
        token_usage={"input_tokens": 3}, warnings=["w"],
    )

    (task,) = run_turns(codex, ("do the thing", "ctx-1"))
    state, text, data = codex.terminal(task)

    assert (state, text) == ("completed", "answer")
    assert data == [
        {
            "tool_call_count": 7,
            "cli_attempts": 2,
            "token_usage": {"input_tokens": 3},
            "warnings": ["w"],
        },
        {"usage": {"tool_call_count": 7}},
    ]


def test_a_cli_error_is_a_failed_task_with_the_capped_message(codex, calls):
    """error_type/error_message come from the framework's part alone, capped at 4000."""
    error = "codex CLI exited with code 1: " + "x" * 5000
    calls.result_for = lambda kwargs: codex_bridge.CodexResult(response="", tool_call_count=7, session_id="tid-1", error=error)

    (task,) = run_turns(codex, ("do the thing", "ctx-1"))
    state, text, (own, framework) = codex.terminal(task)

    assert (state, text) == ("failed", error)
    assert set(own) == {"tool_call_count"}
    assert framework == {
        "usage": {"tool_call_count": 7},
        "error_type": "agent_error",
        "error_code": "cli_error",
        "error_message": error[:4000],
    }


def test_a_bridge_exception_is_an_infra_error_with_its_partial_trajectory(codex, monkeypatch):
    async def _raise(**kwargs):
        kwargs["event_callback"]({"type": "thread.started", "thread_id": "tid-raise"})
        raise RuntimeError("codex CLI event limit exceeded")

    monkeypatch.setattr(a2a_server, "run_codex", _raise)

    async def exercise():
        async with codex.client() as client:
            task = await codex.send(client, "go", context_id="ctx-raise")
            trajectory = await client.post(TRAJECTORY_PATH, json={"task_id": task["id"]})
            return task, trajectory

    task, trajectory = asyncio.run(exercise())
    state, text, (framework,) = codex.terminal(task)

    assert (state, text) == ("failed", "codex CLI event limit exceeded")
    assert framework == {
        "error_type": "infra_error",
        "error_code": "infra_error",
        "error_message": "codex CLI event limit exceeded",
    }
    assert trajectory.json() == {"trajectory": [{"type": "thread.started", "thread_id": "tid-raise"}]}
    assert codex.agent.session_ref_for_context("ctx-raise") == "tid-raise"


def test_a_context_resumes_its_thread_and_a_new_context_starts_one(codex, calls):
    calls.result_for = lambda kwargs: codex_bridge.CodexResult(response="answer", tool_call_count=7, session_id="tid-7")

    run_turns(codex, ("one", "ctx-resume"), ("two", "ctx-resume"), ("three", "ctx-other"))

    assert [call["thread_id"] for call in calls] == [None, "tid-7", None]


def test_agent_config_reaches_the_cli(codex, calls):
    schema = {"type": "json_schema", "schema": {"type": "object"}}

    async def exercise():
        async with codex.client() as client:
            await client.post(AGENT_CONFIG_PATH, json={
                "model": "gpt-x", "effort": "high", "system_prompt": "be brief",
                "output_format": schema, "timeout_seconds": 42,
                "project_id": "p-1", "task_id": "t-1",
            })
            await codex.send(client, "line one", context_id="ctx-cfg")

    asyncio.run(exercise())

    (call,) = calls
    call.pop("event_callback")
    assert call == {
        "prompt": "line one", "thread_id": None, "model": "gpt-x", "effort": "high",
        "system_prompt": "be brief", "output_format": schema, "timeout_s": 42,
        "project_id": "p-1", "task_id": "t-1", "config_overrides": [],
    }


def test_the_default_timeout_is_thirty_minutes(codex, calls):
    run_turns(codex, ("go", None))

    assert (calls[0]["timeout_s"], calls[0]["model"], calls[0]["effort"]) == (1800, None, None)


def test_session_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.CodexAgent, codex_kit.DurableSessionRefs)
