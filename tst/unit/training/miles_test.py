"""``agent_env.training.miles`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake
steps: the rows ``write_rows`` writes, and the agent function playing a row's task through a stand-in for Miles'
session endpoint, which answers each chat request from a script and records it; with the reward hook and the filter
Miles calls beside it."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_env.openenv.environment import END_TURN
from agent_env.openenv.reward import _unscored
from agent_env.training._openenv import Server
from agent_env.training.miles.__main__ import main
from agent_env.training.miles.agent import keep_scored_groups, reward, run
from agent_env.training.miles.prepare import write_rows
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN
from tst.unit.training.nemo_gym_test import PLAYED

ADD_ITEM = {"type": "function", "function": {"name": "add_item", "description": "add_item.",
                                             "parameters": {"type": "object"}}}
FUNCTION_END_TURN = {"type": "function", "function": {"name": "end_turn", "description": END_TURN.description,
                                                      "parameters": END_TURN.input_schema}}
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 256}


def _call(arguments, id="call_1", name="add_item"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def _says(text):
    return {"role": "assistant", "content": text}


@pytest.fixture
def session():
    """Miles' session endpoint, answering the n-th chat request with ``script[n]``, the last again after it ends, with
    the HTTP ``status`` after ``delay`` seconds; ``.requests`` are the bodies it got."""
    servers = []

    def serving(*script, status=200, delay=0.0):
        requests = []

        async def completions(request):
            requests.append(await request.json())
            await asyncio.sleep(delay)
            message = script[min(len(requests), len(script)) - 1]
            return JSONResponse({"choices": [{"index": 0, "message": message}]}, status_code=status)

        server = Server(Starlette(routes=[Route("/sessions/s1/v1/chat/completions", completions, methods=["POST"])]))
        server.requests, server.base_url = requests, f"{server.url}/sessions/s1"
        servers.append(server)
        return server

    yield serving
    for server in servers:
        server.stop()


def _row(server, tmp_path, **options):
    folder = write_rows(server.url, tmp_path / "rows", **options)
    return json.loads((folder / "all.jsonl").read_text().splitlines()[0])


def test_a_servers_tasks_become_a_row_per_task_of_each_split(serve, tmp_path, teardowns, monkeypatch):
    monkeypatch.setenv("DOCK_TOKEN", "unread")
    server = serve({"a": PLAYED, "b": [ITEMS, ALICE, {**TURN, "prompt": "Add a barrel."}, REPLIES]}, {"smoke": ["b"]},
                   name="Dock Yard")

    folder = write_rows(f"{server.url}/", tmp_path / "rows", tool_timeout=30.0, reset_timeout=90.0, model_timeout=120.0,
                        max_turns=3, token_var="DOCK_TOKEN")
    server.stop()

    [row] = [json.loads(line) for line in (folder / "smoke.jsonl").read_text().splitlines()]
    assert sorted(path.name for path in folder.iterdir()) == ["all.jsonl", "smoke.jsonl"]
    assert row == {
        "messages": [{"role": "system", "content": "You keep the items."}, {"role": "user", "content": "Add a barrel."}],
        "metadata": {"split": "smoke", "index": 0, "task_id": "b", "tools": [ADD_ITEM, FUNCTION_END_TURN],
                     "agentenv": {"url": server.url, "tool_timeout": 30.0, "reset_timeout": 90.0,
                                  "model_timeout": 120.0, "max_turns": 3, "token_var": "DOCK_TOKEN"}},
    }
    assert [json.loads(line)["metadata"]["task_id"] for line in (folder / "all.jsonl").read_text().splitlines()] == [
        "a", "b"]
    assert len(teardowns.finished) == 1


def test_the_agent_plays_a_rows_task_through_the_session_and_returns_its_reward(serve, session, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)
    model = session(_call('{"item": "crate"}'), _says("Added a crate."))

    played = asyncio.run(run(model.base_url, row["messages"], SAMPLING, row["metadata"]))
    server.stop()

    assert played == {"reward": 1.0, "agentenv_scores": {"replies": 1.0}, "agentenv_tool_calls": 1,
                      "agentenv_invalid_calls": 0}
    first, second = model.requests
    assert first == {"model": "default", "messages": row["messages"], "tools": row["metadata"]["tools"], **SAMPLING}
    assert second["messages"] == [*row["messages"], _call('{"item": "crate"}'),
                                  {"role": "tool", "tool_call_id": "call_1", "content": second["messages"][-1]["content"]}]
    assert second["messages"][-1]["content"]


def test_a_call_whose_arguments_arent_an_object_is_answered_and_the_episode_goes_on(serve, session, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path)
    model = session(_call('"{\\"item\\": \\"crate\\"}"'), _call("not json", id="call_2"), _call("", id="call_3"),
                    _says("Added a crate."))

    played = asyncio.run(run(model.base_url, row["messages"], SAMPLING, row["metadata"]))
    server.stop()

    assert (played["reward"], played["agentenv_tool_calls"], played["agentenv_invalid_calls"]) == (1.0, 0, 3)
    answers = [message["content"] for message in model.requests[-1]["messages"] if message["role"] == "tool"]
    assert answers[0] == "add_item wasn't called: its arguments must be a JSON object, not str"
    assert answers[1].startswith("add_item wasn't called: its arguments aren't JSON")
    assert answers[2].startswith("add_item wasn't called: its arguments aren't JSON")
    assert "agentenv_failure" not in played


def test_a_rollout_that_reaches_its_turn_cap_ends_the_turn_without_a_reply(serve, session, tmp_path):
    server = serve({"a": PLAYED})
    row = _row(server, tmp_path, max_turns=2)
    model = session(_call('{"item": "crate"}'))

    played = asyncio.run(run(model.base_url, row["messages"], SAMPLING, row["metadata"]))
    server.stop()

    assert (played["reward"], played["agentenv_tool_calls"], len(model.requests)) == (0.0, 2, 2)


@pytest.mark.parametrize(("failing", "why"), [("env", "couldn't open a session on the server"),
                                              ("session", "HTTPStatusError: Server error '503"),
                                              ("model", "ReadTimeout"), ("reward", "the episode ended without a reward")])
def test_an_episode_that_fails_or_ends_without_a_reward_is_flagged_and_its_group_left_out(serve, session, tmp_path,
                                                                                           failing, why):
    server = serve({"a": PLAYED}, **({"reward": _unscored} if failing == "reward" else {}))
    row = _row(server, tmp_path, reset_timeout=5.0, **({"model_timeout": 0.5} if failing == "model" else {}))
    model = session(_says("Added a crate."), status=503 if failing == "session" else 200,
                    delay=2.0 if failing == "model" else 0.0)
    if failing == "env":
        server.stop()

    played = asyncio.run(run(model.base_url, row["messages"], SAMPLING, row["metadata"]))
    server.stop()

    assert played["reward"] == 0.0 and played["agentenv_failure"].startswith(why)
    failed, scored = _sample({**played}), _sample({"reward": 1.0})
    assert keep_scored_groups(None, [scored, failed]) is False
    assert keep_scored_groups(None, [[scored], [_sample({"reward": 0.0})]]) is True
    assert keep_scored_groups(None, [scored, _sample({"reward": 0.0}, status="ABORTED")]) is False
    assert keep_scored_groups(None, [scored, _sample({"session_id": "s1"})]) is False


def test_the_reward_hook_reads_each_samples_reward_from_its_metadata():
    one, two = _sample({"reward": 0.25}), _sample({})

    assert asyncio.run(reward(None, one)) == 0.25
    assert asyncio.run(reward(None, [one, two])) == [0.25, 0.0]


def test_the_command_writes_the_rows(serve, tmp_path, teardowns):
    server = serve({"a": PLAYED})

    result = CliRunner().invoke(main, [server.url, str(tmp_path / "rows"), "--max-turns", "2"])
    server.stop()

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in (tmp_path / "rows").iterdir()) == ["all.jsonl"]
    [line] = (tmp_path / "rows" / "all.jsonl").read_text().splitlines()
    assert json.loads(line)["metadata"]["agentenv"]["max_turns"] == 2
    assert len(teardowns.finished) == 1


def _sample(metadata, status="COMPLETED"):
    return SimpleNamespace(metadata=metadata, status=SimpleNamespace(name=status))
