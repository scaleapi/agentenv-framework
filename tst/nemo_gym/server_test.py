"""``AgentEnvResourcesServer`` driven as NeMo Gym's ``simple_agent`` drives it, through a client that keeps the session
cookie Gym keeps per rollout, over the bundle the unit tier serves on loopback: a row's seed starts its task's episode,
the tool routes answer as text, verify ends the turns left open and scores the episode or masks the rollout, a closed
session ends its episode, and a bundle named in the config is served for the server's lifetime. CI runs it on the
nemo-gym ``uv.lock`` pins."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tst.unit.bundle._support import layout
from tst.unit.openenv.environment_test import AGAIN
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN
from tst.unit.training.nemo_gym_test import ADD_ITEM, FUNCTION_END_TURN, PLAYED
from tst.util.capabilities import NEMO_GYM, missing_capability_reason

server = pytest.importorskip("agent_env.training.nemo_gym.server", reason=missing_capability_reason(NEMO_GYM),
                             exc_type=ImportError)
server_utils = pytest.importorskip("nemo_gym.server_utils", reason=missing_capability_reason(NEMO_GYM))


def test_a_rows_seed_plays_its_task_and_verify_ends_its_turn_with_the_last_reply_and_scores_it(served, teardowns):
    server = served({"a": PLAYED})

    with TestClient(_app(url=server.url)) as client:
        assert client.post("/seed_session", json=_row()).json() == {}
        assert client.post("/add_item", json={"item": "crate"}).text == "add_item {'item': 'crate'}"
        verified = _verify(client, _row(), "Added a crate.")
        client.post("/seed_session", json=_row())
        unanswered = _verify(client, _row(), None)
    server.stop()

    assert (verified["reward"], verified["mask_sample"], verified["failure_reason"], verified["task_id"]) == (
        1.0, False, None, "a")
    assert (unanswered["reward"], unanswered["mask_sample"]) == (0.0, False)
    assert len(teardowns.finished) == 2


def test_end_turn_answers_with_the_next_turns_prompt_and_a_turn_left_open_ends_empty(served):
    server = served({"a": [ITEMS, ALICE, TURN, AGAIN, REPLIES]})

    with TestClient(_app(url=server.url)) as client:
        client.post("/seed_session", json=_row())
        assert client.post("/end_turn", json={"response": "Added a crate."}).text == "Add another."
        played = _verify(client, _row(), "Added another.")
        client.post("/seed_session", json=_row())
        stopped = _verify(client, _row(), "Added a crate.")

    assert (played["reward"], stopped["reward"], stopped["mask_sample"]) == (1.0, 0.0, False)


@pytest.mark.parametrize(("tasks", "row", "why"), [
    ({"a": [{**ITEMS, "fail": True}, ALICE, TURN, REPLIES]}, {}, "the env didn't come up"),
    ({"a": PLAYED}, {"task_id": "b"}, "the episode plays task 'a', not the row's 'b'"),
    ({"a": PLAYED}, {"tools": [FUNCTION_END_TURN]},
     "task 'a' offers the tools ['add_item', 'end_turn'], not the ['end_turn'] the row shows the model"),
])
def test_a_rollout_the_environment_fails_or_that_isnt_its_rows_is_masked(served, tasks, row, why):
    server = served(tasks)

    with TestClient(_app(url=server.url)) as client:
        client.post("/seed_session", json=_row(**row))
        answer = client.post("/add_item", json={"item": "crate"}).text
        verified = _verify(client, _row(**row), "Added a crate.")

    assert answer == "The environment failed to answer, so this episode won't be scored."
    assert (verified["reward"], verified["mask_sample"]) == (0.0, True)
    assert why in verified["failure_reason"]


def test_a_closed_session_ends_its_episode_and_a_session_with_none_answers_so(served, teardowns):
    server = served({"a": PLAYED})

    with TestClient(_app(url=server.url)) as client:
        assert client.post("/add_item", json={"item": "crate"}).status_code == 400
        client.post("/seed_session", json=_row())
        client.post("/add_item", json={"item": "crate"})
        client.post("/close_session", json={"resources_session_id": "a", "episode_id": _EPISODE})
        verified = _verify(client, _row(), "Added a crate.")
    server.stop()

    assert (verified["mask_sample"], verified["failure_reason"]) == (
        True, "the rollout has no episode: its session wasn't seeded, or was closed")
    assert len(teardowns.finished) == 1


def test_a_bundle_named_in_the_config_is_served_while_the_server_runs(tmp_path, monkeypatch, teardowns):
    monkeypatch.setenv("HOME", str(tmp_path))
    unbound = {key: value for key, value in ALICE.items() if key not in ("env_ids", "role")}
    bundle = layout(tmp_path / "dock", {"tasks/a.json": json.dumps([ITEMS, unbound, TURN, REPLIES])})

    row = _row(task_id="@local/~/dock/a")

    with TestClient(_app(bundle=str(bundle), max_concurrent_envs=2)) as client:
        client.post("/seed_session", json=row)
        client.post("/add_item", json={"item": "crate"})
        verified = _verify(client, row, "Added a crate.")
        client.post("/seed_session", json=row)
        client.post("/add_item", json={"item": "crate"})

    assert verified["reward"] == 1.0
    assert len(teardowns.finished) == 2


_EPISODE = {"rollout_id": "rollout"}


def _app(**config):
    return server.AgentEnvResourcesServer(
        config=server.AgentEnvResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="dock",
                                                    **config),
        server_client=MagicMock(spec=server_utils.ServerClient),
    ).setup_webserver()


def _row(task_id="a", tools=(ADD_ITEM, FUNCTION_END_TURN)):
    return {"responses_create_params": {"input": [{"role": "user", "content": "Add a crate."}], "tools": list(tools)},
            "split": "all", "index": 0, "task_id": task_id}


def _verify(client, row, reply):
    output = [] if reply is None else [{"id": "msg", "type": "message", "role": "assistant", "status": "completed",
                                        "content": [{"type": "output_text", "text": reply, "annotations": []}]}]
    response = {"id": "resp", "created_at": 0.0, "model": "policy", "object": "response", "output": output,
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
    return client.post("/verify", json={**row, "response": response}).json()

