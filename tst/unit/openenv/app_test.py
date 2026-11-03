"""``build_app`` over bundle folders, served in process: the splits it serves and the tasks it leaves out, the reward
and bundles it refuses, an episode played over OpenEnv's own WebSocket after a step before reset is refused,
namespace routing for each of the app's runs, and the token a server given one asks of every request and WebSocket."""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from agent_env.bundle import BundleError
from agent_env.openenv import build_app
from agent_env.openenv.environment import NOT_RESET
from agent_env.store.routing import namespace_routing, namespace_routing_enabled
from agent_env.task import Task
from tst.unit.bundle._support import layout
from tst.unit.task.episode.run_test import ITEMS, REPLIES, TURN

ALICE = {"id": "alice", "type": "deploy_agent", "agent_name": "alice", "a2a_agent_id": "nobody",
         "system_prompt": "You keep the items."}
PLAYED = [ITEMS, ALICE, TURN, REPLIES]
WIRING = [ITEMS, REPLIES]


@pytest.fixture
def bundles(tmp_path, monkeypatch, teardowns):
    monkeypatch.setenv("HOME", str(tmp_path))

    def make(name, files):
        return layout(tmp_path / name, {path: content if isinstance(content, str) else json.dumps(content)
                                        for path, content in files.items()})

    yield make
    assert not namespace_routing_enabled()


def _tasks(client, name, split):
    response = client.post(f"/{name}/tasks", json={"split": split})
    assert response.status_code == 200, response.text
    return [task["name"] for task in response.json()["tasks"]]


def test_all_and_each_eval_are_splits_and_a_task_that_cant_be_played_is_left_out(bundles, caplog):
    root = bundles("dock", {"tasks/a.json": PLAYED, "tasks/b.json": PLAYED, "tasks/wiring.json": WIRING,
                            "evals/smoke.toml": 'tasks = ["a", "wiring"]\n'})
    with caplog.at_level(logging.WARNING, logger="agent_env.openenv.app"):
        app = build_app(str(root))

    with TestClient(app) as client:
        assert [split["name"] for split in client.get("/dock/splits").json()] == ["all", "smoke"]
        assert (_tasks(client, "dock", "smoke"), _tasks(client, "dock", "all")) == (["a"], ["a", "b"])
        task = client.post("/dock/task", json={"split": "all", "index": 1}).json()["task"]
    assert (task["prompt"], task["system_prompt"], task["agent"]) == ("Add a crate.", "You keep the items.", "alice")
    assert "left out: task '@local/~/dock/wiring' prompts no agent" in caplog.text


def test_a_mapping_serves_each_bundle_as_its_split(bundles):
    train = bundles("train", {"tasks/t1.json": PLAYED, "tasks/t2.json": PLAYED})
    held_out = bundles("held-out", {"tasks/e1.json": PLAYED})

    with TestClient(build_app({"train": train, "eval": held_out})) as client:
        assert [split["name"] for split in client.get("/train+held-out/splits").json()] == ["train", "eval"]
        assert (_tasks(client, "train+held-out", "train"), _tasks(client, "train+held-out", "eval")) == (
            ["t1", "t2"], ["e1"])


def test_an_episode_is_played_over_openenvs_websocket_and_the_session_closing_tears_it_down(bundles):
    root = bundles("dock", {"tasks/a.json": PLAYED})

    app = build_app(str(root))
    assert not namespace_routing_enabled()

    with TestClient(app) as client:
        assert namespace_routing_enabled()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "step", "data": {"type": "list_tools"}})
            refused = ws.receive_json()
            ws.send_json({"type": "reset", "data": {"split": "all", "index": 0}})
            reset = ws.receive_json()["data"]
            ws.send_json({"type": "step", "data": {"type": "call_tool", "tool_name": "end_turn",
                                                   "arguments": {"response": "Added a crate."}}})
            ended = ws.receive_json()["data"]
            ws.send_json({"type": "reset", "data": {}})
            ws.receive_json()
            ws.send_json({"type": "state"})
            open_episode = ws.receive_json()["data"]["episode_id"]

    assert (refused["type"], refused["data"]["message"]) == ("error", NOT_RESET)
    assert reset["observation"]["metadata"]["prompt"] == "Add a crate."
    assert (ended["done"], ended["reward"]) == (True, 1.0)
    assert not namespace_routing_enabled()
    with namespace_routing():
        assert Task.get_instance(open_episode).status == "cancelled"
    with TestClient(app):
        assert namespace_routing_enabled()
    assert not namespace_routing_enabled()


@pytest.mark.parametrize(("reward", "message"), [
    (None, "records a score at each of replies, again; name the one that is its reward with reward=<step id>"),
    ("nope", "records no score at step 'nope'; it records them at ['replies', 'again']"),
])
def test_a_reward_that_cant_score_every_task_is_refused(bundles, reward, message):
    root = bundles("dock", {"tasks/a.json": [*PLAYED, {**REPLIES, "id": "again", "verifier_id": "again"}]})

    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        build_app(str(root), reward=reward)
    with TestClient(build_app(str(root), reward="again")):
        pass
    with pytest.raises(ValidationError, match="max_concurrent_envs"):
        build_app(str(root), reward="again", max_concurrent_envs=0)


def test_a_bundle_with_no_task_to_play_or_an_eval_named_all_is_refused_with_why(bundles):
    root = bundles("dock", {"tasks/wiring.json": WIRING})
    named_all = bundles("named-all", {"tasks/a.json": PLAYED, "tasks/b.json": PLAYED, "evals/all.toml": 'tasks = ["a"]\n'})

    with pytest.raises(BundleError, match="no task can be served as an OpenEnv env: task '@local/~/dock/wiring'"):
        build_app(str(root))
    with pytest.raises(BundleError, match="eval 'all' has the name of the split that holds every task"):
        build_app(str(named_all))


def test_a_path_is_a_folder_even_where_an_installed_bundle_has_its_name(bundles, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(BundleError, match="hello"):
        build_app(Path("hello"))


def test_a_dotted_bundle_name_is_served_dashed_as_openenvs_web_interface_reads_a_name_as_a_module(
        bundles, monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "false")
    root = bundles("dock.v1", {"tasks/a.json": PLAYED})

    with TestClient(build_app(root)) as client:
        assert [split["name"] for split in client.get("/dock-v1/splits").json()] == ["all"]


def test_a_server_given_a_token_asks_it_of_every_request_and_websocket_but_health(bundles):
    root = bundles("dock", {"tasks/a.json": PLAYED})

    with TestClient(build_app(str(root), token="s3cret")) as client:
        bare = client.get("/list_environments")
        wrong = client.get("/list_environments", headers={"Authorization": "Bearer guess"})
        right = client.get("/list_environments", headers={"Authorization": "Bearer s3cret"})
        health = client.get("/health")
        with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect("/ws"):
            pass
        with client.websocket_connect("/ws", headers={"Authorization": "Bearer s3cret"}) as ws:
            ws.send_json({"type": "reset", "data": {"split": "all", "index": 0}})
            reset = ws.receive_json()["data"]

    assert (bare.status_code, wrong.status_code, right.status_code, health.status_code) == (401, 401, 200, 200)
    assert bare.json() == {"detail": "this server needs Authorization: Bearer <its token>"}
    assert right.json() == ["dock"]
    assert refused.value.code == 1008
    assert reset["observation"]["metadata"]["prompt"] == "Add a crate."
