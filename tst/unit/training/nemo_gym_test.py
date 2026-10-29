"""``write_server`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake steps: the
resources server's folder, its config and its rows, from a server's URL and from a bundle served while it writes."""

import json
import tomllib
from importlib.metadata import version

import pytest
import yaml
from click.testing import CliRunner

from agent_env.openenv.app import _app, _score_at
from agent_env.openenv.environment import END_TURN, Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from agent_env.training._openenv import Server
from agent_env.training.nemo_gym.__main__ import main
from agent_env.training.nemo_gym.prepare import APP, write_server
from tst.unit.bundle._support import layout
from tst.unit.task.episode.run_test import (
    ALICE,
    ITEMS,
    ITEMS_MCP,
    PANTRY,
    PANTRY_MCP,
    REPLIES,
    TURN,
    _tool_sessions,
)

PLAYED = [ITEMS, ALICE, TURN, REPLIES]
ADD_ITEM = {"type": "function", "name": "add_item", "description": "add_item.", "parameters": {"type": "object"},
            "strict": False}
FUNCTION_END_TURN = {"type": "function", "name": "end_turn", "description": END_TURN.description,
                     "parameters": END_TURN.input_schema, "strict": False}


@pytest.fixture
def served(teardowns, tools):
    """Serves ``tasks`` by name in the split ``all`` and those ``splits`` name in theirs, as ``build_app`` serves a
    bundle's; stopped after the test, if the test hasn't stopped it to wait out its episodes' teardowns."""
    servers = []

    def serve(tasks, splits=None, max_concurrent_envs=8):
        by_name = {}
        for name, steps in tasks.items():
            task = Task.from_dict({"id": name, "version": 1, "steps": steps})
            by_name[name] = ServedTask(task, name, EpisodeSpec.from_task(task), _score_at("replies"))
        named = {split: tuple(by_name[name] for name in names) for split, names in (splits or {}).items()}
        servers.append(Server(_app(Served(splits={"all": tuple(by_name.values()), **named}, sandbox=None), "Dock Yard",
                                   max_concurrent_envs)))
        return servers[-1]

    yield serve
    for server in servers:
        server.stop()


def test_a_servers_tasks_become_a_resources_server_its_config_and_a_row_per_task_of_each_split(served, tmp_path,
                                                                                            teardowns):
    server = served({"a": PLAYED, "b": [ITEMS, ALICE, {**TURN, "prompt": "Add a barrel."}, REPLIES]}, {"smoke": ["b"]})
    url = server.url

    folder = write_server(url, tmp_path)
    server.stop()

    assert folder == tmp_path / "resources_servers" / "dock_yard"
    assert (folder / "app.py").read_text() == APP
    project = tomllib.loads((folder / "pyproject.toml").read_text())
    assert (project["project"]["name"], project["project"]["dependencies"], project["tool"]["uv"]["sources"]) == (
        "dock-yard-resources-server", ["nemo-gym", f"agentenv-framework[nemo-gym]=={version('agentenv-framework')}"],
        {"nemo-gym": {"path": "../..", "editable": True}})
    assert yaml.safe_load((folder / "configs" / "dock_yard.yaml").read_text()) == {
        "dock_yard": {"resources_servers": {"dock_yard": {
            "entrypoint": "app.py", "domain": "agent", "description": "Dock Yard's tasks, served by AgentEnv", "url": url,
        }}},
        "dock_yard_simple_agent": {"responses_api_agents": {"simple_agent": {
            "entrypoint": "app.py", "resources_server": {"type": "resources_servers", "name": "dock_yard"},
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }}},
        "dock_yard_environment_server": {"environment_servers": {"legacy_agent": {
            "entrypoint": "app.py", "agent_server": {"type": "responses_api_agents", "name": "dock_yard_simple_agent"},
        }}},
    }
    system = {"role": "system", "content": "You keep the items."}
    agent = {"type": "responses_api_agents", "name": "dock_yard_simple_agent"}
    assert _rows(folder / "data" / "smoke.jsonl") == [{
        "responses_create_params": {"input": [system, {"role": "user", "content": "Add a barrel."}],
                                    "tools": [ADD_ITEM, FUNCTION_END_TURN]},
        "agent_ref": agent, "split": "smoke", "index": 0, "task_id": "b",
    }]
    assert [(row["split"], row["index"], row["task_id"]) for row in _rows(folder / "data" / "all.jsonl")] == [
        ("all", 0, "a"), ("all", 1, "b")]
    assert len(teardowns.finished) == 1


def test_each_toolset_is_probed_once_and_a_rows_tools_are_its_tasks(served, tmp_path, monkeypatch, teardowns):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["add_item"], PANTRY_MCP: ["take_item"]})
    server = served({"a": PLAYED, "p": [PANTRY, {**ALICE, "env_ids": ["pantry"]}, TURN, REPLIES], "c": PLAYED})

    folder = write_server(server.url, tmp_path, name="dock")
    server.stop()

    assert [[tool["name"] for tool in row["responses_create_params"]["tools"]]
            for row in _rows(folder / "data" / "all.jsonl")] == [
        ["add_item", "end_turn"], ["take_item", "end_turn"], ["add_item", "end_turn"]]
    assert len(teardowns.finished) == 2


def test_a_split_whose_name_isnt_a_file_name_is_refused_before_anything_is_written(served, tmp_path):
    server = served({"a": PLAYED}, {"../escape": ["a"]})

    with pytest.raises(ValueError, match=r"the server's splits \['\.\./escape'\] can't name a data file"):
        write_server(server.url, tmp_path / "gym")

    assert list(tmp_path.iterdir()) == []


def test_a_tool_named_after_a_route_gym_serves_is_refused(served, tmp_path, tools):
    tools("verify")
    server = served({"a": PLAYED})

    with pytest.raises(ValueError, match="the env's tool 'verify' can't be a NeMo Gym tool"):
        write_server(server.url, tmp_path)


def test_a_bundle_is_served_while_the_rows_are_written_and_the_config_has_the_server_serve_it(tmp_path, monkeypatch,
                                                                                               teardowns):
    monkeypatch.setenv("HOME", str(tmp_path))
    unbound = {key: value for key, value in ALICE.items() if key not in ("env_ids", "role")}
    bundle = layout(tmp_path / "dock", {"tasks/a.json": json.dumps([ITEMS, unbound, TURN, REPLIES])})

    result = CliRunner().invoke(main, [str(bundle), str(tmp_path / "gym"), "--name", "dock", "--sandbox", "local",
                                       "--max-concurrent-envs", "32"])

    assert result.exit_code == 0, result.output
    config = yaml.safe_load((tmp_path / "gym" / "resources_servers" / "dock" / "configs" / "dock.yaml").read_text())
    assert {key: value for key, value in config["dock"]["resources_servers"]["dock"].items()
            if key in ("bundle", "max_concurrent_envs", "sandbox", "url")} == {
        "bundle": str(bundle.resolve()), "max_concurrent_envs": 32, "sandbox": "local"}
    assert len(_rows(tmp_path / "gym" / "resources_servers" / "dock" / "data" / "all.jsonl")) == 1
    assert len(teardowns.finished) == 1


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
