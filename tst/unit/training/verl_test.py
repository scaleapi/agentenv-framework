"""``write_dataset`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake steps:
verl's parquet rows for each split and the config of the agent loop that plays them, and the command that writes both."""

import json

import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner

from agent_env.openenv.environment import END_TURN
from agent_env.training.verl.__main__ import main
from agent_env.training.verl.prepare import LOOP, ROW, write_dataset
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
from tst.unit.training.nemo_gym_test import PLAYED

ADD_ITEM = {"type": "function", "function": {"name": "add_item", "description": "add_item.",
                                             "parameters": {"type": "object"}}}
FUNCTION_END_TURN = {"type": "function", "function": {"name": "end_turn", "description": END_TURN.description,
                                                      "parameters": END_TURN.input_schema}}


def test_a_servers_tasks_become_a_row_per_task_of_each_split_and_the_agent_loop_that_plays_them(serve, tmp_path,
                                                                                            teardowns):
    server = serve({"a": PLAYED, "b": [ITEMS, ALICE, {**TURN, "prompt": "Add a barrel."}, REPLIES]}, {"smoke": ["b"]},
                   name="Dock Yard")

    out = write_dataset(f"{server.url}/", tmp_path / "data", tool_timeout=30.0, reset_timeout=90.0, attempts=2)
    server.stop()

    assert out == tmp_path / "data"
    assert yaml.safe_load((out / "agent_loop.yaml").read_text()) == [{
        "name": "agentenv", "_target_": LOOP, "url": server.url, "tool_timeout": 30.0, "reset_timeout": 90.0,
        "attempts": 2,
    }]
    [row] = _rows(out / "smoke.parquet")
    assert row == {
        "data_source": "agentenv/Dock Yard",
        "prompt": [{"role": "system", "content": "You keep the items."}, {"role": "user", "content": "Add a barrel."}],
        "agent_name": "agentenv",
        "extra_info": {"split": "smoke", "index": 0, "task_id": "b", "tools": [ADD_ITEM, FUNCTION_END_TURN]},
    }
    assert [(row["extra_info"]["index"], row["extra_info"]["task_id"]) for row in _rows(out / "all.parquet")] == [
        (0, "a"), (1, "b")]
    assert len(teardowns.finished) == 1


def test_each_toolset_is_probed_once_and_a_rows_tools_are_its_tasks(serve, tmp_path, monkeypatch, teardowns):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["add_item"], PANTRY_MCP: ["take_item"]})
    server = serve({"a": PLAYED, "p": [PANTRY, {**ALICE, "env_ids": ["pantry"]}, TURN, REPLIES], "c": PLAYED})

    out = write_dataset(server.url, tmp_path)
    server.stop()

    assert [[tool["function"]["name"] for tool in row["extra_info"]["tools"]] for row in _rows(out / "all.parquet")] == [
        ["add_item", "end_turn"], ["take_item", "end_turn"], ["add_item", "end_turn"]]
    assert len(teardowns.finished) == 2


def test_a_split_the_server_keeps_empty_has_a_rows_columns(serve, tmp_path):
    server = serve({"a": PLAYED}, {"empty": []})

    out = write_dataset(server.url, tmp_path)
    server.stop()

    empty = pq.read_table(out / "empty.parquet")
    assert (empty.num_rows, empty.schema) == (0, ROW)
    assert pq.read_table(out / "all.parquet").schema == ROW


def test_a_split_whose_name_isnt_a_file_name_is_refused_before_anything_is_written(serve, tmp_path):
    server = serve({"a": PLAYED}, {"../escape": ["a"]})

    with pytest.raises(ValueError, match=r"the server's splits \['\.\./escape'\] can't name a data file"):
        write_dataset(server.url, tmp_path / "data")

    assert list(tmp_path.iterdir()) == []


def test_a_rollout_is_played_at_least_once(tmp_path):
    with pytest.raises(ValueError, match="attempts must be at least 1, not 0"):
        write_dataset("http://127.0.0.1:1", tmp_path / "data", attempts=0)
    assert list(tmp_path.iterdir()) == []


def test_the_command_writes_the_rows_and_the_agent_loop_config(serve, tmp_path, teardowns):
    server = serve({"a": PLAYED})

    result = CliRunner().invoke(main, [server.url, str(tmp_path / "data"), "--tool-timeout", "15",
                                       "--reset-timeout", "120", "--attempts", "4"])
    server.stop()

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in (tmp_path / "data").iterdir()) == ["agent_loop.yaml", "all.parquet"]
    [config] = yaml.safe_load((tmp_path / "data" / "agent_loop.yaml").read_text())
    assert (config["tool_timeout"], config["reset_timeout"], config["attempts"]) == (15.0, 120.0, 4)
    assert len(teardowns.finished) == 1


def _rows(path):
    return [{**row, "extra_info": {**row["extra_info"], "tools": json.loads(row["extra_info"]["tools"])}}
            for row in pq.read_table(path).to_pylist()]
