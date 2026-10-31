"""Writes a bundle's tasks as verl's RL dataset: a parquet file of rows per split, which ``data.train_files`` and
``data.val_files`` take, and ``agent_loop.yaml``, which ``actor_rollout_ref.rollout.agent.agent_loop_config_path``
takes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from agent_env.training._openenv import check_split_names, key, messages, probe, read_tasks, toolsets

#: The ``agent_name`` every row plays under, and the name ``agent_loop.yaml`` registers the loop at.
AGENT = "agentenv"
LOOP = "agent_env.training.verl.loop.AgentEnvAgentLoop"


def write_dataset(url: str, out: str | Path, *, tool_timeout: float = 60.0, reset_timeout: float = 600.0,
                  attempts: int = 3) -> Path:
    """The tasks of the OpenEnv server at ``url`` as ``out/<split>.parquet``, a row per task of each split, and
    ``out/agent_loop.yaml``, which points the rows' agent loop at ``url`` with the deadlines and the ``attempts`` a
    rollout the env fails is played; returns ``out``. A row's prompt is its task's system prompt and first prompt, and
    its tools every tool its task's turns offer, found by playing one task of each toolset to its last turn, unscored,
    so writing deploys those."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, not {attempts}")
    url = url.rstrip("/")
    env_name, tasks = read_tasks(url, local="serve the bundle with agent_env.openenv.build_app first")
    check_split_names(tasks)
    tools = {each: [_function(tool) for tool in asyncio.run(probe(url, toolset, reset_timeout))[1]]
             for each, toolset in toolsets(env_name, tasks).items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split, described in tasks.items():
        pq.write_table(pa.Table.from_pylist([{
            "data_source": f"agentenv/{env_name}",
            "prompt": messages(task["system_prompt"], task["prompt"]),
            "agent_name": AGENT,
            "extra_info": {"split": split, "index": task["index"], "task_id": task["id"],
                           "tools": json.dumps(tools[key(task)])},
        } for task in described]), out / f"{split}.parquet")
    (out / "agent_loop.yaml").write_text(yaml.safe_dump([{
        "name": AGENT, "_target_": LOOP, "url": url, "tool_timeout": tool_timeout, "reset_timeout": reset_timeout,
        "attempts": attempts,
    }], sort_keys=False))
    return out


def _function(tool: dict[str, Any]) -> dict[str, Any]:
    """An MCP tool as the function tool a chat template renders."""
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description") or "",
        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
    }}
