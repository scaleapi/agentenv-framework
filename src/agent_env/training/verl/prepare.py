"""Writes a bundle's tasks as verl's RL dataset: a parquet file of rows per split, which ``data.train_files`` and
``data.val_files`` take, and ``agent_loop.yaml``, which ``actor_rollout_ref.rollout.agent.agent_loop_config_path``
takes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from agent_env.training._openenv import function_tool, key, longest_prompt, messages, tasks_and_tools, token_in

#: The ``agent_name`` every row plays under, and the name ``agent_loop.yaml`` registers the loop at.
AGENT = "agentenv"
LOOP = "agent_env.training.verl.loop.AgentEnvAgentLoop"
#: A row's columns, so a split the server keeps empty still has them.
ROW = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("agent_name", pa.string()),
    ("extra_info", pa.struct([("split", pa.string()), ("index", pa.int64()), ("task_id", pa.string()),
                              ("tools", pa.string())])),
])


@dataclass(frozen=True)
class Written:
    """What ``write_dataset`` wrote: the folder, and the longest prompt in tokens with its tools when it counted
    them."""

    folder: Path
    longest_prompt: int | None = None


def write_dataset(url: str, out: str | Path, *, tool_timeout: float = 60.0, reset_timeout: float = 600.0,
                  attempts: int = 3, token_var: str | None = None, tokenizer: str | None = None,
                  max_prompt_length: int | None = None, chat_template_kwargs: dict[str, Any] | None = None) -> Written:
    """The tasks of the OpenEnv server at ``url`` as ``out/<split>.parquet``, a row per task of each split, and
    ``out/agent_loop.yaml``, which points the rows' agent loop at ``url`` with the deadlines and the ``attempts`` a
    rollout the env fails is played. A row's prompt is its task's system prompt and first prompt, and its tools every
    tool its task's turns offer, found by playing one task of each toolset to its last turn, unscored, so writing
    deploys those.

    ``token_var`` names the environment variable that holds the token a server given one asks for
    (``build_app(token=...)``): the writer reads it now and the loop in each of verl's workers, so the token is in no
    file. With ``tokenizer``, the model's as transformers loads it, each row's prompt is counted with its tools as the
    model's chat template renders them with ``chat_template_kwargs``, verl's ``data.apply_chat_template_kwargs``, and a
    row over ``max_prompt_length`` is refused before anything is written, where verl would cut it from the left; verl's
    own length filter doesn't count a row's tools."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, not {attempts}")
    if tokenizer is None and (max_prompt_length is not None or chat_template_kwargs):
        raise ValueError("max_prompt_length and chat_template_kwargs count tokens, so they take the model's tokenizer")
    url = url.rstrip("/")
    env_name, tasks, probed = tasks_and_tools(url, local="serve the bundle with agent_env.openenv.build_app first",
                                              timeout=reset_timeout, token=token_in(token_var))
    tools = {each: [function_tool(tool) for tool in found] for each, found in probed.items()}
    rows = {split: [{
        "data_source": f"agentenv/{env_name}",
        "prompt": messages(task["system_prompt"], task["prompt"]),
        "agent_name": AGENT,
        "extra_info": {"split": split, "index": task["index"], "task_id": task["id"],
                       "tools": json.dumps(tools[key(task)])},
    } for task in described] for split, described in tasks.items()}
    longest = None if tokenizer is None else _longest(rows, AutoTokenizer.from_pretrained(tokenizer), max_prompt_length,
                                                      chat_template_kwargs or {})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split, written in rows.items():
        pq.write_table(pa.Table.from_pylist(written, schema=ROW), out / f"{split}.parquet")
    (out / "agent_loop.yaml").write_text(yaml.safe_dump([{
        "name": AGENT, "_target_": LOOP, "url": url, "tool_timeout": tool_timeout, "reset_timeout": reset_timeout,
        "attempts": attempts, **({"token_var": token_var} if token_var else {}),
    }], sort_keys=False))
    return Written(out, longest)


def _longest(rows: dict[str, list[dict[str, Any]]], tokenizer: Any, limit: int | None,
             chat_template_kwargs: dict[str, Any]) -> int:
    lengths = {(split, row["extra_info"]["task_id"]): len(tokenizer.apply_chat_template(
        row["prompt"], tools=json.loads(row["extra_info"]["tools"]), add_generation_prompt=True, tokenize=True,
        return_dict=True, **chat_template_kwargs)["input_ids"]) for split, written in rows.items() for row in written}
    return longest_prompt(lengths, limit, cut="which verl would cut from the left: raise it, and data.max_prompt_length "
                                              "with it")
