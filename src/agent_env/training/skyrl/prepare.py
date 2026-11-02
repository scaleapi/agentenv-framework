"""Writes a bundle's tasks as SkyRL's RL dataset: a parquet file of rows per split, which ``data.train_data`` and
``data.val_data`` take. SkyRL's generator shows the model no tools, so each row's system prompt carries its tools as
the model's chat template renders them with ``tools=``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from agent_env.training._openenv import function_tool, key, longest_prompt, messages, tasks_and_tools, token_in
from agent_env.training.skyrl.calls import FORMATS

#: The ``env_class`` every row plays under, and the id the env is registered at.
AGENT = "agentenv"
#: A row's columns, so a split the server keeps empty still has them.
ROW = pa.schema([
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("env_class", pa.string()),
    ("data_source", pa.string()),
    ("episode", pa.struct([("split", pa.string()), ("index", pa.int64()), ("task_id", pa.string()),
                           ("tools", pa.string())])),
    ("server", pa.struct([("url", pa.string()), ("tool_timeout", pa.float64()), ("reset_timeout", pa.float64()),
                          ("attempts", pa.int64()), ("tool_format", pa.string()), ("token_var", pa.string())])),
])
#: What stands in for a system prompt while finding where a chat template puts it.
_MARK = "AGENTENVSYSTEMPROMPT"


@dataclass(frozen=True)
class Written:
    """What ``write_dataset`` wrote: the folder, and the longest prompt in tokens, its tools included."""

    folder: Path
    longest_prompt: int


def write_dataset(url: str, out: str | Path, *, tokenizer: str, tool_format: str = "hermes",
                  tool_timeout: float = 60.0, reset_timeout: float = 600.0, attempts: int = 3,
                  token_var: str | None = None, max_prompt_length: int | None = None,
                  chat_template_kwargs: dict[str, Any] | None = None) -> Written:
    """The tasks of the OpenEnv server at ``url`` as ``out/<split>.parquet``, each row's tools in its system prompt as
    ``tokenizer``'s chat template renders them, since SkyRL's generator passes the template none. ``token_var`` names
    the variable holding the server's token, read here and in the trainer, so the token is in no file."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, not {attempts}")
    if tool_format not in FORMATS:
        raise ValueError(f"tool_format must be one of {', '.join(FORMATS)}, not {tool_format!r}")
    url = url.rstrip("/")
    env_name, tasks, probed = tasks_and_tools(url, local="serve the bundle with agent_env.openenv.build_app first",
                                              timeout=reset_timeout, token=token_in(token_var))
    template = AutoTokenizer.from_pretrained(tokenizer)
    kwargs = chat_template_kwargs or {}
    tools = {each: [function_tool(tool) for tool in found] for each, found in probed.items()}
    server = {"url": url, "tool_timeout": tool_timeout, "reset_timeout": reset_timeout, "attempts": attempts,
              "tool_format": tool_format, "token_var": token_var}
    rows = {split: [{
        "prompt": with_tools(template, messages(task["system_prompt"], task["prompt"]), tools[key(task)], kwargs),
        "env_class": AGENT,
        "data_source": f"{AGENT}/{env_name}",
        "episode": {"split": split, "index": task["index"], "task_id": task["id"],
                    "tools": json.dumps(tools[key(task)])},
        "server": server,
    } for task in described] for split, described in tasks.items()}
    longest = _longest(rows, template, max_prompt_length, kwargs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split, written in rows.items():
        pq.write_table(pa.Table.from_pylist(written, schema=ROW), out / f"{split}.parquet")
    return Written(out, longest)


def token_vars(files: list[str]) -> set[str]:
    """The environment variables the rows in the local parquet ``files`` name for their servers' tokens; a file that
    isn't on this machine, or holds other rows, names none."""
    ours = [file for file in files if Path(file).is_file() and "server" in pq.read_schema(file).names]
    return {server["token_var"] for file in ours
            for server in pq.read_table(file, columns=["server"]).column("server").to_pylist() if server["token_var"]}


def with_tools(tokenizer: Any, prompt: list[dict[str, str]], tools: list[dict[str, Any]],
               chat_template_kwargs: dict[str, Any]) -> list[dict[str, str]]:
    """``prompt`` with ``tools`` in its system prompt, as the chat template renders them with ``tools=``, which
    SkyRL's generator doesn't pass: the system prompt that renders the same text without them. Raises for a template
    that puts tools anywhere but the system turn."""
    target = _rendered(tokenizer, prompt, chat_template_kwargs, tools=tools)
    rest = prompt[1:] if prompt[0]["role"] == "system" else prompt
    marked = _rendered(tokenizer, [{"role": "system", "content": _MARK}, *rest], chat_template_kwargs)
    if marked.count(_MARK) == 1:
        before, after = marked.split(_MARK)
        if target.startswith(before) and target.endswith(after) and len(target) >= len(before) + len(after):
            found = [{"role": "system", "content": target[len(before):len(target) - len(after)]}, *rest]
            if _rendered(tokenizer, found, chat_template_kwargs) == target:
                return found
    raise ValueError(f"{tokenizer.name_or_path}'s chat template doesn't render tools in the system prompt, where this "
                     "writer puts them, since SkyRL's generator passes the template no tools")


def _rendered(tokenizer: Any, prompt: list[dict[str, str]], chat_template_kwargs: dict[str, Any], **tools: Any) -> str:
    return tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False, **tools,
                                         **chat_template_kwargs)


def _longest(rows: dict[str, list[dict[str, Any]]], tokenizer: Any, limit: int | None,
             chat_template_kwargs: dict[str, Any]) -> int:
    """Counted as SkyRL's prompt filter counts a row, without ``chat_template_kwargs``, and as its generator renders it,
    with them."""
    lengths = {(split, row["episode"]["task_id"]): max(
        len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True,
                                          **kwargs)["input_ids"])
        for kwargs in ({}, chat_template_kwargs)) for split, written in rows.items() for row in written}
    return longest_prompt(lengths, limit, cut="which SkyRL's prompt filter would drop: raise it, and "
                                              "trainer.max_prompt_length with it")
