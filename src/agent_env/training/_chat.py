"""What the Miles and slime helpers share: a server's tasks as chat rows, and a row's task played as one episode by a
model chatting at a trainer's OpenAI chat endpoint, a session of the trainer's that records the model's turns for
training. A row holds the task's system prompt and first prompt as ``messages``, and in ``metadata`` its episode, the
tools its toolset offers and the server it plays on with its settings, so the trainer's rollout processes need no
settings of their own."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import httpx

from agent_env.training._openenv import Session, Shared, function_tool, key, messages, tasks_and_tools, token_in


@click.command()
@click.argument("url")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--tool-timeout", type=float, default=60.0, show_default=True,
              help="One tool call's deadline in seconds; the server ends a call that misses it, and the episode.")
@click.option("--reset-timeout", type=float, default=600.0, show_default=True,
              help="How long an episode may take to deploy, and each turn to be ended, in seconds; also how long a "
                   "rollout waits for room on the server.")
@click.option("--model-timeout", type=float, default=600.0, show_default=True,
              help="How long a rollout waits for each of the model's replies, in seconds; an episode whose "
                   "model misses it fails, and its group is left out.")
@click.option("--max-turns", type=int, default=4, show_default=True,
              help="The model turns a rollout may take; the episode is ended after the last.")
@click.option("--token-var", help="The environment variable that holds the token the server asks for, read now and in "
                                  "each of the trainer's rollout processes.")
def command(url: str, out: str, tool_timeout: float, reset_timeout: float, model_timeout: float, max_turns: int,
         token_var: str | None) -> None:
    """Writes the tasks of the OpenEnv server at URL as OUT/<split>.jsonl, a row per task, for the trainer's
    --prompt-data with --input-key messages and --metadata-key metadata."""
    folder = write_rows(url, out, tool_timeout=tool_timeout, reset_timeout=reset_timeout, model_timeout=model_timeout,
                        max_turns=max_turns, token_var=token_var)
    click.echo(f"wrote {', '.join(sorted(path.name for path in folder.iterdir()))} to {folder}")



def write_rows(url: str, out: str | Path, *, tool_timeout: float = 60.0, reset_timeout: float = 600.0,
               model_timeout: float = 600.0, max_turns: int = 4, token_var: str | None = None) -> Path:
    """The tasks of the OpenEnv server at ``url`` as ``out/<split>.jsonl``, a row per task of each split: its system
    prompt and first prompt as ``messages``, and as ``metadata`` its split, index and task id, the tools its toolset
    offers, and where and how ``play`` plays it: the server, its deadlines and the model's in seconds, the model turns a
    rollout may take, and the environment variable holding the token a server given one asks for, read in each of the
    trainer's rollout processes."""
    url = url.rstrip("/")
    _, tasks, probed = tasks_and_tools(url, local="serve the bundle with agent_env.openenv.build_app first",
                                       timeout=reset_timeout, token=token_in(token_var))
    tools = {each: [function_tool(tool) for tool in found] for each, found in probed.items()}
    server = {"url": url, "tool_timeout": tool_timeout, "reset_timeout": reset_timeout, "model_timeout": model_timeout,
              "max_turns": max_turns, **({"token_var": token_var} if token_var else {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split, described in tasks.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps({
            "messages": messages(task["system_prompt"], task["prompt"]),
            "metadata": {"split": split, "index": task["index"], "task_id": task["id"], "tools": tools[key(task)],
                         "agentenv": server},
        }) + "\n" for task in described))
    return out


class _Episode(Session):
    """One sample's episode on the server its row names."""

    offered_by = "its row lists"

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared


async def play(base_url: str, prompt: list[dict[str, Any]], request_kwargs: dict[str, Any] | None = None,
               metadata: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Plays the row's task as one episode: the model chats at ``base_url``, a session of the trainer's that records its
    turns, with the row's tools; each call is a call on the episode, and the model's last reply, or the turn cap, ends
    the turn. A call whose arguments aren't a JSON object is answered with why, as the model's error, and the episode
    goes on. Returns the sample's metadata: the episode's reward and scores, or, when the env or the session failed it,
    the model missed the row's ``model_timeout`` or the episode ended without a reward, a reward of 0 and why, which
    ``keep_scored_groups`` leaves out of training; and the calls made and refused. A failed episode isn't played again:
    the session already holds its turns."""
    row = metadata or {}
    server, tools = row["agentenv"], row["tools"]
    episode = _Episode(Shared(server["url"], frozenset(tool["function"]["name"] for tool in tools),
                              server["tool_timeout"], server["reset_timeout"], patience=server["reset_timeout"],
                              waits_for_server=True, token=token_in(server.get("token_var"))))
    counts = {"agentenv_tool_calls": 0, "agentenv_invalid_calls": 0}
    try:
        await episode._start(row["split"], int(row["index"]), row["task_id"])
        reply = await _converse(episode, base_url, list(prompt), tools, request_kwargs or {}, server, counts)
        rollout = episode._rollout
        if not (rollout.done or rollout.failure):
            await episode._finish(rollout, reply)
        if rollout.failure is None and rollout.reward is None:
            rollout.failure = "the episode ended without a reward"
        if rollout.failure is not None:
            return {"reward": 0.0, "agentenv_failure": rollout.failure, **counts}
        return {"reward": float(rollout.reward), "agentenv_scores": rollout.scores or {}, **counts}
    except Exception as e:
        return {"reward": 0.0, "agentenv_failure": f"{type(e).__name__}: {e}", **counts}
    finally:
        await episode._close()


async def reward(args: Any, samples: Any, **kwargs: Any) -> float | list[float]:
    """Each sample's episode reward, which ``play`` put in its metadata."""
    if isinstance(samples, list):
        return [float(sample.metadata.get("reward", 0.0)) for sample in samples]
    return float(samples.metadata.get("reward", 0.0))


def keep_scored_groups(args: Any, samples: list[Any], **kwargs: Any) -> bool:
    """Whether to train on a group: not when an episode of it failed in the env, its episode never returned (a child
    process that died mid-episode leaves the session's turns with no reward), or the trainer aborted a sample of it,
    none of them the policy's doing."""
    return not any(sample.metadata.get("agentenv_failure") or "reward" not in sample.metadata
                   or getattr(sample.status, "name", None) == "ABORTED" for sample in _samples(samples))


async def _converse(episode: _Episode, base_url: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                    request_kwargs: dict[str, Any], server: dict[str, Any], counts: dict[str, int]) -> str:
    """The model's turns, until it replies without a call, the episode ends, or the server's ``max_turns`` pass; its
    last reply."""
    async with httpx.AsyncClient(timeout=server["model_timeout"]) as client:
        for _ in range(server["max_turns"]):
            response = await client.post(f"{base_url}/v1/chat/completions", json={
                "model": "default", "messages": messages, "tools": tools, **request_kwargs})
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            # Verbatim: the session server matches each message a request replays against the one it recorded.
            messages.append(message)
            if not (calls := message.get("tool_calls")):
                return message.get("content") or ""
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": await _answer(episode, call, counts)})
            if episode._rollout.done or episode._rollout.failure:
                return ""
    return ""


async def _answer(episode: _Episode, call: dict[str, Any], counts: dict[str, int]) -> str:
    name, arguments = call["function"]["name"], call["function"].get("arguments")
    try:
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as e:
        counts["agentenv_invalid_calls"] += 1
        return f"{name} wasn't called: its arguments aren't JSON ({e})"
    if not isinstance(arguments, dict):
        counts["agentenv_invalid_calls"] += 1
        return f"{name} wasn't called: its arguments must be a JSON object, not {type(arguments).__name__}"
    counts["agentenv_tool_calls"] += 1
    return await episode._step(name, arguments)


def _samples(group: list[Any]) -> Iterator[Any]:
    for sample in group:
        yield from sample if isinstance(sample, list) else [sample]
