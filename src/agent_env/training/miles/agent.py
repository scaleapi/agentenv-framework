"""What Miles calls for each sample of the rows ``python -m agent_env.training.miles`` writes: the agent function
(``--custom-agent-function-path agent_env.training.miles.agent.run``), the reward hook
(``--custom-rm-path agent_env.training.miles.agent.reward``) and the dynamic-sampling filter
(``--dynamic-sampling-filter-path agent_env.training.miles.agent.keep_scored_groups``). Imports nothing of Miles: it
calls them by their arguments."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from agent_env.training._openenv import Session, Shared, token_in


class _Episode(Session):
    """One sample's episode on the server its row names."""

    offered_by = "its row lists"

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared


async def run(base_url: str, prompt: list[dict[str, Any]], request_kwargs: dict[str, Any] | None = None,
              metadata: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Plays the row's task as one episode: the model chats at ``base_url``, the sample's session, which Miles records,
    with the row's tools; each call is a call on the episode, and the model's last reply, or the turn cap, ends the
    turn. A call whose arguments aren't a JSON object is answered with why, as the model's error, and the episode goes
    on. Returns what Miles merges into the sample's metadata: the episode's reward and scores, or, when the env or the
    session failed it, the model missed the row's ``model_timeout`` or the episode ended without a reward, a reward of 0
    and why, which ``keep_scored_groups`` leaves out of training; and the calls made and refused. A failed episode
    isn't played again: the session already holds its turns."""
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
    """Each sample's episode reward, which ``run`` put in its metadata."""
    if isinstance(samples, list):
        return [float(sample.metadata.get("reward", 0.0)) for sample in samples]
    return float(samples.metadata.get("reward", 0.0))


def keep_scored_groups(args: Any, samples: list[Any], **kwargs: Any) -> bool:
    """Whether to train on a group: not when an episode of it failed in the env, its agent function never returned
    (a child process that died mid-episode leaves the session's turns with no reward), or Miles aborted a sample of
    it, none of them the policy's doing."""
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
