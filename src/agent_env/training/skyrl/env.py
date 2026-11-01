"""SkyRL's env for the rows ``python -m agent_env.training.skyrl`` writes. ``AgentEnvTextEnv`` is a skyrl-gym
``BaseTextEnv`` playing a row's task as one episode on the bundle's OpenEnv server: each step runs the tool calls in the
model's reply on the episode and answers with their results, and a reply without one ends the turn it left open, which
scores the episode once its last turn ends. SkyRL calls an env from a thread of its pool, so the episode's session
runs on the event loop this process's sessions share."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
    from skyrl_gym.metrics import default_aggregate_metrics
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] != "skyrl_gym":
        raise
    raise ImportError("agent_env.training.skyrl.env runs in SkyRL's trainer: install 'agentenv-framework[skyrl]' into "
                      "SkyRL's environment") from exc

from agent_env.training._openenv import CLOSE_SECONDS, GRACE_SECONDS, Rollout, Session, Shared, on_session_loop, token_in
from agent_env.training.skyrl.calls import Call, Unparsed, parse

logger = logging.getLogger(__name__)


class EpisodeFailed(RuntimeError):
    """The env failed the episode: a deploy, a call past its deadline, or a server that went away. AgentEnvGenerator
    plays the rollout again on a new episode."""


class AgentEnvTextEnv(BaseTextEnv):
    """Plays the row's task as one episode on the OpenEnv server its row names. ``init`` starts the episode, and
    returns before it is deployed, so the deploy overlaps the model's first turn. Each ``step`` runs the tool calls of
    the model's reply in order, a tool call SkyRL's parse finds nothing to call with answered with why, and returns
    their answers as ``tool`` messages; a reply without a tool call ends the turn it left open, and any after it end
    empty, as does the reply of the ``generator.max_turns``-th step, and the step that ends the episode's last turn
    returns its reward. A failure of the env raises EpisodeFailed; a task that ends unscored without failing, or offers
    a tool its row doesn't list, is a configuration error no replay changes, and raises ValueError.

    ``get_metrics`` holds each of the episode's scores by the id of the step that recorded it (``score/<step id>``),
    the tool calls run and the unparsed ones; SkyRL logs their means per step and per eval as
    ``environment/avg_<metric>``."""

    def __init__(self, env_config: Any = None, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        self._episode, server = extras["episode"], extras["server"]
        self.max_turns = int(extras.get("max_turns") or 1)
        self._tools = {tool["function"]["name"]: tool["function"] for tool in json.loads(self._episode["tools"])}
        self._format = server["tool_format"]
        self._waits = server["reset_timeout"] + max(server["reset_timeout"], server["tool_timeout"]) \
            + 2 * GRACE_SECONDS + CLOSE_SECONDS
        self._session = _Session(Shared(server["url"], frozenset(self._tools), server["tool_timeout"],
                                        server["reset_timeout"], patience=server["reset_timeout"],
                                        waits_for_server=True, token=token_in(server.get("token_var"))))
        self._scores: dict[str, float] = {}
        self._run = self._unparsed = 0

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        self._wait(self._session.start(self._episode), self._waits)
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        calls, reply = parse(action, self._format, self._tools)
        answers = []
        for call in calls:
            if isinstance(call, Unparsed):
                self._unparsed += 1
                answers.append({"role": "tool", "content": f"Error: {call.why}"})
                continue
            self._run += 1
            answers.append({"role": "tool", "content": self._wait(self._session.call(call), self._waits)})
            if self._session.rollout.done or self._session.rollout.failure:
                break
        rollout = self._session.rollout
        if not (rollout.done or rollout.failure) and (not calls or self.turns >= self.max_turns):
            self._wait(self._session.finish(reply), None)
        if rollout.failure is not None:
            self.close()
            if rollout.misgrouped:
                raise ValueError(f"{self._task()}: {rollout.failure}. Its row lists the tools of its toolset's first "
                                 "task, so a task must offer no tool that one doesn't")
            raise EpisodeFailed(f"{self._task()}: {rollout.failure}")
        if not rollout.done:
            return BaseTextEnvStepOutput(observations=answers, reward=0.0, done=False, metadata={})
        if rollout.reward is None:
            self.close()
            raise ValueError(f"{self._task()} ended without a reward: it records no score its server's reward reads, "
                             "so it can't be trained on")
        self._scores = rollout.scores or {}
        return BaseTextEnvStepOutput(observations=[], reward=float(rollout.reward), done=True, metadata={})

    def get_metrics(self) -> dict[str, Any]:
        return {"tool_calls": self._run, "unparsed_calls": self._unparsed,
                **{f"score/{step}": score for step, score in self._scores.items()}}

    @staticmethod
    def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Each metric's mean over the rollouts, named ``avg_<metric>`` as SkyRL names a mean: where SkyRL joins
        groups' metrics, as its fully async trainer does, it averages a mean so named and adds the others up."""
        return {f"avg_{key}": value for key, value in default_aggregate_metrics(metrics).items()}

    def close(self) -> None:
        on_session_loop(self._session.close(), CLOSE_SECONDS)

    def _wait(self, coroutine: Any, timeout: float | None) -> Any:
        """The session's answer within ``timeout``, past which the episode fails: a call's deadlines and a grace, or
        none for ending the turns left open, each of which has its own deadline."""
        try:
            return on_session_loop(coroutine, timeout)
        except TimeoutError as e:
            self.close()
            raise EpisodeFailed(f"{self._task()}: the env didn't answer within {timeout:g}s") from e

    def _task(self) -> str:
        return f"task {self._episode['task_id']} of split {self._episode['split']}"


class _Session(Session):
    """One rollout's session: it starts the row's episode, runs the model's tool calls on it, and ends the turn left
    open with the model's reply."""

    offered_by = "its row lists"

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared

    @property
    def rollout(self) -> Rollout:
        return self._rollout

    async def start(self, episode: dict[str, Any]) -> None:
        await self._start(episode["split"], int(episode["index"]), episode["task_id"])

    async def call(self, call: Call) -> str:
        return await self._step(call.name, call.arguments)

    async def finish(self, reply: str) -> None:
        await self._finish(self._rollout, reply)

    async def close(self) -> None:
        await self._close()
