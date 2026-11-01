"""SkyRL's env for the rows ``python -m agent_env.training.skyrl`` writes, playing each row's task as one episode on the
bundle's OpenEnv server. SkyRL calls envs from threads, so sessions run on the event loop this process's sessions share."""

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


class AgentEnvTextEnv(BaseTextEnv):
    """An episode the env fails ends with reward 0 and its ``failed`` metric set, for AgentEnvGenerator to play again,
    so SkyRL builds every trajectory itself; a task that can't be scored at all raises."""

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
        self._failed = False

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        try:
            self._wait(self._session.start(self._episode), self._waits)
        except _Failed as e:
            self._fail(e)
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        if self._failed:
            return _ended(0.0)
        try:
            return self._stepped(action)
        except _Failed as e:
            self._fail(e)
            return _ended(0.0)

    def get_metrics(self) -> dict[str, Any]:
        return {"tool_calls": self._run, "unparsed_calls": self._unparsed, "failed": int(self._failed),
                **{f"score/{step}": score for step, score in self._scores.items()}}

    @staticmethod
    def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Named ``avg_<metric>`` as SkyRL names a mean, so where it joins groups' metrics, as its fully async trainer
        does, it averages them rather than adding them up."""
        return {f"avg_{key}": value for key, value in default_aggregate_metrics(metrics).items()}

    def close(self) -> None:
        on_session_loop(self._session.close(), CLOSE_SECONDS)

    def _stepped(self, action: str) -> BaseTextEnvStepOutput:
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
            if rollout.misgrouped:
                self.close()
                raise ValueError(f"{self._task()}: {rollout.failure}. Its row lists the tools of its toolset's first "
                                 "task, so a task must offer no tool that one doesn't")
            raise _Failed(rollout.failure)
        if not rollout.done:
            return BaseTextEnvStepOutput(observations=answers, reward=0.0, done=False, metadata={})
        if rollout.reward is None:
            self.close()
            raise ValueError(f"{self._task()} ended without a reward: it records no score its server's reward reads, "
                             "so it can't be trained on")
        self._scores = rollout.scores or {}
        return _ended(float(rollout.reward))

    def _fail(self, failure: _Failed) -> None:
        self._failed = True
        logger.warning("%s: %s", self._task(), failure)
        self.close()

    def _wait(self, coroutine: Any, timeout: float | None) -> Any:
        try:
            return on_session_loop(coroutine, timeout)
        except TimeoutError as e:
            raise _Failed(f"the env didn't answer within {timeout:g}s") from e

    def _task(self) -> str:
        return f"task {self._episode['task_id']} of split {self._episode['split']}"


class _Failed(Exception):
    pass


def _ended(reward: float) -> BaseTextEnvStepOutput:
    return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})


class _Session(Session):
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
