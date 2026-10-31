"""verl's agent loop for the rows ``python -m agent_env.training.verl`` writes. ``AgentEnvAgentLoop`` is verl's
``ToolAgentLoop``, which generates on verl's rollout server, parses the model's tool calls with
``multi_turn.format``'s parser and adds their answers to the context, with the row's tools in place of verl's own: each
call is a call to the tool on the row's episode, the model's last reply ends the turn it left open, and the episode's
reward is the rollout's ``reward_score``, so verl's reward function isn't called. Verified with verl 0.9.1."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] != "verl":
        raise
    raise ImportError("agent_env.training.verl.loop runs in verl's agent loop workers: install "
                      "'agentenv-framework[verl]' into verl's environment") from exc

from agent_env.training._openenv import Rollout, Session, Shared

logger = logging.getLogger(__name__)


class _Session(Session):
    offered_by = "its row lists"

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared


class AgentEnvAgentLoop(ToolAgentLoop):
    """Plays a row's task as one episode on the OpenEnv server at ``url``. A rollout the env fails, a deploy, a call
    past its deadline or a server that went away, is played again from the start on a new episode, a fresh sample of
    the same policy, up to ``attempts`` times; one that fails every time is kept out of the loss, its response masked,
    with the reward 0. ``agent_loop.yaml`` sets the arguments; verl passes the rest.

    Args:
        url: The OpenEnv server's base URL.
        tool_timeout: One tool call's deadline in seconds. The server ends a call that misses it, and the episode.
        reset_timeout: How long an episode may take to deploy, and each turn to be ended, in seconds; also how long
            a rollout waits for room on a server at its limit.
        attempts: How many times a rollout the env fails is played.
    """

    def __init__(self, *args: Any, url: str, tool_timeout: float = 60.0, reset_timeout: float = 600.0,
                 attempts: int = 3, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._url, self._attempts = url.rstrip("/"), attempts
        self._tool_timeout, self._reset_timeout = tool_timeout, reset_timeout
        self._reply = ""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        row = kwargs["extra_info"]
        schemas = json.loads(row["tools"])
        shared = Shared(self._url, frozenset(schema["function"]["name"] for schema in schemas), self._tool_timeout,
                        self._reset_timeout, patience=self._reset_timeout)
        for attempt in range(1, self._attempts + 1):
            session = _Session(shared)
            try:
                output, rollout = await self._play(session, schemas, row, sampling_params, kwargs)
            finally:
                await session._close()
            if rollout.failure is None and rollout.reward is not None:
                output.reward_score = float(rollout.reward)
                output.extra_fields["agentenv"] = {"attempts": attempt, "scores": rollout.scores}
                return output
            failure = rollout.failure or "the episode ended without a reward"
            logger.warning("task %s of split %s, attempt %d of %d: %s", row["task_id"], row["split"], attempt,
                           self._attempts, failure)
        output.response_mask = [0] * len(output.response_mask)
        output.reward_score = 0.0
        output.extra_fields["agentenv"] = {"attempts": self._attempts, "failure": failure}
        return output

    async def _play(self, session: _Session, schemas: list[dict[str, Any]], row: dict[str, Any],
                    sampling_params: dict[str, Any], kwargs: dict[str, Any]) -> tuple[AgentLoopOutput, Rollout]:
        await session._start(row["split"], int(row["index"]), row["task_id"])
        self.tools = {schema["function"]["name"]: _tool(session, schema) for schema in schemas}
        self.tool_schemas = schemas
        self._reply = ""
        output = await super().run(sampling_params, **kwargs)
        rollout = session._rollout
        await session._finish(rollout, self._reply)
        return output, rollout

    def _build_assistant_message(self, content: str, agent_data: Any) -> dict[str, Any]:
        """Keeps the turn's text, without the special tokens the parser leaves in it, as the reply to end the turn
        with."""
        reply = content or ""
        for token in self.tokenizer.all_special_tokens:
            reply = reply.replace(token, "")
        self._reply = reply.strip()
        return super()._build_assistant_message(content, agent_data)


def _tool(session: _Session, schema: dict[str, Any]) -> FunctionTool:
    """The episode's tool as the function tool ``ToolAgentLoop`` calls; its schema is built unvalidated, since verl's
    schema models keep no nested JSON Schema, and the chat template renders the row's own."""
    name = schema["function"]["name"]

    async def call(**arguments: Any) -> str:
        return await session._step(name, arguments)

    return FunctionTool(name=name, fn=call, tool_schema=OpenAIFunctionToolSchema.model_construct(**schema),
                        is_async=True)
