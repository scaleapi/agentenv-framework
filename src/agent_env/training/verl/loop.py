"""verl's agent loop for the rows ``python -m agent_env.training.verl`` writes. ``AgentEnvAgentLoop`` is verl's
``ToolAgentLoop``, which generates on verl's rollout server, parses the model's tool calls with
``multi_turn.format``'s parser and adds their answers to the context, with the row's tools in place of verl's own: each
call is a call to the tool on the row's episode, the model's last reply ends the turn it left open, and the episode's
reward is the rollout's ``reward_score``, so verl's reward function isn't called. CI checks it on verl 0.9.1 and
main."""

from __future__ import annotations

import json
import logging
import time
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


class AgentEnvAgentLoop(ToolAgentLoop):
    """Plays a row's task as one episode on the OpenEnv server at ``url``. A rollout the env fails, a deploy, a call
    past its deadline or a server that went away, is played again from the start on a new episode, a fresh sample of
    the same policy, up to ``attempts`` times, but not once ``reset_timeout`` seconds have passed since it began; one
    that fails every time is kept out of the loss, its response masked, with the reward 0. A task that ends unscored
    without failing, and a prompt that doesn't fit ``rollout.prompt_length`` with its tools, are configuration errors no
    replay changes, and raise. ``agent_loop.yaml`` sets the arguments; verl passes the rest.

    It leans on verl internals, which CI checks on verl 0.9.1 and main: verl builds an agent loop per trajectory, so an
    instance holds one rollout's state, and ``run`` refuses a second; ``ToolAgentLoop.run`` reads the tools from
    ``self.tools`` and ``self.tool_schemas``; ``_build_assistant_message`` is overridden to keep the reply, and
    ``AgentLoopBase._cap_text_prompt_length`` to refuse a prompt verl would cut; and the tools are verl's
    ``FunctionTool`` with ``OpenAIFunctionToolSchema``.

    Args:
        url: The OpenEnv server's base URL.
        tool_timeout: One tool call's deadline in seconds. The server ends a call that misses it, and the episode.
        reset_timeout: How long an episode may take to deploy, and each turn to be ended, in seconds; also how long
            a rollout waits for room on a server at its limit, and how long after it began it may be played again.
        attempts: How many times a rollout the env fails is played at most.
    """

    def __init__(self, *args: Any, url: str, tool_timeout: float = 60.0, reset_timeout: float = 600.0,
                 attempts: int = 3, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._url, self._attempts = url.rstrip("/"), attempts
        self._tool_timeout, self._reset_timeout = tool_timeout, reset_timeout
        self._row: dict[str, Any] = {}
        self._reply = ""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        if self._row:
            raise RuntimeError("verl ran an AgentEnvAgentLoop a second time, but one holds a single rollout's state: "
                               "this verl doesn't build an agent loop per trajectory, as 0.9 does")
        row = self._row = kwargs["extra_info"]
        schemas = json.loads(row["tools"])
        shared = Shared(self._url, frozenset(schema["function"]["name"] for schema in schemas), self._tool_timeout,
                        self._reset_timeout, patience=self._reset_timeout)
        began = time.monotonic()
        for attempt in range(1, self._attempts + 1):
            session = _Session(shared)
            try:
                output, rollout = await self._play(session, schemas, sampling_params, kwargs)
            finally:
                await session.close()
            if rollout.failure is None:
                if rollout.reward is None:
                    raise ValueError(f"{self._task()} ended without a reward: it records no score its server's reward "
                                     "reads, so it can't be trained on")
                output.reward_score = float(rollout.reward)
                output.extra_fields["agentenv"] = {"attempts": attempt, "scores": rollout.scores}
                return output
            logger.warning("%s, attempt %d of %d: %s", self._task(), attempt, self._attempts, rollout.failure)
            if time.monotonic() - began >= self._reset_timeout:
                break
        output.response_mask = [0] * len(output.response_mask)
        output.reward_score = 0.0
        output.extra_fields["agentenv"] = {"attempts": attempt, "failure": rollout.failure}
        return output

    async def _play(self, session: _Session, schemas: list[dict[str, Any]], sampling_params: dict[str, Any],
                    kwargs: dict[str, Any]) -> tuple[AgentLoopOutput, Rollout]:
        await session.start(self._row)
        self.tools = {schema["function"]["name"]: session.tool(schema) for schema in schemas}
        self.tool_schemas = schemas
        self._reply = ""
        output = await super().run(sampling_params, **kwargs)
        return output, await session.finish(self._reply)

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        """Refuses the first prompt when it doesn't fit, where verl would cut it from the left, losing the system
        prompt and the tools; verl's own length filter doesn't count a row's tools."""
        if len(prompt_ids) > (limit := self.rollout_config.prompt_length):
            raise ValueError(f"{self._task()}'s prompt is {len(prompt_ids)} tokens with its tools, over "
                             f"rollout.prompt_length={limit}: raise data.max_prompt_length to fit it")
        return prompt_ids

    def _build_assistant_message(self, content: str, agent_data: Any) -> dict[str, Any]:
        """Keeps the turn's text, without the special tokens the parser leaves in it, as the reply to end the turn
        with."""
        reply = content or ""
        for token in self.tokenizer.all_special_tokens:
            reply = reply.replace(token, "")
        self._reply = reply.strip()
        return super()._build_assistant_message(content, agent_data)

    def _task(self) -> str:
        return f"task {self._row['task_id']} of split {self._row['split']}"


class _Session(Session):
    """One rollout's session: it starts the row's episode, offers the row's tools as verl's function tools, and ends
    the turn left open with the model's reply."""

    offered_by = "its row lists"

    def __init__(self, shared: Shared):
        super().__init__()
        self._shared = shared

    async def start(self, row: dict[str, Any]) -> None:
        await self._start(row["split"], int(row["index"]), row["task_id"])

    def tool(self, schema: dict[str, Any]) -> FunctionTool:
        name = schema["function"]["name"]

        async def call(**arguments: Any) -> str:
            return await self._step(name, arguments)

        return FunctionTool(name=name, fn=call, tool_schema=_parser_schema(schema), is_async=True)

    async def finish(self, reply: str) -> Rollout:
        rollout = self._rollout
        await self._finish(rollout, reply)
        return rollout

    async def close(self) -> None:
        await self._close()


def _parser_schema(schema: dict[str, Any]) -> OpenAIFunctionToolSchema:
    """The tool as verl's parsers read it: those that convert arguments (``qwen3_coder``) or match a call to a tool by
    its parameters (``deepseek_v4``) take each parameter's type, description and enum. verl's schema models keep no
    nested JSON Schema, so the chat template renders the row's own, and a parameter without a type is left out, which
    a parser takes as a string."""
    function = schema["function"]
    parameters = function.get("parameters") or {}
    return OpenAIFunctionToolSchema.model_validate({"type": "function", "function": {
        "name": function["name"], "description": function.get("description") or "",
        "parameters": {"type": "object", "required": parameters.get("required") or [], "properties": {
            argument: {key: spec[key] for key in ("type", "description", "enum") if key in spec}
            for argument, spec in (parameters.get("properties") or {}).items() if "type" in spec}},
    }})
