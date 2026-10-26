"""The OpenHands SDK side of the agent: a task is one ``arun()`` of the context's persisted conversation, with the
default tools, the task's MCP servers and its skills."""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from agentenv_protocol.a2a_agent import Usage
from openhands.sdk import LLM, Agent, AgentContext, Conversation, ImageContent, Message, TextContent
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, Event
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.event.error_classification import FailureKind
from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.mcp import MCPServer
from openhands.sdk.skills.skill import load_skills_from_dir
from openhands.tools.preset.default import get_default_condenser, get_default_tools
from pydantic import PrivateAttr

from conversation_run import ConversationRun
from openhands_agent import Outcome, Turn

# The AgentEnv endpoint is a LiteLLM gateway; this provider prefix sends LiteLLM's requests to it as is.
_PROVIDER: Final[str] = "litellm_proxy"


class _GatewayLLM(LLM):
    """The SDK's LLM with the gateway's headers sent on every call but kept out of the conversation's persisted state,
    which a snapshot uploads: the SDK saves the agent's LLM there and masks only its key fields. The conversation's
    stats don't count it, as they count only LLMs of exactly the SDK's class, so a turn's spend is read from its own."""

    _gateway_headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def _get_litellm_auth_values(self) -> tuple[str | None, dict[str, str]]:
        api_key, headers = super()._get_litellm_auth_values()
        return api_key, {**self._gateway_headers, **headers}


def _llm(turn: Turn) -> LLM:
    options: dict[str, Any] = {}
    if turn.config.effort is not None:
        options["reasoning_effort"] = turn.config.effort
    llm = _GatewayLLM(
        usage_id="agent",
        model=f"{_PROVIDER}/{turn.config.model}",
        base_url=turn.endpoint.base_url,
        api_key=turn.endpoint.api_key,
        **options,
    )
    llm._gateway_headers = dict(turn.endpoint.extra_headers)
    return llm


def _agent(turn: Turn) -> Agent:
    skills = []
    if turn.skill_names and turn.skills_dir.is_dir():
        loaded = load_skills_from_dir(turn.skills_dir, strict=False)
        skills = [skill for group in loaded for skill in group.values() if skill.name in turn.skill_names]
    mcp_config = {
        # Literal, so a ${VAR} in a URL or header is sent as written, not filled from the agent's environment.
        name: MCPServer(url=registration["url"], transport="http", headers=registration.get("headers") or None)
        .as_literal()
        for name, registration in turn.mcp_servers.items()
    }
    options: dict[str, Any] = {}
    if turn.config.system_prompt:
        options["system_prompt"] = turn.config.system_prompt
    llm = _llm(turn)
    return Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        mcp_config=mcp_config,
        # The SDK's default agent condenses a conversation that outgrows its window; without one, a context's
        # conversation, which every task adds to, fails every turn once it does.
        condenser=get_default_condenser(llm=llm.model_copy(update={"usage_id": "condenser"})),
        agent_context=AgentContext(
            skills=skills,
            system_message_suffix=turn.config.append_system_prompt,
            load_user_skills=False,
            load_public_skills=False,
        ),
        **options,
    )


def _usage(agent: Agent, tool_calls: int) -> Usage:
    """What the turn's LLMs have spent: they are made for the turn, so they start at nothing."""
    llms = (agent.llm, getattr(agent.condenser, "llm", None))
    metrics = Metrics()
    for each in {id(llm.metrics): llm.metrics for llm in llms if llm is not None}.values():
        metrics.merge(each)

    def spent(name: str) -> int:
        return getattr(metrics.accumulated_token_usage, name, 0) or 0

    input_tokens, output_tokens = spent("prompt_tokens"), spent("completion_tokens")
    cost = metrics.accumulated_cost
    return Usage(
        tool_call_count=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost or None,
        provider_details={
            "cache_read_tokens": spent("cache_read_tokens"),
            "cache_write_tokens": spent("cache_write_tokens"),
            "reasoning_tokens": spent("reasoning_tokens"),
        },
    )


def _unstarted(exc: Exception) -> Outcome:
    return Outcome(f"OpenHands could not start the turn: {type(exc).__name__}: {exc}", Usage(), "openhands.start",
                   "infra_error")


def _failure(
    error: ConversationErrorEvent | None,
    fallback: str,
    usage: Usage,
    error_type: Literal["agent_error", "infra_error"],
) -> Outcome:
    """The run's last error event as a failure, typed by its classification when it has one, else ``error_type``."""
    if error is None:
        return Outcome(f"OpenHands failed: {fallback}", usage, "openhands.error", error_type)
    if error.classification is not None:
        error_type = "agent_error" if error.classification.kind == FailureKind.AGENT_ACTION else "infra_error"
    return Outcome(f"OpenHands failed: {error.detail}", usage, f"openhands.{error.code}", error_type)


@dataclass(frozen=True, slots=True)
class SdkRuntime:
    _runs: set[ConversationRun] = field(default_factory=set, init=False, repr=False)

    def stop(self) -> None:
        for run in list(self._runs):
            run.stop()

    async def run(self, turn: Turn) -> Outcome:
        actions = 0
        # The SDK reports why a run failed only as events, whether or not it raises.
        errors: list[ConversationErrorEvent] = []

        def on_event(event: Event) -> None:
            nonlocal actions
            if isinstance(event, ActionEvent):
                actions += 1
            elif isinstance(event, ConversationErrorEvent):
                errors.append(event)
            turn.on_event(event.model_dump(mode="json"))

        agent = _agent(turn)

        def open_conversation() -> BaseConversation:
            return Conversation(
                agent=agent,
                workspace=str(turn.workspace),
                persistence_dir=str(turn.conversations_dir),
                conversation_id=uuid.UUID(turn.conversation_id),
                callbacks=[on_event],
                max_iteration_per_run=turn.config.max_iterations,
                visualizer=None,
                delete_on_close=False,
            )

        try:
            conversation = await asyncio.to_thread(open_conversation)
        except Exception as exc:  # a persisted conversation the SDK can't resume, for one
            return _unstarted(exc)
        # Listed before it starts, so the server's shutdown stops it however far it has got.
        run = ConversationRun(conversation)
        self._runs.add(run)
        try:
            content = [TextContent(text=turn.text), *(ImageContent(image_urls=[image]) for image in turn.images)]
            try:
                await asyncio.to_thread(conversation.send_message, Message(role="user", content=content))
            except Exception as exc:  # the agent's tools or MCP servers failing to start, for one
                return _unstarted(exc)
            try:
                await run.to_its_end()
            except ConversationRunError as exc:
                usage = _usage(agent, actions)
                error = exc.conversation_error or (errors[-1] if errors else None)
                return _failure(error, str(exc.original_exception), usage, "infra_error")
            except asyncio.CancelledError:
                turn.on_usage(_usage(agent, actions))
                raise
            usage = _usage(agent, actions)
            status = conversation.state.execution_status
            # Only the server's shutdown stops a run whose caller wasn't cancelled.
            if run.stopped.is_set() and status != ConversationExecutionStatus.FINISHED:
                return Outcome("OpenHands was stopped: the agent is shutting down", usage, "openhands.shutdown",
                               "infra_error")
            if status == ConversationExecutionStatus.ERROR:
                # A run limit, such as the iteration cap, ends the run without raising: the agent's own failure.
                return _failure(errors[-1] if errors else None, "the run ended in error", usage, "agent_error")
            if status != ConversationExecutionStatus.FINISHED:
                return Outcome(f"OpenHands stopped with status '{status.value}'", usage, f"openhands.{status.value}")
            reply = get_agent_final_response(conversation.state.events)
            return Outcome(reply or "The agent finished without a reply.", usage)
        finally:
            self._runs.discard(run)
            await asyncio.to_thread(conversation.close)
