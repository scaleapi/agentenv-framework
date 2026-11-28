"""The OpenHands SDK side of the agent: a task is one ``arun()`` of the context's persisted conversation, with the
default tools, the task's MCP servers and its skills."""

import asyncio
import uuid
from collections.abc import Callable
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
from openhands.sdk.mcp import MCPServer
from openhands.sdk.skills.skill import load_skills_from_dir
from openhands.tools.preset.default import get_default_condenser, get_default_tools
from pydantic import PrivateAttr

from conversation_run import ConversationRun
from openhands_agent import CallFeed, Outcome, Turn, turn_usage

# The AgentEnv endpoint is a LiteLLM gateway; this provider prefix sends LiteLLM's requests to it as is.
_PROVIDER: Final[str] = "litellm_proxy"
_Retried = Callable[[int, int, BaseException | None], None]


class _GatewayLLM(LLM):
    """The SDK's LLM with the gateway's headers sent on every call but kept out of the conversation's persisted state,
    which a snapshot uploads: the SDK saves the agent's LLM there and masks only its key fields. The conversation's
    stats don't count it, as they count only LLMs of exactly the SDK's class, so a turn's spend is read from its own."""

    _gateway_headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def _get_litellm_auth_values(self) -> tuple[str | None, dict[str, str]]:
        api_key, headers = super()._get_litellm_auth_values()
        return api_key, {**self._gateway_headers, **headers}


def _llm(turn: Turn, retried: _Retried | None) -> LLM:
    options: dict[str, Any] = {}
    if turn.config.effort is not None:
        options["reasoning_effort"] = turn.config.effort
    if retried is not None:
        options["retry_listener"] = retried
    llm = _GatewayLLM(
        usage_id="agent",
        model=f"{_PROVIDER}/{turn.config.model}",
        base_url=turn.endpoint.base_url,
        api_key=turn.endpoint.api_key,
        **options,
    )
    llm._gateway_headers = dict(turn.endpoint.extra_headers)
    return llm


def _agent(turn: Turn, retried: _Retried | None) -> Agent:
    """The turn's agent; ``retried`` hears of each model call its LLMs, the condenser's copy too, fail and make
    again."""
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
    llm = _llm(turn, retried)
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


def _metered(agent: Agent) -> list[LLM]:
    """The turn's LLMs, each metrics object once: the condenser's is a copy of the agent's, which may share it."""
    return list({id(llm.metrics): llm for llm in (agent.llm, getattr(agent.condenser, "llm", None))
                 if llm is not None}.values())


def _usage(agent: Agent, tool_calls: int) -> Usage:
    """What the turn's LLMs have spent, call by call: they are made for the turn, so they start at nothing. Each
    response they got has a latency recorded, with or without usage."""
    llms = _metered(agent)
    calls = [(usage, llm.uses_responses_api()) for llm in llms for usage in llm.metrics.token_usages]
    responses = sum(len(llm.metrics.response_latencies) for llm in llms)
    cost = sum(llm.metrics.accumulated_cost for llm in llms)
    return turn_usage(calls, responses=responses, tool_calls=tool_calls, cost=cost or None)


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
        feed = CallFeed(turn.on_event, turn.on_aux, model=turn.config.model)

        def on_event(event: Event) -> None:
            nonlocal actions
            if isinstance(event, ActionEvent):
                actions += 1
            elif isinstance(event, ConversationErrorEvent):
                errors.append(event)
            feed.event(event.model_dump(mode="json"))

        agent = _agent(turn, feed.retried if feed.builds else None)
        feed.llms = _metered(agent)
        feed.clock()

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
            if feed.builds:
                # As the agent offers them to its model, which the system prompt's event doesn't say.
                feed.tools([tool.to_openai_tool(add_security_risk_prediction=True)
                            for tool in conversation.agent.tools_map.values()])
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
            # A response no event followed, as one the run was cut off after.
            feed.end()
            await asyncio.to_thread(conversation.close)
