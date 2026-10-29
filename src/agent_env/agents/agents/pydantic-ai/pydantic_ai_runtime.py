"""The Pydantic AI side of the agent: a task is one run of a Pydantic AI agent with the Pydantic AI Harness ``Coder``
tools in the workspace, the task's MCP servers and its skills, continuing the context's message history."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agentenv_protocol.a2a_agent import Usage
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelMessagesTypeAdapter, UserContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import Skills
from pydantic_ai_harness.coder import Coder

from pydantic_ai_agent import MESSAGES, Outcome, Turn, to_completion

# Consecutive retries a tool gets for arguments it refuses, as many as Coder gives its file tools, so a model that
# fumbles a command twice doesn't end the task.
_TOOL_RETRIES: Final[int] = 5
# Without a key the OpenAI client refuses to start; an endpoint that needs none ignores it.
_NO_KEY: Final[str] = "unset"
# The Pydantic AI providers a model name's prefix picks for the endpoint, as in ``vllm:qwen3.8-27b``: each speaks
# OpenAI Chat Completions, and ``vllm`` fits its requests to the model's family, as a vLLM server needs.
_PROVIDERS: Final = {"openai": OpenAIProvider, "vllm": VLLMProvider}


def _model(turn: Turn) -> OpenAIChatModel:
    prefix, _, name = turn.config.model.partition(":")
    provider = _PROVIDERS.get(prefix) if name else None
    if provider is None:
        provider, name = OpenAIProvider, turn.config.model
    return OpenAIChatModel(name, provider=provider(base_url=turn.endpoint.base_url,
                                                   api_key=turn.endpoint.api_key or _NO_KEY))


def _settings(turn: Turn) -> ModelSettings:
    # Sent on every call and kept out of the messages, which a snapshot uploads.
    settings = ModelSettings(extra_headers=dict(turn.endpoint.extra_headers))
    if turn.config.effort is not None:
        settings["thinking"] = turn.config.effort
    return settings


def _agent(turn: Turn) -> Agent:
    # The agent runs in a sandbox of its own, so its file tools reach the files attached outside the workspace.
    capabilities: list[Any] = [LocalWorkspace(turn.workspace), Coder(unrestricted_filesystem=True)]
    if turn.skill_names:
        capabilities.append(Skills(turn.skills_dir, include=turn.skill_names))
    toolsets = [
        MCPToolset(registration["url"], id=name, headers=dict(registration.get("headers") or {}))
        for name, registration in turn.mcp_servers.items()
    ]
    return Agent(
        _model(turn),
        instructions=turn.config.system_prompt,
        model_settings=_settings(turn),
        retries={"tools": _TOOL_RETRIES},
        capabilities=capabilities,
        toolsets=toolsets,
    )


def _load(turn: Turn) -> list[ModelMessage]:
    path = turn.conversation_dir / MESSAGES
    return ModelMessagesTypeAdapter.validate_json(path.read_bytes()) if path.is_file() else []


def _save(turn: Turn, messages: Sequence[ModelMessage]) -> None:
    """The conversation's messages, written aside and moved into place, so a snapshot never reads half of them."""
    turn.conversation_dir.mkdir(parents=True, exist_ok=True)
    staged = turn.conversation_dir / f".{MESSAGES}.tmp"
    staged.write_bytes(ModelMessagesTypeAdapter.dump_json(list(messages)))
    staged.replace(turn.conversation_dir / MESSAGES)


def _events(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")


def _usage(usage: RunUsage) -> Usage:
    return Usage(
        tool_call_count=usage.tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        provider_details={
            "requests": usage.requests,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
        },
    )


def _failure(exc: Exception, usage: Usage) -> Outcome:
    """A run that raised, typed by whose failure it was: the model endpoint's, or the agent's own."""
    if isinstance(exc, UsageLimitExceeded):
        return Outcome(f"Pydantic AI stopped: {exc}", usage, "pydantic_ai.usage_limit")
    if isinstance(exc, ModelHTTPError):
        return Outcome(f"The model endpoint answered {exc.status_code}: {exc.body}", usage,
                       f"pydantic_ai.model_http_{exc.status_code}", "infra_error")
    if isinstance(exc, ModelAPIError):
        return Outcome(f"The model endpoint failed: {exc}", usage, "pydantic_ai.model_api", "infra_error")
    if isinstance(exc, UnexpectedModelBehavior):
        return Outcome(f"The model misbehaved: {exc}", usage, "pydantic_ai.unexpected_model_behavior")
    # An MCP server or a tool failing to start, for one.
    return Outcome(f"Pydantic AI failed: {type(exc).__name__}: {exc}", usage, "pydantic_ai.error", "infra_error")


@dataclass(slots=True)
class PydanticAIRuntime:
    _runs: set[asyncio.Task[Any]] = field(default_factory=set, init=False, repr=False)

    async def stop(self) -> None:
        runs = list(self._runs)
        for run in runs:
            run.cancel()
        await asyncio.gather(*runs, return_exceptions=True)

    def check_messages(self, data: bytes) -> None:
        ModelMessagesTypeAdapter.validate_json(data)

    async def run(self, turn: Turn) -> Outcome:
        task = asyncio.current_task()
        self._runs.add(task)
        try:
            return await self._run(turn)
        finally:
            self._runs.discard(task)

    async def _run(self, turn: Turn) -> Outcome:
        try:
            history = await asyncio.to_thread(_load, turn)
        except (OSError, ValueError) as exc:
            return Outcome(f"Pydantic AI could not read the conversation: {exc}", Usage(), "pydantic_ai.history",
                           "infra_error")
        prompt: list[UserContent] = [turn.text]
        for image in turn.images:
            prompt.append(BinaryContent(data=await asyncio.to_thread(image.path.read_bytes), media_type=image.media_type))
        agent = _agent(turn)
        recorded = len(history)
        usage = RunUsage()
        try:
            async with agent.iter(prompt, message_history=history,
                                  usage_limits=UsageLimits(request_limit=turn.config.request_limit)) as run:
                async for _node in run:
                    messages = run.all_messages()
                    for event in _events(messages[recorded:]):
                        turn.on_event(event)
                    recorded = len(messages)
                    usage = run.usage
            result = run.result
        except asyncio.CancelledError:
            turn.on_usage(_usage(usage))
            raise
        except Exception as exc:  # every failure is the task's, typed by whose it was
            return _failure(exc, _usage(usage))
        messages = result.all_messages()
        for event in _events(messages[recorded:]):
            turn.on_event(event)
        # A turn whose run finished is saved whole even if it is cancelled meanwhile, before its locks are let go.
        await to_completion(_save, turn, messages)
        return Outcome(result.output.strip() or "The agent finished without a reply.", _usage(result.usage))
