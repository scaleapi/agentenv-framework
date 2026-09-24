"""Custom LLM agent with lifespan and common SDK-operation overrides."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from agentenv_protocol.a2a_agent import (
    AGENT_CONFIG_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    TriggerDecideRequest,
    TriggerRegisterRequest,
    a2a_agent,
    create_app,
    extension,
)
from starlette.applications import Starlette

try:
    from .reference_model import OpenAICompatibleModel
except ImportError:  # Support ``python examples/advanced_agent.py``.
    from reference_model import OpenAICompatibleModel


class AdvancedConfig(AgentConfig):
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a concise, helpful assistant."
    response_prefix: str = ""


@asynccontextmanager
async def runtime_lifespan(app: Starlette) -> AsyncIterator[None]:
    """Initialize and close runtime resources around the ASGI application."""

    app.state.reference_runtime = {"ready": True}
    try:
        yield
    finally:
        app.state.reference_runtime["ready"] = False


@a2a_agent(
    identity=AgentIdentity(
        name="reference-advanced-agent",
        description="Shows ASGI lifespan and common SDK-operation overrides.",
        version="1.0.0",
    ),
    config=AdvancedConfig,
    lifespan=runtime_lifespan,
)
class AdvancedAgent(AgentEnvAgent):
    def __init__(self, model_client: Any | None = None) -> None:
        self.model_client = model_client or OpenAICompatibleModel()

    async def run(self, request: TaskRequest[AdvancedConfig]) -> TaskResult:
        text = "\n".join(
            part.text for part in request.parts if isinstance(part, TextPart)
        )
        completion = await self.model_client.complete(
            model=request.config.model,
            messages=[
                {"role": "system", "content": request.config.system_prompt},
                {"role": "user", "content": text},
            ],
            metadata=request.metadata,
        )
        response = completion.text
        if request.config.response_prefix:
            response = f"{request.config.response_prefix}: {response}"
        return TaskResult.text(response)

    # SDK-owned operation sets are overridden atomically, even when most
    # operations simply retain their default implementation.
    @extension(AGENT_CONFIG_V1.set)
    async def set_config(self, request: AdvancedConfig) -> dict[str, Any]:
        return await self.default_handlers.call(AGENT_CONFIG_V1.set, request)

    @extension(AGENT_CONFIG_V1.get)
    async def get_config(self) -> dict[str, Any]:
        return await self.default_handlers.call(AGENT_CONFIG_V1.get)

    @extension(TRIGGERS_V1.register)
    async def register_triggers(
        self, request: TriggerRegisterRequest
    ) -> dict[str, Any]:
        return await self.default_handlers.call(TRIGGERS_V1.register, request)

    @extension(TRIGGERS_V1.decide)
    async def decide_trigger(self, request: TriggerDecideRequest) -> dict[str, Any]:
        decision = await self.default_handlers.call(TRIGGERS_V1.decide, request)
        reached_limit = request.turn >= 3
        if reached_limit:
            decision["done"] = True
            decision["fired"] = [*decision["fired"], "reference-turn-limit"]
        return decision

    @extension(TRIGGERS_V1.state)
    async def trigger_state(self) -> dict[str, Any]:
        return await self.default_handlers.call(TRIGGERS_V1.state)


agent = AdvancedAgent()
app = create_app(agent)


if __name__ == "__main__":
    agent.serve()
