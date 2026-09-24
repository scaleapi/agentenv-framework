"""Streaming custom LLM agent using progress and terminal result helpers."""

from collections.abc import AsyncIterator
from typing import Any

from agentenv_protocol.a2a_agent import (
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStreamItem,
    TextPart,
    a2a_agent,
    create_app,
)

try:
    from .reference_model import OpenAICompatibleModel
except ImportError:  # Support ``python examples/streaming_agent.py``.
    from reference_model import OpenAICompatibleModel


class StreamingConfig(AgentConfig):
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a concise, helpful assistant."


@a2a_agent(
    identity=AgentIdentity(
        name="reference-streaming-agent",
        description="Demonstrates request-scoped A2A streaming.",
        version="1.0.0",
    ),
    config=StreamingConfig,
)
class StreamingAgent(AgentEnvAgent):
    def __init__(self, model_client: Any | None = None) -> None:
        self.model_client = model_client or OpenAICompatibleModel()

    async def run(
        self, request: TaskRequest[StreamingConfig]
    ) -> AsyncIterator[TaskStreamItem]:
        prompt = "\n".join(
            part.text for part in request.parts if isinstance(part, TextPart)
        )
        messages = [
            {"role": "system", "content": request.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        chunks = []
        async for chunk in self.model_client.stream(
            model=request.config.model,
            messages=messages,
            metadata=request.metadata,
        ):
            chunks.append(chunk)
            yield TaskProgress.text(chunk)
        yield TaskResult.text("".join(chunks))


agent = StreamingAgent()
app = create_app(agent)


if __name__ == "__main__":
    agent.serve()
