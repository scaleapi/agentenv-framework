"""Minimal custom LLM agent using the framework-owned task lifecycle."""

from typing import Any

from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    WriteOnly,
    a2a_agent,
    create_app,
)

try:
    from .reference_model import OpenAICompatibleModel
except ImportError:  # Support ``python examples/basic_agent.py``.
    from reference_model import OpenAICompatibleModel


class BasicAgentConfig(AgentConfig):
    """Every field is advertised and delivered to ``run`` as typed config."""

    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a concise, helpful assistant."
    max_input_chars: int = 1_000
    provider_token: WriteOnly[str | None] = None


@a2a_agent(
    identity=AgentIdentity(
        name="reference-llm-agent",
        description="A custom LLM agent using the normal SDK path.",
        version="1.0.0",
    ),
    config=BasicAgentConfig,
    config_description="Configure the reference model runtime.",
    extensions=(TRAJECTORY_V1,),
)
class BasicAgent(AgentEnvAgent):
    def __init__(self, model_client: Any | None = None) -> None:
        self.model_client = model_client or OpenAICompatibleModel()

    async def run(self, request: TaskRequest[BasicAgentConfig]) -> TaskResult:
        prompt = "\n".join(
            part.text for part in request.parts if isinstance(part, TextPart)
        )
        prompt = prompt[: request.config.max_input_chars]
        completion = await self.model_client.complete(
            model=request.config.model,
            messages=[
                {"role": "system", "content": request.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            metadata=request.metadata,
            api_key=request.config.provider_token,
        )

        # Redirect this A2A context to the runtime-native session it owns.
        session_ref = request.session_ref or f"model-session-{request.context_id}"
        return (
            TaskResult.builder()
            .succeeded()
            .add_text(completion.text)
            .session_ref(session_ref)
            .usage(
                Usage(
                    tool_call_count=0,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                )
            )
            .native_trajectory(
                format="reference-model-events/v1",
                payload=[
                    {
                        "type": "model_completion",
                        "model": request.config.model,
                        "input": prompt,
                        "output": completion.text,
                    }
                ],
            )
            .build()
        )


agent = BasicAgent()
app = create_app(agent)


if __name__ == "__main__":
    agent.serve()
