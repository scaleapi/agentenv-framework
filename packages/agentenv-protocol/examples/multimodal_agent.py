"""Custom vision agent sending text and image file parts to an LLM."""

from typing import Any

from agentenv_protocol.a2a_agent import (
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    FilePart,
    TaskRequest,
    TaskResult,
    TextPart,
    a2a_agent,
    create_app,
)

try:
    from .reference_model import OpenAICompatibleModel
except ImportError:  # Support ``python examples/multimodal_agent.py``.
    from reference_model import OpenAICompatibleModel


class MultimodalConfig(AgentConfig):
    model: str = "gpt-4o-mini"
    system_prompt: str = "Describe supplied images accurately and concisely."


@a2a_agent(
    identity=AgentIdentity(
        name="reference-multimodal-agent",
        description="A custom vision agent accepting text and PNG images.",
        version="1.0.0",
        input_modes=("text", "image/png"),
        output_modes=("text", "application/json"),
    ),
    config=MultimodalConfig,
)
class MultimodalAgent(AgentEnvAgent):
    def __init__(self, model_client: Any | None = None) -> None:
        self.model_client = model_client or OpenAICompatibleModel()

    async def run(self, request: TaskRequest[MultimodalConfig]) -> TaskResult:
        prompt = "\n".join(
            part.text for part in request.parts if isinstance(part, TextPart)
        )
        images = [
            {
                "name": part.name,
                "mime_type": part.mime_type,
                "transport": "inline" if part.bytes is not None else "uri",
            }
            for part in request.parts
            if isinstance(part, FilePart) and part.mime_type == "image/png"
        ]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{part.mime_type};base64,{part.bytes}"
                        if part.bytes is not None
                        else part.uri
                    )
                },
            }
            for part in request.parts
            if isinstance(part, FilePart) and part.mime_type == "image/png"
        )
        completion = await self.model_client.complete(
            model=request.config.model,
            messages=[
                {"role": "system", "content": request.config.system_prompt},
                {"role": "user", "content": content},
            ],
            metadata=request.metadata,
        )

        return (
            TaskResult.builder()
            .succeeded()
            .add_text(completion.text)
            .add_structured_output({"prompt": prompt, "images": images})
            .build()
        )


agent = MultimodalAgent()
app = create_app(agent)


if __name__ == "__main__":
    agent.serve()
