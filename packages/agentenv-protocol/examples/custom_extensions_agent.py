"""Custom LLM agent with single- and multi-operation custom extensions."""

from typing import Any

from agentenv_protocol.a2a_agent import (
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    ExtensionDefinition,
    FieldSchema,
    ImplementationOwner,
    OperationDefinition,
    TaskRequest,
    TaskResult,
    TextPart,
    a2a_agent,
    create_app,
    custom_extension,
    extension,
)
from pydantic import BaseModel, ConfigDict

try:
    from .reference_model import OpenAICompatibleModel
except ImportError:  # Support ``python examples/custom_extensions_agent.py``.
    from reference_model import OpenAICompatibleModel


class WordCountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class NoteCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class CustomAgentConfig(AgentConfig):
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a concise, helpful assistant."


NOTES_V1 = ExtensionDefinition(
    uri="urn:example:notes/v1",
    description="Create and list in-memory notes.",
    endpoint="/ext/notes/v1",
    core_operations={
        "create": OperationDefinition(
            name="create",
            method="POST",
            path="/ext/notes/v1",
            implementation=ImplementationOwner.RUNTIME,
            request=NoteCreateRequest,
            response=FieldSchema(required=("id", "text")),
        ),
        "list": OperationDefinition(
            name="list",
            method="GET",
            path="/ext/notes/v1",
            implementation=ImplementationOwner.RUNTIME,
            response=FieldSchema(required=("notes",)),
        ),
    },
)


@a2a_agent(
    identity=AgentIdentity(
        name="reference-extension-agent",
        description="Shows both custom-extension authoring forms.",
        version="1.0.0",
    ),
    config=CustomAgentConfig,
)
class CustomExtensionsAgent(AgentEnvAgent):
    def __init__(self, model_client: Any | None = None) -> None:
        self._notes: list[str] = []
        self.model_client = model_client or OpenAICompatibleModel()

    async def run(self, request: TaskRequest[CustomAgentConfig]) -> TaskResult:
        prompt = "\n".join(
            part.text for part in request.parts if isinstance(part, TextPart)
        )
        completion = await self.model_client.complete(
            model=request.config.model,
            messages=[
                {"role": "system", "content": request.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            metadata=request.metadata,
        )
        return TaskResult.text(completion.text)

    # Use custom_extension for one operation under a custom URI.
    @custom_extension(
        uri="urn:example:word-count/v1",
        operation="count",
        method="POST",
        path="/ext/word-count/v1",
        description="Count words in supplied text.",
        request=WordCountRequest,
        response=FieldSchema(required=("words",)),
    )
    async def count_words(self, request: WordCountRequest) -> dict[str, int]:
        return {"words": len(request.text.split())}

    # Use one shared definition when a custom URI has multiple operations.
    @extension(NOTES_V1.create)
    async def create_note(self, request: NoteCreateRequest) -> dict[str, object]:
        self._notes.append(request.text)
        return {"id": len(self._notes), "text": request.text}

    @extension(NOTES_V1.list)
    async def list_notes(self) -> dict[str, list[dict[str, object]]]:
        return {
            "notes": [
                {"id": index, "text": text}
                for index, text in enumerate(self._notes, start=1)
            ]
        }


agent = CustomExtensionsAgent()
app = create_app(agent)


if __name__ == "__main__":
    agent.serve()
