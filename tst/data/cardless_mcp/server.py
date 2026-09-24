"""A deliberately CARDLESS AgentEnvEnvironment, for integration-testing name resolution.

Declares no ``@environment_card``, so its identity comes purely from the SDK's
resolution chain. Since the ``SERVICE_NAME`` fallback was dropped, that chain is
``card name -> ENVIRONMENT_NAME -> class name``, and with no card the only thing standing
between this server and its class name is the ``ENVIRONMENT_NAME`` the gateway injects.

The class name is deliberately distinctive: if resolution ever regresses, the served card
and the ``{environment_name}``-templated tool come back as ``UnnamedProbeEnv`` instead of
the env's registered name, which is unmissable in the assertion.
"""
import json
from typing import Annotated

from pydantic import Field

from agentenv_protocol import AgentEnvEnvironment, DataPart, add_data, get_data, reset_data, tool


class UnnamedProbeEnv(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.store: list = []
        self.create_app()

    @reset_data
    async def _reset(self) -> None:
        self.store.clear()

    @add_data
    async def _add(self, parts: list) -> None:
        for part in parts:
            if part.kind == "data":
                self.store.extend(part.data.get("items", []))

    @get_data
    async def _state(self) -> list:
        return [DataPart(data={"items": self.store})]

    @tool(name="{environment_name}_add_item")
    async def add_item(self, item: Annotated[str, Field(description="The item to add.")]) -> str:
        """Add an item to the store."""
        self.store.append(item)
        return json.dumps({"count": len(self.store)})


if __name__ == "__main__":
    UnnamedProbeEnv().serve()
