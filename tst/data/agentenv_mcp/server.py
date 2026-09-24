"""Minimal in-memory MCP server built on AgentEnvEnvironment, for integration testing."""
import json
import random
from typing import Annotated
from urllib.parse import urlparse

import httpx
from pydantic import Field

from agentenv_protocol import AgentEnvEnvironment, DataPart, EnvironmentCapabilities, EnvironmentExtension, add_data, environment_card, extension, get_data, reset_data, tool


@environment_card(
    name="items",
    capabilities=EnvironmentCapabilities(
        extensions=[
            EnvironmentExtension(
                uri="urn:agentenv:disable-tool/v1",
                description="Disable an MCP tool by name via the gateway tool-access endpoint.",
                params={
                    "endpoint": "/tools/disable",
                    "methods": {
                        "disable": {
                            "method": "POST",
                            "request": {
                                "type": "object",
                                "properties": {"role": {"type": "string"}, "tools": {"type": "array"}},
                                "required": ["role", "tools"],
                            },
                        }
                    },
                },
            )
        ]
    ),
)
class ItemsEnv(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.store: list = []
        self.errors: dict = {}
        self.env_get_time_url: str | None = None
        self.create_app()
        # list_items stays imperatively registered — @tool is additive; both styles coexist.
        self.mcp.tool(name="list_items")(self.list_items)

    @reset_data
    async def _reset(self) -> None:
        self.store.clear()

    @add_data
    async def _add(self, parts: list) -> None:
        for part in parts:
            if part.kind == "data":
                self.store.extend(part.data.get("items", []))
            elif part.kind == "file" and part.file.uri.startswith("file://"):
                path = urlparse(part.file.uri).path
                mt = getattr(part.file, "mimeType", None)
                if mt == "application/json" or path.endswith(".json"):
                    with open(path, encoding="utf-8") as f:
                        self.store.extend(json.load(f).get("items", []))
                else:
                    # Read as bytes so binary content types don't raise
                    # UnicodeDecodeError; decode defensively for the text repr.
                    with open(path, "rb") as f:
                        text = f.read().decode("utf-8", errors="replace").strip()
                    self.store.append(f"file:{mt}:{text}")

    @get_data
    async def _state(self) -> list:
        return [DataPart(data={"items": self.store})]

    @extension(uri="urn:agentenv:set-errors/v1", description="Make a tool start raising at a given error rate.")
    async def set_errors(self, tool_name: str, error_rate: float) -> dict:
        self.errors[tool_name] = error_rate
        return {"tool_name": tool_name, "error_rate": error_rate}

    @extension(uri="urn:agentenv:clock/v1", description="Sync this server to the gateway's virtual clock.")
    async def sync_time(self, env_get_time_url: str) -> dict:
        """Read the gateway's virtual time through the handed env_get_time_url (proves server->gateway reachability)."""
        self.env_get_time_url = env_get_time_url
        async with httpx.AsyncClient() as client:
            resp = await client.get(env_get_time_url, timeout=10)
            resp.raise_for_status()
            return {"env_get_time_url": env_get_time_url, "gateway_virtual_time": resp.json()["virtual_time"]}

    @tool(name="{environment_name}_add_item")
    async def add_item(
        self,
        item: Annotated[str, Field(description="The item to add.")],
        times: Annotated[int, Field(description="How many copies to add.")] = 1,
    ) -> str:
        """Add an item to the store."""
        self.store.extend([item] * times)
        return json.dumps({"count": len(self.store)})

    def list_items(self) -> str:
        """List all items currently stored."""
        rate = self.errors.get("list_items", 0.0)
        if rate and random.random() < rate:  # nosec B311 - test-only fault injection, not security
            raise RuntimeError("injected error for list_items")
        return json.dumps({"items": self.store})

if __name__ == "__main__":
    ItemsEnv().serve()
