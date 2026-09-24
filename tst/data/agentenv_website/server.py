"""Minimal Starlette website backend on AgentEnvStarletteApplication, for integration
testing the website env's v1 data-plane (WebsiteEnv.load_environment_artifact)."""
import json
import os
from urllib.parse import urlparse

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agentenv_protocol import AgentEnvStarletteApplication, DataPart, EnvironmentCard, add_data, get_data, reset_data

WEBITEMS_CARD = EnvironmentCard(name="webitems")


class WebItemsEnv:
    def __init__(self) -> None:
        self.store: list = []
        self.app = Starlette(routes=[Route("/api/health", self._health, methods=["GET"])])
        AgentEnvStarletteApplication(environment_card=WEBITEMS_CARD, handler=self).add_routes_to_app(self.app)

    async def _health(self, request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

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

    def run(self) -> None:
        uvicorn.run(self.app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    WebItemsEnv().run()
