"""A computer env as the ``osworld_*`` steps see one, and a stand-in for the OSWorld controller's routes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
from agentenv_protocol.types import WELL_KNOWN_PATH

from agent_env.env.env import DeployedEnv
from agent_env.osworld import GUEST_EXTENSION

ENV_URL = "https://computer.example"
TOKEN = "s3cret-control-token"
CONTROLLER = "https://controller.example"


def card(endpoint: str = "/svc/osworld-server/osworld", *, uri: str = GUEST_EXTENSION) -> dict:
    """A computer's card with one app, which advertises ``uri`` at ``endpoint``."""
    app = {"name": "osworld-server", "url": "/svc/osworld-server/agentenv",
           "capabilities": {"extensions": [{"uri": uri, "params": {"endpoint": endpoint,
                                                                   "methods": {"guest": {"method": "GET"}}}}]}}
    return {"name": "desk", "capabilities": {"extensions": []}, "children_environments": [app]}


def computer(card_: dict | None = None, **metadata) -> DeployedEnv:
    return DeployedEnv(env_id="desk", env_version=1, environment_card_url=f"{ENV_URL}{WELL_KNOWN_PATH}",
                       environment_card=card() if card_ is None else card_, control_token=TOKEN,
                       metadata=dict(metadata))


@dataclass
class Controller:
    """The controller's routes, answered from ``answers`` by path (a list is answered in turn), and the requests made."""

    answers: dict[str, object] = field(default_factory=dict)
    seen: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        answer = self.answers.get(request.url.path, httpx.Response(404, text="no route"))
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.seen if r.url.path == path]

    def paths(self) -> list[str]:
        return [r.url.path for r in self.seen]


def route_httpx(monkeypatch, module, controller: Controller) -> None:
    """Send ``module``'s httpx clients to ``controller``."""
    real = httpx.AsyncClient

    def client(*args, **kwargs):
        return real(*args, transport=httpx.MockTransport(controller.handler), **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client)
