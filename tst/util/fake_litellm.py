"""A fake LiteLLM proxy key API, as much of it as model budgets use, answered in process through
``httpx.MockTransport``: ``/key/generate``, ``/key/block`` and ``/key/delete``, shaped as LiteLLM 1.106 answers them.
Every request is recorded; ``refuse`` answers the next ``/key/generate`` with an error, ``fail_block`` every
``/key/block`` for those key ids with a 500. ``serve`` makes it the configured endpoint."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from typing import Optional

import httpx

from agent_env import model_budget
from agent_env.config.model import ModelConfig

PROXY = "https://proxy.test:4000"


@dataclass
class FakeKey:
    key: str
    alias: str
    max_budget: float
    duration: str
    metadata: dict
    blocked: bool = False


@dataclass
class FakeLiteLLM:
    #: Who a made key belongs to: None, as for keys the master key makes; a user's id, as for an internal user's.
    user_id: Optional[str] = None
    refuse: Optional[tuple[int, str]] = None
    fail_block: set[str] = field(default_factory=set)
    keys: dict[str, FakeKey] = field(default_factory=dict)  # by key id, LiteLLM's hash of the key
    requests: list[httpx.Request] = field(default_factory=list)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def calls(self, path: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.url.path == path]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/key/generate":
            if self.refuse is not None:
                status, message = self.refuse
                self.refuse = None
                return _error(status, message)
            body = json.loads(request.content)
            if any(made.alias == body["key_alias"] for made in self.keys.values()):
                return _error(400, f"Key with alias '{body['key_alias']}' already exists.")
            key = "sk-" + secrets.token_urlsafe(16)
            key_id = hashlib.sha256(key.encode()).hexdigest()
            self.keys[key_id] = FakeKey(key, body["key_alias"], body["max_budget"], body["duration"], body["metadata"])
            return httpx.Response(200, json={
                "key": key, "token": key_id, "token_id": key_id, "key_alias": body["key_alias"],
                "max_budget": body["max_budget"], "user_id": self.user_id, "spend": 0.0,
            })
        if path == "/key/block":
            key_id = json.loads(request.content)["key"]
            if key_id in self.fail_block:
                return _error(500, "database unavailable")
            if key_id not in self.keys:
                return _error(404, "Key not found.")
            self.keys[key_id].blocked = True
            return httpx.Response(200, json={"token": key_id, "key_alias": self.keys[key_id].alias, "blocked": True})
        if path == "/key/delete":
            ids = [key_id for key_id in json.loads(request.content)["keys"] if key_id in self.keys]
            if not ids:
                return _error(404, "{'error': 'No keys found'}")
            for key_id in ids:
                del self.keys[key_id]
            return httpx.Response(200, json={"deleted_keys": ids})
        return _error(404, f"no route {path}")


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "error", "code": str(status)}})


def serve(monkeypatch, config, **model) -> FakeLiteLLM:
    """A fake LiteLLM at ``PROXY``, made the endpoint of ``config`` (``local_stores``), with ``[model]`` keys
    ``api_key = "sk-shared"`` and ``admin_api_key = "sk-admin"``; ``model`` sets or adds others."""
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setattr(config, "_model_cfg", ModelConfig(
        **{"base_url": PROXY, "api_key": "sk-shared", "admin_api_key": "sk-admin", **model}))
    fake = FakeLiteLLM()
    monkeypatch.setattr(model_budget, "_client", fake.client)
    return fake
