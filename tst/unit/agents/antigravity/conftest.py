"""A fresh agent per test, driven over ASGI the way agent-env reaches it: its workspace, conversations, skills and
changelog under the test's own folder, the SDK's stand-in reset with no scripted turns, and a stand-in for the turn's
model relay, since unit tests open no loopback servers."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from tst.unit.agents.antigravity import fake_sdk
from tst.unit.agents.antigravity.agent import a2a_server, antigravity_bridge, antigravity_date


class Harness:
    """One fresh agent and its app, reached the way agent-env reaches it: over HTTP."""

    def __init__(self, agent: a2a_server.AntigravityAgent) -> None:
        self.agent = agent
        self.app = a2a_server.build_app(agent)

    @property
    def card(self) -> dict:
        return self.app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True, by_alias=True)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://antigravity")

    @staticmethod
    async def send(client: httpx.AsyncClient, parts: str | list[dict], context_id: str = "ctx-1") -> dict:
        """message/send, blocking; the terminal task."""
        if isinstance(parts, str):
            parts = [{"kind": "text", "text": parts}]
        message = {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex, "contextId": context_id,
                   "parts": parts}
        response = await client.post("/a2a", json={
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": message, "configuration": {"blocking": True}},
        })
        response.raise_for_status()
        return response.json()["result"]


@pytest.fixture(autouse=True)
def sdk():
    """The SDK's stand-in, with no turns scripted and nothing recorded."""
    fake_sdk.Agent.turns, fake_sdk.Agent.configs, fake_sdk.Agent.prompts = [], [], []
    return fake_sdk


@pytest.fixture(autouse=True)
def relay(monkeypatch):
    """A turn's model relay as the bridge opens it, without its loopback server: it hands the harness a stand-in root
    and records what it was opened with, and its counted calls stay as the test leaves them."""
    opened: list[dict] = []

    @asynccontextmanager
    async def stand_in(upstream, calls, **kwargs):
        opened.append({"upstream": upstream, "calls": calls, **kwargs})
        yield "http://relay.test/token"

    monkeypatch.setattr(antigravity_bridge, "model_relay", stand_in)
    return opened


@pytest.fixture(autouse=True)
def folders(monkeypatch, tmp_path):
    """The agent's workspace, conversations, skills and changelog in the test's folder."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    conversations = tmp_path / "home" / "conversations"
    times = antigravity_date.Records(conversations / "env-times.json")
    images = antigravity_date.Records(conversations / "turn-images.json")
    filed = antigravity_date.Records(conversations / "conversation-records.json")
    for module in (a2a_server, antigravity_bridge):
        monkeypatch.setattr(module, "SAVE_DIR", conversations)
        monkeypatch.setattr(module, "SKILLS_DIR", tmp_path / "home" / "skills")
        monkeypatch.setattr(module, "ENV_TIMES", times)
        monkeypatch.setattr(module, "TURN_IMAGES", images)
        monkeypatch.setattr(module, "IMAGES_DIR", conversations / "images")
        monkeypatch.setattr(module, "CONVERSATION_RECORDS", filed)
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "INPUTS_DIR", workspace / "inputs")
    monkeypatch.setattr(a2a_server, "_CHANGELOG_DIR", tmp_path / "changelog")
    monkeypatch.setattr(a2a_server, "_CHANGELOG_SNAR", tmp_path / "changelog" / "state.snar")
    monkeypatch.setattr(a2a_server, "_CHANGELOG_META_DIR", tmp_path / f"changelog{a2a_server._META_DIR_SUFFIX}")
    return workspace


@pytest.fixture
def ag() -> Harness:
    """The agent as the image serves it, changelog included."""
    return Harness(a2a_server.AntigravityChangelogAgent())
