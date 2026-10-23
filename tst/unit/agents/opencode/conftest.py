"""A fresh agent per test, driven over ASGI the way agent-env reaches it, its workspace, sessions and changelog under
the test's own folder."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import httpx
import pytest

from tst.unit.agents.opencode.agent import a2a_server, opencode_bridge


class Harness:
    """One fresh agent and its app, reached the way agent-env reaches it: over HTTP."""

    def __init__(self, agent: a2a_server.OpenCodeAgent) -> None:
        self.agent = agent
        self.app = a2a_server.build_app(agent)

    @property
    def card(self) -> dict:
        return self.app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True, by_alias=True)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://opencode")

    @staticmethod
    async def send(client: httpx.AsyncClient, parts: str | list[dict], context_id: str = "ctx-1", *,
                   blocking: bool = True) -> dict:
        """message/send; the task, terminal unless ``blocking`` is False."""
        if isinstance(parts, str):
            parts = [{"kind": "text", "text": parts}]
        message = {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex, "contextId": context_id,
                   "parts": parts}
        response = await client.post("/a2a", json={
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": message, "configuration": {"blocking": blocking}},
        })
        response.raise_for_status()
        return response.json()["result"]


def result(**fields) -> SimpleNamespace:
    """What run_opencode returns, with a finished turn's defaults."""
    defaults = {"response": "done", "error": None, "error_code": None, "session_id": "ses_1", "trajectory": [],
                "tool_call_count": 0}
    return SimpleNamespace(**{**defaults, **fields})


@pytest.fixture(autouse=True)
def folders(monkeypatch, tmp_path):
    """The agent's workspace, sessions and changelog in the test's folder."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "INPUTS_DIR", workspace / "inputs")
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(a2a_server, "_CHANGELOG_DIR", tmp_path / "changelog")
    monkeypatch.setattr(a2a_server, "_CHANGELOG_SNAR", tmp_path / "changelog" / "state.snar")
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    return workspace


@pytest.fixture
def oc() -> Harness:
    """The agent as the image serves it, changelog included."""
    return Harness(a2a_server.OpenCodeChangelogAgent())
