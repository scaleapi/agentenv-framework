"""A fresh agent per test, driven over ASGI the way agent-env reaches it, its workspace, Hermes profiles and skills
under the test's own folder."""

from __future__ import annotations

import os
import stat
import sys
import uuid
from types import SimpleNamespace

import httpx
import pytest

from tst.unit.agents.hermes.agent import a2a_server, hermes_bridge

FAKE_HERMES = os.path.join(os.path.dirname(__file__), "fake_hermes.py")


class Harness:
    """One fresh agent and its app, reached the way agent-env reaches it: over HTTP."""

    def __init__(self, agent: a2a_server.HermesAgent) -> None:
        self.agent = agent
        self.app = a2a_server.build_app(agent)

    @property
    def card(self) -> dict:
        return self.app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True, by_alias=True)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://hermes")

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
    """What run_hermes returns, with a finished turn's defaults."""
    defaults = {"response": "done", "error": None, "error_code": None, "session_id": "s-1", "trajectory": [],
                "tool_call_count": 0, "structured_output": None, "usage": {}}
    return SimpleNamespace(**{**defaults, **fields})


@pytest.fixture(autouse=True)
def folders(monkeypatch, tmp_path):
    """The agent's workspace, Hermes profiles and skills in the test's folder."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(a2a_server, "INPUTS_DIR", workspace / "inputs")
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hermes_bridge, "HERMES_ROOT", tmp_path / "hermes")
    return workspace


@pytest.fixture
def fake_hermes(monkeypatch, tmp_path):
    """A `hermes` that runs fake_hermes.py; what it was run with is in the returned record file."""
    launcher = tmp_path / "bin" / "hermes"
    launcher.parent.mkdir()
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_HERMES} \"$@\"\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    record = tmp_path / "record.json"
    monkeypatch.setattr(hermes_bridge, "HERMES_BIN", str(launcher))
    monkeypatch.setenv("FAKE_HERMES_RECORD", str(record))
    monkeypatch.setenv("FAKE_HERMES_PIDS", str(tmp_path / "pids"))
    return record


@pytest.fixture
def hermes() -> Harness:
    """The agent as the image serves it."""
    return Harness(a2a_server.HermesAgent())
