"""A fresh agent per test, its directories under the test's own, and a fake object store behind agent-env's
grants."""

from __future__ import annotations

import httpx
import pytest
from agentenv_protocol import transfers as protocol_transfers
from starlette.testclient import TestClient

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import FakeStore


@pytest.fixture(autouse=True)
def agent_dirs(monkeypatch, tmp_path):
    """The workspace and the CLI's state under the test's folder, never this machine's ~/.claude."""
    state = tmp_path / "claude-state"
    sessions = state / "projects" / "-app"
    for module in (a2a_server, claude_bridge):
        monkeypatch.setattr(module, "CLAUDE_STATE_DIR", state)
        monkeypatch.setattr(module, "SESSION_JSONL_DIR", sessions)
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", state / "skills")
    for module in (a2a_server, claude_bridge):
        monkeypatch.setattr(module, "WORKSPACE_DIR", tmp_path / "app")
    monkeypatch.setattr(a2a_server, "INPUTS_DIR", tmp_path / "app" / "inputs")
    return state


@pytest.fixture
def app():
    return a2a_server.build_app(a2a_server.ClaudeCodeAgent())


@pytest.fixture
def agent(app):
    return app.state.agentenv_a2a.agent


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    """Every grant transfer against one fake store."""
    fake = FakeStore()
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(protocol_transfers.httpx, "AsyncClient",
                        lambda **kwargs: real_async_client(transport=httpx.MockTransport(fake), **kwargs))
    monkeypatch.setattr(protocol_transfers, "_RETRY_BACKOFF_SECONDS", 0)
    return fake
