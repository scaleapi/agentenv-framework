"""A fresh agent per test, driven over HTTP the way agent-env drives it; a gemini home whose cwd slug is bootstrapped;
and a fake object store behind agent-env's grants."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import transfers as protocol_transfers
from starlette.testclient import TestClient

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge
from tst.unit.agents.gemini.support import FakeStore, Staging


@pytest.fixture
def home(monkeypatch, tmp_path) -> Path:
    """HOME, holding a gemini home whose cwd slug is bootstrapped, as the lifespan does at
    startup. GEMINI_CLI_HOME is unset, so the MCP settings land in HOME too."""
    root = tmp_path / "home"
    gemini_home = root / ".gemini"
    cwd = str(tmp_path / "workspace")
    (gemini_home / "tmp" / "workspace" / "chats").mkdir(parents=True)
    (gemini_home / "projects.json").write_text(json.dumps({"projects": {cwd: "workspace"}}))
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("GEMINI_CWD", cwd)
    monkeypatch.delenv("GEMINI_CLI_HOME", raising=False)
    for module in (gemini_bridge, a2a_server):
        monkeypatch.setattr(module, "GEMINI_HOME", gemini_home)
        monkeypatch.setattr(module, "GEMINI_TMP", gemini_home / "tmp")
        monkeypatch.setattr(module, "GEMINI_PROJECTS_JSON", gemini_home / "projects.json")
    monkeypatch.setattr(a2a_server, "GEMINI_CWD", cwd)
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", gemini_home / "skills")
    monkeypatch.setattr(a2a_server, "INPUTS_DIR", Path(cwd) / "inputs")
    return root


@pytest.fixture
def changelog_dir(monkeypatch, tmp_path) -> Path:
    """The changelog's working directory, private to the test, named as the agent names its own."""
    root = tmp_path / "gemini-changelog-test"
    monkeypatch.setattr(a2a_server, "changelog_dir", lambda environ: root)
    return root


@pytest.fixture
def app(home, changelog_dir):
    return a2a_server.build_app(a2a_server.GeminiChangelogAgent())


@pytest.fixture
def agent(app) -> a2a_server.GeminiAgent:
    return app.state.agentenv_a2a.agent


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    """Every grant transfer, the agent's and the framework's own, against one fake store."""
    fake = FakeStore()
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        protocol_transfers.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=httpx.MockTransport(fake), **kwargs),
    )
    monkeypatch.setattr(protocol_transfers, "_RETRY_BACKOFF_SECONDS", 0)
    return fake


@pytest.fixture
def staging(socket_enabled):
    """The changelog hook uploads to a server on loopback, as it reaches the agent's own staging."""
    server = Staging()
    yield server
    server.close()
