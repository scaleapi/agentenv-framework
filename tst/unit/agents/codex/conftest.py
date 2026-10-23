"""A fresh agent per test, driven over ASGI the way agent-env reaches it, its CLI state under the test's own folder."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from tst.unit.agents.codex.agent import a2a_server, codex_bridge


class Harness:
    """One fresh agent and its app, reached the way agent-env reaches it: over HTTP."""

    def __init__(self) -> None:
        self.agent = a2a_server.CodexAgent()
        self.app = a2a_server.build_app(self.agent)

    @property
    def card(self) -> dict:
        return self.app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True, by_alias=True)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://codex")

    async def send(
        self, client: httpx.AsyncClient, text: str, context_id: str | None = None, *, blocking: bool = True
    ) -> dict:
        """message/send; the task, terminal unless ``blocking`` is False."""
        message = {
            "kind": "message",
            "role": "user",
            "messageId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": text}],
        }
        if context_id is not None:
            message["contextId"] = context_id
        params: dict = {"message": message}
        if not blocking:
            params["configuration"] = {"blocking": False}
        response = await client.post(
            "/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": params}
        )
        response.raise_for_status()
        return response.json()["result"]

    @staticmethod
    def answer_turns(monkeypatch, *turns: list[dict], gate: asyncio.Event | None = None) -> None:
        """Stand in for the CLI: each turn reports its events, then waits for ``gate`` if one is given."""
        remaining = iter(turns)

        async def fake_run_codex(**kwargs):
            events = next(remaining)
            for event in events:
                kwargs["event_callback"](event)
            if gate is not None:
                await gate.wait()
            return codex_bridge.CodexResult(response="done", tool_call_count=0, session_id=None, trajectory=events)

        monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    @staticmethod
    def terminal(task: dict) -> tuple[str, str, list[dict]]:
        """A terminal task's state, its text, and its data parts in order."""
        parts = task["status"]["message"]["parts"]
        text = next(part["text"] for part in parts if part["kind"] == "text")
        return task["status"]["state"], text, [part["data"] for part in parts if part["kind"] == "data"]


@pytest.fixture
def codex(monkeypatch, tmp_path) -> Harness:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(a2a_server, "_active_execution_tasks", {})
    monkeypatch.setattr(a2a_server, "_context_locks", {})
    monkeypatch.setattr(a2a_server, "_workspace_barrier", a2a_server._WorkspaceBarrier())
    return Harness()
