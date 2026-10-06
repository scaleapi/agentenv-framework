from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import agent_env.providers.sandbox_providers.sandbox_provider as provider_mod
from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.config import configure
from agent_env.task_step.context import TaskStepContext
from tst.util.granting_object_store import GrantingObjectStore


@pytest.fixture
def store(tmp_path) -> GrantingObjectStore:
    store = GrantingObjectStore(str(tmp_path))
    configure(object_store=store)
    return store


@pytest.fixture
def requests(monkeypatch) -> list[dict]:
    sent: list[dict] = []

    async def fake_request(
        self, method, url, *, json=None, timeout=None, **kwargs
    ):  # noqa: A002
        sent.append({"method": method, "url": url, "json": json})
        return httpx.Response(
            200,
            json={"ok": True, "count": 2, "context_id": "restored"},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    class _Sandbox:
        mode = "modal"

        async def exec_with_output(self, *args):
            return 0, "marker-token", ""

    class _Provider:
        async def get_sandbox(self, sandbox_id):
            return _Sandbox()

        async def close(self):
            return None

    monkeypatch.setattr(provider_mod, "build_sandbox_provider", lambda spec: _Provider())
    persisted = SimpleNamespace(metadata={})
    persisted.update_metadata = lambda metadata: setattr(persisted, "metadata", metadata)
    monkeypatch.setattr(A2AAgent, "get", classmethod(lambda cls, agent_id: persisted))
    return sent


def _context(capture: dict, *apply_variants: list[str]) -> TaskStepContext:
    method = {
        "endpoint": "/ext/snapshot/changelog",
        "request": {"oneOf": [{"required": fields} for fields in apply_variants]},
    }
    apply_agent = SimpleNamespace(
        agent_name="apply",
        a2a_url="https://agent",
        api_url="https://agent",
        a2a_card={
            "capabilities": {
                "extensions": [
                    {
                        "uri": A2AAgent.EXT_SNAPSHOT,
                        "params": {
                            "methods": {A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG: method}
                        },
                    }
                ]
            }
        },
        sandbox_type="local",
        sandbox_id="sandbox-1",
    )
    context = TaskStepContext()
    context.deployed_agents = [apply_agent]
    context.metadata["agent_changelog"] = [{"agent_name": "capture", **capture}]
    return context


async def _validate(context: TaskStepContext) -> dict:
    await A2AAgentValidator._validate_agent_changelog(
        SimpleNamespace(id="agent-1"),
        context,
        capture_agent_name="capture",
        apply_agent_name="apply",
        marker_path="/tmp/marker",
        token="marker-token",
    )
    return context.metadata["verifications"]["a2a_agent_changelog"]


@pytest.mark.asyncio
async def test_validator_applies_a_captured_portable_changelog(store, requests):
    for name in ("000001.tar", "000000.tar"):
        store.put(f"changelog/run-1/{name}", b"increment")
    context = _context(
        {"object_url": store.object_url("changelog/run-1"), "transfer_mode": "objects"},
        ["increments"],
    )

    verification = await _validate(context)

    assert [item["sequence"] for item in requests[0]["json"]["increments"]] == [0, 1]
    assert verification == {
        "methods_advertised": True,
        "save": True,
        "apply": True,
        "roundtrip": True,
        "note": "",
    }


@pytest.mark.asyncio
async def test_validator_records_an_apply_agent_without_the_object_form(store, requests):
    store.put("changelog/run-1/000000.tar", b"increment")
    context = _context(
        {"object_url": store.object_url("changelog/run-1"), "transfer_mode": "objects"},
        ["s3_prefix"],
    )

    verification = await _validate(context)

    assert not requests
    assert store.granted == []
    assert verification["apply"] is False
    assert "does not advertise the object form" in verification["note"]


@pytest.mark.asyncio
async def test_validator_records_a_capture_it_cannot_send_instead_of_raising(store, requests):
    for name in ("000000.tar", "notes.txt"):
        store.put(f"changelog/run-1/{name}", b"increment")
    context = _context(
        {"object_url": store.object_url("changelog/run-1"), "transfer_mode": "objects"},
        ["increments"],
    )

    verification = await _validate(context)

    assert not requests
    assert verification["apply"] is False
    assert verification["note"].startswith("apply request failed: ")
