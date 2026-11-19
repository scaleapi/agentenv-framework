"""A snapshot of an agent that serves atif/1 carries its SDK's own state, the context's turn count, as a JSON object
beside the agent's: its save is granted the object, and its load the stored one. An agent that doesn't keep one is
granted none, and nothing it would refuse."""

from __future__ import annotations

from typing import Any

import pytest
from agentenv_protocol.a2a_agent import (
    SNAPSHOT_V1,
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    TaskRequest,
    TaskResult,
    a2a_agent,
    enable,
    extension,
)
from agentenv_protocol.atif import Adapter, atif_adapter, native
from starlette.testclient import TestClient

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import (
    SNAPSHOT_STATE_MAX_BYTES,
    snapshot_load_call,
    snapshot_save_call,
)
from tst.util.granting_object_store import GrantingObjectStore


@atif_adapter(native_format="toy/1", version="1.0.0", input_includes_cache=None)
class _ToyAtif(Adapter):
    @native(type="say")
    def say(self, record: dict[str, Any], turn: Any) -> None:
        turn.message(step="say", text=record["text"])
        turn.step(key="say", llm_call_count=0)


def _card(*, atif: bool) -> dict[str, Any]:
    trajectory = enable(TRAJECTORY_V1, native="toy/1", **({"atif": _ToyAtif} if atif else {}))

    @a2a_agent(identity=AgentIdentity(name="toy", description="test", version="1"), extensions=(trajectory,))
    class _Toy(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return {"context_id": request.context_id, "objects": {}}

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest):
            return {"context_id": "restored"}

    return TestClient(_Toy().create_app()).get("/.well-known/agent-card.json").json()


def _method(card: dict[str, Any], name: str) -> dict[str, Any] | None:
    return A2AAgent.operation(A2AAgent.find_extension(card, SNAPSHOT_V1.uri), name)[0]


def _nulls(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [None for item in value.values() if item is None] + [n for item in value.values() for n in _nulls(item)]
    if isinstance(value, list):
        return [n for item in value for n in _nulls(item)]
    return []


@pytest.fixture
def store(tmp_path) -> GrantingObjectStore:
    return GrantingObjectStore(str(tmp_path))


def test_an_agents_card_says_whether_it_serves_atif() -> None:
    assert A2AAgent.serves_atif(_card(atif=True))
    assert not A2AAgent.serves_atif(_card(atif=False))
    assert not A2AAgent.serves_atif({})


@pytest.mark.parametrize("atif", [True, False], ids=["serves-atif", "native-only"])
def test_a_snapshot_save_grants_the_state_only_to_an_agent_that_keeps_one(store, atif) -> None:
    card = _card(atif=atif)

    call = snapshot_save_call(
        _method(card, "save"), store, agent_name="toy", context_id="ctx",
        capture_prefix=store.object_url("snap/"), sandbox_type="local", state=A2AAgent.serves_atif(card),
    )

    objects = call.payload["objects"]
    assert set(objects) == {"trajectory", "workspace", *(["state"] if atif else [])}
    if atif:
        assert (objects["state"]["media_type"], objects["state"]["max_bytes"]) == (
            "application/json", SNAPSHOT_STATE_MAX_BYTES)
    # A pinned agent's request model forbids a field it doesn't know, a null one included.
    assert _nulls(call.payload) == []


@pytest.mark.parametrize(
    "atif, stored, granted",
    [(True, True, True), (True, False, False), (False, True, False)],
    ids=["serves-atif", "capture-without-state", "native-only"],
)
def test_a_snapshot_load_grants_the_stored_state_to_an_agent_that_keeps_one(store, atif, stored, granted) -> None:
    card = _card(atif=atif)
    names = {"trajectory", "workspace", *(["state"] if stored else [])}
    for name in names:
        store.put(f"snap/{name}", b'{"turns": 2}' if name == "state" else b"bytes")

    call = snapshot_load_call(
        _method(card, "load"), store, agent_name="toy", bundle_url=store.object_url("snap/"), file_names=names,
        target_context_id=None, sandbox_type="local", state=A2AAgent.serves_atif(card),
    )

    objects = call.payload["objects"]
    assert ("state" in objects) is granted
    if granted:
        assert (objects["state"]["media_type"], objects["state"]["size_bytes"]) == ("application/json", 12)
