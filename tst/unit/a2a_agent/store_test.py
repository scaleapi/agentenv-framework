"""Round-trip test for DeployedA2AAgent through A2AAgentInstanceStore."""
from __future__ import annotations

from agent_env.a2a_agent.a2a_agent import DeployedA2AAgent
from agent_env.a2a_agent.store import A2AAgentInstanceStore
from agent_env.config import set_document_store
from tst.unit.store.fakes import FakeDocumentStore


def test_deployed_a2a_agent_roundtrip_preserves_all_fields():
    store = A2AAgentInstanceStore()
    set_document_store(FakeDocumentStore())
    original = DeployedA2AAgent(
        agent_id="a2a-default-hardened",
        agent_version=1,
        a2a_url="https://agent.example/",
        sandbox_id="sb-agent",
        agent_card={"name": "Agent Gateway", "capabilities": {"extensions": []}},
        sandbox_type="modal",
        instance_id=None,
        created_at_utc=None,
        expires_at_utc=None,
    )
    persisted = store.create_instance(original, ttl_seconds=60)
    rehydrated = store.get(persisted.instance_id)

    stamped_by_store = {"instance_id", "created_at_utc", "expires_at_utc"}
    for f in DeployedA2AAgent.__dataclass_fields__:
        if f in stamped_by_store:
            continue
        assert getattr(rehydrated, f) == getattr(original, f), f"field {f!r} not preserved through store round-trip"
