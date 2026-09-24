"""Round-trip test for DeployedEnv through EnvInstanceStore.

Guards against the bug we hit while developing i6pn: when a new field is added to
DeployedEnv but the store's explicit doc/constructor dicts forget to plumb it
through, the field round-trips as None and the env silently deploys without it.
"""
from __future__ import annotations

from agent_env.env.env import DeployedEnv
from agent_env.env.store import EnvInstanceStore
from agent_env.config import set_document_store
from tst.unit.store.fakes import FakeDocumentStore


def test_deployed_env_roundtrip_preserves_all_fields():
    store = EnvInstanceStore()
    set_document_store(FakeDocumentStore())
    original = DeployedEnv(
        env_id="multi-test",
        env_version=3,
        gateway_url="https://gw.example/",
        mcp_url="https://gw.example/mcp",
        db_web_url="https://gw.example/dbweb/",
        sandbox_id="sb-abc",
        db_mcp_url="https://gw.example/dbmcp/mcp",
        website_frontend_urls={"a": "https://gw.example/site/a/"},
        sandbox_type="modal",
        vnc_url="vnc://example:5900",
        metadata={"k": "v"},
        instance_id=None,
        created_at_utc=None,
        expires_at_utc=None,
        gateway_mode="performance",
        sandbox_ids={
            "gateway_server": "sb-gw",
            "mcp_server": {"slack": "sb-1", "email": "sb-2"},
            "service_db": {"servicedb": "sb-pg", "pgweb": "sb-pgweb", "db-mcp": "sb-dbmcp"},
        },
        env_state_instance_ids=["esi-abc123"],
        environment_card_url="https://gw.example/.well-known/agent-env.json",
        environment_card={"name": "env1234", "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}],
                          "children_environments": [{"name": "slack", "url": "/svc/mcp-slack/agentenv"}]},
        environment_card_read_at_utc="2026-09-23T23:00:00.123456+00:00",
    )
    persisted = store.create_instance(original, ttl_seconds=60)
    rehydrated = store.get(persisted.instance_id)
    assert persisted.environment_card == original.environment_card  # create_instance rebuilds the record from its doc

    stamped_by_store = {"instance_id", "created_at_utc", "expires_at_utc"}
    for f in DeployedEnv.__dataclass_fields__:
        if f in stamped_by_store:
            continue
        assert getattr(rehydrated, f) == getattr(original, f), f"field {f!r} not preserved through store round-trip"


def test_deployed_env_from_dict_tolerates_missing_env_state_instance_ids():
    """Env instance docs written before env_state_instance_ids existed must still load
    (EnvInstanceStore.get -> from_dict), defaulting the field to an empty list."""
    legacy_doc = {
        "env_id": "multi-test",
        "env_version": 1,
        "gateway_url": "https://gw.example/",
        "mcp_url": "https://gw.example/mcp",
        "sandbox_id": "sb-abc",
        # no env_state_instance_ids key — predates the field
    }
    rehydrated = DeployedEnv.from_dict(legacy_doc)
    assert rehydrated.env_state_instance_ids == []


def test_deployed_env_from_dict_tolerates_records_without_a_card_or_gateway():
    """Records written before the stored card load without one, and gateway_url may be absent."""
    doc = {"env_id": "mcp-test", "env_version": 1, "mcp_url": "https://sb.example/mcp", "sandbox_id": "sb-abc"}
    rehydrated = DeployedEnv.from_dict(doc)
    assert rehydrated.gateway_url is None
    assert rehydrated.environment_card is None and rehydrated.environment_card_read_at_utc is None


class _RecordingStore:
    """Real sqlite store that also records the UpdateSpecs issued through it."""

    def __init__(self, inner):
        self._inner = inner
        self.updates: list = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def update(self, collection, filter, update, upsert=False):
        self.updates.append(update)
        return self._inner.update(collection, filter, update, upsert)

    def set_keys(self) -> list[str]:
        return [key for spec in self.updates for key in spec.set]


def _instance_store(tmp_path, doc):
    from agent_env.store.document_store.sqlite_document_store import (
        LocalSqliteDocumentStore,
    )

    from agent_env.env.store import ENV_INSTANCES_COLLECTION

    inner = LocalSqliteDocumentStore(str(tmp_path / "docs.sqlite"))
    inner.insert(ENV_INSTANCES_COLLECTION, doc)
    store = EnvInstanceStore()
    recording = _RecordingStore(inner)
    set_document_store(recording)
    return store, recording


def _read(tmp_path_store):
    from agent_env.store.document_store import Filter

    from agent_env.env.store import ENV_INSTANCES_COLLECTION

    store, recording = tmp_path_store
    return recording.find_one(ENV_INSTANCES_COLLECTION, Filter.of(instance_id="i-1"))


def test_merge_metadata_persists_when_the_record_has_metadata_null(tmp_path):
    """The shape create_instance actually writes.

    ``create_instance`` persists ``asdict(deployed_env)``, so an unset metadata
    field is stored as null rather than absent. MongoDB refuses to create a child
    under a scalar, so a dotted ``$set`` fails with PathNotViable — and because
    callers treat this as best-effort, the error is swallowed and the annotation
    is silently lost on exactly these records. Backends disagree (sqlite rewrites
    the null, mongomock silently no-ops), which is why the assertion below is on
    the operation shape and not just the resulting document.
    """
    bundle = _instance_store(tmp_path, {"instance_id": "i-1", "metadata": None})
    store, recording = bundle

    store.merge_metadata("i-1", {"deploy_step_id": "s1", "deployed_by_oauth_subject": "sub-42"})

    doc = _read(bundle)
    assert doc["metadata"]["deploy_step_id"] == "s1"
    assert doc["metadata"]["deployed_by_oauth_subject"] == "sub-42"
    # The load-bearing part: no dotted path was ever written beneath the null.
    assert not [k for k in recording.set_keys() if k.startswith("metadata.")], (
        "a dotted $set under a null parent is rejected by MongoDB; "
        f"issued keys were {recording.set_keys()}"
    )


def test_merge_metadata_persists_when_metadata_is_absent(tmp_path):
    bundle = _instance_store(tmp_path, {"instance_id": "i-1"})
    store, recording = bundle

    store.merge_metadata("i-1", {"deployed_by_oauth_subject": "sub-42"})

    assert _read(bundle)["metadata"]["deployed_by_oauth_subject"] == "sub-42"
    assert not [k for k in recording.set_keys() if k.startswith("metadata.")]


def test_merge_metadata_merges_without_clobbering_existing_keys(tmp_path):
    # Once metadata is a real subdocument the per-key merge applies, so a
    # concurrent writer of a different key is not overwritten.
    bundle = _instance_store(
        tmp_path, {"instance_id": "i-1", "metadata": {"deployed_agent": "keep-me"}}
    )
    store, recording = bundle

    store.merge_metadata("i-1", {"deployed_by_oauth_subject": "sub-42"})

    doc = _read(bundle)
    assert doc["metadata"]["deployed_agent"] == "keep-me"
    assert doc["metadata"]["deployed_by_oauth_subject"] == "sub-42"
    assert "metadata.deployed_by_oauth_subject" in recording.set_keys()


def test_merge_metadata_is_a_noop_for_empty_values(tmp_path):
    bundle = _instance_store(tmp_path, {"instance_id": "i-1", "metadata": None})
    store, recording = bundle

    store.merge_metadata("i-1", {})

    assert recording.updates == []
    assert _read(bundle)["metadata"] is None
