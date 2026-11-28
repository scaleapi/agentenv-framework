"""Eval instances and their slots: the conformance kit on SQLite and through namespace routing, where an ``@local`` or
``@hf`` eval's records land in this machine's local store and a registry eval's in the configured one."""

import pytest

from agent_env.config import configure, get_config, reset_config
from agent_env.eval import instance_store
from agent_env.eval.instance import (
    EvalInstance,
    EvalInstanceStatus,
    EvalSettings,
    EvalSource,
    Lineage,
    RunnerDeclarations,
    Slot,
    SlotState,
    SourceKind,
    new_eval_instance_id,
    planned_slots,
)
from agent_env.runner.local_runner import LocalRunner
from agent_env.store import Filter, LocalSqliteDocumentStore
from agent_env.store.routing import RoutingDocumentStore
from tst.store import eval_instance_conformance

REGISTRY_EVAL = "evals/smoke"
LOCAL_EVAL = "@local/~/bundle/evals/smoke"
HUB_EVAL = "@hf/acme/ds/hello/evals/smoke"


@pytest.fixture
def sqlite(tmp_path):
    configure(document_store=LocalSqliteDocumentStore(str(tmp_path / "documents.db")))
    yield
    reset_config()


@pytest.fixture
def routed(tmp_path, cli_routing):
    """A configured store standing in for a shared one, routed beside this machine's local stores."""
    configured = LocalSqliteDocumentStore(str(tmp_path / "configured.db"))
    configure(document_store=configured)
    router = get_config().get_document_store()
    assert isinstance(router, RoutingDocumentStore)
    yield router, configured
    reset_config()


@pytest.mark.parametrize("case", eval_instance_conformance.CASES, ids=lambda case: case.__name__)
def test_sqlite_passes_the_eval_instance_conformance_kit(case, sqlite):
    case(REGISTRY_EVAL)


@pytest.mark.parametrize("root", [REGISTRY_EVAL, LOCAL_EVAL, HUB_EVAL], ids=["registry", "local", "hf"])
@pytest.mark.parametrize("case", eval_instance_conformance.CASES, ids=lambda case: case.__name__)
def test_routing_passes_the_eval_instance_conformance_kit(case, root, routed):
    case(root)


@pytest.mark.parametrize("root, namespace", [(LOCAL_EVAL, "local"), (HUB_EVAL, "hf")])
def test_a_local_evals_records_stay_in_its_namespaces_store(root, namespace, routed):
    router, configured = routed
    instance = _instance(root)
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [(f"{root}/tasks/a", 1)], k=2))
    instance_store.record_slot(instance.id, f"{root}/tasks/a@1#2", SlotState.GRADED, {"result": "passed"})

    local = router.namespaces[namespace]
    assert instance.id.startswith(root)
    assert [doc["id"] for doc in local.query(instance_store.EVAL_INSTANCES_COLLECTION, Filter())] == [instance.id]
    assert {doc["id"] for doc in local.query(instance_store.EVAL_SLOTS_COLLECTION, Filter())} == {
        f"{root}/tasks/a@1#1", f"{root}/tasks/a@1#2"}
    for collection in (instance_store.EVAL_INSTANCES_COLLECTION, instance_store.EVAL_SLOTS_COLLECTION):
        assert configured.query(collection, Filter()) == []
    assert instance_store.get_eval_instance(instance.id).slots_rev == 1


def test_a_registry_evals_records_go_to_the_configured_store(routed):
    router, configured = routed
    instance = _instance(REGISTRY_EVAL)
    instance_store.create_eval_instance(instance, planned_slots(instance.id, [("tasks/a", 1)], k=1))

    assert [doc["id"] for doc in configured.query(instance_store.EVAL_INSTANCES_COLLECTION, Filter())] == [instance.id]
    assert len(configured.query(instance_store.EVAL_SLOTS_COLLECTION, Filter())) == 1
    assert all(store.query(instance_store.EVAL_INSTANCES_COLLECTION, Filter()) == []
               for store in router.namespaces.values() if store.path.exists())


def test_an_eval_instance_id_keeps_the_evals_namespace():
    for root in (REGISTRY_EVAL, LOCAL_EVAL, HUB_EVAL):
        made = new_eval_instance_id(root)
        assert made.startswith(f"{root}-") and len(made) == len(root) + 9
    assert new_eval_instance_id(LOCAL_EVAL) != new_eval_instance_id(LOCAL_EVAL)


def test_a_record_written_before_later_fields_still_loads():
    instance = EvalInstance.from_dict({
        "id": "evals/smoke-abcd1234", "source": {"kind": "eval", "eval_id": "evals/smoke"},
        "runner": {"type": "local"}, "created_at": "2026-11-27T10:00:00+00:00"})
    assert (instance.status, instance.settings, instance.lineage, instance.slots_rev) == (
        EvalInstanceStatus.PLANNING, EvalSettings(), Lineage(), 0)
    slot = Slot.from_dict({"eval_instance_id": instance.id, "task_id": "t", "task_version": 1, "replica": 1})
    assert (slot.id, slot.state, slot.attempts, slot.result) == ("t@1#1", SlotState.PLANNED, [], None)


def test_an_eval_instance_records_what_its_runner_declared():
    declared = RunnerDeclarations.of(LocalRunner())
    assert declared == RunnerDeclarations(type="local", runs_outlive_submitter=False, retries_infra=False)
    assert RunnerDeclarations.from_dict(declared.to_dict()) == declared


def test_each_source_kind_round_trips():
    for source in (
        EvalSource(kind="eval", eval_id="evals/smoke", eval_version=3),
        EvalSource(kind="bundle", ref="./bench", content_hash="sha256:ab", eval_id=LOCAL_EVAL, eval_version=1),
        EvalSource(kind="tasks", tasks=[{"task_id": "tasks/a", "task_version": 2}]),
    ):
        assert EvalSource.from_dict(source.to_dict()) == source
        assert isinstance(source.kind, SourceKind)


def _instance(root) -> EvalInstance:
    return EvalInstance(id=new_eval_instance_id(root), source=EvalSource(kind="eval", eval_id=root),
                        settings=EvalSettings(k=2), runner=RunnerDeclarations(type="local"),
                        created_at="2026-11-27T10:00:00+00:00")
