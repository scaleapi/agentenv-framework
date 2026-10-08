"""The recorded-run reads on the plugin surface: task_instances, count_task_instances and find_task_instance."""

from __future__ import annotations

import pytest

from agent_env.config import set_document_store
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION,
    TaskInstance,
    TaskStepStatus,
    count_task_instances,
    find_task_instance,
    register_task_instance,
    task_instances,
)

TASK = "@local/mycorp-demo/triage/route-ticket"


@pytest.fixture
def docs(tmp_path) -> LocalSqliteDocumentStore:
    store = LocalSqliteDocumentStore(str(tmp_path / "runs.db"))
    set_document_store(store)
    return store


def _run(docs, instance_id: str, created: str, *, task_id: str = TASK, version: int = 1, status: str = "completed"):
    docs.insert(TASK_INSTANCES_COLLECTION, {
        "instance_id": instance_id, "task_id": task_id, "task_version": version, "status": status,
        "current_step": 1, "total_steps": 1, "created_at_utc": created,
        "completed_steps": [{"step_id": "grade", "status": "success"}],
        "context": {"metadata": {"verifications": {"grade": {"score": 1.0}}}},
    })


def test_a_tasks_runs_come_newest_first_and_only_that_tasks(docs):
    _run(docs, "r1", "2026-10-08 09:00 UTC")
    _run(docs, "r3", "2026-10-08 11:00 UTC")
    _run(docs, "r2", "2026-10-08 10:00 UTC")
    _run(docs, "other", "2026-10-08 12:00 UTC", task_id=f"{TASK}-v2")

    runs = task_instances(TASK)

    assert [r.instance_id for r in runs] == ["r3", "r2", "r1"]
    assert all(isinstance(r, TaskInstance) for r in runs)
    assert runs[0].completed_steps[0].status is TaskStepStatus.SUCCESS
    assert runs[0].context["metadata"]["verifications"]["grade"]["score"] == 1.0
    assert count_task_instances(TASK) == 3


def test_a_version_filter_and_a_page(docs):
    for minute in range(5):
        _run(docs, f"v1-{minute}", f"2026-10-08 10:0{minute} UTC")
        _run(docs, f"v2-{minute}", f"2026-10-08 10:0{minute} UTC", version=2)

    assert [r.instance_id for r in task_instances(TASK, task_version=2, limit=2, offset=1)] == ["v2-3", "v2-2"]
    assert count_task_instances(TASK, task_version=2) == 5
    assert count_task_instances(TASK) == 10
    assert task_instances(TASK, task_version=3) == []


def test_one_run_by_id_or_none(docs):
    _run(docs, "r1", "2026-10-08 09:00 UTC")

    assert find_task_instance("r1").task_id == TASK
    assert find_task_instance("missing") is None
    assert task_instances("@local/mycorp-demo/triage/nothing") == []
    assert count_task_instances("@local/mycorp-demo/triage/nothing") == 0


def test_a_registered_run_reads_back(docs):
    registered = register_task_instance(TASK, 4, total_steps=3)

    found = find_task_instance(registered.instance_id)

    assert (found.task_id, found.task_version, found.status, found.total_steps) == (TASK, 4, "running", 3)
    assert [r.instance_id for r in task_instances(TASK)] == [registered.instance_id]


def test_runs_of_one_minute_page_in_descending_instance_id_order_without_overlap(docs):
    for instance_id in ("c", "a", "d", "b"):
        _run(docs, instance_id, "2026-10-08 10:00 UTC")
    _run(docs, "later", "2026-10-08 10:01 UTC")

    pages = [task_instances(TASK, limit=2, offset=offset) for offset in (0, 2, 4)]

    assert [[r.instance_id for r in page] for page in pages] == [["later", "d"], ["c", "b"], ["a"]]
