"""The step-journal suite on DynamoDbDocumentStore under moto, plus concurrent completions."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from moto import mock_aws

from agent_env.config import set_document_store
from agent_env.store import DynamoDbDocumentStore
from agent_env.task.store import TaskInstanceStore, set_task_instance_store
from tst.unit.task import step_journal_test as journal
from tst.unit.task.step_journal_test import *  # noqa: F403


@pytest.fixture
def store():
    with mock_aws():
        set_document_store(DynamoDbDocumentStore(table_prefix="test_", region="us-west-2"))
        s = TaskInstanceStore()
        set_task_instance_store(s)
        try:
            yield s
        finally:
            set_task_instance_store(None)


def test_concurrent_completions_all_journal_and_an_undo_replays_the_survivors(store, monkeypatch):
    iid = journal._new(store, total=8, seed_md={"base": 1})
    docs = store._doc_store
    read = docs._candidates

    def read_slowly(*args):
        found = read(*args)
        time.sleep(0.005)
        return found

    monkeypatch.setattr(docs, "_candidates", read_slowly)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: journal._record(store, iid, f"s{i}", journal._ops({}, {f"k{i}": i}), total=8), range(8)))
    assert sorted(e["step_id"] for e in journal._journal(store, iid)) == ["__seed__", *(f"s{i}" for i in range(8))]
    assert journal._replay(store, iid) == journal._doc(store, iid)["context"]
    after = store.undo_steps_sync(iid, {"s0"})
    assert after["context"]["metadata"] == {"base": 1, **{f"k{i}": i for i in range(1, 8)}}
