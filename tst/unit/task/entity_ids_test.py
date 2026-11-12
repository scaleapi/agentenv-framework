"""Task.entity_ids: the ids a bundle gave what a task names, stored with the task and put in each run's metadata."""

import pytest

from agent_env.config.runtime import Config
from agent_env.task import Task
from agent_env.task_step.context import ENTITY_IDS_KEY, TaskStepContext, run_entity_id
from agent_env.task_step.task_step import TaskStep

IDS = {"world": "@local/~/runs/pn/world"}


class _Seen(TaskStep):
    """Records the metadata its run reached it with."""

    type = "entity_ids_test"
    entity_refs = ()
    seen: list[dict] = []

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        _Seen.seen.append(dict(context.metadata))
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Seen.type: _Seen}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Seen, "seen", [])


def _task(**kw) -> Task:
    return Task(id="t", version=None, steps=[_Seen(id="s", version=None)], **kw)


def test_they_are_stored_with_the_task_and_left_out_when_there_are_none():
    assert Task.from_dict(_task(entity_ids=IDS).to_dict()).entity_ids == IDS
    assert "entity_ids" not in _task().to_dict()
    assert "entity_ids" not in _task(entity_ids={}).to_dict()


@pytest.mark.asyncio
async def test_a_run_puts_them_in_its_metadata_for_run_entity_id(local_stores):
    await _task(entity_ids=IDS).run(context=TaskStepContext())

    metadata = _Seen.seen[0]
    assert metadata[ENTITY_IDS_KEY] == IDS
    assert run_entity_id(metadata, "world") == "@local/~/runs/pn/world"
    assert run_entity_id(metadata, "elsewhere") == "elsewhere"


@pytest.mark.asyncio
async def test_a_task_stored_as_written_adds_nothing(local_stores):
    await _task().run(context=TaskStepContext())

    assert ENTITY_IDS_KEY not in _Seen.seen[0]
    assert run_entity_id(_Seen.seen[0], "world") == "world"
