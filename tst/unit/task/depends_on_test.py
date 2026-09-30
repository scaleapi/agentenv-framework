"""A step's ``depends_on``: each entry a step id or ``{"task_step_id": <step id>}``, what else is refused, the
form a step is stored in, and the edges a task checks and runs its steps by."""

import asyncio
import re

import pytest

from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.store import Filter
from agent_env.task import Task
from agent_env.task.store import get_task_store
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency, dependencies


class _Step(TaskStep):
    """Logs when it starts and ends."""

    type = "depends_on_test"
    entity_refs = ()
    log: list[str] = []

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        _Step.log.append(f"start {self.id}")
        await asyncio.sleep(0.02)
        _Step.log.append(f"end {self.id}")
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Step.type: _Step}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Step, "log", [])


def _step(id, depends_on=None):
    return _Step(id=id, version=None, depends_on=depends_on)


@pytest.mark.parametrize("entry", ["a", {"task_step_id": "a"}, TaskStepDependency("a")])
def test_an_entry_is_a_step_id_or_its_object_form(entry):
    assert dependencies([entry]) == [TaskStepDependency("a")]
    assert _step("b", [entry]).depends_on == _Step.from_dict({"id": "b", "depends_on": [entry]}).depends_on == [
        TaskStepDependency("a")]


@pytest.mark.parametrize(("depends_on", "problem"), [
    ("a", "depends_on is a list of step ids, not 'a'"),
    ({"task_step_id": "a"}, "depends_on is a list of step ids, not {'task_step_id': 'a'}"),
    ([7], 'a depends_on entry is a step id or {"task_step_id": "<step id>"}, not 7'),
    ([{"id": "a"}], 'a depends_on entry is a step id or {"task_step_id": "<step id>"}, not {\'id\': \'a\'}'),
    ([{"task_step_id": 7}], 'a depends_on entry is a step id or {"task_step_id": "<step id>"}, not '
     "{'task_step_id': 7}"),
])
def test_anything_else_is_refused_saying_what_it_takes(depends_on, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        _step("b", depends_on)
    with pytest.raises(ValueError, match=re.escape(problem)):
        _Step.from_dict({"id": "b", "depends_on": depends_on})


def test_a_step_id_is_stored_in_the_object_form_and_reads_back(local_stores):
    steps = [_step("a"), _Step.from_dict({"id": "b", "depends_on": ["a"]}), _step("c", ["a", {"task_step_id": "b"}])]
    get_task_store().put_document(Task(id="t", version=None, steps=steps))

    stored = get_config().get_document_store().find_one("tasks", Filter.of(id="t"))
    assert [step.get("depends_on") for step in stored["steps"]] == [
        None, [{"task_step_id": "a"}], [{"task_step_id": "a"}, {"task_step_id": "b"}]]
    assert Task.get("t").steps[2].depends_on == [TaskStepDependency("a"), TaskStepDependency("b")]


@pytest.mark.parametrize(("steps", "problem"), [
    ([_step("ask", ["deploy"]), _step("deploy")],
     "Step 'ask' at position 0 declares depends_on 'deploy', which doesn't come before it (forward refs are not "
     "allowed: a step depends only on the steps before it)"),
    ([_step("deploy"), _step("ask", ["ask"])],
     "Step 'ask' at position 1 declares depends_on 'ask', which doesn't come before it (forward refs are not "
     "allowed: a step depends only on the steps before it)"),
    ([_step("deploy"), _step("ask", ["deplyo"])],
     "Step 'ask' at position 1 declares depends_on 'deplyo', which names no step in this task (steps before it: "
     "['deploy'])"),
], ids=["later", "itself", "unknown"])
def test_a_step_depends_only_on_a_step_before_it_and_a_typo_is_told_apart_from_a_forward_ref(steps, problem):
    with pytest.raises(ValueError) as caught:
        Task(id="t", version=None, steps=steps)
    assert str(caught.value) == problem


@pytest.mark.asyncio
async def test_bare_and_object_entries_gate_the_scheduler_alike(local_stores):
    task = Task(id="t", version=None, steps=[
        _step("a"), _step("b", ["a"]), _step("c", [{"task_step_id": "a"}]), _step("d", ["b", {"task_step_id": "c"}]),
    ])

    await task.run(context=TaskStepContext())

    assert _Step.log[:2] == ["start a", "end a"]
    assert sorted(_Step.log[2:4]) == ["start b", "start c"]
    assert sorted(_Step.log[4:6]) == ["end b", "end c"]
    assert _Step.log[6:] == ["start d", "end d"]
