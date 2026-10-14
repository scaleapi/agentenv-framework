"""A step's ``when`` and ``if_dependency_skipped``: what they take, how the run's seed decides a skip, how a
skip carries to the steps that name it in ``depends_on``, and what a skipped step leaves behind."""

import pytest

from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.store import Filter
from agent_env.task import Task
from agent_env.task.store import TaskStepStatus, find_task_instance, get_task_store
from agent_env.task.task import substitute_seed_in_steps
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, evaluate_when, record_skip, skip_reason


class _Step(TaskStep):
    """Records that it ran."""

    type = "when_test"
    entity_refs = ()
    ran: list[str] = []

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        _Step.ran.append(self.id)
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Step.type: _Step}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Step, "ran", [])


def _step(id, depends_on=None, when=None, fail_task_on_error=True, if_dependency_skipped="skip"):
    return _Step(id=id, version=None, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                 when=when, if_dependency_skipped=if_dependency_skipped)


def _task(*steps):
    return Task(id="t", version=None, steps=list(steps))


def _statuses(ctx):
    return {s.step_id: s.status for s in find_task_instance(ctx.instance_id).completed_steps}


@pytest.mark.parametrize("when", [None, True, "true", "TRUE", " yes ", "on", "1"])
def test_these_run(when):
    assert evaluate_when(when) is True


@pytest.mark.parametrize("when", [False, "false", "False", "no", "off", "0", "", "  "])
def test_these_skip(when):
    assert evaluate_when(when) is False


def test_a_placeholder_the_seed_never_filled_is_an_error_naming_it():
    with pytest.raises(ValueError, match="still holds a placeholder"):
        evaluate_when("<needs_heavy_env>")


def test_anything_else_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="is not a boolean"):
        evaluate_when("maybe")


def test_when_is_a_bool_or_a_string():
    with pytest.raises(ValueError, match="when is a bool or a string, not 3"):
        _step("a", when=3)


def test_if_dependency_skipped_is_skip_or_run():
    with pytest.raises(ValueError, match="if_dependency_skipped is 'skip' or 'run', not 'always'"):
        _step("a", if_dependency_skipped="always")


@pytest.mark.asyncio
async def test_a_skip_carries_down_the_chain_that_names_it(local_stores):
    task = _task(_step("a"), _step("b", ["a"], when="false"), _step("c", ["b"]), _step("d", ["c"]))
    ctx = TaskStepContext()

    await task.run(context=ctx)

    assert _Step.ran == ["a"]
    assert _statuses(ctx) == {
        "a": TaskStepStatus.SUCCESS, "b": TaskStepStatus.SKIPPED,
        "c": TaskStepStatus.SKIPPED, "d": TaskStepStatus.SKIPPED}
    assert ctx.metadata["skipped_steps"] == [
        {"step_id": "b", "reason": "when='false'"},
        {"step_id": "c", "reason": "dependency 'b' was skipped"},
        {"step_id": "d", "reason": "dependency 'c' was skipped"},
    ]
    assert find_task_instance(ctx.instance_id).status == "completed"
    assert "failed_steps" not in ctx.metadata


@pytest.mark.asyncio
async def test_one_skipped_dependency_is_enough(local_stores):
    task = _task(_step("a", []), _step("b", [], when="false"), _step("c", ["a", "b"]))

    await task.run(context=TaskStepContext())

    assert _Step.ran == ["a"]


@pytest.mark.asyncio
async def test_a_join_that_runs_after_a_skipped_dependency(local_stores):
    task = _task(
        _step("deploy-heavy-env", [], when="false"),
        _step("init-heavy-env", ["deploy-heavy-env"]),
        _step("prepare", []),
        _step("create", ["prepare", "init-heavy-env"], if_dependency_skipped="run"),
        _step("collect", ["create"]),
    )
    ctx = TaskStepContext()

    await task.run(context=ctx)

    assert sorted(_Step.ran) == ["collect", "create", "prepare"]
    assert _statuses(ctx)["init-heavy-env"] == TaskStepStatus.SKIPPED


@pytest.mark.asyncio
async def test_implicit_ordering_does_not_carry_a_skip(local_stores):
    task = _task(_step("a", when="false"), _step("b"))

    await task.run(context=TaskStepContext())

    assert _Step.ran == ["b"]


@pytest.mark.asyncio
async def test_a_step_that_opts_to_run_still_obeys_its_own_when(local_stores):
    task = _task(_step("a", [], when="false"), _step("b", ["a"], when="false", if_dependency_skipped="run"))

    await task.run(context=TaskStepContext())

    assert _Step.ran == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "ran"), [("true", ["a", "b"]), ("false", [])])
async def test_the_runs_seed_decides(local_stores, value, ran):
    task = _task(_step("a", when="<needs_heavy_env>"), _step("b", ["a"]))
    ctx = TaskStepContext()
    ctx.metadata["seed"] = {"needs_heavy_env": value}

    await task.run(context=ctx)

    assert _Step.ran == ran


@pytest.mark.asyncio
async def test_a_run_without_the_seed_value_fails_the_step_instead_of_guessing(local_stores):
    task = _task(_step("a", when="<needs_heavy_env>"), _step("b", ["a"]))
    ctx = TaskStepContext()
    ctx.metadata["seed"] = {"brief": "unrelated"}

    with pytest.raises(ValueError, match="still holds a placeholder"):
        await task.run(context=ctx)

    assert _Step.ran == []


@pytest.mark.asyncio
async def test_a_tolerant_step_with_a_bad_when_fails_tolerantly(local_stores):
    task = _task(_step("a", when="maybe", fail_task_on_error=False), _step("b", ["a"]))
    ctx = TaskStepContext()

    await task.run(context=ctx)

    assert _Step.ran == ["b"]
    assert [f["step_id"] for f in ctx.metadata["failed_steps"]] == ["a"]


def test_a_resumed_run_or_a_one_step_runner_reads_earlier_skips_from_the_context():
    ctx = TaskStepContext()
    ctx.metadata["skipped_steps"] = [{"step_id": "a", "reason": "when='false'"}]

    assert skip_reason(_step("b", ["a"]), ctx) == "dependency 'a' was skipped"
    assert skip_reason(_step("b", ["a"], if_dependency_skipped="run"), ctx) is None
    assert skip_reason(_step("c", ["x"]), ctx) is None


def test_both_fields_round_trip_through_the_task_store_and_seed_substitution(local_stores):
    get_task_store().put_document(_task(
        _step("a", when="<needs_heavy_env>"), _step("b", ["a"], if_dependency_skipped="run"), _step("c")))

    stored = get_config().get_document_store().find_one("tasks", Filter.of(id="t"))
    assert [(s.get("when"), s.get("if_dependency_skipped")) for s in stored["steps"]] == [
        ("<needs_heavy_env>", None), (None, "run"), (None, None)]
    steps = Task.get("t").steps
    assert (steps[0].when, steps[1].if_dependency_skipped, steps[2].if_dependency_skipped) == (
        "<needs_heavy_env>", "run", "skip")
    rendered = substitute_seed_in_steps(steps, {"needs_heavy_env": "no"})
    assert [(s.when, s.if_dependency_skipped) for s in rendered] == [("no", "skip"), (None, "run"), (None, "skip")]


def test_put_keeps_both_fields(local_stores):
    saved = _Step.put(id="a", when="<needs_heavy_env>", if_dependency_skipped="run")

    loaded = TaskStep.get("a")
    assert (saved.when, saved.if_dependency_skipped) == (loaded.when, loaded.if_dependency_skipped) == (
        "<needs_heavy_env>", "run")


def test_recording_the_same_skip_twice_keeps_one_entry():
    ctx = TaskStepContext()
    step = _step("a", when="false")

    record_skip(step, ctx, "when='false'")
    record_skip(step, ctx, "when='false'")

    assert ctx.metadata["skipped_steps"] == [{"step_id": "a", "reason": "when='false'"}]
