"""A step sees the attempt epoch it was dispatched under, which a rolled-back and retried span moves on, so the usage
it records says which attempt spent it."""

import pytest

from agent_env.config.runtime import Config
from agent_env.task import Task
from agent_env.task_step.context import ATTEMPT_EPOCH, TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep


class _Step(TaskStep):
    """Records the epoch it runs under; fails its first run when ``flaky``."""

    type = "attempt_epoch_test"
    entity_refs = ()
    seen: list[tuple[str, int | None]] = []
    failed: set[str] = set()

    def __init__(self, *, flaky: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.flaky = flaky

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        _Step.seen.append((self.id, ATTEMPT_EPOCH.get()))
        if self.flaky and self.id not in _Step.failed:
            _Step.failed.add(self.id)
            raise RuntimeError("first attempt fails")
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Step.type: _Step}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Step, "seen", [])
    monkeypatch.setattr(_Step, "failed", set())


@pytest.mark.asyncio
async def test_a_retried_span_runs_under_the_next_epoch(local_stores):
    task = Task(id="t", version=None, steps=[
        _Step(id="a", version=None),
        _Step(id="b", version=None, flaky=True, depends_on=["a"],
              retry_config=RetryConfig(retry_from_step_id="b", max_retries=1)),
    ])

    await task.run(context=TaskStepContext())

    assert _Step.seen == [("a", 0), ("b", 0), ("b", 1)]
    assert ATTEMPT_EPOCH.get() is None
