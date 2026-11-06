from __future__ import annotations

import asyncio
import threading

import pytest

from agent_env.config import set_document_store
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task import teardown
from agent_env.task.store import TaskInstanceStore, set_task_instance_store
from agent_env.task.task import Task
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


@pytest.fixture
def store(tmp_path):
    set_document_store(LocalSqliteDocumentStore(str(tmp_path / "instances.db")))
    s = TaskInstanceStore()
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


class _Sandbox:
    mode, type, tunnel_urls, vnc_url, network_policy = "vm", "fake", {}, None, NetworkPolicy()

    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id


class _Provider:
    ON_THIS_MACHINE = False

    def __init__(self):
        self.created: list[str] = []
        self.terminated: list[str] = []

    async def create_vm(self, **_):
        self.created.append(f"sb-{len(self.created) + 1}")
        return _Sandbox(self.created[-1])

    async def terminate_sandbox(self, sandbox_id: str):
        self.terminated.append(sandbox_id)

    async def close(self):
        pass


class _FailsFirstAttemptOnceReady(TaskStep):
    type = "flaky"

    def __init__(self, *args, ready: asyncio.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready, self.attempts = ready, 0

    async def execute(self, context):
        self.attempts += 1
        if self.attempts == 1:
            await self.ready.wait()
            raise RuntimeError("flaky")
        return context


@pytest.mark.asyncio
async def test_a_sibling_cancelled_during_its_completion_write_is_still_torn_down(store, monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    monkeypatch.setattr(teardown, "build_sandbox_provider", lambda _: provider)
    loop = asyncio.get_running_loop()
    deploy_writing = asyncio.Event()
    undo_landed, deploy_write_done = threading.Event(), threading.Event()
    real_complete, real_undo = store.record_step_complete_sync, store.undo_steps_sync
    held = []

    def complete_after_the_undo(instance_id, entry, *args):
        if entry["step_id"] != "deploy" or held:
            return real_complete(instance_id, entry, *args)
        held.append(entry)
        loop.call_soon_threadsafe(deploy_writing.set)
        undo_landed.wait(10)
        try:
            return real_complete(instance_id, entry, *args)
        finally:
            deploy_write_done.set()

    def undo_then_release(*args, **kwargs):
        try:
            return real_undo(*args, **kwargs)
        finally:
            undo_landed.set()

    monkeypatch.setattr(store, "record_step_complete_sync", complete_after_the_undo)
    monkeypatch.setattr(store, "undo_steps_sync", undo_then_release)
    flaky = _FailsFirstAttemptOnceReady("flaky", 1, depends_on=["a"], ready=deploy_writing,
                            retry_config=RetryConfig(retry_from_step_id="flaky"))
    survivor = DeploySandboxTaskStep(id="a", version=1, sandbox_name="base", sandbox_mode="vm")
    deploy = DeploySandboxTaskStep(id="deploy", version=1, sandbox_name="box", sandbox_mode="vm", depends_on=["a"])

    context = await Task(id="t", version=1, steps=[survivor, flaky, deploy]).run(context=TaskStepContext())
    assert deploy_write_done.wait(10)

    assert (flaky.attempts, provider.created) == (2, ["sb-1", "sb-2", "sb-3"])
    [reset] = context.metadata["retry_resets"]
    assert reset["rolled_back_sandboxes"] == {
        "deployed_envs": [], "deployed_agents": [], "deployed_sandboxes": [{"sandbox_id": "sb-2", "sandbox_type": "fake"}],
    }
    report = await teardown_run(context)
    assert sorted(provider.terminated) == ["sb-1", "sb-2", "sb-3"]
    assert report.still_up == ()
