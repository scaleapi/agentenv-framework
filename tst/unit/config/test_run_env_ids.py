"""A run's ``gateway_env_id`` / ``service_db_env_id`` overrides name the infra envs its deploys use, only within it."""

from __future__ import annotations

import asyncio

import pytest

from agent_env.config import gateway_env_id, get_config, service_db_env_id, set_document_store
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext


class _Probe:
    type = "probe"
    id, depends_on, retry_config, fail_task_on_error = "probe", None, None, True

    def __init__(self, both_running: asyncio.Barrier):
        self.both_running, self.seen = both_running, None

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        await self.both_running.wait()
        self.seen = (gateway_env_id(get_config()), service_db_env_id(get_config()))
        return context


@pytest.mark.asyncio
async def test_concurrent_runs_each_use_their_own_infra_envs_and_only_within_the_run(tmp_path):
    set_document_store(LocalSqliteDocumentStore(str(tmp_path / "runs.db")))
    both_running = asyncio.Barrier(2)
    probes = {name: _Probe(both_running) for name in ("a", "b")}

    await asyncio.gather(*(Task(id=f"t-{name}", version=1, steps=[probe]).run(context=TaskStepContext(metadata={
        "user_overrides": {"gateway_env_id": f"gw-{name}", "service_db_env_id": f"db-{name}"}}))
        for name, probe in probes.items()))

    assert {name: probe.seen for name, probe in probes.items()} == {"a": ("gw-a", "db-a"), "b": ("gw-b", "db-b")}
    assert gateway_env_id(get_config()) == get_config().default_gateway_env_id
