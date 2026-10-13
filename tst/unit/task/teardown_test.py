"""``teardown_run``: which sandboxes a finished run's context gives it, through which provider, what it does
when a terminate fails, hangs or is cancelled, and which local work folders it removes."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, remove_local_work_dir
from agent_env.task import teardown
from agent_env.task.teardown import RecordedSandbox, TeardownReport, recorded_sandboxes, teardown_run
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps import teardown_sandboxes
from agent_env.task_step.task_steps.teardown_sandboxes import TORN_DOWN_KEY


class _FakeSandbox:
    ON_THIS_MACHINE = False

    def __init__(self, provider, sandbox_id):
        self.provider, self.sandbox_id = provider, sandbox_id

    async def terminate(self):
        behavior = self.provider.log.behaviors.get(self.sandbox_id)
        if behavior == "raise":
            raise RuntimeError("boom")
        if behavior == "hang":
            await asyncio.Event().wait()
        self.provider.log.append((self.provider.name, self.sandbox_id))


class _FakeProvider:
    def __init__(self, name, log):
        self.name, self.log = name, log

    async def get_sandbox(self, sandbox_id):
        if self.log.behaviors.get(sandbox_id) == "gone":
            raise RuntimeError(f"no sandbox {sandbox_id}")
        return _FakeSandbox(self, sandbox_id)


class _Log(list):
    """The (provider, sandbox_id) terminate log, and how given sandboxes misbehave."""

    def __init__(self):
        super().__init__()
        self.behaviors: dict[str, str] = {}


@pytest.fixture
def terminated(monkeypatch):
    log = _Log()

    def build(name):
        if name not in {"modal", "modal_vm"}:
            raise ValueError(f"unknown backend {name}")
        return _FakeProvider(name, log)

    monkeypatch.setattr(teardown, "build_sandbox_provider", build)
    monkeypatch.setattr(teardown_sandboxes, "build_sandbox_provider", build)
    monkeypatch.setattr(teardown, "_DEFAULT_PROVIDERS", {
        slot: (lambda slot=slot: _FakeProvider(f"default-{slot}", log)) for slot in ("agent", "env", "sandbox")
    })
    return log


def _context() -> TaskStepContext:
    return TaskStepContext(
        deployed_sandboxes=[DeployedSandbox(sandbox_name="box", sandbox_id="sb-box", sandbox_mode="vm")],
        deployed_envs=[
            DeployedGatewayEnv(
                env_id="shop", env_version=1, gateway_url="http://g", mcp_url="http://m", sandbox_id="sb-gw",
                sandbox_type="modal", sandbox_ids={"service_db": {"orders": "sb-db"}, "modal_vm": "sb-vm"},
            ),
            DeployedEnv(env_id="hosted", env_version=1, mcp_url="http://hosted"),
        ],
        deployed_agents=[
            DeployedAgent(agent_name="solver", api_url="http://a", sandbox_id="sb-solver", sandbox_type="modal"),
            DeployedAgent(agent_name="on-box", api_url="http://b", sandbox_id="sb-box", sandbox_type="modal"),
            DeployedAgent(agent_name="human", api_url="http://h"),
        ],
    )


def _run(context, **kwargs) -> TeardownReport:
    return asyncio.run(teardown_run(context, **kwargs))


def test_every_recorded_sandbox_is_found_once_and_attributed_to_what_made_it():
    assert recorded_sandboxes(_context()) == (
        RecordedSandbox("sb-box", None, "sandbox"),
        RecordedSandbox("sb-gw", "modal", "env"),
        RecordedSandbox("sb-db", "modal", "env"),
        RecordedSandbox("sb-vm", "modal_vm", "env"),
        RecordedSandbox("sb-solver", "modal", "agent"),
    )


def test_each_sandbox_is_terminated_through_its_backend_and_recorded_so_a_second_call_does_nothing(terminated):
    context = _context()

    report = _run(context)

    assert sorted(terminated) == [("default-sandbox", "sb-box"), ("modal", "sb-db"), ("modal", "sb-gw"),
                                  ("modal", "sb-solver"), ("modal_vm", "sb-vm")]
    assert {sandbox.sandbox_id for sandbox in report.terminated} == {"sb-box", "sb-db", "sb-gw", "sb-solver", "sb-vm"}
    assert report.still_up == ()
    assert sorted(context.metadata[TORN_DOWN_KEY]) == ["sb-box", "sb-db", "sb-gw", "sb-solver", "sb-vm"]

    terminated.clear()
    assert _run(context) == TeardownReport()
    assert terminated == []


def test_a_sandbox_a_step_already_tore_down_is_not_terminated_again(terminated):
    context = _context()
    context.metadata[TORN_DOWN_KEY] = ["sb-gw", "sb-db", "sb-vm"]

    report = _run(context)

    assert sorted(sandbox_id for _, sandbox_id in terminated) == ["sb-box", "sb-solver"]
    assert len(report.terminated) == 2


@pytest.mark.parametrize("behavior, why", [
    ("raise", "RuntimeError: boom"),
    ("gone", "RuntimeError: no sandbox sb-gw"),
    ("hang", "timed out after 0.05s"),
])
def test_a_terminate_that_fails_is_reported_not_raised_and_the_others_still_go(terminated, behavior, why):
    terminated.behaviors["sb-gw"] = behavior
    context = _context()

    report = _run(context, timeout=0.05)

    assert report.failed == ((RecordedSandbox("sb-gw", "modal", "env"), why),)
    assert report.still_up == (RecordedSandbox("sb-gw", "modal", "env"),)
    assert len(report.terminated) == 4
    assert "sb-gw" not in context.metadata[TORN_DOWN_KEY]


def test_cancelling_it_returns_what_it_reached_and_leaves_the_rest(terminated):
    terminated.behaviors["sb-gw"] = "hang"
    context = _context()

    async def cancelled_mid_teardown():
        running = asyncio.ensure_future(teardown_run(context))
        while len(terminated) < 4:
            await asyncio.sleep(0.01)
        running.cancel()
        return await running

    report = asyncio.run(cancelled_mid_teardown())

    assert report.left == (RecordedSandbox("sb-gw", "modal", "env"),)
    assert len(report.terminated) == 4
    assert TeardownReport.skipped(context).left == report.left


def test_a_skipped_teardown_leaves_everything_not_yet_torn_down():
    context = _context()
    context.metadata[TORN_DOWN_KEY] = ["sb-gw"]

    assert [sandbox.sandbox_id for sandbox in TeardownReport.skipped(context).left] == [
        "sb-box", "sb-db", "sb-vm", "sb-solver"]


@pytest.fixture
def sandbox_root(tmp_path, monkeypatch):
    root = tmp_path / "sandboxes"
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    return root


def test_a_local_sandbox_is_brought_down_and_its_work_folder_removed(sandbox_root):
    sandbox = LocalSandbox()
    (sandbox.work_dir / "greeting").mkdir()
    (sandbox.work_dir / "greeting/hello.txt").write_text("hello")
    context = TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm", sandbox_type="local")])

    report = _run(context)

    assert report.terminated == (RecordedSandbox(sandbox.sandbox_id, "local", "sandbox"),)
    assert list(sandbox_root.iterdir()) == []


def test_a_reattached_local_agent_has_its_container_and_work_folder_removed(sandbox_root, monkeypatch):
    agent = LocalSandbox()
    (agent.work_dir / ".agent-container-mode").write_text(agent.container_name)
    scripts = []

    async def record(self, script, *, max_retries=0):
        scripts.append(script)
        return ""

    monkeypatch.setattr(LocalSandbox, "exec_script", record)
    context = TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name="solver", api_url="http://agent", sandbox_id=agent.sandbox_id, sandbox_type="local")])

    report = _run(context)

    assert report.terminated == (RecordedSandbox(agent.sandbox_id, "local", "agent"),)
    assert scripts == [f"docker rm -f {agent.container_name} >/dev/null 2>&1 || true"]
    assert list(sandbox_root.iterdir()) == []


def test_a_slow_folder_removal_does_not_count_against_the_terminate_timeout(sandbox_root, monkeypatch):
    sandbox = LocalSandbox()
    context = TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm", sandbox_type="local")])
    removed = []

    def slow_remove(sandbox_id):
        time.sleep(0.3)
        removed.append(sandbox_id)

    async def terminate(self):
        pass

    monkeypatch.setattr(LocalSandbox, "terminate", terminate)
    monkeypatch.setattr(teardown, "remove_local_work_dir", slow_remove)

    report = _run(context, timeout=0.1)

    assert report.terminated == (RecordedSandbox(sandbox.sandbox_id, "local", "sandbox"),)
    assert removed == [sandbox.sandbox_id]


def test_a_local_sandbox_a_step_already_tore_down_still_has_its_folder_removed(sandbox_root):
    sandbox = LocalSandbox()
    context = TaskStepContext(
        deployed_sandboxes=[DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm",
                                            sandbox_type="local")],
        metadata={TORN_DOWN_KEY: [sandbox.sandbox_id]},
    )

    assert _run(context) == TeardownReport()
    assert list(sandbox_root.iterdir()) == []


def test_only_a_real_folder_directly_under_the_sandbox_root_is_removed(sandbox_root, tmp_path):
    kept = LocalSandbox()
    legacy = tmp_path / "tmp" / "agent-env-local-legacy1-abc"
    legacy.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "precious.txt").write_text("keep me")
    os.symlink(elsewhere, sandbox_root / "agent-env-local-linked1-abc")

    assert remove_local_work_dir("local-legacy1") is None
    assert remove_local_work_dir("local-linked1") is None
    assert remove_local_work_dir("local-missing") is None
    assert legacy.is_dir()
    assert (elsewhere / "precious.txt").read_text() == "keep me"
    assert remove_local_work_dir(kept.sandbox_id) == kept.work_dir
    assert not kept.work_dir.exists()


def test_a_work_folder_that_cant_be_removed_is_reported_not_swallowed(sandbox_root, monkeypatch):
    sandbox = LocalSandbox()
    context = TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm", sandbox_type="local")])

    def refuse(sandbox_id):
        raise PermissionError("owned by root")

    monkeypatch.setattr(teardown, "remove_local_work_dir", refuse)

    report = _run(context)

    assert report.terminated == ()
    assert report.failed == ((RecordedSandbox(sandbox.sandbox_id, "local", "sandbox"),
                              "terminated, but its work folder wasn't removed: owned by root"),)
    assert context.metadata[TORN_DOWN_KEY] == [sandbox.sandbox_id]
