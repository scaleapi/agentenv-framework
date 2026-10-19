"""The ``teardown_sandboxes`` step: which sandboxes it terminates, through which provider, and what
it records."""

from __future__ import annotations

import copy

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.context_ops import build_context_update_ops
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import teardown_sandboxes
from agent_env.task_step.task_steps.teardown_sandboxes import TORN_DOWN_KEY, TeardownSandboxesTaskStep

_BACKENDS = {"modal", "modal_vm"}


class _FakeSandbox:
    def __init__(self, provider, sandbox_id):
        self.provider, self.sandbox_id = provider, sandbox_id

    async def terminate(self):
        if self.sandbox_id in self.provider.failing:
            raise RuntimeError("boom")
        self.provider.terminated.append((self.provider.name, self.sandbox_id))


class _FakeProvider:
    def __init__(self, name, terminated, failing):
        self.name, self.terminated, self.failing = name, terminated, failing

    async def get_sandbox(self, sandbox_id):
        return _FakeSandbox(self, sandbox_id)

    async def terminate_sandbox(self, sandbox_id):
        await (await self.get_sandbox(sandbox_id)).terminate()

    async def close(self):
        pass


class _Log(list):
    """The (provider, sandbox_id) terminate log, plus the sandbox ids whose terminate should fail."""

    def __init__(self):
        super().__init__()
        self.failing: set[str] = set()


@pytest.fixture
def terminated(monkeypatch):
    """Route every provider lookup through fakes; returns the terminate log."""
    log = _Log()
    failing = log.failing

    def build(name):
        if name not in _BACKENDS:
            raise ValueError(f"unknown backend {name}")
        return _FakeProvider(name, log, failing)

    monkeypatch.setattr(teardown_sandboxes, "build_sandbox_provider", build)
    monkeypatch.setattr(teardown_sandboxes, "_DEFAULT_PROVIDERS", {
        slot: (lambda slot=slot: _FakeProvider(f"default-{slot}", log, failing))
        for slot in ("agent", "env", "sandbox")
    })
    return log


def _context() -> TaskStepContext:
    return TaskStepContext(
        deployed_agents=[
            DeployedAgent(agent_name="solver", api_url="http://a", sandbox_id="sb-solver", sandbox_type="modal"),
            DeployedAgent(agent_name="judge", api_url="http://j", sandbox_id="sb-judge", sandbox_type="modal"),
            DeployedAgent(agent_name="human", api_url="http://h"),
        ],
        deployed_sandboxes=[
            DeployedSandbox(sandbox_name="gpu-box", sandbox_id="sb-gpu", sandbox_mode="container"),
        ],
        deployed_envs=[
            DeployedGatewayEnv(
                env_id="shop", env_version=1, gateway_url=None, mcp_url="http://m", db_web_url=None,
                sandbox_id="sb-gw", sandbox_type="modal",
                sandbox_ids={"gateway": "sb-gw", "service_db": {"orders": "sb-db"}, "modal_vm": "sb-vm"},
            ),
        ],
    )


def _step(**targets):
    return TeardownSandboxesTaskStep(id="teardown", version=1, **targets)


# --- definition --------------------------------------------------------------------------------


def test_registered_and_round_trips():
    assert get_task_step_registry().get("teardown_sandboxes") is TeardownSandboxesTaskStep
    step = _step(agent_names=["solver"], env_ids=["shop"], sandbox_names=["gpu-box"])
    data = step.to_dict()
    assert data["type"] == "teardown_sandboxes"
    again = TeardownSandboxesTaskStep.from_dict(data)
    assert (again.agent_names, again.env_ids, again.sandbox_names) == (["solver"], ["shop"], ["gpu-box"])


def test_best_effort_by_default_including_from_dict():
    assert _step(agent_names=["solver"]).fail_task_on_error is False
    assert TeardownSandboxesTaskStep.from_dict(
        {"id": "t", "version": 1, "agent_names": ["solver"]}
    ).fail_task_on_error is False


def test_fail_task_on_error_is_rejected():
    with pytest.raises(ValueError, match="best-effort"):
        TeardownSandboxesTaskStep(id="t", version=1, agent_names=["solver"], fail_task_on_error=True)
    with pytest.raises(ValueError, match="best-effort"):
        TeardownSandboxesTaskStep.from_dict(
            {"id": "t", "version": 1, "agent_names": ["solver"], "fail_task_on_error": True}
        )


def test_requires_a_target():
    with pytest.raises(ValueError):
        _step()
    with pytest.raises(ValueError):
        TeardownSandboxesTaskStep.from_dict({"id": "t", "version": 1})


# --- execute -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminates_only_the_named_agent(terminated):
    context = await _step(agent_names=["solver"]).execute(_context())
    assert terminated == [("modal", "sb-solver")]
    assert context.metadata[TORN_DOWN_KEY] == ["sb-solver"]


@pytest.mark.asyncio
async def test_an_env_tears_down_every_sandbox_behind_it_on_its_own_backend(terminated):
    await _step(env_ids=["shop"]).execute(_context())
    assert sorted(terminated) == [
        ("modal", "sb-db"),         # nested {service: id} inherits the env's backend
        ("modal", "sb-gw"),         # primary and "gateway" entry are one sandbox
        ("modal_vm", "sb-vm"),      # keyed by a backend name: that backend, not the env's
    ]


@pytest.mark.asyncio
async def test_a_typed_sandbox_is_terminated_by_id_without_a_handle(terminated, monkeypatch):
    async def no_handle(self, sandbox_id):
        raise AssertionError(f"asked {self.name} for a handle to {sandbox_id}")

    async def by_id(self, sandbox_id):
        self.terminated.append((self.name, sandbox_id))

    monkeypatch.setattr(_FakeProvider, "get_sandbox", no_handle)
    monkeypatch.setattr(_FakeProvider, "terminate_sandbox", by_id)
    await _step(agent_names=["solver"]).execute(_context())
    assert terminated == [("modal", "sb-solver")]


@pytest.mark.asyncio
async def test_an_untyped_sandbox_uses_its_slot_default(terminated):
    await _step(sandbox_names=["gpu-box"]).execute(_context())
    assert terminated == [("default-sandbox", "sb-gpu")]


@pytest.mark.asyncio
async def test_an_env_outside_our_sandboxes_is_left_to_its_own_lifetime(terminated):
    context = _context()
    context.deployed_envs.append(DeployedEnv(env_id="hosted", env_version=1, env_provider_type="hosted"))
    await _step(env_ids=["hosted"]).execute(context)
    assert terminated == []


@pytest.mark.asyncio
async def test_a_sandboxless_agent_is_skipped(terminated):
    context = await _step(agent_names=["human"]).execute(_context())
    assert terminated == []
    assert TORN_DOWN_KEY not in context.metadata


@pytest.mark.asyncio
async def test_rerunning_skips_sandboxes_already_torn_down(terminated):
    step = _step(agent_names=["solver", "judge"])
    context = await step.execute(_context())
    terminated.clear()
    context = await step.execute(context)
    assert terminated == []
    assert sorted(context.metadata[TORN_DOWN_KEY]) == ["sb-judge", "sb-solver"]


@pytest.mark.asyncio
async def test_missing_targets_are_skipped_not_raised(terminated, caplog):
    context = await _step(agent_names=["solver", "ghost"], env_ids=["nope"]).execute(_context())
    assert terminated == [("modal", "sb-solver")]
    assert "agent 'ghost'" in caplog.text and "env 'nope'" in caplog.text
    assert context.metadata[TORN_DOWN_KEY] == ["sb-solver"]


@pytest.mark.asyncio
async def test_a_failed_terminate_is_logged_and_the_rest_still_go(terminated):
    terminated.failing.add("sb-solver")
    context = await _step(agent_names=["solver", "judge"]).execute(_context())
    assert terminated == [("modal", "sb-judge")]
    assert context.metadata[TORN_DOWN_KEY] == ["sb-judge"]


@pytest.mark.asyncio
async def test_a_rerun_over_rolled_back_context_retries_without_raising(terminated):
    """A retry of an earlier step can restore the context from before this step ran, so a re-run
    meets sandboxes it already terminated but no longer has recorded; a lookup that now fails for
    them is logged, not raised, and the rest still go."""
    step = _step(agent_names=["solver", "judge"])
    await step.execute(_context())
    terminated.clear()
    terminated.failing.add("sb-solver")  # e.g. the provider can't find a terminated sandbox
    context = await step.execute(_context())
    assert terminated == [("modal", "sb-judge")]
    assert context.metadata[TORN_DOWN_KEY] == ["sb-judge"]


@pytest.mark.asyncio
async def test_the_record_survives_the_context_diff(terminated):
    pre = _context()
    post = await _step(agent_names=["solver"]).execute(copy.deepcopy(pre))
    ops = build_context_update_ops(pre, post)
    assert ops.add_to_sets.get(f"context.metadata.{TORN_DOWN_KEY}") == ["sb-solver"]
