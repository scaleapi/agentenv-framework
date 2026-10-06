"""Which container collect_artifacts and verify_sandbox reach into, per kind of sandbox.

A container sandbox on a container provider (Modal) is the runtime itself; one on a VM-backed provider (local, modal_vm,
e2b) runs beside its host, so its files are only reachable through `docker exec <container>`. A VM-backed provider
reattaches every sandbox as a VM, so a deploy_sandbox sandbox's mode is the one deploy_sandbox recorded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.sandbox_providers import sandbox_provider as sp_mod
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.verifiers import verify_sandbox
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep

CONTAINER = "agent-local-1"


def _sandbox(kind: str):
    """`vm`: a VM with an agent container; `vm-backed-container`: local/modal_vm-style; `container`: Modal-style."""
    sandbox = AsyncMock(spec=VmSandbox) if kind != "container" else AsyncMock()
    sandbox.sandbox_id = "sb-1"
    sandbox.mode = "vm" if kind == "vm" else "container"
    sandbox.container_name = CONTAINER
    sandbox.exec_with_output = AsyncMock(return_value=(0, f"{CONTAINER}\n", ""))
    return sandbox


def _provide(monkeypatch, sandbox):
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    for module in (sp_mod, verify_sandbox):  # verify_sandbox binds the getters at import
        monkeypatch.setattr(module, "get_agent_sandbox_provider", lambda: provider)
        monkeypatch.setattr(module, "get_sandbox_provider", lambda: provider)


def _agent_context():
    ctx = TaskStepContext()
    ctx.deployed_agents = [DeployedAgent(agent_name="solver", api_url="http://x", sandbox_id="sb-1")]
    return ctx


def _sandbox_context(recorded_mode: str):
    ctx = TaskStepContext()
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode=recorded_mode)]
    return ctx


async def _collect_container(monkeypatch, step, ctx):
    seen = {}

    async def _items(self, provider, sandbox, container, *rest):
        seen["container"] = container
        return {}, {}, []

    monkeypatch.setattr(CollectArtifactsTaskStep, "_collect_items", _items)
    # Nothing is collected from the stub, so the step then fails; the container it chose is already recorded.
    with pytest.raises(RuntimeError, match="produced no file_artifact_universe"):
        await step.execute(ctx)
    return seen["container"]


async def _verify_container(monkeypatch, step, ctx):
    seen = {}

    async def _eval(self, sandbox, container, criterion):
        seen["container"] = container
        return {"passed": True, "score": 1.0, "justification": ""}

    monkeypatch.setattr(VerifySandboxTaskStep, "_eval_criterion", _eval)
    await step.execute(ctx)
    return seen["container"]


_CRITERIA = [{"type": "probe_file_exists", "criterion": "c", "paths": ["out.txt"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,expected", [("vm", CONTAINER), ("container", None)])
async def test_collect_reaches_an_agents_container(monkeypatch, kind, expected):
    _provide(monkeypatch, _sandbox(kind))
    step = CollectArtifactsTaskStep(id="c", version=None, agent_name="solver", artifact_paths=["out.txt"])
    assert await _collect_container(monkeypatch, step, _agent_context()) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,recorded,expected", [
    ("vm", "vm", None),
    ("vm", "container", CONTAINER),  # a VM-backed provider reattaches a container sandbox as a VM
    ("vm-backed-container", "container", CONTAINER),
    ("container", "container", None),
])
async def test_collect_reaches_a_named_sandboxs_container(monkeypatch, kind, recorded, expected):
    _provide(monkeypatch, _sandbox(kind))
    step = CollectArtifactsTaskStep(id="c", version=None, sandbox_name="box", artifact_paths=["out.txt"])
    assert await _collect_container(monkeypatch, step, _sandbox_context(recorded)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,expected", [("vm", CONTAINER), ("container", None)])
async def test_verify_sandbox_probes_an_agents_container(monkeypatch, kind, expected):
    _provide(monkeypatch, _sandbox(kind))
    step = VerifySandboxTaskStep(id="v", version=None, agent_name="solver", criteria=_CRITERIA)
    assert await _verify_container(monkeypatch, step, _agent_context()) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,recorded,expected", [
    ("vm", "vm", None),
    ("vm", "container", CONTAINER),
    ("vm-backed-container", "container", CONTAINER),
    ("container", "container", None),
])
async def test_verify_sandbox_probes_a_named_sandboxs_container(monkeypatch, kind, recorded, expected):
    _provide(monkeypatch, _sandbox(kind))
    step = VerifySandboxTaskStep(id="v", version=None, sandbox_name="box", criteria=_CRITERIA)
    assert await _verify_container(monkeypatch, step, _sandbox_context(recorded)) == expected
