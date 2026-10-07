"""LoadArtifactTaskStep with `sandbox_name` naming a container-mode deploy_sandbox sandbox: files land in its container
the way an agent's container is loaded, not on a VM host the sandbox does not have."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.providers.sandbox_providers import sandbox_provider as sp_mod
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

UNIVERSE_ID = "bundler-inputs"


@pytest.fixture
def universe(monkeypatch):
    uni = FileArtifactUniverse(id=UNIVERSE_ID, version=2, file_artifact_refs={})
    staged = {
        "task.json": FileArtifact(
            id="fa-1", version=1, description="task", filename="task.json",
            content_type="application/json", s3_url="s3://bucket/task.json",
        ),
        "kb/INDEX.md": FileArtifact(
            id="fa-2", version=1, description="index", filename="INDEX.md",
            content_type="text/markdown", s3_url="s3://bucket/INDEX.md",
        ),
    }
    monkeypatch.setattr(type(uni), "get_file_artifacts", lambda self: staged)
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: uni))
    return uni


def _context(sandbox, monkeypatch):
    async def _get_sandbox(sandbox_id):
        return sandbox

    monkeypatch.setattr(
        sp_mod, "get_sandbox_provider",
        lambda: type("P", (), {"get_sandbox": staticmethod(_get_sandbox)})(),
    )
    ctx = TaskStepContext()
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="bundler", sandbox_id="sb-1", sandbox_mode="container")]
    return ctx


def _container_sandbox(*, vm_backed):
    sandbox = AsyncMock(spec=VmSandbox) if vm_backed else AsyncMock()
    sandbox.mode = "container"
    sandbox.container_name = "agent-local-1"
    return sandbox


@pytest.mark.asyncio
async def test_a_container_sandbox_is_written_directly(universe, monkeypatch):
    sandbox = _container_sandbox(vm_backed=False)
    step = LoadArtifactTaskStep(
        id="load", version=None, sandbox_name="bundler", artifact_id=UNIVERSE_ID, destination_path="/work/inputs",
    )

    ctx = await step.execute(_context(sandbox, monkeypatch))

    assert sorted(c.args for c in sandbox.exec.await_args_list) == [
        ("mkdir", "-p", "/work/inputs"), ("mkdir", "-p", "/work/inputs/kb"),
    ]
    assert sorted(c.args for c in sandbox.write_file_from_object.await_args_list) == [
        ("s3://bucket/INDEX.md", "/work/inputs/kb/INDEX.md"),
        ("s3://bucket/task.json", "/work/inputs/task.json"),
    ]
    sandbox.exec_script.assert_not_awaited()
    sandbox.load_object_file.assert_not_awaited()
    (loaded,) = ctx.metadata["loaded_file_artifact_universes"]
    assert loaded["sandbox_name"] == "bundler"
    assert loaded["destination_path"] == "/work/inputs"
    assert sorted(loaded["files"]) == ["kb/INDEX.md", "task.json"]


@pytest.mark.asyncio
async def test_a_container_on_a_vm_backed_provider_is_loaded_through_its_container(universe, monkeypatch):
    sandbox = _container_sandbox(vm_backed=True)
    step = LoadArtifactTaskStep(
        id="load", version=None, sandbox_name="bundler", artifact_id=UNIVERSE_ID, destination_path="/work/inputs",
    )

    await step.execute(_context(sandbox, monkeypatch))

    scripts = [c.args[0] for c in sandbox.exec_script.await_args_list]
    assert scripts and all(s.startswith("docker exec -u 0 agent-local-1 mkdir -p ") for s in scripts)
    assert len(sandbox.write_file_from_object.await_args_list) == 2
    sandbox.load_object_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_urls_go_into_a_container_sandbox(monkeypatch):
    sandbox = _container_sandbox(vm_backed=False)
    step = LoadArtifactTaskStep(
        id="load", version=None, sandbox_name="bundler", urls=["https://example.com/data.zip"],
        destination_path="/work/inputs",
    )

    await step.execute(_context(sandbox, monkeypatch))

    sandbox.write_file_from_url.assert_awaited_once_with("https://example.com/data.zip", "/work/inputs/data.zip")
    sandbox.exec_script.assert_not_awaited()
