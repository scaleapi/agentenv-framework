"""`container_name` is VmSandbox-only, so `LoadArtifactTaskStep(agent_name=...)` must resolve the staging container by subtype, not by `sandbox.mode`.
An environment of several files is staged file by file at their paths, by the same subtype."""

from __future__ import annotations

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.providers.sandbox_providers import sandbox as sandbox_mod
from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER, SANDBOX_MODE_VM
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import load_artifact as mod
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

ARTIFACT_ID = "filesystem-payload"


class _ModalLikeSandbox(Sandbox):
    type = "modal"

    def __init__(self) -> None:
        self.mode = SANDBOX_MODE_CONTAINER
        self.sandbox_id = "sb-01M07Z0EFQGE0Z8KA3WGRTQK7X"
        self.tunnel_urls = {}
        self.vnc_url = None

    async def terminate(self) -> None:  # pragma: no cover - never called here
        pass


class _LocalLikeSandbox(VmSandbox):
    type = "local"

    def __init__(self) -> None:
        self.mode = SANDBOX_MODE_CONTAINER
        self.sandbox_id = "local-1"
        self.tunnel_urls = {}
        self.vnc_url = None

    @property
    def container_name(self) -> str:
        return f"agent-{self.sandbox_id}"

    async def terminate(self) -> None:  # pragma: no cover - never called here
        pass


@pytest.fixture
def artifact(monkeypatch):
    art = EnvironmentArtifact(
        id=ARTIFACT_ID, version=1, description="fs payload",
        environment_name="filesystem",
    )
    monkeypatch.setattr(
        type(art), "get_file_artifact",
        lambda self: FileArtifact(
            id="fa-1", version=1, description="payload", filename="data.json",
            content_type="application/json", s3_url="s3://bucket/data.json",
        ),
    )
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: art))
    return art


@pytest.fixture
def staged(monkeypatch):
    seen: list[dict] = []

    async def _fake_stage(sandbox, container_name, environment_artifact, destination):
        seen.append({
            "sandbox": sandbox, "container": container_name, "destination": destination,
        })
        return ["notes/readme.txt"]

    monkeypatch.setattr(mod, "_stage_environment_payload_into_container", _fake_stage)
    return seen


def _context_with_agent(
    monkeypatch, sandbox, installed_container: str | None = None, on_host: bool = False,
) -> TaskStepContext:
    async def _get_sandbox(sandbox_id):
        return sandbox

    provider = type("P", (), {"get_sandbox": staticmethod(_get_sandbox)})()

    from agent_env.providers.sandbox_providers import sandbox_provider as sp_mod

    monkeypatch.setattr(sp_mod, "build_sandbox_provider", lambda _type: provider)
    monkeypatch.setattr(sp_mod, "get_agent_sandbox_provider", lambda: provider)

    ctx = TaskStepContext()
    ctx.deployed_agents = [
        DeployedAgent(
            agent_name="solver", api_url="http://agent", sandbox_id=sandbox.sandbox_id,
            sandbox_type=sandbox.type, installed_container=installed_container, on_host=on_host,
        )
    ]
    return ctx


@pytest.mark.asyncio
async def test_container_mode_sandbox_without_container_name_does_not_crash(
    monkeypatch, artifact, staged
):
    sandbox = _ModalLikeSandbox()
    assert not hasattr(sandbox, "container_name")

    ctx = _context_with_agent(monkeypatch, sandbox)
    step = LoadArtifactTaskStep(
        id="stage", version=None, agent_name="solver",
        artifact_id=ARTIFACT_ID, destination_path="/app/files",
    )
    ctx = await step.execute(ctx)

    assert staged == [{"sandbox": sandbox, "container": None, "destination": "/app/files"}]
    entry = ctx.metadata["loaded_environment_artifacts"][0]
    assert entry["agent_name"] == "solver"
    assert entry["file_count"] == 1


@pytest.mark.asyncio
async def test_vm_backed_agent_container_keeps_its_per_sandbox_name(
    monkeypatch, artifact, staged
):
    sandbox = _LocalLikeSandbox()
    assert sandbox.mode != SANDBOX_MODE_VM
    assert isinstance(sandbox, VmSandbox)

    ctx = _context_with_agent(monkeypatch, sandbox)
    step = LoadArtifactTaskStep(
        id="stage", version=None, agent_name="solver",
        artifact_id=ARTIFACT_ID, destination_path="/app/files",
    )
    await step.execute(ctx)

    assert staged[0]["container"] == "agent-local-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("installed_in,on_host,expected", [
    (None, False, "agent-api"), ("agent-api", False, "agent-api"), ("task", False, "task"), (None, True, None)])
async def test_plain_vm_agent_gets_the_default_container_name_the_one_it_was_installed_in_or_its_host(
    monkeypatch, artifact, staged, installed_in, on_host, expected
):
    class _Vm(VmSandbox):
        type = "platform_vm"

        def __init__(self) -> None:
            self.mode = SANDBOX_MODE_VM
            self.sandbox_id = "vm-1"
            self.tunnel_urls = {}
            self.vnc_url = None

        async def terminate(self) -> None:  # pragma: no cover - never called here
            pass

    ctx = _context_with_agent(monkeypatch, _Vm(), installed_in, on_host)
    step = LoadArtifactTaskStep(
        id="stage", version=None, agent_name="solver", artifact_id=ARTIFACT_ID,
    )
    await step.execute(ctx)

    assert staged[0]["container"] == expected
    assert staged[0]["destination"] == "/app/files"


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox, container, on_host", [
    (_ModalLikeSandbox(), None, False),
    (_LocalLikeSandbox(), "agent-local-1", False),
    (_LocalLikeSandbox(), None, True),
], ids=["container-sandbox", "vm-container", "vm-host"])
async def test_an_environment_of_several_files_is_staged_file_by_file_at_their_paths(monkeypatch, sandbox, container,
                                                                                    on_host):
    calls = []

    async def stage(*args, **kwargs):
        calls.append((args, kwargs))
        return {"inbox/a.json": "/app/files/inbox/a.json", "sent.json": "/app/files/sent.json"}

    monkeypatch.setattr(sandbox_mod, "stage_files_into_container", stage)
    files = {"inbox/a.json": object(), "sent.json": object()}
    monkeypatch.setattr(EnvironmentArtifact, "get_file_artifacts", lambda self: files)
    mail = EnvironmentArtifact(id="mail", version=1, environment_name="mail", file_artifact_refs={
        "inbox/a.json": ArtifactRef(id="fa-a", version=1), "sent.json": ArtifactRef(id="fa-b", version=1)})

    staged = await mod._stage_environment_payload_into_container(sandbox, container, mail, "/app/files")

    assert staged == ["inbox/a.json", "sent.json"]
    assert calls == [((sandbox, files, "/app/files"), {"container": container, "on_host": on_host})]
