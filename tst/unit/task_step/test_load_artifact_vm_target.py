"""LoadArtifactTaskStep with `sandbox_name` naming a VM directly: previously reachable only alongside `container_name`, so files could never land on a VM host."""

from __future__ import annotations

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

UNIVERSE_ID = "harbor-bundle-universe"


@pytest.fixture
def universe(monkeypatch):
    uni = FileArtifactUniverse(id=UNIVERSE_ID, version=3, file_artifact_refs={})
    staged = {
        "instruction.md": FileArtifact(
            id="fa-1", version=1, description="instruction", filename="instruction.md",
            content_type="text/markdown", s3_url="s3://bucket/instruction.md",
        ),
        "tests/test.sh": FileArtifact(
            id="fa-2", version=1, description="verifier", filename="test.sh",
            content_type="text/x-shellscript", s3_url="s3://bucket/test.sh",
        ),
    }
    monkeypatch.setattr(type(uni), "get_file_artifacts", lambda self: staged)
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: uni))
    return uni


@pytest.fixture
def vm(monkeypatch):
    """A named VM sandbox in context, with every call against it recorded."""
    calls: list[tuple] = []

    class _Vm:
        def scoped_name(self, name):
            return name

        async def exec_script(self, script, **kw):
            calls.append(("exec", script))
            return ""

        async def load_object_file(self, object_url, destination_path):
            calls.append(("object", object_url, destination_path))

        async def load_url_file(self, url, destination_path):  # pragma: no cover
            # Deliberately present: the step must NOT depend on a sandbox method.
            calls.append(("url", url, destination_path))

        async def write_file_from_url(self, url, destination_path):  # pragma: no cover
            calls.append(("container_url", url, destination_path))

        async def docker_cp(self, source, destination, *, remove_source=False):
            calls.append(("cp", source, destination, remove_source))

    from agent_env.providers.sandbox_providers import sandbox_provider as sp_mod

    async def _get_sandbox(sandbox_id):
        return _Vm()

    monkeypatch.setattr(
        sp_mod, "get_sandbox_provider",
        lambda: type("P", (), {"get_sandbox": staticmethod(_get_sandbox)})(),
    )

    ctx = TaskStepContext()
    ctx.deployed_sandboxes = [
        DeployedSandbox(sandbox_name="mk", sandbox_id="sb-1", sandbox_mode="vm")
    ]
    return ctx, calls


class TestConstructorValidation:
    def test_sandbox_name_alone_is_now_accepted(self):
        LoadArtifactTaskStep(
            id="s", version=None, sandbox_name="mk", artifact_id=UNIVERSE_ID,
        )

    def test_container_name_still_requires_sandbox_name(self):
        with pytest.raises(ValueError, match="`container_name` requires `sandbox_name`"):
            LoadArtifactTaskStep(
                id="s", version=None, container_name="c", artifact_id=UNIVERSE_ID,
            )

    def test_no_target_at_all_is_still_rejected(self):
        with pytest.raises(ValueError, match="Must pass at least one of"):
            LoadArtifactTaskStep(id="s", version=None, artifact_id=UNIVERSE_ID)

    def test_urls_accept_a_bare_sandbox(self):
        LoadArtifactTaskStep(
            id="s", version=None, sandbox_name="mk", urls=["https://x/y.tar.gz"],
        )


class TestUniverseOntoVm:
    @pytest.mark.asyncio
    async def test_files_land_on_the_vm_with_no_docker(self, universe, vm):
        ctx, calls = vm
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk",
            artifact_id=UNIVERSE_ID, destination_path="/app/bundle",
        )
        ctx = await step.execute(ctx)

        assert not any("docker" in c[1] for c in calls if c[0] == "exec"), calls
        # Pulled straight from the object store to the final path — no VM temp file, no copy inward.
        objects = [c for c in calls if c[0] == "object"]
        assert objects == [
            ("object", "s3://bucket/instruction.md", "/app/bundle/instruction.md"),
            ("object", "s3://bucket/test.sh", "/app/bundle/tests/test.sh"),
        ]
        assert ("exec", "mkdir -p /app/bundle") in calls
        assert ("exec", "mkdir -p /app/bundle/tests") in calls

    @pytest.mark.asyncio
    async def test_it_is_recorded_against_the_sandbox_not_an_agent(self, universe, vm):
        ctx, _ = vm
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk",
            artifact_id=UNIVERSE_ID, destination_path="/app/bundle",
        )
        ctx = await step.execute(ctx)

        entry = ctx.metadata["loaded_file_artifact_universes"][0]
        assert entry["sandbox_name"] == "mk"
        assert entry["container_name"] is None
        assert entry["agent_name"] is None
        assert entry["destination_path"] == "/app/bundle"

    @pytest.mark.asyncio
    async def test_no_deployed_agents_needed(self, universe, vm):
        """The point of this path: staging can happen before install_agent runs."""
        ctx, _ = vm
        assert ctx.deployed_agents == []
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk", artifact_id=UNIVERSE_ID,
        )
        await step.execute(ctx)  # must not raise

    @pytest.mark.asyncio
    async def test_unknown_sandbox_name_is_a_clear_error(self, universe, vm):
        ctx, _ = vm
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="nope", artifact_id=UNIVERSE_ID,
        )
        with pytest.raises(RuntimeError, match="Sandbox 'nope' not found"):
            await step.execute(ctx)


class TestUrlsOntoVm:
    @pytest.mark.asyncio
    async def test_urls_use_the_host_downloader(self, vm):
        ctx, calls = vm
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk",
            urls=["https://example.com/repo.tar.gz"], destination_path="/app/seeds",
        )
        ctx = await step.execute(ctx)

        # Curled onto the host by the step's own helper — no sandbox method, no container writer.
        execs = [c[1] for c in calls if c[0] == "exec"]
        assert "mkdir -p /app/seeds" in execs, execs
        curls = [e for e in execs if e.startswith("curl -fsSL")]
        assert len(curls) == 1 and curls[0].endswith("-o /app/seeds/repo.tar.gz"), curls
        assert not [c for c in calls if c[0] == "url"]
        assert not [c for c in calls if c[0] == "container_url"]
        assert ctx.metadata["loaded_urls"][0]["sandbox_name"] == "mk"

    @pytest.mark.asyncio
    async def test_a_named_url_is_curled_to_its_filename(self, vm):
        ctx, calls = vm
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk", destination_path="/app/seeds",
            urls=[{"url": "https://files.example/objects/obj-4f9c2a", "filename": "repo.tar.gz"}],
        )
        ctx = await step.execute(ctx)

        curls = [c[1] for c in calls if c[0] == "exec" and c[1].startswith("curl -fsSL")]
        assert len(curls) == 1, curls
        assert "https://files.example/objects/obj-4f9c2a" in curls[0] and curls[0].endswith("-o /app/seeds/repo.tar.gz")
        assert ctx.metadata["loaded_urls"][0]["files"] == ["repo.tar.gz"]


class TestContainerPathUnchanged:
    @pytest.mark.asyncio
    async def test_container_name_still_copies_inward(self, universe, vm, monkeypatch):
        """With a container named, behaviour is exactly as before."""
        ctx, _ = vm
        ctx.metadata["deployed_docker_containers"] = [
            {"container_name": "task-container", "sandbox_name": "mk"}
        ]
        seen: list[dict] = []

        from agent_env.task_step.task_steps import load_artifact as mod

        async def _fake_into_container(sandbox, container_name, uni, destination):
            seen.append({"container": container_name, "destination": destination})
            return ["instruction.md"]

        monkeypatch.setattr(mod, "_load_universe_into_container", _fake_into_container)

        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk", container_name="task-container",
            artifact_id=UNIVERSE_ID, destination_path="/loaded",
        )
        ctx = await step.execute(ctx)

        assert seen == [{"container": "task-container", "destination": "/loaded"}]
        assert ctx.metadata["loaded_file_artifact_universes"][0]["container_name"] == "task-container"

    @pytest.mark.asyncio
    async def test_urls_go_into_the_named_container_not_the_agents(self, vm):
        ctx, calls = vm
        ctx.metadata["deployed_docker_containers"] = [{"container_name": "task-container", "sandbox_name": "mk"}]
        step = LoadArtifactTaskStep(
            id="stage", version=None, sandbox_name="mk", container_name="task-container",
            urls=["https://example.com/data.csv"], destination_path="/work",
        )

        await step.execute(ctx)

        [curl] = [c[1] for c in calls if c[0] == "exec" and c[1].startswith("curl ")]
        assert ("cp", curl.rsplit(" -o ", 1)[1], "task-container:/work/data.csv", True) in calls
        assert ("exec", "docker exec -u 0 task-container mkdir -p /work") in calls
        assert not [c for c in calls if c[0] == "container_url"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_a_failed_url_load_reports_its_own_error_and_cleans_up(self, cleanup_fails):
        scripts = []

        class _Vm:
            async def exec_script(self, script, **kw):
                scripts.append(script)
                if script.startswith("curl "):
                    raise RuntimeError("curl: (22) 404")
                if script.startswith("rm -f ") and cleanup_fails:
                    raise RuntimeError("exec transport closed")
                return ""

        step = LoadArtifactTaskStep(id="s", version=None, sandbox_name="mk", container_name="c", urls=["https://x/y"])

        with pytest.raises(RuntimeError, match="404"):
            await step._load_url_into_container(_Vm(), "c", "https://x/y", "/work/y")

        assert scripts[-1].startswith("rm -f /tmp/_load_url_")


class TestUrlHelper:
    @pytest.mark.asyncio
    async def test_load_url_onto_vm_curls_straight_to_the_path(self):
        """The helper lives on the step, not the sandbox — it has one caller."""
        scripts: list[str] = []

        class _Vm:
            async def exec_script(self, script, **kw):
                scripts.append(script)
                return ""

        step = LoadArtifactTaskStep(
            id="s", version=None, sandbox_name="mk", urls=["https://x/y"],
        )
        await step._load_url_onto_vm(_Vm(), "https://example.com/repo.tar.gz", "/app/seeds/repo.tar.gz")

        assert not any("docker" in s for s in scripts), scripts
        assert "mkdir -p /app/seeds" in scripts, scripts
        curls = [s for s in scripts if s.startswith("curl -fsSL")]
        assert len(curls) == 1 and curls[0].endswith("-o /app/seeds/repo.tar.gz"), curls
        # retry policy comes from the sandbox layer, not re-declared here
        assert "--retry 5" in curls[0], curls
