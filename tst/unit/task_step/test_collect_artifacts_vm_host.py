"""CollectArtifactsTaskStep with `sandbox_name` naming a VM directly: the agent path discovers `agent-api` and raises when it is absent, which is every host-mode install."""

from __future__ import annotations

import base64

import pytest

from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep

FILES = {"caa_10/MODIFICATIONS.md": b"# what changed\n", "caa_10/alignment_x/task.toml": b"schema=1\n"}


class _Stdout:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self):
        return self._data


class _Proc:
    def __init__(self, data: bytes):
        self.stdout = _Stdout(data)
        self.stderr = _Stdout(b"")

    async def wait(self):
        return 0


class _VmSandbox:
    """A VM-mode sandbox that serves the fake tree, recording every command."""

    mode = "vm"

    def __init__(self):
        self.calls: list[tuple] = []

    async def exec_with_output(self, *args):
        self.calls.append(args)
        # the liveness probe (`echo ok`) is exempt; anything touching files is not
        if args[0] != "echo":
            assert args[0] == "sudo", f"VM-host command must run as root: {args}"
        if "find" in args:
            if "-maxdepth" in args:
                return 0, "caa_10\n", ""
            return 0, "".join(f"{p}\n" for p in FILES), ""
        if "stat" in args:
            path = args[-1]
            rel = path.split("/app/artifact/", 1)[-1]
            return (0, str(len(FILES[rel])), "") if rel in FILES else (-1, "", "no such file")
        return 0, "", ""

    async def exec(self, *args):
        self.calls.append(args)
        assert args[0] == "sudo", f"VM-host command must run as root: {args}"
        bash_cmd = args[-1]
        path = bash_cmd.split("base64 < ", 1)[1].strip().strip("'")
        rel = path.split("/app/artifact/", 1)[-1]
        return _Proc(base64.b64encode(FILES[rel]))


@pytest.fixture
def vm(monkeypatch):
    sandbox = _VmSandbox()
    from agent_env.providers.sandbox_providers import sandbox_provider as sp_mod

    async def _get_sandbox(sandbox_id):
        return sandbox

    class _Provider:
        get_sandbox = staticmethod(_get_sandbox)

        async def close(self):
            return None

    monkeypatch.setattr(sp_mod, "get_sandbox_provider", lambda: _Provider())

    uploaded: list[str] = []

    class _Store:
        """Stands in for the real artifact store; S3 and Mongo are both unreachable in tests."""

        def put_object_file(self, *, object_name, **kw):
            uploaded.append(object_name)
            return f"s3://bucket/{object_name}"

        def next_version(self, _id):
            return 1

        def put_document(self, _doc):
            return None

    from agent_env.artifact import store as artifact_store_mod

    monkeypatch.setattr(artifact_store_mod, "get_artifact_store", lambda: _Store())

    ctx = TaskStepContext()
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="caa", sandbox_id="sb-1", sandbox_mode="vm")]
    return ctx, sandbox, uploaded


class TestConstructorValidation:
    def test_sandbox_name_alone_is_accepted(self):
        CollectArtifactsTaskStep(id="c", version=None, sandbox_name="caa", base_path="/app/artifact")

    def test_sandbox_name_with_explicit_agent_is_rejected(self):
        with pytest.raises(ValueError, match="exclusive with"):
            CollectArtifactsTaskStep(
                id="c", version=None, sandbox_name="caa", agent_name="creator",
            )

    def test_sandbox_name_with_env_id_is_rejected(self):
        with pytest.raises(ValueError, match="exclusive with"):
            CollectArtifactsTaskStep(id="c", version=None, sandbox_name="caa", env_id="cua")

    def test_container_name_still_requires_sandbox_name(self):
        with pytest.raises(ValueError, match="`container_name` requires `sandbox_name`"):
            CollectArtifactsTaskStep(id="c", version=None, container_name="task-container")

    def test_default_agent_name_does_not_count_as_explicit(self):
        """agent_name defaults to 'default-agent'; that must not block the VM path."""
        step = CollectArtifactsTaskStep(id="c", version=None, sandbox_name="caa")
        assert step.sandbox_name == "caa"


class TestNoDockerOnTheVmPath:
    @pytest.mark.asyncio
    async def test_enumeration_and_reads_use_no_container(self, vm):
        ctx, sandbox, _ = vm
        step = CollectArtifactsTaskStep(
            id="collect", version=None, sandbox_name="caa", base_path="/app/artifact",
        )
        await step.execute(ctx)

        flat = [" ".join(str(a) for a in c) for c in sandbox.calls]
        assert not any("docker" in c for c in flat), flat
        # sudo'd but not container-wrapped: host-mode agents write as uid 0
        assert all(c.startswith(("sudo ", "echo ")) for c in flat), flat
        assert any(c.startswith("sudo find /app/artifact") for c in flat), flat
        assert any(c.startswith("sudo bash -c base64 < ") for c in flat), flat

    @pytest.mark.asyncio
    async def test_it_does_not_need_a_deployed_agent(self, vm):
        """The agent path raises 'No agent container found'; this one must not care."""
        ctx, _, _ = vm
        assert ctx.deployed_agents == []
        step = CollectArtifactsTaskStep(
            id="collect", version=None, sandbox_name="caa", base_path="/app/artifact",
        )
        await step.execute(ctx)  # must not raise

    @pytest.mark.asyncio
    async def test_unknown_sandbox_name_is_a_clear_error(self, vm):
        ctx, _, _ = vm
        step = CollectArtifactsTaskStep(
            id="collect", version=None, sandbox_name="nope", base_path="/app/artifact",
        )
        with pytest.raises(RuntimeError, match="Sandbox 'nope' not found"):
            await step.execute(ctx)

    @pytest.mark.asyncio
    async def test_every_file_under_base_path_is_collected(self, vm):
        ctx, _, _ = vm
        step = CollectArtifactsTaskStep(
            id="collect", version=None, sandbox_name="caa", base_path="/app/artifact",
        )
        ctx = await step.execute(ctx)
        entry = ctx.metadata["collected_artifacts"]["collect"]
        assert set(entry["artifacts"]) == set(FILES), entry["artifacts"]
        # the legacy flat key mirrors it
        assert set(ctx.metadata["artifacts"]) == set(FILES)


class TestUniverseIdOverride:
    """Without a suffix every collect in a run shares one artifact id."""

    @pytest.mark.asyncio
    async def test_suffix_names_this_step_s_universe(self, vm, monkeypatch):
        ctx, _, _ = vm
        ctx.metadata["universe_id"] = "caa_105"
        seen = []
        from agent_env.artifact.artifacts import file_artifact_universe as fau
        monkeypatch.setattr(fau.FileArtifactUniverse, "put",
                            classmethod(lambda cls, id, file_artifacts: seen.append(id) or
                                        cls(id=id, version=1, file_artifact_refs={})))
        await CollectArtifactsTaskStep(
            id="collect-records", version=None, sandbox_name="caa",
            base_path="/app/artifact", universe_id_suffix="-records",
        ).execute(ctx)
        assert seen == ["caa_105-records"], seen

    @pytest.mark.asyncio
    async def test_without_a_suffix_it_uses_the_run_id(self, vm, monkeypatch):
        ctx, _, _ = vm
        ctx.metadata["universe_id"] = "caa_105"
        seen = []
        from agent_env.artifact.artifacts import file_artifact_universe as fau
        monkeypatch.setattr(fau.FileArtifactUniverse, "put",
                            classmethod(lambda cls, id, file_artifacts: seen.append(id) or
                                        cls(id=id, version=1, file_artifact_refs={})))
        await CollectArtifactsTaskStep(
            id="collect", version=None, sandbox_name="caa", base_path="/app/artifact",
        ).execute(ctx)
        assert seen == ["caa_105"], seen

    def test_it_round_trips(self):
        step = CollectArtifactsTaskStep(
            id="c", version=None, sandbox_name="caa", universe_id_suffix="-records",
        )
        assert step.to_dict()["universe_id_suffix"] == "-records"
        assert CollectArtifactsTaskStep.from_dict(step.to_dict()).universe_id_suffix == "-records"

    def test_it_defaults_to_none(self):
        assert CollectArtifactsTaskStep(id="c", version=None, sandbox_name="caa").universe_id_suffix is None


class TestExecArgs:
    """The three-way wrap: container, VM host, container-mode sandbox."""

    def _args(self, mode, container):
        from agent_env.task_step.task_steps.collect_artifacts import _exec_args
        return _exec_args(type("S", (), {"mode": mode})(), container, ("find", "/app"))

    def test_vm_with_container_execs_into_it(self):
        assert self._args("vm", "agent-api") == ("sudo", "docker", "exec", "agent-api", "find", "/app")

    def test_vm_without_container_still_sudoes(self):
        assert self._args("vm", None) == ("sudo", "find", "/app")

    def test_container_mode_runs_bare(self):
        assert self._args("container", None) == ("find", "/app")


class TestAgentPathUnchanged:
    @pytest.mark.asyncio
    async def test_agent_target_still_discovers_its_container(self, monkeypatch):
        """With an agent named, the old discovery path must still run."""
        seen = {}

        async def _fake_agent_collect(self, context, store, artifact_id, version):
            seen["called"] = True
            return {}, {}, []

        monkeypatch.setattr(
            CollectArtifactsTaskStep, "_collect_via_agent_container", _fake_agent_collect
        )
        ctx = TaskStepContext()
        ctx.deployed_agents = [DeployedAgent(agent_name="solver", api_url="http://x", sandbox_id="sb-1")]
        step = CollectArtifactsTaskStep(id="collect", version=None, agent_name="solver")
        await step.execute(ctx)
        assert seen.get("called") is True
