"""`universe_id_suffix` names a collect step's FileArtifactUniverse.

Two collect steps in one run otherwise derive the same id from the run and the
second lands as another version of the first. `run-batch` loads the task once and
runs every seed against the same TaskStep objects, so the suffix must be read-only
state and the resulting id must depend only on the per-run context.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep

FILES = {"bundle/MODIFICATIONS.md": b"# what changed\n"}


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
    mode = "vm"

    async def exec_with_output(self, *args):
        if "find" in args:
            if "-maxdepth" in args:
                return 0, "bundle\n", ""
            return 0, "".join(f"{p}\n" for p in FILES), ""
        if "stat" in args:
            return 0, str(len(next(iter(FILES.values())))), ""
        return 0, "", ""

    async def exec(self, *args):
        await asyncio.sleep(0)  # yield, so concurrent collects genuinely interleave
        return _Proc(base64.b64encode(next(iter(FILES.values()))))


@pytest.fixture
def env(monkeypatch):
    """Stub the sandbox provider and artifact store; record every universe id created."""
    from agent_env.providers.sandbox_providers import sandbox_provider as sp_mod

    class _Provider:
        @staticmethod
        async def get_sandbox(sandbox_id):
            return _VmSandbox()

        async def close(self):
            return None

    monkeypatch.setattr(sp_mod, "get_sandbox_provider", lambda: _Provider())

    class _Store:
        def put_object_file(self, *, object_name, **kw):
            return f"s3://bucket/{object_name}"

        def next_version(self, _id):
            return 1

        def put_document(self, _doc):
            return None

    from agent_env.artifact import store as artifact_store_mod

    monkeypatch.setattr(artifact_store_mod, "get_artifact_store", lambda: _Store())

    created: list[str] = []
    from agent_env.artifact.artifacts import file_artifact_universe as fau

    def _put(cls, id, file_artifacts):
        created.append(id)
        return cls(id=id, version=1, file_artifact_refs={})

    monkeypatch.setattr(fau.FileArtifactUniverse, "put", classmethod(_put))
    return created


def _ctx(universe_id: str | None = None, instance_id: str | None = None) -> TaskStepContext:
    ctx = TaskStepContext()
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="caa", sandbox_id="sb-1", sandbox_mode="vm")]
    if universe_id:
        ctx.metadata["universe_id"] = universe_id
    if instance_id:
        ctx.instance_id = instance_id
    return ctx


def _step(step_id: str, suffix: str | None = None, base: str = "/app/artifact"):
    return CollectArtifactsTaskStep(
        id=step_id, version=None, sandbox_name="caa", base_path=base, universe_id_suffix=suffix,
    )


class TestNaming:
    @pytest.mark.asyncio
    async def test_two_collects_in_one_run_get_distinct_names(self, env):
        ctx = _ctx(universe_id="caa_105")
        await _step("collect-records", "-records", "/app/records").execute(ctx)
        await _step("collect-artifacts").execute(ctx)
        assert env == ["caa_105-records", "caa_105"]

    @pytest.mark.asyncio
    async def test_no_suffix_leaves_the_run_id_alone(self, env):
        await _step("collect").execute(_ctx(universe_id="caa_105"))
        assert env == ["caa_105"]

    @pytest.mark.asyncio
    async def test_it_falls_back_to_the_instance_id(self, env):
        await _step("collect", "-records").execute(_ctx(instance_id="inst-42"))
        assert env == ["inst-42-records"]

    @pytest.mark.asyncio
    async def test_runs_whose_ids_share_a_100_char_prefix_get_distinct_names(self, env):
        step, prefix = _step("collect", "-records"), "c" * 100
        await step.execute(_ctx(instance_id=f"{prefix}-run-1"))
        await step.execute(_ctx(instance_id=f"{prefix}-run-2"))
        assert env == [f"{prefix}-run-1-records", f"{prefix}-run-2-records"]


class TestConcurrentSeeds:
    """run-batch shares one TaskStep across seeds — ids must come only from context."""

    @pytest.mark.asyncio
    async def test_shared_step_objects_do_not_leak_between_seeds(self, env):
        records, bundle = _step("collect-records", "-records", "/app/records"), _step("collect-artifacts")
        seeds = [f"caa_{n}" for n in range(1, 13)]

        async def run_one(seed):
            ctx = _ctx(universe_id=seed)          # each seed gets its own context
            await records.execute(ctx)
            await bundle.execute(ctx)

        await asyncio.gather(*(run_one(s) for s in seeds))

        assert len(env) == len(set(env)) == 2 * len(seeds), env
        for s in seeds:
            assert s in env and f"{s}-records" in env

    @pytest.mark.asyncio
    async def test_a_seed_only_ever_sees_its_own_name(self, env):
        step = _step("collect-records", "-records", "/app/records")
        await asyncio.gather(*(step.execute(_ctx(universe_id=f"caa_{n}")) for n in range(20)))
        assert sorted(env) == sorted(f"caa_{n}-records" for n in range(20))

    def test_the_suffix_is_read_only_config(self):
        """Nothing may rebind it per run, or one seed would rename another's artifact."""
        import ast
        import inspect
        import textwrap

        cls = ast.parse(textwrap.dedent(inspect.getsource(CollectArtifactsTaskStep))).body[0]
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name == "__init__":
                continue
            for node in ast.walk(fn):
                targets = node.targets if isinstance(node, ast.Assign) else []
                for t in targets:
                    assert not (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                    ), f"{fn.name} mutates self.{t.attr}; shared across concurrent seeds"


class TestSerialization:
    def test_it_round_trips(self):
        step = _step("collect-records", "-records")
        assert step.to_dict()["universe_id_suffix"] == "-records"
        assert CollectArtifactsTaskStep.from_dict(step.to_dict()).universe_id_suffix == "-records"

    def test_it_defaults_to_none(self):
        assert _step("collect").universe_id_suffix is None
        assert CollectArtifactsTaskStep.from_dict(_step("collect").to_dict()).universe_id_suffix is None
