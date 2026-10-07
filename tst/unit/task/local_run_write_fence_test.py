"""An ``@local`` task run against configured stores that refuse every write: it reads the registry,
and everything it writes lands in the per-user local stores or is refused before it happens."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_env.providers.sandbox_providers.sandbox_provider as sandbox_provider
from agent_env.artifact import EnvironmentUniverseArtifact, FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import Outcome, RunInterrupted, run_bundle
from agent_env.bundle.installed import find_bundle
from agent_env.config import configure, get_config
from agent_env.config.paths import state_root
from agent_env.env.env import DeployedGatewayEnv
from agent_env.store import Filter, LocalSqliteDocumentStore
from agent_env.store.ids import derive_id
from agent_env.store.image_store import OciRegistryImageStore
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.store.routing import LocalRunWriteError, run_scope
from agent_env.task.task import Task
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.snapshot_utils.snapshot_series import SnapshotConfig, SnapshotSeries
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import RunContainerUnitTestsVerifierTaskStep
from tst.unit.store.fakes import CollectingVm

LOCAL_TASK = "@local/~/bundle/tasks/t"
LOCAL_ENV = "@local/~/bundle/envs/tickets"
_DOCUMENT_WRITES = ("insert", "update", "update_one_and_get", "replace", "delete")
_OBJECT_WRITES = ("put", "put_file", "put_file_at", "signed_put_url", "issue_write_grant", "issue_upload_policy")


def _refusing(cls, methods):
    """``cls`` with every write in ``methods`` raising once ``armed``, and each attempt recorded."""

    def refuse(name):
        def write(self, *args, **kwargs):
            if self.armed:
                self.attempts.append((name, args[:1]))
                raise AssertionError(f"configured store written: {name}{args[:1]}")
            return getattr(cls, name)(self, *args, **kwargs)
        return write

    return type(f"Sentinel{cls.__name__}", (cls,), {"armed": False, **{m: refuse(m) for m in methods}})


class _Step(TaskStep):
    """Runs ``action`` as its body."""

    type = "local_run_fence_test_step"

    def __init__(self, id, action):
        super().__init__(id, 1)
        self.action = action

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        self.action(context)
        return context


@pytest.fixture
def sentinels(tmp_path, cli_routing):
    documents = _refusing(LocalSqliteDocumentStore, _DOCUMENT_WRITES)(str(tmp_path / "configured.db"))
    objects = _refusing(LocalFilesystemObjectStore, _OBJECT_WRITES)(str(tmp_path / "configured-objects"))
    images = _refusing(OciRegistryImageStore, ("ensure_repository",))(registry_host="registry.example.com")
    for store in (documents, objects, images):
        store.attempts = []
    documents.insert("envs", {"id": "rocket", "version": 1, "type": "mcp_server"})
    registry_object = objects.put("shared/seed.json", b"registry data")
    configure(document_store=documents, object_store=objects, image_store=images)
    for store in (documents, objects, images):
        store.armed = True
    yield documents, objects, images, registry_object
    assert [store.attempts for store in (documents, objects, images)] == [[], [], []]


@pytest.mark.asyncio
async def test_an_local_run_reads_the_registry_and_writes_only_locally(sentinels, tmp_path):
    documents, _, _, registry_object = sentinels
    source = tmp_path / "out.txt"
    source.write_text("produced by the run")

    def read_registry(context):
        config = get_config()
        context.metadata["env"] = config.get_document_store().find_one("envs", Filter.of(id="rocket"))["id"]
        context.metadata["seed"] = config.get_object_store().get(registry_object).decode()

    def write_outputs(context):
        FileArtifactUniverse.put_bundled("@local/~/bundle/tasks/t/out", files={"out.txt": source})
        get_config().get_object_store().put("prompt_agent_trajectories/p/trajectory.json", b"{}")

    context = await Task(id=LOCAL_TASK, version=1, steps=[_Step("read", read_registry), _Step("write", write_outputs)]).run(
        context=TaskStepContext()
    )

    assert context.metadata["env"] == "rocket"
    assert context.metadata["seed"] == "registry data"
    local = get_config().get_document_store().local
    assert local.find_one("task_instances", Filter.of(task_id=LOCAL_TASK)) is not None
    assert local.find_one("artifacts", Filter.of(id="@local/~/bundle/tasks/t/out")) is not None
    universe = FileArtifactUniverse.get("@local/~/bundle/tasks/t/out")
    assert {name: fa.load() for name, fa in universe.get_file_artifacts().items()} == {"out.txt": b"produced by the run"}
    trajectories = state_root() / "object_store" / "prompt_agent_trajectories" / "p" / "trajectory.json"
    assert trajectories.read_bytes() == b"{}"
    assert documents.count("task_instances", Filter()) == 0


@pytest.mark.asyncio
async def test_an_local_run_that_writes_a_bare_entity_fails_before_any_upload(sentinels, tmp_path):
    source = tmp_path / "out.txt"
    source.write_text("x")

    def write_bare(context):
        FileArtifactUniverse.put_bundled("cli-rocket", files={"out.txt": source})

    with pytest.raises(LocalRunWriteError, match="'cli-rocket'"):
        await Task(id=LOCAL_TASK, version=1, steps=[_Step("write", write_bare)]).run(context=TaskStepContext())


def test_the_shipped_hello_writes_nothing_to_the_configured_stores(sentinels, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    hello = find_bundle("hello")

    for reused in (False, True):
        result = run_bundle(hello.root, id_root=hello.id_root)
        assert [run.outcome for run in result.runs] == [Outcome.PASSED]
        assert [write.reused for write in result.materialization.writes] == [reused, reused]
        assert list((tmp_path / "sandboxes").iterdir()) == []


def test_a_cancelled_local_run_is_marked_and_torn_down_in_the_local_stores_only(sentinels, tmp_path, monkeypatch,
                                                                               sigint_handled):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    root = tmp_path / "interrupted"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks/stop.json").write_text(json.dumps([
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"},
        {"id": "stop", "type": "verify_sandbox", "sandbox_name": "box", "base_dir": "/app", "verifier_id": "stop",
         "criteria": [{"type": "bash_cmd_succeeds", "criterion": "interrupts", "bash_cmd": "kill -INT $PPID; sleep 30"}]},
    ]))

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(root)

    (run,) = stopped.value.result.runs
    instance = get_config().get_document_store().local.find_one("task_instances", Filter.of(instance_id=run.instance_id))
    assert (instance["status"], instance["error"]) == ("cancelled", "cancelled by Ctrl-C")
    assert list((tmp_path / "sandboxes").iterdir()) == []


@pytest.mark.asyncio
async def test_a_task_run_on_its_own_leaves_its_sandboxes_up(sentinels, tmp_path, monkeypatch):
    """Tearing down is the run command's: ``Task.run`` leaves what it deployed, as the hub and the worker need."""
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    task = Task.from_dict({"id": LOCAL_TASK, "version": 1, "steps": [
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"}]})

    context = await task.run(context=TaskStepContext())

    (sandbox,) = context.deployed_sandboxes
    assert LocalSandbox.find_work_dir(sandbox.sandbox_id).parent == tmp_path / "sandboxes"


@pytest.mark.asyncio
async def test_a_registry_run_still_writes_to_the_configured_stores(tmp_path, cli_routing):
    documents = LocalSqliteDocumentStore(str(tmp_path / "configured.db"))
    configure(document_store=documents)

    await Task(id="registry-task", version=1, steps=[_Step("noop", lambda context: None)]).run(context=TaskStepContext())

    assert documents.find_one("task_instances", Filter.of(task_id="registry-task")) is not None
    assert not (state_root() / "document_store").exists()


@pytest.mark.asyncio
async def test_concurrent_runs_keep_their_own_scope(tmp_path, cli_routing):
    documents = LocalSqliteDocumentStore(str(tmp_path / "configured.db"))
    configure(document_store=documents)

    class _Interleaved(TaskStep):
        type = "local_run_interleaving_test_step"

        def __init__(self, id, owner):
            super().__init__(id, 1)
            self.owner = owner

        async def execute(self, context):
            for n in range(3):
                await asyncio.sleep(0)
                await asyncio.to_thread(get_config().get_document_store().insert, "runs", {"owner": self.owner, "n": n})
            return context

    owners = [LOCAL_TASK, "registry-task", "@local/~/bundle/tasks/u", "other-registry-task"]
    await asyncio.gather(*(Task(id=o, version=1, steps=[_Interleaved("s", o)]).run(context=TaskStepContext()) for o in owners))

    local = get_config().get_document_store().local
    assert {doc["owner"] for doc in local.query("runs", Filter())} == {LOCAL_TASK, "@local/~/bundle/tasks/u"}
    assert {doc["owner"] for doc in documents.query("runs", Filter())} == {"registry-task", "other-registry-task"}


@pytest.fixture
def vm(monkeypatch):
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=CollectingVm()), close=lambda: asyncio.sleep(0))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    return DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode="vm")


@pytest.mark.asyncio
async def test_an_local_run_collects_into_a_universe_named_after_the_run(sentinels, vm):
    step = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", base_path="/app/artifact",
                                    artifact_paths=["report.pdf", "sub/notes.md"], universe_id_suffix="-docs")

    context = await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=TaskStepContext(deployed_sandboxes=[vm]))

    universe = FileArtifactUniverse.get(f"{context.instance_id}-docs")
    assert universe.file_artifact_ids == {
        name: derive_id(universe.id, hashlib.sha256(name.encode()).hexdigest()[:16]) for name in ("report.pdf", "sub/notes.md")
    }
    assert {name: fa.load() for name, fa in universe.get_file_artifacts().items()} == {"report.pdf": b"%PDF", "sub/notes.md": b"# notes"}


@pytest.mark.asyncio
async def test_an_local_run_refuses_a_bare_universe_to_collect_into_before_reading_the_sandbox(sentinels, vm, monkeypatch):
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: pytest.fail("the sandbox was read"))
    context = TaskStepContext(deployed_sandboxes=[vm])
    context.metadata["universe_id"] = "VPC Endpoints"
    step = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", artifact_paths=["report.pdf"])

    with pytest.raises(LocalRunWriteError, match="'VPC Endpoints'"):
        await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=context)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", ["Q&A: onboarding", "VPC Endpoints "])
async def test_an_local_run_refuses_an_invalid_local_universe_to_collect_into_before_reading_the_sandbox(
        sentinels, vm, monkeypatch, seed):
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: pytest.fail("the sandbox was read"))
    context = TaskStepContext(deployed_sandboxes=[vm])
    context.metadata["universe_id"] = f"{LOCAL_TASK}-v1-{seed}"
    step = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", artifact_paths=["report.pdf"])

    with pytest.raises(ValueError, match="@local"):
        await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=context)


class _Gmail:
    """An env with one service, gmail."""

    version = 1

    def __init__(self, id, version=None):
        self.id = id
        self.mcp_server_envs = [SimpleNamespace(environment_name="gmail")]


async def _export(gateway_url, name, tmp_path, timeout, deployed=None):
    Path(tmp_path).write_text(json.dumps({"service": name}))
    return ".json"


@pytest.fixture
def exports(monkeypatch):
    """Envs resolve to ``_Gmail``, and each service exports ``{"service": <name>}``."""
    monkeypatch.setattr("agent_env.env.env.Env.get", _Gmail)
    monkeypatch.setattr(SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(_export))


def _deployed(env_id):
    return DeployedGatewayEnv(env_id=env_id, env_version=1, gateway_url="https://gw", mcp_url="https://gw/mcp", db_web_url=None,
                              sandbox_id="sb-1")


def _assert_snapshot(universe_id):
    (gmail,) = EnvironmentUniverseArtifact.get(universe_id).get_environment_artifacts()
    assert (gmail.id, json.loads(gmail.get_file_artifact().load())) == (f"{universe_id}-gmail", {"service": "gmail"})


@pytest.mark.asyncio
@pytest.mark.parametrize("env_id, snapshot_id", [(LOCAL_ENV, None), ("rocket", None), ("rocket", f"{LOCAL_TASK}/greeting")],
                         ids=["local-env", "registry-env", "named"])
async def test_an_local_runs_env_snapshot_is_named_after_the_run_unless_the_task_names_it(sentinels, exports, env_id, snapshot_id):
    step = SnapshotEnvTaskStep(id="snap", version=None, env_id=env_id, snapshot_id=snapshot_id)

    context = await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=TaskStepContext(deployed_envs=[_deployed(env_id)]))

    universe_id = snapshot_id or derive_id(context.instance_id, "snapshot-snap")
    assert context.metadata["env_snapshotted_universes"]["snap"] == {"id": universe_id, "version": 1}
    _assert_snapshot(universe_id)


@pytest.mark.asyncio
async def test_an_local_run_refuses_a_snapshot_step_id_its_universe_cant_take_before_resolving_the_env(sentinels, monkeypatch):
    monkeypatch.setattr(SnapshotEnvTaskStep, "_resolve_deployed_env", lambda self, context: pytest.fail("the env was resolved"))
    step = SnapshotEnvTaskStep(id="snap:final", version=None, env_id=LOCAL_ENV)

    with pytest.raises(ValueError, match="contains ':'"):
        await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=TaskStepContext(deployed_envs=[_deployed(LOCAL_ENV)]))


@pytest.mark.asyncio
async def test_an_local_runs_capture_series_names_its_captures_after_the_run(sentinels, exports):
    instance_id = f"{LOCAL_TASK}-ab12cd34"
    series = SnapshotSeries(step_id="solve", agent_name="solver", prompt_id="p1", a2a_context_id="c1",
                            config=SnapshotConfig(env_id="rocket"), trajectory_output_prefix="unused", instance_id=instance_id)
    row: dict = {}

    with run_scope(LOCAL_TASK):
        assert await series._capture_env_state(TaskStepContext(instance_id=instance_id, deployed_envs=[_deployed("rocket")]), row, 30) is None
        assert get_artifact_store().next_version(series.workspace_artifact_id) == 1

    assert series.workspace_artifact_id == derive_id(instance_id, "solve-workspace")
    assert row["env_universe_id"] == derive_id(instance_id, "snapshot-solve")
    _assert_snapshot(row["env_universe_id"])


def test_an_local_runs_capture_series_refuses_a_step_id_its_captures_cant_take_when_it_is_built():
    def series(instance_id):
        return SnapshotSeries(step_id="solve:1", agent_name="solver", prompt_id="p1", a2a_context_id="c1",
                              config=SnapshotConfig(env_id="rocket"), trajectory_output_prefix="unused", instance_id=instance_id)

    with pytest.raises(ValueError, match="contains ':'"):
        series(f"{LOCAL_TASK}-ab12cd34")
    assert series("rocket-run-ab12cd34").workspace_artifact_id == "rocket-run-ab12cd34__solve:1-workspace"


@pytest.mark.asyncio
async def test_an_local_runs_verifier_outputs_are_named_after_the_run(sentinels, vm):
    context = TaskStepContext(deployed_sandboxes=[vm])
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "box"}]
    step = RunContainerUnitTestsVerifierTaskStep(id="verify", version=None, sandbox_name="box", container_name="c", command="true")

    context = await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=context)

    entry = context.metadata["verifications"]["verify"]
    for stream, text in (("stdout", b"ok"), ("stderr", b"")):
        artifact = FileArtifact.get(entry[f"{stream}_artifact"]["id"])
        assert artifact.id.startswith(derive_id(context.instance_id, f"verifier-{stream}-verify-"))
        assert artifact.load() == text


@pytest.mark.asyncio
async def test_an_local_run_refuses_a_verifier_step_id_its_outputs_cant_take_before_the_tests_run(sentinels, vm, monkeypatch):
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: pytest.fail("the sandbox was read"))
    context = TaskStepContext(deployed_sandboxes=[vm])
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "box"}]
    step = RunContainerUnitTestsVerifierTaskStep(id="verify:1", version=None, sandbox_name="box", container_name="c", command="true")

    with pytest.raises(ValueError, match="contains ':'"):
        await Task(id=LOCAL_TASK, version=1, steps=[step]).run(context=context)
