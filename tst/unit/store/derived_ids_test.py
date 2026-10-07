"""The ids core names after an entity or a run: ``derive_id(base, suffix)``, for a bare base and an ``@local`` one
alike, so the base's namespace carries over. Consumers rebuild two of these spellings to link a validation to its
task, ``<env>__validate-v<n>`` and ``<env>__validate-universe-compat-v<ev>-<universe>-v<uv>``, so every table here
pins its bare row."""

import asyncio
import re
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import agent_env.providers.sandbox_providers.sandbox_provider as sandbox_provider
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.artifact import DockerImageArtifact, FileArtifactUniverse
from agent_env.cli import cli
from agent_env.config import get_config, set_object_store
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.snapshot_store import _DOCKER_NAME, EnvSnapshot, _image_tag
from agent_env.task import Task
from agent_env.task_step import VerifyUniverseLoadExportRoundtripStep
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.snapshot_utils.snapshot_series import SnapshotConfig, SnapshotSeries
from agent_env.task_step.task_steps.add_skills import _build_skill_for_loaded_file_artifact_universe
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from agent_env.task_step.task_steps.verifiers import run_container_unit_tests_verifier as verifier_module
from tst.unit.store.fakes import CollectingVm, SigningObjectStore, reattach_for_snapshot

LOCAL = "@local/~/work/triage"
LOCAL_ENV = f"{LOCAL}/envs/tickets"
LOCAL_UNIVERSE = f"{LOCAL}/artifacts/seed"
LOCAL_AGENT = f"{LOCAL}/agents/solver"
LOCAL_TASK = f"{LOCAL}/tasks/t"
BARE_RUN = "triage-1-abcd1234"
LOCAL_RUN = f"{LOCAL_TASK}-abcd1234"
# fs_safe and key_segment of LOCAL_ENV and LOCAL_UNIVERSE, and key_segment of LOCAL_RUN.
LOCAL_ENV_FILENAME, LOCAL_ENV_KEY = "local-work-triage-envs-tickets-ca64d3fb6d09", "local/work-triage-envs-tickets-ca64d3fb6d09"
LOCAL_UNIVERSE_FILENAME, LOCAL_UNIVERSE_KEY = (
    "local-work-triage-artifacts-seed-ddf2904a1148", "local/work-triage-artifacts-seed-ddf2904a1148",
)
LOCAL_RUN_KEY = "local/work-triage-tasks-t-abcd1234-d34bacfa44b2"


class _Written(Exception):
    """Raised by a patched write once it has what the code under test writes."""


def _written_by(run, monkeypatch, owner=Task, method="put") -> dict:
    """The keywords of the one ``owner.method`` call ``run`` makes, which stops there."""
    written = []

    def write(cls, **kwargs):
        written.append(kwargs)
        raise _Written

    monkeypatch.setattr(owner, method, classmethod(write))
    with pytest.raises(_Written):
        asyncio.run(run())
    (kwargs,) = written
    return kwargs


_VALIDATIONS = {
    "mcp_server": lambda id: MCPServerEnv(id=id, version=3, docker_image_artifact=MagicMock(), environment_name="slack").validate(),
    "website": lambda id: WebsiteEnv(id=id, version=3, backend_docker_image_artifact=MagicMock(),
                                     frontend_docker_image_artifact=MagicMock(), environment_name="shop").validate(),
    "multi": lambda id: MultiEnv(id=id, version=3, mcp_server_envs=[]).validate(),
    "a2a_agent": lambda id: A2AAgentValidator.validate(A2AAgent(id=id, version=22, docker_image_artifact=MagicMock())),
}


@pytest.mark.parametrize("kind, entity_id, task_id", [
    ("mcp_server", "slack-mcp", "slack-mcp__validate-v3"),
    ("mcp_server", LOCAL_ENV, f"{LOCAL_ENV}__validate-v3"),
    ("website", "shop", "shop__validate-v3"),
    ("website", LOCAL_ENV, f"{LOCAL_ENV}__validate-v3"),
    ("multi", "crm-suite", "crm-suite__validate-v3"),
    ("multi", LOCAL_ENV, f"{LOCAL_ENV}__validate-v3"),
    ("a2a_agent", "claude-code", "claude-code__validate-a2a-v22"),
    ("a2a_agent", LOCAL_AGENT, f"{LOCAL_AGENT}__validate-a2a-v22"),
], ids=["mcp_server-bare", "mcp_server-local", "website-bare", "website-local", "multi-bare", "multi-local",
        "a2a_agent-bare", "a2a_agent-local"])
def test_a_validation_task_is_named_after_its_entity(local_stores, tmp_path, monkeypatch, kind, entity_id, task_id):
    set_object_store(SigningObjectStore(str(tmp_path / "signing")))
    assert _written_by(lambda: _VALIDATIONS[kind](entity_id), monkeypatch)["id"] == task_id


@pytest.mark.parametrize("env_id, universe_id, task_id, fau_id", [
    ("crm-suite", "crm-universe",
     "crm-suite__validate-universe-compat-v3-crm-universe-v2", "crm-suite__validate-v3-crm-universe-v2-fau"),
    (LOCAL_ENV, "crm-universe",
     f"{LOCAL_ENV}__validate-universe-compat-v3-crm-universe-v2", f"{LOCAL_ENV}__validate-v3-crm-universe-v2-fau"),
    (LOCAL_ENV, LOCAL_UNIVERSE,
     f"{LOCAL_ENV}__validate-universe-compat-v3-{LOCAL_UNIVERSE_FILENAME}-v2",
     f"{LOCAL_ENV}__validate-v3-{LOCAL_UNIVERSE_FILENAME}-v2-fau"),
    ("crm-suite", LOCAL_UNIVERSE,
     f"{LOCAL_UNIVERSE}__validate-universe-compat-v2-crm-suite-v3", f"{LOCAL_UNIVERSE}__validate-v2-crm-suite-v3-fau"),
], ids=["bare", "local-env", "both-local", "local-universe"])
def test_a_compatibility_validation_is_named_after_its_env_unless_only_the_universe_is_local(
        monkeypatch, env_id, universe_id, task_id, fau_id):
    universe = SimpleNamespace(id=universe_id, version=2, get_environment_artifacts=lambda: [])
    monkeypatch.setattr("agent_env.artifact.EnvironmentUniverseArtifact.get", lambda *args: universe)

    task = _written_by(lambda: MultiEnv(id=env_id, version=3, mcp_server_envs=[]).validate_universe_compatibility(universe_id),
                       monkeypatch)

    (load,) = [step for step in task["steps"] if step.type == "load_artifact"]
    assert (task["id"], [artifact["id"] for artifact in load.artifacts]) == (task_id, [fau_id])


@pytest.mark.parametrize("env_id, export", [
    ("crm-suite", "crm-suite__validate-v3-crm-universe-v7-export1"),
    (LOCAL_ENV, f"{LOCAL_ENV}__validate-v3-crm-universe-v7-export1"),
], ids=["bare", "local"])
def test_roundtrip_exports_are_named_after_the_validation(local_stores, cli_routing, env_id, export):
    step = VerifyUniverseLoadExportRoundtripStep(id="rt", version=None, env_id=env_id, universe_artifact_id="crm-universe")

    universe = step._create_universe_artifact([SimpleNamespace(environment_name="slack")], {"slack": {"messages": []}},
                                              env_version=3, universe_version=7)

    (service,) = universe.get_environment_artifacts()
    assert (universe.id, service.id, service.get_file_artifact().id) == (export, f"{export}-svc-slack", f"{export}-slack")


@pytest.mark.parametrize("env_id, task_id, cli_id", [
    ("slack-mcp", "slack-mcp__create-cli-v3", "slack-mcp__cli"),
    (LOCAL_ENV, f"{LOCAL_ENV}__create-cli-v3", f"{LOCAL_ENV}__cli"),
], ids=["bare", "local"])
def test_create_cli_names_its_task_and_cli_after_the_env(monkeypatch, env_id, task_id, cli_id):
    env = MCPServerEnv(id=env_id, version=3, docker_image_artifact=MagicMock(), environment_name="slack")

    task = _written_by(lambda: env.create_cli(force=True), monkeypatch)

    assert (task["id"], task["steps"][1].cli_artifact_id) == (task_id, cli_id)


@pytest.mark.parametrize("agent_id, universe_id", [
    ("claude-code", "claude-code__validate-install-image-v22-1790000000"),
    (LOCAL_AGENT, f"{LOCAL_AGENT}__validate-install-image-v22-1790000000"),
], ids=["bare", "local"])
def test_the_install_image_universe_is_named_after_the_agent(local_stores, cli_routing, monkeypatch, agent_id, universe_id):
    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)
    agent = A2AAgent(id=agent_id, version=22, docker_image_artifact=MagicMock())

    assert A2AAgentValidator._upload_install_test_image_fixture(agent).id == universe_id


@pytest.mark.parametrize("env_id, universe_id, artifact_id, image_tag, key", [
    ("crm-suite", "crm-universe", "crm-suite__env-snapshot", "env-snapshot-crm-suite-crm-universe",
     "env-snapshots/crm-suite/crm-universe/env-snapshot-crm-suite-crm-universe"),
    (LOCAL_ENV, LOCAL_UNIVERSE, f"{LOCAL_ENV}__env-snapshot", f"env-snapshot-{LOCAL_ENV_FILENAME}-{LOCAL_UNIVERSE_FILENAME}",
     f"env-snapshots/{LOCAL_ENV_KEY}/{LOCAL_UNIVERSE_KEY}/env-snapshot-{LOCAL_ENV_FILENAME}-{LOCAL_UNIVERSE_FILENAME}"),
], ids=["bare", "local"])
def test_an_env_snapshot_is_named_after_its_env(local_stores, tmp_path, monkeypatch, env_id, universe_id, artifact_id,
                                                image_tag, key):
    set_object_store(SigningObjectStore(str(tmp_path / "signing")))
    sandbox = reattach_for_snapshot(monkeypatch, env_id, 3, universe_id, 2)

    image = _written_by(lambda: EnvSnapshot.create("inst-1"), monkeypatch, DockerImageArtifact, "put_tar")

    assert (image["id"], image["image_name"]) == (artifact_id, image_tag)
    assert re.search(rf"/{re.escape(key)}-[0-9a-f]{{16}}\.tar\.gz$", image["tar_gz_s3_url"])
    assert f"docker build --platform linux/amd64 -t {image_tag} /tmp/snapshot-build" in sandbox.scripts


@pytest.mark.parametrize("universe_id, image_tag", [
    ("crm-universe", "env-snapshot-crm-suite-crm-universe"),
    (f"crm-run-{'a' * 210}__snapshot-snap", "env-snapshot-crm-suite-16d9f6db8bf8"),
    ("Crm-Run-abcd1234__snapshot-snap", "env-snapshot-crm-suite-a655c6c500a1"),
], ids=["short", "233-chars", "mixed-case"])
def test_an_env_snapshot_image_name_is_a_docker_name_of_at_most_200_chars(universe_id, image_tag):
    tag = _image_tag("crm-suite", universe_id)
    assert tag == image_tag
    assert _DOCKER_NAME.fullmatch(tag)
    assert len(tag) <= 200


@pytest.mark.parametrize("instance_id, snapshot_id", [
    (BARE_RUN, "triage-1-abcd1234__snapshot-snap"),
    (LOCAL_RUN, f"{LOCAL_RUN}__snapshot-snap"),
], ids=["bare", "local"])
def test_an_env_snapshot_step_snapshots_into_a_universe_named_after_the_run(instance_id, snapshot_id):
    step = SnapshotEnvTaskStep(id="snap", version=None, env_id="crm-suite")
    assert step._derive_snapshot_id(TaskStepContext(instance_id=instance_id)) == snapshot_id


def _series(instance_id: str) -> SnapshotSeries:
    return SnapshotSeries(step_id="solve", agent_name="solver", prompt_id="p1", a2a_context_id="c1",
                          config=SnapshotConfig(env_id="crm-suite"), trajectory_output_prefix="unused", instance_id=instance_id)


def _captured_snapshot_id(series: SnapshotSeries, monkeypatch) -> str:
    """The universe ``series`` snapshots its env into on a capture."""
    snapshot_ids = []

    async def snapshot_env_state(**kwargs):
        snapshot_ids.append(kwargs["snapshot_id"])
        return SimpleNamespace(environment_universe_artifact_id=kwargs["snapshot_id"], environment_universe_artifact_version=1)

    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *args: SimpleNamespace(id="crm-suite"))
    monkeypatch.setattr(SnapshotEnvTaskStep, "snapshot_env_state", staticmethod(snapshot_env_state))
    deployed = DeployedGatewayEnv(env_id="crm-suite", env_version=1, gateway_url="https://gw", mcp_url="https://gw/mcp",
                                  db_web_url=None, sandbox_id="sb-1")
    assert asyncio.run(series._capture_env_state(TaskStepContext(deployed_envs=[deployed]), {}, 30)) is None
    (snapshot_id,) = snapshot_ids
    return snapshot_id


@pytest.mark.parametrize("instance_id, workspace_id, snapshot_id", [
    (BARE_RUN, "triage-1-abcd1234__solve-workspace", "triage-1-abcd1234__snapshot-solve"),
    (LOCAL_RUN, f"{LOCAL_RUN}__solve-workspace", f"{LOCAL_RUN}__snapshot-solve"),
], ids=["bare", "local"])
def test_a_capture_series_names_its_artifacts_after_the_run(monkeypatch, instance_id, workspace_id, snapshot_id):
    series = _series(instance_id)
    assert (series.workspace_artifact_id, _captured_snapshot_id(series, monkeypatch)) == (workspace_id, snapshot_id)


def test_an_env_snapshot_step_and_a_capture_series_in_one_run_snapshot_one_env_into_different_universes(monkeypatch):
    step = SnapshotEnvTaskStep(id="snap", version=None, env_id="crm-suite")
    from_step = step._derive_snapshot_id(TaskStepContext(instance_id=BARE_RUN))
    assert from_step != _captured_snapshot_id(_series(BARE_RUN), monkeypatch)


class _VerifierSandbox:
    async def exec_script(self, script: str) -> str:
        return ""

    async def exec_with_output(self, *args) -> tuple[int, str, str]:
        return 0, "ok", ""


@pytest.mark.parametrize("instance_id, base, run", [
    (BARE_RUN, "triage-1-abcd1234", "triage-1-abcd1234-1790000000-0123456789ab"),
    (LOCAL_RUN, LOCAL_RUN, f"{LOCAL_RUN_KEY}-1790000000-0123456789ab"),
    (None, "adhoc-0123456789ab", "1790000000-0123456789ab"),
], ids=["bare", "local", "no-instance"])
def test_verifier_outputs_are_named_and_keyed_after_the_run(local_stores, monkeypatch, instance_id, base, run):
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=_VerifierSandbox()))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    monkeypatch.setattr(verifier_module.time, "time", lambda: 1_790_000_000.0)
    monkeypatch.setattr(verifier_module.uuid, "uuid4", lambda: uuid.UUID(hex="0123456789ab" + "0" * 20))
    uploads: list[tuple[str, str]] = []

    def record(text, artifact_id, description, s3_url):
        uploads.append((artifact_id, s3_url))
        return SimpleNamespace(id=artifact_id, version=1, object_url=s3_url)

    monkeypatch.setattr(verifier_module.RunContainerUnitTestsVerifierTaskStep, "_upload_text_artifact", staticmethod(record))
    ctx = TaskStepContext(instance_id=instance_id)
    ctx.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    ctx.deployed_sandboxes.append(DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm"))
    step = verifier_module.RunContainerUnitTestsVerifierTaskStep(id="scrape", version=None, sandbox_name="h", container_name="c", command="true")

    asyncio.run(step.execute(ctx))

    outputs = get_config().get_object_store().object_url(f"verifier-outputs/scrape/{run}")
    assert uploads == [(f"{base}__verifier-{stream}-scrape-1790000000-0123456789ab", f"{outputs}/{stream}.txt")
                       for stream in ("stdout", "stderr")]


@pytest.mark.parametrize("instance_id", [BARE_RUN, LOCAL_RUN], ids=["bare", "local"])
def test_collected_files_are_named_after_the_run(local_stores, cli_routing, monkeypatch, instance_id):
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=CollectingVm()),
                               close=lambda: asyncio.sleep(0))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    ctx = TaskStepContext(instance_id=instance_id)
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode="vm")]
    step = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", base_path="/app/artifact",
                                    artifact_paths=["report.pdf", "sub/notes.md"])

    asyncio.run(step.execute(ctx))

    universe = FileArtifactUniverse.get(ctx.metadata["file_artifact_universe"]["id"])
    assert universe.id == instance_id
    assert universe.file_artifact_ids == {
        "report.pdf": f"{instance_id}__6466e450a16b77b8", "sub/notes.md": f"{instance_id}__d72324ebb0d7e97a",
    }


def test_a_seed_universe_longer_than_a_filename_collects_on_the_local_store(local_stores, monkeypatch):
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=CollectingVm()),
                               close=lambda: asyncio.sleep(0))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    universe_id = "triage__v2-" + ("Escalated tickets from the EU desk, " * 8).rstrip(", ")
    ctx = TaskStepContext(instance_id=BARE_RUN)
    ctx.metadata["universe_id"] = universe_id
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode="vm")]
    step = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", base_path="/app/artifact",
                                    artifact_paths=["report.pdf", "sub/notes.md"])

    asyncio.run(step.execute(ctx))

    universe = FileArtifactUniverse.get(universe_id)
    assert {name: fa.load() for name, fa in universe.get_file_artifacts().items()} == {"report.pdf": b"%PDF", "sub/notes.md": b"# notes"}


@pytest.mark.parametrize("universe_id, name", [
    ("crm-universe", "crm-universe-506b24e3579f-files"),
    ("Crm_Universe", "crm-universe-4c998975eec1-files"),
    ("x" * 60, "x" * 45 + "-42f2d9733566-files"),
    (LOCAL_UNIVERSE, "seed-ddf2904a1148-files"),
    (f"{LOCAL}/artifacts/" + "Ticket Desk (EU) " * 5 + "x", "ticket-desk-eu-ticket-desk-eu-ticket-desk-eu-e6f6e0683a17-files"),
    ("@local/~/work/日本", "e3629cd92c57-files"),
], ids=["bare", "bare-mixed-case", "bare-60-chars", "local", "local-long", "local-no-slug"])
def test_a_loaded_universes_skill_is_named_after_it(universe_id, name):
    skill = _build_skill_for_loaded_file_artifact_universe(universe_id, {"destination_path": "/app/files"})
    assert skill.name == name
    skill.validate()


@pytest.mark.parametrize("task_id, universe_id", [
    ("triage", "triage__v2-VPC Endpoints"),
    (LOCAL_TASK, f"{LOCAL_TASK}__v2-VPC Endpoints"),
], ids=["bare", "local"])
def test_a_seeded_runs_universe_is_named_after_its_task_and_seed(tmp_path, monkeypatch, task_id, universe_id):
    seen: list[str] = []

    class _Task:
        id, version, steps = task_id, 2, []

        async def run(self, context, **kwargs):
            seen.append(context.metadata["universe_id"])
            return context

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: _Task()))
    (tmp_path / "seeds.csv").write_text("name\nVPC Endpoints\n")

    result = CliRunner().invoke(cli, ["task", "run-batch", "--id", task_id, "--seeds", str(tmp_path / "seeds.csv"),
                                      "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert seen == [universe_id]


def _collect(suffix=None) -> CollectArtifactsTaskStep:
    return CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", artifact_paths=["report.pdf"],
                                    universe_id_suffix=suffix)


@pytest.mark.parametrize(("task_steps", "seeds", "refused", "runs"), [
    ([_collect()], ["fine", "Q&A: onboarding"], 2, 0),
    ([], ["fine", "Q&A: onboarding"], None, 2),
    ([_collect("-records")], ["VPC Endpoints "], None, 1),
    ([_collect(":records")], ["fine"], 1, 0),
], ids=["collects", "doesnt-collect", "suffix-makes-it-valid", "suffix-makes-it-invalid"])
def test_a_seed_that_cant_name_an_local_universe_stops_a_collecting_run_batch_before_any_run(
        tmp_path, monkeypatch, task_steps, seeds, refused, runs):
    ran: list[TaskStepContext] = []

    class _Task:
        id, version, steps = LOCAL_TASK, 2, task_steps

        async def run(self, context, **kwargs):
            ran.append(context)
            return context

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: _Task()))
    (tmp_path / "seeds.csv").write_text("".join(f"{row}\n" for row in ["name", *seeds]))

    result = CliRunner().invoke(cli, ["task", "run-batch", "--id", _Task.id, "--seeds", str(tmp_path / "seeds.csv"),
                                      "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == (1 if refused else 0), result.output
    assert (f"seed {refused} can't name an @local universe" in result.output) is bool(refused)
    assert len(ran) == runs
