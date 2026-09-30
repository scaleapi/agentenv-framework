"""The ids core names after an entity or a run. A bare base keeps the spelling it has always had: stored
tasks and artifacts, retried activities and the links consumers rebuild all record these strings, so the
bare goldens here are frozen. An ``@local`` base keeps its namespace."""

import asyncio
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
from agent_env.config import set_object_store
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.snapshot_store import EnvSnapshot
from agent_env.task import Task
from agent_env.task_step import VerifyUniverseLoadExportRoundtripStep
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.add_skills import _build_skill_for_loaded_file_artifact_universe
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.verifiers import run_container_unit_tests_verifier as verifier_module
from tst.unit.store.fakes import CollectingVm, SigningObjectStore, reattach_for_snapshot

LOCAL_ENV = "@local/~/work/triage/envs/tickets"
LOCAL_UNIVERSE = "@local/~/work/triage/artifacts/seed"


class _Written(Exception):
    """Raised by the patched ``Task.put`` once it has the task a validator writes."""


def _task_put_by(validate, monkeypatch) -> dict:
    written = []

    def put(cls, **kwargs):
        written.append(kwargs)
        raise _Written

    monkeypatch.setattr(Task, "put", classmethod(put))
    with pytest.raises(_Written):
        asyncio.run(validate())
    (task,) = written
    return task


@pytest.mark.parametrize("validate, task_id", [
    (lambda: MCPServerEnv(id="slack-mcp", version=3, docker_image_artifact=MagicMock(), environment_name="slack").validate(),
     "validate-slack-mcp-v3"),
    (lambda: WebsiteEnv(id="shop", version=3, backend_docker_image_artifact=MagicMock(), frontend_docker_image_artifact=MagicMock(),
                        environment_name="shop").validate(),
     "validate-shop-v3"),
    (lambda: MultiEnv(id="crm-suite", version=3, mcp_server_envs=[]).validate(), "validate-crm-suite-v3"),
    (lambda: A2AAgentValidator.validate(A2AAgent(id="claude-code", version=22, docker_image_artifact=MagicMock())),
     "validate-a2a-claude-code-v22"),
], ids=["mcp_server", "website", "multi", "a2a_agent"])
def test_a_validation_task_keeps_its_bare_name(local_stores, tmp_path, monkeypatch, validate, task_id):
    set_object_store(SigningObjectStore(str(tmp_path / "signing")))
    assert _task_put_by(validate, monkeypatch)["id"] == task_id


def test_a_compatibility_validation_keeps_its_bare_names(monkeypatch):
    universe = SimpleNamespace(id="crm-universe", version=2, get_environment_artifacts=lambda: [])
    monkeypatch.setattr("agent_env.artifact.EnvironmentUniverseArtifact.get", lambda *args: universe)

    task = _task_put_by(lambda: MultiEnv(id="crm-suite", version=3, mcp_server_envs=[]).validate_universe_compatibility("crm-universe"),
                        monkeypatch)

    assert task["id"] == "validate-universe-compat-crm-suite-v3-crm-universe-v2"
    (load,) = [step for step in task["steps"] if step.type == "load_artifact"]
    assert [artifact["id"] for artifact in load.artifacts] == ["validate-crm-suite-v3-crm-universe-v2-fau"]


def test_create_cli_keeps_its_bare_task_and_artifact_ids(monkeypatch):
    env = MCPServerEnv(id="slack-mcp", version=3, docker_image_artifact=MagicMock(), environment_name="slack")

    task = _task_put_by(lambda: env.create_cli(force=True), monkeypatch)

    assert task["id"] == "create-cli-slack-mcp-v3"
    assert task["steps"][1].cli_artifact_id == "cli-slack-mcp"


def test_the_install_image_universe_keeps_its_bare_id(local_stores, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)
    agent = A2AAgent(id="claude-code", version=22, docker_image_artifact=MagicMock())

    universe = A2AAgentValidator._upload_install_test_image_fixture(agent)

    assert universe.id == "validate-install-image-claude-code-v22-1790000000"


def test_an_env_snapshot_keeps_its_bare_artifact_id_image_tag_and_key(local_stores, tmp_path, monkeypatch):
    set_object_store(SigningObjectStore(str(tmp_path / "signing")))
    sandbox = reattach_for_snapshot(monkeypatch, "crm-suite", 3, "crm-universe", 2)

    snapshot = asyncio.run(EnvSnapshot.create("inst-1"))

    image = DockerImageArtifact.get(snapshot.db_image_artifact_id)
    assert (image.id, image.image_name) == ("env-snapshot-crm-suite", "env-snapshot-crm-suite-crm-universe")
    assert image.tar_gz_object_url.endswith("/env-snapshots/crm-suite/crm-universe/env-snapshot-crm-suite-crm-universe.tar.gz")
    assert "docker build --platform linux/amd64 -t env-snapshot-crm-suite-crm-universe /tmp/snapshot-build" in sandbox.scripts


class _VerifierSandbox:
    async def exec_script(self, script: str) -> str:
        return ""

    async def exec_with_output(self, *args) -> tuple[int, str, str]:
        return 0, "ok", ""


@pytest.mark.parametrize("instance_id, run", [
    ("verify-triage-1-abcd1234", "verify-triage-1-abcd1234-1790000000-0123456789ab"),
    (None, "1790000000-0123456789ab"),
])
def test_verifier_outputs_keep_their_bare_ids(local_stores, monkeypatch, instance_id, run):
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=_VerifierSandbox()))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    monkeypatch.setattr(verifier_module.time, "time", lambda: 1_790_000_000.0)
    monkeypatch.setattr(verifier_module.uuid, "uuid4", lambda: uuid.UUID(hex="0123456789ab" + "0" * 20))
    ids: list[str] = []

    def record(text, artifact_id, description, s3_url):
        ids.append(artifact_id)
        return SimpleNamespace(id=artifact_id, version=1, object_url=s3_url)

    monkeypatch.setattr(verifier_module.RunContainerUnitTestsVerifierTaskStep, "_upload_text_artifact", staticmethod(record))
    ctx = TaskStepContext(instance_id=instance_id)
    ctx.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    ctx.deployed_sandboxes.append(DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm"))
    step = verifier_module.RunContainerUnitTestsVerifierTaskStep(id="scrape", version=None, sandbox_name="h", container_name="c", command="true")

    asyncio.run(step.execute(ctx))

    assert ids == [f"verifier-stdout-scrape-{run}", f"verifier-stderr-scrape-{run}"]


def test_collected_files_keep_their_bare_ids(local_stores, monkeypatch):
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=CollectingVm()),
                               close=lambda: asyncio.sleep(0))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    ctx = TaskStepContext(instance_id="triage-task-1-abcd1234")
    ctx.deployed_sandboxes = [DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode="vm")]
    step = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", base_path="/app/artifact",
                                    artifact_paths=["report.pdf", "sub/notes.md"])

    asyncio.run(step.execute(ctx))

    universe = FileArtifactUniverse.get(ctx.metadata["file_artifact_universe"]["id"])
    assert universe.id == "triage-task-1-abcd1234"
    assert universe.file_artifact_ids == {
        "report.pdf": "triage-task-1-abcd1234-report.pdf", "sub/notes.md": "triage-task-1-abcd1234-sub_notes.md",
    }


def test_roundtrip_exports_keep_their_bare_ids(local_stores):
    step = VerifyUniverseLoadExportRoundtripStep(id="rt", version=None, env_id="crm-suite", universe_artifact_id="crm-universe")

    universe = step._create_universe_artifact([SimpleNamespace(environment_name="slack")], {"slack": {"messages": []}},
                                              env_version=3, universe_version=7)

    (service,) = universe.get_environment_artifacts()
    assert universe.id == "validate-crm-suite-v3-crm-universe-v7-export1"
    assert service.id == "validate-crm-suite-v3-crm-universe-v7-export1-svc-slack"
    assert service.get_file_artifact().id == "validate-crm-suite-v3-crm-universe-v7-export1-slack"


@pytest.mark.parametrize("env_id, universe_id, expected", [
    ("crm-suite", "crm-universe", "validate-universe-compat-crm-suite-v3-crm-universe-v2"),
    (LOCAL_ENV, "crm-universe", f"{LOCAL_ENV}__validate-universe-compat-v3-crm-universe-v2"),
    (LOCAL_ENV, LOCAL_UNIVERSE, f"{LOCAL_ENV}__validate-universe-compat-v3-local-work-triage-artifacts-seed-ddf2904a1148-v2"),
    ("crm-suite", LOCAL_UNIVERSE, f"{LOCAL_UNIVERSE}__validate-universe-compat-v2-crm-suite-v3"),
], ids=["bare", "local-env", "both-local", "local-universe"])
def test_a_compatibility_validation_is_owned_by_its_local_side(env_id, universe_id, expected):
    assert VerifyUniverseLoadExportRoundtripStep.validation_id("validate-universe-compat", env_id, 3, universe_id, 2) == expected


@pytest.mark.parametrize("universe_id, name, valid", [
    ("crm-universe", "crm-universe-files", True),
    ("Crm_Universe", "Crm_Universe-files", False),
    ("x" * 60, "x" * 60 + "-files", False),
    (LOCAL_UNIVERSE, "seed-ddf2904a1148-files", True),
    ("@local/~/work/triage/artifacts/" + "Ticket Desk (EU) " * 5 + "x",
     "ticket-desk-eu-ticket-desk-eu-ticket-desk-eu-e6f6e0683a17-files", True),
    ("@local/~/work/日本", "e3629cd92c57-files", True),
], ids=["bare", "bare-mixed-case", "bare-60-chars", "local", "local-long", "local-no-slug"])
def test_a_loaded_universes_skill_is_named_after_it(universe_id, name, valid):
    skill = _build_skill_for_loaded_file_artifact_universe(universe_id, {"destination_path": "/app/files"})
    assert skill.name == name
    if valid:
        skill.validate()
    else:
        with pytest.raises(ValueError, match="does not match the spec"):
            skill.validate()


@pytest.mark.parametrize("task_id, universe_id", [
    ("triage", "VPC Endpoints"),
    ("@local/~/work/triage/tasks/t", "@local/~/work/triage/tasks/t-v2-VPC Endpoints"),
])
def test_a_seeded_runs_universe_is_named_after_its_seed(tmp_path, monkeypatch, task_id, universe_id):
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


@pytest.mark.parametrize(("collects", "exit_code", "runs"), [(True, 1, 0), (False, 0, 2)], ids=["collects", "doesnt-collect"])
def test_a_seed_that_cant_name_an_local_universe_stops_a_collecting_run_batch_before_any_run(
        tmp_path, monkeypatch, collects, exit_code, runs):
    ran: list[TaskStepContext] = []
    collect = CollectArtifactsTaskStep(id="collect", version=None, sandbox_name="box", artifact_paths=["report.pdf"])

    class _Task:
        id, version, steps = "@local/~/work/triage/tasks/t", 2, [collect] if collects else []

        async def run(self, context, **kwargs):
            ran.append(context)
            return context

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: _Task()))
    (tmp_path / "seeds.csv").write_text("name\nfine\nQ&A: onboarding\n")

    result = CliRunner().invoke(cli, ["task", "run-batch", "--id", _Task.id, "--seeds", str(tmp_path / "seeds.csv"),
                                      "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == exit_code, result.output
    assert ("seed 2 can't name an @local universe" in result.output) is collects
    assert len(ran) == runs
