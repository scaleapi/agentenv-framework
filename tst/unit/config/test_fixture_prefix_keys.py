"""The fixture prefix reaches every key core builds by hand for artifacts and validator fixtures,
not only the ones ArtifactStore builds, so a control plane sharing a bucket keeps to its prefix."""

import asyncio
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, set_object_store
from agent_env.env.snapshot_store import _snapshot_servicedb
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import TrajectoryFilter
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.store.fakes import SigningObjectStore

_AGENT = SimpleNamespace(id="solver", version=3)


@pytest.fixture
def prefixed(local_stores, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_FIXTURE_PREFIX", "fx")
    configure()
    reset_artifact_store()
    return get_config()


def _prefixed_url(config, key: str) -> str:
    return config.get_object_store().object_url(f"fx/{key}")


def test_a_skill_bundle_is_written_under_the_prefix(prefixed, tmp_path):
    skill_dir = tmp_path / "review"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: review\ndescription: Reviews code.\n---\nBody\n")
    skill = SkillArtifact.put("review", skill_dir=skill_dir)
    assert skill.skill_object_url.startswith(_prefixed_url(prefixed, "artifacts/skill/review/1"))
    assert prefixed.get_object_store().exists("fx/artifacts/skill/review/1/SKILL.md")


def test_a_cli_bundle_is_written_under_the_prefix(prefixed, tmp_path):
    cli_dir = tmp_path / "slack"
    cli_dir.mkdir()
    (cli_dir / "run.sh").write_text("#!/bin/sh\n")
    cli = CliArtifact.put("slack-cli", command_name="slack", entrypoint="run.sh", cli_dir=cli_dir)
    assert cli.cli_object_url.startswith(_prefixed_url(prefixed, "artifacts/cli/slack-cli/1"))


def test_the_validator_fixtures_are_written_under_the_prefix(prefixed, tmp_path):
    set_object_store(SigningObjectStore(str(tmp_path / "signing")))
    store = get_config().get_object_store()
    skill_url = A2AAgentValidator._upload_skill_fixture(_AGENT, name="probe", description="d", body="b")
    assert skill_url == store.object_url("fx/a2a_validator/validator_skill/solver-v3/probe/")
    fixtures = A2AAgentValidator._upload_probe_fixtures(_AGENT, skill_url)
    assert fixtures.png_object_uri == store.object_url("fx/a2a_validator/probe_fixtures/solver-v3/red.png")
    image = A2AAgentValidator._upload_install_test_image_fixture(_AGENT)
    assert image.bundle_object_url.startswith(store.object_url("fx/a2a_validator/install_test_image/"))


def test_a_default_trajectory_prefix_is_under_the_prefix_and_an_explicit_one_is_kept(prefixed):
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver")
    assert step._trajectory_prefix() == _prefixed_url(prefixed, f"prompt_agent_trajectories/prompt_id={step.prompt_id}/")
    explicit = "s3://elsewhere/trajectories/"
    kept = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver", trajectory_output_prefix=explicit
    )
    assert kept._trajectory_prefix() == explicit


class _Sandbox:
    """Answers the servicedb container lookup and records every script; nothing runs."""

    def __init__(self) -> None:
        self.scripts: list[str] = []

    async def exec_script(self, script: str) -> str:
        self.scripts.append(script)
        return "container-1\n"


def test_an_env_snapshot_is_uploaded_under_the_prefix(prefixed, tmp_path):
    set_object_store(SigningObjectStore(str(tmp_path / "signing")))
    sandbox = _Sandbox()
    url = asyncio.run(_snapshot_servicedb(sandbox, "slack-env", "acme", lambda *args: None))
    key = "fx/env-snapshots/slack-env/acme/env-snapshot-slack-env-acme.tar.gz"
    assert url == get_config().get_object_store().object_url(key)
    assert any(f"https://objects.example.test/{key}" in script for script in sandbox.scripts)


def test_a_compacted_trajectory_is_written_under_the_prefix(prefixed):
    store = prefixed.get_object_store()
    raw = store.put("fx/prompt_agent_trajectories/raw.json", b"[]", content_type="application/json")
    verifier = RubricsVerifierTaskStep(
        id="v", version=1, criteria=[{"id": "c", "description": "d"}], prompt_id="p1", verifier_id="vid"
    )
    compact_url, _ = verifier._filter_trajectory(raw, TrajectoryFilter())
    assert store.get_object_key(compact_url).startswith("fx/compacted-trajectories/")
