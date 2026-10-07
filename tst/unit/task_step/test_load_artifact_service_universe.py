"""LoadArtifactTaskStep dispatch for a EnvironmentUniverseArtifact.

The split is on the TARGET, not the type:
  - aimed at an env       → restore into the live MCP services (unchanged)
  - aimed at an agent /
    container             → stage the frozen per-service exports as files, so a
                            judge can grade them without deploying an env

The staged case must also record `loaded_file_artifact_universes`, because that
is the key rubrics_verifier reads to tell its judge which files exist.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps import load_artifact as load_artifact_mod
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

UNIVERSE_ID = "snapshot-multi-slack-email-abc123"


@pytest.fixture
def universe(monkeypatch):
    """A EnvironmentUniverseArtifact returned by Artifact.get, with a stubbed file view."""
    uni = EnvironmentUniverseArtifact(id=UNIVERSE_ID, version=4, environment_artifact_refs=[])
    staged = {
        "slack/slack.json": FileArtifact(
            id="fa-slack", version=1, description="slack", filename="slack.json",
            content_type="application/json", s3_url="s3://bucket/slack.json",
        ),
    }
    monkeypatch.setattr(type(uni), "get_file_artifacts", lambda self: staged)
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: uni))
    return uni


@pytest.fixture
def agent_ctx(monkeypatch):
    """Context with one deployed agent; A2A staging is captured, not performed."""
    from agent_env.a2a_agent import A2AAgent
    from agent_env.a2a_agent import store as a2a_store

    calls: list[dict] = []

    async def _fake_load(deployed, universe, destination):
        calls.append({"universe": universe, "destination": destination})
        return {"slack/slack.json": f"{destination}/slack/slack.json"}

    monkeypatch.setattr(A2AAgent, "load_file_artifact_universe", staticmethod(_fake_load))
    monkeypatch.setattr(
        a2a_store, "get_a2a_agent_instance_store",
        lambda: type("S", (), {"get": staticmethod(lambda iid: object())})(),
    )
    ctx = TaskStepContext()
    ctx.deployed_agents = [
        DeployedAgent(agent_name="judge", api_url="http://judge", sandbox_id="sb-1", instance_id="inst-1")
    ]
    return ctx, calls


def _deployed_env(env_id: str = "multi-slack-email") -> DeployedEnv:
    return DeployedGatewayEnv(
        env_id=env_id, env_version=7, gateway_url="http://gw", mcp_url="http://gw/mcp",
        db_web_url=None, sandbox_id="sb-env", instance_id="env-inst-1",
    )


class TestStagedOntoAgent:
    @pytest.mark.asyncio
    async def test_staged_as_files_and_recorded_for_the_judge(self, universe, agent_ctx):
        ctx, calls = agent_ctx
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="judge",
            artifact_id=UNIVERSE_ID, destination_path="/tmp/env_state",
        )
        ctx = await step.execute(ctx)

        assert len(calls) == 1
        assert calls[0]["universe"] is universe
        assert calls[0]["destination"] == "/tmp/env_state"

        # `loaded_file_artifact_universes` is the key rubrics_verifier reads to
        # tell its judge which files exist; `artifact_type` distinguishes a
        # staged snapshot from a genuine FileArtifactUniverse.
        entries = ctx.metadata["loaded_file_artifact_universes"]
        assert entries == [{
            "id": UNIVERSE_ID,
            "version": 4,
            "artifact_type": "environment_universe",
            "env_id": None,
            "agent_name": "judge",
            "destination_path": "/tmp/env_state",
            "files": {"slack/slack.json": "/tmp/env_state/slack/slack.json"},
        }]


class TestStagedIntoContainer:
    @pytest.mark.asyncio
    async def test_service_universe_is_staged_into_container(self, universe, monkeypatch):
        calls: list[dict] = []

        async def _fake_container_load(sandbox, container_name, uni, destination):
            calls.append({"container": container_name, "universe": uni, "destination": destination})
            return ["slack/slack.json"]

        monkeypatch.setattr(load_artifact_mod, "_load_universe_into_container", _fake_container_load)
        monkeypatch.setattr(
            load_artifact_mod, "get_sandbox_provider",
            lambda: type("P", (), {"get_sandbox": staticmethod(_async_none)})(),
            raising=False,
        )
        from agent_env.providers.sandbox_providers import sandbox_provider

        monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda _t: _FakeProvider())

        ctx = TaskStepContext()
        ctx.deployed_sandboxes = [
            DeployedSandbox(sandbox_name="box", sandbox_id="sb-9", sandbox_mode="vm", sandbox_type="modal")
        ]
        ctx.metadata["deployed_docker_containers"] = [
            {"container_name": "verifier", "sandbox_name": "box"}
        ]
        step = LoadArtifactTaskStep(
            id="load", version=None, sandbox_name="box", container_name="verifier",
            artifact_id=UNIVERSE_ID, destination_path="/loaded/env_state",
        )
        ctx = await step.execute(ctx)

        assert calls[0]["container"] == "verifier"
        assert calls[0]["universe"] is universe
        entry = ctx.metadata["loaded_file_artifact_universes"][0]
        assert entry["container_name"] == "verifier"
        assert entry["artifact_type"] == "environment_universe"
        assert entry["files"] == ["slack/slack.json"]


class TestEnvTargetUnchanged:
    """The regression guard that matters: an env target must still RESTORE."""

    @pytest.mark.asyncio
    async def test_service_universe_with_env_id_still_loads_into_env(self, universe, monkeypatch):
        loaded: list = []

        class _FakeEnv:
            @classmethod
            async def from_deployed_env(cls, deployed):
                return cls()

            async def load_environment_universe_artifact(self, artifact):
                loaded.append(artifact)

        from agent_env.env.env import Env

        monkeypatch.setattr(Env, "get", classmethod(lambda cls, id, version=None: _FakeEnv()))

        ctx = TaskStepContext()
        ctx.deployed_envs = [_deployed_env()]
        step = LoadArtifactTaskStep(
            id="load", version=None, env_id="multi-slack-email", artifact_id=UNIVERSE_ID,
        )
        ctx = await step.execute(ctx)

        assert loaded == [universe]
        # Restoring into an env is not a file staging — nothing recorded.
        assert "loaded_file_artifact_universes" not in ctx.metadata


class TestFileArtifactUniverseUnchanged:
    @pytest.mark.asyncio
    async def test_fau_onto_agent_still_works_and_is_tagged(self, monkeypatch, agent_ctx):
        fau = FileArtifactUniverse(id="fau-1", version=2, file_artifact_ids={"a.txt": "fa-a"})
        monkeypatch.setattr(type(fau), "get_file_artifacts", lambda self: {"a.txt": None})
        monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: fau))

        ctx, calls = agent_ctx
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="judge", artifact_id="fau-1",
        )
        ctx = await step.execute(ctx)

        assert calls[0]["universe"] is fau
        assert ctx.metadata["loaded_file_artifact_universes"][0]["artifact_type"] == "file_artifact_universe"


async def _async_none(*a, **kw):
    return None


class _FakeProvider:
    async def get_sandbox(self, sandbox_id):
        return SimpleNamespace(scoped_name=lambda name: name)
