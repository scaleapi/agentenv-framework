"""Tests for the MCP CLI builder: codegen output + task-step registry roundtrip.

One happy (codegen produces a script that compiles and renders --help correctly
for boolean, enum, anyOf-null, JSON-encoded array, and Python-keyword params)
and one plumbing test (BuildMcpCliTaskStep round-trips through the task-step
registry).
"""

from __future__ import annotations

import asyncio
import os
import py_compile
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from agent_env.artifact.store import ArtifactStore
from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step import AddSkillsTaskStep, BuildMcpCliTaskStep, LoadArtifactTaskStep
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.add_skills import _build_skill_for_installed_cli
from agent_env.task_step.task_steps.mcp_cli_builder import build_mcp_cli, generate_cli_script
from tst.unit.event_loop_probe import on_event_loop


def test_codegen_compiles_and_renders_help():
    """The generated script compiles and its group --help renders the header.

    Tools are now discovered from the gateway at runtime (not baked in at codegen
    time), so subcommands don't appear in --help without a reachable gateway. With
    a gateway URL set but unreachable, list_commands swallows the fetch error and
    --help still renders (exit 0). Per-tool rendering is covered by integration tests.
    """
    src = generate_cli_script("slack")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "slack")
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        os.chmod(path, 0o755)

        env = {"PATH": os.environ["PATH"], "AGENT_ENV_GATEWAY_URL": "http://127.0.0.1:1"}
        group_help = subprocess.run([path, "--help"], capture_output=True, text=True, env=env, timeout=15)
        assert group_help.returncode == 0
        assert "Auto-generated CLI for slack" in group_help.stdout


def test_codegen_loads_sibling_env_file():
    src = generate_cli_script("svc")

    with tempfile.TemporaryDirectory() as tmp:
        bin_dir = os.path.join(tmp, "bin")
        os.makedirs(bin_dir)
        script_path = os.path.join(bin_dir, "svc")
        with open(script_path, "w") as f:
            f.write(src)
        os.chmod(script_path, 0o755)

        clean_env = {"PATH": os.environ["PATH"]}

        # No .env, no env var -> exits 2 with error mentioning the var name and the .env path
        result = subprocess.run([script_path, "ping"], capture_output=True, text=True, env=clean_env)
        assert result.returncode == 2
        assert "AGENT_ENV_GATEWAY_URL" in result.stderr
        assert ".env" in result.stderr

        # With .env containing the URL, no env var -> script tries to connect (no "set ... or write" error)
        env_path = os.path.join(bin_dir, ".env")
        with open(env_path, "w") as f:
            f.write("AGENT_ENV_GATEWAY_URL=http://127.0.0.1:1\n")
        result = subprocess.run([script_path, "ping"], capture_output=True, text=True, env=clean_env, timeout=15)
        assert result.returncode != 0
        # Critical: it didn't bail with "set ... or write KEY=VALUE pairs"; it actually tried to call.
        assert "or write KEY=VALUE pairs" not in result.stderr

        # Env var beats .env file (priority check). With both set, set env var to a different URL —
        # both will fail to connect, but we just need to confirm the script reached the call path.
        result = subprocess.run([script_path, "ping"], capture_output=True, text=True,
                                env={**clean_env, "AGENT_ENV_GATEWAY_URL": "http://127.0.0.1:2"}, timeout=15)
        assert result.returncode != 0
        assert "or write KEY=VALUE pairs" not in result.stderr


def test_build_mcp_cli_step_dict_roundtrip():
    step = BuildMcpCliTaskStep(id="s1", version=None, env_id="slack-mcp", command_name="slack")
    assert step.cli_artifact_id == "slack-mcp__cli"  # eager default

    payload = step.to_dict()
    cls = get_task_step_registry()["build_mcp_cli"]
    restored = cls.from_dict(payload)

    assert restored.id == "s1"
    assert restored.env_id == "slack-mcp"
    assert restored.command_name == "slack"
    assert restored.cli_artifact_id == "slack-mcp__cli"
    assert restored.to_dict() == payload



@pytest.mark.asyncio
async def test_build_mcp_cli_needs_a_gateway():
    """The CLI it generates talks to the gateway's /step and /state, so an env deployed without one is refused up front."""
    from agent_env.env.env import DeployedSandboxEnv, EnvNeedsGateway
    from agent_env.task_step.context import TaskStepContext

    step = BuildMcpCliTaskStep(id="s1", version=None, env_id="slack-bare", command_name="slack")
    record = DeployedSandboxEnv(env_id="slack-bare", env_version=1, sandbox_id="srv")
    with pytest.raises(EnvNeedsGateway, match="build_mcp_cli needs a gateway; env 'slack-bare' was deployed without one"):
        await step.execute(TaskStepContext(deployed_envs=[record]))

def test_add_skills_step_roundtrip_with_cli_artifact_ids():
    step = AddSkillsTaskStep(id="s1", version=None, skills=[], cli_artifact_ids=["cli-foo", "cli-bar"], agent_name="my-agent")
    payload = step.to_dict()
    cls = get_task_step_registry()["add_skills"]
    restored = cls.from_dict(payload)
    assert restored.cli_artifact_ids == ["cli-foo", "cli-bar"]
    assert restored.skills == []
    assert restored.agent_name == "my-agent"
    assert restored.to_dict() == payload


def test_build_skill_for_installed_cli():
    from agent_env.artifact.artifacts.skill import _validate_frontmatter, parse_skill_md
    skill = _build_skill_for_installed_cli(
        "cli-slack-mcp",
        {"command_name": "slack", "install_path": "/opt/cli/slack/bin/slack"},
    )
    assert skill.name == "slack-cli"
    skill_md = skill.to_skill_md()
    fm, body = parse_skill_md(skill_md.encode("utf-8"))
    _validate_frontmatter(fm, expected_name="slack-cli")
    assert "/opt/cli/slack/bin/slack" in body
    assert "--help" in body


def test_load_artifact_step_roundtrip_with_agent_name():
    step = LoadArtifactTaskStep(id="s1", version=None, env_id="slack-mcp", artifact_id="cli-slack-mcp", agent_name="my-agent")
    assert step.artifacts == [{"id": "cli-slack-mcp", "version": None}]
    payload = step.to_dict()
    cls = get_task_step_registry()["load_artifact"]
    restored = cls.from_dict(payload)
    assert restored.env_id == "slack-mcp"
    assert restored.artifacts == [{"id": "cli-slack-mcp", "version": None}]
    assert restored.agent_name == "my-agent"
    assert restored.to_dict() == payload

    step_no_agent = LoadArtifactTaskStep(id="s2", version=None, env_id="x", artifact_id="y")
    assert step_no_agent.agent_name is None
    assert restored.from_dict(step_no_agent.to_dict()).agent_name is None


def test_load_artifact_step_plural_artifacts():
    step = LoadArtifactTaskStep(
        id="s3", version=None, env_id="multi-x",
        artifacts=[{"id": "cli-a", "version": 1}, {"id": "cli-b", "version": None}],
        agent_name="agent-1",
    )
    payload = step.to_dict()
    assert payload["artifacts"] == [{"id": "cli-a", "version": 1}, {"id": "cli-b", "version": None}]
    cls = get_task_step_registry()["load_artifact"]
    restored = cls.from_dict(payload)
    assert restored.artifacts == step.artifacts
    assert restored.to_dict() == payload


def test_load_artifact_step_legacy_dict_compat():
    legacy = {
        "id": "old", "type": "load_artifact", "version": None,
        "env_id": "e", "artifact_id": "a", "artifact_version": 7,
        "agent_name": "agent-1",
    }
    cls = get_task_step_registry()["load_artifact"]
    restored = cls.from_dict(legacy)
    assert restored.artifacts == [{"id": "a", "version": 7}]
    assert restored.agent_name == "agent-1"
    new = restored.to_dict()
    assert "artifact_id" not in new and "artifact_version" not in new
    assert new["artifacts"] == [{"id": "a", "version": 7}]


def test_load_artifact_step_constructor_validation():
    with pytest.raises(ValueError, match="not both"):
        LoadArtifactTaskStep(id="s", version=None, env_id="e",
                              artifacts=[{"id": "a", "version": None}], artifact_id="a")
    with pytest.raises(ValueError, match="Must pass at least one"):
        LoadArtifactTaskStep(id="s", version=None, env_id="e")


@pytest.mark.asyncio
async def test_build_mcp_cli_uploads_the_bundle_off_the_event_loop(monkeypatch):
    on_loop: list[bool] = []

    async def no_manifest(gateway_url, environment_name):
        return None

    def put(**kw):
        on_loop.append(on_event_loop())
        return SimpleNamespace(id=kw["id"], version=1, type="cli", command_name=kw["command_name"])

    monkeypatch.setattr(build_mcp_cli.Env, "get", staticmethod(lambda id, version=None: SimpleNamespace(environment_name="slack")))
    monkeypatch.setattr(build_mcp_cli, "fetch_interface_manifest", no_manifest)
    monkeypatch.setattr(build_mcp_cli.CliArtifact, "put", staticmethod(put))
    record = DeployedGatewayEnv(env_id="slack-mcp", env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None, sandbox_id="sb")

    await BuildMcpCliTaskStep(id="s1", version=None, env_id="slack-mcp", command_name="slack").execute(
        TaskStepContext(deployed_envs=[record])
    )

    assert on_loop == [False]


def _gateway_record(env_id: str = "slack-mcp") -> DeployedGatewayEnv:
    return DeployedGatewayEnv(env_id=env_id, env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None, sandbox_id="sb")


def _no_gateway(monkeypatch) -> None:
    async def no_manifest(gateway_url, environment_name):
        return None

    monkeypatch.setattr(build_mcp_cli.Env, "get", staticmethod(lambda id, version=None: SimpleNamespace(environment_name="slack")))
    monkeypatch.setattr(build_mcp_cli, "fetch_interface_manifest", no_manifest)


@pytest.mark.asyncio
async def test_a_cancel_mid_upload_leaves_the_bundle_to_the_upload(monkeypatch, caplog):
    _no_gateway(monkeypatch)
    uploading, release, dirs = threading.Event(), threading.Event(), []

    def put(**kw):
        dirs.append(kw["cli_dir"])
        uploading.set()
        release.wait(5)
        assert (kw["cli_dir"] / "bin" / kw["command_name"]).is_file()
        return SimpleNamespace(id=kw["id"], version=1, type="cli", command_name=kw["command_name"])

    monkeypatch.setattr(build_mcp_cli.CliArtifact, "put", staticmethod(put))
    step = BuildMcpCliTaskStep(id="s1", version=None, env_id="slack-mcp", command_name="slack")
    build = asyncio.create_task(step.execute(TaskStepContext(deployed_envs=[_gateway_record()])))
    await asyncio.to_thread(uploading.wait, 5)
    build.cancel()
    with pytest.raises(asyncio.CancelledError):
        await build
    with caplog.at_level("INFO", logger="agent_env.task_step.thread_work"):
        release.set()
        for _ in range(100):
            if "finished after its caller was cancelled" in caplog.text:
                break
            await asyncio.sleep(0.02)
    assert "Uploading CLI slack-mcp__cli finished after its caller was cancelled" in caplog.text
    assert not dirs[0].exists()


@pytest.mark.asyncio
async def test_concurrent_builds_of_one_cli_each_get_their_own_version(local_stores, monkeypatch):
    _no_gateway(monkeypatch)
    next_version = ArtifactStore.next_version

    def slow_next_version(self, id):
        version = next_version(self, id)
        time.sleep(0.05)  # widen the window between taking a version and writing at it
        return version

    monkeypatch.setattr(ArtifactStore, "next_version", slow_next_version)
    step = BuildMcpCliTaskStep(id="s1", version=None, env_id="slack-mcp", command_name="slack", cli_artifact_id="cli-shared")
    contexts = await asyncio.gather(*(step.execute(TaskStepContext(deployed_envs=[_gateway_record()])) for _ in range(4)))

    assert sorted(c.metadata["cli_artifact"]["version"] for c in contexts) == [1, 2, 3, 4]
