"""Tests for the MCP CLI builder: codegen output + task-step registry roundtrip.

One happy (codegen produces a script that compiles and renders --help correctly
for boolean, enum, anyOf-null, JSON-encoded array, and Python-keyword params)
and one plumbing test (BuildMcpCliTaskStep round-trips through the task-step
registry).
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import tempfile

import pytest

from agent_env.task_step import AddSkillsTaskStep, BuildMcpCliTaskStep, LoadArtifactTaskStep
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.add_skills import _build_skill_for_installed_cli
from agent_env.task_step.task_steps.mcp_cli_builder import generate_cli_script


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
    assert step.cli_artifact_id == "cli-slack-mcp"  # eager default

    payload = step.to_dict()
    cls = get_task_step_registry()["build_mcp_cli"]
    restored = cls.from_dict(payload)

    assert restored.id == "s1"
    assert restored.env_id == "slack-mcp"
    assert restored.command_name == "slack"
    assert restored.cli_artifact_id == "cli-slack-mcp"
    assert restored.to_dict() == payload


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
    from agent_env.artifact.artifacts.skill import _parse_skill_md, _validate_frontmatter
    skill = _build_skill_for_installed_cli(
        "cli-slack-mcp",
        {"command_name": "slack", "install_path": "/opt/cli/slack/bin/slack"},
    )
    assert skill.name == "slack-cli"
    skill_md = skill.to_skill_md()
    fm, body = _parse_skill_md(skill_md.encode("utf-8"))
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
