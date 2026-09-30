"""`env mcp-server put` and `env multi put` only register by default. `--validate` opts into
validation (the release gate, for an MCP server), `--override` implies it, and `--skip-validation`
is not an option. The build, the SDK put and validation are patched out."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli import cli

_MCP_PUT = ["env", "mcp-server", "put", "--id", "x", "--environment-name", "items"]
_MULTI_PUT = ["env", "multi", "put", "--id", "crm-suite", "--mcp-server", "slack"]
_BUILDS = ["local", "github"]


@pytest.mark.parametrize("build", _BUILDS)
def test_mcp_server_put_registers_without_validating(tmp_path, build):
    result, gate = _mcp_put(tmp_path, build)
    assert result.exit_code == 0, result.output
    assert "Created MCPServerEnv" in result.output and not gate.called


@pytest.mark.parametrize("build", _BUILDS)
def test_validate_runs_the_release_gate(tmp_path, build):
    result, gate = _mcp_put(tmp_path, build, "--validate")
    assert result.exit_code == 0, result.output
    assert gate.call_count == 1 and gate.call_args.args[2] is False


@pytest.mark.parametrize("build", _BUILDS)
def test_override_runs_the_release_gate_without_validate(tmp_path, build):
    result, gate = _mcp_put(tmp_path, build, "--override")
    assert result.exit_code == 0, result.output
    assert gate.call_count == 1 and gate.call_args.args[2] is True


def test_mcp_server_put_has_no_skip_validation(tmp_path):
    result, gate = _mcp_put(tmp_path, "local", "--skip-validation")
    assert result.exit_code == 2 and "No such option '--skip-validation'" in result.output
    assert not gate.called


def test_multi_put_registers_without_validating():
    result, multi_env = _multi_put()
    assert result.exit_code == 0, result.output
    assert "Created MultiEnv" in result.output and not multi_env.validate.called


def test_multi_put_validates_with_validate():
    result, multi_env = _multi_put("--validate")
    assert result.exit_code == 0, result.output
    assert multi_env.validate.await_count == 1 and "Validation task: inst-1" in result.output


def test_multi_put_has_no_skip_validation():
    result, multi_env = _multi_put("--skip-validation")
    assert result.exit_code == 2 and "No such option '--skip-validation'" in result.output
    assert not multi_env.validate.called


def _mcp_put(tmp_path, build: str, *flags: str):
    env = MagicMock(id="x", version=1, environment_name="items", env_provider_type="gateway")
    with patch("agent_env.cli.env.mcp_server._gate_release") as gate:
        if build == "github":
            with patch("agent_env.cli.env.mcp_server.MCPServerEnv.put_from_github", AsyncMock(return_value=env)):
                result = CliRunner().invoke(
                    cli, [*_MCP_PUT, "--dockerfile-github-url", "https://github.com/o/r/tree/main/Dockerfile", *flags])
            return result, gate
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        with patch("agent_env.cli.env.mcp_server.build_image"), \
                patch("agent_env.cli.env.mcp_server.DockerImageArtifact.put", return_value=MagicMock(id="a", version=1)), \
                patch("agent_env.cli.env.mcp_server.detect_env_metadata", return_value={}), \
                patch("agent_env.cli.env.mcp_server.MCPServerEnv.put", return_value=env):
            result = CliRunner().invoke(cli, [*_MCP_PUT, "--dockerfile", str(dockerfile), *flags])
    return result, gate


def _multi_put(*flags: str):
    server = MagicMock(type="mcp_server", id="slack", version=3)
    multi_env = MagicMock(id="crm-suite", version=1, validate=AsyncMock(return_value="inst-1"))
    with patch("agent_env.cli.env.multi.Env.get", return_value=server), \
            patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
            patch("agent_env.cli.env.multi.MultiEnv.put", return_value=multi_env):
        result = CliRunner().invoke(cli, [*_MULTI_PUT, *flags])
    return result, multi_env
