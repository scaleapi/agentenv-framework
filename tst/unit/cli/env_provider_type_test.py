"""`env mcp-server put --env-provider-type` reaches MCPServerEnv.put on both build paths, defaults to gateway, and refuses an
unknown type before anything is built. The build and the SDK put are patched out, as in environment_name_alias_test."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli import cli

_DECLARED = [(["--env-provider-type", "server"], "server"), ([], "gateway")]


@pytest.mark.parametrize("flag, put_with", _DECLARED, ids=["server", "default"])
def test_the_local_build_puts_the_declared_type(tmp_path, flag, put_with):
    captured, output = _local_put(tmp_path, flag)
    assert captured.get("env_provider_type") == put_with and f"env_provider_type={put_with}" in output


@pytest.mark.parametrize("flag, put_with", _DECLARED, ids=["server", "default"])
def test_the_github_build_puts_the_declared_type(flag, put_with):
    captured, output = _github_put(flag)
    assert captured.get("env_provider_type") == put_with and f"env_provider_type={put_with}" in output


def test_an_unknown_type_is_refused_before_anything_is_built(tmp_path):
    with patch("agent_env.cli.env.mcp_server.build_image") as build, patch("agent_env.cli.env.mcp_server.MCPServerEnv.put") as put:
        result = CliRunner().invoke(cli, [*_PUT, "--dockerfile", str(_dockerfile(tmp_path)), "--env-provider-type", "vm"])
    assert result.exit_code == 2 and "Invalid value for '--env-provider-type'" in result.output
    assert (build.called, put.called) == (False, False)


_PUT = ["env", "mcp-server", "put", "--id", "x", "--environment-name", "items"]


def _local_put(tmp_path, flag: list[str]) -> tuple[dict, str]:
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _put_env(kwargs)

    with patch("agent_env.cli.env.mcp_server.build_image"), \
            patch("agent_env.cli.env.mcp_server.DockerImageArtifact.put", return_value=MagicMock(id="a", version=1)), \
            patch("agent_env.cli.env.mcp_server.detect_env_metadata", return_value={}), \
            patch("agent_env.cli.env.mcp_server.MCPServerEnv.put", side_effect=_capture):
        result = CliRunner().invoke(cli, [*_PUT, "--dockerfile", str(_dockerfile(tmp_path)), *flag])
    return captured, result.output


def _github_put(flag: list[str]) -> tuple[dict, str]:
    captured = {}

    async def _fake_put(**kwargs):
        captured.update(kwargs)
        return _put_env(kwargs)

    with patch("agent_env.cli.env.mcp_server.MCPServerEnv.put_from_github", side_effect=_fake_put):
        result = CliRunner().invoke(cli, [*_PUT, "--dockerfile-github-url", "https://github.com/o/r/tree/main/Dockerfile", *flag])
    return captured, result.output


def _put_env(kwargs: dict):
    return MagicMock(id="x", version=1, environment_name="items", env_provider_type=kwargs.get("env_provider_type"))


def _dockerfile(tmp_path):
    path = tmp_path / "Dockerfile"
    path.write_text("FROM scratch\n")
    return path
