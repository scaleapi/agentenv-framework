"""Tests for the shared docker build --platform CLI option."""

from click.testing import CliRunner

from agent_env.cli.utils import DEFAULT_BUILD_PLATFORM, docker_build_platform_args


def test_default_platform_preserves_amd64():
    # Backward compatibility: unchanged callers still build amd64.
    assert DEFAULT_BUILD_PLATFORM == "linux/amd64"
    assert docker_build_platform_args(DEFAULT_BUILD_PLATFORM) == ["--platform", "linux/amd64"]


def test_explicit_arm64_platform():
    assert docker_build_platform_args("linux/arm64") == ["--platform", "linux/arm64"]


def test_empty_platform_builds_host_native():
    # Empty / None omits the flag entirely so docker uses the host platform.
    assert docker_build_platform_args("") == []
    assert docker_build_platform_args(None) == []


def test_platform_option_wired_on_mcp_server_put():
    from agent_env.cli.env.mcp_server import put as mcp_server_put

    result = CliRunner().invoke(mcp_server_put, ["--help"])
    assert result.exit_code == 0
    assert "--platform" in result.output
    assert "linux/amd64" in result.output  # default shown
