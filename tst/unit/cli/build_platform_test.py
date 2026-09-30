"""Each put command builds through ``build_image`` with its ``--platform``, and a build that fails is reported
as one line, not a traceback."""

import importlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.env.envs.website_browser import PLAYWRIGHT_MCP_VERSION, WEBSITE_BROWSER_IMAGE_TAG
from agent_env.utils.docker_build import DockerBuildError

gateway = importlib.import_module("agent_env.cli.env.gateway")  # the packages export the click commands
service_db = importlib.import_module("agent_env.cli.env.service_db")
website_browser = importlib.import_module("agent_env.cli.env.website_browser")


def test_platform_option_wired_on_mcp_server_put():
    result = CliRunner().invoke(cli, ["env", "mcp-server", "put", "--help"])
    assert result.exit_code == 0
    assert "--platform" in result.output
    assert "linux/amd64" in result.output  # default shown


def _dockerfile(tmp_path, name="Dockerfile"):
    path = tmp_path / name
    path.write_text("FROM scratch\n")
    return path


def _put_commands(tmp_path):
    """Each put command, and the first build it makes: its Dockerfile, context and tag, and its build args."""
    dockerfile = _dockerfile(tmp_path)
    backend, frontend = _dockerfile(tmp_path, "backend.Dockerfile"), _dockerfile(tmp_path, "frontend.Dockerfile")
    return {
        "a2a-agent": ("agent_env.cli.a2a_agent.put", ["a2a-agent", "put", "--id", "x", "--dockerfile", str(dockerfile)],
                      (dockerfile, tmp_path, "a2a-agent-x"), None),
        "mcp-server": ("agent_env.cli.env.mcp_server",
                       ["env", "mcp-server", "put", "--id", "x", "--environment-name", "items", "--dockerfile",
                        str(dockerfile)], (dockerfile, tmp_path, "mcp-server-x"), None),
        "website": ("agent_env.cli.env.website",
                    ["env", "website", "put", "--id", "x", "--environment-name", "shop", "--backend-dockerfile", str(backend),
                     "--frontend-dockerfile", str(frontend)], (backend, tmp_path, "website-backend-x"), None),
        "gateway": ("agent_env.cli.env.gateway", ["env", "gateway", "put", "--id", "x"],
                    (gateway.GATEWAY_DOCKERFILE, gateway.GATEWAY_CONTEXT, gateway.GATEWAY_IMAGE_TAG), None),
        "service-db": ("agent_env.cli.env.service_db", ["env", "service-db", "put", "--id", "x"],
                       (service_db.SERVICE_DB_DOCKERFILE, service_db.SERVICE_DB_DOCKERFILE.parent,
                        service_db.SERVICE_DB_IMAGE_NAME), None),
        "website-browser": ("agent_env.cli.env.website_browser", ["env", "website-browser", "put", "--id", "x"],
                            (website_browser.WEBSITE_BROWSER_DOCKERFILE, website_browser.WEBSITE_BROWSER_CONTEXT,
                             WEBSITE_BROWSER_IMAGE_TAG), {"PLAYWRIGHT_MCP_VERSION": PLAYWRIGHT_MCP_VERSION}),
    }


@pytest.mark.parametrize("name", ["a2a-agent", "mcp-server", "website", "gateway", "service-db", "website-browser"])
def test_each_put_builds_through_build_image_and_a_failed_build_is_one_line(tmp_path, name):
    module, argv, (dockerfile, context, tag), build_args = _put_commands(tmp_path)[name]

    with patch(f"{module}.build_image", side_effect=DockerBuildError(f"docker build of {tag} failed (exit 1):\nboom")) \
            as build:
        result = CliRunner().invoke(cli, [*argv, "--platform", "linux/arm64"])

    expected = {"platform": "linux/arm64", **({"build_args": build_args} if build_args else {})}
    assert build.call_args.args == (dockerfile, context, tag)
    assert build.call_args.kwargs == expected
    assert result.exit_code == 1
    assert result.output.endswith(f"Error: docker build of {tag} failed (exit 1):\nboom\n")
    assert "Traceback" not in result.output
