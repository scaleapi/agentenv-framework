"""`--environment-name` is the only name flag; the deprecated `--service-name`
alias is removed.

Drives the `put` commands that expose the name flag through CliRunner and asserts the
resolution contract: the new flag works, the old flag is rejected as an unknown option,
and an omitted flag is an error (unless the name can be derived from the card). The SDK
call is patched out so these stay pure argument-parsing tests (no Mongo, no Docker).
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.cli.utils import resolve_environment_name
from agent_env.utils.card_naming import card_name_from_source


# (argv prefix for the command, the patch target whose kwargs we inspect)
PUT_COMMANDS = [
    pytest.param(
        ["env", "mcp-server", "put", "--id", "x", "--dockerfile-github-url", "https://github.com/o/r/tree/main/Dockerfile"],
        "agent_env.cli.env.mcp_server.MCPServerEnv.put_from_github",
        id="mcp-server-put",
    ),
    pytest.param(
        [
            "env", "website", "put", "--id", "x",
            "--backend-dockerfile-github-url", "https://github.com/o/r/tree/main/b/Dockerfile",
            "--frontend-dockerfile-github-url", "https://github.com/o/r/tree/main/f/Dockerfile",
            "--skip-validation",
        ],
        "agent_env.cli.env.website.WebsiteEnv.put_from_github",
        id="website-put",
    ),
]


def _invoke(argv, put_target, extra):
    """Run `argv + extra` with the SDK put patched out; return (result, captured kwargs)."""
    captured = {}
    module = put_target.rsplit(".", 2)[0]

    async def _fake_put(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock(id="x", version=1, environment_name=kwargs.get("environment_name"))

    with patch(put_target, side_effect=_fake_put), \
            patch(f"{module}.card_name_from_github", return_value=None):
        result = CliRunner().invoke(cli, argv + extra)
    return result, captured


@pytest.mark.parametrize("argv,put_target", PUT_COMMANDS)
def test_environment_name_is_passed_through(argv, put_target):
    result, captured = _invoke(argv, put_target, ["--environment-name", "email"])
    assert captured.get("environment_name") == "email", result.output
    assert "service_name" not in captured
    assert "deprecated" not in result.output


@pytest.mark.parametrize("argv,put_target", PUT_COMMANDS)
def test_service_name_flag_is_rejected(argv, put_target):
    result, captured = _invoke(argv, put_target, ["--service-name", "email"])
    assert result.exit_code != 0
    assert "No such option" in result.output
    assert captured == {}


@pytest.mark.parametrize("argv,put_target", PUT_COMMANDS)
def test_neither_flag_is_an_error(argv, put_target):
    result, captured = _invoke(argv, put_target, [])
    assert result.exit_code != 0
    assert "--environment-name" in result.output
    assert captured == {}


@pytest.mark.parametrize("argv,put_target", PUT_COMMANDS)
def test_help_lists_environment_name_only(argv, put_target):
    result = CliRunner().invoke(cli, argv[:3] + ["--help"])
    assert result.exit_code == 0
    assert "--environment-name" in result.output
    assert "--service-name" not in result.output


def test_artifact_environment_put_resolves_environment_name(tmp_path):
    """`artifact environment put` is the third flag site; it writes a FileArtifact first, so
    patch at EnvironmentArtifact.put and assert the resolved kwarg."""
    payload = tmp_path / "data.json"
    payload.write_text("{}")
    captured = {}

    def _fake_service_put(*args, **kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    argv = [
        "artifact", "environment", "put", str(payload),
        "--id", "x", "--description", "d",
    ]
    with patch("agent_env.cli.artifact.environment.EnvironmentArtifact.put", side_effect=_fake_service_put), \
            patch("agent_env.cli.artifact.environment.FileArtifact.put"):
        result = CliRunner().invoke(cli, argv + ["--environment-name", "email"])
    assert captured.get("environment_name") == "email", result.output


def test_mcp_put_derives_name_from_card_when_flag_omitted(tmp_path):
    """With no name flag, the local-build put derives environment_name from the source's card."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    with patch("agent_env.cli.env.mcp_server.card_name_from_source", return_value="items") as derive, \
            patch("agent_env.cli.env.mcp_server.build_image"), \
            patch("agent_env.cli.env.mcp_server.DockerImageArtifact.put", return_value=MagicMock(id="a", version=1)), \
            patch("agent_env.cli.env.mcp_server.detect_env_metadata", return_value={}), \
            patch("agent_env.cli.env.mcp_server.MCPServerEnv.put", side_effect=_capture):
        result = CliRunner().invoke(cli, [
            "env", "mcp-server", "put", "--id", "x",
            "--dockerfile", str(dockerfile),
        ])
    assert captured.get("environment_name") == "items", result.output
    derive.assert_called_once()


def test_website_put_derives_name_from_card_when_flag_omitted(tmp_path):
    """Website put derives environment_name by reading the BACKEND source's card."""
    backend = tmp_path / "Dockerfile.backend"
    backend.write_text("FROM scratch\n")
    frontend = tmp_path / "Dockerfile.frontend"
    frontend.write_text("FROM scratch\n")
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    with patch("agent_env.cli.env.website.card_name_from_source", return_value="webitems") as derive, \
            patch("agent_env.cli.env.website.build_image"), \
            patch("agent_env.cli.env.website.DockerImageArtifact.put", return_value=MagicMock(id="a", version=1)), \
            patch("agent_env.cli.env.website.detect_env_metadata", return_value={}), \
            patch("agent_env.cli.env.website.WebsiteEnv.put", side_effect=_capture):
        result = CliRunner().invoke(cli, [
            "env", "website", "put", "--id", "x",
            "--backend-dockerfile", str(backend), "--frontend-dockerfile", str(frontend),
            "--skip-validation",
        ])
    assert captured.get("environment_name") == "webitems", result.output
    derive.assert_called_once()
    assert derive.call_args.args[0] == str(backend)


def test_github_put_derives_name_from_github_source():
    """github-url put with no flag derives environment_name from the GitHub source (no local checkout)."""
    captured = {}

    async def _fake_put(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock(id="x", version=1, environment_name=kwargs.get("environment_name"))

    with patch("agent_env.cli.env.mcp_server.MCPServerEnv.put_from_github", side_effect=_fake_put), \
            patch("agent_env.cli.env.mcp_server.card_name_from_github", return_value="email") as derive:
        result = CliRunner().invoke(cli, [
            "env", "mcp-server", "put", "--id", "x",
            "--dockerfile-github-url", "https://github.com/o/r/tree/main/svc/Dockerfile",
        ])
    assert captured.get("environment_name") == "email", result.output
    derive.assert_called_once()


class TestResolveEnvironmentName:
    """The shared resolver, unit-tested directly."""

    def test_name_resolves_when_given(self):
        assert resolve_environment_name("email") == "email"

    def test_missing_raises(self):
        import click

        with pytest.raises(click.UsageError, match="--environment-name"):
            resolve_environment_name(None)

    def test_missing_with_allow_missing_returns_none(self):
        assert resolve_environment_name(None, allow_missing=True) is None


class TestCardNameFromSource:
    """Static @environment_card(name=...) extraction from source (no build/run)."""

    def _df(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM scratch\n")
        return str(tmp_path / "Dockerfile")

    def test_single_literal(self, tmp_path):
        df = self._df(tmp_path)
        (tmp_path / "svc.py").write_text('from agentenv_protocol import environment_card\n@environment_card(name="foo")\nclass S:\n    pass\n')
        assert card_name_from_source(df, str(tmp_path)) == "foo"

    def test_aliased_decorator(self, tmp_path):
        df = self._df(tmp_path)
        (tmp_path / "svc.py").write_text('import agentenv_protocol as ep\n@ep.environment_card(name="foo")\nclass S:\n    pass\n')
        assert card_name_from_source(df, str(tmp_path)) == "foo"

    def test_none_when_absent(self, tmp_path):
        df = self._df(tmp_path)
        (tmp_path / "svc.py").write_text("class S:\n    pass\n")
        assert card_name_from_source(df, str(tmp_path)) is None

    def test_none_when_ambiguous(self, tmp_path):
        df = self._df(tmp_path)
        (tmp_path / "a.py").write_text('@environment_card(name="foo")\nclass A:\n    pass\n')
        (tmp_path / "b.py").write_text('@environment_card(name="bar")\nclass B:\n    pass\n')
        assert card_name_from_source(df, str(tmp_path)) is None

    def test_none_when_nonliteral(self, tmp_path):
        df = self._df(tmp_path)
        (tmp_path / "svc.py").write_text('N = "foo"\n@environment_card(name=N)\nclass S:\n    pass\n')
        assert card_name_from_source(df, str(tmp_path)) is None

    def test_syntax_error_file_skipped(self, tmp_path):
        df = self._df(tmp_path)
        (tmp_path / "broken.py").write_text("def (:\n")
        (tmp_path / "svc.py").write_text('@environment_card(name="foo")\nclass S:\n    pass\n')
        assert card_name_from_source(df, str(tmp_path)) == "foo"
