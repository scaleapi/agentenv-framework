"""The pre-rename CLI spellings are GONE. These guards stop them coming back.

Removing a CLI surface is not like removing a Python symbol: nothing imports it, so nothing
fails at build time. A legacy spelling that survives here is invisible until someone notices
the CLI still answers to it, and a canonical spelling accidentally dropped alongside its alias
is invisible until a script breaks. Both directions are asserted.

Also pins the ServiceDB surface, a permanent exemption the removal must NOT cross, and
`--service-version`: its field is deleted everywhere, and no command takes the flag.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agent_env.cli import cli

# (group path, removed noun, surviving replacement)
_REMOVED_NOUNS = [
    (["artifact"], "service", "environment"),
    (["artifact"], "service-universe", "environment-universe"),
    (["env", "mcp-server"], "load-service-artifact", "load-environment-artifact"),
    (["env", "multi"], "load-service-artifact", "load-environment-artifact"),
    (["env", "multi"], "load-service-universe-artifact", "load-environment-universe-artifact"),
    (["env", "website"], "load-service-artifact", "load-environment-artifact"),
]

# (command path, removed flag, surviving replacement)
_REMOVED_FLAGS = [
    (["artifact", "environment-universe", "put"], "--service-artifact", "--environment-artifact"),
    (["env", "mcp-server", "load-environment-artifact"], "--service-artifact-id", "--environment-artifact-id"),
    (["env", "multi", "load-environment-artifact"], "--service-artifact-id", "--environment-artifact-id"),
    (["env", "multi", "load-environment-universe-artifact"], "--service-universe-artifact-id", "--environment-universe-artifact-id"),
    (["env", "website", "load-environment-artifact"], "--service-artifact-id", "--environment-artifact-id"),
]


def _resolve(path):
    node = cli
    for part in path:
        node = node.commands[part]
    return node


@pytest.mark.parametrize("path,removed,replacement", _REMOVED_NOUNS)
def test_removed_noun_is_gone_and_replacement_survives(path, removed, replacement):
    group = _resolve(path)
    assert removed not in group.commands, f"legacy command {removed!r} is still registered"
    assert replacement in group.commands, f"replacement {replacement!r} went missing with it"
    assert group.commands[replacement].deprecated is False


@pytest.mark.parametrize("path,removed,replacement", _REMOVED_FLAGS)
def test_removed_flag_is_gone_and_replacement_survives(path, removed, replacement):
    opts = {o for p in _resolve(path).params for o in p.opts}
    assert removed not in opts, f"legacy flag {removed!r} is still accepted"
    assert replacement in opts, f"replacement {replacement!r} went missing with it"


def test_the_website_browser_has_no_put_command_of_its_own():
    """It's an MCP server env, which `env mcp-server put` writes, and the remote-store hint names that."""
    env = cli.commands["env"].commands
    assert "website-browser" not in env and "playwright-mcp" not in env
    assert "mcp-server" in env


def test_no_deprecated_command_is_registered_anywhere():
    """Catches a legacy spelling registered under a name this module does not enumerate."""

    def walk(node, path=()):
        for name, cmd in getattr(node, "commands", {}).items():
            yield path + (name,), cmd
            yield from walk(cmd, path + (name,))

    deprecated = [" ".join(p) for p, c in walk(cli) if c.deprecated]
    assert deprecated == [], f"deprecated CLI commands still registered: {deprecated}"


def test_the_alias_helper_module_is_gone():
    """`cli/aliases.py` existed only for these registrations."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_env.cli.aliases")


def test_servicedb_cli_surface_is_never_renamed():
    """Permanent exemption; a mechanical --service-* sweep would take all three."""
    assert "service-db" in cli.commands["env"].commands
    assert "environment-db" not in cli.commands["env"].commands
    deploy_flags = {o for p in cli.commands["env"].commands["deploy"].params for o in p.opts}
    assert "--service-db" in deploy_flags
    run_flags = {o for p in cli.commands["task"].commands["run"].params for o in p.opts}
    assert "--service-db-env-id" in run_flags


def test_service_version_is_gone_from_every_put():
    """No command takes the flag, and there is no `--environment-version` in its place."""
    for path in (["env", "mcp-server", "put"], ["env", "website", "put"], ["artifact", "environment", "put"]):
        opts = {o for p in _resolve(path).params for o in p.opts}
        assert "--service-version" not in opts
        assert "--environment-version" not in opts


_GITHUB = "https://github.com/o/r/tree/main/Dockerfile"

_ENV_PUTS = [
    pytest.param(["env", "mcp-server", "put", "--dockerfile-github-url", _GITHUB],
                 "agent_env.cli.env.mcp_server.MCPServerEnv.put_from_github", id="mcp-server"),
    pytest.param(["env", "website", "put", "--backend-dockerfile-github-url", _GITHUB, "--frontend-dockerfile-github-url", _GITHUB],
                 "agent_env.cli.env.website.WebsiteEnv.put_from_github", id="website"),
]


@pytest.mark.parametrize("argv,put_target", _ENV_PUTS)
def test_service_version_is_refused_before_anything_is_built(argv, put_target):
    with patch(put_target) as put:
        result = CliRunner().invoke(cli, [*argv, "--id", "x", "--environment-name", "email", "--service-version", "2"])
    assert result.exit_code == 2, result.output
    assert "No such option" in result.output and "--service-version" in result.output
    put.assert_not_called()
