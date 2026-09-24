"""The pre-rename CLI spellings are GONE. These guards stop them coming back.

Removing a CLI surface is not like removing a Python symbol: nothing imports it, so nothing
fails at build time. A legacy spelling that survives here is invisible until someone notices
the CLI still answers to it, and a canonical spelling accidentally dropped alongside its alias
is invisible until a script breaks. Both directions are asserted.

Also pins the two boundaries the removal must NOT cross: the ServiceDB surface is a permanent
exemption, and `--service-version` survives on the env commands (an MCPServerEnv/WebsiteEnv
field) while it is gone from `artifact environment put` (the deleted artifact field).
"""

import pytest

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
    from agent_env.cli import cli

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


def test_no_deprecated_command_is_registered_anywhere():
    """Catches a legacy spelling registered under a name this module does not enumerate."""
    from agent_env.cli import cli

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
    from agent_env.cli import cli

    assert "service-db" in cli.commands["env"].commands
    assert "environment-db" not in cli.commands["env"].commands
    deploy_flags = {o for p in cli.commands["env"].commands["deploy"].params for o in p.opts}
    assert "--service-db" in deploy_flags
    run_flags = {o for p in cli.commands["task"].commands["run"].params for o in p.opts}
    assert "--service-db-env-id" in run_flags


def test_service_version_survives_on_envs_and_is_gone_from_the_artifact():
    """The field split: MCPServerEnv/WebsiteEnv keep `service_version`; the artifact's was
    deleted, so its flag goes with it. There is no `--environment-version` in either case."""
    from agent_env.cli import cli

    for group, cmd in (
        (cli.commands["env"].commands["mcp-server"], "put"),
        (cli.commands["env"].commands["website"], "put"),
    ):
        flags = {o for p in group.commands[cmd].params for o in p.opts}
        assert "--service-version" in flags
        assert "--environment-version" not in flags

    artifact_put = {o for p in cli.commands["artifact"].commands["environment"].commands["put"].params for o in p.opts}
    assert "--service-version" not in artifact_put
    assert "--environment-version" not in artifact_put
