"""Stage is a platform concern: no built-in command may declare ``--stage``. Plugin groups are
outside this walk: a plugin may give its own commands whatever options it likes."""

import click

from agent_env.cli import cli

BUILT_IN_GROUPS = ("a2a-agent", "artifact", "env", "eval", "plugin", "task")


def _declaring(group, flag, path=()):
    for name, command in group.commands.items():
        if any(flag in param.opts for param in command.params):
            yield " ".join((*path, name))
        if isinstance(command, click.Group):
            yield from _declaring(command, flag, (*path, name))


def test_no_built_in_command_declares_stage():
    found = {site for name in BUILT_IN_GROUPS for site in _declaring(cli.commands[name], "--stage", (name,))}
    assert found == set()
