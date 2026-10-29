"""Offer to install what a bundle's bundle.toml requires when a command is refused for it in a terminal, then run the
command again."""

from __future__ import annotations

import os
import shlex
import sys

import click

from agent_env.bundle.requires import UnmetRequirements
from agent_env.cli import _plugin_changes
from agent_env.cli._installers import InstallerError

# Set for the command run again once the install worked, so it never asks twice.
_ASKED = "AGENT_ENV_INSTALLED_REQUIRES"


def offer_to_install(error: UnmetRequirements) -> None:
    """When ``error`` refused a command run in a terminal, print it and offer to install what it requires with
    ``agent-env plugin add``; once that works, run the same command again in a new process, which loads the plugins it
    added. Without a terminal, or when the command was already run again, it returns having printed nothing, for the
    caller to raise ``error`` as it would have."""
    if os.environ.get(_ASKED) or not _interactive():
        return
    click.echo("\n".join([f"Error: {error}", *getattr(error, "__notes__", ())]), err=True)
    if not click.confirm(f"Install {', '.join(error.requirements)} with agent-env plugin add?", default=False):
        raise SystemExit(1)
    try:
        code = _plugin_changes.add(list(error.requirements), installer=None, index_url=None, dry_run=False, yes=True,
                                   keep=False, tools=_plugin_changes.Tools())
    except InstallerError as exc:
        raise click.ClickException(str(exc)) from exc
    if code:
        raise SystemExit(code)
    argv = [sys.executable, *sys.orig_argv[1:]]
    click.echo(f"Running it again: {shlex.join(['agent-env', *sys.argv[1:]])}")
    os.environ[_ASKED] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, argv)


def _interactive() -> bool:
    """Whether a person at a terminal can answer: both stdin and stdout are one."""
    return sys.stdin.isatty() and sys.stdout.isatty()
