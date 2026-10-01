"""A rule the caller broke prints as one line; a defect keeps its traceback.

Click formats only `ClickException` specially, so every other error reached the user as a
stack trace through click's own frames — 56 lines for an unknown `AGENT_ENV_DOCUMENT_STORE`,
35 for a missing artifact. Both read as agent-env crashing rather than as the answer.
"""

import click
import pytest
from click.testing import CliRunner

from agent_env.cli import _UserErrorsAreNotCrashes, cli
from agent_env.config.errors import ConfigError
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import DuplicateKeyError
from agent_env.store.document_store.sqlite_document_store import DatabaseLockedError


def _group_raising(exc):
    @click.group(cls=_UserErrorsAreNotCrashes)
    @click.option("--verbose", "-v", is_flag=True)
    def root(verbose):
        pass

    @root.command()
    def boom():
        raise exc

    return root


@pytest.mark.parametrize("exc, message", [
    (ValueError("id '@ns/x' starts with the reserved '@' prefix"), "id '@ns/x' starts with the reserved '@' prefix"),
    (ConfigError("Unknown AGENT_ENV_DOCUMENT_STORE='bogus'"), "Unknown AGENT_ENV_DOCUMENT_STORE='bogus'"),
    (NotFoundError("Artifact definitely-missing not found"), "Artifact definitely-missing not found"),
    (DatabaseLockedError("/s/documents.db is locked: another process"), "/s/documents.db is locked: another process"),
], ids=["reserved-id", "config", "not-found", "locked"])
def test_a_rule_the_caller_broke_prints_one_line(exc, message):
    result = CliRunner().invoke(_group_raising(exc), ["boom"])

    assert result.exit_code == 1
    assert result.output.strip() == f"Error: {message}"
    assert "Traceback" not in result.output


def test_the_errors_notes_follow_it():
    exc = ValueError("tasks/t.json: step 'box': not a known step type")
    exc.add_note("while preflighting tasks/t.json (@local/~/triage/t)")

    result = CliRunner().invoke(_group_raising(exc), ["boom"])

    assert result.output == (
        "Error: tasks/t.json: step 'box': not a known step type\nwhile preflighting tasks/t.json (@local/~/triage/t)\n")


@pytest.mark.parametrize("exc", [
    TypeError("got an unexpected keyword argument 'agent_id'"),
    DuplicateKeyError("could not assign a version after 5 retries"),
], ids=["type-error", "duplicate-key"])
def test_a_defect_keeps_its_traceback(exc):
    """Neither is an answer to the caller: one is a bug, the other a write that lost a race
    every time it retried. Collapsing those to one line would hide where they came from."""
    result = CliRunner().invoke(_group_raising(exc), ["boom"])

    assert type(result.exception) is type(exc)


def test_verbose_defers_the_traceback_rather_than_losing_it():
    result = CliRunner().invoke(_group_raising(ValueError("x")), ["--verbose", "boom"])

    assert isinstance(result.exception, ValueError)


def test_clicks_own_errors_are_left_alone():
    result = CliRunner().invoke(_group_raising(ValueError("x")), ["no-such-command"])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_the_shipped_root_group_is_wired_to_it():
    """Every assertion above is about a group built in this file; without this one they could
    all pass while the real CLI kept printing tracebacks."""
    assert isinstance(cli, _UserErrorsAreNotCrashes)
