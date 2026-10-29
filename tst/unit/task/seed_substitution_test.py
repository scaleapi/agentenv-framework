"""A run's seed fills each step's ``<key>`` placeholders in one pass. A value bound for a field a step runs as shell
text must be one the shell reads literally, so a seed row can name an id there but can't add a command."""

from __future__ import annotations

import pytest

from agent_env.task.task import substitute_seed_in_steps
from agent_env.task_step.task_steps.run_docker_container import RunDockerContainerTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep


def _verifier(**fields):
    return RunContainerUnitTestsVerifierTaskStep(id="check", version=None, sandbox_name="h", container_name="c",
                                                 **{"command": "true", **fields})


_SHELL_TEXT = [
    (_verifier(command="scrape <url>"), "command"),
    (_verifier(setup_commands=["pip install <url>"]), "setup_commands"),
    (VerifySandboxTaskStep(id="check", version=None, base_dir="/app", verifier_id="v",
                           criteria=[{"type": "bash_cmd_succeeds", "bash_cmd": "test -e <url>"}]), "criteria.bash_cmd"),
    (RunDockerContainerTaskStep(id="box", version=None, sandbox_name="h", docker_context_artifact_id="u",
                                command_override="serve <url>"), "command_override"),
    (RunDockerContainerTaskStep(id="box", version=None, sandbox_name="h", docker_context_artifact_id="u",
                                ready_command="curl <url>"), "ready_command"),
]


@pytest.mark.parametrize("value", ["https://x/a; rm -rf /", "$(id)", "`id`", "a b", "x'y", 'x"y', "a&b", "*", ""])
@pytest.mark.parametrize(("step", "field"), _SHELL_TEXT, ids=[field for _, field in _SHELL_TEXT])
def test_a_seed_value_with_shell_syntax_is_refused_for_shell_text(step, field, value):
    with pytest.raises(ValueError, match=rf"Seed 'url' can't go into step '{step.id}' field '{field}'"):
        substitute_seed_in_steps([step], {"url": value})


def test_an_id_is_substituted_into_shell_text_as_it_is():
    step = _verifier(command='qc --task-id <task_id> --universe-id "<universe_id>" --turns <turns>')

    seeded, = substitute_seed_in_steps([step], {"task_id": "task-12", "universe_id": "hg4/v3.1", "turns": 8})

    assert seeded.command == 'qc --task-id task-12 --universe-id "hg4/v3.1" --turns 8'


def test_any_value_fills_a_field_that_isnt_shell_text():
    step = _verifier(env_vars={"URL": "<url>"})

    seeded, = substitute_seed_in_steps([step], {"url": "https://x/a?b=1&c=$(id)"})

    assert seeded.env_vars == {"URL": "https://x/a?b=1&c=$(id)"}


def test_a_criterion_that_isnt_a_shell_command_takes_any_value():
    step = VerifySandboxTaskStep(id="check", version=None, base_dir="/app", verifier_id="v", criteria=[
        {"type": "probe_file_contains", "paths": ["<report>"], "expected": "<pattern>"},
        {"type": "bash_cmd_succeeds", "bash_cmd": "test -s <name>"}])

    seeded, = substitute_seed_in_steps([step], {"report": "out dir/report.txt", "pattern": "[0-9]+ passed", "name": "ok"})

    assert seeded.criteria == [
        {"type": "probe_file_contains", "paths": ["out dir/report.txt"], "expected": "[0-9]+ passed"},
        {"type": "bash_cmd_succeeds", "bash_cmd": "test -s ok"}]


def test_substituted_text_is_never_rescanned():
    step = _verifier(env_vars={"A": "<a>", "B": "<b>"})

    seeded, = substitute_seed_in_steps([step], {"a": "<b>", "b": "$(id)"})

    assert seeded.env_vars == {"A": "<b>", "B": "$(id)"}
