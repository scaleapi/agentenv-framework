"""A run's seed fills each step's ``<key>`` placeholders in one left-to-right pass, so text a value brings in is
never read again as another placeholder."""

from __future__ import annotations

from agent_env.task.task import substitute_seed_in_steps
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)


def _verifier(**fields):
    return RunContainerUnitTestsVerifierTaskStep(id="check", version=None, sandbox_name="h", container_name="c",
                                                 **{"command": "true", **fields})


def test_each_placeholder_takes_its_value():
    step = _verifier(command='qc --task-id <task_id> --universe-id "<universe_id>" --turns <turns>')

    seeded, = substitute_seed_in_steps([step], {"task_id": "task-12", "universe_id": "hg4/v3.1", "turns": 8})

    assert seeded.command == 'qc --task-id task-12 --universe-id "hg4/v3.1" --turns 8'


def test_substituted_text_is_never_rescanned():
    step = _verifier(env_vars={"A": "<a>", "B": "<b>"})

    seeded, = substitute_seed_in_steps([step], {"a": "<b>", "b": "$(id)"})

    assert seeded.env_vars == {"A": "<b>", "B": "$(id)"}
