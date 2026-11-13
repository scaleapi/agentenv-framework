"""`task run` shows an always_on step its response as it shows a prompt_agent step's."""

from __future__ import annotations

from types import SimpleNamespace

from agent_env.cli.task.run import _step_output_lines
from agent_env.task_step.context import PromptResponse, TaskStepContext


def test_an_always_on_steps_response_is_shown():
    context = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="loop", response="Recap sent.", step_id="loop")])

    [lines] = _step_output_lines(SimpleNamespace(type="always_on", id="loop"), context)

    assert "prompt_id: loop" in lines and "Recap sent." in lines


def test_an_always_on_step_without_a_response_shows_nothing():
    assert _step_output_lines(SimpleNamespace(type="always_on", id="loop"), TaskStepContext()) == []


def test_an_always_on_step_shows_its_own_response_when_a_parallel_step_added_a_later_one():
    context = TaskStepContext(prompt_responses=[
        PromptResponse(prompt_id="loop", response="Recap sent.", step_id="loop"),
        PromptResponse(prompt_id="side", response="Side answer.", step_id="side")])

    [lines] = _step_output_lines(SimpleNamespace(type="always_on", id="loop"), context)

    assert "Recap sent." in lines and "Side answer." not in lines
