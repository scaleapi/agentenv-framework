"""always_on_agent_verifier: the rubrics_verifier for an always_on step's prompt.

An always_on run may take 40–60 wakes, each with its own trajectory: together they may not fit a judge, and cost a lot.
This verifier grades the run from what the user saw and the run's context instead: the user's chat (the prompt's
response and conversation), the trigger log, and the run wake by wake with the app changes each made
(``render_for_judge``), never a wake's trajectory. So when the task loads it refuses what needs the trajectories: an
output format that cites per-criterion screenshots, use_trajectory, and grade_up_to_turn (the conversation's turns are
not the run's wakes); a per-run grade_up_to_turn is refused when the verifier runs. Every other rubrics_verifier
setting works as it does there. An always_on step's prompt takes this verifier, never a plain rubrics_verifier, and
this verifier only an always_on step's prompt: a task that pairs them otherwise is refused when it loads.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Optional

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.always_on import ALWAYS_ON_KEY, AlwaysOnTaskStep, render_for_judge
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JudgeOutputFormat,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RUN_CONTEXT_PARTS, RubricsVerifierTaskStep


class AlwaysOnAgentVerifier(RubricsVerifierTaskStep):
    type: ClassVar[str] = "always_on_agent_verifier"
    entity_refs = RubricsVerifierTaskStep.entity_refs  # the judge, as rubrics_verifier's
    context_paths = RubricsVerifierTaskStep.context_paths

    def __init__(
        self,
        id: str,
        version: Optional[int],
        *,
        use_trajectory: bool = False,
        grade_up_to_turn: Optional[int] = None,
        run_context_for_judge: Optional[list[str]] = None,
        output_format: JudgeOutputFormat | str = JudgeOutputFormat.RUBRIC_BINARY,
        default_model: Optional[str] = None,
        default_model_api_base: Optional[str] = None,
        effort: Optional[str] = None,
        max_thinking_tokens: Optional[int] = None,
        judge_a2a_agent_id: Optional[str] = None,
        judge_sandbox_type: Optional[str] = None,
        judge_batch_size: Optional[int] = None,
        judge_batch_concurrency: Optional[int] = None,
        **fields: Any,
    ):
        """``run_context_for_judge`` defaults to every part of the run's context; the rest is rubrics_verifier's."""
        output_format = JudgeOutputFormat(output_format)
        where = f"{self.type} '{id}'"
        if use_trajectory:
            raise ValueError(f"{where}: use_trajectory must be off; an always_on step's prompt is graded from the "
                             "user's chat and the run's context, never its wakes' trajectories")
        if get_judge_output_format_spec(output_format).cites_evidence:
            raise ValueError(f"{where}: output_format={output_format.value} cites per-criterion screenshots, and an "
                             "always_on step's prompt is graded without its wakes' trajectories")
        if grade_up_to_turn is not None:
            raise ValueError(f"{where}: grade_up_to_turn can't cut an always_on step's prompt, whose conversation "
                             "turns are not its wakes")
        super().__init__(
            id, version, use_trajectory=False, run_context_for_judge=run_context_for_judge or list(RUN_CONTEXT_PARTS),
            output_format=output_format, default_model=default_model, default_model_api_base=default_model_api_base,
            effort=effort, max_thinking_tokens=max_thinking_tokens, judge_a2a_agent_id=judge_a2a_agent_id,
            judge_sandbox_type=judge_sandbox_type, judge_batch_size=judge_batch_size,
            judge_batch_concurrency=judge_batch_concurrency, **fields,
        )

    @classmethod
    def from_dict(cls, data: dict) -> AlwaysOnAgentVerifier:
        return super().from_dict({**data, "use_trajectory": data.get("use_trajectory", False)})

    def task_problems(self, steps: list[TaskStep]) -> list[str]:
        """Its prompt must be an always_on step's: one only another prompt step makes is refused. A prompt no step of
        the task makes is checked when the verifier runs."""
        prompts = [s for s in steps if isinstance(s, PromptAgentTaskStep) and s.prompt_id == self.prompt_id]
        if prompts and not any(isinstance(s, AlwaysOnTaskStep) for s in prompts):
            return [f"step '{self.id}' ({self.type}) grades prompt '{self.prompt_id}' of {prompts[0].type} step "
                    f"'{prompts[0].id}', which isn't an always_on step; grade it with rubrics_verifier"]
        return []

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        if "grade_up_to_turn" in self.step_param_overrides(context):
            raise RuntimeError(f"Verifier '{self.verifier_id}': grade_up_to_turn can't cut an always_on step's "
                               "prompt, whose conversation turns are not its wakes.")
        return await super().execute(context)

    async def judge_prompt_sections(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[str]:
        """The run wake by wake, from the always_on step's record of it."""
        record = (context.metadata.get(ALWAYS_ON_KEY) or {}).get(prompt_response.step_id)
        if record is None:
            raise RuntimeError(f"Verifier '{self.verifier_id}': prompt_id='{self.prompt_id}' is not an always_on "
                               f"step's prompt (step '{prompt_response.step_id}' left no always_on record); grade it "
                               "with rubrics_verifier.")
        return [await asyncio.to_thread(render_for_judge, record)]
