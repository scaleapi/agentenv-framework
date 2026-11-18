"""rubrics_verifier leaves the legacy entries main left on every path that writes one, or none, with the judge faked
below the seams the verifier's own code owns (``_prompt_llm_judge``, ``_invoke_judge_a2a``)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    CompactionType,
    FrameStrategy,
    TrajectoryFilter,
    TrajectoryText,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import JudgeTimeoutError, RubricsVerifierTaskStep
from tst.util import verifier_golden

FILE = "rubrics_verifier.json"
_JPEG = "/9j/" + "A" * 300
TWO = [{"id": "table-reserved", "description": "A table is reserved"},
       {"id": "party-size", "description": "Party size is four"}]
EVIDENCE = [*TWO, {"id": "no-autocomplete", "description": "No suggestion was accepted", "weight": -1}]


def _span(tool: str, args: dict, screenshot: bool = True) -> dict:
    completion = {"result": "ok", **({"screenshot": _JPEG} if screenshot else {})}
    return {"name": tool, "attributes": {"gen_ai.operation.name": "execute_tool",
                                         "gen_ai.prompt": json.dumps({"tool": tool, "input": args}),
                                         "gen_ai.completion": json.dumps(completion)}}


SHOTS = json.dumps([_span("gui_tap", {"x": i}) for i in range(1, 12)])
NO_SHOTS = json.dumps([_span("gui_tap", {"x": 1}, screenshot=False)])


def _criteria(n: int) -> list[dict]:
    return [{"id": f"crit-{i:02d}", "title": f"Criterion {i}", "description": "d" * 40} for i in range(1, n + 1)]


def _ids(prompt: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'"id": "(c\d+)"', prompt)))


def _context(overrides: dict | None = None, **response) -> TaskStepContext:
    fields = dict(prompt_id="p1", response="done", prompt_text="do it", agent_trajectory_object_url="mem://traj/raw.json")
    context = TaskStepContext(prompt_responses=[PromptResponse(**{**fields, **response})], instance_id="inst-1")
    if overrides is not None:
        context.metadata["user_overrides"] = overrides
    return context


def _step(cls: type[RubricsVerifierTaskStep] = RubricsVerifierTaskStep, **kw) -> RubricsVerifierTaskStep:
    return cls(**{"id": "verify", "version": 1, "criteria": TWO, "prompt_id": "p1", "use_agent_judge": False,
                  "use_trajectory": False, "verifier_id": "vid", "default_model": "judge-model", **kw})


def _evidence_step(output_format: str, **kw) -> RubricsVerifierTaskStep:
    frames = TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT, frame_strategy=FrameStrategy.PER_CRITERION,
                              frame_budget=6, trajectory_text=TrajectoryText.ACTION_LOG)
    return _step(output_format=output_format, criteria=EVIDENCE, use_trajectory=True, trajectory_filter=frames, **kw)


def _direct_judge(step: RubricsVerifierTaskStep, respond: Callable[[list[str], int], str],
                  trajectory: str = SHOTS) -> RubricsVerifierTaskStep:
    calls: list[str] = []

    async def judge(prompt, *, model, context, image_blocks=None, **_):
        calls.append(prompt)
        return {"response": respond(_ids(prompt), len(calls)), "trajectory_object_url": None}

    step._prompt_llm_judge = judge
    step._read_trajectory_text = lambda url: trajectory
    step._filter_trajectory = lambda url, trajectory_filter: ("mem://compact/1.json", [])
    return step


def _binary(ids: list[str], failing=(), why: Callable[[str], str] = lambda i: f"Checked record {i} with the sheet tool"
            ) -> str:
    return json.dumps({"results": [{"id": i, "score": 0.0 if i in failing else 1.0, "justification": why(i)}
                                   for i in ids]})


def _evidence(ids: list[str], mistakes: bool = False) -> str:
    status = ["visible", "missing", "not_applicable"]
    body = {"reasoning": "the frames show the booking", "checks": [{"id": "typed", "passed": True, "evidence": "x"}],
            "results": [{"id": i, "frame": f"[{i}] action 3", "evidence_status": status[k % 3],
                         "justification": f"about {i}", "score": 0.0 if k % 3 == 1 else 1.0}
                        for k, i in enumerate(ids)]}
    if mistakes:
        body.update(findings=[{"severity": "high", "step": 4, "should": "type it", "description": "tapped a suggestion"}],
                    botched=False, botched_reason="", summary="one shortcut")
    return json.dumps(body)


class _WithRows(RubricsVerifierTaskStep):
    async def unjudged_rows(self, context, prompt_response):
        return [{"id": "auto", "result": True, "score": 1.0}, {"id": "na", "result": False, "score": 0.0, "weight": 0}]


def _judge_unreachable(step: RubricsVerifierTaskStep) -> RubricsVerifierTaskStep:
    async def judge(prompt, **_):
        raise RuntimeError("LLM judge failed after 3 attempts")

    step._prompt_llm_judge = judge
    return step


def _agent_judge_timing_out_once() -> tuple[RubricsVerifierTaskStep, TaskStepContext]:
    step = _step(criteria=_criteria(2), use_agent_judge=True, agent_name="judge", judge_timeout_retries=1)
    attempts: list[int] = []

    async def judge(*, eval_prompt, judge_a2a_url, judge_agent_card, judge_agent):
        attempts.append(1)
        if len(attempts) == 1:
            raise JudgeTimeoutError("ran out", "mem://judge/timed-out.json")
        return {"response": _binary(_ids(eval_prompt)), "trajectory_object_url": "mem://judge/a2a.json"}

    step._invoke_judge_a2a = judge
    context = _context()
    context.deployed_agents = [SimpleNamespace(
        agent_name="judge", sandbox_id="sb-1", a2a_url="http://judge.test", api_url=None, a2a_card={},
        installed_container=None, sandbox_type="local", control_headers={})]
    return step, context


TIMEOUT = {"error_type": "timeout", "response": ""}
CASES: dict[str, Callable[[], tuple[RubricsVerifierTaskStep, TaskStepContext]]] = {
    "binary": lambda: (_direct_judge(_step(), lambda ids, n: _binary(ids, failing={"c2"})), _context()),
    "partial_weighted": lambda: (_direct_judge(_step(
        output_format="rubric_partial", score_aggregator="weighted_average",
        criteria=[{"id": "q", "description": "quality", "outcomes": {"min": {"value": -1}, "max": {"value": 1}}},
                  {"id": "r", "description": "recall"}]),
        lambda ids, n: json.dumps({"results": [{"id": i, "score": 0.5, "justification": f"half {i}"} for i in ids]})),
        _context()),
    "batches_with_a_bulk_verdict_retry": lambda: (_direct_judge(
        _step(criteria=_criteria(5), judge_batch_size=2),
        lambda ids, n: _binary(ids, why=(lambda i: "Pass.") if n == 1 else (lambda i: f"Checked row {i}"))), _context()),
    "an_answer_retry_for_a_missing_id": lambda: (_direct_judge(
        _step(criteria=_criteria(3)), lambda ids, n: _binary(ids[:-1] if n == 1 else ids)), _context()),
    "a_compacted_trajectory": lambda: (_direct_judge(
        _step(criteria=_criteria(3), use_trajectory=True, trajectory_filter=TrajectoryFilter()),
        lambda ids, n: _binary(ids), trajectory="[]"), _context()),
    "grade_up_to_turn": lambda: (_direct_judge(
        _step(criteria=_criteria(2), use_trajectory=True), lambda ids, n: _binary(ids), trajectory="turn text"),
        _context({"step_params": {"verify": {"grade_up_to_turn": 1}}},
                 target_agent_per_turn_trajectory_object_urls=["mem://t/1.json", "mem://t/2.json"])),
    "prompt_error": lambda: (_direct_judge(_step(), lambda ids, n: _binary(ids)), _context(**TIMEOUT)),
    "the_gate_on_an_unmet_row": lambda: (_direct_judge(
        _step(fail_task_when_not_passing=True), lambda ids, n: _binary(ids, failing={"c1"})), _context()),
    "the_gate_on_a_prompt_error": lambda: (_direct_judge(
        _step(fail_task_when_not_passing=True), lambda ids, n: _binary(ids)), _context(**TIMEOUT)),
    "an_answer_the_judge_never_fixes": lambda: (_direct_judge(_step(), lambda ids, n: "not json"), _context()),
    "a_judge_that_cannot_be_reached": lambda: (_judge_unreachable(_step()), _context()),
    "rubric_evidence": lambda: (_direct_judge(
        _evidence_step("rubric_evidence", grading_policy_prompt="Type every value in full."),
        lambda ids, n: _evidence(ids)), _context()),
    "rubric_evidence_without_screenshots": lambda: (_direct_judge(
        _evidence_step("rubric_evidence"), lambda ids, n: _evidence(ids), trajectory=NO_SHOTS), _context()),
    "with_mistakes": lambda: (_direct_judge(
        _evidence_step("rubric_evidence_with_mistakes"), lambda ids, n: _evidence(ids, mistakes=True)), _context()),
    "with_mistakes_without_screenshots": lambda: (_direct_judge(
        _evidence_step("rubric_evidence_with_mistakes"), lambda ids, n: _evidence(ids, mistakes=True),
        trajectory=NO_SHOTS), _context()),
    "with_mistakes_on_a_prompt_error": lambda: (_direct_judge(
        _evidence_step("rubric_evidence_with_mistakes"), lambda ids, n: _evidence(ids, mistakes=True)),
        _context(**TIMEOUT)),
    "with_mistakes_and_no_findings": lambda: (_direct_judge(
        _evidence_step("rubric_evidence_with_mistakes"), lambda ids, n: _evidence(ids)), _context()),
    "unjudged_rows": lambda: (_direct_judge(
        _step(_WithRows, criteria=_criteria(2), score_aggregator="weighted_average"), lambda ids, n: _binary(ids)),
        _context()),
    "an_agent_judge_timing_out_once": _agent_judge_timing_out_once,
    "a_runs_criteria_and_judge_overrides": lambda: (_direct_judge(_step(criteria=_criteria(2)), lambda ids, n: _binary(ids)),
                                                    _context({"judge_model": "judge-x", "judge_effort": "high",
                                                              "judge_batch_size": 1, "step_params": {"verify": {
                                                                  "criteria": [{"id": "only", "description": "it"}]}}})),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(CASES))
async def test_a_run_leaves_the_legacy_entries_main_left(case):
    step, context = CASES[case]()
    raised = None
    try:
        await step.execute(context)
    except Exception as exc:
        raised = exc

    verifier_golden.check(FILE, case, verifier_golden.observed(context, raised))
