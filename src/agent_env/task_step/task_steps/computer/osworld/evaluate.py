"""``osworld_evaluate``: a classic OSWorld task graded in the worker, never where the agent can reach it: each of its
``evaluator``'s funcs is upstream OSWorld's metric over what its ``result`` and ``expected`` getters read off the
computer's guest server, with the env's control token (:mod:`.runtime`), and its ``options``; the scores are combined by
its ``conj`` as OSWorld's ``DesktopEnv.evaluate`` combines them: ``and`` is 0 when any func scores 0, else their mean;
``or`` is their max. The verdict goes to ``context.metadata["verifications"][verifier_id]`` as ``{"results", "score"}``.

agent-env ships no LLM judge. An LLM-judge func (``agent_judge`` and its variants) is graded by the judge the worker's
deployment registers for it (``register_judge``), in the worker, with the run's judge model; with none registered it
scores 0 saying so. An OSWorld-V2 task, whose evaluate is the task's own code, fails the step."""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Optional

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.computer.osworld import guest, runtime
from agent_env.task_step.task_steps.computer.osworld.initialize import SETUP_ERROR_CHARS, V2_UNSUPPORTED, run_setup

logger = logging.getLogger(__name__)

PASS_SCORE, FAIL_SCORE = 1.0, 0.0
#: How much of the agent's stop message a partial-credit marker keeps.
BUDGET_DETAIL_CHARS = 300

#: What an agent answers a task it finds infeasible with.
INFEASIBLE_TAG = "[INFEASIBLE]"
INFEASIBLE = "infeasible"
#: The LLM-judge funcs (``agent_judge`` and its per-format ``agent_judge_<format>`` variants) take a ``model`` option, a
#: task's ``rubric`` and ``parameters``. A deterministic func is called as ``metric(result, expected, **options)``, which
#: a ``model`` it doesn't take fails.
JUDGE_FUNC_PREFIX = "agent_judge"

#: A judge: given one func's payload, ``{"func", "options", "result"?, "expected"?, "rubric"?, "parameters"?,
#: "instruction"?}``, with the judge model in ``options["model"]`` when one is named, and the run's context, it answers
#: ``{"score", "error"?, "message"?}``. It runs in the worker, which holds the model key.
Judge = Callable[[dict, TaskStepContext], Awaitable[dict]]
_JUDGES: dict[str, Judge] = {}


def register_judge(func: str, judge: Judge) -> None:
    """Grade the LLM-judge func ``func`` with ``judge``; ``JUDGE_FUNC_PREFIX`` itself serves every variant without a
    judge of its own."""
    if not is_judge(func):
        raise ValueError(f"{func!r} isn't an LLM-judge func ({JUDGE_FUNC_PREFIX} or {JUDGE_FUNC_PREFIX}_<format>)")
    _JUDGES[func] = judge


def judge_for(func: str) -> Judge | None:
    return _JUDGES.get(func) or _JUDGES.get(JUDGE_FUNC_PREFIX)


def is_judge(func: str) -> bool:
    return func == JUDGE_FUNC_PREFIX or func.startswith(f"{JUDGE_FUNC_PREFIX}_")


def normalize_score(score: Any) -> float:
    """A score as a float: a bool as 1 or 0, a number as it is, a dict as the mean of its numbers, else 0."""
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict):
        numbers = [float(v) for v in score.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return sum(numbers) / len(numbers) if numbers else 0.0
    return 0.0


def combined(scores: list[float], conj: str) -> float:
    """``scores`` combined as OSWorld's ``DesktopEnv.evaluate`` does: ``or`` their max; ``and`` 0 when any is 0, else
    their mean."""
    if conj == "or":
        return max(scores)
    return FAIL_SCORE if any(s == FAIL_SCORE for s in scores) else sum(scores) / len(scores)
# An agent that ran out of turns or time leaves a desktop worth grading; every other stop leaves none. Matched on the
# sentences the harness and the shipped agents stop with, since their error types name both kinds: the harness's
# "Max turns (N) reached" and "agent timed out after N"; Claude Code's and Codex's "Reached maximum number of turns
# (N)"; each CLI's "<name> timed out after Ns" (claude CLI, codex CLI, gemini CLI, Letta Code, hermes, opencode).
_BUDGET_EXHAUSTED = re.compile(
    r"Max turns \(\d+\) reached|Reached maximum number of turns \(\d+\)"
    r"|\b(?:agent|CLI|Letta Code|hermes|opencode) timed out after \d+", re.I)


def budget_exhaustion(prompt_response: PromptResponse | None) -> str | None:
    """The stop reason when the agent stopped on its turn or time budget, else None."""
    if prompt_response is None or not prompt_response.error_type:
        return None
    for text in (prompt_response.error_message, prompt_response.response):
        if text and _BUDGET_EXHAUSTED.search(str(text)):
            return str(text)
    return None


class OSWorldEvaluateTaskStep(TaskStep):
    type: ClassVar[str] = "osworld_evaluate"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, evaluator: Optional[dict] = None,
                 verifier_id: Optional[str] = None, osworld_v2_task_path: Optional[str] = None,
                 osworld_v2_task_url: Optional[str] = None, judge_model: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        """``evaluator`` is OSWorld's: ``funcs``, a name or a list, with ``result``, ``expected`` and ``options`` of
        each, in parallel lists for several, ``conj`` (``and`` or ``or``), ``postconfig``, ``rubric`` and
        ``instruction``. A judge func's ``model`` is the run's ``user_overrides.judge_model``, else its own option's, else
        ``judge_model``; with none the judge's own default applies. ``osworld_v2_task_url`` is the field the hub's CUA
        importer writes an OSWorld-V2 task's URL to; either one fails the step (``V2_UNSUPPORTED``)."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.evaluator = evaluator or {}
        self.verifier_id = verifier_id or uuid.uuid4().hex
        self.osworld_v2_task_path = osworld_v2_task_path or osworld_v2_task_url
        self.judge_model = judge_model

    def to_dict(self) -> dict:
        base = {**super().to_dict(), "env_id": self.env_id, "verifier_id": self.verifier_id, "evaluator": self.evaluator}
        if self.osworld_v2_task_path is not None:
            base["osworld_v2_task_path"] = self.osworld_v2_task_path
        if self.judge_model is not None:
            base["judge_model"] = self.judge_model
        return base

    @classmethod
    def from_dict(cls, data: dict) -> OSWorldEvaluateTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], evaluator=data.get("evaluator"),
                   verifier_id=data.get("verifier_id"), osworld_v2_task_path=data.get("osworld_v2_task_path"),
                   osworld_v2_task_url=data.get("osworld_v2_task_url"), judge_model=data.get("judge_model"))

    def _record(self, context: TaskStepContext, verification: dict) -> TaskStepContext:
        context.metadata.setdefault("verifications", {})[self.verifier_id] = verification
        logger.info(f"osworld_evaluate '{self.verifier_id}': score={verification['score']}")
        return context

    def _skipped(self, context: TaskStepContext, prompt_response: PromptResponse) -> TaskStepContext:
        message = f"Skipped: prompt had error_type={prompt_response.error_type}"
        logger.warning(f"osworld_evaluate '{self.verifier_id}': {message}")
        return self._record(context, {"results": [{"id": "prompt_error", "score": FAIL_SCORE, "result": False,
                                                   "message": message}], "score": FAIL_SCORE})

    async def _stage_result_files(self, guest_url: str, headers: dict[str, str], context: TaskStepContext) -> None:
        """A run's earlier outputs, ``step_overrides`` ``stage_files`` ``[{url, path}]``, downloaded onto the computer
        before anything reads it, to grade a finished run again on a fresh one. Raises on any failure: an unstaged
        computer would be graded on the task's own inputs."""
        files = self.step_param_overrides(context).get("stage_files")
        if not files:
            return
        if not isinstance(files, list) or not all(
                isinstance(f, dict) and isinstance(f.get("url"), str) and f["url"].startswith(("http://", "https://"))
                and f.get("path") for f in files):
            raise ValueError(f"osworld_evaluate '{self.verifier_id}': stage_files must be a list of {{url, path}}, "
                             f"each url http(s) and fetchable from the computer; got {files!r}")
        entries = [{"url": f["url"], "path": f["path"]} for f in files]
        logger.info(f"osworld_evaluate '{self.verifier_id}': staging {[e['path'] for e in entries]} before grading")
        await run_setup(guest_url, headers, [{"type": "download", "parameters": {"files": entries}}],
                        what=f"osworld_evaluate '{self.verifier_id}' staging result files")

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise StepContextError(f"Env '{self.env_id}' not found in context.deployed_envs")
        if self.osworld_v2_task_path:
            raise ValueError(f"osworld_evaluate '{self.verifier_id}': {V2_UNSUPPORTED}")
        prompt_response = context.prompt_responses[-1] if context.prompt_responses else None
        budget = budget_exhaustion(prompt_response)
        # A turn that failed is recorded as skipped before the computer is touched, so a computer the failure left
        # unreachable still gets its zero-score verification.
        if prompt_response is not None and prompt_response.error_type and budget is None:
            return self._skipped(context, prompt_response)
        guest_url, headers = await guest.guest_access(deployed)
        await self._stage_result_files(guest_url, headers, context)
        if postconfig := self.evaluator.get("postconfig"):
            try:
                await run_setup(guest_url, headers, postconfig, what=f"osworld_evaluate '{self.verifier_id}' postconfig")
            except Exception as e:
                logger.warning(f"osworld_evaluate '{self.verifier_id}': postconfig failed, grading anyway: "
                               f"{type(e).__name__}: {str(e)[:SETUP_ERROR_CHARS]}")
        run_judge_model = (context.metadata.get("user_overrides") or {}).get("judge_model")
        judge_models = [self._judge_model(options, run_judge_model)
                        for func, _, _, options in self._funcs() if is_judge(func)]
        if judge_models:
            # The models the judge funcs are sent, each func's own option included; None leaves the judge's default.
            named = sorted({m for m in judge_models if m})
            context.record_model("judge", self.id, ", ".join(named) or None)
        verification = await self._evaluate_funcs(guest_url, headers, context, prompt_response, run_judge_model)
        if budget is not None and prompt_response is not None:
            # The score is the end state's, earned after the agent ran out of room; the marker beside the verifications
            # keeps such runs separable, since score readers project only each verifier's score.
            marker = {"reason": "budget_exhausted", "error_type": str(prompt_response.error_type),
                      "detail": budget[:BUDGET_DETAIL_CHARS]}
            verification["partial_credit"] = marker
            existing = context.metadata.get("partial_credit")
            context.metadata["partial_credit"] = {**(existing if isinstance(existing, dict) else {}),
                                                  self.verifier_id: marker}
        return self._record(context, verification)

    def _funcs(self) -> list[tuple[str, Any, Any, dict]]:
        if "funcs" not in self.evaluator:
            raise ValueError(f"osworld_evaluate '{self.verifier_id}': the evaluator names no funcs "
                             f"(its keys are {sorted(self.evaluator)})")
        funcs = self.evaluator["funcs"]
        if not isinstance(funcs, list):
            return [(funcs, self.evaluator.get("result"), self.evaluator.get("expected"),
                     self.evaluator.get("options") or {})]
        return list(zip(funcs, self._per_func("result", len(funcs), None), self._per_func("expected", len(funcs), None),
                        [o or {} for o in self._per_func("options", len(funcs), {})], strict=True))

    def _per_func(self, key: str, n: int, absent: Any) -> list:
        """The evaluator's ``key``, one entry per func, or ``absent`` for each when it has none; a list of another
        length is refused, since zipping it would grade on fewer checks than the task names."""
        value = self.evaluator.get(key)
        if not value:
            return [absent] * n
        if not isinstance(value, list) or len(value) != n:
            got = f"a list of {len(value)}" if isinstance(value, list) else type(value).__name__
            raise ValueError(f"osworld_evaluate '{self.verifier_id}': the evaluator's {key!r} must be a list of {n}, "
                             f"one per func, got {got}")
        return value

    def _judge_model(self, options: dict, run_judge_model: str | None) -> str | None:
        """A judge func's model: the run's (``user_overrides.judge_model``), else its own option's, else the step's
        ``judge_model``; None leaves the judge's default."""
        return run_judge_model or options.get("model") or self.judge_model

    def _judge_payload(self, func: str, result: Any, expected: Any, options: dict, instruction: str | None,
                       run_judge_model: str | None) -> dict:
        """What a judge func's judge is given (``Judge``), its ``model`` as ``_judge_model`` picks it."""
        model = self._judge_model(options, run_judge_model)
        options = {key: value for key, value in options.items() if key != "model"}
        if model:
            options["model"] = model
        payload: dict[str, Any] = {"func": func, "options": options}
        if result is not None:
            payload["result"] = result
        if expected is not None:
            payload["expected"] = expected
        payload.update({key: self.evaluator[key] for key in ("parameters", "rubric") if key in self.evaluator})
        if instruction:
            payload["instruction"] = instruction
        return payload

    async def _graded(self, func: str, result: Any, expected: Any, options: dict, guest_url: str,
                      headers: dict[str, str], context: TaskStepContext, instruction: str | None,
                      run_judge_model: str | None) -> dict:
        """One func's row: a judge func's judge's answer, else upstream's metric's score."""
        if not is_judge(func):
            # A ``model`` a task copied onto a deterministic func would fail its metric call.
            options = {key: value for key, value in options.items() if key != "model"}
            score = await runtime.run_evaluate(guest_url, headers, func, result, expected, options)
            return {"id": func, "score": float(score), "result": True, "message": ""}
        if (judge := judge_for(func)) is None:
            return {"id": func, "score": FAIL_SCORE, "result": False,
                    "message": f"no judge is registered for {func} in this worker (register_judge)"}
        answer = await judge(self._judge_payload(func, result, expected, options, instruction, run_judge_model),
                             context)
        return {"id": func, "score": normalize_score(answer.get("score", FAIL_SCORE)),
                "result": not answer.get("error", False), "message": answer.get("message", "")}

    async def _evaluate_funcs(self, guest_url: str, headers: dict[str, str], context: TaskStepContext,
                              prompt_response: PromptResponse | None, run_judge_model: str | None) -> dict:
        answer_text = prompt_response.response if prompt_response is not None else None
        instruction = self.evaluator.get("instruction") or (prompt_response.prompt_text if prompt_response else None)
        results, scores, metrics = [], [], []
        funcs = self._funcs()
        for func, result, expected, options in funcs:
            try:
                if func == INFEASIBLE:
                    found = answer_text is not None and INFEASIBLE_TAG in answer_text
                    sub = {"id": INFEASIBLE, "score": PASS_SCORE if found else FAIL_SCORE, "result": found,
                           "message": "Correctly identified infeasible task" if found
                           else "Did not identify infeasible task"}
                elif answer_text and INFEASIBLE_TAG in answer_text:
                    sub = {"id": "infeasible_on_feasible", "score": FAIL_SCORE, "result": False,
                           "message": "Agent declared infeasible on a feasible task"}
                else:
                    sub = await self._graded(func, result, expected, options, guest_url, headers, context, instruction,
                                             run_judge_model)
                passed = True
            except Exception as e:
                logger.warning(f"osworld_evaluate '{self.verifier_id}': {func} failed: {type(e).__name__}: "
                               f"{str(e)[:SETUP_ERROR_CHARS]}")
                sub = {"id": func, "score": FAIL_SCORE, "result": False,
                       "message": f"Exception: {str(e)[:SETUP_ERROR_CHARS]}"}
                passed = False
            # The row's own score and result are what aggregate_verifiers reads; ``results`` is cua_evaluate's shape.
            results.append({"func": func, "score": float(sub["score"]), "result": float(sub["score"]) >= PASS_SCORE,
                            "results": [sub]})
            scores.append(float(sub["score"]))
            metrics.append({"func": func, "passed": passed, "score": float(sub["score"]), "message": sub["message"]})
        conj = self.evaluator.get("conj", "and")
        score = combined(scores, conj)
        errored = any(not m["passed"] for m in metrics)
        return {"results": results, "score": score,
                "debug": {"overall_score": score, "conjunction": conj if len(funcs) > 1 else None, "metrics": metrics,
                          "error": errored, "message": "Evaluation completed with errors" if errored
                          else "Evaluation passed" if score == PASS_SCORE else "Evaluation failed"}}
