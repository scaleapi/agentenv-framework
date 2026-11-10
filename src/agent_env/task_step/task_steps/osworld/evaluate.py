"""``osworld_evaluate``: an OSWorld task graded on a computer by the controller ``osworld_deploy_controller`` deployed
beside it, so no grading runs where the agent can reach it. An OSWorld-V2 task is graded by its ``TASK_CLASS.evaluate()``
on the controller; a classic one by each of its ``evaluator``'s funcs, sent to the controller's ``/evaluate`` one at a
time with its ``result`` and ``expected`` getters and its ``options``, and combined by its ``conj``. The verdict goes
to ``context.metadata["verifications"][verifier_id]`` as ``{"results", "score"}``."""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, ClassVar, Optional

from agent_env.entity.refs import EntityRef
from agent_env.osworld import controller, guest
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.osworld.initialize import run_setup
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)

PASS_SCORE, FAIL_SCORE = 1.0, 0.0

#: What an agent answers a task it finds infeasible with.
INFEASIBLE_TAG = "[INFEASIBLE]"
INFEASIBLE = "infeasible"
#: The LLM-judge funcs, which take a ``model`` option, a task's ``rubric`` and ``parameters``. A deterministic func is
#: called as ``metric(result, expected, **options)``, which a ``model`` it doesn't take fails.
JUDGE_FUNCS = frozenset({"agent_judge", "agent_judge_multi", "agent_judge_docx", "agent_judge_pptx", "agent_judge_pdf"})
#: The deterministic funcs that also take a task's ``rubric`` and ``parameters``.
CONTEXT_FUNCS = JUDGE_FUNCS | {"compare_pdfs", "compare_csv", "compare_images", "compare_python_pure_text",
                               "diff_text_file", "website_verifier"}
# An agent that ran out of turns or time leaves a desktop worth grading; every other stop leaves none. Matched on the
# harness's own sentence, since its error types name both kinds.
_BUDGET_EXHAUSTED = re.compile(r"Max turns \(\d+\) reached|agent timed out after \d+", re.I)


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
        ``instruction``. ``judge_model`` is the ``model`` option the judge funcs get when theirs set none; without it the
        controller's own default applies."""
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

    async def _stage_result_files(self, controller_url: str, context: TaskStepContext) -> None:
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
        await run_setup(controller_url, [{"type": "download", "parameters": {"files": entries}}],
                        what=f"osworld_evaluate '{self.verifier_id}' staging result files")

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        url = guest.controller_url(deployed)
        await self._stage_result_files(url, context)
        prompt_response = context.prompt_responses[-1] if context.prompt_responses else None
        if self.osworld_v2_task_path:
            if prompt_response is not None and prompt_response.error_type:
                return self._skipped(context, prompt_response)
            return self._record(context, await self._evaluate_v2(url))

        if postconfig := self.evaluator.get("postconfig"):
            try:
                await run_setup(url, postconfig, what=f"osworld_evaluate '{self.verifier_id}' postconfig")
            except Exception as e:
                logger.warning(f"osworld_evaluate '{self.verifier_id}': postconfig failed, grading anyway: {e}")
        budget = budget_exhaustion(prompt_response)
        if prompt_response is not None and prompt_response.error_type and budget is None:
            return self._skipped(context, prompt_response)
        verification = await self._evaluate_funcs(url, prompt_response)
        if budget is not None:
            # The score is the end state's, earned after the agent ran out of room; the marker beside the verifications
            # keeps such runs separable, since score readers project only each verifier's score.
            marker = {"reason": "budget_exhausted", "error_type": str(prompt_response.error_type),
                      "detail": budget[:300]}
            verification["partial_credit"] = marker
            existing = context.metadata.get("partial_credit")
            context.metadata["partial_credit"] = {**(existing if isinstance(existing, dict) else {}),
                                                  self.verifier_id: marker}
        return self._record(context, verification)

    async def _evaluate_v2(self, controller_url: str) -> dict:
        logger.info(f"OSWorld-V2 evaluate '{self.verifier_id}': task_ref={self.osworld_v2_task_path}")
        try:
            answer = await controller.evaluate(
                controller_url, {"osworld_v2_task_url": controller.resolve_task_ref(self.osworld_v2_task_path)})
        except controller.ControllerCallError as e:
            logger.warning(f"OSWorld-V2 evaluate '{self.verifier_id}': controller error: {e}")
            return {"results": [{"id": "osworld_v2", "score": FAIL_SCORE, "result": False, "message": str(e)[:500]}],
                    "score": FAIL_SCORE}
        score = controller.normalize_score(answer.get("score", FAIL_SCORE))
        return {"results": [{"id": "osworld_v2", "score": score, "result": not answer.get("error", False),
                             "message": answer.get("message", "")}], "score": score}

    def _funcs(self) -> list[tuple[str, Any, Any, dict]]:
        if "funcs" not in self.evaluator:
            raise ValueError(f"osworld_evaluate '{self.verifier_id}': the evaluator names no funcs "
                             f"(its keys are {sorted(self.evaluator)})")
        funcs = self.evaluator["funcs"]
        if not isinstance(funcs, list):
            return [(funcs, self.evaluator.get("result"), self.evaluator.get("expected"),
                     self.evaluator.get("options") or {})]
        n = len(funcs)
        return list(zip(funcs, self.evaluator.get("result") or [None] * n, self.evaluator.get("expected") or [None] * n,
                        [o or {} for o in self.evaluator.get("options") or [{}] * n]))

    def _payload(self, func: str, result: Any, expected: Any, options: dict, instruction: str | None) -> dict:
        options = dict(options)
        if func in JUDGE_FUNCS:
            if self.judge_model:
                options.setdefault("model", self.judge_model)
        else:
            options.pop("model", None)
        payload: dict[str, Any] = {"func": func, "options": options}
        if result is not None:
            payload["result"] = result
        if expected is not None:
            payload["expected"] = expected
        if func in CONTEXT_FUNCS:
            payload.update({key: self.evaluator[key] for key in ("parameters", "rubric") if key in self.evaluator})
        if instruction:
            payload["instruction"] = instruction
        return payload

    async def _evaluate_funcs(self, controller_url: str, prompt_response: PromptResponse | None) -> dict:
        answer_text = prompt_response.response if prompt_response is not None else None
        instruction = self.evaluator.get("instruction") or (prompt_response.prompt_text if prompt_response else None)
        results, scores, metrics = [], [], []
        funcs = self._funcs()
        for func, result, expected, options in funcs:
            try:
                if func == INFEASIBLE:
                    found = bool(answer_text) and INFEASIBLE_TAG in answer_text
                    sub = {"id": INFEASIBLE, "score": PASS_SCORE if found else FAIL_SCORE, "result": found,
                           "message": "Correctly identified infeasible task" if found
                           else "Did not identify infeasible task"}
                elif answer_text and INFEASIBLE_TAG in answer_text:
                    sub = {"id": "infeasible_on_feasible", "score": FAIL_SCORE, "result": False,
                           "message": "Agent declared infeasible on a feasible task"}
                else:
                    answer = await controller.evaluate(controller_url,
                                                       self._payload(func, result, expected, options, instruction))
                    sub = {"id": func, "score": controller.normalize_score(answer.get("score", FAIL_SCORE)),
                           "result": not answer.get("error", False), "message": answer.get("message", "")}
                passed = True
            except Exception as e:
                logger.warning(f"osworld_evaluate '{self.verifier_id}': {func} failed: {e}")
                sub = {"id": func, "score": FAIL_SCORE, "result": False, "message": f"Exception: {str(e)[:300]}"}
                passed = False
            results.append({"func": func, "results": [sub]})
            scores.append(float(sub["score"]))
            metrics.append({"func": func, "passed": passed, "score": float(sub["score"]), "message": sub["message"]})
        conj = self.evaluator.get("conj", "and")
        if len(scores) == 1:
            score = scores[0]
        else:
            score = aggregate_score([{"score": s, "result": s == PASS_SCORE} for s in scores],
                                    ScoreAggregator.ANY_PASS if conj == "or" else ScoreAggregator.ALL_PASS)
        errored = any(not m["passed"] for m in metrics)
        return {"results": results, "score": score,
                "debug": {"overall_score": score, "conjunction": conj if len(funcs) > 1 else None, "metrics": metrics,
                          "error": errored, "message": "Evaluation completed with errors" if errored
                          else "Evaluation passed" if score == PASS_SCORE else "Evaluation failed"}}
