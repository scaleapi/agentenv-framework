"""The user_satisfaction_grading step: after a run with the built-in user sim, a judge reads the run from the user's
side and logs what went wrong for them and when; the step's score is computed from that log. It grades the assistant, as
its user would, and is recorded only: it never fails the task and decides no verdict. A task writer adds it after the
run's user_sim_judge (so a run the user sim spoiled is never graded) and rate_user_satisfaction (whose lines say what a
5, a 3 and a 1 mean to this user).

What a task can set on it, beyond every step's ``id``, ``type``, ``version``, ``depends_on``, ``fail_task_on_error``
(false unless set: a judge that can't answer is a failed grade, never a failed task) and ``retry_config``:

- ``run_step_id``, required: the always_on or prompt_agent step whose run it grades;
- ``judge_model``: unset, the run's ``judge_model`` override, then the config's judge model (``ModelRole.JUDGE``);
- ``judge_timeout_seconds``: how long the judge's call may take;
- ``verifier_id`` and ``min_passing_score`` (0.5, a 3 out of 5, unless set): the grade's, which nothing gates on.

What the judge reads is what the user saw, read as user_sim_judge reads it (``user_sim_judge._read_run``): the user
sim's instructions rebuilt from its record and checked against its ``prompt_sha256``, and every check-in, what the user
sim was sent (the assistant's messages and the user's notifications) and how it answered. From the run's
rate_user_satisfaction entry it takes the rating's lines and what the user read only after the last check-in. It never
sees the user's accounts, so it never checks whether what the assistant said happened: the task's end-state checks do.

What the assistant could know is judged from what it was told: the run step's ``system_prompt``, read when the task
loads (as user_sim_judge reads the task's graders; ``seeded_from`` keeps it) with the run's seed filled in. A person's
rule its instructions don't give, and the user hasn't said, binds it only when they lead to it. A step built outside a
task, or a run step with no system prompt, judges as if the user sim's rules and should-already-know lines were in its
notes, and the entry's ``assistant_instructions`` says which.

One call answers for the whole run (``user_satisfaction_grading_core``): every action the assistant reported and every
question or request it put to the user, then every incident at the check-in where the user could first see it, and the
end outcome. The step's score is the log's 1-5 score as 0-1 (``user_satisfaction_grading_core.normalized``); its rows,
one per dimension and one for the outcome, each weigh 0. Its entry keeps the score's parts and the run's details, and
the whole log is stored at ``log_object_url`` (inline as ``log`` when the store refuses it). A run that ended on an
error, or whose user sim never read, is skipped; a judge that gives no log after ``MAX_ATTEMPTS`` calls fails the grade,
with no score.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import Any, ClassVar, Optional

from agentenv_protocol.a2a_agent import Usage

from agent_env.config import get_config, model_call
from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy
from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.agent.user_sim_common import SATISFACTION_KEY
from agent_env.task_step.task_steps.verifiers import user_satisfaction_grading_core as core
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_usage
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_usage import JudgeUsage
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import _JUDGE_KEY_HOW, _record_failed_call
from agent_env.task_step.task_steps.verifiers.user_sim_judge import (
    ALWAYS_ON,
    ALWAYS_ON_KEY,
    CHAT,
    READS_KEY,
    _cost,
    _load,
    _prompt_response,
    _read_run,
    _usage,
)
from agent_env.task_step.task_steps.verifiers.verifier import Verification, VerifierTaskStep, _store_output
from agent_env.task_step.thread_work import finish_on_thread

logger = logging.getLogger(__name__)

SETTINGS = ("run_step_id", "judge_model", "judge_timeout_seconds")
DEFAULT_MIN_PASSING_SCORE = 0.5  # a 3 out of 5: "fine"
MAX_ATTEMPTS = 3  # the judge's call, again when it fails or its answer isn't JSON
MAX_TOKENS = 16000  # the answer: two checklists, the incidents and a line per check-in
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "user_satisfaction_grading", "schema": core.RUN_SCHEMA}}


class UserSatisfactionGrading(VerifierTaskStep):
    type: ClassVar[str] = "user_satisfaction_grading"
    entity_refs = ()
    DEFAULT_JUDGE_TIMEOUT_SECONDS: ClassVar[int] = 600

    def __init__(self, id: str, version: Optional[int], *, run_step_id: str,
                 judge_model: Optional[str] = None, judge_timeout_seconds: Optional[float] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = False,
                 retry_config: Optional[RetryConfig | dict] = None, verifier_id: Optional[str] = None,
                 min_passing_score: Optional[float] = None):
        passing = DEFAULT_MIN_PASSING_SCORE if min_passing_score is None else min_passing_score
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         retry_config=retry_config, verifier_id=verifier_id, min_passing_score=passing)
        self.run_step_id = run_step_id
        self.judge_model = judge_model
        self.judge_timeout_seconds = judge_timeout_seconds
        self._assistant_prompt: Optional[str] = None  # the run step's system_prompt, read when the task loads

    # ---- settings ----------------------------------------------------------------------------------------------
    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        where = f"{self.type} '{self.id}'"
        if not isinstance(self.run_step_id, str) or not self.run_step_id.strip():
            problems.append(f"{where}: run_step_id must name the always_on or prompt_agent step to grade")
        if self.judge_model is not None and (not isinstance(self.judge_model, str) or not self.judge_model.strip()):
            problems.append(f"{where}: judge_model must be a model name")
        t = self.judge_timeout_seconds
        if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0):
            problems.append(f"{where}: judge_timeout_seconds must be a number of seconds, more than 0")
        return problems

    def task_problems(self, steps: list[TaskStep]) -> list[str]:
        """The run it grades must be an always_on or prompt_agent step's, played by a built-in user sim, and come before
        it. It must run after that run and after the task's user_sim_judge and rate_user_satisfaction steps on the same
        run, when there are some: the judge so a run the user sim spoiled is never graded, the rating for its lines."""
        from agent_env.task.task import _ancestor_ids, _dependency_ids  # here: the task module imports the steps
        from agent_env.task_step.task_steps.agent.rate_user_satisfaction import (
            RateUserSatisfactionTaskStep,
            plays_built_in_user_sim,
            run_step_of,
        )
        from agent_env.task_step.task_steps.verifiers.user_sim_judge import UserSimJudge

        run, found = run_step_of(self, steps, does="grade")
        if run is None:
            return found
        self._assistant_prompt = getattr(run, "system_prompt", None) or None
        where = f"{self.type} '{self.id}'"
        if steps.index(run) > steps.index(self):
            found.append(f"{where} comes before step '{run.id}', the run it grades; put it after")
        if not plays_built_in_user_sim(run, steps):
            found.append(f"{where}: the user of {run.type} step '{run.id}' isn't a user sim a deploy_user_sim_agent "
                         "step deploys; it grades only the built-in user sim's run")
        ancestors = _ancestor_ids(self.id, _dependency_ids(steps))
        needs = [(run, "the run it grades")]
        for step in steps:
            if isinstance(step, (UserSimJudge, RateUserSatisfactionTaskStep)) and run_step_of(step, steps)[0] is run:
                needs.append((step, "so a run the user sim spoiled is never graded" if isinstance(step, UserSimJudge)
                              else "for the rating's lines and what the user read last"))
        for step, why in needs:
            if step.id not in ancestors:
                found.append(f"{where}: it must depend on '{step.id}' ({step.type}), {why}")
        return found

    def seeded_from(self, written: UserSatisfactionGrading) -> None:
        """This step is ``written`` with the run's seed filled in (``substitute_seed_in_steps``): keep what the
        assistant was told, which ``written`` read from the task's run step; the seed is filled in when it's read."""
        self._assistant_prompt = written._assistant_prompt

    def criterion_rows(self) -> Optional[int]:
        return len(core.DIMENSIONS) + 1

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({k: getattr(self, k) for k in SETTINGS if getattr(self, k) is not None})
        if self.min_passing_score == DEFAULT_MIN_PASSING_SCORE:
            base.pop("min_passing_score", None)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> UserSatisfactionGrading:
        base = {**cls._base_from_dict(data), "fail_task_on_error": data.get("fail_task_on_error", False)}
        base.pop("score_aggregator", None)
        return cls(**base, **{k: data[k] for k in SETTINGS if k in data})

    def verifier_settings(self, context: TaskStepContext) -> dict[str, Any]:
        settings = super().verifier_settings(context)
        settings.pop("score_aggregator", None)  # the score is the log's, not an aggregate of its rows
        settings["judge_model"] = self._judge_model(context)
        return settings

    def _judge_model(self, context: TaskStepContext) -> str:
        """The run's ``judge_model``, the step's own, then the config's judge model, as rubrics_verifier picks it."""
        overrides = context.metadata.get("user_overrides", {})
        return get_config().resolve_model(ModelRole.JUDGE, run=overrides.get("judge_model"),
                                          step=self.judge_model).model

    # ---- the judge's record, as a judge step keeps it (rubrics_verifier's) ---------------------------------------
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        with judge_usage.scope():
            return await super().execute(context)

    async def settle(self, context: TaskStepContext) -> None:
        if (judge := judge_usage.current()) is not None:
            await judge.settle()

    def grade_fields(self, context: TaskStepContext) -> dict[str, Any]:
        calls = judge_usage.calls()
        return {"judge_calls": calls} if calls else {}

    # ---- the log -----------------------------------------------------------------------------------------------
    async def verify(self, context: TaskStepContext) -> Verification:
        problems = self.config_problems()
        if problems:
            raise ValueError("; ".join(problems))
        kind, step_id = self._find_run(context)
        response = _prompt_response(context, step_id)
        record = (context.metadata.get(ALWAYS_ON_KEY) or {}).get(step_id) if kind == ALWAYS_ON else None
        model = self._judge_model(context)
        entry: dict[str, Any] = {"run_step_id": step_id, "run_kind": kind, "judge_model": model}
        error_type = (record or {}).get("error_type") or (response.error_type if response else None)
        if error_type:
            return self._skip(f"the run ended with error_type={error_type}", entry)
        if kind == CHAT and not (context.metadata.get(READS_KEY) or {}).get(step_id):
            return self._skip("the user sim never read: the run has no user_sim_reads", entry)
        run, about = await asyncio.to_thread(_read_run, kind, step_id, record, context)
        entry.update(user_sim=about["user_sim"], instructions=about["instructions"], turns=len(run.turns),
                     notes=list(run.notes))
        if not run.turns:
            return self._skip("the user sim never read: the run has no check-ins", entry)
        satisfaction, unread, read_at = await self._rating(context, step_id, entry)
        run = core.with_final_read(run, unread, read_at)  # what the user read after the last check-in, as one more
        entry["read_at_end"] = len(run.turns) - entry["turns"]
        told = _seeded(self._assistant_prompt, context.metadata.get("seed") or {})
        entry["assistant_instructions"] = "shown" if told else "unknown: the user sim's rules taken as in its notes"

        context.record_model(ModelRole.JUDGE.record_key, self.id, model)
        judge = judge_usage.start(context, self.id)
        got, error = await self._ask(context, judge, model, core.run_prompt(run, satisfaction, told), told is not None)
        answered = [call for call in judge_usage.calls() if call["outcome"] == "answered"]
        entry.update(calls=len(answered), usage=_usage(answered), **_cost(answered))
        if got is None:  # raised, not returned: a failed grade, with no score in the entry for anyone to read
            raise RuntimeError(f"{self.type} '{self.id}': the judge gave no log after {MAX_ATTEMPTS} calls ({error})")

        ledger = core.Ledger()
        core.apply_run(ledger, got, run, set(run.tagged_lines or {}) | set(satisfaction or {}), told)
        scored = core.score(ledger, [t.n for t in run.turns])
        entry.update({k: scored[k] for k in ("score", "incident_cost", "outcome_cost", "lost", "tally", "free",
                                             "curve")},
                     outcome=(ledger.outcome or {}).get("outcome"), incidents=len(ledger.incidents),
                     problems=len(ledger.problems))
        entry.update(await self._store_log(context, ledger))
        logger.info(f"{self.type} '{self.id}': {kind} '{step_id}' scored {scored['score']} from "
                    f"{len(ledger.incidents)} incidents, outcome {entry['outcome']}")
        return Verification(criteria=_rows(scored, entry["outcome"]), score=core.normalized(scored["score"]),
                            output=entry)

    def _skip(self, why: str, entry: dict) -> Verification:
        logger.warning(f"{self.type} '{self.id}': skipped, {why}")
        return Verification(skipped=why, output=entry)

    def _find_run(self, context: TaskStepContext) -> tuple[str, str]:
        """The kind of run its run step left and that step's id, as user_sim_judge finds it: always_on when it left an
        always_on record, else a chat."""
        records = context.metadata.get(ALWAYS_ON_KEY) or {}
        reads = context.metadata.get(READS_KEY) or {}
        step_id = self.run_step_id
        if step_id in records:
            return ALWAYS_ON, step_id
        if step_id in reads or _prompt_response(context, step_id) is not None:
            return CHAT, step_id
        raise StepContextError(f"{self.type} '{self.id}': step '{step_id}' left no run to grade (always_on records: "
                               f"{sorted(records)}; prompt_agent user sim reads: {sorted(reads)})")

    async def _rating(self, context: TaskStepContext, step_id: str,
                      entry: dict) -> tuple[Optional[dict[str, str]], Optional[list[dict]], Optional[str]]:
        """From the run's rate_user_satisfaction entry: its lines (S1...), and what the user read only after the last
        check-in, from the rating's stored ask, with when they read it. None of them without a rating; its lines alone
        when the ask can't be read (said in the entry's ``notes``)."""
        rating = (context.metadata.get(SATISFACTION_KEY) or {}).get(step_id) or {}
        if not rating:
            entry["notes"].append("the run has no rate_user_satisfaction entry: no rating lines, and nothing read "
                                  "after the last check-in")
            return None, None, None
        unread = read_at = None
        if rating.get("object_url"):
            try:
                full = await asyncio.to_thread(_load, rating["object_url"])
                unread = (full.get("read") or {}).get("assistant_messages") or []
                read_at = (full.get("read") or {}).get("time")
            except Exception as e:
                entry["notes"].append(f"the rating's stored ask could not be read ({type(e).__name__}): what the user "
                                      "read after the last check-in isn't shown")
        return rating.get("tagged_lines") or None, unread, read_at

    async def _ask(self, context: TaskStepContext, judge: JudgeUsage, model: str, prompt: str,
                   told: bool) -> tuple[Optional[dict], Optional[str]]:
        """The judge's answer, through core's one model call (``model_call``) as rubrics_verifier's direct judge makes
        it, with the run's judge key and base: up to ``MAX_ATTEMPTS`` calls, each in the judge's record, until one
        answers with a whole log (``answer_problems``); else None and why."""
        overrides = context.metadata.get("user_overrides", {})
        route = model_call.resolve_route(
            model, api_key=overrides.get("judge_litellm_api_key") or overrides.get("litellm_api_key"),
            api_base=overrides.get("judge_litellm_base_url") or overrides.get("litellm_base_url"),
            who=f"{self.type} '{self.id}'", key_how=_JUDGE_KEY_HOW)
        messages = [{"role": "system", "content": core.system_text(told)}, {"role": "user", "content": prompt}]
        # Only the answer's room: no temperature, as rubrics_verifier's judge sends none (Claude's thinking models take
        # only their default); a setting the model refuses anyway is dropped once.
        settings = {"max_tokens": MAX_TOKENS} if model_call.takes(route, "max_tokens") else {}
        timeout = self.judge_timeout_seconds or self.DEFAULT_JUDGE_TIMEOUT_SECONDS
        error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            judge_usage.attempt(attempt)
            sent = await judge.input({"messages": messages, "response_format": RESPONSE_FORMAT,
                                      "settings": model_call.sent_params(route, settings)})
            try:
                reply = await model_call.complete(route, messages, timeout=timeout, response_format=RESPONSE_FORMAT,
                                                  settings=settings)
            except asyncio.CancelledError as exc:
                _record_failed_call(judge, exc, model=model, endpoint_host=route.endpoint_host, sent=sent)
                raise
            except Exception as exc:
                _record_failed_call(judge, exc, model=model, endpoint_host=route.endpoint_host, sent=sent)
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
                logger.warning(f"{self.type} '{self.id}': judge attempt {attempt}/{MAX_ATTEMPTS} failed ({error})")
                if settings and model_call.refused_setting(exc):
                    settings = {}
                continue
            judge.call(ModelRole.JUDGE, Usage.from_model_response(reply.usage) if reply.usage else None,
                       model=reply.model, endpoint_host=reply.endpoint_host, reported_by=ReportedBy.PROVIDER,
                       cost=reply.cost.fields(), sent=sent)
            got = core.parse(reply.content)
            wrong = core.answer_problems(got) if got is not None else []
            if got is not None and not wrong:
                return got, None
            error = (f"its answer isn't a whole log: {'; '.join(wrong)}" if got is not None
                     else f"its answer isn't a JSON object: {reply.content[:300]!r}")
            logger.warning(f"{self.type} '{self.id}': judge attempt {attempt}/{MAX_ATTEMPTS}: {error}")
        return None, error

    async def _store_log(self, context: TaskStepContext, ledger: core.Ledger) -> dict[str, Any]:
        """The whole log in the object store, as a verifier's large output is stored (``verifier._store_output``):
        ``log_object_url``; inline as ``log``, with ``log_store_error``, when the store refuses it."""
        body = json.dumps(ledger.to_dict(), ensure_ascii=False).encode()
        try:
            url = await finish_on_thread(functools.partial(_store_output, context.instance_id, self.id, body),
                                         f"Storing {self.type} '{self.id}''s log")
        except Exception as e:
            logger.warning(f"{self.type} '{self.id}': kept the log inline, the object store refused it: {e}")
            return {"log": ledger.to_dict(), "log_store_error": f"{type(e).__name__}: {str(e)[:300]}"}
        return {"log_object_url": url}


def _seeded(text: Optional[str], seed: dict) -> Optional[str]:
    """``text`` with the run's seed filled in, as prompt_agent fills its system prompt (``<key>`` for each key)."""
    if not text:
        return None
    for key, value in seed.items():
        text = text.replace(f"<{key}>", str(value))
    return text


def _rows(scored: dict, outcome: Optional[str]) -> list[dict[str, Any]]:
    """The grade's rows, each weighing 0 so that nothing pooling them moves: one per dimension, its score what the
    dimension cost out of the score's range, and one for the end outcome."""
    span = core.TOP - core.FLOOR
    rows = []
    for dim in core.DIMENSIONS:
        lost = scored["lost"].get(dim, 0.0)
        rows.append({"id": dim, "score": round(max(0.0, 1.0 - lost / span), 4), "result": lost == 0, "weight": 0.0,
                     "lost": lost, "incidents": sum(scored["tally"].get(dim, {}).values())})
    end = scored["outcome_cost"]
    rows.append({"id": "outcome", "score": round(max(0.0, 1.0 - end / span), 4), "result": end == 0, "weight": 0.0,
                 "outcome": outcome, "lost": end})
    return rows
