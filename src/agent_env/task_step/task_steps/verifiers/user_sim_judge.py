"""The user_sim_judge step: after a run with the built-in user sim, grade the user sim (the AI playing the user), not
the assistant, and fail the task when the user sim slipped in a way that could change the assistant's grade, so the
run is done again rather than scored. A task writer adds it after the run's step.

It is a rubrics_verifier (``RubricsVerifierTaskStep``) with its own criteria and its own context. What a task can set
on it, beyond every step's ``id``, ``type``, ``version``, ``depends_on`` and ``retry_config`` (``SETTINGS``):

- ``run_step_id``, required: the always_on or prompt_agent step whose user sim it judges;
- ``rubric_items`` and ``rubric_persona_sha256`` (below);
- ``judge_model``: unset, rubrics_verifier's default, the config's judge model; ``judge_timeout_seconds``: unset,
  rubrics_verifier's;
- ``fail_task_when_not_passing``: the gate (below), on unless set to false;
- ``verifier_id``: the key of its entry.

Every step's ``fail_task_on_error`` must stay true, its default. The rest is fixed, and a task that sets it is refused
when it loads: the judge asks all seven checks, every check but style can fail a run (``checks`` and ``error_checks``
are refused), and each rubrics_verifier setting the judge fixes (``PINNED``: the direct judge, ``use_trajectory`` off,
``output_format`` ``user_sim_checks``, no batch size, no criteria of its own ...) may only be given its fixed value. The
model call, its key and base, its retries and the calls' concurrency are rubrics_verifier's, and a run's
``user_overrides`` change them as they do for every judge (``judge_model``; ``use_agent_judge`` sends the same prompt to
an agent judge, without the system message and the cache marks).

It judges either kind of run the built-in user sim plays (``deploy_user_sim_agent``), the one ``run_step_id`` names:

- an ``always_on`` run: every ``user_sim_read`` in the timeline stored at ``metadata["always_on"][step]["object_url"]``
  (else the record's ``timeline_tail``, which keeps answers but not what the user sim was sent);
- a ``prompt_agent`` chat run: each short entry in ``metadata["user_sim_reads"][step]``, read whole from its
  ``object_url`` (else the entry, which keeps the answer only).

The user sim's instructions are rebuilt from its ``metadata["user_sims"]`` record by ``deploy_user_sim_agent``'s own
path (``recorded_person``, then ``user_sim_versions.setup``), its opening's files included, and checked against the
record's ``prompt_sha256``. ``used`` is never shown to the judge: it is the user sim's own pulse check, not evidence.

What the judge reads: a system message saying it grades the user sim, the format's framing, then the run
(``judge_trajectory_content``: the rebuilt instructions and every check-in, ``user_sim_judge_core.render_run``), then
the criteria. Everything up to the end of the run is the same in every call on a run, so it is marked for the prompt
cache, and the run's first call goes alone so the rest read it from the cache.

The criteria (``user_sim_judge_core.plan_run``): each of the seven checks (``user_sim_judge_core.CHECKS``) is one
criterion and one call, and a task's ``rubric_items`` ride in the call of the check that owns their family, at most
``MAX_ITEMS_PER_CALL`` to a call. Each row has a weight: 1 for every check but style
(``user_sim_judge_core.DEFAULT_ERROR_CHECKS``) and for an item that can make the run a user sim error (its
``can_error``, unset as its family's check), else 0. A weight-0 row is recorded and never gates.

A rubric item is the shape of a criterion in a task's ``usersim.yaml`` (psi-benchmark-tasks), which that repo turns
into this step's settings; this step reads no file:

- ``id`` and ``family`` (A1-A5, F1-F4, G1-G5), both required; ids are unique, and none is a check's name.
- ``kind``: ``hard`` (an exact rule from the user sim's instructions: the judge fails it on any breach), ``general``
  (a judgement: it fails only on a clear instance; the default) or ``auto`` (never sent to the model: ``metric``
  ``max_words`` or ``max_sentences`` is counted on each graded answer, and any answer over ``max`` fails it, naming
  the turn; see ``user_sim_judge_core.count_sentences``). If none is over but the record kept only the start of one,
  or dropped some answers (always_on's tail, when its stored timeline can't be read), the item is ``error``: those
  lengths can't be counted.
- ``check``: the text the judge applies. ``text`` is its old name and still loads.
- ``line``: instead of ``check``, a tag of the user sim's instructions ("P1", "T1", "C2", "D3"): the judge applies
  that line's text from the run's ``metadata["user_sims"][name]["tagged_lines"]``, and the item's row records ``line``
  and ``line_text``. A hard or general item has exactly one of ``check`` and ``line``; an auto item, counted rather
  than judged, takes no ``line``. A tag names the same line only for the same persona, so a line item needs
  ``rubric_persona_sha256``, and grades only on a run whose fingerprint matches it: on a run that recorded none, it
  is ``error``. A tag the run doesn't have makes the item ``error`` ("the user sim's instructions have no line P1").
  Neither makes a call.
- ``name`` and ``short``: a title and one line for reviewers, on the item's row but never shown to the judge.
- ``na_when``: the situation in which it never came up, told to the judge. ``always`` makes it n/a with no call,
  recording ``static_na_reason`` (required with it); ``never`` means it always applies, and an n/a is asked again.
  An auto item is counted, not judged, so it takes only ``always``; any other ``na_when`` is refused when the task loads.
- ``can_error``: whether its fail can make the run a user sim error (weight 1, or 0); unset, as the check that owns
  its family.

A task item never restates a persona line. The persona is the one place that says what the user sim knows and must
not say, and the judge reads all of it with every check. Task items are for counted limits (``auto``) and judgment
calls specific to the task, or they point at a persona line with ``line``.

For example::

    rubric_persona_sha256: 3f1c...  # the run's persona_sha256, all 64 hex digits
    rubric_items:
    - {id: a1_call_rae, family: A1, kind: hard, name: "Rae: yes if it fits",
       check: "If the assistant asks about Rae's order, Tess says yes only if it fits, gives both floors ...",
       na_when: "The assistant didn't raise Rae's order."}
    - {id: f2_her_length, family: F2, kind: auto, check: "Every reply is at most 25 words.", metric: max_words, max: 25}
    - {id: g1_engagement, family: G1, kind: hard, line: P1}

``rubric_persona_sha256`` is the fingerprint of the user sim the items were written for:
``user_sim_common.persona_fingerprint`` of its persona and ``opening_prompt``, which deploy_user_sim_agent records as
the run's ``persona_sha256``. When it's set and differs from the run's, the task items aren't graded and the entry's
``rubric_persona`` (``status``, ``rubric``, ``run``, ``note``) says so; the general checks still run. A run that
recorded no ``persona_sha256`` is ``unknown``, and the items are graded. Edits to the task that leave the persona and
the opening as they were keep the items graded.

What it writes, all in one place, ``metadata["verifications"][verifier_id]``:

- ``format`` (``user_sim_checks``), ``results`` and ``score``, as rubrics_verifier writes them. Each row: ``id`` (a
  check's name or an item's id); what a reviewer reads (a check's ``title`` and ``families``; an item's ``family``,
  ``kind``, ``check`` or ``line`` and ``line_text``, ``name``, ``short``, ``na_when``, ``asked_with``: the call it
  rode in, or ``computed``, ``static_na``, ``no_line``); its ``weight``; then its ``verdict`` (pass, fail, n/a, or
  error when undecided), ``reason``, ``turns``, ``could_change_grade``, ``assistant_asked`` (whether the assistant
  asked for or invited what was graded: recorded only, never part of the gate; None on a row no model graded),
  ``score`` and ``result``. A row lowers the score only on a fail that could change the grade at a weight other
  than 0, so ``score`` is 0 when ``user_sim_error`` is yes, else 1. ``can_be_reward`` is False: the score checks the
  run, so it is never the task's reward.
- ``user_sim_error``: ``value`` yes / no / unknown, ``because`` and ``unknown`` (the row ids), ``fails_task``, and
  ``not_failing`` (why a yes didn't fail the task). ``flag`` when ``value`` is unknown: "user sim judge undecided", or
  "the judge couldn't decide any check — check its setup" when every check is undecided.
- ``run_step_id``, ``run_kind``, ``user_sim``, ``judge_model``; ``instructions`` (``source``, ``prompt_sha256``,
  ``recorded_prompt_sha256``, ``matches``); ``rubric_persona`` (when ``rubric_persona_sha256`` is set); ``notes``
  (what the judge was told about gaps in the record); ``graded_turns``, ``turns``; ``calls``, ``usage``, ``cost`` (the
  calls' costs summed, each what the endpoint billed, else LiteLLM's price map's figure; None when no call's has one)
  and ``unpriced_calls`` (the calls whose cost has no figure, when there are some); and rubrics_verifier's
  ``judge_output_retries`` / ``judge_output_discrepancies`` / ``judge_batches``.
- ``skipped`` instead, with ``score`` None and no rows, for a run that isn't judged: one that ended in an error (the
  always_on record's or the prompt response's ``error_type``: every verifier skips its prompt anyway), a chat run
  whose user sim never read, and a run with no graded turn.

The gate, ``fail_task_when_not_passing`` (on by default): the step raises ``VerifierNotPassingError``, naming the failing rows, when ``user_sim_error`` is
yes and the rebuilt instructions match the recorded ``prompt_sha256``. The task then ends FAILED with no score, and
the run is done again. Whoever caused the slip, the run is spoiled for grading the assistant, so ``assistant_asked``
never changes it. Off, the verdicts are recorded only. When the gate is on, every step that can be the task's reward
(one with a ``verifier_id`` and ``can_be_reward``) must depend on this step, directly or through others, or the task is
refused when it loads.

A judge that can't reach its model fails the run at once, with ``JudgeSetupError`` ("the user sim judge can't reach
its model: ..."): agent-env's ``ConfigError`` (no key, missing config) and LiteLLM's ``AuthenticationError``,
``PermissionDeniedError``, ``NotFoundError`` (an unknown model), ``BadRequestError`` and its kinds,
``UnprocessableEntityError`` and ``BudgetExceededError`` (``_setup_error``). Every other failure is taken as flaky
(``Timeout``, ``RateLimitError``, ``ServiceUnavailableError``, ``InternalServerError``, ``BadGatewayError``,
``APIConnectionError``, an agent judge's ``TimeoutError``, ``JudgeTimeoutError`` or ``AgentUnreachableError``, an answer
still unusable after its retries), and unknown never fails the task: a run is failed
only on a slip the judge found, so a flaky judge never throws away a good run. A judge answer the format can't use is
asked again with the format's correction, up to 3 more times (``DEFAULT_MAX_RETRIES``, 4, counts the first ask;
rubrics_verifier tries a failed model call as many times). A call still without a usable answer leaves its rows
``error``, ``user_sim_error`` unknown and the entry flagged; the step doesn't raise, and nothing the assistant is
scored on changes.

When the task loads it is refused without ``run_step_id``, when that step isn't an always_on or prompt_agent step
(``rate_user_satisfaction.run_step_of``), when the run's user isn't a user sim a deploy_user_sim_agent step deploys,
when this step doesn't come after the run (its ``depends_on``, when set, must name it), when a reward step doesn't
depend on it (above), when ``fail_task_on_error`` is false, when it sets ``checks``, ``error_checks`` or a fixed
rubrics_verifier setting to another value, and when it still sets ``mark_user_sim_error``, which
``fail_task_when_not_passing`` replaces.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from agent_env.config import model_call
from agent_env.task_step.context import PromptResponse, StepContextError, TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import JudgeOutputFormat
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_usage
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerdict, RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator
from agent_env.task_step.task_steps.verifiers.verifier import Verification

logger = logging.getLogger(__name__)

# Not imported: always_on and prompt_agent import the verifiers' peers. Each is the key its module names.
ALWAYS_ON_KEY = "always_on"  # always_on.ALWAYS_ON_KEY
READS_KEY = "user_sim_reads"  # prompt_agent's _BuiltInUserSim
USER_SIMS_KEY = "user_sims"  # deploy_user_sim_agent.USER_SIMS_KEY
ALWAYS_ON, CHAT = "always_on", "prompt_agent"
SYSTEM = ("You grade a user simulator: an AI playing a real person who messages their personal AI assistant. You "
          "grade the user sim, never the assistant.")
COST_DECIMALS = 6  # the entry keeps the judge's cost to a millionth of a dollar
FLAG_UNDECIDED = "user sim judge undecided"
FLAG_NONE_DECIDED = "the judge couldn't decide any check — check its setup"
# The judge's own settings: what a task may set on it, beyond every step's (id, type, version, depends_on ...).
SETTINGS = ("run_step_id", "rubric_items", "rubric_persona_sha256", "judge_model", "judge_timeout_seconds",
            "fail_task_when_not_passing", "verifier_id")
# rubrics_verifier settings the judge fixes, with the value it fixes them at: a task may give only that value.
PINNED: dict[str, Any] = {
    "use_trajectory": False, "use_agent_judge": False, "output_format": "user_sim_checks", "score_aggregator": "all_pass",
    **dict.fromkeys(("criteria", "prompt_id", "default_model", "default_model_api_base", "agent_name",
                     "trajectory_filter", "max_thinking_tokens", "effort", "judge_a2a_agent_id", "judge_sandbox_type",
                     "grading_policy_prompt", "context_facts_for_judge", "grade_up_to_turn", "run_context_for_judge",
                     "judge_batch_size", "judge_batch_concurrency", "judge_timeout_retries", "judge_max_turns",
                     "min_passing_score"))}
NOT_FAILING_OFF = "fail_task_when_not_passing is off: verdicts are recorded only"
NOT_FAILING_UNCONFIRMED = ("the judge's copy of the user sim's instructions isn't confirmed to be the one it ran with, "
                           "so it doesn't fail the task")
FLAG_ASSISTANT = "this answer looks like the assistant's (very long, or markdown)"
FLAG_LAST = "after the run's last wake: this answer never reached the assistant"
FLAG_HELD = "its reply was still waiting to be delivered when the run ended: it never reached the assistant"
# always_on.ASSISTANT_LIKE_MESSAGE_CHARS: a chat run's reads carry no flag of their own, so it's computed the same way.
ASSISTANT_LIKE_MESSAGE_CHARS = 1200
SCHEMA_ERRORS_SHOWN_CHARS = 300  # of a malformed answer's schema errors, in the judge's turn note
USER_SIM_ERROR_SHOWN_CHARS = 200  # of a user sim agent error, in the judge's turn note
ERROR_SHOWN_CHARS = 300  # of why a call got no usable answer, on its rows
# Why a line item isn't graded on a run that recorded no persona fingerprint: its tag may name another line now.
LINE_UNCHECKED = ("the run recorded no persona fingerprint, so its tags can't be checked against the persona the item "
                  "was written for")


@dataclass
class _ThisRun:
    """One run's judging: what it asks, and how its first call went. Held by the run's own copy of the step, never by
    the step a task shares across runs."""
    plan: core.Plan
    text: str  # the run as the judge reads it
    first: Optional[asyncio.Event] = None  # set once the run's first call is over
    first_error: Optional[BaseException] = None  # what the first call raised, if it did


class UserSimJudge(RubricsVerifierTaskStep):
    type: ClassVar[str] = "user_sim_judge"
    entity_refs = ()
    context_paths = ()
    field_scopes = TaskStep.field_scopes
    can_be_reward: ClassVar[bool] = False
    # Each call's answer is asked for up to this many times: the first ask and 3 corrections. rubrics_verifier tries a
    # failed model call the same number of times.
    DEFAULT_MAX_RETRIES: ClassVar[int] = 4

    def __init__(
        self,
        id: str,
        version: Optional[int],
        run_step_id: Optional[str] = None,
        judge_model: Optional[str] = None,
        rubric_items: Optional[list[dict]] = None,
        rubric_persona_sha256: Optional[str] = None,
        fail_task_when_not_passing: bool = True,
        verifier_id: Optional[str] = None,
        judge_timeout_seconds: Optional[int] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        score_aggregator: Optional[ScoreAggregator | str] = None,
        min_passing_score: Optional[float] = None,
    ):
        """``run_step_id`` is required (it is a keyword with a default only so the refusal can say what's missing)."""
        if not isinstance(run_step_id, str) or not run_step_id.strip():
            raise ValueError(f"user_sim_judge '{id}': run_step_id is required: name the always_on or prompt_agent "
                             "step whose user sim it judges")
        if fail_task_on_error is not True:
            raise ValueError(f"user_sim_judge '{id}': fail_task_on_error can't be false: a judge that fails has to "
                             "fail the task, or the steps after it would score a run nobody checked. Leave it out "
                             "(true is the default); set fail_task_when_not_passing to false to record the verdicts "
                             "only.")
        super().__init__(
            id, version, criteria=[], prompt_id=None, default_model=judge_model, verifier_id=verifier_id,
            use_agent_judge=False, use_trajectory=False, output_format=JudgeOutputFormat.USER_SIM_CHECKS,
            judge_timeout_seconds=judge_timeout_seconds, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
            fail_task_when_not_passing=fail_task_when_not_passing is True, score_aggregator=score_aggregator,
            min_passing_score=min_passing_score)
        self.fail_task_when_not_passing = fail_task_when_not_passing  # as given, so config_problems sees a non-bool
        self.run_step_id = run_step_id
        self.judge_model = judge_model  # None: rubrics_verifier's default, the config's judge model
        self.rubric_items = list(rubric_items or [])
        self.rubric_persona_sha256 = rubric_persona_sha256
        self._run: Optional[_ThisRun] = None

    # ---- settings ----------------------------------------------------------------------------------------------
    def criterion_rows(self) -> Optional[int]:
        """Its rows follow the conversation, not its criteria alone."""
        return None

    def config_problems(self) -> list[str]:
        return _problems(self.judge_model, self.rubric_items, self.fail_task_when_not_passing,
                         self.rubric_persona_sha256)

    def task_problems(self, steps: list[TaskStep]) -> list[str]:
        """The run it judges must be an always_on or prompt_agent step's, played by a built-in user sim, and come
        before it; with the gate on, every step that can be the task's reward must depend on it."""
        from agent_env.task.task import _ancestor_ids, _dependency_ids  # here: the task module imports the steps
        from agent_env.task_step.task_steps.agent.rate_user_satisfaction import plays_built_in_user_sim, run_step_of

        run, found = run_step_of(self, steps, does="judge")
        if run is None:
            return found
        where = f"{self.type} '{self.id}'"
        if steps.index(run) > steps.index(self):
            found.append(f"{where} comes before step '{run.id}', the run it judges; put it after")
        if self.depends_on is not None and run.id not in {d.task_step_id for d in self.depends_on}:
            found.append(f"{where}: its depends_on must name '{run.id}', the run it judges, so it runs once that's over")
        if not plays_built_in_user_sim(run, steps):
            found.append(f"{where}: the user of {run.type} step '{run.id}' isn't a user sim a deploy_user_sim_agent "
                         "step deploys; the judge grades only the built-in user sim")
        if self.fail_task_when_not_passing is True:
            deps = _dependency_ids(steps)
            for step in steps:
                if (step is not self and getattr(step, "verifier_id", None) and getattr(step, "can_be_reward", True)
                        and self.id not in _ancestor_ids(step.id, deps)):
                    found.append(f"step '{step.id}' ({step.type}) can be the task's reward but doesn't depend on "
                                 f"{where}, so a run the user sim spoiled could still be scored; make it depend on "
                                 f"'{self.id}', or set fail_task_when_not_passing to false to record the verdicts only")
        return found

    def to_dict(self) -> dict:
        base = TaskStep.to_dict(self)
        base.update(run_step_id=self.run_step_id, judge_model=self.judge_model, rubric_items=self.rubric_items,
                    rubric_persona_sha256=self.rubric_persona_sha256,
                    fail_task_when_not_passing=self.fail_task_when_not_passing, verifier_id=self.verifier_id,
                    judge_timeout_seconds=self.judge_timeout_seconds)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "UserSimJudge":
        where = f"user_sim_judge '{data.get('id')}'"
        if "mark_user_sim_error" in data:
            raise ValueError(
                f"{where}: mark_user_sim_error is gone; fail_task_when_not_passing (on by default) takes its place: "
                "a run whose user sim slipped in a way that could change the grade fails the task, so it is run "
                "again. Remove mark_user_sim_error, and set fail_task_when_not_passing to false to record the "
                "verdicts only.")
        if gone := [k for k in ("checks", "error_checks") if k in data]:
            raise ValueError(
                f"{where}: {' and '.join(gone)} can't be set: the judge always asks all seven checks, and every "
                "check but style can fail a run. Remove them; a task item's can_error says whether it can.")
        if pinned := [k for k, v in PINNED.items() if k in data and data[k] != v]:
            raise ValueError(
                f"{where}: {', '.join(pinned)} can't be set on the user sim judge, which fixes "
                f"{'them' if len(pinned) > 1 else 'it'} ({', '.join(f'{k}={PINNED[k]!r}' for k in pinned)}). "
                "Remove " + ("them" if len(pinned) > 1 else "it") + "; its settings are " + ", ".join(SETTINGS)
                + ".")
        return cls(**cls._base_from_dict(data), run_step_id=data.get("run_step_id"),
                   judge_model=data.get("judge_model"), rubric_items=data.get("rubric_items"),
                   rubric_persona_sha256=data.get("rubric_persona_sha256"),
                   fail_task_when_not_passing=data.get("fail_task_when_not_passing", True),
                   judge_timeout_seconds=data.get("judge_timeout_seconds"))

    def _recorded_values(self, context: TaskStepContext) -> dict[str, Any]:
        """rubrics_verifier's, but the criteria, which each run's plan replaces, and the trajectory filter: its judge
        reads no trajectory."""
        return {k: v for k, v in super()._recorded_values(context).items() if k not in ("criteria", "trajectory_filter")}

    # ---- the run -----------------------------------------------------------------------------------------------
    async def verify(self, context: TaskStepContext) -> Verification:
        problems = self.config_problems()
        if problems:
            raise ValueError(f"user_sim_judge '{self.id}': " + "; ".join(problems))
        items = [core.RubricItem.from_dict(i) for i in self.rubric_items]
        kind, step_id = self._find_run(context)
        response = _prompt_response(context, step_id)
        model = self._judge_model(context)
        entry: dict[str, Any] = {"run_step_id": step_id, "run_kind": kind, "judge_model": model}
        record = (context.metadata.get(ALWAYS_ON_KEY) or {}).get(step_id) if kind == ALWAYS_ON else None
        error_type = (record or {}).get("error_type") or (response.error_type if response else None)
        skipped = None
        if error_type:
            skipped = f"the run ended with error_type={error_type}"
        elif kind == CHAT and not (context.metadata.get(READS_KEY) or {}).get(step_id):
            skipped = "the user sim never read: the run has no user_sim_reads"
        else:
            run, about = await asyncio.to_thread(_read_run, kind, step_id, record, context)
            entry.update(user_sim=about["user_sim"], instructions=about["instructions"])
            if self.rubric_persona_sha256 is not None and items:
                us = (context.metadata.get(USER_SIMS_KEY) or {}).get(about["user_sim"]) or {}
                entry["rubric_persona"] = _rubric_persona(self.rubric_persona_sha256, us.get("persona_sha256"))
                if entry["rubric_persona"]["status"] == "mismatch":
                    items = []
            graded = [t.n for t in run.turns if t.graded]
            entry.update(graded_turns=graded, turns=len(run.turns), notes=list(run.notes))
            if not graded:
                skipped = "no graded turns"
            elif response is None:
                raise StepContextError(f"user_sim_judge '{self.id}': step '{step_id}' left no prompt response")
        if skipped:
            logger.warning(f"user_sim_judge '{self.id}': skipped, {skipped}")
            return Verification(skipped=skipped, output={
                **self._entry_format_fields(), **entry, "user_sim_error": {**core.NOT_JUDGED, "fails_task": False}})

        unusable = LINE_UNCHECKED if (entry.get("rubric_persona") or {}).get("status") == "unknown" else ""
        plan = core.plan_run(run, checks=list(core.CHECK_NAMES), items=items,
                             error_checks=list(core.DEFAULT_ERROR_CHECKS), lines_unusable=unusable)
        judge = copy.copy(self)  # this run's own copy: the step is shared by every run of the task
        judge.prompt_id, judge._run = response.prompt_id, _ThisRun(plan=plan, text=core.run_text(run))
        return self._finish(await judge._judged(context), entry, judge._run)

    def _finish(self, judged: RubricsVerdict, entry: dict, this: _ThisRun) -> Verification:
        """The judge's verdict with its rows in the plan's order and shape, the run's verdict and details, and its
        score: 0 when a row fails, else 1. A slip the gate counts fails the step once recorded."""
        rows = core.finish_rows(judged.criteria, this.plan)
        error = core.user_sim_error(rows)
        error["fails_task"] = False
        if error["value"] == "yes":
            if self.fail_task_when_not_passing is not True:
                error["not_failing"] = NOT_FAILING_OFF
            elif not (entry.get("instructions") or {}).get("matches"):
                error["not_failing"] = NOT_FAILING_UNCONFIRMED
            else:
                error["fails_task"] = True
        answered = [call for call in judge_usage.calls() if call["outcome"] == "answered"]
        output = {**judged.output, **entry, "user_sim_error": error, "calls": len(answered),
                  "usage": _usage(answered), **_cost(answered)}
        if error["value"] == "unknown":
            checks = [r for r in rows if r["id"] in core.CHECKS]
            output["flag"] = FLAG_NONE_DECIDED if checks and all(r["verdict"] == "error" for r in checks) \
                else FLAG_UNDECIDED
        logger.info(f"user_sim_judge '{self.id}': {entry['run_kind']} '{entry['run_step_id']}' user_sim_error="
                    f"{error['value']} ({', '.join(error['because'] + error['unknown']) or 'no row'}), fails_task="
                    f"{error['fails_task']}; " + ", ".join(f"{r['id']}={r['verdict']}" for r in rows))
        slipped = VerifierNotPassingError(
            f"user_sim_judge '{self.id}': the user sim slipped in a way that could change the assistant's grade "
            f"({', '.join(error['because'])}), so the run doesn't count; run it again") if error["fails_task"] else None
        return Verification(criteria=rows, score=0.0 if any(r.get("result") is False for r in rows) else 1.0,
                            output=output, raise_after=slipped)

    def _find_run(self, context: TaskStepContext) -> tuple[str, str]:
        """The kind of run ``run_step_id`` left and its step id: always_on when it left an always_on record, else a
        chat (a chat whose user sim never read leaves a response and no reads, and is skipped)."""
        records = context.metadata.get(ALWAYS_ON_KEY) or {}
        reads = context.metadata.get(READS_KEY) or {}
        if self.run_step_id in records:
            return ALWAYS_ON, self.run_step_id
        if self.run_step_id in reads or _prompt_response(context, self.run_step_id) is not None:
            return CHAT, self.run_step_id
        raise StepContextError(f"user_sim_judge '{self.id}': step '{self.run_step_id}' left no run to judge "
                               f"(always_on records: {sorted(records)}; prompt_agent user sim reads: {sorted(reads)})")

    # ---- rubrics_verifier's hooks, on the run's own copy ---------------------------------------------------------
    def _this(self) -> _ThisRun:
        if self._run is None:
            raise RuntimeError(f"user_sim_judge '{self.id}': the judge's hooks run only on a run's own copy of the step")
        return self._run

    def _resolved_criteria(self, context: TaskStepContext) -> list[dict]:
        return [dict(c) for c in self._this().plan.criteria]

    async def judge_trajectory_content(self, context: TaskStepContext, prompt_response: PromptResponse) -> str:
        return self._this().text

    async def unjudged_rows(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[dict]:
        return list(self._this().plan.unjudged)

    def _criteria_batches(self, criteria: list[dict], size: Optional[int]) -> list[list[dict]]:
        """One call per check, with the task items riding with it, then the calls that ask only items."""
        out, start = [], 0
        for n in self._this().plan.batches:
            out.append(criteria[start:start + n])
            start += n
        return out

    async def _prompt_llm_judge(self, eval_prompt: str, *, model: str, context: TaskStepContext,
                                image_blocks: Optional[list[dict]] = None, system: Optional[str] = None,
                                cached_prefix: Optional[str] = None) -> dict:
        """rubrics_verifier's call, with the system message and everything up to the end of the run marked for the
        prompt cache. The run's first call goes alone, so the others read the run from the cache."""
        this = self._this()
        end = eval_prompt.find(this.text)
        head = eval_prompt[:end + len(this.text)] if end >= 0 else None

        async def call() -> dict:
            return await super(UserSimJudge, self)._prompt_llm_judge(
                eval_prompt, model=model, context=context, image_blocks=image_blocks, system=SYSTEM,
                cached_prefix=head)

        if this.first is None:
            this.first = asyncio.Event()
            try:
                return await call()
            except Exception as e:
                this.first_error = e
                raise
            finally:
                this.first.set()
        await this.first.wait()
        if this.first_error is not None and _setup_error(this.first_error) is not None:
            raise RuntimeError("the run's first judge call couldn't reach its model") from this.first_error
        return await call()

    async def _run_judge_with_output_retries(self, *, criteria: list[dict], **kwargs: Any
                                             ) -> tuple[list[dict], int, list[dict], Optional[str]]:
        """rubrics_verifier's answer retries; a call still without a usable answer leaves its criteria undecided
        (``error``) rather than failing the step: an unknown never fails the task."""
        try:
            return await super()._run_judge_with_output_retries(criteria=criteria, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as e:  # TimeoutError: an agent judge's (run-wide use_agent_judge)
            if (setup := _setup_error(e)) is not None:
                raise JudgeSetupError(f"the user sim judge can't reach its model: {type(setup).__name__}: "
                                      f"{str(setup)[:ERROR_SHOWN_CHARS]}") from e
            cause = _root(e)
            error = (f"no usable answer after {self.DEFAULT_MAX_RETRIES} tries: {type(cause).__name__}: "
                     f"{str(cause)[:ERROR_SHOWN_CHARS]}")
            logger.warning(f"user_sim_judge '{self.id}': {len(criteria)} criteria undecided, {error}")
            return [core.undecided_row(c, error) for c in criteria], 0, [], None


class JudgeSetupError(RuntimeError):
    """The judge can't reach its model as it is set up (no key, a bad key, a model that doesn't exist, a request the
    model refuses): the run fails at once, rather than every check going unknown."""


def _chain(e: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    while e is not None and e not in out:
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def _root(e: BaseException) -> BaseException:
    """What the judge call itself raised: rubrics_verifier wraps a model call that kept failing in a RuntimeError."""
    return _chain(e)[-1]


def _setup_error(e: BaseException) -> Optional[BaseException]:
    """The setup error behind a failed judge call (``model_call.setup_errors``), or None for a flaky one. Anything
    else is flaky (Timeout, RateLimitError, ServiceUnavailableError, InternalServerError, BadGatewayError,
    APIConnectionError, an agent judge's TimeoutError, JudgeTimeoutError or AgentUnreachableError, an answer still
    unusable after its retries) and leaves the call's rows undecided."""
    setup = model_call.setup_errors()
    return next((x for x in _chain(e) if isinstance(x, setup)), None)


def _cost(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's ``cost``: its answered calls' costs summed, None when no call's cost has a figure, and
    ``unpriced_calls``, the calls left out of the sum, when there are some."""
    priced = [call["cost_usd"] for call in calls if call.get("cost_usd") is not None]
    unpriced = len(calls) - len(priced)
    total = round(float(sum(priced)), COST_DECIMALS) if priced or not calls else None
    return {"cost": total, **({"unpriced_calls": unpriced} if unpriced else {})}


def _usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    """The answered calls' token counts summed, as core reads each call's usage (``Usage``), input counting the cache."""
    keys = ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")
    total = dict.fromkeys(keys, 0)
    for call in calls:
        for key in keys:
            total[key] += (call.get("usage") or {}).get(key) or 0
    return total


def _problems(model: Any, items: Any, gate: Any, rubric_persona: Any = None) -> list[str]:
    out = []
    if rubric_persona is not None and not (isinstance(rubric_persona, str)
                                           and re.fullmatch(r"[0-9a-f]{64}", rubric_persona)):
        out.append(f"rubric_persona_sha256 must be a persona fingerprint (64 lowercase hex digits), not "
                   f"{rubric_persona!r}")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        out.append("judge_model must be a model name")
    if not isinstance(gate, bool):
        out.append("fail_task_when_not_passing must be true or false")
    if not isinstance(items, list):
        out.append("rubric_items must be a list of {id, family, kind, check, ...}")
    else:
        out += core.item_problems(items)
        out += [f"rubric_items[{i}] ({item['id']}): id '{item['id']}' is a check's name; give the item another id"
                for i, item in enumerate(items) if isinstance(item, dict) and isinstance(item.get("id"), str)
                and item["id"] in core.CHECKS]
        if rubric_persona is None:
            out += [f"rubric_items[{i}] ({item.get('id')}): a line item needs rubric_persona_sha256, so its tag is "
                    "graded only on runs of the persona it was written for"
                    for i, item in enumerate(items) if isinstance(item, dict) and item.get("line") is not None]
    return out


def _rubric_persona(rubric: str, run: Optional[str]) -> dict:
    """Whether the task items were written for this run's user sim: ``status`` match, mismatch (the items aren't
    graded) or unknown (the run recorded no ``persona_sha256``; they are)."""
    out: dict[str, Any] = {"rubric": rubric, "run": run}
    if run is None:
        return {**out, "status": "unknown",
                "note": "the run recorded no persona fingerprint, so the task items are graded as written"}
    if run != rubric:
        return {**out, "status": "mismatch",
                "note": "the task items were written for a different persona or opening than this run's user sim had, "
                        "so they weren't graded; the general checks were"}
    return {**out, "status": "match"}


def _prompt_response(context: TaskStepContext, step_id: str) -> Optional[PromptResponse]:
    return next((p for p in reversed(context.prompt_responses) if p.step_id == step_id), None)


# ---- the run as the judge reads it ----------------------------------------------------------------------------------
def _read_run(kind: str, step_id: str, record: Optional[dict], context: TaskStepContext) -> tuple[core.UserSimRun, dict]:
    """The user sim's run, and what the entry records about it: ``user_sim`` (its name) and ``instructions``."""
    metadata = context.metadata
    if kind == ALWAYS_ON:
        reads, notes, missing = _always_on_reads(record)
        name = (record.get("user_sim") or {}).get("name")
        multi_day = record.get("multi_day")
    else:
        reads, notes = _chat_reads(metadata.get(READS_KEY, {}).get(step_id) or [])
        missing = 0  # a chat run keeps a summary of every read
        name = _chat_user_sim(metadata, step_id)
        multi_day = False
    us = (metadata.get(USER_SIMS_KEY) or {}).get(name) if name else None
    instructions, check, more = _instructions(us, metadata.get("seed") or {}, multi_day)
    turns = [_turn(n, e) for n, e in enumerate(reads, 1)]
    more += _reply_limit_note(turns, (us or {}).get("max_replies"))
    run = core.UserSimRun(name=name or "the user", instructions=instructions, turns=turns, notes=notes + more,
                          tagged_lines=(us or {}).get("tagged_lines"), missing_answers=missing)
    return run, {"user_sim": name, "instructions": check}


def _reply_limit_note(turns: list[core.Turn], max_replies: Any) -> list[str]:
    """A note when the user sim reached its ``max_replies``, which its instructions never mention: the ending check
    needs it."""
    replies = [t.n for t in turns if t.outcome == "reply" and t.message.strip()]
    if isinstance(max_replies, bool) or not isinstance(max_replies, int) or not 0 < max_replies <= len(replies):
        return []
    return [f"the run ended at the user sim's reply limit: its reply at turn {replies[max_replies - 1]} was the last of "
            f"its {max_replies} and it was never asked again (its instructions don't mention the limit)"]


def _load(url: str) -> dict:
    from agent_env.config import get_config

    return json.loads(get_config().get_object_store_at(url).get(url))


def _always_on_reads(record: dict) -> tuple[list[dict], list[str], int]:
    """Every ``user_sim_read`` in the always_on step's stored timeline, else its tail; with notes for the judge and how
    many of the run's reads the tail dropped (by the record's ``counts.user_sim_reads``)."""
    if record.get("object_url"):
        try:
            timeline = _load(record["object_url"])["timeline"]
            return _with_delivery(timeline, timeline), [], 0
        except Exception as e:
            note = (f"the run's stored record could not be read ({type(e).__name__}); only its last events, without "
                    "what the user sim was sent, are shown")
    else:
        note = "the run's full record wasn't stored; only its last events, without what the user sim was sent, are shown"
    from agent_env.task_step.task_steps.always_on import PENDING_TEXT_CHARS

    tail = record.get("timeline_tail") or []
    reads = []
    for e in _with_delivery(tail, tail):
        message = e.get("message") or ""  # the tail keeps each string's first PENDING_TEXT_CHARS, then "…"
        reads.append({**e, "message_cut": len(message) == PENDING_TEXT_CHARS + 1 and message.endswith("…")})
    total = (record.get("counts") or {}).get("user_sim_reads")
    missing = total - len(reads) if isinstance(total, int) and not isinstance(total, bool) and total > len(reads) else 0
    notes = [note] + ([f"the first {missing} of the run's {total} check-ins aren't kept, so they aren't shown"]
                      if missing else [])
    return reads, notes, missing


def _with_delivery(reads_from: list[dict], timeline: list[dict]) -> list[dict]:
    """The ``user_sim_read`` events, each reply marked ``held`` when no ``user_reply_delivered`` event carries it (its
    ``virtual_time`` is the reply's ``deliver_at``): it was still waiting when the run ended. A delivery comes after its
    read, so a tail that has the read has its delivery too."""
    delivered = {e.get("virtual_time") for e in timeline if e.get("event") == "user_reply_delivered"}
    return [{**e, "held": bool(e.get("deliver_at")) and not e.get("last_read") and e["deliver_at"] not in delivered}
            for e in reads_from if e.get("event") == "user_sim_read"]


def _chat_reads(summaries: list[dict]) -> tuple[list[dict], list[str]]:
    """Each read of a chat run, whole from its ``object_url``, else its summary, which keeps only the answer."""
    reads, missing = [], 0
    for summary in summaries:
        full = None
        if summary.get("object_url"):
            try:
                full = _load(summary["object_url"])
            except Exception as e:
                logger.warning(f"user_sim_judge: could not read {summary['object_url']}: {type(e).__name__}: {e}")
        if full is None:
            missing += 1
        read = {**summary, **(full or {})}
        message = read.get("message") or ""
        read["looks_like_assistant"] = (len(message) > ASSISTANT_LIKE_MESSAGE_CHARS
                                        or bool(re.search(r"^\s*(#|\|)", message, re.M)))
        reads.append(read)
    notes = [f"{missing} of the run's {len(summaries)} check-ins weren't stored whole, so what the user sim was sent "
             "at them isn't shown"] if missing else []
    return reads, notes


def _chat_user_sim(metadata: dict, step_id: str) -> Optional[str]:
    """Which user sim played a chat run: its conversation's user, else the run's only user sim."""
    users = metadata.get(USER_SIMS_KEY) or {}
    conversation_id = (metadata.get("a2a_conversations") or {}).get(step_id)
    if conversation_id:
        try:
            from agent_env.a2a_agent import conversation_store

            source = (conversation_store.get_conversation(conversation_id) or {}).get("source_agent_name")
            name = next((k for k, u in users.items() if source in (k, u.get("agent_name"))), None)
            if name:
                return name
        except Exception as e:
            logger.warning(f"user_sim_judge: could not read conversation {conversation_id}: {type(e).__name__}: {e}")
    return next(iter(users)) if len(users) == 1 else None


def _turn(n: int, e: dict) -> core.Turn:
    flags = []
    if e.get("looks_like_assistant"):
        flags.append(FLAG_ASSISTANT)
    if e.get("schema_errors"):
        flags.append("its answer was malformed, so nothing was sent: " + "; ".join(e["schema_errors"])[:SCHEMA_ERRORS_SHOWN_CHARS])
    elif e.get("retried"):
        flags.append("its first answer was malformed and it answered again")
    if e.get("user_sim_error"):
        flags.append(f"the user sim agent failed at this check-in: {str(e['user_sim_error'])[:USER_SIM_ERROR_SHOWN_CHARS]}")
    if e.get("last_read"):
        flags.append(FLAG_LAST)
    elif e.get("held"):
        flags.append(FLAG_HELD)
    when = e.get("virtual_time") or (e.get("read") or {}).get("time")
    return core.Turn(n=n, outcome=e.get("outcome") or "no_reply", message=e.get("message") or "",
                     input=e.get("input"), flags=flags, time=None if e.get("input") or not when else f"{when} UTC",
                     cut=bool(e.get("message_cut")))


def _instructions(us: Optional[dict], seed: dict, multi_day: Optional[bool]) -> tuple[Optional[str], dict, list[str]]:
    """The user sim's system prompt, rebuilt from its ``user_sims`` record as deploy_user_sim_agent wrote it, what the
    entry records of the check against the record's ``prompt_sha256``, and notes for the judge."""
    if not us:
        return None, {"source": "missing", "matches": False}, [
            "the user sim's settings were not found, so its instructions aren't shown"]
    from agent_env.task_step.task_steps.agent import user_sim_versions
    from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import recorded_person

    try:
        person = recorded_person(us, seed, multi_day)
        prompt = user_sim_versions.setup(us.get("user_version") or user_sim_versions.VERSION, person)["system_prompt"]
    except Exception as e:
        return None, {"source": "missing", "matches": False, "error": f"{type(e).__name__}: {e}"}, [
            f"the user sim's instructions could not be rebuilt ({type(e).__name__}), so they aren't shown"]
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    recorded = us.get("prompt_sha256")
    check = {"source": "rebuilt", "prompt_sha256": sha, "recorded_prompt_sha256": recorded,
             "matches": recorded == sha}
    if recorded is None:
        return prompt, check, ["these instructions were rebuilt from the user sim's settings, and the run recorded no "
                               "hash of the ones it got, so they may differ from what it got"]
    if recorded != sha:
        return prompt, check, ["these instructions were rebuilt from the user sim's settings and differ from the ones "
                               "it got, so a line may be missing or changed"]
    if not person["opening"]:
        return prompt, check, ["the opening request wasn't recorded, so it is missing from these instructions"]
    return prompt, check, []
