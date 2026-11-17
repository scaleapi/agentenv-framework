"""always_on_agent_verifier: the rubrics_verifier for an always_on step's prompt.

The judge's prompt holds what the user saw and the run's context: the user's chat (the prompt's response and
conversation), the trigger log, and the run wake by wake with the app changes each made (``render_for_judge``). With
use_trajectory, an agent judge also gets every tool call of every wake, staged in its sandbox as files it searches
(``always_on_staging``): one file per wake with one event per line, matched to its wake by the run's timeline, an
index, every event in one file and the trigger firings. A direct judge never gets the trajectories, so the task is
refused when it loads with use_trajectory and a direct judge, an output format that cites per-criterion screenshots,
or grade_up_to_turn (the conversation's turns are not the run's wakes); a run whose overrides ask for either is
refused when the verifier runs. Unlike rubrics_verifier, its judge has 2,400 s and 400 turns per call by default, a
call that runs out of time is tried once more, at most 3 batches run at once, and the judge has only Bash and Read: no
helper agents, no web, no writes, no messaging other agents. An always_on step's prompt takes this verifier, never a
plain rubrics_verifier, and this verifier only an always_on step's prompt: a task that pairs them otherwise is refused when it loads.

For a run of the PSI neutral harness with compaction (its spans name their agent, gen_ai.agent.name), the judge is told the compaction
summary is the assistant's own and graded as its own, and, with a per-criterion format, says for each failed criterion
whether it traces to context a compaction lost (``context_loss`` on the row; the entry counts them): reported next to
the score, never changing it. Every entry also says what the run records of the assistant's harness
(``assistant_harness``: its agent name, its card's name and version and, when its wakes are staged, the settings
its trace reports).

With ``grade_user_rules`` (off by default; it needs use_trajectory) the judge also grades the user's rules, the user
sim's ``persona.permissions``: one criterion per rule (``always_on_user_rules``), graded from ``rules.md``, which lists
every app change the assistant made, every write call no app change recorded, every message it sent the user and
every message from the user in the order they happened, each reply with the message it answers. The entry's
``user_rules`` has each rule's result with what it cites, the verdict, and ``unlogged_writes``: per tool, the write
calls no app's change log recorded. The score is the task's own criteria aggregated as usual (``task_criteria_score``),
0 when an ask_first or never rule failed, and SCORE_RULE_PENALTY less for each just_do or tell_after rule broken. A
judge call that leaves rules unanswered is asked again, up to RULE_RETRIES times, each time for only the rules still
unanswered, with the same prompt and files; a rule unanswered after that makes the run ungradable (a half-finished
grade is never reported): the entry has no score and the step raises UngradableRunError, as a judge that can't answer at all does, so the run is
left out rather than failed. A run whose timeline can't be read is refused rather than graded without its rules.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import re
import uuid
from contextvars import ContextVar
from typing import Any, ClassVar, Optional

from agent_env.providers.sandbox_providers.sandbox import into_container
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.always_on import ALWAYS_ON_KEY, AlwaysOnTaskStep, render_for_judge, stored_timeline
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.always_on_user_rules import (
    RULES_FILE,
    answered,
    is_rule_row,
    persona_rules,
    rule_criteria,
    rule_events,
    rules_block,
    rules_file,
    rules_score,
    run_permissions,
    unlogged_by_tool,
    unlogged_writes,
)
from agent_env.task_step.task_steps.verifiers.always_on_staging import (
    EVENTS_FILE,
    INDEX_FILE,
    RESULTS_DIR,
    TRIGGERS_FILE,
    WAKES_DIR,
    StagedRun,
    judge_reads,
    stage_run,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JudgeOutputFormat,
    JudgeOutputFormatSpec,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import TurnEvents, turn_events
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    RUN_CONTEXT_PARTS,
    RubricsVerifierTaskStep,
    StagedTrajectory,
)
from agent_env.task_step.task_steps.verifiers.run_context import Run, load_run_context
from agent_env.task_step.task_steps.verifiers.scoring import aggregate_score

logger = logging.getLogger(__name__)

# Where the run's files go in the judge's sandbox, under a directory of the run's own.
STAGING_DIR = "/tmp/always_on_run"
# Trajectories read, and files written into the judge's sandbox, at once.
STAGING_CONCURRENCY = 8
# The only tools of its own the judge gets: it reads and searches the files (the CLI's search is Bash's grep), and its
# answer comes back through the structured output the CLI adds for the output format. No helper agents, which ran a
# second judge with its own verdict; no web; no writes; and no messaging other agents (peer_tools).
JUDGE_TOOLS = ("Bash", "Read")
_FILES_NOTE = """

## The run's files
The trajectory path above is a directory holding the whole run, one JSON event per line:
- {index}: start here. A block per wake: its pretend time, what woke the assistant, the user's message, each write \
action (tool and key arguments; how actions were told apart is at its top) and message to the user with the \
`file:line` it is at, and errors.
- {first} … {last}: one file per wake. First what woke the assistant ("woke", the user's own messages in full as \
"user_message", the whole text it was sent as "wake_input"); then every tool call of its turn with input and result \
("tool_call", numbered "call"), its own text ("assistant_text", which nobody saw) and its thinking when it was \
recorded ("assistant_thinking": why it acted; what it did is in the tool calls and app changes); then the rest of the wake in time \
order: every app change ("app_change", "by": "agent" for its own), its closing reply ("turn_end"; the user never saw \
it), each message the user got ("message_to_user"), a nudge and its calls, and the user's reads of its messages.
- {results}/: what is too long for a line (a tool's input or result, a long message): the line names the file \
("<field>_file", such as "result_file" or "text_file") and shows its start ("<field>_preview"). Email bodies are \
decoded (body.text). A line longer than 1,800 characters is split across lines there, nothing dropped.
- {events}: every wake's events in one file, so "did it ever do X" is one search. {triggers}: every trigger firing, \
fired or failed, with its pretend time and the wake it came after.

How to read them:
- Don't load whole files. Search with grep -n, then Read with an offset and limit.
- Later actions replace earlier ones: after you find something, look for later corrections, edits, cancellations or \
follow-ups before you decide.
- Before you conclude that something never happened, or was never said or done, search every wake for it, long \
texts included: grep -r {wakes}/ (the wake files and {results}/, where a text too long for a line is kept whole; \
{events} has only its first 300 characters), and name that search in your reason.
- The user's messages are the "user_message" lines (and the first wake's "wake_input"); the user saw only the \
"message_to_user" lines.
- Cite the wake (and call) number of every piece of evidence, and quote it.
- Only the assistant's tool calls and its "agent" app changes are its acts. A draft is not a sent message. Messages \
from the user, the user simulator or contacts are never the assistant's. Order events by their pretend time ("at"), \
not by where they sit in the files.
- You have Bash and Read only: no web, no other agents, and nothing to change.{gaps}{harness}"""
# Added to _FILES_NOTE for a run whose trajectories say which agent made each step (the PSI harness's: compaction,
# memory, helpers), where these lines appear; a run without them gets the note as before.
_HARNESS_NOTE = """
- "harness_note" lines are from the harness that runs the assistant, never from the assistant or the user: a message \
the harness sent the assistant, or a compaction summary that replaced older history in what the assistant sees.
- Lines with "step": "pre_compaction_save" are a turn the harness asked for before a compaction. What the assistant \
wrote there went to the harness, not the user, and it ran before the assistant saw that wake's message.
- Tool calls marked "internal" are tools the harness itself provides (notes, look-ups of past wakes). They count as \
what the assistant did, but they change nothing in any app.
- Lines with "subagent" are from a helper agent the assistant started.
- Harness notes have no "at": read them in file order.
- A compaction summary is the assistant's own summary of its history: its own model wrote it. If the summary lost or \
got wrong something the assistant needed, the assistant is responsible for what follows, the same as for any other \
mistake. Grade it normally."""
# Added after _HARNESS_NOTE when the judge's rows are scored per criterion (rubric_binary, rubric_partial): its schema
# then takes the two fields (``_with_context_loss``).
_CONTEXT_LOSS_NOTE = """
- For each criterion you fail, also return "context_loss": true when the failure traces to information the \
assistant no longer had because of a compaction (a fact that was in a summarized wake but not in the summary, or that \
the summary got wrong), else false; and "context_loss_evidence": which compaction note (file:line) and which earlier \
wake held the fact, or "" when false. This is reported next to the score and never changes it."""
# The per-criterion row fields a judge of such a run may add, and the formats whose rows take them.
CONTEXT_LOSS_FIELDS = {"context_loss": {"type": "boolean"}, "context_loss_evidence": {"type": "string"}}
_CONTEXT_LOSS_FORMATS = frozenset({JudgeOutputFormat.RUBRIC_BINARY, JudgeOutputFormat.RUBRIC_PARTIAL})
# Whether the run being graded (this task's context) has the harness's agent labels and a per-criterion format, so its
# judge is asked for context loss: set by stage_trajectory, read by _spec, per run (one step serves concurrent runs).
_CONTEXT_LOSS_RUN: ContextVar[bool] = ContextVar("always_on_context_loss_run", default=False)
_RULES_NOTE = """

## The user's rules
{path} lists the user's rules (the user_rules.* criteria) and every app change the assistant made (C…), write call \
no app change recorded (W…), message it sent the user (M…) and message from the user (U…), in the order they \
happened, each reply with the message it answers. Decide which changes fall under each rule from what they did, then check the order there. Open the wake \
files it points to for a message's whole text or a change's tool call."""
# The staging record's key for the events rules.md numbered, taken off the record once the rules are graded.
_EVENTS_KEY = "_user_rules_events"


# The formats whose rows are one per criterion, which the user's rules are graded as.
_RULE_FORMATS = frozenset({JudgeOutputFormat.RUBRIC_BINARY, JudgeOutputFormat.RUBRIC_PARTIAL})


def _rules_problem(use_trajectory: bool, output_format: JudgeOutputFormat) -> Optional[str]:
    """Why grade_user_rules can't be set with these settings, or None."""
    if not use_trajectory:
        return ("grade_user_rules needs use_trajectory: the rules are graded from the run's events staged as files "
                "for the agent judge")
    if output_format not in _RULE_FORMATS:
        return (f"grade_user_rules needs a format with a row per criterion ({', '.join(sorted(f.value for f in _RULE_FORMATS))}); "
                f"output_format={output_format.value} has no row for each rule, so every run would be ungradable")
    return None


def _rule_rows_may_be_missing(spec: JudgeOutputFormatSpec) -> JudgeOutputFormatSpec:
    """``spec`` whose responses may leave out user-rule rows: a response whose only problem is missing rule rows is
    parsed against the criteria it answered, and each rule it left out gets an unanswered row (score 0, no
    justification), as a parse failure's stand-in rows are. Any other problem is the format's own."""
    diagnose = spec.diagnose_response

    def diagnose_with_missing_rules(response: str, criteria: list[dict]):
        rows, problem = diagnose(response, criteria)
        rules = {c["id"] for c in criteria if "user_rule" in c}
        if (problem is None or problem.parse_error or problem.duplicate_ids or problem.unknown_ids
                or problem.non_finite_score_ids or not problem.missing_ids or not set(problem.missing_ids) <= rules):
            return rows, problem
        missing = set(problem.missing_ids)
        answered_rows, again = diagnose(response, [c for c in criteria if c["id"] not in missing])
        if again is not None:
            return rows, problem
        by_id = {r["id"]: r for r in answered_rows}
        return [by_id.get(c["id"]) or {**c, "score": 0.0, "result": False, "justification": ""}
                for c in criteria], None

    return dataclasses.replace(spec, diagnose_response=diagnose_with_missing_rules)


class UngradableRunError(RuntimeError):
    """The judge gave no answer for a user rule, asked RULE_RETRIES more times: the run wasn't graded."""


# How many more times a judge call that leaves user rules unanswered is made, each for the rules still unanswered.
RULE_RETRIES = 3
_RETRY_NOTE = ("\n\nThe other criteria were graded already. These {count} were left without an answer: return exactly "
               "these {count}, each with its justification.")


def _rules_retry_prompt(prompt: str, criteria: list[dict], batch_note: str) -> str:
    """``prompt`` asking for ``criteria`` alone: its criteria list (between "## Criteria" and the next section)
    replaced by theirs, a batch's note on its share of the criteria dropped, and a note that only these are left."""
    head, sep, rest = prompt.partition("## Criteria\n")
    if not sep:
        raise RuntimeError("the judge's prompt has no '## Criteria' section to ask the unanswered rules from")
    _, nxt, tail = rest.partition("\n\n## ")
    pattern = re.escape(batch_note)
    for key in ("first", "last", "total", "count"):
        pattern = pattern.replace(re.escape("{" + key + "}"), r"\d+")
    listed = [{**c, "negative": True} if (c.get("weight") is not None and c["weight"] < 0) else c for c in criteria]
    body = head + sep + json.dumps(listed, indent=2) + nxt + tail
    return re.sub(pattern, "", body) + _RETRY_NOTE.format(count=len(criteria))


class AlwaysOnAgentVerifier(RubricsVerifierTaskStep):
    type: ClassVar[str] = "always_on_agent_verifier"
    entity_refs = RubricsVerifierTaskStep.entity_refs  # the judge, as rubrics_verifier's
    context_paths = RubricsVerifierTaskStep.context_paths
    DEFAULT_JUDGE_TIMEOUT_SECONDS: ClassVar[int] = 2400
    DEFAULT_JUDGE_BATCH_CONCURRENCY: ClassVar[int] = 3
    DEFAULT_JUDGE_TIMEOUT_RETRIES: ClassVar[int] = 1
    # A judge reading a 100+ wake run spends a few calls a criterion (index, search, read) on top of finding its way:
    # 30 criteria unbatched is 100-200 turns. The cap bounds a judge that loops; the time limit (2,400 s at about 5-10
    # s a turn) bounds it near here anyway.
    DEFAULT_JUDGE_MAX_TURNS: ClassVar[int] = 400

    def __init__(
        self,
        id: str,
        version: Optional[int],
        *,
        use_trajectory: bool = False,
        use_agent_judge: bool = True,
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
        judge_timeout_retries: Optional[int] = None,
        judge_max_turns: Optional[int] = None,
        grade_user_rules: bool = False,
        **fields: Any,
    ):
        """``run_context_for_judge`` defaults to every part of the run's context; ``grade_user_rules`` grades the
        user sim's rules too (the module's docstring); the rest is rubrics_verifier's."""
        output_format = JudgeOutputFormat(output_format)
        where = f"{self.type} '{id}'"
        if use_trajectory and not use_agent_judge:
            raise ValueError(f"{where}: use_trajectory needs the agent judge (use_agent_judge): an always_on run's "
                             "wakes are staged as files for a judge that searches them, never inlined")
        if get_judge_output_format_spec(output_format).cites_evidence:
            raise ValueError(f"{where}: output_format={output_format.value} cites per-criterion screenshots, and an "
                             "always_on step's prompt is graded without its wakes' screenshots")
        if grade_user_rules and (problem := _rules_problem(use_trajectory, output_format)):
            raise ValueError(f"{where}: {problem}")
        if grade_up_to_turn is not None:
            raise ValueError(f"{where}: grade_up_to_turn can't cut an always_on step's prompt, whose conversation "
                             "turns are not its wakes")
        super().__init__(
            id, version, use_trajectory=use_trajectory, use_agent_judge=use_agent_judge,
            run_context_for_judge=run_context_for_judge or list(RUN_CONTEXT_PARTS), output_format=output_format,
            default_model=default_model, default_model_api_base=default_model_api_base, effort=effort,
            max_thinking_tokens=max_thinking_tokens, judge_a2a_agent_id=judge_a2a_agent_id,
            judge_sandbox_type=judge_sandbox_type, judge_batch_size=judge_batch_size,
            judge_batch_concurrency=judge_batch_concurrency,
            judge_timeout_retries=(self.DEFAULT_JUDGE_TIMEOUT_RETRIES if judge_timeout_retries is None
                                   else judge_timeout_retries),
            judge_max_turns=self.DEFAULT_JUDGE_MAX_TURNS if judge_max_turns is None else judge_max_turns,
            **fields,
        )
        self.grade_user_rules = bool(grade_user_rules)

    @classmethod
    def from_dict(cls, data: dict) -> AlwaysOnAgentVerifier:
        step = super().from_dict({**data, "use_trajectory": data.get("use_trajectory", False)})
        if data.get("grade_user_rules"):
            if problem := _rules_problem(step.use_trajectory, step.output_format):
                raise ValueError(f"{cls.type} '{step.id}': {problem}")
            step.grade_user_rules = True
        return step

    def to_dict(self) -> dict:
        base = super().to_dict()
        if self.grade_user_rules:
            base["grade_user_rules"] = True
        return base

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
        if self.use_trajectory and (context.metadata.get("user_overrides") or {}).get("use_agent_judge") is False:
            raise RuntimeError(f"Verifier '{self.verifier_id}': use_trajectory needs the agent judge; this run's "
                               "use_agent_judge=False would grade without the wakes it was set to read.")
        return await super().execute(context)

    async def _grade(self, context: TaskStepContext) -> TaskStepContext:
        """rubrics_verifier's grade; then, for a staged run, what each judge call read of it (``judge_reads``); for a
        run with agent labels, how many failed criteria the judge traced to a compaction (``context_loss``); for every
        run, the assistant's harness as the run records it (``assistant_harness``); and with grade_user_rules, the
        rules' results and the verdict they set, last, so an ungradable run's entry has all the rest when
        UngradableRunError is raised."""
        token = _CONTEXT_LOSS_RUN.set(False)
        try:
            context = await super()._grade(context)
            asked = _CONTEXT_LOSS_RUN.get()
        finally:
            _CONTEXT_LOSS_RUN.reset(token)
        entry = (context.metadata.get("verifications") or {}).get(self.verifier_id)
        if entry is None:
            return context
        staged_rules = None
        if "trajectory_staging" in entry:
            staged_rules = entry["trajectory_staging"].pop(_EVENTS_KEY, None)
            entry["trajectory_staging"]["judge_reads"] = await self._judge_reads(entry)
        if asked:
            entry["context_loss"] = context_loss(entry.get("results") or [])
        if harness := await self._assistant_harness(context, staged="trajectory_staging" in entry):
            entry["assistant_harness"] = harness
        if self.grade_user_rules and staged_rules is not None:
            self._grade_rules(context, entry, staged_rules)
        return context

    async def _run_judge_with_output_retries(self, **kwargs: Any) -> tuple[list[dict], int, list[dict], Optional[str]]:
        """rubrics_verifier's judge call; with grade_user_rules, a call that leaves rules unanswered (no
        justification) is made again, up to RULE_RETRIES times, each time with only the rules still unanswered (the
        same prompt and files, its criteria list cut to them), until every rule has an answer. One
        ``judge_output_discrepancies`` entry keeps the retries by rule id: how many, each one's rules, those it
        answered and its trajectory, and any still unanswered."""
        rows, retries, log, url = await super()._run_judge_with_output_retries(**kwargs)
        blank = [r["id"] for r in rows if "user_rule" in r and not answered(r)] if self.grade_user_rules else []
        if not blank:
            return rows, retries, log, url
        # The judge's short criterion keys (c1…) as the rules' own ids, for the record.
        rule_of = {c["id"]: c["user_rule"]["id"] for c in kwargs["criteria"] if "user_rule" in c}
        first, attempts = list(blank), []
        for n in range(1, RULE_RETRIES + 1):
            logger.warning("Verifier '%s': the judge left %d user rules unanswered (%s); retry %d of %d",
                           self.verifier_id, len(blank), blank, n, RULE_RETRIES)
            criteria = [c for c in kwargs["criteria"] if c["id"] in blank]
            prompt = _rules_retry_prompt(kwargs["eval_prompt"], criteria, self._BATCH_NOTE)
            again, more, log_again, url_again = await super()._run_judge_with_output_retries(
                **{**kwargs, "criteria": criteria, "eval_prompt": prompt})
            fixed = {r["id"]: r for r in again if r["id"] in blank and answered(r)}
            rows = [fixed.get(r["id"], r) for r in rows]
            retries, log = retries + more + 1, [*log, *log_again]
            attempts.append({"retry": n, "asked": [rule_of[i] for i in blank],
                             "answered": [rule_of[i] for i in blank if i in fixed],
                             "judge_trajectory_object_url": url_again})
            blank = [i for i in blank if i not in fixed]
            if not blank:
                break
        log.append({"kind": "unanswered_user_rules", "ids": [rule_of[i] for i in first], "retries": len(attempts),
                    "still_unanswered": [rule_of[i] for i in blank], "attempts": attempts})
        return rows, retries, log, url

    def _rules(self, context: TaskStepContext) -> list:
        """The user sim's rules this run graded, from deploy_user_sim_agent's record of the user it ran with."""
        prompt_response = next((pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id), None)
        if prompt_response is None:
            return []
        permissions = run_permissions(context.metadata, self._record(context, prompt_response))
        if permissions is None:
            raise RuntimeError(f"Verifier '{self.verifier_id}': grade_user_rules needs the user sim's rules, but no "
                               "deploy_user_sim_agent record of the run's user sim is in its metadata")
        return persona_rules(permissions)

    def _resolved_criteria(self, context: TaskStepContext) -> list[dict]:
        """The task's criteria, then with grade_user_rules one per user rule, graded from rules.md."""
        criteria = super()._resolved_criteria(context)
        if not self.grade_user_rules:
            return criteria
        return [*criteria, *rule_criteria(self._rules(context), RULES_FILE)]

    def _grade_rules(self, context: TaskStepContext, entry: dict, staged: dict) -> None:
        """The entry's ``user_rules`` and its score (``rules_score`` of the task's own criteria, aggregated as usual,
        1.0 with none). An ungradable run keeps its entry with no score, and raises UngradableRunError."""
        rows = entry.get("results") or []
        events, writes = staged["events"], staged["writes"]
        block = rules_block([r for r in rows if is_rule_row(r)], self._rules(context), events, writes)
        own = [r for r in rows if not is_rule_row(r)]
        block["task_criteria_score"] = aggregate_score(own, self.score_aggregator) if own else 1.0
        block["unlogged_writes"] = unlogged_by_tool(writes, events)
        if staged["unchecked"]:
            block["unlogged_writes_unchecked_wakes"] = staged["unchecked"]
        entry["user_rules"] = block
        entry["score"] = rules_score(block, block["task_criteria_score"])
        logger.info("Verifier '%s': the user's rules: %s (%s); score %s", self.verifier_id, block["verdict"],
                    ", ".join(f"{r['id']} {r['result']}" for r in block["rules"]) or "no rules", entry["score"])
        if block["verdict"] == "ungradable":
            unanswered = [r["id"] for r in block["rules"] if r["result"] == "unknown"]
            entry["ungradable"] = f"the judge gave no answer for {unanswered} after {RULE_RETRIES} retries"
            raise UngradableRunError(f"Verifier '{self.verifier_id}': ungradable: the judge gave no answer for user "
                                     f"rules {unanswered} after {RULE_RETRIES} retries")

    def _spec(self) -> JudgeOutputFormatSpec:
        """``_context_loss_spec``; with grade_user_rules, a response that leaves out only rule rows is taken with
        those rows unanswered (``_rule_rows_may_be_missing``), so the rows it did answer are kept and the rules
        left out go to the rule-only retries instead of the whole batch being asked again."""
        spec = self._context_loss_spec()
        # rubrics_verifier's constructor reads the format before this step's own settings are set.
        return _rule_rows_may_be_missing(spec) if getattr(self, "grade_user_rules", False) else spec

    def _context_loss_spec(self) -> JudgeOutputFormatSpec:
        """rubrics_verifier's format; for a run with agent labels (``_CONTEXT_LOSS_RUN``), its rows also take ``context_loss`` and
        ``context_loss_evidence``."""
        spec = super()._spec()
        return _with_context_loss(spec) if _CONTEXT_LOSS_RUN.get() else spec

    async def _assistant_harness(self, context: TaskStepContext, *, staged: bool) -> dict:
        """The graded assistant's harness as the run records it: its agent name, its A2A card's name and version, and,
        for a run whose wakes were ``staged`` (so its trajectories are read anyway), the settings its trace reports
        (``harness_settings`` of the last wake's trajectory). A run graded without its trajectories has none read.
        What isn't there is left out."""
        response = next((pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id), None)
        if response is None:
            return {}
        out: dict = {}
        if response.agent_name:
            out["agent_name"] = response.agent_name
            agent = next((a for a in context.deployed_agents if a.agent_name == response.agent_name), None)
            card = {k: (agent.a2a_card or {}).get(k) for k in ("name", "version")} if agent else {}
            if card := {k: v for k, v in card.items() if v}:
                out["agent_card"] = card
        urls = [u for u in (response.target_agent_per_turn_trajectory_object_urls or []) if u]
        url = urls[-1] if urls else response.agent_trajectory_object_url
        if staged and url:
            try:
                settings = harness_settings(await asyncio.to_thread(self._read_trajectory_text, url))
            except Exception as e:
                logger.warning("Verifier '%s': the assistant's harness settings could not be read from %s (%s: %s)",
                               self.verifier_id, url, type(e).__name__, e)
                settings = {}
            if settings:
                out["settings"] = settings
                out["settings_from"] = url
        return out

    def judge_agent_config(self, judge_agent_card: dict) -> dict:
        """No messaging other agents, and, when the judge's card offers them by these names (Claude Code's), only
        Bash and Read of its own tools. Another harness names its tools differently and would refuse Claude Code's
        names, so its own tools are left as they are, and the log says so."""
        config: dict = {"peer_tools": False}
        if set(JUDGE_TOOLS) <= _builtin_tool_names(judge_agent_card):
            config["builtin_tools"] = {"only": list(JUDGE_TOOLS)}
        else:
            logger.warning("Verifier '%s': the judge agent doesn't offer %s as built-in tools, so its own tools aren't "
                           "narrowed to them", self.verifier_id, list(JUDGE_TOOLS))
        return config

    def _record(self, context: TaskStepContext, prompt_response: PromptResponse) -> dict:
        record = (context.metadata.get(ALWAYS_ON_KEY) or {}).get(prompt_response.step_id)
        if record is None:
            raise RuntimeError(f"Verifier '{self.verifier_id}': prompt_id='{self.prompt_id}' is not an always_on "
                               f"step's prompt (step '{prompt_response.step_id}' left no always_on record); grade it "
                               "with rubrics_verifier.")
        return record

    async def judge_prompt_sections(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[str]:
        """The run wake by wake, from the always_on step's record of it."""
        return [await asyncio.to_thread(render_for_judge, self._record(context, prompt_response))]

    async def stage_trajectory(self, context: TaskStepContext, prompt_response: PromptResponse,
                               run_context: Optional[Run], sandbox: Any,
                               container: Optional[str]) -> Optional[StagedTrajectory]:
        """The run as files (``stage_run``) in the judge's sandbox. Without its stored timeline, which matches each
        trajectory to its wake, None: the per-turn trajectories are loaded as rubrics_verifier loads them."""
        record = self._record(context, prompt_response)
        if self.grade_user_rules and not record.get("object_url"):
            raise RuntimeError(f"Verifier '{self.verifier_id}': grade_user_rules needs the always_on run's stored "
                               "timeline, and this run kept none")
        if not record.get("object_url"):
            logger.warning("Verifier '%s': the always_on run kept no stored timeline; its wakes' trajectories are "
                           "loaded by position instead", self.verifier_id)
            return None
        try:
            timeline = await asyncio.to_thread(stored_timeline, record)
        except Exception as e:
            if self.grade_user_rules:
                raise RuntimeError(f"Verifier '{self.verifier_id}': grade_user_rules needs the always_on run's stored "
                                   f"timeline, which could not be read ({type(e).__name__}: {e})") from e
            logger.warning("Verifier '%s': the always_on run's timeline could not be read (%s: %s); its wakes' "
                           "trajectories are loaded by position instead", self.verifier_id, type(e).__name__, e)
            return None
        urls = list(dict.fromkeys(e["trajectory_url"] for e in timeline
                                  if e.get("event") in ("agent_turn_end", "nudge") and e.get("trajectory_url")))
        turns = await self._read_turns(urls)
        run = run_context or await load_run_context(context, prompt_response)
        staged = await asyncio.to_thread(stage_run, timeline, turns, firings=run.firings_read(),
                                         unavailable=run.unavailable)
        _CONTEXT_LOSS_RUN.set(bool(staged.record.get("agent_labeled_trajectories"))
                              and self.output_format in _CONTEXT_LOSS_FORMATS)
        path = f"{STAGING_DIR}/{uuid.uuid4().hex[:12]}"
        note = _files_note(path, staged, context_loss=_CONTEXT_LOSS_RUN.get())
        if self.grade_user_rules:
            events = rule_events(timeline)
            writes, unchecked = unlogged_writes(staged.files, events)
            staged.files[RULES_FILE] = rules_file(self._rules(context), events, writes, timeline, staged.files,
                                                  unchecked)
            staged.record[_EVENTS_KEY] = {"events": events, "writes": writes, "unchecked": unchecked}
            note += _RULES_NOTE.format(path=f"{path}/{RULES_FILE}")
        await self._write_files(sandbox, {f"{path}/{name}": text for name, text in staged.files.items()}, container)
        self._log_staging(staged)
        return StagedTrajectory(path=path, prompt_note=note, record=staged.record)

    async def _read_turns(self, urls: list[str]) -> dict[str, Optional[TurnEvents]]:
        gate = asyncio.Semaphore(STAGING_CONCURRENCY)

        async def read(url: str) -> Optional[TurnEvents]:
            async with gate:
                try:
                    return await asyncio.to_thread(lambda: turn_events(self._read_trajectory_text(url)))
                except Exception as e:
                    logger.warning("Verifier '%s': trajectory %s could not be read (%s: %s)", self.verifier_id, url,
                                   type(e).__name__, e)
                    return None

        return dict(zip(urls, await asyncio.gather(*(read(u) for u in urls))))

    async def _write_files(self, sandbox: Any, files: dict[str, str], container: Optional[str]) -> None:
        gate = asyncio.Semaphore(STAGING_CONCURRENCY)

        async def write(path: str, text: str) -> None:
            async with gate:
                await sandbox.write_file_from_text(text, path, **into_container(container))

        await asyncio.gather(*(write(path, text) for path, text in files.items()))

    def _log_staging(self, staged: StagedRun) -> None:
        r = staged.record
        logger.info("Verifier '%s': staged %d wakes as %d files; %d trajectories (%s) of %d bytes staged as %d",
                    self.verifier_id, r["wakes"], r["files"], r["trajectories"], r["formats"], r["raw_bytes"],
                    r["staged_bytes"])
        if r.get("not_compacted"):
            logger.error("Verifier '%s': compaction removed nothing from %d trajectories (unknown format, or nothing "
                         "to drop), first in wake %s; the rest are in trajectory_staging.not_compacted",
                         self.verifier_id, len(r["not_compacted"]), r["not_compacted"][0]["wake"])
        for key in ("missing_trajectories", "unreadable_trajectories"):
            if r.get(key):
                logger.warning("Verifier '%s': %s in wakes %s", self.verifier_id, key.replace("_", " "), r[key])

    async def _judge_reads(self, entry: dict) -> list[dict]:
        """``judge_reads`` of each judge call's trajectory (each batch's, and each timed-out attempt's)."""
        calls = [{"batch": i, "url": url} for i, url in enumerate(
            entry.get("judge_batch_trajectory_object_urls") or [entry.get("judge_trajectory_object_url")], 1)]
        calls += [{"batch": d.get("batch", 1), "timed_out": True, "url": d["judge_trajectory_object_url"]}
                  for d in entry.get("judge_output_discrepancies") or []
                  if d.get("kind") == "judge_timeout" and d.get("judge_trajectory_object_url")]
        out = []
        for call in calls:
            if not call["url"]:
                out.append({**call, "unreadable": "no trajectory was kept"})
                continue
            try:
                events = await asyncio.to_thread(lambda u=call["url"]: turn_events(self._read_trajectory_text(u)))
                out.append({**call, **judge_reads(events)})
            except Exception as e:
                out.append({**call, "unreadable": f"{type(e).__name__}: {str(e)[:200]}"})
        return out


def _builtin_tool_names(card: dict) -> set[str]:
    """The built-in tool names a judge's card lets a caller select: the names its agent-config schema lists for
    ``builtin_tools``."""
    from agent_env.a2a_agent import A2AAgent

    ext = A2AAgent.find_extension(card or {}, A2AAgent.EXT_AGENT_CONFIG) or {}
    schema = ((((ext.get("params") or {}).get("methods") or {}).get("set") or {}).get("request") or {}).get("schema") or {}
    defs = schema.get("$defs") or {}
    names: set[str] = set()
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/") and ref not in seen:
                seen.add(ref)
                walk(defs.get(ref.rsplit("/", 1)[-1]))
            names.update(v for v in node.get("enum") or [] if isinstance(v, str))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk((schema.get("properties") or {}).get("builtin_tools"))
    return names


def _files_note(path: str, staged: StagedRun, *, context_loss: bool = False) -> str:
    r = staged.record
    wakes = sorted(name for name in staged.files if name.startswith(f"{WAKES_DIR}/wake_"))
    gaps = sorted(set(r.get("missing_trajectories", [])) | set(r.get("unreadable_trajectories", [])))
    gap_note = ("\n- These wakes' tool calls are not in the files (their trajectory is missing or unreadable), so "
                "for them only the run's record is there: " + ", ".join(str(n) for n in gaps) + ".") if gaps else ""
    return _FILES_NOTE.format(index=f"{path}/{INDEX_FILE}", wakes=f"{path}/{WAKES_DIR}", first=f"{path}/{wakes[0] if wakes else WAKES_DIR}",
                              last=f"{path}/{wakes[-1] if wakes else WAKES_DIR}", results=f"{path}/{RESULTS_DIR}",
                              events=f"{path}/{EVENTS_FILE}", triggers=f"{path}/{TRIGGERS_FILE}", gaps=gap_note,
                              harness=(_HARNESS_NOTE + (_CONTEXT_LOSS_NOTE if context_loss else ""))
                              if r.get("agent_labeled_trajectories") else "")


def _with_context_loss(spec: JudgeOutputFormatSpec) -> JudgeOutputFormatSpec:
    """``spec`` whose per-criterion rows also take the optional ``CONTEXT_LOSS_FIELDS`` (rubric_binary's and
    rubric_partial's rows keep every field the judge returns, so nothing else changes)."""
    output_format = copy.deepcopy(spec.output_format)
    output_format["schema"]["properties"]["results"]["items"]["properties"].update(CONTEXT_LOSS_FIELDS)
    return dataclasses.replace(spec, output_format=output_format)


def context_loss(results: list[dict]) -> dict:
    """The run's count of failed criteria and of those the judge traced to a compaction, with their ids. A passed
    criterion's context-loss fields are dropped (they mean nothing there); a failed one's are kept, ``context_loss``
    as a bool; a failed one the judge said nothing about is counted in ``not_said``."""
    failed, caused, not_said = 0, [], 0
    for row in results:
        if row.get("result"):
            row.pop("context_loss", None)
            row.pop("context_loss_evidence", None)
            continue
        failed += 1
        if "context_loss" not in row:
            not_said += 1
            continue
        row["context_loss"] = row["context_loss"] is True or str(row["context_loss"]).lower() == "true"
        if row["context_loss"]:
            caused.append(row.get("id"))
    out = {"failed_criteria": failed, "caused_by_compaction": len(caused), "ids": caused}
    if not_said:
        out["not_said"] = not_said
    return out


def harness_settings(raw: str) -> dict:
    """The settings an assistant's harness reports in its trace: the PSI neutral harness's ``harness.config.*``
    attributes on its ``harness.usage`` span, without the prefix. Nothing for a trace with none (another harness,
    another format)."""
    prefix = "harness.config."
    try:
        spans = json.loads(raw)
    except ValueError:
        return {}
    for span in spans if isinstance(spans, list) else []:
        attrs = span.get("attributes") if isinstance(span, dict) else None
        found = {k[len(prefix):]: v for k, v in attrs.items() if k.startswith(prefix)} if isinstance(attrs, dict) else {}
        if found:
            return found
    return {}
