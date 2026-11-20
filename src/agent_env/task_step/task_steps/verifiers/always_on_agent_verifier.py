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

A run that kept no timeline has its turns staged the same way by position, one wake file a turn, so the judge never
reads a trajectory as stored.

For a run of the PSI neutral harness (its trace carries ``harness.*`` labels), the judge is told whose a compaction
summary's losses are, by who wrote it (``harness.compaction.written_by``: the assistant's own model, graded as its own,
or the harness, not held against the assistant), and, with a per-criterion format, says for each failed criterion
whether it traces to context a compaction lost (``context_loss`` on the row; the entry counts them): reported next to
the score, never changing it. A trajectory whose compaction doesn't say who wrote its summary leaves the run ungraded
(``UngradableRunError``), never misread. Every entry also says what the run records of the assistant's harness
(``assistant_harness``: its agent name, its card's name and version and, when its wakes are staged, the harness version
and settings its trace reports).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import uuid
from contextvars import ContextVar
from typing import Any, ClassVar, Optional

from agent_env.providers.sandbox_providers.sandbox import into_container
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.always_on import ALWAYS_ON_KEY, AlwaysOnTaskStep, render_for_judge, stored_timeline
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
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
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import (
    TrajectoryFormatError,
    TurnEvents,
    turn_events,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    RUN_CONTEXT_PARTS,
    RubricsVerdict,
    RubricsVerifierTaskStep,
    StagedTrajectory,
)
from agent_env.task_step.task_steps.verifiers.run_context import Run, load_run_context

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
- You have Bash and Read only: no web, no other agents, and nothing to change.{by_position}{gaps}{harness}"""
# Added to _FILES_NOTE for a run staged by position (it kept no timeline).
_BY_POSITION_NOTE = """
- This run kept no timeline, so each wake file is one turn of the assistant by its position, in the order the turns \
ran (a nudge's turn is one of its own), not a wake of the run. It has the text the turn was sent ("wake_input") and \
the turn's own events, and no times, app changes, messages to the user or reads of them: for those, use the run wake \
by wake and the user's chat in this prompt."""
# Added to _FILES_NOTE for a run whose trajectories carry the PSI harness's labels (compaction, memory, helpers), where
# these lines appear; a run without them gets none of it.
_HARNESS_NOTE = """
- "harness_note" lines are from the harness that runs the assistant, never from the assistant or the user: a message \
the harness sent the assistant, or a compaction summary that replaced older history in what the assistant sees.
- Lines with "step": "pre_compaction_save" are a turn the harness asked for before a compaction. What the assistant \
wrote there went to the harness, not the user, and it ran before the assistant saw that wake's message.
- Tool calls marked "internal" are tools the harness itself provides (notes, look-ups of past wakes). They count as \
what the assistant did, but they change nothing in any app.
- Lines with "subagent" are from a helper agent the assistant started.
- Harness notes have no "at": read them in file order."""
# Added after _HARNESS_NOTE for a run with a compaction, by who wrote its summary (its note's "written_by", in
# "compaction"): whose a loss the summary caused is.
_COMPACTION_NOTES = {
    "assistant": """
- A compaction summary whose note says "written_by": "assistant" is the assistant's own summary of its history: its \
own model wrote it. If the summary lost or got wrong something the assistant needed, the assistant is responsible \
for what follows, the same as for any other mistake. Grade it normally.""",
    "harness": """
- A compaction summary whose note says "written_by": "harness" was written by the harness, not the assistant. If it \
lost or got wrong something the assistant needed, that loss is not the assistant's fault: don't hold it against the \
assistant, grade what it did with what it still had, and name the loss in your reason (the compaction note's \
file:line and the earlier wake that held the fact).""",
}
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


class UngradableRunError(RuntimeError):
    """A run whose trajectories break the labels they carry: it is left without a score, to be graded again once they
    can be read, never graded from a misreading."""


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

    def criterion_rows(self) -> Optional[int]:
        """Its rows follow the run's timeline, not its criteria alone."""
        return None

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
        **fields: Any,
    ):
        """``run_context_for_judge`` defaults to every part of the run's context; the rest is rubrics_verifier's."""
        output_format = JudgeOutputFormat(output_format)
        where = f"{self.type} '{id}'"
        if use_trajectory and not use_agent_judge:
            raise ValueError(f"{where}: use_trajectory needs the agent judge (use_agent_judge): an always_on run's "
                             "wakes are staged as files for a judge that searches them, never inlined")
        if get_judge_output_format_spec(output_format).cites_evidence:
            raise ValueError(f"{where}: output_format={output_format.value} cites per-criterion screenshots, and an "
                             "always_on step's prompt is graded without its wakes' screenshots")
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

    async def _judged(self, context: TaskStepContext) -> RubricsVerdict:
        """rubrics_verifier's verdict, refused for a run whose overrides cut the prompt or take the wakes from its judge;
        then, for a staged run, what each judge call read of it (``judge_reads``); for a run with agent labels (the PSI
        harness's), how many failed criteria the judge traced to a compaction (``context_loss``); and for every run,
        the assistant's harness as the run records it (``assistant_harness``)."""
        if "grade_up_to_turn" in self.step_param_overrides(context):
            raise RuntimeError(f"Verifier '{self.verifier_id}': grade_up_to_turn can't cut an always_on step's "
                               "prompt, whose conversation turns are not its wakes.")
        if self.use_trajectory and (context.metadata.get("user_overrides") or {}).get("use_agent_judge") is False:
            raise RuntimeError(f"Verifier '{self.verifier_id}': use_trajectory needs the agent judge; this run's "
                               "use_agent_judge=False would grade without the wakes it was set to read.")
        token = _CONTEXT_LOSS_RUN.set(False)
        try:
            verdict = await super()._judged(context)
            asked = _CONTEXT_LOSS_RUN.get()
        finally:
            _CONTEXT_LOSS_RUN.reset(token)
        output = dict(verdict.output)
        if "trajectory_staging" in output:
            output["trajectory_staging"]["judge_reads"] = await self._judge_reads(output)
        if asked:
            output["context_loss"] = context_loss(verdict.criteria)
        if harness := await self._assistant_harness(context, staged="trajectory_staging" in output):
            output["assistant_harness"] = harness
        return dataclasses.replace(verdict, output=output)

    def _spec(self) -> JudgeOutputFormatSpec:
        """rubrics_verifier's format; for a run with agent labels (``_CONTEXT_LOSS_RUN``), its rows also take ``context_loss`` and
        ``context_loss_evidence``."""
        spec = super()._spec()
        return _with_context_loss(spec) if _CONTEXT_LOSS_RUN.get() else spec

    async def _assistant_harness(self, context: TaskStepContext, *, staged: bool) -> dict:
        """The graded assistant's harness as the run records it: its agent name, its A2A card's name and version, and,
        for a run whose wakes were ``staged`` (so its trajectories are read anyway), the harness version and settings
        its trace reports (``harness_usage`` of the last wake's trajectory, read from ``settings_from``). A run graded
        without its trajectories has none read. What isn't there is left out."""
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
                reported = harness_usage(await asyncio.to_thread(self._read_trajectory_text, url))
            except Exception as e:
                logger.warning("Verifier '%s': the assistant's harness settings could not be read from %s (%s: %s)",
                               self.verifier_id, url, type(e).__name__, e)
                reported = {}
            if reported:
                out.update(reported)
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
        """The run as files (``stage_run``) in the judge's sandbox, each trajectory matched to its wake by the run's
        stored timeline. Without that timeline (not kept, or unreadable), the turns are staged the same way by
        position (``turns_by_position``). A trajectory that breaks the harness labels it carries raises
        ``UngradableRunError``."""
        record = self._record(context, prompt_response)
        timeline: Optional[list[dict]] = None
        if not record.get("object_url"):
            logger.warning("Verifier '%s': the always_on run kept no stored timeline; its turns are staged by position "
                           "instead", self.verifier_id)
        else:
            try:
                timeline = await asyncio.to_thread(stored_timeline, record)
            except Exception as e:
                logger.warning("Verifier '%s': the always_on run's timeline could not be read (%s: %s); its turns are "
                               "staged by position instead", self.verifier_id, type(e).__name__, e)
        by_position = timeline is None
        if timeline is None:
            timeline = turns_by_position(prompt_response)
        urls = list(dict.fromkeys(e["trajectory_url"] for e in timeline
                                  if e.get("event") in ("agent_turn_end", "nudge") and e.get("trajectory_url")))
        turns = await self._read_turns(urls, _turn_names(timeline, by_position=by_position))
        run = run_context or await load_run_context(context, prompt_response)
        staged = await asyncio.to_thread(stage_run, timeline, turns, firings=run.firings_read(),
                                         unavailable=run.unavailable, by_position=by_position)
        _CONTEXT_LOSS_RUN.set(bool(staged.record.get("agent_labeled_trajectories"))
                              and self.output_format in _CONTEXT_LOSS_FORMATS)
        path = f"{STAGING_DIR}/{uuid.uuid4().hex[:12]}"
        await self._write_files(sandbox, {f"{path}/{name}": text for name, text in staged.files.items()}, container)
        self._log_staging(staged)
        return StagedTrajectory(path=path, prompt_note=_files_note(path, staged, context_loss=_CONTEXT_LOSS_RUN.get()),
                                record=staged.record)

    async def _read_turns(self, urls: list[str], names: dict[str, str]) -> dict[str, Optional[TurnEvents]]:
        """Each trajectory's events, by url; None for one that couldn't be read. One that breaks the harness labels it
        carries raises ``UngradableRunError`` naming its wake (``names``)."""
        gate = asyncio.Semaphore(STAGING_CONCURRENCY)

        async def read(url: str) -> Optional[TurnEvents]:
            async with gate:
                try:
                    return await asyncio.to_thread(lambda: turn_events(self._read_trajectory_text(url)))
                except TrajectoryFormatError as e:
                    raise UngradableRunError(
                        f"Verifier '{self.verifier_id}': {names.get(url, 'a turn')}'s trajectory ({url}) can't be "
                        f"read: {e}. The run is left ungraded, to be graded again once its trajectory is fixed."
                    ) from e
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
    writers = r.get("compaction_summaries_written_by") or {}
    harness = (_HARNESS_NOTE + "".join(note for writer, note in _COMPACTION_NOTES.items() if writer in writers)
               + (_CONTEXT_LOSS_NOTE if context_loss else "")) if r.get("agent_labeled_trajectories") else ""
    return _FILES_NOTE.format(index=f"{path}/{INDEX_FILE}", wakes=f"{path}/{WAKES_DIR}", first=f"{path}/{wakes[0] if wakes else WAKES_DIR}",
                              last=f"{path}/{wakes[-1] if wakes else WAKES_DIR}", results=f"{path}/{RESULTS_DIR}",
                              events=f"{path}/{EVENTS_FILE}", triggers=f"{path}/{TRIGGERS_FILE}",
                              by_position=_BY_POSITION_NOTE if r.get("turns_by_position") else "", gaps=gap_note,
                              harness=harness)


def turns_by_position(prompt_response: PromptResponse) -> list[dict]:
    """A timeline for a run that kept none: one wake a turn, in the order the turns ran (a nudge's turn is one of its
    own), with its trajectory and the text it was sent (the prompt response keeps each turn's, cut)."""
    sent = prompt_response.source_agent_per_turn_prompt_parts or []
    timeline: list[dict] = []
    for n, url in enumerate(prompt_response.target_agent_per_turn_trajectory_object_urls or [], 1):
        parts = (sent[n - 1] if n <= len(sent) else None) or []
        text = "\n".join(p.get("text") or "" for p in parts if isinstance(p, dict) and p.get("kind") == "text")
        timeline += [{"event": "wake", "n": n, "parts_text": text},
                     {"event": "agent_turn_end", "n": n, "trajectory_url": url}]
    return timeline


def _turn_names(timeline: list[dict], *, by_position: bool) -> dict[str, str]:
    """Each trajectory's turn in words, by its url: "wake 4", "the nudge after wake 2", or by position "turn 3"."""
    names: dict[str, str] = {}
    for e in timeline:
        url = e.get("trajectory_url")
        if not url or url in names:
            continue
        if by_position:
            names[url] = f"turn {e.get('n')}"
        elif e.get("event") == "nudge":
            names[url] = f"the nudge after wake {e.get('wake')}"
        elif e.get("event") == "agent_turn_end":
            names[url] = f"wake {e.get('n')}"
    return names


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


def harness_usage(raw: str) -> dict:
    """What an assistant's harness reports of itself in its trace, on the PSI neutral harness's ``harness.usage`` span:
    its ``version`` (``harness.version``) and ``settings`` (its ``harness.config.*`` attributes, without the prefix).
    Nothing for a trace with no such span (another harness, another format)."""
    prefix = "harness.config."
    try:
        spans = json.loads(raw)
    except ValueError:
        return {}
    for span in spans if isinstance(spans, list) else []:
        attrs = span.get("attributes") if isinstance(span, dict) and span.get("name") == "harness.usage" else None
        if not isinstance(attrs, dict):
            continue
        out: dict = {}
        if attrs.get("harness.version") is not None:
            out["version"] = attrs["harness.version"]
        if settings := {k[len(prefix):]: v for k, v in attrs.items() if k.startswith(prefix)}:
            out["settings"] = settings
        return out
    return {}
