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
"""

from __future__ import annotations

import asyncio
import logging
import uuid
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
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import TurnEvents, turn_events
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    RUN_CONTEXT_PARTS,
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
("tool_call", numbered "call") and its own text ("assistant_text", which nobody saw); then the rest of the wake in time \
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
- The user's messages are the "user_message" lines (and the first wake's "wake_input"); the user saw only the \
"message_to_user" lines.
- Cite the wake (and call) number of every piece of evidence, and quote it.
- Only the assistant's tool calls and its "agent" app changes are its acts. A draft is not a sent message. Messages \
from the user, the user simulator or contacts are never the assistant's. Order events by their pretend time ("at"), \
not by where they sit in the files.
- You have Bash and Read only: no web, no other agents, and nothing to change.{gaps}"""


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

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        if "grade_up_to_turn" in self.step_param_overrides(context):
            raise RuntimeError(f"Verifier '{self.verifier_id}': grade_up_to_turn can't cut an always_on step's "
                               "prompt, whose conversation turns are not its wakes.")
        if self.use_trajectory and (context.metadata.get("user_overrides") or {}).get("use_agent_judge") is False:
            raise RuntimeError(f"Verifier '{self.verifier_id}': use_trajectory needs the agent judge; this run's "
                               "use_agent_judge=False would grade without the wakes it was set to read.")
        return await super().execute(context)

    async def _grade(self, context: TaskStepContext) -> TaskStepContext:
        """rubrics_verifier's grade; then, for a staged run, what each judge call read of it (``judge_reads``)."""
        context = await super()._grade(context)
        entry = (context.metadata.get("verifications") or {}).get(self.verifier_id) or {}
        if "trajectory_staging" in entry:
            entry["trajectory_staging"]["judge_reads"] = await self._judge_reads(entry)
        return context

    def judge_agent_config(self) -> dict:
        return {"builtin_tools": {"only": list(JUDGE_TOOLS)}, "peer_tools": False}

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
        if not record.get("object_url"):
            logger.warning("Verifier '%s': the always_on run kept no stored timeline; its wakes' trajectories are "
                           "loaded by position instead", self.verifier_id)
            return None
        try:
            timeline = await asyncio.to_thread(stored_timeline, record)
        except Exception as e:
            logger.warning("Verifier '%s': the always_on run's timeline could not be read (%s: %s); its wakes' "
                           "trajectories are loaded by position instead", self.verifier_id, type(e).__name__, e)
            return None
        urls = list(dict.fromkeys(e["trajectory_url"] for e in timeline
                                  if e.get("event") in ("agent_turn_end", "nudge") and e.get("trajectory_url")))
        turns = await self._read_turns(urls)
        run = run_context or await load_run_context(context, prompt_response)
        staged = await asyncio.to_thread(stage_run, timeline, turns, firings=run.firings_read(),
                                         unavailable=run.unavailable)
        path = f"{STAGING_DIR}/{uuid.uuid4().hex[:12]}"
        await self._write_files(sandbox, {f"{path}/{name}": text for name, text in staged.files.items()}, container)
        self._log_staging(staged)
        return StagedTrajectory(path=path, prompt_note=_files_note(path, staged), record=staged.record)

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


def _files_note(path: str, staged: StagedRun) -> str:
    r = staged.record
    wakes = sorted(name for name in staged.files if name.startswith(f"{WAKES_DIR}/wake_"))
    gaps = sorted(set(r.get("missing_trajectories", [])) | set(r.get("unreadable_trajectories", [])))
    gap_note = ("\n- These wakes' tool calls are not in the files (their trajectory is missing or unreadable), so "
                "for them only the run's record is there: " + ", ".join(str(n) for n in gaps) + ".") if gaps else ""
    return _FILES_NOTE.format(index=f"{path}/{INDEX_FILE}", first=f"{path}/{wakes[0] if wakes else WAKES_DIR}",
                              last=f"{path}/{wakes[-1] if wakes else WAKES_DIR}", results=f"{path}/{RESULTS_DIR}",
                              events=f"{path}/{EVENTS_FILE}", triggers=f"{path}/{TRIGGERS_FILE}", gaps=gap_note)
