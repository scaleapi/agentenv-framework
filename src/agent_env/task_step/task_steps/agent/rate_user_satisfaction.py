"""The rate_user_satisfaction step: once an always_on or prompt_agent run is over, the user sim that played its user
rates the job 1-5 with a one-line reason, by this step's ``satisfaction`` lines (user_sim.md, "The rating at the end").

The lines are this step's, never the persona's, so the user sim's system prompt never has them and the run behaves as
it does without a rating. The step runs after the run step, as a step of its own, so the rating takes no time from
the run's budget, and it changes nothing in the run step.

Which run: ``run_step_id``, else the task's only always_on or prompt_agent step; the task is refused when it loads if
that step doesn't exist, isn't one of the two, or isn't run with a user sim a deploy_user_sim_agent step deploys, when
the task has several and no ``run_step_id`` picks one, when this step doesn't come after it (``depends_on``, when set,
must name it), or when two rating steps rate the same run. ``satisfaction`` is checked as the persona's sections are:
a list of non-empty lines.

The ask is one message to the user sim, in the session the run used, so it remembers the run:

- always_on: the session its run record names (``metadata["always_on"][run step id]["user_sim"]["context_id"]``),
  the one every read of the run used. The messages the user never read are the record's ``unread_by_user`` at the end
  (from the stored timeline's end event, else the record's own bounded copy), and the ask is at the time of her last
  read, or the run's end when that's later.
- prompt_agent: the run's conversation id (``metadata["a2a_conversations"][run step id]``), the session its built-in
  user sim's reads use (``_UserSide._ask_built_in``). The messages the user never read are the agent's replies after
  her last read (``metadata["user_sim_reads"]``), from the run's conversation, and the ask is at the task clock's time.

The message (``user_sim_versions.render_rating``) has those unread messages, the ``satisfaction`` lines under "HOW
YOU'D RATE THE JOB" tagged S1..., and the 1-5 ask; a malformed answer gets one retry (``user_sim_versions.rate``). The
rating is recorded at ``metadata["user_sim_satisfaction"][run step id]``: ``user_sim``, ``score`` (None when it
failed), ``reason``, ``used``, ``at``, ``unread_shown``, ``schema_errors``, ``retried``, ``user_sim_error``,
``user_sim_turns``, ``user_sim_trajectory_urls``, ``rated_by`` (this step's id), ``context_id``, ``session_from``
and ``tagged_lines`` (``{"S1": line, ...}``, the lines the ask tagged, so ``used`` resolves; a persona tag resolves by
the deploy record's ``tagged_lines``), the whole ask and raw answers in the object store at its ``object_url``. It never reaches the
agent, the graded conversation, or the verifications.

Not rated, recorded with ``skipped`` and why: a run that ended on an error (always_on's error end reasons; a
prompt_agent run whose agent turn or user sim failed; a run step that failed), and a run whose user sim agent is no
longer deployed. A rating that fails or runs past ``timeout_seconds`` (by default one always_on read's worst case) is
recorded with its error; the step fails only when it can't find the run or its user, and by default
(``fail_task_on_error`` false) that doesn't fail the task.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Optional

from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent, conversation_store, protocol
from agent_env.a2a_agent.object_transfer import send_and_wait
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.store import get_config
from agent_env.task_step.context import DeployedAgent, PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY, DeployUserSimAgentTaskStep
from agent_env.task_step.task_steps.agent.user_sim_common import SATISFACTION_KEY, iso, person_of, task_zone, utc
from agent_env.task_step.task_steps.always_on import (
    ALWAYS_ON_KEY,
    ERROR_ENDS,
    READ_ASKS,
    TRAJECTORY_FETCH_SECONDS,
    TURN_GRACE_SECONDS,
    AlwaysOnTaskStep,
)
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep, is_prompt_step

logger = logging.getLogger(__name__)

# One always_on read's worst case at its default user sim timeout (AlwaysOnTaskStep.read_cap_seconds).
USER_TIMEOUT_DEFAULT_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = READ_ASKS * (USER_TIMEOUT_DEFAULT_SECONDS + TURN_GRACE_SECONDS + TRAJECTORY_FETCH_SECONDS)
SIM_FAILED_PREFIX = "The user simulator failed: "  # prompt_agent's error_message for a user sim error


@dataclass
class _Session:
    """What the step needs of the run: whose user sim, the session, what she never read, and when she's asked."""

    session_from: str
    user_sim: Optional[str] = None
    context_id: Optional[str] = None
    time_zone: Optional[str] = None
    unread: list[dict] = field(default_factory=list)
    at: Optional[str] = None
    trajectory_prefix: Optional[str] = None
    skipped: Optional[str] = None


class RateUserSatisfactionTaskStep(TaskStep):
    """Once the run step is over, its user sim rates the job 1-5 by ``satisfaction``, recorded in
    ``metadata["user_sim_satisfaction"][run step id]`` and sent nowhere."""

    type: ClassVar[str] = "rate_user_satisfaction"
    entity_refs = ()

    def __init__(self, id: str, version: Optional[int], *, satisfaction: list[str], run_step_id: Optional[str] = None,
                 timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 poll_interval_seconds: float = PromptAgentTaskStep.DEFAULT_POLL_INTERVAL_SECONDS,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = False):
        """satisfaction: how the user would rate the job, one line each (what 5, 3 and 1 mean for it, what moves the
        score). run_step_id: the always_on or prompt_agent step to rate; unset, the task's only one. timeout_seconds:
        the longest the rating may take, both asks included."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        found = user_sim_versions.satisfaction_problems(satisfaction)
        if run_step_id is not None and (not isinstance(run_step_id, str) or not run_step_id.strip()):
            found.append("run_step_id must name the always_on or prompt_agent step to rate")
        for name, value in (("timeout_seconds", timeout_seconds), ("poll_interval_seconds", poll_interval_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (
                    name == "timeout_seconds" and value == 0):
                found.append(f"{name} must be a number of seconds" + (", more than 0" if name == "timeout_seconds"
                                                                       else ", 0 or more"))
        if found:
            raise ValueError(f"{self.type} '{id}': " + "; ".join(found))
        self.satisfaction = list(satisfaction)
        self.run_step_id = run_step_id
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    _OWN = ("satisfaction", "run_step_id", "timeout_seconds", "poll_interval_seconds")

    def to_dict(self) -> dict:
        return {**super().to_dict(), **{k: getattr(self, k) for k in self._OWN}}

    @classmethod
    def from_dict(cls, data: dict) -> RateUserSatisfactionTaskStep:
        base = {**cls._base_from_dict(data), "fail_task_on_error": data.get("fail_task_on_error", False)}
        return cls(**base, **{k: data[k] for k in cls._OWN if k in data})

    # ---- the load check ---------------------------------------------------------------------------------------------
    def _run_step(self, steps: list[TaskStep]) -> tuple[Optional[PromptAgentTaskStep], list[str]]:
        """The step this one rates, or why there's none."""
        where = f"{self.type} '{self.id}'"
        if self.run_step_id is not None:
            run = next((s for s in steps if s.id == self.run_step_id), None)
            if run is None:
                return None, [f"{where}: run_step_id '{self.run_step_id}' names no step of the task"]
            if not isinstance(run, PromptAgentTaskStep):
                return None, [f"{where}: run_step_id '{self.run_step_id}' is a {run.type} step; name the always_on or "
                              "prompt_agent step to rate"]
            return run, []
        runs = [s for s in steps if isinstance(s, PromptAgentTaskStep)]
        if not runs:
            return None, [f"{where}: the task has no always_on or prompt_agent step to rate"]
        if len(runs) > 1:
            return None, [f"{where}: the task has several always_on and prompt_agent steps "
                          f"({', '.join(s.id for s in runs)}); set run_step_id to the one to rate"]
        return runs[0], []

    def task_problems(self, steps: list[TaskStep]) -> list[str]:
        run, found = self._run_step(steps)
        if run is None:
            return found
        where = f"{self.type} '{self.id}'"
        if steps.index(run) > steps.index(self):
            found.append(f"{where} comes before step '{run.id}', the run it rates; put it after")
        if self.depends_on is not None and run.id not in {d.task_step_id for d in self.depends_on}:
            found.append(f"{where}: its depends_on must name '{run.id}', the run it rates, so it runs once that's over")
        user = _user_of(run)
        if user is None or not any(isinstance(s, DeployUserSimAgentTaskStep) and s.agent_name == user for s in steps):
            found.append(f"{where}: the user of {run.type} step '{run.id}' isn't a user sim a deploy_user_sim_agent "
                         "step deploys; only that user sim can rate the run")
        earlier = next((s for s in steps[:steps.index(self)] if isinstance(s, RateUserSatisfactionTaskStep)
                        and s._run_step(steps)[0] is run), None)
        if earlier is not None:
            found.append(f"{where}: step '{earlier.id}' already rates '{run.id}'; a run is rated once")
        return found

    # ---- the run ----------------------------------------------------------------------------------------------------
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        run_id = self.run_step_id or _only_run(context, self.id)
        session = await self._session(context, run_id)
        us = (context.metadata.get(USER_SIMS_KEY) or {}).get(session.user_sim) if session.user_sim else None
        summary: dict = {"user_sim": session.user_sim, "score": None, "reason": None, "rated_by": self.id}
        context.metadata.setdefault(SATISFACTION_KEY, {})[run_id] = summary
        if session.skipped is None and us is None:
            raise StepContextError(f"{self.type} '{self.id}': step '{run_id}' ran with no user sim a "
                                   "deploy_user_sim_agent step recorded, so there's no one to ask")
        agent = next((a for a in context.deployed_agents if a.agent_name == us.get("agent_name")), None) if us else None
        if session.skipped is None and agent is None:
            session.skipped = f"the user sim's agent '{us.get('agent_name')}' is no longer deployed"
        if session.skipped is not None:
            summary["skipped"] = session.skipped
            logger.info(f"{self.type} '{self.id}': '{run_id}' not rated: {session.skipped}")
            return context
        if not session.context_id:
            raise StepContextError(f"{self.type} '{self.id}': step '{run_id}' recorded no user sim session to ask in")

        person = {**person_of(us), "time_zone": session.time_zone}
        read: dict = {"time": session.at or await _clock_now(context),
                      "assistant_messages": [{"at": m.get("at"), "text": m["text"]} for m in session.unread]}
        calls: list[dict] = []
        deadline = time.monotonic() + self.timeout_seconds

        async def ask(text: str) -> dict:
            got = await self._ask(agent, session, text, deadline)
            calls.append(got)
            return got

        got: dict = {}
        error = None
        try:
            got = await asyncio.wait_for(user_sim_versions.rate(
                us.get("user_version") or user_sim_versions.VERSION, person, self.satisfaction, read, ask),
                self.timeout_seconds) or {}
        except Exception as e:  # TimeoutError included: the rating is recorded with its error
            error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        error = error or (calls[-1].get("error") if calls else None)
        summary.update({k: got.get(k) for k in ("score", "reason", "used", "schema_errors", "retried")},
                       at=read["time"], unread_shown=len(read["assistant_messages"]), user_sim_error=error,
                       user_sim_turns=got.get("turns", len(calls)),
                       user_sim_trajectory_urls=[c.get("trajectory_url") for c in calls],
                       context_id=session.context_id, session_from=session.session_from,
                       tagged_lines=user_sim_versions.rating_tagged_lines(self.satisfaction))
        logger.info(f"{self.type} '{self.id}': '{run_id}' rated {summary['score']} (error={error})")
        full = {**summary, "run_step_id": run_id, "satisfaction": self.satisfaction, "read": read,
                **{k: got.get(k) for k in ("input", "raw", "first_answer")}}
        try:
            config = get_config()
            key = (f"{config.get_artifact_key_prefix()}user_sim_ratings/instance_id={context.instance_id or 'adhoc'}/"
                   f"{run_id}-{self.id}-{uuid.uuid4().hex[:8]}.json")
            summary["object_url"] = await asyncio.to_thread(
                config.get_object_store().put, key, json.dumps(full, indent=1, default=str).encode(),
                content_type="application/json")
        except Exception as e:
            logger.warning(f"{self.type} '{self.id}': could not store the rating: {e}")
            summary["store_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return context

    async def _session(self, context: TaskStepContext, run_id: str) -> _Session:
        """The run's user sim session, what she never read, and whether the run is rated at all."""
        response = next((r for r in reversed(context.prompt_responses) if r.step_id == run_id), None)
        prefix = response.agent_trajectory_object_prefix if response is not None else None
        record = (context.metadata.get(ALWAYS_ON_KEY) or {}).get(run_id)
        if record is not None:
            session = _always_on_session(record)
        elif response is not None:
            session = await _prompt_agent_session(context, run_id, response)
        else:
            failed = _failed(context, run_id)
            if failed is None:
                raise StepContextError(f"{self.type} '{self.id}': step '{run_id}' left no run to rate; run this step "
                                       "after it")
            return _Session("none", skipped=f"the run step failed: {failed}")
        session.trajectory_prefix = prefix + "user_sim/" if prefix else None
        if session.skipped is None and (failed := _failed(context, run_id)) is not None:
            session.skipped = f"the run step failed: {failed}"
        return session

    async def _ask(self, agent: DeployedAgent, session: _Session, text: str, deadline: float) -> dict:
        """One user sim turn on ``text`` in the run's session, cut off at the step's ``deadline`` (time.monotonic()):
        ``{"text", "structured", "trajectory_url"}``, or ``{"text": None, "structured": None, "error"}``."""
        limit = deadline - time.monotonic()
        if limit <= 0:
            return {"text": None, "structured": None, "error": "the step's time limit passed"}
        url = agent.a2a_url or agent.api_url
        message_id = uuid.uuid4().hex
        try:
            task_id, result = await asyncio.wait_for(send_and_wait(
                url, [{"kind": "text", "text": text}], agent=agent if agent.sandbox_id else None,
                message_id=message_id, context_id=session.context_id, timeout_seconds=limit,
                poll_interval_seconds=self.poll_interval_seconds), limit)
        except TimeoutError as e:
            return {"text": None, "structured": None, "error": f"timeout: {e}"}
        status = result.get("status") or {}
        if status.get("state") != TaskState.completed and str(status.get("state")) != "completed":
            return {"text": None, "structured": None, "error": f"state {status.get('state')}"}
        msg = status.get("message") or {}
        term = protocol.TerminalResponse.from_message(msg)
        raw = next((p["text"] for p in msg.get("parts") or [] if p.get("kind") == "text"), term.response_text)
        structured = term.structured_output if isinstance(term.structured_output, dict) else None
        return {"text": raw, "structured": structured,
                "trajectory_url": await self._trajectory(agent, session, task_id, message_id, deadline)}

    async def _trajectory(self, agent: DeployedAgent, session: _Session, task_id: str, message_id: str,
                          deadline: float) -> Optional[str]:
        """The user sim's trajectory for one turn, stored where the run stored its reads'; None when its card offers
        none or the run left no trajectory prefix."""
        card = agent.a2a_card or {}
        ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if not ext or not session.trajectory_prefix:
            return None
        try:
            # prompt_agent's fetch uses nothing of its step, so the run's own code stores it.
            return await asyncio.wait_for(PromptAgentTaskStep._fetch_trajectory(
                None, agent.a2a_url or agent.api_url, ext, task_id, session.trajectory_prefix, message_id,
                card=card, sandbox_type=agent.sandbox_type),
                max(0.0, min(TRAJECTORY_FETCH_SECONDS, deadline - time.monotonic())))
        except Exception as e:
            logger.warning(f"user sim trajectory fetch failed (continuing): {e}")
            return None


# ---- the run, as the run step recorded it ----------------------------------------------------------------------------
def _user_of(run: PromptAgentTaskStep) -> Optional[str]:
    """The agent a run step talks to as its user: always_on's ``user_sim``; prompt_agent's ``user_agent_name``, unless
    a ``user_a2a_url`` puts someone else there."""
    if isinstance(run, AlwaysOnTaskStep):
        return run.user_sim
    return run.user_agent_name if run.user_a2a_url is None else None


def _only_run(context: TaskStepContext, step_id: str) -> str:
    """With no ``run_step_id``: the task's one run step (the load check made sure there's one), by what it left."""
    ids = list(dict.fromkeys([r.step_id for r in context.prompt_responses if r.step_id]
                             + [f["step_id"] for f in context.metadata.get("failed_steps") or []
                                if isinstance(f, dict) and is_prompt_step({"type": f.get("step_type")})]))
    if len(ids) != 1:
        raise StepContextError(f"{RateUserSatisfactionTaskStep.type} '{step_id}': "
                               + (f"several runs ran ({', '.join(ids)}); set run_step_id" if ids
                                  else "no always_on or prompt_agent step ran before it"))
    return ids[0]


def _failed(context: TaskStepContext, run_id: str) -> Optional[str]:
    """The run step's error, when it failed and the task went on; a failure a retry recovered from is no failure."""
    return next((f.get("error") or f.get("error_type") or "an error" for f in context.metadata.get("failed_steps") or []
                 if isinstance(f, dict) and f.get("step_id") == run_id and not f.get("retried")), None)


def _always_on_session(record: dict) -> _Session:
    """From always_on's record: its user sim session, what she never read, and her last read's time or the end's."""
    sim = record.get("user_sim") or {}
    session = _Session("always_on record", user_sim=sim.get("name"), context_id=sim.get("context_id"),
                       time_zone=record.get("time_zone"))
    if record.get("end_reason") in ERROR_ENDS:
        session.skipped = f"the run ended on {record['end_reason']}"
        return session
    timeline = record.get("timeline_tail") or []
    if record.get("object_url"):
        try:
            url = record["object_url"]
            timeline = json.loads(get_config().get_object_store_at(url).get(url))["timeline"]
        except Exception as e:  # the record's own copy is bounded, but has them
            logger.warning(f"rate_user_satisfaction: could not read always_on's timeline: {e}")
    end = next((e for e in reversed(timeline) if e.get("event") == "end"), {})
    unread = ((end.get("pending") or {}).get("unread_by_user") if isinstance(end.get("pending"), dict) else None)
    if not isinstance(unread, list):
        unread = ((record.get("pending_at_end") or {}).get("unread_by_user") or {}).get("items") or []
    session.unread = [m for m in unread if isinstance(m, dict) and isinstance(m.get("text"), str)]
    times = [utc(e["virtual_time"]) for e in timeline
             if e.get("event") in ("user_sim_read", "end") and e.get("virtual_time")]
    session.at = iso(max(times)) if times else None
    return session


async def _prompt_agent_session(context: TaskStepContext, run_id: str, response: PromptResponse) -> _Session:
    """From prompt_agent's run: its conversation, the session its built-in user sim's reads used; the agent's replies
    after her last read; the task clock's time."""
    conversation_id = (context.metadata.get("a2a_conversations") or {}).get(run_id)
    doc = conversation_store.get_conversation(conversation_id) if conversation_id else None
    session = _Session("prompt_agent conversation", user_sim=(doc or {}).get("source_agent_name"),
                       context_id=conversation_id)
    if response.error_type == "user_sim_error":
        session.skipped = f"the user sim failed: {(response.error_message or '').removeprefix(SIM_FAILED_PREFIX)}"
        return session
    if response.error_type or response.error_message:
        session.skipped = "the agent's turn failed"
        return session
    us = (context.metadata.get(USER_SIMS_KEY) or {}).get(session.user_sim) if session.user_sim else None
    if us is not None:
        session.time_zone = task_zone(us, context.metadata.get("clock_configurations") or [],
                                      f"{RateUserSatisfactionTaskStep.type} on '{run_id}'")
    session.at = await _clock_now(context)
    last_read = max((r.get("turn") or 0 for r in (context.metadata.get("user_sim_reads") or {}).get(run_id) or []),
                    default=0)
    replies = [m for m in (doc or {}).get("messages") or [] if m.get("role") == "agent"]
    for m in replies[last_read:]:
        text = "\n".join(p.get("text", "") for p in m.get("parts") or [] if p.get("kind") == "text")
        clocks = [utc(c["virtual_time"]) for c in m.get("env_clock") or [] if c.get("virtual_time")]
        session.unread.append({"at": iso(min(clocks)) if clocks else session.at, "text": text})
    return session


async def _clock_now(context: TaskStepContext) -> str:
    """The task clock's time when one is armed (the earliest), else now; RFC 3339 UTC."""
    times = []
    for env in (d for d in context.deployed_envs if d.supports(EXT_CLOCK_URI, "state")):
        try:
            state = await env.invoke(EXT_CLOCK_URI, "state", timeout=5)
        except Exception as e:
            logger.warning(f"clock state read failed for env '{env.env_id}' (continuing): {e}")
            continue
        if state and state.get("armed") and state.get("virtual_time"):
            times.append(utc(state["virtual_time"]))
    return iso(min(times, default=datetime.now(timezone.utc)))
