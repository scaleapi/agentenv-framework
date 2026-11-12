"""The always_on step: a task that runs over simulated days, the way you'd hand a job to a personal assistant, with an
LLM user sim.

The ``deploy_user_sim_agent`` step (``deploy_user_sim_agent.py``, which documents the interface this step uses)
deploys the agent that plays the user, with its system prompt and model set, and records the user in the run's
metadata under its ``agent_name``. The ``always_on`` step names it with ``user_sim`` and runs the loop:

- The first wake is the user's request. Each wake opens with what came due for the agent on notifications/v1 (a
  trigger's ``notify``, a scheduled check), labelled ``[time · channel]``, then the user's own messages, each labelled
  with its time; while the task's clock is set, the agent can read it with the gateway's ``get_time``.
- Only what the agent sends with the gateway's ``message_user`` tool (user-chat/v1) reaches the user; its plain reply
  goes to the log. Each wake should end with ``message_user`` or ``stay_quiet``: one that ends with closing text and
  neither gets one ``nudge`` turn. ``stay_quiet`` and the nudge are on unless the task turns them off; the rule itself is
  in the agent's system prompt. A reminder at the end of every wake (``wake_reminder``) is off by default: with
  ``stay_quiet`` on, no wake in Opus 5.5 and GPT 6.1 runs of Surprise 30th ended in a message left in closing text.
  The step takes only its agent's role's messages: another role's stay queued, and what the role sent before the run
  is recorded apart (``user_chat_before_run``), never read by the user.
- The user sim is asked only after the agent sent something, at the time the user would read it (``schedule``), with
  everything sent meanwhile in one read: the time, the agent's messages, what triggers sent the user's phone
  (``notify`` to ``user_phone``) and the user's own lines (the opening request, ``user_says`` lines, ``final``).
  Nothing meant for the agent is in it. Its answer is reply / no_reply / done; a reply, or a done with text, is held
  for the response time. A schedule and response times are task time, so a user sim that sets either needs an armed
  clock (sync_env_clock): without one the step refuses to start. Done ends a one-sitting run (after the agent's turn on its text); a run whose end_time falls
  on a later day than its start goes on to end_time.
- Scripted user lines are triggers' ``notify`` actions to the ``user_says`` recipient: sent to the agent as the
  user's message at their time.
- The clock runs at the task's speed (sync_env_clock) only during the agent's turns: the step freezes it between them
  (clock/v1 set_rate), so settles and reads don't move it, and puts the task's rate back at the end. When nothing is
  due, the step advances it to the earliest of: the next trigger event, the next notification, a held reply, a
  pending read, or end_time, never past one; then waits for the world to settle. A contact's reply (an AI-played contact, a trigger's executor) counts as in
  progress until it ends or its time limit passes; other trigger work gets ``settle_timeout_seconds``. A contact's
  time limit is the one it was registered with (register_env_triggers' ``executor_timeout_seconds``), and one
  trigger's replies run side by side (``contact_replies_in_parallel``). What comes due in the same minute wakes the
  agent once.
- The agent can schedule its own wake-ups (scheduled checks) when the run is multi-day, at least
  ``scheduled_check_min_interval`` apart; ``scheduled_checks`` turns them on or off whatever the run's length. A
  check's wake counts toward ``max_wakes``.
- At end_time there is always a final check-in: the user's ``final`` line, or the step's own (``FINAL_DEFAULT_TEXT``)
  when the task sets none. An agent's turn that ends past end_time is followed by it, with whatever came due (a read
  due before end_time goes first), never by an ordinary wake. The run's last wake, at end_time or the one
  ``max_wakes`` makes the last, carries the last-turn notice (``final_reminder``). After it the user sim reads what
  the agent sent (its wrap-up), no earlier than the task's time when the run ended, and the user's reaction, and any
  reply still held, are recorded; then the run ends.
- A real-time budget keeps the run inside the hub's 8-hour job (``effective_time_budget`` gives the rule): once one
  more wake might end past ``time_budget_seconds`` (7 h, lowered to fit the task's own timeouts), the agent gets its
  closing wake, with the step's wrap-up line and the last-turn notice, and the run ends with end reason time_budget.
- End reasons: end_time, user_done (one sitting), max_wakes, time_budget, finished_nothing_pending or stuck (nothing
  can wake the agent, with nothing pending or with something still pending), agent_turn_failed, agent_turn_timeout,
  user_sim_error, step_error. With no end_time, a world that only repeats ends the run too (``IDLE_JUMPS_MAX``, in the
  loop's docstring), so a recurring trigger that never wakes the agent can't move the clock forever. What was still pending is recorded (``pending_at_end``): replies held, messages the user
  never read, user lines and alerts never delivered, the agent's checks not yet due, contact replies in progress.
- Graders read what the user saw. The prompt response is every message_user text in order, each with its pretend
  time, never the agent's closing text; messages no turn took (a turn that timed out after sending) are taken at the
  end and included, and listed as never read. The conversation (``metadata["a2a_conversations"][step_id]``) holds the user's
  own messages and the agent's message_user texts, then the user's last word that never reached the agent and an end
  marker with the end reason. An agent turn that doesn't complete (failed, canceled, rejected, timed out) ends the run
  and sets the response's ``error_type`` (as prompt_agent does for a failed one), so verifiers skip the run.
- Who did what, and in what order, even under a frozen clock: each app change (urn:agentenv:changes/v1) is recorded
  with who made it and the wake it was seen in, the agent's own with the wake that made it (``app_changes`` events);
  each reply records the agent's message it answers (``reply_to``: its wake, time and text); and each read leaves a
  record in ``metadata["post_turn_hooks"][step_id]`` in the shape the old psi-wake hook wrote, ``user_reads_at``
  being when the reply reached the agent, so checks built on those records grade the run instead of skipping it. A
  reply that never reached it (her last read's, or one still held at the end) has ``delivered`` false and no
  ``user_reads_at``.
- The run's metadata keeps a summary in ``metadata["always_on"][step_id]`` (small enough for a running job to report
  back); the full timeline goes to the object store, at the summary's ``object_url``.
- An always_on_agent_verifier (``verifiers/always_on_agent_verifier.py``) grades the step's prompt from the user's chat
  and the run's context, the run wake by wake among it (``render_for_judge``); a plain rubrics_verifier on it is
  refused when the task loads.
- The agent: the message_user rules go in its system prompt when the step sets one and the agent's card takes it,
  else on its first wake; its session is the step's ``context_id`` when set, as prompt_agent's is, else the run's own.
- When the step ends it puts back the gateway settings it changed for the run (the user chat's roles and stay_quiet,
  the triggers' concurrent_fires, the roles with scheduled checks, the clock's rate), so the task's later steps see
  them as they were.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any, ClassVar, Optional
from zoneinfo import ZoneInfo

from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent, conversation_store, protocol
from agent_env.a2a_agent.object_transfer import send_and_wait
from agent_env.artifact import FileArtifact
from agent_env.env.env import EnvCapabilityUnsupported
from agent_env.env.gateway.notifications import DEFAULT_MAX_ACTIVE
from agent_env.env.gateway.constants import (
    CHANGES_PAGE_MAX,
    DEFAULT_ROLE,
    EXT_CHANGES_URI,
    EXT_CLOCK_URI,
    EXT_NOTIFICATIONS_URI,
    EXT_STATE_URI,
    EXT_TRIGGERS_URI,
    EXT_USER_CHAT_URI,
)
from agent_env.store import DuplicateKeyError, get_config
from agent_env.task_step.context import PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps.notification_delivery import delivered_parts, label_time, take_due
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.agent.user_sim_common import iso as _iso, person_of, task_zone, utc as _utc

logger = logging.getLogger(__name__)

ALWAYS_ON_KEY = "always_on"
USER_SAYS = "user_says"
USER_PHONE = "user_phone"
IN_FLIGHT = ("firing", "queued")
POLL_SECONDS = 0.5
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
FINAL_REMINDER = ("(This is the last time you'll be woken for this job. Only what you send with message_user reaches "
                  "{first}; your normal reply is not shown to them.)")
NUDGE_TEXT = ("Your closing text was not delivered. If the user should see it, send it with message_user; otherwise "
              "call stay_quiet.")
WAKE_REMINDER_TEXT = "Reminder: only message_user reaches the user."
MESSAGE_RULES = ("The user never sees your normal replies. The only way to reach them is message_user. Text outside it "
                 "goes to a private log.")
QUIET_RULE = "If nothing is worth saying, call stay_quiet."
CHECKS_RULE = ("You only run when something wakes you. To do something at a later time, schedule a check with "
               "schedule_check: you'll be woken then with your note.")
CHECK_MIN_INTERVAL = "PT3H"
EMPTY_REPLIES = frozenset({"(The assistant's reply was empty.)"})  # a harness's stand-in for an empty final reply
# The hub runs a step as one job of at most this long, retrying one that runs over; the step keeps this much of it spare.
JOB_LIMIT_SECONDS = 8 * 3600
JOB_HEADROOM_SECONDS = 600
DEFAULT_TIME_BUDGET_SECONDS = 7 * 3600
# The time limit of a contact (a trigger's executor) is the one it was registered with, read from each env's triggers
# state when the run starts. Until then (a step's settings are checked when the task loads), and for a gateway that
# doesn't report it, the step takes the gateway's default for an executor registered without one.
CONTACT_TIMEOUT_DEFAULT_SECONDS = 120.0
# Past a contact's time limit, how long the settle wait still gives its call to come back.
CONTACT_GRACE_SECONDS = 30.0
# The hard caps that bound one turn of the loop (wake_reserve_seconds adds them up): past an agent's or user sim's turn
# limit, how long send_and_wait's own retries and polls still get before the turn is cut off; how long a trajectory
# fetch may take; how many user sim turns one read may take; and what the step's own gateway and store calls (takes,
# changes, the chat, clock reads) are allowed per wake and per read, beyond which the loop's turn is cut off.
TURN_GRACE_SECONDS = 60.0
TRAJECTORY_FETCH_SECONDS = 120.0
READ_ASKS = 2
GATEWAY_ALLOWANCE_SECONDS = 300.0
# At the end, how long taking the agent's messages no turn took may take (the run's record comes first).
LEFTOVER_TAKE_SECONDS = 60.0
# The final check-in's line when the task sets no `final`: the step's, not the user's.
FINAL_DEFAULT_TEXT = "Time's up on this job. Send {first} a short wrap-up: what's done, and anything still open."
NOTHING_CAN_WAKE = "nothing_can_wake_the_agent"  # until finish() tells finished_nothing_pending from stuck
ERROR_ENDS = ("agent_turn_failed", "agent_turn_timeout", "user_sim_error", "step_error")
# What each end reason means, for the end marker in the graded conversation; {wakes} and {first} are filled in.
END_NOTES = {
    "end_time": "The run reached its end time.",
    "user_done": "{first} signed off.",
    "max_wakes": "The run was cut off at its cap of {wakes} wakes.",
    "time_budget": "The run was cut off by its real-time budget.",
    "finished_nothing_pending": "Nothing could wake the assistant again, and nothing was pending.",
    "stuck": "Nothing could wake the assistant again, though something was still pending.",
    "agent_turn_failed": "The assistant's turn failed.",
    "agent_turn_timeout": "The assistant's turn timed out.",
    "user_sim_error": "The user simulator failed.",
    "step_error": "The run stopped on an error.",
}
# The run's metadata keeps a summary; the full timeline goes to the object store. When storing fails, the summary keeps
# this many of the last events instead, without these fields.
# The records a post-turn hook keeps (metadata["post_turn_hooks"][step id]), written in the shape the old psi-wake hook
# wrote them, so checks that read its reply times (user_reads_at) grade a run on this step.
HOOK_RECORDS_KEY = "post_turn_hooks"
HOOK_NAME = "always_on"
CHANGE_SUMMARY_CHARS = 200
# The app changes one read of an env's changelog keeps in the timeline; the rest are counted (counts.app_changes counts
# them all, by who made them).
CHANGES_KEPT_PER_READ = 200
TIMELINE_TAIL = 20
# With no end_time, the run ends once this many clock jumps in a row have been to repeating triggers' marks alone (see
# the loop's docstring): the world can go on ticking, but nothing in it has woken the agent.
IDLE_JUMPS_MAX = 24
# The clock jumps between two wakes the timeline keeps one by one; the rest are folded into one entry.
JUMPS_KEPT_PER_GAP = 20
PENDING_KEPT = 20
PENDING_TEXT_CHARS = 300
# What _small keeps of nesting: past this many levels, a list or dict is summed up as its count.
KEPT_NESTING_LEVELS = 6
# A user sim message longer than this, or one with a markdown heading or table, reads like an assistant's reply: its
# read is flagged (looks_like_assistant).
ASSISTANT_LIKE_MESSAGE_CHARS = 1200
BULKY_EVENT_FIELDS = ("input", "raw", "first_answer", "read", "parts_text", "changes")
# What the prompt response keeps of each wake's parts (source_agent_per_turn_prompt_parts): each text part cut to
# TURN_TEXT_CHARS, and past TURN_TEXT_TOTAL_CHARS in all, a note for the rest; every wake's full text is in the stored
# timeline (its wake event's parts_text).
TURN_TEXT_CHARS = 4000
TURN_TEXT_TOTAL_CHARS = 400_000


# ==================================================================================================================
class AlwaysOnTaskStep(PromptAgentTaskStep):
    type: ClassVar[str] = "always_on"
    entity_refs = ()
    field_scopes = {k: v for k, v in PromptAgentTaskStep.field_scopes.items() if k != "user_model"}

    def __init__(self, id: str, version: Optional[int], *, user_sim: str, end_time: Optional[str] = None,
                 final: Optional[str] = None, max_wakes: int = 60, settle_timeout_seconds: float = 300,
                 user_timeout_seconds: int = 300, final_reminder: Optional[str] = FINAL_REMINDER,
                 stay_quiet: bool = True, nudge: bool = True, nudge_text: str = NUDGE_TEXT,
                 wake_reminder: bool = False, wake_reminder_text: str = WAKE_REMINDER_TEXT,
                 scheduled_checks: Optional[bool] = None, scheduled_check_min_interval: str = CHECK_MIN_INTERVAL,
                 contact_replies_in_parallel: bool = True, time_budget_seconds: Optional[float] = None,
                 model: Optional[str] = None, model_params: Optional[dict] = None, effort: Optional[str] = None,
                 max_thinking_tokens: Optional[int] = None, harness: Optional[str] = None,
                 trajectory_output_prefix: Optional[str] = None, **prompt_fields: Any):
        for unsupported in ("max_conversation_turns", "user_agent_name", "user_a2a_url", "post_turn_hooks",
                            "user_output_format", "user_model", "snapshot_config"):
            prompt_fields.pop(unsupported, None)
        # The user's first message is the user sim's opening_prompt, read when the run starts: whoever plays the user
        # sends it, so the step has no prompt of its own.
        given = [k for k in ("prompt", "parts") if prompt_fields.get(k) is not None]
        if given:
            raise ValueError(f"always_on takes no {' or '.join(given)}: the user's first message is the "
                             "deploy_user_sim_agent step's opening_prompt; move it there")
        prompt_fields["prompt"] = ""  # for prompt_agent's check; unset again below
        super().__init__(id, version, max_conversation_turns=max(1, int(max_wakes)), model=model,
                         model_params=model_params, effort=effort, max_thinking_tokens=max_thinking_tokens,
                         harness=harness, trajectory_output_prefix=trajectory_output_prefix, **prompt_fields)
        self.prompt = None
        if not user_sim:
            raise ValueError("always_on needs user_sim, the agent_name of a deploy_user_sim_agent step")
        if end_time is not None:
            _end_instant(end_time, datetime.now(timezone.utc))  # validates
        self.user_sim = user_sim
        self.end_time = end_time
        self.final = final
        self.max_wakes = int(max_wakes)
        self.settle_timeout_seconds = float(settle_timeout_seconds)
        self.user_timeout_seconds = int(user_timeout_seconds)
        self.final_reminder = final_reminder
        self.stay_quiet = bool(stay_quiet)
        self.nudge = bool(nudge)
        self.nudge_text = nudge_text
        self.wake_reminder = bool(wake_reminder)
        self.wake_reminder_text = wake_reminder_text
        self.scheduled_checks = scheduled_checks
        self.scheduled_check_min_interval = scheduled_check_min_interval
        self.contact_replies_in_parallel = bool(contact_replies_in_parallel)
        self.time_budget_seconds = None if time_budget_seconds is None else float(time_budget_seconds)
        self.effective_time_budget()  # validates

    _OWN = ("user_sim", "end_time", "final", "max_wakes", "settle_timeout_seconds", "user_timeout_seconds",
            "final_reminder", "stay_quiet", "nudge", "nudge_text", "wake_reminder", "wake_reminder_text",
            "scheduled_checks", "scheduled_check_min_interval", "contact_replies_in_parallel", "time_budget_seconds")
    _PROMPT_FIELDS = ("prompt", "parts", "prompt_id", "agent_name", "timeout_seconds", "trajectory_output_prefix",
                      "model", "model_params", "system_prompt", "max_turns", "output_format", "max_thinking_tokens",
                      "effort", "harness", "agentenv_tools", "poll_interval_seconds", "context_id")

    def settle_cap_seconds(self, contact_seconds: float = CONTACT_TIMEOUT_DEFAULT_SECONDS) -> float:
        """The longest one settle waits, whatever is still at work (queued fires, nl actions in a row): the longest
        contact time limit (``contact_seconds``: the run's, read when it starts, else the gateway's default) and its
        grace, then settle_timeout_seconds."""
        return contact_seconds + CONTACT_GRACE_SECONDS + self.settle_timeout_seconds

    def read_cap_seconds(self) -> float:
        """The longest one read's answer may take: READ_ASKS user sim turns, each with its grace and trajectory
        fetch; an ask past it is refused, and an answer past it ends the read as a user sim error."""
        return READ_ASKS * (self.user_timeout_seconds + TURN_GRACE_SECONDS + TRAJECTORY_FETCH_SECONDS)

    def wake_reserve_seconds(self, contact_seconds: float = CONTACT_TIMEOUT_DEFAULT_SECONDS) -> float:
        """The longest one turn of the loop, or the run's closing, can take, each part hard-capped: a clock jump's
        settle (settle_cap_seconds), then a wake (the agent's turn and, with nudges on, its nudge turn, each at most
        timeout_seconds + TURN_GRACE_SECONDS plus a trajectory fetch; one settle; GATEWAY_ALLOWANCE_SECONDS of the
        step's own calls), then a read (read_cap_seconds and GATEWAY_ALLOWANCE_SECONDS). The loop cuts off a turn of
        its own that runs past this (a step error), so it is a real bound. ``contact_seconds`` is the longest contact
        time limit, as for settle_cap_seconds."""
        turn = self.timeout_seconds + TURN_GRACE_SECONDS + TRAJECTORY_FETCH_SECONDS
        settle = self.settle_cap_seconds(contact_seconds)
        wake = turn * (2 if self.nudge else 1) + settle + GATEWAY_ALLOWANCE_SECONDS
        return settle + wake + self.read_cap_seconds() + GATEWAY_ALLOWANCE_SECONDS

    def effective_time_budget(self, contact_seconds: float = CONTACT_TIMEOUT_DEFAULT_SECONDS) -> float:
        """The real time after which the run closes: ``time_budget_seconds``, else 7 h, lowered so the closing still
        fits the hub's job. The rule:

        The hub runs a step as one job of at most 8 h (``JOB_LIMIT_SECONDS``) and retries one that runs over, which
        would start the step again against a world whose clock has already moved. So the step never starts a wake or a
        read that might end past the budget: once the time spent plus one wake's worst case
        (``wake_reserve_seconds``) reaches it, it gives the agent its closing wake instead (whatever is due, the
        step's wrap-up line and the last-turn notice), lets the user sim read the wrap-up, and ends with end reason
        ``time_budget``. A run therefore ends by the budget plus one wake's worst case, and that must fit the job
        with ``JOB_HEADROOM_SECONDS`` to spare: the default budget is lowered to make it fit, and a budget a task sets
        that can't is refused, as is a wake whose worst case leaves no room for a budget at all.

        ``contact_seconds`` is the longest contact time limit: the gateway's default when the task loads, the
        contacts' own once the run has read them (``_configure_contacts``), which checks the budget again."""
        reserve = self.wake_reserve_seconds(contact_seconds)
        room = JOB_LIMIT_SECONDS - JOB_HEADROOM_SECONDS - reserve
        if room <= reserve:
            raise ValueError(f"always_on: one wake can take up to {reserve:.0f}s (wake_reserve_seconds: a settle, the "
                             f"agent's turn and nudge, a settle and a read), so no run fits the hub's "
                             f"{JOB_LIMIT_SECONDS // 3600} h job; lower timeout_seconds or user_timeout_seconds")
        if self.time_budget_seconds is None:
            return min(DEFAULT_TIME_BUDGET_SECONDS, room)
        if not 0 < self.time_budget_seconds <= room:
            raise ValueError(f"always_on: time_budget_seconds={self.time_budget_seconds:.0f} can't fit: with one wake's "
                             f"worst case ({reserve:.0f}s) after it, the run must close within {room:.0f}s to end "
                             f"inside the hub's {JOB_LIMIT_SECONDS // 3600} h job")
        return self.time_budget_seconds

    def to_dict(self) -> dict:
        base = TaskStep.to_dict(self)
        for k in self._PROMPT_FIELDS:
            base[k] = getattr(self, k)
        for k in self._OWN:
            base[k] = getattr(self, k)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> AlwaysOnTaskStep:
        fields = {k: data[k] for k in cls._PROMPT_FIELDS if k in data and data[k] is not None}
        own = {k: data[k] for k in cls._OWN if k in data and data[k] is not None}
        return cls(**cls._base_from_dict(data), retry_config=data.get("retry_config"), **fields, **own)

    def task_problems(self, steps: list[TaskStep]) -> list[str]:
        """Its prompt is graded by always_on_agent_verifier: a plain rubrics_verifier, which would grade it from every
        wake's trajectory, is refused."""
        # Here, not at the top: that module imports this one.
        from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import AlwaysOnAgentVerifier
        from agent_env.task_step.task_steps.verifiers.rubrics_verifier import is_rubrics_verifier

        return [f"step '{s.id}' ({s.type}) grades the prompt '{self.prompt_id}' of always_on step '{self.id}', which "
                f"is graded from the user's chat and the run's context, not its wakes' trajectories: use type "
                f"{AlwaysOnAgentVerifier.type}"
                for s in steps if is_rubrics_verifier(s) and s.prompt_id == self.prompt_id
                and not isinstance(s, AlwaysOnAgentVerifier)]

    async def _execute(self, context: TaskStepContext) -> TaskStepContext:
        run = _Run(self, context)
        try:
            await run.go()
        finally:
            try:
                await run.restore()
            finally:
                run.close()
        return context


# ==================================================================================================================
class _Run:
    """One always-on run: its state, its timeline and its loop."""

    def __init__(self, step: AlwaysOnTaskStep, context: TaskStepContext):
        self.step = step
        self.ctx = context
        self.timeline: list[dict] = []
        self.t0_real = time.time()
        self.wakes = 0
        self.end_reason: Optional[str] = None
        self.unread: list[dict] = []          # the agent's messages the user hasn't read
        self.read_at: Optional[datetime] = None
        self.read_mode = "now"
        self.held: list[dict] = []            # user replies held for their response time: {deliver_at, text}
        self.you_sent: list[dict] = []        # the user's own lines since their last read: {at, text}
        self.history: list[dict] = []         # every read and its answer, for the plug-in
        self.replies = 0
        self.gave_up = False
        self.final_sent = False
        self.last_reply_at: Optional[datetime] = None
        self.per_turn_trajectory_urls: list[Optional[str]] = []
        self.per_turn_parts: list[Optional[list[dict]]] = []
        self.tool_calls: Optional[int] = None
        self.all_messages: list[dict] = []
        self.final_task_id = ""
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None  # set as prompt_agent sets it, so verifiers skip a run that broke
        self.error_code: Optional[str] = None
        self.chat_pending: list[dict] = []  # the agent's message_user texts not yet in the graded conversation
        self.reaction: Optional[dict] = None  # her last read's answer when she said nothing
        self.change_cursors: dict[str, tuple[Any, int]] = {}  # env id -> (env, the last changelog id read)
        self.change_counts = {"by_agent": 0, "by_others": 0, "no_actor": 0}
        self.hook_records: list[dict] = []
        self._plugin_paths: list[str] = []
        self._plugin_modules: list[str] = []
        self.nudges = 0
        self.zone = "UTC"
        self.multi_day = False
        self.checks: dict = {"on": False}
        self.signed_off = False  # a one-sitting run's user said done: no more reads
        self.contacts: dict = {}
        self.contact_seconds = CONTACT_TIMEOUT_DEFAULT_SECONDS  # the longest contact time limit, read at the start
        self.last_settle: dict = {}
        self.pending: dict = {}
        self.end_note = ""
        self.time_budget = step.effective_time_budget()
        self.idle_jumps = 0           # clock jumps in a row to repeating triggers' marks alone, since the last wake
        self.idle_since: Optional[str] = None
        self.clock_jumps = 0
        self.jumps_in_gap = 0         # clock jumps since the last wake, the first JUMPS_KEPT_PER_GAP of them logged
        self.folded: Optional[dict] = None
        self.clock: dict = {}                  # the task's clock rate, and whether the step freezes it between turns
        self.clock_rates: list[tuple[Any, float]] = []  # (env, the task's rate) for each clock the step freezes
        self.undo: list[tuple[str, Any]] = []  # (what, coroutine function): the gateway settings to put back at the end
        self.rules_in_system_prompt = False   # whether the agent took the rules in its system prompt

    async def restore(self) -> None:
        """Put back what the step changed on the gateways, latest first, so later steps of the task see the settings
        they had before it: the user chat's roles, user name and stay_quiet; the triggers' concurrent_fires; the roles
        with scheduled checks; the clock's rate. Each is tried on its own; one that fails is logged."""
        while self.undo:
            what, put_back = self.undo.pop()
            try:
                await put_back()
            except Exception as e:
                logger.warning(f"always_on: could not put back {what} after the run: {type(e).__name__}: {e}")

    def _undo_config(self, env, uri: str, what: str, body: dict) -> None:
        """At the end, set ``env``'s ``uri`` config back to ``body``, what it was before the step changed it."""
        async def put_back() -> None:
            await _call(env, uri, "config", body, attempts=3)
        self.undo.append((f"{what} on {env.env_id}", put_back))

    def close(self) -> None:
        for p in self._plugin_paths:
            if os.path.exists(p):
                os.unlink(p)
        for m in self._plugin_modules:
            sys.modules.pop(m, None)

    def spent(self) -> float:
        """Real seconds since the step started."""
        return time.time() - self.t0_real

    # ---- record ----------------------------------------------------------------------------------------------
    def log(self, event: str, **fields: Any) -> dict:
        """Add an event to the run's timeline, which keeps it whole; the process log gets only its name, ids, times,
        counts and sizes (``_log_view``), since texts and reads carry what people wrote."""
        entry = {"event": event, "real_time": datetime.now(timezone.utc).isoformat(),
                 "real_elapsed_s": round(time.time() - self.t0_real, 1), **fields}
        self.timeline.append(entry)
        logger.info(f"always_on {event}: " + json.dumps(_log_view(fields), default=str))
        return entry

    # ---- setup -----------------------------------------------------------------------------------------------
    async def go(self) -> None:
        step, ctx = self.step, self.ctx
        self.agent = next((a for a in ctx.deployed_agents if a.agent_name == step.agent_name), None)
        if self.agent is None:
            raise StepContextError(f"Agent with name '{step.agent_name}' not found in context.deployed_agents")
        self.target_url = self.agent.a2a_url or self.agent.api_url
        self.recipient = self.agent.role or DEFAULT_ROLE
        us = (ctx.metadata.get(USER_SIMS_KEY) or {}).get(step.user_sim)
        if us is None:
            raise StepContextError(f"always_on: no deploy_user_sim_agent step named '{step.user_sim}' ran before this step")
        self.us = us
        self.zone = _task_zone(us, ctx.metadata.get("clock_configurations") or [])
        self.person = {**person_of(us), "time_zone": self.zone}
        self.version = us["user_version"]
        self.user_agent = next((a for a in ctx.deployed_agents if a.agent_name == us["agent_name"]), None)
        if self.user_agent is None:
            raise StepContextError(f"always_on: the user sim agent '{us['agent_name']}' is not deployed")
        self.user_url = self.user_agent.a2a_url or self.user_agent.api_url
        self.plugin = await self._load_plugin(us.get("plugin"))
        self.tz = ZoneInfo(self.zone)

        self.clock_envs = [d for d in ctx.deployed_envs if d.supports(EXT_CLOCK_URI, "state")]
        self.notification_envs = [d for d in ctx.deployed_envs if d.supports(EXT_NOTIFICATIONS_URI, "take")]
        self.chat_envs = [d for d in ctx.deployed_envs if d.supports(EXT_USER_CHAT_URI, "take")]
        if not self.chat_envs:
            raise RuntimeError(f"always_on: no deployed env offers {EXT_USER_CHAT_URI} (message_user); rebuild its "
                               "gateway from a branch that has user_chat.py")
        timed = [k for k in ("schedule", "response_times") if us.get(k)]
        if timed and not await self._armed():
            # Read times and reply delays are task time; with no clock to jump, the loop would wait them out in real
            # time, spinning on the clock it can't move.
            raise ValueError(f"always_on: user sim '{step.user_sim}' sets {' and '.join(timed)}, which need the "
                             "task's clock, and no env's clock is armed; run sync_env_clock before this step, or drop "
                             f"{' and '.join(timed)} from the user sim's persona")
        if not us.get("opening"):
            raise ValueError(f"always_on: the user sim '{step.user_sim}' has no opening_prompt, the user's first "
                             "message; give the deploy_user_sim_agent step one")
        now = await self.now()
        self.opening = {"at": _iso(now), "text": self._effective_prompt()}
        self.person["opening"] = self.opening
        t0 = await self._t0() or now
        self.end = _end_instant(step.end_time, t0) if step.end_time else None
        self.multi_day = self.end is not None and self.end.astimezone(self.tz).date() > t0.astimezone(self.tz).date()
        if us.get("prompt_set") and us.get("multi_day") != self.multi_day:
            kind = "multi-day" if self.multi_day else "one-sitting"
            raise ValueError(f"always_on: this run is {kind} (end_time {step.end_time!r}), but the user sim "
                             f"'{step.user_sim}' was deployed with multi_day {us.get('multi_day')!r}, so its prompt has "
                             f"the other done line; set multi_day {str(self.multi_day).lower()} on the "
                             "deploy_user_sim_agent step")
        self.person["done_ends_run"] = not self.multi_day  # for setup, which writes her prompt before any read
        chat_config = {"roles": [self.recipient], "user_name": self.person.get("name") or "your user"}
        if step.stay_quiet:
            chat_config["stay_quiet"] = True
        for env in self.chat_envs:
            before = await _call(env, EXT_USER_CHAT_URI, "state") if env.supports(EXT_USER_CHAT_URI, "state") else {}
            self._undo_config(env, EXT_USER_CHAT_URI, "the user chat's config",
                              {"roles": before.get("roles") or [], "user_name": before.get("user_name") or "your user",
                               **({"stay_quiet": True} if before.get("stay_quiet") else {})})
            await _call(env, EXT_USER_CHAT_URI, "config", chat_config)
        await self._configure_clock()
        await self._configure_checks()
        await self._configure_contacts()
        await self._configure_agents()
        await self._start_changes()

        self.conversation_id = uuid.uuid4().hex
        try:
            conversation_store.create_conversation(conversation_id=self.conversation_id,
                                                   task_instance_id=ctx.instance_id or "",
                                                   source_agent_name=us["agent_name"],
                                                   target_agent_name=step.agent_name)
        except DuplicateKeyError:
            pass
        ctx.metadata.setdefault("a2a_conversations", {})[step.id] = self.conversation_id
        self.user_context_id = uuid.uuid4().hex   # one user sim session per run
        # The agent's session: the step's context_id, as prompt_agent uses it, so a run can go on with the session an
        # earlier step left; else a new one. The graded conversation is the run's own either way.
        self.solver_context_id = step.context_id or uuid.uuid4().hex

        # From here the conversation exists, so whatever fails still closes it and leaves the run's record.
        try:
            self.hook_records.append({"hook": HOOK_NAME, "turn": 0, "started_at": _iso(await self.now())})
            ctx.metadata.setdefault(HOOK_RECORDS_KEY, {})[step.id] = self.hook_records
            now = await self.now()
            self.log("start", virtual_time=_iso(now), end_time=_iso(self.end) if self.end else None,
                     agent=step.agent_name, agent_role=self.recipient, user_sim=step.user_sim, version=self.version,
                     user_model=self.user_model, max_wakes=step.max_wakes, time_zone=self.zone,
                     multi_day=self.multi_day, scheduled_checks=self.checks, clock_rate=await self._rate())
            # What the agent's role sent before the run (a task that gave it message_user earlier) isn't this run's:
            # set it aside, recorded, so the first wake doesn't take it as its own.
            before, quiet = await self._take_messages()
            if before or quiet:
                self.log("user_chat_before_run", messages=before, stay_quiet_calls=len(quiet))
            await self.loop(now)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.end_reason = self.end_reason or "step_error"
            self.log("error", error=self.error)
            raise
        finally:
            await self.finish()

    async def _load_plugin(self, spec: Optional[dict]) -> ModuleType:
        if not spec:
            return user_sim_versions
        artifact = FileArtifact.get(spec["file_artifact_id"], spec.get("file_artifact_version"))
        source = await asyncio.to_thread(artifact.load)
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, source)
        os.close(fd)
        self._plugin_paths.append(path)
        name = f"user_sim_plugin_{uuid.uuid4().hex[:8]}"
        mspec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(mspec)
        sys.modules[name] = module
        self._plugin_modules.append(name)
        mspec.loader.exec_module(module)
        if hasattr(module, "configure"):
            module.configure(spec.get("params") or {})
        return module

    async def _configure_agents(self) -> None:
        step, ctx = self.step, self.ctx
        model, effort = step._resolve_model_and_effort(ctx)
        self.model = model
        overrides = ctx.metadata.get("user_overrides", {})
        card = self.agent.a2a_card or {}
        desired = {"model": model, "system_prompt": self._system_prompt(), "effort": effort,
                   "harness": ctx.agent_harness or step.harness,
                   "max_turns": overrides.get("agent_max_turns") or step.max_turns,
                   "max_thinking_tokens": overrides.get("agent_max_thinking_tokens") or step.max_thinking_tokens,
                   "output_format": step.output_format, "timeout_seconds": step.timeout_seconds,
                   "task_id": ctx.metadata.get("task_id"), "agentenv_tools": step.agentenv_tools,
                   "builtin_tools": card.get("builtin_tools")}
        desired = {k: v for k, v in desired.items() if v is not None}
        model_params = get_config().get_model_params(step.model_params)
        if model_params:
            desired["model_params"] = model_params
        negotiated = A2AAgent.negotiate_agent_config(card, desired)
        if negotiated:
            await protocol.post_agent_config(self.target_url + negotiated.endpoint, negotiated.fields)
        ctx.record_model("solver", step.id, negotiated.fields.get("model") if negotiated else None)
        self.rules_in_system_prompt = bool(desired.get("system_prompt") and negotiated
                                           and "system_prompt" in negotiated.fields)
        if desired.get("system_prompt") and not self.rules_in_system_prompt:
            logger.warning("always_on: the agent's card did not accept system_prompt, so the message_user rules go "
                           "on its first wake instead")

        if self.us.get("prompt_set"):  # deploy_user_sim_agent set the user sim's prompt and model at deploy
            self.user_model, self.user_model_from = self.us["model"], self.us["model_source"]
            self.user_setup = dict(self.us["setup"])
            return
        setup = self.plugin.setup(self.version, self.person)
        self.user_model, self.user_model_from = self._user_model()
        user_config = {"model": self.user_model, "system_prompt": setup.get("system_prompt"),
                       "output_format": setup.get("output_format")}
        user_config = {k: v for k, v in user_config.items() if v is not None}
        negotiated = A2AAgent.negotiate_agent_config(self.user_agent.a2a_card or {}, user_config)
        if negotiated:
            await protocol.post_agent_config(self.user_url + negotiated.endpoint, negotiated.fields)
        accepted = sorted(negotiated.fields) if negotiated else []
        ctx.record_model("user simulator", step.id, negotiated.fields.get("model") if negotiated else None)
        self.user_setup = {"model": self.user_model, "model_from": self.user_model_from,
                           "user_sim_step_model": self.us.get("model"),
                           "model_set": "model" in accepted,  # else the agent ran its own
                           "system_prompt_chars": len(setup.get("system_prompt") or ""),
                           "output_format": setup.get("output_format"), "accepted_fields": accepted}
        if "system_prompt" not in accepted and setup.get("system_prompt"):
            logger.warning("always_on: the user sim agent's card did not accept system_prompt; it keeps its deploy one")

    async def _configure_clock(self) -> None:
        """The clock runs at the task's rate only while the agent works (FD-5366): freeze it (rate 0) now, and between
        turns, so a slow contact reply or user sim read doesn't move the task's time. With clock/v1 ``set_rate``,
        which keeps the clock's marks (a re-arm with set_time, as sync_env_clock arms it, would re-anchor every
        relative or recurring time trigger); the task's rate is put back at the end. A clock already frozen (rate 0)
        is left alone; a gateway without set_rate keeps running between turns, as the record says."""
        envs = []
        for env in await self._armed():
            state = await _call(env, EXT_CLOCK_URI, "state")
            rate = float(state.get("virtual_seconds_per_real_second") or 0)
            if rate <= 0:
                continue
            if not env.supports(EXT_CLOCK_URI, "set_rate"):
                envs.append({"env_id": env.env_id, "rate": rate, "frozen_between_turns": False,
                             "why": "the gateway has no clock set_rate"})
                continue
            self.clock_rates.append((env, rate))
            envs.append({"env_id": env.env_id, "rate": rate, "frozen_between_turns": True})
        if self.clock_rates:
            self.undo.append(("the clock's rate", lambda: self._clock_runs(True)))
            await self._clock_runs(False)
        self.clock = {"envs": envs}

    async def _clock_runs(self, running: bool) -> None:
        """Run each clock the step freezes at the task's rate (``running``), or freeze it."""
        for env, rate in self.clock_rates:
            await _call(env, EXT_CLOCK_URI, "set_rate", {"virtual_seconds_per_real_second": rate if running else 0},
                        attempts=3)

    async def _agent_turn(self, parts: list[dict], msg_id: str, record) -> tuple[str, dict]:
        """One turn of the agent on ``parts``: the task's clock runs at its rate while the agent works, and is frozen
        again after."""
        step = self.step
        await self._clock_runs(True)
        try:
            # Cut off past its limit and grace, whatever send_and_wait's own retries do: a TimeoutError, as its own.
            return await asyncio.wait_for(
                send_and_wait(self.target_url, parts, agent=self.agent, message_id=msg_id,
                              context_id=self.solver_context_id, timeout_seconds=step.timeout_seconds,
                              poll_interval_seconds=step.poll_interval_seconds, before_send=record),
                step.timeout_seconds + TURN_GRACE_SECONDS)
        finally:
            await self._clock_runs(False)

    async def _configure_contacts(self) -> None:
        """On each env with triggers, let one trigger's replies run side by side (``contact_replies_in_parallel``; a
        gateway too old for it keeps its ways), and read its contacts' time limits: each executor's own, as the task
        registered it (register_env_triggers' ``executor_timeout_seconds``). The longest (``contact_seconds``; the
        gateway's default for a gateway that doesn't report them) caps every settle, and with it one wake's worst case,
        so the run's time budget is worked out again from it, and refused when that can't fit the hub's job."""
        step = self.step
        applied, limits = [], []
        for env in self.ctx.deployed_envs:
            if not env.supports(EXT_TRIGGERS_URI, "state"):
                continue
            state = await _call(env, EXT_TRIGGERS_URI, "state")
            timeouts = (state.get("config") or {}).get("executor_timeouts")
            limits += [CONTACT_TIMEOUT_DEFAULT_SECONDS] if timeouts is None else [t["timeout_seconds"] for t in timeouts]
            entry = {"env_id": env.env_id, "contact_timeouts": timeouts}  # None: the gateway doesn't report them
            if not env.supports(EXT_TRIGGERS_URI, "config"):
                applied.append({**entry, "applied": False, "why": "the gateway has no triggers config"})
                continue
            before = state.get("settings") or {}
            self._undo_config(env, EXT_TRIGGERS_URI, "the triggers' settings",
                              {k: v for k, v in before.items() if k == "concurrent_fires" and v is not None})
            got = await _call(env, EXT_TRIGGERS_URI, "config", {"concurrent_fires": step.contact_replies_in_parallel})
            applied.append({**entry, "applied": True, "concurrent_fires": got.get("concurrent_fires")})
        self.contact_seconds = float(max(limits, default=0.0))
        self.time_budget = step.effective_time_budget(self.contact_seconds)
        self.contacts = {"replies_in_parallel": step.contact_replies_in_parallel,
                         "longest_timeout_seconds": self.contact_seconds, "envs": applied}

    def rules(self) -> str:
        """What the agent is told about reaching the user, and about waking itself when it can."""
        return " ".join([MESSAGE_RULES, *([QUIET_RULE] if self.step.stay_quiet else []),
                         *([CHECKS_RULE] if self.checks["on"] else [])])

    def _user_model(self) -> tuple[str, str]:
        """The user sim agent's model for this run, and where it came from: the user_sim step's own ``step_params``
        model, else the run's ``user_model`` (``task run`` / ``eval run --user-model``, a bundle's ``user_model``, as
        prompt_agent's ``_resolve_user_model`` honours it), else the user_sim step's ``model``."""
        overrides = self.ctx.metadata.get("user_overrides") or {}
        own = ((overrides.get("step_params") or {}).get(self.us.get("step_id")) or {}).get("model")
        if own:
            return own, "user_sim_step_params"
        if overrides.get("user_model"):
            return overrides["user_model"], "run_user_model"
        return self.us["model"], "user_sim_step"

    async def _configure_checks(self) -> None:
        """Give the agent's role scheduled checks (notifications/v1) on each env whose clock is set: by default when
        the run is multi-day, else as ``scheduled_checks`` says. A role the task already gave checks keeps its own
        settings; other roles the task named keep theirs."""
        step = self.step
        wanted = self.multi_day if step.scheduled_checks is None else step.scheduled_checks
        envs = [e for e in await self._armed() if e.supports(EXT_NOTIFICATIONS_URI, "config")]
        applied = []
        for env in envs:
            state = await _call(env, EXT_NOTIFICATIONS_URI, "state")
            roles = list(state.get("roles") or [])
            if self.recipient in roles:
                applied.append({"env_id": env.env_id, "set_by": "task", "min_interval": state.get("min_interval"),
                                "max_active": state.get("max_active")})
            elif wanted:
                body = {"roles": [*roles, self.recipient],
                        "max_active": state.get("max_active", DEFAULT_MAX_ACTIVE) if roles else DEFAULT_MAX_ACTIVE,
                        "min_interval": state.get("min_interval") if roles else step.scheduled_check_min_interval}
                self._undo_config(env, EXT_NOTIFICATIONS_URI, "the roles with scheduled checks",
                                  {"roles": roles, **{k: state[k] for k in ("max_active", "min_interval")
                                                      if state.get(k) is not None}})
                got = await _call(env, EXT_NOTIFICATIONS_URI, "config", body)
                applied.append({"env_id": env.env_id, "set_by": "step", "min_interval": got.get("min_interval"),
                                "max_active": got.get("max_active")})
        self.checks = {"on": bool(applied), "envs": applied}

    def _system_prompt(self) -> Optional[str]:
        """The step's system prompt with the rules after it; without one, or when the agent's card doesn't accept it,
        the agent keeps its own and the first wake opens with the rules."""
        return f"{self.step.system_prompt.rstrip()}\n\n{self.rules()}" if self.step.system_prompt else None

    # ---- clock and world ------------------------------------------------------------------------------------
    async def _armed(self) -> list:
        armed = []
        for env in self.clock_envs:
            try:
                if (await _call(env, EXT_CLOCK_URI, "state")).get("armed"):
                    armed.append(env)
            except Exception as e:
                logger.warning(f"clock state of {env.env_id} failed: {e}")
        return armed

    async def now(self) -> datetime:
        envs = await self._armed()
        if not envs:
            return datetime.now(timezone.utc)
        return min([_utc((await _call(env, EXT_CLOCK_URI, "state"))["virtual_time"]) for env in envs])

    async def _t0(self) -> Optional[datetime]:
        envs = await self._armed()
        return min([_utc((await _call(env, EXT_CLOCK_URI, "state"))["t0"]) for env in envs]) if envs else None

    async def _rate(self) -> Optional[float]:
        envs = await self._armed()
        return (await _call(envs[0], EXT_CLOCK_URI, "state")).get("virtual_seconds_per_real_second") if envs else None

    async def advance(self, target: datetime, reason: str) -> None:
        before = await self.now()
        if target <= before:
            return
        armed = await self._armed()
        if not armed:  # nothing to move: the loop would come straight back here, spinning in real time
            raise RuntimeError(f"always_on: can't move the task's clock to {_iso(target)} ({reason}): no env's clock "
                               "is armed")
        for env in armed:
            try:
                await _call(env, EXT_CLOCK_URI, "advance", {"virtual_time": _iso(target)}, attempts=2)
            except Exception:
                state = await _call(env, EXT_CLOCK_URI, "state")
                if not (state.get("armed") and _utc(state["virtual_time"]) >= target):
                    raise
        stuck = await self.settle()
        self.clock_jumps += 1
        self.jumps_in_gap += 1
        if self.jumps_in_gap <= JUMPS_KEPT_PER_GAP:
            self.log("clock_jump", **{"from": _iso(before), "to": _iso(target), "reason": reason,
                                      "settled": not stuck, "still_at_work": stuck, "settle": self.last_settle})
            return
        # Past the first jumps since the last wake, one entry sums the rest up, so a long gap can't grow the timeline.
        if self.folded is None:
            self.folded = self.log("clock_jumps_folded", count=0, **{"from": _iso(before)}, reasons={},
                                   unsettled=0, settle_seconds=0.0)
        f = self.folded
        f["count"] += 1
        f["to"] = _iso(target)
        f["reasons"][reason] = f["reasons"].get(reason, 0) + 1
        f["unsettled"] += 1 if stuck else 0
        f["settle_seconds"] = round(f["settle_seconds"] + self.last_settle.get("seconds", 0), 1)
        if stuck:
            f["last_still_at_work"] = stuck

    async def settle(self) -> list[dict]:
        """Wait until the world has settled: no trigger still at work, bar a fire waiting for a later time. A contact's
        reply (a call to a trigger's executor) counts as in progress until it ends or its time limit passes (plus
        ``CONTACT_GRACE_SECONDS``); any other work, an event due that an advance to now doesn't set going among it,
        gets ``settle_timeout_seconds``, from the start or from the last contact call seen. A gateway that reports no
        executor calls gets ``settle_timeout_seconds`` in all. However much is still at work (queued fires, nl
        actions in a row), the wait never runs past ``settle_cap_seconds`` of the longest contact time limit.
        Returns, by env, the triggers still at work when it gave up (empty when the world settled); ``last_settle``
        says how long it waited and how many contact calls it saw."""
        started = time.monotonic()
        cap = started + self.step.settle_cap_seconds(self.contact_seconds)  # however much is at work, never past this
        envs = [e for e in await self._armed() if e.supports(EXT_TRIGGERS_URI, "state")]
        waiting = {i: {"live_call_seen": started, "contact_until": started} for i in range(len(envs))}
        stuck: list[dict] = []
        calls_seen = 0
        while waiting:
            for i in sorted(waiting):
                env, track = envs[i], waiting[i]
                clock = await _call(env, EXT_CLOCK_URI, "state")
                state = await _call(env, EXT_TRIGGERS_URI, "state")
                now, mono = _utc(clock["virtual_time"]), time.monotonic()
                rows = state.get("triggers", [])
                live = [c for row in rows for c in row.get("executor_calls") or [] if c.get("seconds_left", 0) > 0]
                if live:
                    calls_seen = max(calls_seen, len(live))
                    track["live_call_seen"] = mono
                    track["contact_until"] = max(track["contact_until"], mono + CONTACT_GRACE_SECONDS
                                                 + max(c["seconds_left"] for c in live))
                busy = [row for row in rows if _busy(row, now)]
                due = state.get("next_event_at") is not None and _utc(state["next_event_at"]) <= now
                limit = min(cap, max(track["contact_until"],
                                     track["live_call_seen"] + self.step.settle_timeout_seconds))
                if not due and not busy:
                    waiting.pop(i)
                elif mono >= limit:  # a due event that won't go, like other work, gets settle_timeout_seconds
                    stuck.append({"env_id": env.env_id, "triggers": [_at_work(row) for row in busy],
                                  **({"due_not_going": state["next_event_at"]} if due else {})})
                    waiting.pop(i)
                elif due:  # an event already due: an advance to now sets it going
                    try:
                        await _call(env, EXT_CLOCK_URI, "advance", {"virtual_time": _iso(now)}, attempts=1)
                    except Exception:
                        pass
            if waiting:
                self.ctx.heartbeat()
                await asyncio.sleep(POLL_SECONDS)
        self.last_settle = {"seconds": round(time.monotonic() - started, 1), "contact_calls_seen": calls_seen}
        return stuck

    async def _next_world_event(self) -> Optional[datetime]:
        out = []
        for env in await self._armed():
            if env.supports(EXT_TRIGGERS_URI, "state"):
                st = await _call(env, EXT_TRIGGERS_URI, "state")
                if st.get("next_event_at"):
                    out.append(_utc(st["next_event_at"]))
        return min(out) if out else None

    async def _trigger_rows(self) -> list[dict]:
        rows = []
        for env in await self._armed():
            if env.supports(EXT_TRIGGERS_URI, "state"):
                rows += (await _call(env, EXT_TRIGGERS_URI, "state")).get("triggers", [])
        return rows

    async def _only_repeating(self) -> bool:
        """Whether all the world has ahead is recurring time triggers' marks: no trigger has a fire in flight (running,
        queued, waiting out a delay or hours, calling a contact, evaluating) and none has a one-off mark to come."""
        for row in await self._trigger_rows():
            if (row.get("status") in IN_FLIGHT or row.get("running") or row.get("pending") or row.get("waits")
                    or row.get("waiting") or row.get("executor_calls") or row.get("evaluating")):
                return False
            when = row.get("when") or {}
            if row.get("type") == "time" and row.get("status") == "armed" and "every" not in when \
                    and (row.get("next_mark") or "after" not in when):
                return False
        return True

    async def _repeating_ids(self) -> list[str]:
        return sorted({row["id"] for row in await self._trigger_rows() if row.get("type") == "time"
                       and row.get("status") == "armed" and "every" in (row.get("when") or {})})

    async def _notes_state(self, to: str) -> list[dict]:
        out = []
        for env in self.notification_envs:
            if env.supports(EXT_NOTIFICATIONS_URI, "state"):
                out.append(await _call(env, EXT_NOTIFICATIONS_URI, "state", {"to": to}))
        return out

    async def _due_count(self, to: str, now: datetime) -> int:
        n = 0
        for st in await self._notes_state(to):
            n += sum(1 for c in st.get("checks", []) if c.get("status") == "active" and _utc(c["next"]) <= now)
            n += sum(1 for p in st.get("pending_notifications", []) if not p.get("at") or _utc(p["at"]) <= now)
        return n

    async def _next_due(self, to: str, now: datetime) -> Optional[datetime]:
        """When the next check or notification for ``to`` comes due after ``now``."""
        times = []
        for st in await self._notes_state(to):
            times += [_utc(c["next"]) for c in st.get("checks", []) if c.get("status") == "active"]
            times += [_utc(p["at"]) for p in st.get("pending_notifications", []) if p.get("at")]
        return min((t for t in times if t > now), default=None)

    # ---- the user's schedule ---------------------------------------------------------------------------------
    def _schedule(self) -> Optional[dict]:
        return self.us.get("schedule") or None

    def _available(self, t: datetime) -> tuple[datetime, str]:
        """When the user would first look at a message that arrived at ``t``, and in what mode (morning, work, free):
        while awake and not busy. Sleep may run past midnight (wake 08:00, sleep 00:30). No schedule: at once, mode
        "now"."""
        sched = self._schedule()
        if not sched:
            return t, "now"
        tz = self.tz
        wake, sleep = _hm(sched.get("wake", "07:30")), _hm(sched.get("sleep", "23:30"))
        work = [_hm(x) for x in sched.get("work", ["09:00", "18:00"])]
        work_days = set(sched.get("work_days", DAYS[:5]))
        busy = [(datetime.fromisoformat(b["start"]).replace(tzinfo=tz), datetime.fromisoformat(b["end"]).replace(tzinfo=tz))
                for b in sched.get("busy") or []]
        lt = t.astimezone(tz)
        mode = None
        # Each pass moves past a busy block or a night, or ends: a block once passed is behind, and a night always ends
        # at a waking time; so the passes are bounded by two per block and one more.
        for _ in range(2 * len(busy) + 2):
            moved = False
            for start, stop in busy:
                if start <= lt < stop:
                    lt, moved = stop, True
            hm = (lt.hour, lt.minute)
            if _asleep(hm, wake, sleep):
                day = lt.date() if hm < wake else lt.date() + timedelta(days=1)
                lt, mode, moved = datetime(day.year, day.month, day.day, *wake, tzinfo=tz), "morning", True
            if not moved:
                break
        else:
            raise RuntimeError(f"always_on: the user's schedule never frees a time to read after {_iso(t)}")
        if mode is None:
            hm = (lt.hour, lt.minute)
            mode = "work" if DAYS[lt.weekday()] in work_days and work[0] <= hm < work[1] else "free"
        return lt.astimezone(timezone.utc), mode

    def _latency(self, mode: str) -> timedelta:
        rt = self.us.get("response_times") or {}
        if mode == "now" or not rt:
            return timedelta(0)
        return timedelta(minutes=float(rt.get(mode, rt.get("free", 0))))

    async def _schedule_read(self) -> None:
        if not self.unread or self.read_at is not None or self.gave_up or self.signed_off:
            return
        # A message has no time when the chat env's clock is unarmed; the task's time (another env's clock, else the
        # computer's) stands in for it.
        first_at = min(_utc(m["at"]) for m in self.unread if m.get("at")) if any(m.get("at") for m in self.unread) \
            else await self.now()
        at, mode = self._available(first_at)
        if self.last_reply_at and at < self.last_reply_at:
            at, mode = self._available(self.last_reply_at)
        self.read_at, self.read_mode = at, mode
        self.log("user_read_scheduled", read_at=_iso(at), mode=mode, unread=len(self.unread))

    # ---- the loop -------------------------------------------------------------------------------------------
    async def loop(self, now: datetime) -> None:
        """Until the run ends: a read that is due, else a wake for what is due, else a clock jump to what comes next.

        A world that only repeats: with no end_time, when nothing the step tracks is ahead (no alert or check for the
        agent, no user line, no held reply, no read) and no trigger has work in flight or a one-off mark to come, the
        world can only go on ticking: its recurring triggers fire again and again. Each such jump after which nothing
        woke the agent counts; ``IDLE_JUMPS_MAX`` of them in a row end the run (``nothing_can_wake``, which ``finish``
        turns into finished_nothing_pending or stuck). A wake or a read starts the count again. The rule only ever ends
        a run in which a recurring trigger has fired that many times running without waking the agent, and it can't
        end one that waits on a known event; with an end_time the run goes on to it, its jumps folded in the timeline
        past ``JUMPS_KEPT_PER_GAP`` per gap."""
        step = self.step
        await self._capped(self._open(), "the opening wake")
        while self.end_reason is None:
            if self.wakes >= step.max_wakes:
                self.end_reason = "max_wakes"
                break
            if not await self._capped(self._next(), "a turn of the loop"):
                break
        await self._capped(self._last_read(), "the last read", step.read_cap_seconds() + GATEWAY_ALLOWANCE_SECONDS)

    async def _capped(self, work, what: str, seconds: Optional[float] = None):
        """``work``, cut off past ``seconds`` (one turn of the loop's worst case, ``wake_reserve_seconds``): every part
        is capped on its own, so only calls hanging past their allowance get here, and the run ends on a step error
        rather than run past the hub's job."""
        seconds = self.step.wake_reserve_seconds(self.contact_seconds) if seconds is None else seconds
        timer = asyncio.timeout(seconds)
        try:
            async with timer:
                return await work
        except TimeoutError:
            if not timer.expired():
                raise
            self.end_reason = "step_error"
            raise RuntimeError(f"always_on: {what} ran past its worst case ({seconds:.0f}s, wake_reserve_seconds)")

    async def _open(self) -> None:
        """The opening wake: the user's request, with what came due for the agent."""
        self.you_sent.append(dict(self.opening))
        opened_at = _utc(self.opening["at"])
        await self._chat_user(self._opening_parts(lambda text: self._chat_part(opened_at, text, kind="opening")))
        due = await take_due(self.notification_envs, self.recipient) if self.notification_envs else []
        await self.wake([*delivered_parts(due),
                         *self._opening_parts(lambda text: self._user_part(opened_at, text, opening=True))],
                        woke_by=[f"notification:{d['channel']}" for d in due] + ["user_request"],
                        notifications=[{k: d.get(k) for k in ("at", "channel", "source", "source_id", "to")}
                                       for d in due])

    async def _next(self) -> bool:
        """One turn of the loop: the closing, a read, a wake or a clock jump; False once the run should end."""
        step = self.step
        now = await self.now()
        spent = self.spent()
        reserve = step.wake_reserve_seconds(self.contact_seconds)
        if spent + reserve >= self.time_budget:
            self.log("time_budget_reached", virtual_time=_iso(now), real_seconds=round(spent, 1),
                     time_budget_seconds=self.time_budget, wake_reserve_seconds=reserve)
            await self._final(now, reason="time_budget")
            return False
        # 1. a read that is due: the user sim reads what the agent sent (one due only at or past end_time is the last
        # read, after the final check-in)
        if self.read_at is not None and self.read_at <= now and self.unread \
                and (self.end is None or self.read_at < self.end):
            if self.read_at < now:  # her time to read passed during the agent's turn: she reads now if she's free
                at, self.read_mode = self._available(now)
                if at > now:  # asleep or busy by now: she reads when she's next free
                    self.read_at = at
                    self.log("user_read_scheduled", read_at=_iso(at), mode=self.read_mode, unread=len(self.unread))
                    return True
            await self.read(now)
            return True
        # past end_time (the clock runs during the agent's turns, so one can end past it): the final check-in next, with
        # whatever else is due, rather than ordinary wakes that would use up max_wakes without one
        if self.end is not None and now >= self.end:
            await self._final(now)
            return False
        # 2. anything due for the agent: notifications, scripted user lines, held replies; and what else comes
        # due in the same minute, so it all arrives in one wake
        if any(h["deliver_at"] <= now for h in self.held) or await self._due_count(self.recipient, now) \
                or await self._due_count(USER_SAYS, now):
            later = await self._later(now)
            same_minute = [t for t in later.values() if t < now.replace(second=0, microsecond=0)
                           + timedelta(minutes=1) and (self.end is None or t < self.end)]
            if same_minute and await self._armed():
                await self.advance(min(same_minute), reason="same_minute")
                return True
            lines = await self._take_user_says()
            replies = [h for h in self.held if h["deliver_at"] <= now]
            self.held = [h for h in self.held if h["deliver_at"] > now]
            await self._wake_with(now, lines, replies)
            return True
        # 3. nothing due: jump to the next thing
        later = await self._later(now)
        soonest = min(later.values()) if later else None
        if self.end is not None and (soonest is None or soonest >= self.end):
            await self._final(now)
            return False
        if soonest is None:
            self.end_reason = NOTHING_CAN_WAKE  # finished or stuck: finish() decides from what is pending
            return False
        if self.end is None and set(later) == {"world_event"} and await self._only_repeating():
            self.idle_jumps += 1
            self.idle_since = self.idle_since or _iso(now)
            if self.idle_jumps > IDLE_JUMPS_MAX:
                self.log("world_only_repeats", virtual_time=_iso(now), jumps=self.idle_jumps - 1,
                         since=self.idle_since, triggers=await self._repeating_ids())
                self.end_reason = NOTHING_CAN_WAKE
                return False
        else:
            self.idle_jumps, self.idle_since = 0, None
        why = sorted(k for k, v in later.items() if v == soonest)
        await self.advance(soonest, reason="+".join(why))
        return True

    async def _later(self, now: datetime) -> dict[str, datetime]:
        """What could wake the agent after ``now``, by kind, each at its earliest time."""
        candidates = {"world_event": await self._next_world_event(),
                      "agent_notification": await self._next_due(self.recipient, now),
                      "user_says": await self._next_due(USER_SAYS, now),
                      "held_reply": min((h["deliver_at"] for h in self.held), default=None),
                      "user_read": self.read_at if self.unread else None}
        return {k: v for k, v in candidates.items() if v is not None and v > now}

    async def _final(self, now: datetime, reason: str = "end_time") -> None:
        """The final check-in: at end_time, the user's ``final`` line, else ``FINAL_DEFAULT_TEXT`` (a line from the
        step, not the user); with whatever else is due, and the last-turn notice. A reply held past now stays held,
        to be recorded as pending."""
        step = self.step
        if reason == "end_time" and self.end > now and await self._armed():  # with no clock, the check-in is now
            await self.advance(self.end, reason="end_time")
        now = await self.now()
        self.final_sent = True
        lines = await self._take_user_says()
        replies = [h for h in self.held if h["deliver_at"] <= now]
        self.held = [h for h in self.held if h["deliver_at"] > now]
        notes = []
        if reason == "end_time" and step.final:
            lines.append({"at": _iso(now), "text": step.final, "source": "final"})
        else:
            notes.append(_step_part(FINAL_DEFAULT_TEXT.format(first=self.person["first"]), "final"))
        await self._wake_with(now, lines, replies, last=True, notes=notes)
        self.end_reason = self.end_reason or reason

    async def _last_read(self) -> None:
        """After the agent's last turn, the user sim reads what it sent since her last read, at the time she would
        (the world's clock stays where the run ended), so her reaction to its wrap-up is recorded; a reply she gives
        is held and recorded as pending, never delivered. Not after done ends a one-sitting run, after she gave up, or
        when the run ended on an error."""
        if not self.unread or self.signed_off or self.gave_up or self.end_reason in ERROR_ENDS:
            return
        await self._schedule_read()
        now = await self.now()
        at = self.read_at or now
        if at < now:  # she'd have looked during the last turn, which went on sending: she reads it all, from now
            at, self.read_mode = self._available(now)
        await self.read(at, last=True)

    def _prompt_parts(self) -> list[dict]:
        """The user's first message, the user sim's opening, as A2A parts, its seed applied."""
        return self.step._apply_seed([{"kind": "text", "text": self.us["opening"]}], self.ctx.metadata.get("seed", {}))

    def _effective_prompt(self) -> str:
        return "\n".join(p.get("text", "") for p in self._prompt_parts() if p.get("kind") == "text")

    def _opening_parts(self, labelled) -> list[dict]:
        """The prompt as the opening wake and the user's chat carry it: its text as one labelled line from the user
        (``labelled(text)``), where its first text part was, and every other part (a file, data) as it is, in the
        prompt's order, as prompt_agent sends them."""
        out, text = [], self._effective_prompt()
        for part in self._prompt_parts():
            if part.get("kind") != "text":
                out.append(part)
            elif text is not None:
                out.append(labelled(text))
                text = None
        return out

    def _user_part(self, at: datetime, text: str, opening: bool = False) -> dict:
        first = self.person["first"]
        body = re.sub(rf"^\s*{re.escape(first)}\s*:\s*", "", text)
        return {"kind": "text", "text": f"[{label_time(_iso(at))}] {first}: {body}",
                "metadata": {"agentenv": {"source": "user", "at": _iso(at), "opening": opening}}}

    # ---- the graded conversation: the user's chat ----------------------------------------------------------------
    # What graders read as the run's conversation (metadata["a2a_conversations"][step id]) is what the user saw: her
    # own messages as user turns (the opening request, user_says and final lines, her replies), and the agent's
    # message_user texts, each with its pretend time, as one agent message per run of them. Never the agent's closing
    # text, the alerts that woke it, or the step's own lines; those are in the per-wake prompt parts and the timeline.
    # It ends with her last word that never reached the agent, and a marker saying how the run ended.

    def _chat_part(self, at: datetime, text: str, **meta: Any) -> dict:
        part = self._user_part(at, text)
        part["metadata"]["agentenv"].update(meta)
        return part

    async def _chat_flush(self) -> None:
        """Write the agent's message_user texts not yet in the conversation, as one agent message."""
        if self.chat_pending:
            parts, self.chat_pending = self.chat_pending, []
            conversation_store.complete_a2a_task(conversation_id=self.conversation_id, parts=parts, role="agent",
                                                 **(await self.step._env_clock(self.clock_envs)))

    async def _chat_user(self, parts: list[dict]) -> None:
        """Write a user turn: what she said, after what the agent sent before it."""
        await self._chat_flush()
        conversation_store.add_a2a_task(conversation_id=self.conversation_id, parts=parts,
                                        a2a_task_id=uuid.uuid4().hex, role="user",
                                        **(await self.step._env_clock(self.clock_envs)))

    async def _chat_end(self, now: datetime) -> None:
        """Close the conversation: the agent's last messages, then one last user turn with her replies that never
        reached the agent, her reaction to its last messages when she said nothing, and the end marker."""
        first = self.person["first"]
        parts = [self._chat_part(h["deliver_at"], f"{h['text']} (sent after the assistant's last turn; it never saw "
                                 "this)", kind="user_reply", delivered=False, **({"done": True} if h.get("done") else {}))
                 for h in self.held]
        if self.reaction:
            parts.append(_step_part(f"({first} {self.reaction['said']})", "user_reaction"))
            parts[-1]["metadata"]["agentenv"].update(outcome=self.reaction["outcome"], at=self.reaction["at"])
        parts.append({"kind": "text", "text": f"[End of run · {label_time(_iso(now))}] {self.end_note} (A note from "
                                              f"the run, not a message from {first}.)",
                      "metadata": {"agentenv": {"source": "end_marker", "end_reason": self.end_reason,
                                                "at": _iso(now)}}})
        await self._chat_user(parts)

    async def _take_user_says(self) -> list[dict]:
        items = await take_due(self.notification_envs, USER_SAYS) if self.notification_envs else []
        out = []
        for it in items:
            text = "\n\n".join(p["text"] for p in it["parts"] if p.get("kind") == "text")
            out.append({"at": it.get("at"), "text": text, "source": "user_says", "trigger_id": it.get("source_id")})
        return out

    async def _wake_with(self, now: datetime, lines: list[dict], replies: list[dict], last: bool = False,
                         notes: Optional[list[dict]] = None) -> None:
        """Wake the agent with what came due for it, then the user's lines and replies in time order, then ``notes``
        (the step's own lines); ``last`` marks the run's last wake."""
        due = await take_due(self.notification_envs, self.recipient) if self.notification_envs else []
        parts = delivered_parts(due)
        user_msgs = sorted([{"at": l["at"] or _iso(now), "text": l["text"], "kind": l.get("source", "user_says"),
                             "trigger_id": l.get("trigger_id")} for l in lines]
                           + [{"at": _iso(r["deliver_at"]), "text": r["text"], "kind": "user_reply",
                               "done": r.get("done", False)} for r in replies],
                           key=lambda m: _utc(m["at"]))
        for r in replies:
            if r.get("hook") is not None:  # it reaches the agent now: the reply time the old psi-wake hook recorded
                r["hook"].update(delivered=True, user_reads_at=_iso(r["deliver_at"]), delivered_in_wake=self.wakes + 1)
        if user_msgs:
            await self._chat_user([self._chat_part(_utc(m["at"]), m["text"], kind=m["kind"],
                                                   **({"done": True} if m.get("done") else {})) for m in user_msgs])
        for m in user_msgs:
            parts.append(self._user_part(_utc(m["at"]), m["text"]))
            if m["kind"] != "user_reply":
                self.you_sent.append({"at": m["at"], "text": m["text"]})
                self.log("scripted_user_line", virtual_time=m["at"], text=m["text"], kind=m["kind"],
                         trigger_id=m.get("trigger_id"))
            else:
                self.log("user_reply_delivered", virtual_time=m["at"], text=m["text"], done=m["done"])
        parts += notes or []
        woke_by = [f"notification:{d['channel']}" for d in due] + [f"{m['kind']}" for m in user_msgs]
        woke_by += [f"step:{n['metadata']['agentenv']['source']}" for n in notes or []]
        await self.wake(parts, woke_by=woke_by, notifications=[{k: d.get(k) for k in
                                                                ("at", "channel", "source", "source_id", "to")}
                                                               for d in due], last=last)
        if self.signed_off and any(m.get("done") for m in user_msgs):
            self.end_reason = self.end_reason or "user_done"

    async def wake(self, parts: list[dict], woke_by: list[str], notifications: Optional[list] = None,
                   last: bool = False) -> None:
        """One wake: the agent's turn on ``parts``, then what it sent the user, then the settle wait. The wake that
        ``last`` marks, or the one ``max_wakes`` makes the last, carries the last-turn notice (``final_reminder``)."""
        step = self.step
        sent_at = await self.now()
        if self.wakes == 0 and not self.rules_in_system_prompt:  # no system prompt took them: the first wake does
            parts = [_step_part(self.rules(), "rules"), *parts]
        last = last or self.wakes + 1 >= step.max_wakes
        if last and step.final_reminder:
            parts.append(_step_part(step.final_reminder.format(first=self.person["first"]), "final_reminder"))
        if step.wake_reminder:
            parts = [*parts, _step_part(step.wake_reminder_text, "reminder")]
        self.wakes += 1
        self.idle_jumps, self.idle_since = 0, None
        self.jumps_in_gap, self.folded = 0, None
        msg_id = uuid.uuid4().hex
        await self._read_changes("before_turn")

        def record() -> None:
            self.per_turn_parts.append(self._kept_parts(parts))

        self.log("wake", n=self.wakes, virtual_time=_iso(sent_at), woke_by=woke_by, notifications=notifications or [],
                 last=last, parts_text="\n".join(p.get("text", "") for p in parts if p.get("kind") == "text"))
        t = time.time()
        try:
            task_id, result = await self._agent_turn(parts, msg_id, record)
        except TimeoutError as e:
            self._failed("agent_turn_timeout", f"agent turn timed out after {step.timeout_seconds}s: {e}")
            self.log("agent_turn_end", n=self.wakes, state="timeout", real_seconds=round(time.time() - t, 1))
            return
        state = result["status"]["state"]
        status_msg = (result.get("status") or {}).get("message") or {}
        term = protocol.TerminalResponse.from_message(status_msg)
        if term.tool_call_count is not None:
            self.tool_calls = (self.tool_calls or 0) + term.tool_call_count
        messages, quiet = await self._take_messages()
        await self._read_changes("turn")
        traj = None
        card = self.agent.a2a_card or {}
        traj_ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if traj_ext:
            try:
                traj = await asyncio.wait_for(
                    step._fetch_trajectory(self.target_url, traj_ext, task_id, step._trajectory_prefix(), msg_id,
                                           card=card, sandbox_type=self.agent.sandbox_type), TRAJECTORY_FETCH_SECONDS)
            except Exception as e:
                logger.warning(f"wake {self.wakes}: trajectory fetch failed (continuing): {e}")
        self.per_turn_trajectory_urls.append(traj)
        self.final_task_id = task_id
        ended = await self.now()
        plain = term.response_text or ""
        extra = {"stay_quiet_calls": len(quiet)} if step.stay_quiet else {}
        self.log("agent_turn_end", n=self.wakes, state=str(state), virtual_time=_iso(ended),
                 virtual_seconds=round((ended - sent_at).total_seconds()), real_seconds=round(time.time() - t, 1),
                 tool_calls=term.tool_call_count, plain_reply=plain, messages_to_user=len(messages),
                 trajectory_url=traj, **extra)
        self._sent(messages)
        if not _completed(state):  # failed, canceled or rejected: the turn didn't run to its end
            self._failed(term.error_type or f"agent_turn_{_state_name(state)}",
                         term.error_message or term.response_text or f"agent turn {_state_name(state)}",
                         term.error_code)
            return
        if step.nudge and not messages and not quiet and _closing_text(plain):
            await self._nudge(plain)
        stuck = await self.settle()
        if stuck:
            self.log("settle_timeout", wake=self.wakes, still_at_work=stuck, settle=self.last_settle)
        await self._read_changes("after_turn")
        await self._schedule_read()

    def _kept_parts(self, parts: list[dict]) -> list[dict]:
        """What the prompt response keeps of one wake's parts: TURN_TEXT_CHARS of each text, until
        TURN_TEXT_TOTAL_CHARS in all; then a note. Other parts (files, data) as they are."""
        kept, used = [], sum(len(p.get("text") or "") for turn in self.per_turn_parts for p in turn or [])
        for part in parts:
            if part.get("kind") != "text":
                kept.append(part)
                continue
            text = part.get("text") or ""
            room = min(TURN_TEXT_CHARS, TURN_TEXT_TOTAL_CHARS - used)
            if room <= 0:
                kept.append({"kind": "text", "text": "(cut: the run's stored timeline has this wake's full text)"})
                break
            if len(text) > room:
                part = {**part, "text": text[:room] + "… (cut)"}
            used += len(part["text"])
            kept.append(part)
        return kept

    async def _nudge(self, closing: str) -> None:
        """The wake ended with closing text and neither message_user nor stay_quiet, so give the agent one more short
        turn with ``nudge_text``; record what the closing text was and what it did after."""
        step = self.step
        self.nudges += 1
        parts = [_step_part(step.nudge_text, "nudge")]
        sent_at = await self.now()
        msg_id = uuid.uuid4().hex

        def record() -> None:
            self.per_turn_parts.append(self._kept_parts(parts))

        t = time.time()
        try:
            task_id, result = await self._agent_turn(parts, msg_id, record)
        except TimeoutError as e:
            self._failed("agent_turn_timeout", f"agent nudge turn timed out after {step.timeout_seconds}s: {e}")
            self.log("nudge", n=self.nudges, wake=self.wakes, virtual_time=_iso(sent_at), closing_text=closing,
                     state="timeout", outcome="timeout", real_seconds=round(time.time() - t, 1))
            return
        state = result["status"]["state"]
        status_msg = (result.get("status") or {}).get("message") or {}
        term = protocol.TerminalResponse.from_message(status_msg)
        if term.tool_call_count is not None:
            self.tool_calls = (self.tool_calls or 0) + term.tool_call_count
        messages, quiet = await self._take_messages()
        await self._read_changes("nudge")
        traj = None
        card = self.agent.a2a_card or {}
        traj_ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if traj_ext:
            try:
                traj = await asyncio.wait_for(
                    step._fetch_trajectory(self.target_url, traj_ext, task_id, step._trajectory_prefix(), msg_id,
                                           card=card, sandbox_type=self.agent.sandbox_type), TRAJECTORY_FETCH_SECONDS)
            except Exception as e:
                logger.warning(f"nudge after wake {self.wakes}: trajectory fetch failed (continuing): {e}")
        self.per_turn_trajectory_urls.append(traj)
        self.final_task_id = task_id
        plain = term.response_text or ""
        outcome = "message_user" if messages else "stay_quiet" if quiet else "neither"
        self.log("nudge", n=self.nudges, wake=self.wakes, virtual_time=_iso(sent_at), closing_text=closing,
                 state=str(state), outcome=outcome, messages_to_user=len(messages), stay_quiet_calls=len(quiet),
                 plain_reply_after=plain, tool_calls=term.tool_call_count, real_seconds=round(time.time() - t, 1),
                 trajectory_url=traj)
        self._sent(messages, after_nudge=True)
        if not _completed(state):
            self._failed(term.error_type or f"agent_turn_{_state_name(state)}",
                         term.error_message or term.response_text or f"agent nudge turn {_state_name(state)}",
                         term.error_code)

    # ---- app changes -------------------------------------------------------------------------------------------------
    async def _start_changes(self) -> None:
        """Where each env's changelog stands before the first wake (urn:agentenv:state/v1's changelog_id), so the
        run reads only its own changes; an env that keeps no changelog is skipped."""
        for env in self.ctx.deployed_envs:
            if not (env.supports(EXT_CHANGES_URI, "get") and env.supports(EXT_STATE_URI, "get")):
                continue
            try:
                changelog_id = (await _call(env, EXT_STATE_URI, "get", attempts=2)).get("changelog_id")
                if changelog_id is None:  # an empty changelog, or none: only the second refuses a read
                    await _call(env, EXT_CHANGES_URI, "get", {"after_id": 0, "limit": 1}, attempts=1)
            except Exception as e:
                logger.info(f"always_on: {env.env_id} keeps no changelog it serves; its changes go unrecorded: {e}")
                continue
            self.change_cursors[env.env_id] = (env, int(changelog_id or 0))

    async def _read_changes(self, phase: str) -> None:
        """Record the app changes made since the last read (urn:agentenv:changes/v1), each with who made it. One the
        agent made (a tool call or website request of its role) carries the wake that made it (``wake``, with
        ``phase`` "turn" or "nudge"); others say when they were seen: before the wake's turn (clock jumps), during it,
        or after it (contacts and triggers while the world settled). A change with no actor is one the gateway didn't
        serve, a website reached directly say."""
        for env_id, (env, cursor) in list(self.change_cursors.items()):
            rows: list[dict] = []
            try:
                while True:
                    got = await _call(env, EXT_CHANGES_URI, "get", {"after_id": cursor, "limit": CHANGES_PAGE_MAX},
                                      attempts=2)
                    page = got.get("changes") or []
                    rows += page
                    if page:
                        cursor = page[-1]["id"]
                    if not page or not got.get("has_more"):
                        break
            except Exception as e:
                logger.warning(f"always_on: reading {env_id}'s changes failed (continuing): {e}")
            self.change_cursors[env_id] = (env, cursor)
            if rows:
                changes = [self._change(env_id, row, phase) for row in rows]
                more = len(changes) - CHANGES_KEPT_PER_READ
                self.log("app_changes", wake=self.wakes, phase=phase, env_id=env_id, count=len(changes),
                         changes=changes[:CHANGES_KEPT_PER_READ], **({"not_kept": more} if more > 0 else {}))

    def _change(self, env_id: str, row: dict, phase: str) -> dict:
        actors = [{k: v for k, v in a.items() if k in ("kind", "role", "tool", "trigger_id", "request", "caused_by",
                                                         "failed") and v not in (None, [], {})}
                  for a in row.get("actors") or []]
        mine = any(a.get("role") == self.recipient and a.get("kind") in ("tool_call", "request") for a in actors)
        by = "agent" if mine else "others" if actors else None
        self.change_counts["by_agent" if mine else "by_others" if actors else "no_actor"] += 1
        summary = row.get("summary")
        if isinstance(summary, str) and len(summary) > CHANGE_SUMMARY_CHARS:
            summary = summary[:CHANGE_SUMMARY_CHARS] + "…"
        return {"env_id": env_id, "id": row.get("id"), "table": f"{row.get('schema_name')}.{row.get('table_name')}",
                "op": row.get("operation"), "row_id": row.get("row_id"), "summary": summary,
                "at": row.get("timestamp"), "by": by, "wake": self.wakes, "phase": phase, "actors": actors}

    async def _take_messages(self) -> tuple[list[dict], list[dict]]:
        """What the agent sent the user with message_user this turn, oldest first, and its stay_quiet calls: its role's
        only, so another role's stay queued for whoever takes them."""
        messages, quiet = [], []
        for env in self.chat_envs:
            got = await _call(env, EXT_USER_CHAT_URI, "take", {"take_id": uuid.uuid4().hex, "from": self.recipient})
            messages += got.get("items") or []
            quiet += got.get("quiet") or []
        messages.sort(key=lambda m: m.get("at") or "")
        return messages, quiet

    def _sent(self, messages: list[dict], after_nudge: bool = False, leftover: bool = False) -> None:
        """Record what the agent sent the user: for her next read, the graded chat and the timeline. ``leftover``
        marks messages no turn took (taken at the end), which she never reads."""
        for m in messages:
            self.log("message_to_user", virtual_time=m.get("at"), text=m["text"], wake=self.wakes,
                     **({"after_nudge": True} if after_nudge else {}), **({"leftover": True} if leftover else {}))
            sent = {"at": m.get("at"), "text": m["text"], "wake": self.wakes}
            self.unread.append(sent)
            self.all_messages.append(sent)
            self.chat_pending.append({"kind": "text", "text": f"[{label_time(m.get('at'))}] {m['text']}",
                                      "metadata": {"agentenv": {"source": "message_user", "at": m.get("at"),
                                                                "wake": self.wakes}}})

    def _failed(self, error_type: str, error: str, error_code: Optional[str] = None) -> None:
        """An agent turn didn't complete (failed, canceled, rejected or timed out): the run ends, and its prompt
        response carries ``error_type`` the way prompt_agent's does, so verifiers skip it instead of grading it as a
        normal run."""
        self.error, self.error_type, self.error_code = error, error_type, error_code
        self.end_reason = "agent_turn_timeout" if error_type == "agent_turn_timeout" else "agent_turn_failed"

    async def read(self, now: datetime, last: bool = False) -> None:
        """The user sim reads what the agent sent since her last read, at ``now``; ``last`` marks the read after the
        run's last wake, whose answer can no longer change how the run ends."""
        phone = []
        if self.notification_envs:
            for it in await take_due(self.notification_envs, USER_PHONE):
                phone.append({"at": it.get("at"), "channel": it["channel"],
                              "text": "\n\n".join(p["text"] for p in it["parts"] if p.get("kind") == "text")})
        self.idle_jumps, self.idle_since = 0, None
        seen = list(self.unread)
        read = {"time": _iso(now), "assistant_messages": [{"at": m.get("at"), "text": m["text"]} for m in seen],
                "phone": phone,
                "you_sent": list(self.you_sent), "done_ends_run": not self.multi_day}
        self.unread, self.you_sent, self.read_at = [], [], None
        calls = []
        cap = self.step.read_cap_seconds()
        deadline = time.monotonic() + cap

        async def ask(text: str) -> dict:
            got = await self._ask_user_sim(text, deadline)
            calls.append(got)
            return got

        t = time.time()
        try:
            ans = await asyncio.wait_for(self.plugin.answer(self.version, self.person, self.history, read, ask), cap)
        except TimeoutError:  # the plug-in ran past the read's cap: a user sim error, as a failed ask is
            ans = {"outcome": "no_reply", "message": ""}
            calls.append({"text": None, "structured": None, "error": f"the read ran past its cap ({cap:.0f}s)"})
        self.history.append({"read": read, "answer": ans})
        mode = self.read_mode
        latency = self._latency(mode)
        flagged = (len(ans.get("message") or "") > ASSISTANT_LIKE_MESSAGE_CHARS
                   or bool(re.search(r"^\s*(#|\|)", ans.get("message") or "", re.M)))
        entry = self.log("user_sim_read", virtual_time=_iso(now), version=self.version, read=read,
                         input=ans.get("input"), raw=ans.get("raw"), outcome=ans.get("outcome"),
                         message=ans.get("message"), used=ans.get("used"), score=ans.get("score"),
                         schema_errors=ans.get("schema_errors"), mode=mode,
                         real_seconds=round(time.time() - t, 1), looks_like_assistant=flagged,
                         user_sim_error=(calls[-1].get("error") if calls else None),
                         user_sim_trajectory_urls=[c.get("trajectory_url") for c in calls],
                         user_sim_turns=ans.get("turns", len(calls)), shown=ans.get("shown"),
                         past_turn_cap=ans.get("past_turn_cap"), last_read=last,
                         retried=ans.get("retried"), first_answer=ans.get("first_answer"))
        outcome = ans.get("outcome")
        said = (ans.get("message") or "").strip()
        hook = {"hook": HOOK_NAME, "turn": self.wakes, "read_at": _iso(now),
                "wrote_at": min((m["at"] for m in seen if m.get("at")), default=None), "outcome": outcome}
        if outcome in ("reply", "done") and said:
            deliver = now + latency
            reply_to = _reply_to(ans, seen)
            entry.update(deliver_at=_iso(deliver), reply_to=reply_to, reply_to_candidates=len(seen))
            # Not delivered until a wake carries it, which sets user_reads_at; one never delivered (her last read's
            # reply, or one still held at the end) keeps delivered False and no user_reads_at, so a check can't count
            # it as received.
            hook.update(delivered=False, deliver_at=_iso(deliver), reply_to=_cut(reply_to))
            self.held.append({"deliver_at": deliver, "text": ans["message"], "done": outcome == "done",
                              "hook": hook})
            self.last_reply_at = deliver
        self.hook_records.append(hook)
        if outcome == "reply" and said:
            self.replies += 1
            max_replies = self.us.get("max_replies")
            if max_replies and self.replies >= int(max_replies):
                self.gave_up = True
                self.log("user_gave_up", replies=self.replies)
        if not said and outcome in ("no_reply", "done") and (last or (outcome == "done" and not self.multi_day)):
            self.reaction = {"outcome": outcome, "at": _iso(now),
                             "said": "read it and signed off without a message." if outcome == "done"
                             else "read it and didn't reply."}
        if outcome == "done" and not last:
            entry["ends_run"] = not self.multi_day
            if not self.multi_day and said:
                self.signed_off = True
            elif not self.multi_day:
                self.end_reason = "user_done"
        if calls and calls[-1].get("error") and not last:
            self.end_reason = self.end_reason or "user_sim_error"

    async def _ask_user_sim(self, text: str, deadline: float) -> dict:
        """One user sim turn on ``text``, cut off at its time limit and grace or at the read's ``deadline``
        (time.monotonic()), whichever comes first."""
        msg_id = uuid.uuid4().hex
        limit = min(self.step.user_timeout_seconds + TURN_GRACE_SECONDS, deadline - time.monotonic())
        if limit <= 0:
            return {"text": None, "structured": None, "error": "the read's time limit passed"}
        try:
            user_task_id, result = await asyncio.wait_for(send_and_wait(
                self.user_url, [{"kind": "text", "text": text}],
                agent=self.user_agent if self.user_agent.sandbox_id else None, message_id=msg_id,
                context_id=self.user_context_id, timeout_seconds=min(self.step.user_timeout_seconds, limit),
                poll_interval_seconds=self.step.poll_interval_seconds), limit)
        except TimeoutError as e:
            return {"text": None, "structured": None, "error": f"timeout: {e}"}
        status = (result.get("status") or {})
        if status.get("state") != TaskState.completed and str(status.get("state")) != "completed":
            return {"text": None, "structured": None, "error": f"state {status.get('state')}"}
        msg = status.get("message") or {}
        term = protocol.TerminalResponse.from_message(msg)
        raw = next((p["text"] for p in msg.get("parts") or [] if p.get("kind") == "text"), term.response_text)
        structured = term.structured_output
        if structured is None:
            try:
                structured = json.loads(raw)
            except (TypeError, ValueError):
                structured = None
        traj = None
        card = self.user_agent.a2a_card or {}
        traj_ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if traj_ext:
            try:
                traj = await asyncio.wait_for(
                    self.step._fetch_trajectory(self.user_url, traj_ext, user_task_id,
                                                self.step._trajectory_prefix() + "user_sim/", msg_id,
                                                card=card, sandbox_type=self.user_agent.sandbox_type),
                    max(0.0, min(TRAJECTORY_FETCH_SECONDS, deadline - time.monotonic())))
            except Exception as e:
                logger.warning(f"user sim trajectory fetch failed (continuing): {e}")
        return {"text": raw, "structured": structured if isinstance(structured, dict) else None,
                "trajectory_url": traj}

    # ---- the end --------------------------------------------------------------------------------------------
    async def _contacts_at_work(self) -> list[dict]:
        """By env, the triggers still at work now: contact replies in progress or waiting for a later time."""
        out = []
        for env in self.ctx.deployed_envs:
            if not env.supports(EXT_TRIGGERS_URI, "state"):
                continue
            try:
                state = await _call(env, EXT_TRIGGERS_URI, "state", attempts=2)
            except Exception as e:
                out.append({"env_id": env.env_id, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            rows = [_at_work(row) for row in state.get("triggers", [])
                    if row.get("status") in IN_FLIGHT or row.get("running") or row.get("executor_calls")
                    or row.get("pending")]
            if rows:
                out.append({"env_id": env.env_id, "triggers": rows})
        return out

    async def _pending_at_end(self) -> dict:
        """What was still pending when the run ended: her replies held for their response time, the agent's messages
        she never read, user lines and alerts for the agent never delivered, its scheduled checks not yet due, and
        contact replies still in progress."""
        alerts, checks = [], []
        for st in await self._notes_state(self.recipient):
            alerts += [{k: n.get(k) for k in ("at", "channel", "text", "trigger_id")}
                       for n in st.get("pending_notifications", [])]
            checks += [{k: c.get(k) for k in ("next", "note", "every")} for c in st.get("checks", [])
                       if c.get("status") == "active"]
        lines = [{k: n.get(k) for k in ("at", "text", "trigger_id")}
                 for st in await self._notes_state(USER_SAYS) for n in st.get("pending_notifications", [])]
        return {"held_replies": [{"at": _iso(h["deliver_at"]), "text": h["text"], "done": h.get("done", False)}
                                 for h in self.held],
                "unread_by_user": [dict(m) for m in self.unread],
                "user_lines_not_delivered": lines,
                "alerts_never_delivered": alerts,
                "scheduled_checks_not_due": checks,
                "contact_replies_in_progress": await self._contacts_at_work()}

    async def finish(self) -> None:
        step, ctx = self.step, self.ctx
        now = await self.now()
        try:  # what the agent sent and no turn took: a turn that timed out or failed mid-way
            leftover, _ = await asyncio.wait_for(self._take_messages(), LEFTOVER_TAKE_SECONDS)
            self._sent(leftover, leftover=True)
        except Exception as e:
            logger.warning(f"always_on: could not take the agent's last messages: {type(e).__name__}: {e}")
        for h in self.held:
            if h.get("hook") is not None:
                h["hook"]["undelivered"] = "the run ended before it reached the agent"
        await self._read_changes("end")
        try:
            self.pending = await self._pending_at_end()
        except Exception as e:  # the record matters more than this part of it
            logger.warning(f"always_on: could not read what was pending at the end: {e}")
            self.pending = {"error": [{"error": f"{type(e).__name__}: {str(e)[:200]}"}]}
        if self.end_reason == NOTHING_CAN_WAKE:
            # Scheduled checks would have woken the agent, so they aren't what makes a run stuck.
            stuck = any(items for kind, items in self.pending.items() if kind != "scheduled_checks_not_due")
            self.end_reason = "stuck" if stuck else "finished_nothing_pending"
        self.end_note = END_NOTES.get(self.end_reason, self.end_reason or "").format(first=self.person["first"],
                                                                                     wakes=self.wakes)
        if self.end_reason in ERROR_ENDS and not self.error_type:
            self.error_type = self.end_reason
        self.log("end", reason=self.end_reason, note=self.end_note, virtual_time=_iso(now), wakes=self.wakes,
                 replies=self.replies, unread_at_end=len(self.unread), held_at_end=len(self.held),
                 pending=self.pending)
        try:
            await self._chat_end(now)
        except Exception as e:
            logger.warning(f"always_on: could not close the graded conversation: {e}")
        conv = conversation_store.get_conversation(self.conversation_id)
        if conv and conv.get("status") == "active":
            conversation_store.mark_closed(self.conversation_id)
        await step._persist_env_trigger_state(ctx)
        counts = {
            "wakes": self.wakes,
            "messages_to_user": len(self.all_messages),
            "user_sim_reads": sum(1 for e in self.timeline if e["event"] == "user_sim_read"),
            "user_replies": self.replies,
            "user_no_replies": sum(1 for e in self.timeline if e["event"] == "user_sim_read"
                                   and e.get("outcome") == "no_reply"),
            "user_dones": sum(1 for e in self.timeline if e["event"] == "user_sim_read" and e.get("outcome") == "done"),
            "scripted_user_lines": sum(1 for e in self.timeline if e["event"] == "scripted_user_line"),
            "clock_jumps": self.clock_jumps,
            "silent_wakes": sum(1 for e in self.timeline if e["event"] == "agent_turn_end"
                                and not e.get("messages_to_user")),
            # reads whose answer didn't match the user sim's answer format (the plug-in's schema_errors), counting a
            # read the plug-in retried after a malformed first answer even when the retry came back well formed
            "user_sim_malformed_answers": sum(1 for e in self.timeline if e["event"] == "user_sim_read"
                                              and (e.get("schema_errors") or e.get("retried"))),
        }
        if self.change_cursors:
            counts["app_changes"] = dict(self.change_counts)
        if step.nudge:
            nudges = [e for e in self.timeline if e["event"] == "nudge"]
            counts["nudges"] = len(nudges)
            counts["nudge_outcomes"] = {o: sum(1 for e in nudges if e.get("outcome") == o)
                                        for o in sorted({e.get("outcome") for e in nudges})}
        if step.stay_quiet:
            counts["stay_quiet_wakes"] = sum(1 for e in self.timeline if e["event"] == "agent_turn_end"
                                             and e.get("stay_quiet_calls"))
        record = {"step_id": step.id, "end_reason": self.end_reason, "end_note": self.end_note, "error": self.error,
                  "error_type": self.error_type, "counts": counts,
                  "user_sim": {"name": step.user_sim, "version": self.version,
                               "agent_name": self.us.get("agent_name"), **self.user_setup},
                  "end_time": _iso(self.end) if self.end else None, "time_zone": self.zone,
                  "multi_day": self.multi_day, "real_seconds": round(time.time() - self.t0_real, 1),
                  "timeline_events": len(self.timeline), "pending_at_end": _bounded(self.pending)}
        record["settings"] = {"stay_quiet": step.stay_quiet, "nudge": step.nudge, "wake_reminder": step.wake_reminder,
                              "clock": self.clock,
                              "rules_via": "system_prompt" if self.rules_in_system_prompt else "first_wake",
                              "scheduled_checks": self.checks, "contacts": self.contacts,
                              "time_budget_seconds": self.time_budget,
                              "wake_reserve_seconds": step.wake_reserve_seconds(self.contact_seconds)}
        await self._store_timeline(record)
        ctx.metadata.setdefault(ALWAYS_ON_KEY, {})[step.id] = record
        # The graded response is the user's whole chat from the agent, each message with its pretend time; never its
        # closing text, which she never saw.
        self._append_prompt_response("\n\n".join(f"[{label_time(m.get('at'))}] {m['text']}" for m in self.all_messages))

    async def _store_timeline(self, record: dict) -> None:
        """Store the run's full record, its timeline included, and give ``record`` (what stays in the run's metadata,
        kept small for the job's report) its ``object_url``. When it can't be stored, ``record`` keeps the last
        ``TIMELINE_TAIL`` events instead, their bulky fields dropped."""
        step, ctx = self.step, self.ctx
        try:
            store = get_config().get_object_store()
            key = (f"{get_config().get_artifact_key_prefix()}always_on_timeline/"
                   f"instance_id={ctx.instance_id or 'adhoc'}/{step.id}-{uuid.uuid4().hex[:8]}.json")
            full = {**record, "timeline": self.timeline}
            record["object_url"] = await asyncio.to_thread(store.put, key,
                                                           json.dumps(full, indent=1, default=str).encode(),
                                                           content_type="application/json")
        except Exception as e:
            logger.warning(f"always_on: could not store the timeline: {e}")
            record["timeline_store_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            record["timeline_tail"] = [_small({k: v for k, v in event.items() if k not in BULKY_EVENT_FIELDS})
                                       for event in self.timeline[-TIMELINE_TAIL:]]

    def _append_prompt_response(self, final_text: str) -> None:
        step, ctx = self.step, self.ctx
        traj = next((u for u in reversed(self.per_turn_trajectory_urls) if u), None)
        ctx.prompt_responses.append(PromptResponse(
            prompt_id=step.prompt_id, response=final_text, prompt_text=self._effective_prompt(),
            agent_trajectory_object_url=traj, agent_trajectory_object_prefix=step._trajectory_prefix(),
            target_agent_per_turn_trajectory_object_urls=self.per_turn_trajectory_urls or None,
            source_agent_per_turn_prompt_parts=self.per_turn_parts or None, tool_call_count=self.tool_calls,
            model=self.model, error_type=self.error_type, error_code=self.error_code, error_message=self.error,
            agent_session_id=self.final_task_id,
            a2a_context_id=self.solver_context_id, agent_name=step.agent_name, step_id=step.id))


# ==================================================================================================================
# What a judge reads of the run beyond the user's chat (always_on_agent_verifier, for an always_on step's prompt).
JUDGE_CHANGES_PER_WAKE = 10
JUDGE_WAKES = 300
_WOKE_BY = {"user_request": "the user's request", "user_reply": "the user's reply", "user_says": "a user message",
            "final": "the user's final line", "step:final": "the end of the job"}


def render_for_judge(record: dict) -> str:
    """The run wake by wake for a judge: each wake's pretend time, what woke the agent, how much it sent the user and
    its tool calls, and the app changes made around it, the agent's own marked; then how the run ended and what was
    still pending. From the stored timeline at ``record["object_url"]`` (else the record's tail)."""
    timeline = record.get("timeline_tail") or []
    if record.get("object_url"):
        try:
            url = record["object_url"]
            timeline = json.loads(get_config().get_object_store_at(url).get(url))["timeline"]
        except Exception as e:
            return f"## The run, wake by wake\n(The run's record could not be read: {type(e).__name__}: {e})"
    turns = {e["n"]: e for e in timeline if e["event"] == "agent_turn_end"}
    nudges = {e["wake"]: e for e in timeline if e["event"] == "nudge"}
    changes: dict[int, list[dict]] = {}
    for e in timeline:
        if e["event"] == "app_changes":
            changes.setdefault(e["wake"], []).extend(e.get("changes") or [])
    lines = ["## The run, wake by wake",
             "The assistant ran only when something woke it. Each wake below gives its pretend time, what woke it, how "
             "many messages it sent the user (their text is in the conversation) and its tool calls, then the app "
             "changes made around it: the assistant's own (\"it changed\"), others' (contacts, triggers), and ones "
             "with no known actor (made outside the env's gateway, such as on a website reached directly), each with "
             "when it was seen. The user saw only the messages the assistant sent."]
    wakes = [e for e in timeline if e["event"] == "wake"]
    for wake in wakes[:JUDGE_WAKES]:
        n = wake["n"]
        woke = ", ".join(_WOKE_BY.get(w, f"an alert ({w.split(':', 1)[1]})" if w.startswith("notification:") else w)
                         for w in wake.get("woke_by") or []) or "nothing named"
        turn = turns.get(n) or {}
        sent = (turn.get("messages_to_user") or 0) + (nudges.get(n, {}).get("messages_to_user") or 0)
        calls = f" · {turn['tool_calls']} tool calls" if turn.get("tool_calls") is not None else ""
        state = f" · its turn {turn['state']}" if turn.get("state") not in (None, "completed", "TaskState.completed") \
            else ""
        lines.append(f"- Wake {n} · {label_time(wake.get('virtual_time'))} · woken by {woke} · sent the user "
                     f"{sent} message{'s' if sent != 1 else ''}{calls}{state}")
        made = changes.get(n, [])
        for c in made[:JUDGE_CHANGES_PER_WAKE]:
            if c.get("by") == "agent":
                who = "it changed"
            elif c.get("by"):
                who = f"{_actor(c)} changed ({c.get('phase')})"
            else:
                who = f"no known actor changed ({c.get('phase')})"
            summary = f" ({c['summary']})" if c.get("summary") else ""
            lines.append(f"  - {who}: {c.get('op')} {c.get('table')} row {c.get('row_id')}{summary}")
        if len(made) > JUDGE_CHANGES_PER_WAKE:
            lines.append(f"  - and {len(made) - JUDGE_CHANGES_PER_WAKE} more changes")
    if len(wakes) > JUDGE_WAKES:
        lines.append(f"(and {len(wakes) - JUDGE_WAKES} more wakes)")
    pending = {kind: v.get("count", 0) for kind, v in (record.get("pending_at_end") or {}).items()
               if isinstance(v, dict) and v.get("count")}
    lines += ["", "## How the run ended",
              f"{record.get('end_note') or ''} (end reason: {record.get('end_reason')})".strip()]
    if pending:
        lines.append("Still pending at the end: " + ", ".join(f"{k.replace('_', ' ')}: {v}"
                                                           for k, v in pending.items()))
    return "\n".join(lines)


def _actor(change: dict) -> str:
    actor = (change.get("actors") or [{}])[0]
    if actor.get("trigger_id"):
        return f"trigger {actor['trigger_id']}"
    if actor.get("caused_by"):
        return f"{actor.get('role')} (for trigger {actor['caused_by'][0].get('trigger_id')})"
    return str(actor.get("role") or actor.get("kind") or "someone")


# The longest wait between two tries of a gateway control call (_call backs off doubling, from 1 s, up to this).
CALL_RETRY_MAX_SLEEP_SECONDS = 20


async def _call(env, uri: str, method: str, params: Optional[dict] = None, *, attempts: int = 6,
                timeout: int = 60) -> Any:
    """A gateway control call, retried on a transient failure (a busy gateway can take seconds to answer).
    Every call the step retries is safe to repeat: reads, config, advance (checked against the clock after) and takes
    with a take_id."""
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await env.invoke(uri, method, params, timeout=timeout)
        except EnvCapabilityUnsupported:
            raise
        except Exception as e:
            last = e
            if attempt + 1 < attempts:
                logger.warning(f"always_on: {method} on {getattr(env, 'env_id', env)} failed ({type(e).__name__}: "
                               f"{e}); retry {attempt + 1}/{attempts - 1}")
                await asyncio.sleep(min(2 ** attempt, CALL_RETRY_MAX_SLEEP_SECONDS))
    raise last


def _task_zone(us: dict, clocks: list[dict]) -> str:
    return task_zone(us, clocks, "always_on")


def _step_part(text: str, source: str) -> dict:
    """A line the step adds to a wake, marked so a reader can tell it from what the user or the world sent."""
    return {"kind": "text", "text": text, "metadata": {"agentenv": {"source": source}}}


def _state_name(state: Any) -> str:
    """An A2A task state's name ("completed", "failed", "canceled", "rejected"), from the enum or its string."""
    return str(getattr(state, "value", state))


def _completed(state: Any) -> bool:
    return state == TaskState.completed or _state_name(state) == "completed"


def _closing_text(plain: str) -> str:
    text = plain.strip()
    return "" if text in EMPTY_REPLIES else text


def _busy(row: dict, now: datetime) -> bool:
    """Whether a trigger is still at work at ``now``: evaluating, calling an executor, or with a fire in flight that
    isn't waiting for a later time."""
    if row.get("evaluating") or row.get("executor_calls"):
        return True
    if "running" in row:  # a gateway that reports each fire in flight and when each waiting one goes on
        later = sum(1 for w in row.get("waits") or [] if w.get("resumes_at") and _utc(w["resumes_at"]) > now)
        return row["running"] > later
    if row.get("status") not in IN_FLIGHT and not row.get("pending"):
        return False
    return not (row.get("waiting") and row.get("resumes_at") and _utc(row["resumes_at"]) > now)


# Words she quotes name one of the agent's messages only when they run to at least this many characters, so a short
# quote ("ok") can't match one by chance.
QUOTE_MIN_CHARS = 8


def _reply_to(answer: dict, seen: list[dict]) -> Optional[dict]:
    """Which of the agent's messages in a read her answer responds to: the one the plug-in names in ``replying_to``
    (its 1-based number in the read, or words quoted from it); else the one her ``used`` quotes; else the only message
    in the read; else the only one that asks something. None when it can't be told. ``matched_by`` says which rule
    found it. Whether she approved what it asked is for the grader to read from her message."""
    def found(m: dict, how: str) -> dict:
        return {"wake": m.get("wake"), "at": m.get("at"), "text": m["text"], "matched_by": how}

    def quoting(quote: Any) -> list[dict]:
        words = " ".join(str(quote or "").split()).lower()
        return [m for m in seen if len(words) >= QUOTE_MIN_CHARS and words in " ".join(m["text"].split()).lower()]

    named = answer.get("replying_to")
    if isinstance(named, int) and not isinstance(named, bool) and 1 <= named <= len(seen):
        return found(seen[named - 1], "replying_to")
    if isinstance(named, str) and len(hits := quoting(named)) == 1:
        return found(hits[0], "replying_to")
    if isinstance(answer.get("used"), str) and len(hits := quoting(answer["used"])) == 1:
        return found(hits[0], "used")
    if len(seen) == 1:
        return found(seen[0], "only_message")
    if len(asks := [m for m in seen if "?" in m["text"]]) == 1:
        return found(asks[0], "only_question")
    return None


def _cut(item: Optional[dict]) -> Optional[dict]:
    """``item`` with its text cut to PENDING_TEXT_CHARS, for what the run's metadata keeps."""
    if not item or len(item.get("text") or "") <= PENDING_TEXT_CHARS:
        return item
    return {**item, "text": item["text"][:PENDING_TEXT_CHARS] + "…"}


def _bounded(pending: dict) -> dict:
    """``pending`` as the run's metadata keeps it: each list's count, and its first ``PENDING_KEPT`` items, kept
    small (``_small``) however deep; the stored timeline's end event has them all."""
    return {kind: {"count": len(items), "items": [_small(item) for item in items[:PENDING_KEPT]]}
            for kind, items in pending.items()}


def _small(value: Any, depth: int = 0) -> Any:
    """``value`` kept small for the run's metadata, at any depth: strings cut to ``PENDING_TEXT_CHARS``, lists to their
    first ``PENDING_KEPT`` items and a count of the rest, nesting past ``KEPT_NESTING_LEVELS`` levels summed up."""
    if isinstance(value, str):
        return value[:PENDING_TEXT_CHARS] + "…" if len(value) > PENDING_TEXT_CHARS else value
    if depth >= KEPT_NESTING_LEVELS and isinstance(value, (list, dict)):
        return f"({len(value)} {'items' if isinstance(value, list) else 'fields'})"
    if isinstance(value, dict):
        return {k: _small(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        kept = [_small(v, depth + 1) for v in value[:PENDING_KEPT]]
        return kept + [f"(and {len(value) - PENDING_KEPT} more)"] if len(value) > PENDING_KEPT else kept
    return value


# The timeline fields the process log may show as they are: names, ids, times, states and settings, never what a person
# or the agent wrote. Any other string is logged as its length, a list as its count, a dict as its keys' view.
_LOGGED_AS_IS = frozenset({
    "virtual_time", "from", "to", "reason", "state", "outcome", "mode", "read_at", "deliver_at", "end_time", "agent",
    "agent_role", "user_sim", "version", "user_model", "time_zone", "kind", "trigger_id", "env_id", "phase",
    "error_type", "matched_by", "at", "wake", "n", "trajectory_url", "object_url"})
# Past this length even one of those fields is logged as its length.
LOGGED_TEXT_CHARS = 200


def _log_view(fields: dict, depth: int = 0) -> dict:
    """``fields`` as the process log may show them: see ``_LOGGED_AS_IS``. Bodies and addresses are PII; the full
    content stays in the stored timeline."""
    out: dict = {}
    for k, v in fields.items():
        if v is None or isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v if k in _LOGGED_AS_IS and len(v) <= LOGGED_TEXT_CHARS else {"chars": len(v)}
        elif isinstance(v, (list, tuple)):
            out[k] = {"items": len(v)}
        elif isinstance(v, dict):
            out[k] = _log_view(v, depth + 1) if depth < 1 else {"keys": len(v)}
        else:
            out[k] = type(v).__name__
    return out


def _at_work(row: dict) -> dict:
    """A trigger still at work, as the run's record keeps it."""
    return {k: row.get(k) for k in ("id", "status", "running", "pending", "executor_calls", "waits", "resumes_at")
            if row.get(k) not in (None, [], 0)}


def _asleep(hm: tuple[int, int], wake: tuple[int, int], sleep: tuple[int, int]) -> bool:
    """Whether local time ``hm`` falls in the night from ``sleep`` to ``wake``; a sleep time after midnight (00:30 with
    a wake at 08:00) makes the night the span between them. Equal times: never asleep."""
    if sleep > wake:
        return hm >= sleep or hm < wake
    return sleep <= hm < wake


def _hm(text: str) -> tuple[int, int]:
    h, m = text.split(":")
    return int(h), int(m)


def _end_instant(end_time: str, t0: datetime) -> datetime:
    if isinstance(end_time, str) and end_time.startswith("P"):
        m = re.fullmatch(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", end_time)
        if m is None or not any(m.groups()):
            raise ValueError(f"invalid ISO 8601 duration {end_time!r}")
        w, d, h, mi, s = (float(g or 0) for g in m.groups())
        return t0 + timedelta(weeks=w, days=d, hours=h, minutes=mi, seconds=s)
    return _utc(end_time)
