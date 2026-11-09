"""The always_on and user_sim steps: a task that runs over simulated days, the way you'd hand a job to a personal
assistant, with an LLM user sim.

The ``user_sim`` step (``user_sim.py``, which documents the interface this step uses) says who the user is and records
it in the run's metadata under its ``name``; a ``deploy_agent`` step deploys the agent that plays them. The
``always_on`` step names it with ``user_sim`` and runs the loop:

- The first wake is the user's request. While the task's clock is set, each wake opens with the task's date and time
  (any other date the agent sees is the computer's clock), then what came due for the agent on notifications/v1 (a
  trigger's ``notify``, a scheduled check), labelled ``[time · channel]``, then the user's own messages.
- Only what the agent sends with the gateway's ``message_user`` tool (user-chat/v1) reaches the user; its plain reply
  goes to the log. Each wake should end with ``message_user`` or ``stay_quiet``: one that ends with closing text and
  neither gets one ``nudge`` turn, and every wake ends with a reminder. All three are on unless the task turns them off.
- The user sim is asked only after the agent sent something, at the time the user would read it (``schedule``), with
  everything sent meanwhile in one read: the time, the agent's messages, what triggers sent the user's phone
  (``notify`` to ``user_phone``) and the user's own lines (the opening request, ``user_says`` lines, ``final``).
  Nothing meant for the agent is in it. Its answer is reply / no_reply / done, and a reply is held for the response
  time. Done ends a one-sitting run; a run whose end_time falls on a later day than its start goes on to end_time.
- Scripted user lines are triggers' ``notify`` actions to the ``user_says`` recipient: sent to the agent as the
  user's message at their time.
- The clock runs at the task's speed during turns (sync_env_clock). When nothing is due, the step advances it to the
  earliest of: the next trigger event, the next notification, a held reply, a pending read, or end_time, never past
  one; then waits up to ``settle_timeout_seconds`` for triggers to finish. What comes due in the same minute wakes
  the agent once.
- The agent can schedule its own wake-ups (scheduled checks) when the run is multi-day, at least
  ``scheduled_check_min_interval`` apart; ``scheduled_checks`` turns them on or off whatever the run's length. A
  check's wake counts toward ``max_wakes``.
- At end_time the user's ``final`` line (with a reminder to use message_user) gives the agent one last turn. The run
  ends on end_time, done (one sitting), max_wakes, or nothing left to wake the agent; the reason is recorded.
- Everything goes into a timeline in ``metadata["always_on"][step_id]`` and the object store.
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
    DEFAULT_ROLE,
    EXT_CLOCK_URI,
    EXT_NOTIFICATIONS_URI,
    EXT_TRIGGERS_URI,
    EXT_USER_CHAT_URI,
)
from agent_env.store import DuplicateKeyError, get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps import user_sim_versions
from agent_env.task_step.task_steps.user_sim import USER_SIMS_KEY
from agent_env.task_step.task_steps.notification_delivery import delivered_parts, label_time, take_due
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

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
TIME_LINE = "It's {when}. Any other date you see is the computer's clock, not the task's."


# ==================================================================================================================
class AlwaysOnTaskStep(PromptAgentTaskStep):
    type: ClassVar[str] = "always_on"
    entity_refs = ()
    field_scopes = {k: v for k, v in PromptAgentTaskStep.field_scopes.items() if k != "user_model"}

    def __init__(self, id: str, version: Optional[int], *, user_sim: str, end_time: Optional[str] = None,
                 final: Optional[str] = None, max_wakes: int = 60, settle_timeout_seconds: float = 300,
                 user_timeout_seconds: int = 300, final_reminder: Optional[str] = FINAL_REMINDER,
                 stay_quiet: bool = True, nudge: bool = True, nudge_text: str = NUDGE_TEXT,
                 wake_reminder: bool = True, wake_reminder_text: str = WAKE_REMINDER_TEXT,
                 scheduled_checks: Optional[bool] = None, scheduled_check_min_interval: str = CHECK_MIN_INTERVAL,
                 model: Optional[str] = None, model_params: Optional[dict] = None, effort: Optional[str] = None,
                 max_thinking_tokens: Optional[int] = None, harness: Optional[str] = None,
                 trajectory_output_prefix: Optional[str] = None, **prompt_fields: Any):
        for unsupported in ("max_conversation_turns", "user_agent_name", "user_a2a_url", "post_turn_hooks",
                            "user_output_format", "user_model", "snapshot_config"):
            prompt_fields.pop(unsupported, None)
        super().__init__(id, version, max_conversation_turns=max(1, int(max_wakes)), model=model,
                         model_params=model_params, effort=effort, max_thinking_tokens=max_thinking_tokens,
                         harness=harness, trajectory_output_prefix=trajectory_output_prefix, **prompt_fields)
        if not user_sim:
            raise ValueError("always_on needs user_sim, the name of a user_sim step's user")
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

    _OWN = ("user_sim", "end_time", "final", "max_wakes", "settle_timeout_seconds", "user_timeout_seconds",
            "final_reminder", "stay_quiet", "nudge", "nudge_text", "wake_reminder", "wake_reminder_text",
            "scheduled_checks", "scheduled_check_min_interval")
    _PROMPT_FIELDS = ("prompt", "parts", "prompt_id", "agent_name", "timeout_seconds", "trajectory_output_prefix",
                      "model", "model_params", "system_prompt", "max_turns", "output_format", "max_thinking_tokens",
                      "effort", "harness", "agentenv_tools", "poll_interval_seconds", "context_id")

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

    async def _execute(self, context: TaskStepContext) -> TaskStepContext:
        run = _Run(self, context)
        try:
            await run.go()
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
        self.last_plain = ""
        self.last_messages: list[dict] = []
        self.all_messages: list[dict] = []
        self.final_task_id = ""
        self.error: Optional[str] = None
        self._plugin_paths: list[str] = []
        self._plugin_modules: list[str] = []
        self.nudges = 0
        self.zone = "UTC"
        self.multi_day = False
        self.checks: dict = {"on": False}

    def close(self) -> None:
        for p in self._plugin_paths:
            if os.path.exists(p):
                os.unlink(p)
        for m in self._plugin_modules:
            sys.modules.pop(m, None)

    # ---- record ----------------------------------------------------------------------------------------------
    def log(self, event: str, **fields: Any) -> dict:
        entry = {"event": event, "real_time": datetime.now(timezone.utc).isoformat(),
                 "real_elapsed_s": round(time.time() - self.t0_real, 1), **fields}
        self.timeline.append(entry)
        logger.info(f"always_on {event}: " + json.dumps({k: v for k, v in fields.items()
                                                         if k not in ("input", "parts_text", "raw")},
                                                        default=str)[:400])
        return entry

    # ---- setup -----------------------------------------------------------------------------------------------
    async def go(self) -> None:
        step, ctx = self.step, self.ctx
        self.agent = next((a for a in ctx.deployed_agents if a.agent_name == step.agent_name), None)
        if self.agent is None:
            raise RuntimeError(f"Agent with name '{step.agent_name}' not found in context.deployed_agents")
        self.target_url = self.agent.a2a_url or self.agent.api_url
        self.recipient = self.agent.role or DEFAULT_ROLE
        us = (ctx.metadata.get(USER_SIMS_KEY) or {}).get(step.user_sim)
        if us is None:
            raise RuntimeError(f"always_on: no user_sim step named '{step.user_sim}' ran before this step")
        self.us = us
        self.zone = _task_zone(us, ctx.metadata.get("clock_configurations") or [])
        self.person = {**self._person(us), "time_zone": self.zone}
        self.version = us["user_version"]
        self.user_agent = next((a for a in ctx.deployed_agents if a.agent_name == us["agent_name"]), None)
        if self.user_agent is None:
            raise RuntimeError(f"always_on: the user sim agent '{us['agent_name']}' is not deployed")
        self.user_url = self.user_agent.a2a_url or self.user_agent.api_url
        self.plugin = await self._load_plugin(us.get("plugin"))
        self.tz = ZoneInfo(self.zone)

        self.clock_envs = [d for d in ctx.deployed_envs if d.supports(EXT_CLOCK_URI, "state")]
        self.notification_envs = [d for d in ctx.deployed_envs if d.supports(EXT_NOTIFICATIONS_URI, "take")]
        self.chat_envs = [d for d in ctx.deployed_envs if d.supports(EXT_USER_CHAT_URI, "take")]
        if not self.chat_envs:
            raise RuntimeError(f"always_on: no deployed env offers {EXT_USER_CHAT_URI} (message_user); rebuild its "
                               "gateway from a branch that has user_chat.py")
        now = await self.now()
        t0 = await self._t0() or now
        self.end = _end_instant(step.end_time, t0) if step.end_time else None
        self.multi_day = self.end is not None and self.end.astimezone(self.tz).date() > t0.astimezone(self.tz).date()
        chat_config = {"roles": [self.recipient], "user_name": self.person.get("name") or "your user"}
        if step.stay_quiet:
            chat_config["stay_quiet"] = True
        for env in self.chat_envs:
            await _call(env, EXT_USER_CHAT_URI, "config", chat_config)
        await self._configure_checks()
        await self._configure_agents()

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
        self.solver_context_id = uuid.uuid4().hex

        now = await self.now()
        self.log("start", virtual_time=_iso(now), end_time=_iso(self.end) if self.end else None,
                 agent=step.agent_name, agent_role=self.recipient, user_sim=step.user_sim, version=self.version,
                 user_model=us["model"], max_wakes=step.max_wakes, time_zone=self.zone, multi_day=self.multi_day,
                 scheduled_checks=self.checks, clock_rate=await self._rate())
        try:
            await self.loop(now)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.end_reason = self.end_reason or "step_error"
            self.log("error", error=self.error)
            raise
        finally:
            await self.finish()

    def _person(self, us: dict) -> dict:
        name = us.get("display_name") or us["name"]
        return {"key": us["name"], "name": name, "first": us.get("first_name") or name.split()[0],
                "email": us.get("email"), "time_zone": us.get("time_zone"),
                "description": us.get("description"), "versions": us.get("versions") or {},
                "output_format": us.get("output_format"),
                "permissions": us.get("permissions"), "keep_private": us.get("keep_private")}

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

        setup = self.plugin.setup(self.version, self.person)
        user_config = {"model": self.us["model"], "system_prompt": setup.get("system_prompt"),
                       "output_format": setup.get("output_format")}
        user_config = {k: v for k, v in user_config.items() if v is not None}
        negotiated = A2AAgent.negotiate_agent_config(self.user_agent.a2a_card or {}, user_config)
        if negotiated:
            await protocol.post_agent_config(self.user_url + negotiated.endpoint, negotiated.fields)
        accepted = sorted(negotiated.fields) if negotiated else []
        ctx.record_model("user simulator", step.id, negotiated.fields.get("model") if negotiated else None)
        self.user_setup = {"system_prompt_chars": len(setup.get("system_prompt") or ""),
                           "output_format": setup.get("output_format"), "accepted_fields": accepted}
        if "system_prompt" not in accepted and setup.get("system_prompt"):
            logger.warning("always_on: the user sim agent's card did not accept system_prompt; it keeps its deploy one")

    def rules(self) -> str:
        """What the agent is told about reaching the user, and about waking itself when it can."""
        return " ".join([MESSAGE_RULES, *([QUIET_RULE] if self.step.stay_quiet else []),
                         *([CHECKS_RULE] if self.checks["on"] else [])])

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
                got = await _call(env, EXT_NOTIFICATIONS_URI, "config", body)
                applied.append({"env_id": env.env_id, "set_by": "step", "min_interval": got.get("min_interval"),
                                "max_active": got.get("max_active")})
        self.checks = {"on": bool(applied), "envs": applied}

    def _system_prompt(self) -> Optional[str]:
        """The step's system prompt with the rules after it; without one, the agent keeps its own and the first wake
        opens with the rules."""
        return f"{self.step.system_prompt.rstrip()}\n\n{self.rules()}" if self.step.system_prompt else None

    def _time_line(self, at: datetime) -> str:
        """The task's date and time, and the user's local time when they aren't in UTC."""
        when = f"{at.strftime('%a %b')} {at.day}, {at.year}, {_clock_time(at)} UTC"
        if self.zone != "UTC":
            local = at.astimezone(self.tz)
            when += f" ({local.strftime('%a')} {_clock_time(local)} for {self.person['first']}, {self.zone})"
        return TIME_LINE.format(when=when)

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
        for env in await self._armed():
            try:
                await _call(env, EXT_CLOCK_URI, "advance", {"virtual_time": _iso(target)}, attempts=2)
            except Exception:
                state = await _call(env, EXT_CLOCK_URI, "state")
                if not (state.get("armed") and _utc(state["virtual_time"]) >= target):
                    raise
        stuck = await self.settle()
        self.log("clock_jump", **{"from": _iso(before), "to": _iso(target), "reason": reason,
                                  "settled": not stuck, "still_firing": stuck})

    async def settle(self) -> list[str]:
        """Wait until no trigger still fires (bar one waiting for a later time); the ids of envs still firing."""
        deadline = time.monotonic() + self.step.settle_timeout_seconds
        envs = [e for e in await self._armed() if e.supports(EXT_TRIGGERS_URI, "state")]
        waiting = set(range(len(envs)))
        while waiting and time.monotonic() < deadline:
            for i in sorted(waiting):
                env = envs[i]
                clock = await _call(env, EXT_CLOCK_URI, "state")
                state = await _call(env, EXT_TRIGGERS_URI, "state")
                now = _utc(clock["virtual_time"])
                if state.get("next_event_at") is not None and _utc(state["next_event_at"]) <= now:
                    try:
                        await _call(env, EXT_CLOCK_URI, "advance", {"virtual_time": _iso(now)}, attempts=1)
                    except Exception:
                        pass
                elif not any(_busy(row, now) for row in state.get("triggers", [])):
                    waiting.discard(i)
            if waiting:
                self.ctx.heartbeat()
                await asyncio.sleep(POLL_SECONDS)
        return sorted(envs[i].env_id for i in waiting)

    async def _next_world_event(self) -> Optional[datetime]:
        out = []
        for env in await self._armed():
            if env.supports(EXT_TRIGGERS_URI, "state"):
                st = await _call(env, EXT_TRIGGERS_URI, "state")
                if st.get("next_event_at"):
                    out.append(_utc(st["next_event_at"]))
        return min(out) if out else None

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
        while awake and not busy. No schedule: at once, mode "now"."""
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
        for _ in range(4):
            moved = False
            for start, stop in busy:
                if start <= lt < stop:
                    lt, moved = stop, True
            hm = (lt.hour, lt.minute)
            if hm >= sleep or hm < wake:
                day = lt.date() if hm < wake else lt.date() + timedelta(days=1)
                lt, mode, moved = datetime(day.year, day.month, day.day, *wake, tzinfo=tz), "morning", True
            if not moved:
                break
        if mode is None:
            hm = (lt.hour, lt.minute)
            mode = "work" if DAYS[lt.weekday()] in work_days and work[0] <= hm < work[1] else "free"
        return lt.astimezone(timezone.utc), mode

    def _latency(self, mode: str) -> timedelta:
        rt = self.us.get("response_times") or {}
        if mode == "now" or not rt:
            return timedelta(0)
        return timedelta(minutes=float(rt.get(mode, rt.get("free", 0))))

    def _schedule_read(self) -> None:
        if not self.unread or self.read_at is not None or self.gave_up:
            return
        first_at = min(_utc(m["at"]) for m in self.unread if m.get("at")) if any(m.get("at") for m in self.unread) \
            else datetime.now(timezone.utc)
        at, mode = self._available(first_at)
        if self.last_reply_at and at < self.last_reply_at:
            at, mode = self._available(self.last_reply_at)
        self.read_at, self.read_mode = at, mode
        self.log("user_read_scheduled", read_at=_iso(at), mode=mode, unread=len(self.unread))

    # ---- the loop -------------------------------------------------------------------------------------------
    async def loop(self, now: datetime) -> None:
        step = self.step
        first = self._effective_prompt()
        self.you_sent.append({"at": _iso(now), "text": first})
        due = await take_due(self.notification_envs, self.recipient) if self.notification_envs else []
        await self.wake([*delivered_parts(due), self._user_part(now, first, opening=True)],
                        woke_by=[f"notification:{d['channel']}" for d in due] + ["user_request"],
                        notifications=[{k: d.get(k) for k in ("at", "channel", "source", "source_id", "to")}
                                       for d in due])
        while self.end_reason is None:
            if self.wakes >= step.max_wakes:
                self.end_reason = "max_wakes"
                break
            now = await self.now()
            # 1. a read that is due: the user sim reads what the agent sent
            if self.read_at is not None and self.read_at <= now and self.unread:
                await self.read(now)
                continue
            # 2. anything due for the agent: notifications, scripted user lines, held replies; and what else comes
            # due in the same minute, so it all arrives in one wake
            if any(h["deliver_at"] <= now for h in self.held) or await self._due_count(self.recipient, now) \
                    or await self._due_count(USER_SAYS, now):
                later = await self._later(now)
                same_minute = [t for t in later.values() if t < now.replace(second=0, microsecond=0)
                               + timedelta(minutes=1) and (self.end is None or t < self.end)]
                if same_minute and await self._armed():
                    await self.advance(min(same_minute), reason="same_minute")
                    continue
                lines = await self._take_user_says()
                replies = [h for h in self.held if h["deliver_at"] <= now]
                self.held = [h for h in self.held if h["deliver_at"] > now]
                await self._wake_with(now, lines, replies, final=False)
                continue
            # 3. nothing due: jump to the next thing
            later = await self._later(now)
            soonest = min(later.values()) if later else None
            if self.end is not None and (soonest is None or soonest >= self.end):
                await self._final(now)
                break
            if soonest is None:
                self.end_reason = "nothing_can_wake_the_agent"
                break
            why = sorted(k for k, v in later.items() if v == soonest)
            await self.advance(soonest, reason="+".join(why))

    async def _later(self, now: datetime) -> dict[str, datetime]:
        """What could wake the agent after ``now``, by kind, each at its earliest time."""
        candidates = {"world_event": await self._next_world_event(),
                      "agent_notification": await self._next_due(self.recipient, now),
                      "user_says": await self._next_due(USER_SAYS, now),
                      "held_reply": min((h["deliver_at"] for h in self.held), default=None),
                      "user_read": self.read_at if self.unread else None}
        return {k: v for k, v in candidates.items() if v is not None and v > now}

    async def _final(self, now: datetime) -> None:
        step = self.step
        if self.end > now:
            await self.advance(self.end, reason="end_time")
        now = await self.now()
        if not step.final:
            self.end_reason = "end_time"
            return
        self.final_sent = True
        lines = await self._take_user_says()
        replies = [h for h in self.held if h["deliver_at"] <= now]
        self.held = []
        lines.append({"at": _iso(now), "text": step.final, "source": "final"})
        await self._wake_with(now, lines, replies, final=True)
        self.end_reason = self.end_reason or "end_time"

    def _effective_prompt(self) -> str:
        parts = self.step._apply_seed(self.step._effective_parts(), self.ctx.metadata.get("seed", {}))
        return "\n".join(p.get("text", "") for p in parts if p.get("kind") == "text")

    def _user_part(self, at: datetime, text: str, opening: bool = False) -> dict:
        first = self.person["first"]
        body = re.sub(rf"^\s*{re.escape(first)}\s*:\s*", "", text)
        return {"kind": "text", "text": f"[{label_time(_iso(at))}] {first}: {body}",
                "metadata": {"agentenv": {"source": "user", "at": _iso(at), "opening": opening}}}

    async def _take_user_says(self) -> list[dict]:
        items = await take_due(self.notification_envs, USER_SAYS) if self.notification_envs else []
        out = []
        for it in items:
            text = "\n\n".join(p["text"] for p in it["parts"] if p.get("kind") == "text")
            out.append({"at": it.get("at"), "text": text, "source": "user_says", "trigger_id": it.get("source_id")})
        return out

    async def _wake_with(self, now: datetime, lines: list[dict], replies: list[dict], final: bool) -> None:
        due = await take_due(self.notification_envs, self.recipient) if self.notification_envs else []
        parts = delivered_parts(due)
        user_msgs = sorted([{"at": l["at"] or _iso(now), "text": l["text"], "kind": l.get("source", "user_says"),
                             "trigger_id": l.get("trigger_id")} for l in lines]
                           + [{"at": _iso(r["deliver_at"]), "text": r["text"], "kind": "user_reply"} for r in replies],
                           key=lambda m: _utc(m["at"]))
        for m in user_msgs:
            parts.append(self._user_part(_utc(m["at"]), m["text"]))
            if m["kind"] != "user_reply":
                self.you_sent.append({"at": m["at"], "text": m["text"]})
                self.log("scripted_user_line", virtual_time=m["at"], text=m["text"], kind=m["kind"],
                         trigger_id=m.get("trigger_id"))
            else:
                self.log("user_reply_delivered", virtual_time=m["at"], text=m["text"])
        if final and self.step.final_reminder:
            parts.append(_step_part(self.step.final_reminder.format(first=self.person["first"]), "final_reminder"))
        woke_by = [f"notification:{d['channel']}" for d in due] + [f"{m['kind']}" for m in user_msgs]
        await self.wake(parts, woke_by=woke_by, notifications=[{k: d.get(k) for k in
                                                                ("at", "channel", "source", "source_id", "to")}
                                                               for d in due])

    async def wake(self, parts: list[dict], woke_by: list[str], notifications: Optional[list] = None) -> None:
        step = self.step
        sent_at = await self.now()
        clock = await step._env_clock(self.clock_envs)
        lead = [_step_part(self._time_line(sent_at), "time")] if clock else []
        if self.wakes == 0 and not step.system_prompt:
            lead = [_step_part(self.rules(), "rules"), *lead]
        parts = [*lead, *parts]
        if step.wake_reminder:
            parts = [*parts, _step_part(step.wake_reminder_text, "reminder")]
        self.wakes += 1
        msg_id = uuid.uuid4().hex

        def record() -> None:
            conversation_store.add_a2a_task(conversation_id=self.conversation_id, parts=parts, a2a_task_id=msg_id,
                                            role="user", **clock)
            self.per_turn_parts.append(list(parts))

        self.log("wake", n=self.wakes, virtual_time=_iso(sent_at), woke_by=woke_by, notifications=notifications or [],
                 parts_text="\n".join(p.get("text", "") for p in parts if p.get("kind") == "text"))
        t = time.time()
        try:
            task_id, result = await send_and_wait(self.target_url, parts, agent=self.agent, message_id=msg_id,
                                                  context_id=self.solver_context_id,
                                                  timeout_seconds=step.timeout_seconds,
                                                  poll_interval_seconds=step.poll_interval_seconds, before_send=record)
        except TimeoutError as e:
            self.error = f"agent turn timed out after {step.timeout_seconds}s: {e}"
            self.end_reason = "agent_turn_timeout"
            self.log("agent_turn_end", n=self.wakes, state="timeout", real_seconds=round(time.time() - t, 1))
            return
        state = result["status"]["state"]
        status_msg = (result.get("status") or {}).get("message") or {}
        term = protocol.TerminalResponse.from_message(status_msg)
        if term.tool_call_count is not None:
            self.tool_calls = (self.tool_calls or 0) + term.tool_call_count
        agent_parts = status_msg.get("parts") or [{"kind": "text", "text": term.response_text}]
        messages, quiet = [], []
        for env in self.chat_envs:
            take_id = uuid.uuid4().hex
            got = await _call(env, EXT_USER_CHAT_URI, "take", {"take_id": take_id})
            messages += got.get("items") or []
            quiet += got.get("quiet") or []
        messages.sort(key=lambda m: m.get("at") or "")
        # The conversation shows graders what the user saw (message_user) and, marked, what only went to the log.
        recorded = [{"kind": "text", "text": f"[message_user · {label_time(m.get('at'))}] {m['text']}",
                     "metadata": {"agentenv": {"source": "message_user", "at": m.get("at")}}} for m in messages]
        recorded += [{**p, "text": f"[plain reply, not shown to the user] {p['text']}",
                      "metadata": {"agentenv": {"source": "plain_reply"}}} if p.get("kind") == "text" else p
                     for p in agent_parts]
        conversation_store.complete_a2a_task(conversation_id=self.conversation_id, parts=recorded, role="agent",
                                             **(await step._env_clock(self.clock_envs)))
        traj = None
        card = self.agent.a2a_card or {}
        traj_ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if traj_ext:
            try:
                traj = await step._fetch_trajectory(self.target_url, traj_ext, task_id, step._trajectory_prefix(),
                                                    msg_id, card=card, sandbox_type=self.agent.sandbox_type)
            except Exception as e:
                logger.warning(f"wake {self.wakes}: trajectory fetch failed (continuing): {e}")
        self.per_turn_trajectory_urls.append(traj)
        self.final_task_id = task_id
        ended = await self.now()
        plain = term.response_text or ""
        self.last_plain, self.last_messages = plain, messages
        extra = {"stay_quiet_calls": len(quiet)} if step.stay_quiet else {}
        self.log("agent_turn_end", n=self.wakes, state=str(state), virtual_time=_iso(ended),
                 virtual_seconds=round((ended - sent_at).total_seconds()), real_seconds=round(time.time() - t, 1),
                 tool_calls=term.tool_call_count, plain_reply=plain, messages_to_user=len(messages),
                 trajectory_url=traj, **extra)
        for m in messages:
            self.log("message_to_user", virtual_time=m.get("at"), text=m["text"], wake=self.wakes)
            self.unread.append({"at": m.get("at"), "text": m["text"]})
            self.all_messages.append({"at": m.get("at"), "text": m["text"], "wake": self.wakes})
        if state == TaskState.failed or str(state) == "failed":
            self.error = term.error_message or term.response_text or "agent turn failed"
            self.end_reason = "agent_turn_failed"
            return
        if step.nudge and not messages and not quiet and _closing_text(plain):
            await self._nudge(plain)
        stuck = await self.settle()
        if stuck:
            self.log("settle_timeout", envs=stuck)
        self._schedule_read()

    async def _nudge(self, closing: str) -> None:
        """The wake ended with closing text and neither message_user nor stay_quiet, so give the agent one more short
        turn with ``nudge_text``; record what the closing text was and what it did after."""
        step = self.step
        self.nudges += 1
        parts = [_step_part(step.nudge_text, "nudge")]
        sent_at = await self.now()
        clock = await step._env_clock(self.clock_envs)
        msg_id = uuid.uuid4().hex

        def record() -> None:
            conversation_store.add_a2a_task(conversation_id=self.conversation_id, parts=parts, a2a_task_id=msg_id,
                                            role="user", **clock)
            self.per_turn_parts.append(list(parts))

        t = time.time()
        try:
            task_id, result = await send_and_wait(self.target_url, parts, agent=self.agent, message_id=msg_id,
                                                  context_id=self.solver_context_id,
                                                  timeout_seconds=step.timeout_seconds,
                                                  poll_interval_seconds=step.poll_interval_seconds, before_send=record)
        except TimeoutError as e:
            self.error = f"agent nudge turn timed out after {step.timeout_seconds}s: {e}"
            self.end_reason = "agent_turn_timeout"
            self.log("nudge", n=self.nudges, wake=self.wakes, virtual_time=_iso(sent_at), closing_text=closing,
                     state="timeout", outcome="timeout", real_seconds=round(time.time() - t, 1))
            return
        state = result["status"]["state"]
        status_msg = (result.get("status") or {}).get("message") or {}
        term = protocol.TerminalResponse.from_message(status_msg)
        if term.tool_call_count is not None:
            self.tool_calls = (self.tool_calls or 0) + term.tool_call_count
        agent_parts = status_msg.get("parts") or [{"kind": "text", "text": term.response_text}]
        messages, quiet = [], []
        for env in self.chat_envs:
            got = await _call(env, EXT_USER_CHAT_URI, "take", {"take_id": uuid.uuid4().hex})
            messages += got.get("items") or []
            quiet += got.get("quiet") or []
        messages.sort(key=lambda m: m.get("at") or "")
        recorded = [{"kind": "text", "text": f"[message_user · {label_time(m.get('at'))}] {m['text']}",
                     "metadata": {"agentenv": {"source": "message_user", "at": m.get("at")}}} for m in messages]
        recorded += [{**p, "text": f"[plain reply, not shown to the user] {p['text']}",
                      "metadata": {"agentenv": {"source": "plain_reply"}}} if p.get("kind") == "text" else p
                     for p in agent_parts]
        conversation_store.complete_a2a_task(conversation_id=self.conversation_id, parts=recorded, role="agent",
                                             **(await step._env_clock(self.clock_envs)))
        traj = None
        card = self.agent.a2a_card or {}
        traj_ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if traj_ext:
            try:
                traj = await step._fetch_trajectory(self.target_url, traj_ext, task_id, step._trajectory_prefix(),
                                                    msg_id, card=card, sandbox_type=self.agent.sandbox_type)
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
        if messages:
            self.last_plain, self.last_messages = plain, messages
        for m in messages:
            self.log("message_to_user", virtual_time=m.get("at"), text=m["text"], wake=self.wakes, after_nudge=True)
            self.unread.append({"at": m.get("at"), "text": m["text"]})
            self.all_messages.append({"at": m.get("at"), "text": m["text"], "wake": self.wakes})
        if state == TaskState.failed or str(state) == "failed":
            self.error = term.error_message or term.response_text or "agent nudge turn failed"
            self.end_reason = "agent_turn_failed"

    async def read(self, now: datetime) -> None:
        phone = []
        if self.notification_envs:
            for it in await take_due(self.notification_envs, USER_PHONE):
                phone.append({"at": it.get("at"), "channel": it["channel"],
                              "text": "\n\n".join(p["text"] for p in it["parts"] if p.get("kind") == "text")})
        read = {"time": _iso(now), "assistant_messages": list(self.unread), "phone": phone,
                "you_sent": list(self.you_sent), "done_ends_run": not self.multi_day}
        self.unread, self.you_sent, self.read_at = [], [], None
        calls = []

        async def ask(text: str) -> dict:
            got = await self._ask_user_sim(text)
            calls.append(got)
            return got

        t = time.time()
        ans = await self.plugin.answer(self.version, self.person, self.history, read, ask)
        self.history.append({"read": read, "answer": ans})
        mode = self.read_mode
        latency = self._latency(mode)
        flagged = len(ans.get("message") or "") > 1200 or bool(re.search(r"^\s*(#|\|)", ans.get("message") or "", re.M))
        entry = self.log("user_sim_read", virtual_time=_iso(now), version=self.version, read=read,
                         input=ans.get("input"), raw=ans.get("raw"), outcome=ans.get("outcome"),
                         message=ans.get("message"), used=ans.get("used"), score=ans.get("score"),
                         schema_errors=ans.get("schema_errors"), mode=mode,
                         real_seconds=round(time.time() - t, 1), looks_like_assistant=flagged,
                         user_sim_error=(calls[-1].get("error") if calls else None),
                         user_sim_trajectory_urls=[c.get("trajectory_url") for c in calls],
                         user_sim_turns=ans.get("turns", len(calls)), shown=ans.get("shown"),
                         past_turn_cap=ans.get("past_turn_cap"))
        outcome = ans.get("outcome")
        if outcome == "reply" and (ans.get("message") or "").strip():
            deliver = now + latency
            self.held.append({"deliver_at": deliver, "text": ans["message"]})
            self.last_reply_at = deliver
            self.replies += 1
            entry["deliver_at"] = _iso(deliver)
            max_replies = self.us.get("max_replies")
            if max_replies and self.replies >= int(max_replies):
                self.gave_up = True
                self.log("user_gave_up", replies=self.replies)
        elif outcome == "done":
            entry["ends_run"] = not self.multi_day
            if not self.multi_day:
                self.end_reason = "user_done"
        if calls and calls[-1].get("error"):
            self.end_reason = self.end_reason or "user_sim_error"

    async def _ask_user_sim(self, text: str) -> dict:
        msg_id = uuid.uuid4().hex
        try:
            user_task_id, result = await send_and_wait(
                self.user_url, [{"kind": "text", "text": text}],
                agent=self.user_agent if self.user_agent.sandbox_id else None, message_id=msg_id,
                context_id=self.user_context_id, timeout_seconds=self.step.user_timeout_seconds,
                poll_interval_seconds=self.step.poll_interval_seconds)
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
                traj = await self.step._fetch_trajectory(self.user_url, traj_ext, user_task_id,
                                                         self.step._trajectory_prefix() + "user_sim/", msg_id,
                                                         card=card, sandbox_type=self.user_agent.sandbox_type)
            except Exception as e:
                logger.warning(f"user sim trajectory fetch failed (continuing): {e}")
        return {"text": raw, "structured": structured if isinstance(structured, dict) else None,
                "trajectory_url": traj}

    # ---- the end --------------------------------------------------------------------------------------------
    async def finish(self) -> None:
        step, ctx = self.step, self.ctx
        now = await self.now()
        self.log("end", reason=self.end_reason, virtual_time=_iso(now), wakes=self.wakes, replies=self.replies,
                 unread_at_end=len(self.unread), held_at_end=len(self.held))
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
            "clock_jumps": sum(1 for e in self.timeline if e["event"] == "clock_jump"),
            "silent_wakes": sum(1 for e in self.timeline if e["event"] == "agent_turn_end"
                                and not e.get("messages_to_user")),
        }
        if step.nudge:
            nudges = [e for e in self.timeline if e["event"] == "nudge"]
            counts["nudges"] = len(nudges)
            counts["nudge_outcomes"] = {o: sum(1 for e in nudges if e.get("outcome") == o)
                                        for o in sorted({e.get("outcome") for e in nudges})}
        if step.stay_quiet:
            counts["stay_quiet_wakes"] = sum(1 for e in self.timeline if e["event"] == "agent_turn_end"
                                             and e.get("stay_quiet_calls"))
        record = {"step_id": step.id, "end_reason": self.end_reason, "error": self.error, "counts": counts,
                  "user_sim": {"name": step.user_sim, "version": self.version, "model": self.us.get("model"),
                               "agent_name": self.us.get("agent_name"), **self.user_setup},
                  "end_time": _iso(self.end) if self.end else None, "time_zone": self.zone,
                  "multi_day": self.multi_day, "timeline": self.timeline}
        record["settings"] = {"stay_quiet": step.stay_quiet, "nudge": step.nudge, "wake_reminder": step.wake_reminder,
                              "scheduled_checks": self.checks}
        try:
            store = get_config().get_object_store()
            key = (f"{get_config().get_artifact_key_prefix()}always_on_timeline/"
                   f"instance_id={ctx.instance_id or 'adhoc'}/{step.id}-{uuid.uuid4().hex[:8]}.json")
            record["object_url"] = await asyncio.to_thread(store.put, key,
                                                           json.dumps(record, indent=1, default=str).encode(),
                                                           content_type="application/json")
        except Exception as e:
            logger.warning(f"always_on: could not store the timeline: {e}")
        ctx.metadata.setdefault(ALWAYS_ON_KEY, {})[step.id] = record
        final_text = "\n\n".join([m["text"] for m in self.last_messages] + ([self.last_plain] if self.last_plain else []))
        traj = next((u for u in reversed(self.per_turn_trajectory_urls) if u), None)
        ctx.prompt_responses.append(PromptResponse(
            prompt_id=step.prompt_id, response=final_text, prompt_text=step.prompt,
            agent_trajectory_object_url=traj, agent_trajectory_object_prefix=step._trajectory_prefix(),
            target_agent_per_turn_trajectory_object_urls=self.per_turn_trajectory_urls or None,
            source_agent_per_turn_prompt_parts=self.per_turn_parts or None, tool_call_count=self.tool_calls,
            model=self.model, error_message=self.error, agent_session_id=self.final_task_id,
            a2a_context_id=self.solver_context_id, agent_name=step.agent_name, step_id=step.id))


# ==================================================================================================================
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
                await asyncio.sleep(min(2 ** attempt, 20))
    raise last


def _task_zone(us: dict, clocks: list[dict]) -> str:
    """The user's time zone, else the one sync_env_clock gave the world; refused when the two disagree."""
    zones = sorted({c["timezone"] for c in clocks if c.get("timezone")})
    zone = us.get("time_zone")
    if zone and zones and zones != [zone]:
        raise ValueError(f"always_on: user sim '{us['name']}' is in {zone} but sync_env_clock set {zones}")
    if not zone and len(zones) > 1:
        raise ValueError(f"always_on: sync_env_clock set several time zones {zones}; give the user_sim step one")
    return zone or (zones[0] if zones else "UTC")


def _step_part(text: str, source: str) -> dict:
    """A line the step adds to a wake, marked so a reader can tell it from what the user or the world sent."""
    return {"kind": "text", "text": text, "metadata": {"agentenv": {"source": source}}}


def _clock_time(at: datetime) -> str:
    return at.strftime("%I:%M %p").lstrip("0")


def _closing_text(plain: str) -> str:
    text = plain.strip()
    return "" if text in EMPTY_REPLIES else text


def _busy(row: dict, now: datetime) -> bool:
    if row.get("evaluating"):
        return True
    if row.get("status") not in IN_FLIGHT and not row.get("pending"):
        return False
    return not (row.get("waiting") and row.get("resumes_at") and _utc(row["resumes_at"]) > now)


def _hm(text: str) -> tuple[int, int]:
    h, m = text.split(":")
    return int(h), int(m)


def _utc(text: str) -> datetime:
    at = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError(f"{text!r} has no UTC offset")
    return at.astimezone(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if value else None


def _end_instant(end_time: str, t0: datetime) -> datetime:
    if isinstance(end_time, str) and end_time.startswith("P"):
        m = re.fullmatch(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", end_time)
        if m is None or not any(m.groups()):
            raise ValueError(f"invalid ISO 8601 duration {end_time!r}")
        w, d, h, mi, s = (float(g or 0) for g in m.groups())
        return t0 + timedelta(weeks=w, days=d, hours=h, minutes=mi, seconds=s)
    return _utc(end_time)
