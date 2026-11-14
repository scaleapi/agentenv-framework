"""The deploy_user_sim_agent step: deploys the agent that plays an always-on task's user, with who the user is already
in its system prompt, and records the user in ``metadata["user_sims"][agent_name]``, where the always_on step that
names it in ``user_sim`` reads them. How to write a user is in user_sim.md.

When it runs, the step renders the system prompt with the built-in user sim (``user_sim_versions.setup``) from
``persona``, ``opening_prompt`` (its ``<seed>`` placeholders filled in) and ``done_ends_run = not multi_day``; deploys the
agent through deploy_agent with that prompt; then sets its model and answer format through agent-config. A card that
takes no system prompt or no model fails the step: a user sim without its persona or its model is a different user.
The model is this step's ``step_params`` ``model``, else the run's ``user_model`` (``task run`` / ``eval run
--user-model``, a bundle's ``user_model``), else the step's ``model``. A run-wide ``a2a_agent_id`` is ignored, so the
agent under test never plays the user; this step's own ``step_params`` can still change it.

``opening_prompt`` is text, or a list of A2A text and file parts in ``prompt_agent``'s ``parts`` format, for a first
message that attaches files (any type). A file part may carry a one-line ``metadata.description`` of what it is, for
the user sim, which can't open files; the agent is sent the file without it. The parts are checked when the task loads:
at least one text part with text, each part a valid text or file part, each description one line. The seed fills
``<placeholder>``s in text and in a file's ``uri`` and ``name``, as in any prompt's parts.

What always_on reads from the record (``metadata["user_sims"][agent_name]``):

- ``name`` (the agent name), ``display_name`` and ``first_name`` (else the display name's first word): how the agent's
  wakes and the user sim name the user; ``agent_name``: the deployed agent always_on asks (``ask`` below).
- ``schedule`` (``wake``, ``sleep``, ``work``, ``work_days``, ``busy``): when the user reads, where ``sleep`` may be
  past midnight (wake 08:00, sleep 00:30); ``response_times`` (minutes by ``morning`` / ``work`` / ``free``): how long
  a reply is held; ``max_replies``: after that many replies the user stops reading. ``schedule`` and
  ``response_times`` need the task's clock (``sync_env_clock``): always_on refuses to start a run that sets either
  with no clock armed. The user's time zone is the task clock's.
- ``description``, ``permissions``, ``keep_private``: in ``person``, for the reads.
- ``opening`` (the opening's text parts, joined) and, for an ``opening_prompt`` given as parts, ``opening_parts``
  (those parts, descriptions included): both as written, the seed not yet filled in. The user's first message, which
  always_on sends, its text as one labelled line where its first text part was and each file in its place; always_on
  has no prompt of its own.
- ``prompt_set``: the system prompt and model are on the agent, so always_on sets neither; ``multi_day``: the done line
  the prompt was written with, which always_on refuses to run against a run of the other kind.
- ``model``, ``model_source``, ``prompt_sha256``, ``user_version``, ``setup``: for the run's record.
- ``persona_sha256``: ``user_sim_common.persona_fingerprint`` of the persona and ``opening_prompt`` as written, which
  user_sim_judge compares a rubric's ``rubric_persona_sha256`` with.
- ``tagged_lines``: each tag the system prompt shows with its line as written there (``{"W1": "...", "D9": "..."}``),
  so an answer's ``used`` resolves against the lines this run's user sim saw, whatever later edits to the persona move.

The built-in user sim (``user_sim_versions``) is used directly; its interface, internal to the two steps and
rate_user_satisfaction:

    setup(version: str, person: dict) -> {"system_prompt": str | None, "output_format": dict | None}
    async answer(version: str, person: dict, history: list[dict], read: dict, ask) -> dict
    async rate(version: str, person: dict, satisfaction: list[str], read: dict, ask) -> dict

``person`` is ``{"key": name, "name", "first", "time_zone", "description", "permissions", "keep_private", "opening",
"done_ends_run"}``, with ``time_zone`` the run's zone (the persona's, else the task clock's, else UTC) and ``opening`` the user's opening
request, ``{"at": RFC 3339 UTC, "text"}``, with ``"files": [{"name", "description"}]`` when it attaches files (the
description only when the part has one, the name its file's ``name``, else the last segment of its ``uri``), so
``setup`` can write it into the system prompt (this step's has no ``at``). The opening is also the first line of the
first read's ``you_sent``. ``done_ends_run`` is set before ``setup``, as each read has it, since the done line is in
the system prompt; the built-in user sim uses the multi-day wording when it's missing.

``answer`` runs once per read, at the time the user would read. ``read`` is ``{"time": RFC 3339 UTC,
"assistant_messages": [{"at", "text"}], "phone": [{"at", "channel", "text"}], "you_sent": [{"at", "text"}],
"done_ends_run": bool}``: the agent's message_user texts since the last read, oldest first; what triggers sent
``user_phone`` since the last read, its sender in ``channel`` (``"Gmail · Carol Reyes (Mom)"``) as in the agent's
alerts; the user's own lines (the opening request, with its ``files``, ``user_says`` lines, ``final``) each at the time
it was sent; never anything meant for the agent; and whether done ends the run (one-sitting) or is only recorded
(multi-day). The user sim reads only after the agent has used message_user. ``history`` is every earlier
``{"read", "answer"}``, oldest first. ``ask(text)`` sends one turn to the user sim agent (one session per run) and
returns ``{"text", "structured", "trajectory_url"}``, or ``{"text": None, "structured": None, "error"}``.

``answer`` returns ``{"outcome": "reply" | "no_reply" | "done", "message": str, "used": str | None}``, and may add
``input``, ``raw``, ``score``, ``schema_errors``, ``turns``, ``retried``, ``first_answer``, ``shown`` and
``past_turn_cap``, which always_on records. A ``reply`` or ``done`` with text is held for the response time, then
wakes the agent as the user's message; the recorded extras (``used``, ``score``, ...) never reach the agent. A reply
that never reaches the agent (one still held when the run ends, or the answer to the last read below) is recorded with
``delivered: false``. ``done`` ends a one-sitting run after the agent's turn on that message, or at once when it has
no text, with no further reads; in a multi-day run it is only recorded.
When the last ``ask`` of a read returned an error, the run ends (``user_sim_error``).

After the run's last wake (the final check-in, or the wake ``max_wakes`` makes the last), the user reads the agent's
wrap-up once more, if it sent one: one more ``answer``, its ``read`` shaped like any other, with the task's ``final``
line, when it sets one, in ``you_sent``. Its answer is recorded and never delivered; a ``done`` or an error on it
changes nothing.

``rate`` is the rate_user_satisfaction step's (rate_user_satisfaction.py), once an always_on or prompt_agent run is
over, never the run step's: ``satisfaction`` is that step's lines, which are never in the persona, so the system
prompt ``setup`` writes never has them. ``read`` is ``{"time": RFC 3339 UTC, "assistant_messages": [{"at", "text"}]}``,
the agent's messages the user never read (often none). It asks the user sim, in the run's session (``ask`` as above),
to rate the outcome 1-5 by those lines with a one-line reason, the answer in the same format as a read's, and returns
``{"score": 1-5 | None, "reason", "used", "schema_errors", "input", "raw", "retried", "turns"}`` (``first_answer``
after a retry). The step records it in ``metadata["user_sim_satisfaction"][run step id]``; it never reaches the agent
or the graded conversation.

prompt_agent with ``user_agent_name`` set to the agent this step deployed reads it with the same ``answer``, in one
sitting: one read per agent reply, ``phone`` empty, ``done_ends_run`` true, its prompt this step's opening: the same
text, and the same files in the same order once the seed is filled in (user_sim.md).
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import re
from datetime import datetime
from typing import Any, ClassVar, Optional

from agent_env.a2a_agent import protocol
from agent_env.config import get_config
from agent_env.entity.scopes import FieldScope
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.user_sim_common import (attached, opening_parts, opening_problems, opening_text,
                                                                  persona_fingerprint, recorded_opening)
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep

logger = logging.getLogger(__name__)

USER_SIMS_KEY = "user_sims"
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
SCHEDULE_KEYS = ("wake", "sleep", "work", "work_days", "busy")
RESPONSE_TIME_KEYS = ("morning", "work", "free")
PERSONA_KEYS = ("display_name", "first_name", "time_zone", "description", "permissions", "keep_private", "schedule",
                "response_times", "max_replies")
# deploy_agent's own fields, which this step takes as they are.
_DEPLOY_FIELDS = tuple(name for name in inspect.signature(DeployAgentTaskStep.__init__).parameters
                       if name not in ("self", "id", "version", "depends_on", "fail_task_on_error"))


def _clock(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", value) is not None


def _local_time(value: Any) -> bool:
    try:
        datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def schedule_problems(schedule: Any, response_times: Any, max_replies: Any) -> list[str]:
    """What's wrong with when the user reads and how long their replies take, one message each."""
    out = []
    if schedule is not None:
        if not isinstance(schedule, dict):
            return [f"schedule must be an object of {list(SCHEDULE_KEYS)}"]
        unknown = sorted(set(schedule) - set(SCHEDULE_KEYS))
        if unknown:
            out.append(f"schedule has unknown keys {unknown}; the keys are {list(SCHEDULE_KEYS)}")
        out += [f"schedule.{key} must be a time like 07:30" for key in ("wake", "sleep")
                if key in schedule and not _clock(schedule[key])]
        work = schedule.get("work", ["09:00", "18:00"])
        if not (isinstance(work, list) and len(work) == 2 and all(_clock(w) for w in work)):
            out.append('schedule.work must be [start, end], like ["09:00", "18:00"]')
        days = schedule.get("work_days", list(DAYS[:5]))
        if not (isinstance(days, list) and set(days) <= set(DAYS)):
            out.append(f"schedule.work_days must be a list of {list(DAYS)}")
        busy = schedule.get("busy") or []
        if not (isinstance(busy, list) and all(isinstance(b, dict) and _local_time(b.get("start"))
                                               and _local_time(b.get("end")) for b in busy)):
            out.append('schedule.busy must be a list of {start, end} local times, like "2026-11-12T09:00"')
    if response_times is not None and not schedule:  # no schedule, or {}: the user reads at once, nothing is held
        out.append("response_times needs a schedule (an empty one counts as none); without one, replies are never held")
    if response_times is not None:
        if not isinstance(response_times, dict):
            out.append(f"response_times must be an object of minutes by {list(RESPONSE_TIME_KEYS)}")
        else:
            unknown = sorted(set(response_times) - set(RESPONSE_TIME_KEYS))
            if unknown:
                out.append(f"response_times has unknown keys {unknown}; the keys are {list(RESPONSE_TIME_KEYS)}")
            out += [f"response_times.{key} must be a number of minutes, 0 or more" for key, v in response_times.items()
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0]
    if max_replies is not None and (isinstance(max_replies, bool) or not isinstance(max_replies, int)
                                    or max_replies < 1):
        out.append("max_replies must be a whole number, 1 or more")
    return out


def persona_problems(persona: Any) -> list[str]:
    """What's wrong with who the user is, one message each; empty when it's fine."""
    if not isinstance(persona, dict):
        return [f"persona must be an object of {list(PERSONA_KEYS)}"]
    out = []
    unknown = sorted(set(persona) - set(PERSONA_KEYS))
    if unknown:
        out.append(f"persona has unknown keys {unknown}; the keys are {list(PERSONA_KEYS)}")
    if not isinstance(persona.get("display_name"), str) or not persona["display_name"].strip():
        out.append("persona.display_name is required: the user's full name")
    if persona.get("time_zone") is not None:
        out += user_sim_versions.zone_problems(persona["time_zone"])
    out += schedule_problems(persona.get("schedule"), persona.get("response_times"), persona.get("max_replies"))
    out += [p for p in user_sim_versions.problems(
        {"name": persona.get("display_name") or "?", "first": persona.get("first_name"),
         "description": persona.get("description"), "permissions": persona.get("permissions"),
         "keep_private": persona.get("keep_private")}) if p != "name is missing"]
    return out


def person(key: str, persona: dict, opening: list[dict], seed: dict, multi_day: Optional[bool]) -> dict:
    """The user as the built-in user sim's ``setup`` takes them: ``persona``'s fields, the ``opening`` parts with the
    run's ``<seed>`` placeholders filled in (as always_on fills them), and the done line ``multi_day`` picks. The step
    writes its prompt from this, and user_sim_judge rebuilds that prompt from the step's record (``recorded_person``)."""
    from agent_env.task_step.task_steps.prompt_agent import apply_seed

    name = persona["display_name"]
    parts = apply_seed(opening, seed)
    text, files = opening_text(parts), attached(parts)
    return {"key": key, "name": name, "first": persona.get("first_name") or name.split()[0],
            "description": persona.get("description"), "permissions": persona.get("permissions"),
            "keep_private": persona.get("keep_private"),
            "opening": {"text": text, **({"files": files} if files else {})} if text else None,
            "done_ends_run": not multi_day}


def recorded_person(us: dict, seed: dict, multi_day: Optional[bool] = None) -> dict:
    """``person`` from the step's ``metadata["user_sims"]`` record, with its ``opening_parts`` when it has them, so
    ``setup`` gives the prompt the step wrote; ``multi_day`` is the record's, else the one given."""
    persona = {**{k: us.get(k) for k in PERSONA_KEYS},
               "display_name": us.get("display_name") or us.get("name") or "the user"}
    return person(us.get("name"), persona, recorded_opening(us), seed,
                  us["multi_day"] if us.get("multi_day") is not None else multi_day)


class DeployUserSimAgentTaskStep(DeployAgentTaskStep):
    """Deploys the agent that plays the task's user, its persona set at deploy, and records the user under
    ``agent_name`` in ``metadata["user_sims"]``."""

    type: ClassVar[str] = "deploy_user_sim_agent"
    system_prompt_required: ClassVar[bool] = True
    entity_refs = DeployAgentTaskStep.entity_refs
    # Which agent plays the user is part of the task, not the user's setup: written out for elsewhere, the task keeps it.
    field_scopes = {**DeployAgentTaskStep.field_scopes, "a2a_agent_id": FieldScope.TASK,
                    "a2a_agent_version": FieldScope.TASK}

    def __init__(self, id: str, version: Optional[int], *, persona: Optional[dict] = None, model: Optional[str] = None,
                 multi_day: Optional[bool] = None, opening_prompt: Optional[str | list[dict]] = None, **deploy: Any):
        if deploy.get("system_prompt") is not None:
            raise ValueError(f"{self.type} '{id}': system_prompt is written from the persona and opening_prompt; don't set it")
        deploy.pop("system_prompt", None)
        if not deploy.get("agent_name"):
            raise ValueError(f"{self.type} '{id}': agent_name is required: the user's key, which always_on's user_sim "
                             "names")
        super().__init__(id, version, **deploy)
        found = []
        if not isinstance(model, str) or not model.strip():
            found.append("model is required")
        # TEMPORARY: multi_day only picks the prompt's done line, so the prompt stays the one tested. always_on knows it
        # from end_time and refuses a run that disagrees; drop this field once the step can learn it from the task.
        if not isinstance(multi_day, bool):
            found.append("multi_day must be true or false: whether the run goes past the day it starts on")
        # Required: always_on starts the run from it, and refuses a user sim without one only after the deploy.
        found += opening_problems(opening_prompt)
        found += persona_problems(persona)
        if found:
            raise ValueError(f"{self.type} '{self.agent_name}': " + "; ".join(found))
        self.persona = persona
        self.model = model
        self.multi_day = multi_day
        self.opening_prompt = opening_prompt

    _OWN = ("persona", "model", "multi_day", "opening_prompt")

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.pop("system_prompt", None)
        return {**base, **{k: getattr(self, k) for k in self._OWN}}

    @classmethod
    def from_dict(cls, data: dict) -> DeployUserSimAgentTaskStep:
        fields = {k: data[k] for k in (*_DEPLOY_FIELDS, *cls._OWN) if k in data}
        if "metadata" in fields:
            fields["metadata"] = dict(fields["metadata"] or {})
        return cls(**cls._base_from_dict(data), **fields)

    def _resolve_agent_ref(self, context: TaskStepContext) -> tuple[str, Optional[int]]:
        """The user sim's agent id and version: this step's ``step_params`` over its stored values. The run-wide
        ``a2a_agent_id`` and ``a2a_agent_version`` pick the agent under test, which never plays the user."""
        step_overrides = self.step_param_overrides(context)
        if "a2a_agent_id" in step_overrides:
            return step_overrides["a2a_agent_id"], step_overrides.get("a2a_agent_version")
        agent_id = self.a2a_agent_id or get_config().get_default_a2a_agent_id()
        return agent_id, step_overrides.get("a2a_agent_version", self.a2a_agent_version)

    def person(self, context: TaskStepContext) -> dict:
        """The user as the built-in user sim's ``setup`` takes them, as always_on hands them over."""
        return person(self.agent_name, self.persona, opening_parts(self.opening_prompt),
                      context.metadata.get("seed") or {}, self.multi_day)

    def _setup(self, context: TaskStepContext) -> dict:
        return user_sim_versions.setup(user_sim_versions.VERSION, self.person(context))

    def _run_system_prompt(self, context: TaskStepContext) -> Optional[str]:
        # Rendered whole, with the seed already in the opening: the persona's own text gets no seed, as before.
        return self._setup(context)["system_prompt"]

    def _user_model(self, context: TaskStepContext) -> tuple[str, str]:
        """The model for this run, and where it came from."""
        own = self.step_param_overrides(context).get("model")
        if own:
            return own, "user_sim_step_params"
        run = (context.metadata.get("user_overrides") or {}).get("user_model")
        if run:
            return run, "run_user_model"
        return self.model, "user_sim_step"

    async def _execute(self, context: TaskStepContext, deployed_agents: list) -> TaskStepContext:
        # Inside deploy_agent's execute, so an agent whose model can't be set is closed like any failed deploy.
        setup = self._setup(context)
        context = await super()._execute(context, deployed_agents)
        agent = next(a for a in reversed(context.deployed_agents) if a.agent_name == self.agent_name)
        try:
            return await self._set_user(context, agent, setup)
        except BaseException:
            context.deployed_agents.remove(agent)  # closed by execute; a retry must not find it
            raise

    async def _set_user(self, context: TaskStepContext, agent, setup: dict) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        model, source = self._user_model(context)
        wanted = {"model": model, "output_format": setup.get("output_format")}
        negotiated = A2AAgent.negotiate_agent_config(agent.a2a_card or {}, {k: v for k, v in wanted.items() if v})
        if negotiated is None or "model" not in negotiated.fields:
            raise RuntimeError(f"{self.type} '{self.id}': agent '{self.agent_name}' takes no model through "
                               f"{A2AAgent.EXT_AGENT_CONFIG}, so it would play the user on a model of its own")
        await protocol.post_agent_config((agent.a2a_url or agent.api_url) + negotiated.endpoint, negotiated.fields)
        context.record_model("user simulator", self.id, model)
        prompt = setup["system_prompt"]
        persona = self.persona
        context.metadata.setdefault(USER_SIMS_KEY, {})[self.agent_name] = {
            "name": self.agent_name, "agent_name": self.agent_name, "step_id": self.id,
            "display_name": persona["display_name"], "first_name": persona.get("first_name"),
            # The user's own zone; unset, always_on uses the task clock's. A clock that sets a different one is refused.
            "time_zone": persona.get("time_zone"),
            "description": persona.get("description"), "permissions": persona.get("permissions"),
            "keep_private": persona.get("keep_private"), "schedule": persona.get("schedule"),
            "response_times": persona.get("response_times"), "max_replies": persona.get("max_replies"),
            # The opening's text, as written; an opening with files also has its parts, as written.
            "opening": opening_text(opening_parts(self.opening_prompt)),
            **({"opening_parts": opening_parts(self.opening_prompt)} if isinstance(self.opening_prompt, list) else {}),
            "user_version": user_sim_versions.VERSION, "model": model, "model_source": source,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_set": True,
            # Who the user is, as written: what user_sim_judge's rubric_persona_sha256 is compared with.
            "persona_sha256": persona_fingerprint(persona, self.opening_prompt),
            # Each tag the prompt shows with its line, so an answer's ``used`` ("D9") names what this run saw.
            "tagged_lines": user_sim_versions.tagged_lines(self.person(context)),
            "multi_day": self.multi_day,
            # The run record's user sim settings, as always_on wrote them when it set the agent up itself.
            "setup": {"model": model, "model_from": source, "user_sim_step_model": self.model, "model_set": True,
                      "system_prompt_chars": len(prompt), "output_format": setup.get("output_format"),
                      "accepted_fields": sorted({"system_prompt", *negotiated.fields})},
        }
        logger.info(f"{self.type} '{self.agent_name}': model {model} ({source}), prompt sha256 "
                    f"{context.metadata[USER_SIMS_KEY][self.agent_name]['prompt_sha256']}")
        return context


# deploy_agent's fields are this step's too, so its signature names them (a scoped field must be one of its own).
DeployUserSimAgentTaskStep.__init__.__signature__ = inspect.Signature([
    *(p for name, p in inspect.signature(DeployUserSimAgentTaskStep.__init__).parameters.items() if name != "deploy"),
    *(p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
      for name, p in inspect.signature(DeployAgentTaskStep.__init__).parameters.items()
      if name in _DEPLOY_FIELDS and name != "system_prompt"),
])
