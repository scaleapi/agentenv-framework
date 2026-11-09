"""The user_sim step: who the user is, once per task. Its settings go into ``metadata["user_sims"][name]``, where the
always_on step reads them. This docstring is the whole interface between the two.

What always_on reads from the settings (``metadata["user_sims"][name]``):

- ``name``; ``display_name`` (else ``name``) and ``first_name`` (else the display name's first word): how the agent's
  wakes and the plug-in name the user.
- ``agent_name``: the deployed agent that plays the user (``ask`` below sends to it); ``model``: set on that agent,
  unless the run overrides it. The model used is this step's ``step_params`` ``model`` (found by the ``step_id`` the
  step records), else the run's ``user_model`` (``task run`` / ``eval run --user-model``, a bundle's ``user_model``),
  else this step's ``model``.
- ``user_version``: the version handed to the plug-in; ``plugin``: ``{"file_artifact_id", "file_artifact_version",
  "params"}`` naming a plug-in file, else the built-in ``user_sim_versions``.
- ``time_zone`` (IANA): the user's zone; ``schedule`` (``wake``, ``sleep``, ``work``, ``work_days``, ``busy``): when
  the user reads, where ``sleep`` may be past midnight (wake 08:00, sleep 00:30); ``response_times`` (minutes by
  ``morning`` / ``work`` / ``free``): how long a reply is held; ``max_replies``: after that many replies the user
  stops reading. ``schedule`` and ``response_times`` need the task's clock (``sync_env_clock``): always_on refuses to
  start a run that sets either with no clock armed.
- ``email``, ``description``, ``versions``, ``output_format``, ``permissions``, ``keep_private``: passed to the
  plug-in in ``person`` only. For the built-in plug-in, ``description`` is an object of sections; user_sim.md says
  how to write one. Its settings are checked when the task loads, as are ``schedule``, ``response_times`` and
  ``max_replies`` for any plug-in.

A plug-in is one Python module with:

    configure(params: dict) -> None                       optional, called once with plugin.params
    setup(version: str, person: dict) -> {"system_prompt": str | None, "output_format": dict | None}
    async answer(version: str, person: dict, history: list[dict], read: dict, ask) -> dict

``setup`` runs once per run; the step sets ``model``, ``system_prompt`` and ``output_format`` on the user sim agent.
``person`` is ``{"key": name, "name", "first", "email", "time_zone", "description", "versions", "output_format",
"permissions", "keep_private", "opening", "done_ends_run"}``, with ``time_zone`` the run's zone (the user's, else the
world clock's, else UTC) and ``opening`` the user's opening request, ``{"at": RFC 3339 UTC, "text"}``, so ``setup``
can write it into the system prompt. The opening is also the first line of the first read's ``you_sent``. The step
sets ``done_ends_run`` before ``setup``, as each read has it, since the done line is in the system prompt; the
built-in plug-in uses the multi-day wording when it's missing.

``answer`` runs once per read, at the time the user would read. ``read`` is ``{"time": RFC 3339 UTC,
"assistant_messages": [{"at", "text"}], "phone": [{"at", "channel", "text"}], "you_sent": [{"at", "text"}],
"done_ends_run": bool}``: the agent's message_user texts since the last read, oldest first; what triggers sent
``user_phone`` since the last read, its sender in ``channel`` (``"Gmail · Carol Reyes (Mom)"``) as in the agent's
alerts; the user's own lines (the opening request, ``user_says`` lines, ``final``) each at the time it was sent; never
anything meant for the agent; and whether done ends the run (one-sitting) or is only recorded (multi-day). The user
sim reads only after the agent has used message_user. ``history`` is every earlier
``{"read", "answer"}``, oldest first. ``ask(text)`` sends one turn to the user sim agent (one session per run) and
returns ``{"text", "structured", "trajectory_url"}``, or ``{"text": None, "structured": None, "error"}``.

``answer`` returns ``{"outcome": "reply" | "no_reply" | "done", "message": str, "used": str | None}``, and may add
``input``, ``raw``, ``score``, ``schema_errors``, ``turns``, ``retried``, ``first_answer``, ``shown`` and
``past_turn_cap``, which the step records. A ``reply`` or ``done`` with text is held for the response time, then
wakes the agent as the user's message; the recorded extras (``used``, ``score``, ...) never reach the agent. A reply
that never reaches the agent (one still held when the run ends, or the answer to the last read below) is recorded with
``delivered: false``. ``done`` ends a one-sitting run after the agent's turn on that message, or at once when it has
no text, with no further reads; in a multi-day run it is only recorded.
When the last ``ask`` of a read returned an error, the run ends (``user_sim_error``).

After the run's last wake (the final check-in, or the wake ``max_wakes`` makes the last), the user reads the agent's
wrap-up once more, if it sent one: one more ``answer``, its ``read`` shaped like any other, with the task's ``final``
line, when it sets one, in ``you_sent``. Its answer is recorded and never delivered; a ``done`` or an error on it
changes nothing.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, ClassVar, Optional

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps import user_sim_versions

logger = logging.getLogger(__name__)

USER_SIMS_KEY = "user_sims"
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
SCHEDULE_KEYS = ("wake", "sleep", "work", "work_days", "busy")
RESPONSE_TIME_KEYS = ("morning", "work", "free")


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


class UserSimTaskStep(TaskStep):
    """Who the user is, once per task: their settings go into the run's metadata under ``name``."""

    type: ClassVar[str] = "user_sim"
    entity_refs = (EntityRef.artifact("plugin.file_artifact_id", version_field="file_artifact_version",
                                      artifact_type="file"),)

    def __init__(self, id: str, version: Optional[int], *, name: str, model: str, agent_name: Optional[str] = None,
                 user_version: str = "ours", display_name: Optional[str] = None, first_name: Optional[str] = None,
                 description: Optional[dict] = None, email: Optional[str] = None, time_zone: Optional[str] = None,
                 schedule: Optional[dict] = None, response_times: Optional[dict] = None,
                 max_replies: Optional[int] = None, permissions: Optional[dict] = None,
                 keep_private: Optional[list] = None, output_format: Optional[dict] = None,
                 versions: Optional[dict] = None, plugin: Optional[dict] = None, depends_on=None,
                 fail_task_on_error: bool = True, retry_config=None):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         retry_config=retry_config)
        if not model:
            raise ValueError("user_sim.model is required")
        found = schedule_problems(schedule, response_times, max_replies)
        if time_zone is not None:
            found += user_sim_versions.zone_problems(time_zone)
        if not plugin:
            if user_version != user_sim_versions.VERSION:
                found.append(f"user_version {user_version!r}: the built-in user sim has one version, "
                             f"{user_sim_versions.VERSION!r}")
            found += user_sim_versions.problems({"name": display_name or name, "first": first_name,
                                                 "description": description, "permissions": permissions,
                                                 "keep_private": keep_private})
        if found:
            raise ValueError(f"user_sim '{name}': " + "; ".join(found))
        self.name = name
        self.model = model
        self.agent_name = agent_name or name
        self.user_version = user_version
        self.display_name = display_name
        self.first_name = first_name
        self.description = description
        self.email = email
        self.time_zone = time_zone
        self.schedule = schedule
        self.response_times = response_times
        self.max_replies = max_replies
        self.permissions = permissions
        self.keep_private = keep_private
        self.output_format = output_format
        self.versions = versions or {}
        self.plugin = plugin

    _FIELDS = ("name", "model", "agent_name", "user_version", "display_name", "first_name", "description", "email",
               "time_zone", "schedule", "response_times", "max_replies", "permissions", "keep_private", "output_format",
               "versions", "plugin")

    def to_dict(self) -> dict:
        return {**super().to_dict(), **{k: getattr(self, k) for k in self._FIELDS}}

    @classmethod
    def from_dict(cls, data: dict) -> UserSimTaskStep:
        return cls(**cls._base_from_dict(data), retry_config=data.get("retry_config"),
                   **{k: data.get(k) for k in cls._FIELDS if k in data})

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        overrides = self.step_param_overrides(context)
        version = overrides.get("user_version") or self.user_version
        if not any(a.agent_name == self.agent_name for a in context.deployed_agents):
            raise RuntimeError(f"user_sim '{self.name}': no deployed agent named '{self.agent_name}'; deploy it first")
        settings = {k: getattr(self, k) for k in self._FIELDS}
        settings.update(user_version=version, step_id=self.id,
                        model=overrides.get("model") or self.model)
        context.metadata.setdefault(USER_SIMS_KEY, {})[self.name] = settings
        logger.info(f"user_sim '{self.name}': version {version}, model {settings['model']}, agent {self.agent_name}")
        return context
