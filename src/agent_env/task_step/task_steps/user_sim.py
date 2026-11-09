"""The user_sim step: who the user is, once per task. Its settings go into ``metadata["user_sims"][name]``, where the
always_on step reads them. This docstring is the whole interface between the two.

What always_on reads from the settings (``metadata["user_sims"][name]``):

- ``name``; ``display_name`` (else ``name``) and ``first_name`` (else the display name's first word): how the agent's
  wakes and the plug-in name the user.
- ``agent_name``: the deployed agent that plays the user (``ask`` below sends to it); ``model``: set on that agent.
- ``user_version``: the version handed to the plug-in; ``plugin``: ``{"file_artifact_id", "file_artifact_version",
  "params"}`` naming a plug-in file, else the built-in ``user_sim_versions``.
- ``time_zone`` (IANA): the user's zone; ``schedule`` (``wake``, ``sleep``, ``work``, ``work_days``, ``busy``): when
  the user reads; ``response_times`` (minutes by ``morning`` / ``work`` / ``free``): how long a reply is held;
  ``max_replies``: after that many replies the user stops reading.
- ``email``, ``description``, ``versions``, ``output_format``, ``permissions``, ``keep_private``: passed to the
  plug-in in ``person`` only.

A plug-in is one Python module with:

    configure(params: dict) -> None                       optional, called once with plugin.params
    setup(version: str, person: dict) -> {"system_prompt": str | None, "output_format": dict | None}
    async answer(version: str, person: dict, history: list[dict], read: dict, ask) -> dict

``setup`` runs once per run; the step sets ``model``, ``system_prompt`` and ``output_format`` on the user sim agent.
``person`` is ``{"key": name, "name", "first", "email", "time_zone", "description", "versions", "output_format",
"permissions", "keep_private"}``, with ``time_zone`` the run's zone (the user's, else the world clock's, else UTC).

``answer`` runs once per read, at the time the user would read. ``read`` is ``{"time": RFC 3339 UTC,
"assistant_messages": [{"at", "text"}], "phone": [{"at", "channel", "text"}], "you_sent": [{"at", "text"}]}``: the
agent's message_user texts since the last read, what triggers sent ``user_phone``, and the user's own lines (the
opening request, ``user_says`` lines, ``final``), never anything meant for the agent. ``history`` is every earlier
``{"read", "answer"}``, oldest first. ``ask(text)`` sends one turn to the user sim agent (one session per run) and
returns ``{"text", "structured", "trajectory_url"}``, or ``{"text": None, "structured": None, "error"}``.

``answer`` returns ``{"outcome": "reply" | "no_reply" | "done", "message": str, "used": str | None}``, and may add
``input``, ``raw``, ``score``, ``schema_errors``, ``turns``, ``shown`` and ``past_turn_cap``, which the step records.
A reply with text is held for the response time, then wakes the agent; ``done`` is never sent to the agent and ends
only a one-sitting run. When the last ``ask`` of a read returned an error, the run ends (``user_sim_error``).
"""
from __future__ import annotations

import logging
from typing import ClassVar, Optional
from zoneinfo import ZoneInfo

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep

logger = logging.getLogger(__name__)

USER_SIMS_KEY = "user_sims"


class UserSimTaskStep(TaskStep):
    """Who the user is, once per task: their settings go into the run's metadata under ``name``."""

    type: ClassVar[str] = "user_sim"
    entity_refs = (EntityRef.artifact("plugin.file_artifact_id", version_field="file_artifact_version",
                                      artifact_type="file"),)

    def __init__(self, id: str, version: Optional[int], *, name: str, model: str, agent_name: Optional[str] = None,
                 user_version: str = "ours", display_name: Optional[str] = None, first_name: Optional[str] = None,
                 description: Optional[str] = None, email: Optional[str] = None, time_zone: Optional[str] = None,
                 schedule: Optional[dict] = None, response_times: Optional[dict] = None,
                 max_replies: Optional[int] = None, permissions: Optional[dict] = None,
                 keep_private: Optional[list] = None, output_format: Optional[dict] = None,
                 versions: Optional[dict] = None, plugin: Optional[dict] = None, depends_on=None,
                 fail_task_on_error: bool = True, retry_config=None):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         retry_config=retry_config)
        if not model:
            raise ValueError("user_sim.model is required")
        if time_zone is not None:
            ZoneInfo(time_zone)
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
