"""Whether a task's step plays the user. Whether it prompts an agent is ``prompt_agent.is_prompt_step``."""

from __future__ import annotations

from typing import Any

from agent_env.task_step.registry import get_task_step_registry


def _type_of(step: Any) -> str | None:
    """The type of ``step``, a step or a step's config, or None when it names none."""
    kind = step.get("type") if isinstance(step, dict) else getattr(step, "type", None)
    return kind if isinstance(kind, str) else None


def plays_user(step: Any) -> bool:
    """Whether ``step``, a step or a step's config, deploys the agent that plays the user: its class's ``plays_user``,
    the class looked up by type for a config; False for a type that isn't registered."""
    if isinstance(step, dict):
        kind = _type_of(step)
        cls = get_task_step_registry().get(kind) if kind is not None else None
        return bool(getattr(cls, "plays_user", False))
    return bool(getattr(step, "plays_user", False))
