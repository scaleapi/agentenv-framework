"""Which part a task's step plays: whether it prompts the agent under test, and whether it plays the user."""

from __future__ import annotations

from typing import Any

from agent_env.task_step.registry import get_task_step_registry

# The step types that prompt the agent under test: its agent_name is the agent a run tests.
_PROMPTS_AGENT_UNDER_TEST = frozenset({"prompt_agent", "always_on"})


def _type_of(step: Any) -> str | None:
    """The type of ``step``, a step or a step's config, or None when it names none."""
    kind = step.get("type") if isinstance(step, dict) else getattr(step, "type", None)
    return kind if isinstance(kind, str) else None


def prompts_agent_under_test(step: Any) -> bool:
    """Whether ``step``, a step or a step's config, prompts the agent under test (its ``agent_name``)."""
    return _type_of(step) in _PROMPTS_AGENT_UNDER_TEST


def plays_user(step: Any) -> bool:
    """Whether ``step``, a step or a step's config, deploys the agent that plays the user: its class's ``plays_user``,
    the class looked up by type for a config; False for a type that isn't registered."""
    if isinstance(step, dict):
        kind = _type_of(step)
        cls = get_task_step_registry().get(kind) if kind is not None else None
        return bool(getattr(cls, "plays_user", False))
    return bool(getattr(step, "plays_user", False))
