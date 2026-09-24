"""Task registry for type-based deserialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_env.task.task import Task


def _get_type(cls: type["Task"]) -> str:
    return cls.type


_registry: dict[str, type["Task"]] | None = None


def get_task_registry() -> dict[str, type["Task"]]:
    """Get the task registry, lazily building it to avoid circular imports."""
    global _registry
    if _registry is None:
        from agent_env.task.task import Task

        _registry = {
            _get_type(Task): Task,
        }
    return _registry
