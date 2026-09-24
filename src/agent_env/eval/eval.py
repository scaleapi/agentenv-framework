"""Eval model for AgentEnv."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from agent_env.eval.store import EvalQuery


@dataclass
class EvalTask:
    """A reference to a task within an eval. Version is optional (None = latest)."""

    task_id: str
    task_version: Optional[int] = None


class Eval:
    """An eval is a versioned collection of task references.

    Each task reference pins a specific (task_id, task_version) pair.
    Evals are immutable — modifications create new versions.
    """

    type: ClassVar[str] = "eval"

    def __init__(self, id: str, version: Optional[int], tasks: list[EvalTask] | None = None):
        self.id = id
        self.version = version
        self.tasks: list[EvalTask] = tasks or []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "version": self.version,
            "tasks": [asdict(t) for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Eval:
        tasks = [EvalTask(**t) for t in data.get("tasks", [])]
        return cls(id=data["id"], version=data.get("version"), tasks=tasks)

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> Eval:
        from .store import get_eval_store

        return get_eval_store().get(id, version)

    @classmethod
    def put(cls, **kwargs) -> Eval:
        from .store import get_eval_store

        kwargs.setdefault("version", None)
        instance = cls(**kwargs)
        return get_eval_store().put_document(instance)

    @classmethod
    def query(cls) -> EvalQuery:
        from .store import EvalQuery, get_eval_store

        return EvalQuery(get_eval_store())
