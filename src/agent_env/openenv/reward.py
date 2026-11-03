"""How a task's recorded scores make its reward: the rules ``build_app`` applies, which harness training shares."""

from __future__ import annotations

from collections.abc import Callable

from agent_env.task import Task


def reward_of(task: Task, reward: str | Callable[[dict[str, float]], float] | None
               ) -> tuple[Callable[[dict[str, float]], float | None] | None, str | None]:
    """``task``'s reward as a function of its scores, or why ``reward`` can't make one."""
    if callable(reward):
        return reward, None
    scoring = [step.id for step in task.steps if getattr(step, "verifier_id", None)]
    if reward is None:
        if len(scoring) > 1:
            return None, (f"task {task.id!r} records a score at each of {', '.join(scoring)}; name the one that is "
                          "its reward with reward=<step id>")
        return (_score_at(scoring[0]) if scoring else _unscored), None
    if reward not in scoring:
        return None, f"task {task.id!r} records no score at step {reward!r}; it records them at {scoring}"
    return _score_at(reward), None


def _score_at(step: str) -> Callable[[dict[str, float]], float]:
    def score(scores: dict[str, float]) -> float:
        if step not in scores:
            raise LookupError(f"the run recorded no score at step {step!r}")
        return scores[step]

    return score


def _unscored(scores: dict[str, float]) -> None:
    return None
