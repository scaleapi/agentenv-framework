"""How a task's recorded scores make its reward: the rules ``build_app`` applies, which harness training shares."""

from __future__ import annotations

from collections.abc import Callable

from agent_env.task import Task


def reward_of(task: Task, reward: str | Callable[[dict[str, float]], float] | None
               ) -> tuple[Callable[[dict[str, float]], float | None] | None, str | None]:
    """``task``'s reward as a function of its scores, or why ``reward`` can't make one. A candidate is a step that
    records a score (it has a ``verifier_id``) unless its class sets ``can_be_reward = False``: its score checks the
    run rather than grading the agent, so it is never picked, and is refused as ``reward=``."""
    if callable(reward):
        return reward, None
    recording = [step for step in task.steps if getattr(step, "verifier_id", None)]
    scoring = [step.id for step in recording if getattr(step, "can_be_reward", True)]
    if reward is None:
        if len(scoring) > 1:
            return None, (f"task {task.id!r} records a score at each of {', '.join(scoring)}; name the one that is "
                          "its reward with reward=<step id>")
        return (_score_at(scoring[0]) if scoring else _unscored), None
    if reward not in scoring:
        if checking := next((step for step in recording if step.id == reward), None):
            return None, (f"task {task.id!r}'s step {reward!r} ({checking.type}) records a score that checks the run "
                          f"and can't be its reward; the scores that can are at {scoring}")
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
