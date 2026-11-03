"""What a trainer's adapter needs without the trainer installed: a tool call's arguments typed by its tool's schema,
and the run group a trainer files its episodes in."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


_JSON_TYPES = {bool: "boolean", int: "integer", float: "number", type(None): "null"}


def typed_arguments(arguments: Any, schema: Mapping[str, Any]) -> Any:
    """``arguments`` with each scalar whose type the tool's schema doesn't allow, but a string, as the string it was
    written as. Renderers that write parameters as untyped text (Qwen3.5's ``<parameter=zip>19122</parameter>``) parse
    each with ``json.loads``, reading digits as a number; Qwen's own parser types values by the schema, as this does."""
    if not isinstance(arguments, dict):
        return arguments
    properties = schema.get("properties") or {}
    typed = dict(arguments)
    for name, value in arguments.items():
        allowed = _allowed_types(properties.get(name) or {})
        kind = _JSON_TYPES.get(type(value))
        widened = kind == "integer" and "number" in allowed
        if kind is not None and "string" in allowed and kind not in allowed and not widened:
            typed[name] = json.dumps(value)
    return typed


def _allowed_types(schema: Mapping[str, Any]) -> set[str]:
    declared = schema.get("type")
    allowed = set(declared) if isinstance(declared, list) else {declared} if declared else set()
    for option in schema.get("anyOf") or schema.get("oneOf") or []:
        allowed |= _allowed_types(option)
    return allowed


def run_group_id(run: str, phase: Literal["train", "eval"], step: int) -> str:
    """The run group a trainer files a group of episodes in: its training run, whether they were played to train or
    to evaluate, and the training step whose policy played them. Every trainer names its groups so, and the explorer
    lists a task's runs by them."""
    return f"{run}/{phase}/step-{step:04d}"


class TrainingSteps:
    """The training step a trainer's batches are played at: a training batch's index, and for an evaluation the step
    after the last training batch handed out, for a trainer that, as ``rl.train`` does, evaluates a step's policy
    before it hands out that step's batch and once more after the last."""

    def __init__(self) -> None:
        self._last_trained: int | None = None

    def trained(self, index: int) -> int:
        self._last_trained = index
        return index

    def evaluated(self) -> int:
        return 0 if self._last_trained is None else self._last_trained + 1


@dataclass(frozen=True)
class RunGroup:
    """Where a trainer files a group of episodes: under the training batch ``trained`` names, fixed when the batch is
    handed out, or, for an evaluation, under the step it is played at, read each time it is played, since a trainer
    may make its evaluation's groups once and play them at every evaluation, as ``rl.train`` does."""

    run: str
    steps: TrainingSteps
    trained: int | None = None

    def id(self) -> str:
        if self.trained is not None:
            return run_group_id(self.run, "train", self.trained)
        return run_group_id(self.run, "eval", self.steps.evaluated())
