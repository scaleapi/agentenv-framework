"""What a task gets wrong before any agent runs (``agent-env task lint``). An error is a config that can't run or grade
correctly, and refuses the run; everything else is a warning, which blocks nothing."""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

from agent_env.entity.refs import RefRole, ref_sites
from agent_env.store.ids import is_local_id
from agent_env.task.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep, attach_base_fields
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import JudgeOutputFormat
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep, trajectory_mistakes_key
from agent_env.task_step.task_steps.verifiers.verifier import VerifierTaskStep


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    rule: str
    message: str
    step_id: str | None = None

    @property
    def text(self) -> str:
        """The message with the step it's about, as a refusal lists it."""
        return f"step {self.step_id!r}: {self.message}" if self.step_id is not None else self.message

    def __str__(self) -> str:
        return f"{self.severity.value}: {self.text}"


def lint(task: Task, step_dicts: Sequence[Mapping[str, Any]] | None = None, *, config: bool = True,
         unpinned: bool = True) -> list[Finding]:
    """``task``'s findings, errors first. ``step_dicts``, the task's steps as a file holds them, adds the unknown-field
    rule, which a stored task can't break since storing drops unknown fields. ``config`` adds each step's own
    ``config_problems`` as errors; a caller that runs ``Task.preflight`` already has them. ``unpinned`` warns for a store
    entity read with no version; a bundle's plan leaves it out, since a plan records each version it reads and a
    registration pins it. Reads no store."""
    findings = [
        *(unknown_fields(step_dicts) if step_dicts is not None else ()),
        *(_step_configs(task.steps) if config else ()),
        *_step_warnings(task.steps),
        *_headlines(task),
        *_aggregate_sources(task.steps),
        *(_unpinned(task.steps) if unpinned else ()),
        *_coverage(task),
    ]
    return sorted(findings, key=lambda finding: finding.severity is not Severity.ERROR)


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.severity is Severity.ERROR]


def is_verifier(step: TaskStep) -> bool:
    """A step that grades: one on the VerifierTaskStep base, or one that still writes a legacy entry under its
    ``verifier_id`` itself, as agentenvhub-sdk's do until they move onto the base."""
    return isinstance(step, VerifierTaskStep) or bool(getattr(step, "verifier_id", None))


class _Reads(dict):
    """A step dict that records each key its type's ``from_dict`` asks for, read or absent; spreading or iterating it
    counts as reading every key."""

    def __init__(self, data: Mapping[str, Any]):
        super().__init__(data)
        self.asked: set[str] = set()

    def __getitem__(self, key):
        self.asked.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.asked.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.asked.add(key)
        return super().__contains__(key)

    def pop(self, key, *default):
        self.asked.add(key)
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        self.asked.add(key)
        return super().setdefault(key, default)

    def __iter__(self):
        self.asked.update(super().keys())
        return super().__iter__()

    def keys(self):
        self.asked.update(super().keys())
        return super().keys()

    def items(self):
        self.asked.update(super().keys())
        return super().items()

    def values(self):
        self.asked.update(super().keys())
        return super().values()


def unknown_fields(step_dicts: Sequence[Mapping[str, Any]]) -> Iterator[Finding]:
    """Each key of a step that its type never asks for when it reads the step, with the nearest field it asks for."""
    registry = get_task_step_registry()
    for step in step_dicts:
        cls = registry.get(step.get("type"))
        if cls is None:
            continue
        tracked = _Reads(copy.deepcopy(dict(step)))
        try:
            attach_base_fields(cls.from_dict(tracked), tracked)
        except Exception:
            continue  # a step that can't be read is reported where the task is built
        for key in sorted(set(step) - tracked.asked - {"type"}):
            near = difflib.get_close_matches(key, sorted(tracked.asked - set(step)), n=1)
            hint = f" (did you mean {near[0]!r}?)" if near else ""
            yield Finding(Severity.ERROR, "unknown-field",
                          f"{cls.type} doesn't read {key!r}{hint}, so it would be ignored", step.get("id"))


def _step_configs(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    for step in steps:
        for problem in step.config_problems():
            yield Finding(Severity.ERROR, "step-config", problem, step.id)


def _step_warnings(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    for step in steps:
        for doubt in step.config_warnings():
            yield Finding(Severity.WARNING, "step-config", doubt, step.id)


def _headlines(task: Task) -> Iterator[Finding]:
    by_id = {step.id: step for step in task.steps}
    for headline in task.headline_verifiers or ():
        step = by_id.get(headline)
        if step is None:
            continue  # Task refuses it when it's built
        if not is_verifier(step):
            yield Finding(Severity.ERROR, "headline-not-a-verifier",
                          f"headline_verifiers names {headline!r}, a {step.type} step, which grades nothing")
        elif not isinstance(step, VerifierTaskStep):
            yield Finding(Severity.WARNING, "headline-without-grade",
                          f"headline_verifiers names {headline!r}, a {step.type} step that writes no named grade yet, "
                          "so a run's result will read it as ungraded")


def _aggregate_sources(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """An aggregate source that no verifier in the task writes, which fails the aggregate when it grades."""
    written = set()
    for step in steps:
        if is_verifier(step):
            written.add(step.verifier_id)
            if (isinstance(step, RubricsVerifierTaskStep)
                    and step.output_format is JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES):
                written.add(trajectory_mistakes_key(step.verifier_id))
    for step in steps:
        if isinstance(step, AggregateVerifiersTaskStep):
            missing = [source for source in step.verifier_ids if source not in written]
            if missing:
                yield Finding(Severity.ERROR, "aggregate-source",
                              f"verifier_ids names {missing!r}, which no verifier in this task writes (it has "
                              f"{sorted(written)})", step.id)


def _unpinned(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """A store entity a step reads with no version, so each run takes whatever is latest. A bundle's own entities and
    installed bundles' are local ids, which their bundles write and pin."""
    for step in steps:
        for site in ref_sites(type(step).entity_refs or (), step.to_dict()):
            if (site.ref.role is RefRole.INPUT and site.version_key is not None and site.version is None
                    and isinstance(site.value, str) and site.value and not is_local_id(site.value)):
                yield Finding(Severity.WARNING, "unpinned-reference",
                              f"{site.path} names {site.ref.kind.value} {site.value!r} with no version, so each run "
                              f"takes its latest; set {site.version_key} to pin it", step.id)


def _coverage(task: Task) -> Iterator[Finding]:
    verifiers = [step for step in task.steps if is_verifier(step)]
    if not verifiers:
        yield Finding(Severity.WARNING, "no-verifier", "no step grades this task, so a run has no score")
    elif len(verifiers) > 1 and not task.headline_verifiers:
        yield Finding(Severity.WARNING, "no-headlines", _no_headlines(verifiers))


def _no_headlines(verifiers: list[TaskStep]) -> str:
    """Several verifiers and no headline: every grade is top-line. Name the aggregate that covers the others, if one
    does."""
    ids = {step.verifier_id for step in verifiers}
    covering = [step.id for step in verifiers if isinstance(step, AggregateVerifiersTaskStep)
                and ids - {step.verifier_id} <= set(step.verifier_ids)]
    suggestion = (f"; {covering[0]!r} aggregates the others, so name it in headline_verifiers" if covering
                  else "; name the one that is its reward in headline_verifiers")
    return (f"{len(verifiers)} verifiers and no headline_verifiers, so every grade is top-line and anything that needs "
            f"one number must name one{suggestion}")
