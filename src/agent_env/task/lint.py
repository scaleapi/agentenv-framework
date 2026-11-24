"""What a task gets wrong before any agent runs (``agent-env task lint``). An error is a config that can't run or grade
correctly, and refuses the run; everything else is a warning, which blocks nothing."""

from __future__ import annotations

import collections
import copy
import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

from agent_env.entity.refs import RefRole, ref_sites
from agent_env.store.ids import is_local_id
from agent_env.task.task import Task, _ancestor_ids, _dependency_ids
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import StepRole, TaskStep, attach_base_fields
from agent_env.task_step.task_steps.prompt_agent import solver_agents
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import (
    POOLED,
    AggregateVerifiersTaskStep,
    legacy_entry_writers,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator
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
    about: str | None = None  # what the finding names, for a summary to count by: an unpinned reference's entity kind

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
    registration pins it. Reads no store itself; a step's ``config_problems`` may, as load_artifact's do."""
    findings = [
        *(unknown_fields(step_dicts) if step_dicts is not None else ()),
        *(_step_configs(task.steps) if config else ()),
        *_step_warnings(task.steps),
        *_headlines(task),
        *_aggregate_sources(task.steps),
        *_judge_as_solver(task.steps),
        *_lopsided(task.steps),
        *(_unpinned(task.steps) if unpinned else ()),
        *_coverage(task),
    ]
    return sorted(findings, key=lambda finding: finding.severity is not Severity.ERROR)


#: How many of the tasks a finding is in a run names, before how many more.
SHOWN_TASKS = 3
UNPINNED = "unpinned-reference"
#: A pooled source's share of the rows below which it barely moves the score.
LOPSIDED_SHARE = 0.10


def said(findings: Iterable[tuple[str, Finding]]) -> list[str]:
    """``findings``, each with the task it's in, as a run says them: each kind once, errors first, after its task when
    only one has it, else with how many and the first few, so a bundle of a thousand tasks says each kind once."""
    where: dict[tuple[Severity, str], list[str]] = {}
    for task, finding in findings:
        where.setdefault((finding.severity, finding.text), []).append(task)
    lines = []
    for (severity, text), tasks in sorted(where.items(), key=lambda item: item[0][0] is not Severity.ERROR):
        if len(tasks) == 1:
            lines.append(f"{tasks[0]}: {severity.value}: {text}")
        else:
            more = len(tasks) - SHOWN_TASKS
            shown = ", ".join(tasks[:SHOWN_TASKS]) + (f" and {more} more" if more > 0 else "")
            lines.append(f"{severity.value}: {text} ({len(tasks)} tasks: {shown}; agent-env task lint lists each)")
    return lines


def unpinned_summary(findings: Iterable[tuple[str, Finding]]) -> str | None:
    """``findings``' unpinned references as one line, by how many tasks name how many of each kind: an eval of
    thousands of tasks naming the latest of their agents and artifacts, as the hub's do, would otherwise say one line
    for each, since each names its own step. None when there are none."""
    tasks, kinds = set(), collections.Counter()
    for task, finding in findings:
        if finding.rule == UNPINNED:
            tasks.add(task)
            kinds[finding.about] += 1
    if not tasks:
        return None
    total = sum(kinds.values())
    counts = ", ".join(f"{kind} {n:,}" for kind, n in kinds.most_common())
    return (f"warning: {len(tasks):,} task{'s' * (len(tasks) != 1)} name{'s' * (len(tasks) == 1)} {total:,} store "
            f"entit{'ies' if total != 1 else 'y'} with no version, so each run takes the latest ({counts}); "
            "agent-env task lint --id TASK lists a task's")


def refuses(finding: Finding, *, skip_lint: bool = False) -> bool:
    """Whether ``finding`` refuses a run. ``skip_lint`` lifts lint's own rules, not a step's ``config_problems``, which
    refused a run before lint did and fail it when it runs."""
    return finding.severity is Severity.ERROR and not (skip_lint and finding.rule != "step-config")


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.severity is Severity.ERROR]


def is_verifier(step: TaskStep) -> bool:
    """A step that grades: one on the VerifierTaskStep base, or one that still writes its own legacy entry, a grader (as
    agent-env's validators are) or a step with a ``verifier_id`` (as agentenvhub-sdk's are)."""
    return (isinstance(step, VerifierTaskStep) or step.step_role is StepRole.GRADER
            or bool(getattr(step, "verifier_id", None)))


def _own(step: TaskStep) -> bool:
    """One of agent-env's own steps, whose behaviour lint knows."""
    return type(step).__module__.partition(".")[0] == "agent_env"


def _seen_into(step: TaskStep) -> bool:
    """A step whose verification writes lint knows: one on the base writes under its ``verifier_id``, and another of
    agent-env's own writes none unless it's a grader, whose keys are its own. An outside step off the base may write
    any key, as agentenvhub-sdk's ``run_lighthouse_audit`` writes under its ``verification_id``."""
    return isinstance(step, VerifierTaskStep) or (_own(step) and step.step_role is not StepRole.GRADER)


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
        if isinstance(step, VerifierTaskStep):
            continue
        if _own(step) and not is_verifier(step):
            yield Finding(Severity.ERROR, "headline-not-a-verifier",
                          f"headline_verifiers names {headline!r}, a {step.type} step, which grades nothing")
        else:
            yield Finding(Severity.WARNING, "headline-without-grade",
                          f"headline_verifiers names {headline!r}, a {step.type} step that writes no named grade, "
                          "so a run's result will read it as ungraded")


def _aggregate_sources(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """An aggregate source that no verifier in the task writes, which fails the aggregate when it grades: an error when
    lint sees every key the task's steps write, else a warning naming the steps it can't see into."""
    written, unseen = legacy_entry_writers(steps), [step.id for step in steps if not _seen_into(step)]
    for step in steps:
        if isinstance(step, AggregateVerifiersTaskStep) and step.names_steps:
            yield from _named_sources(step, steps)
        elif isinstance(step, AggregateVerifiersTaskStep):
            missing = [source for source in step.verifier_ids if isinstance(source, str) and source not in written]
            if missing and not unseen:
                yield Finding(Severity.ERROR, "aggregate-source",
                              f"verifier_ids names {missing!r}, which no verifier in this task writes (it has "
                              f"{sorted(written)})", step.id)
            elif missing:
                yield Finding(Severity.WARNING, "aggregate-source",
                              f"verifier_ids names {missing!r}, which no verifier in this task names as its "
                              f"verifier_id; unless {unseen!r} write it, under keys lint can't see, the aggregate fails "
                              "when it grades", step.id)


def _named_sources(aggregate: AggregateVerifiersTaskStep, steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """Each source or gate an aggregate names by step id that a run can't read a grade of: no step in the task, a step
    off the VerifierTaskStep base, which writes none, or one that doesn't run before it. Its own config problems, such
    as naming itself, are its ``config_problems``."""
    by_id, before = {step.id: step for step in steps}, _ancestor_ids(aggregate.id, _dependency_ids(list(steps)))
    for field in ("sources", "gates"):
        names = getattr(aggregate, field)
        if not isinstance(names, list):
            continue
        for name in dict.fromkeys(name for name in names if isinstance(name, str) and name != aggregate.id):
            source = by_id.get(name)
            if source is None:
                problem = "which is no step in this task"
            elif not isinstance(source, VerifierTaskStep):
                problem = (f"a {source.type} step, which writes no named grade: an aggregate's {field} are verifiers "
                           "on the VerifierTaskStep base, and one of other steps' entries names them in verifier_ids")
            elif name not in before:
                problem = (f"which doesn't run before {aggregate.id!r}, so its grade isn't there when the aggregate "
                           "reads it")
            else:
                continue
            yield Finding(Severity.ERROR, "aggregate-source", f"{field} names {name!r}, {problem}", aggregate.id)


def judges_in_solver(steps: Sequence[TaskStep]) -> list[RubricsVerifierTaskStep]:
    """The agent judges that run in the solver's own agent: their ``agent_name`` names an agent a step prompts as the
    agent under test (``solver_agents``), so the solver grades its own work."""
    solver = solver_agents(steps)
    return [step for step in steps if isinstance(step, RubricsVerifierTaskStep) and step.use_agent_judge
            and isinstance(step.agent_name, str) and step.agent_name in solver]


def _judge_as_solver(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """A warning, not an error: the run still grades, but its grade is the solver's own, which may be the point."""
    for judge in judges_in_solver(steps):
        yield Finding(Severity.WARNING, "judge-as-solver",
                      f"its judge runs in {judge.agent_name!r}, the agent under test, so the solver grades its own work; "
                      "for a judge of its own, deploy a judge agent, load the files it grades into it (load_artifact "
                      "with artifact_from_step_id) and name it in agent_name", judge.id)


def _lopsided(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """A pooled, weighted-average aggregate one of whose sources is under ``LOPSIDED_SHARE`` of its rows, so it barely
    moves the score, counting only the sources whose rows their config says (``criterion_rows``). An aggregate of
    ``verifier_ids`` reads a nested aggregate's entry, which has none, so that source adds nothing; a rubric's
    trajectory-mistakes entry has rows only a run can tell."""
    by_id, writers = {step.id: step for step in steps}, legacy_entry_writers(steps)
    for step in steps:
        if (not isinstance(step, AggregateVerifiersTaskStep)
                or step.score_aggregator is not ScoreAggregator.WEIGHTED_AVERAGE):
            continue
        if step.names_steps:
            if (step.combine or POOLED) != POOLED or not isinstance(step.sources, list):
                continue
            field, sources = "sources", {name: by_id.get(name) for name in step.sources if isinstance(name, str)}
        else:
            field, sources = "verifier_ids", {vid: by_id.get(writers.get(vid)) for vid in step.verifier_ids
                                              if isinstance(vid, str)}
        rows = {}
        for name, source in sources.items():
            if not step.names_steps and getattr(source, "verifier_id", None) != name:
                continue
            if isinstance(source, AggregateVerifiersTaskStep) and not step.names_steps:
                rows[name] = 0
            elif isinstance(source, VerifierTaskStep) and (count := source.criterion_rows()) is not None:
                rows[name] = count
        total = sum(rows.values())
        if len(rows) < 2 or not total:
            continue
        name, count = min(rows.items(), key=lambda item: item[1])
        if count / total >= LOPSIDED_SHARE:
            continue
        moves = ("adds no rows to the pool, so it doesn't move the score" if not count else
                 f"is {count} of the {total} rows it pools ({count / total:.0%}), so it barely moves the score")
        fix = ("combine: weighted_mean weighs each source equally" if step.names_steps else
               "naming the sources in sources with combine: weighted_mean weighs each equally")
        yield Finding(Severity.WARNING, "lopsided-pooling", f"{field} source {name!r} {moves}; {fix}", step.id)


def _unpinned(steps: Sequence[TaskStep]) -> Iterator[Finding]:
    """A store entity a step reads with no version, so each run takes whatever is latest. A bundle's own entities and
    installed bundles' are local ids, which their bundles write and pin."""
    for step in steps:
        for site in ref_sites(type(step).entity_refs or (), step.to_dict()):
            if (site.ref.role is RefRole.INPUT and site.version_key is not None and site.version is None
                    and isinstance(site.value, str) and site.value and not is_local_id(site.value)):
                yield Finding(Severity.WARNING, UNPINNED,
                              f"{site.path} names {site.ref.kind.value} {site.value!r} with no version, so each run "
                              f"takes its latest; set {site.version_key} to pin it", step.id, site.ref.kind.value)


def _coverage(task: Task) -> Iterator[Finding]:
    verifiers = [step for step in task.steps if is_verifier(step)]
    if not verifiers and all(_own(step) for step in task.steps):
        yield Finding(Severity.WARNING, "no-verifier", "no step grades this task, so a run has no score")
    elif len(verifiers) > 1 and not task.headline_verifiers:
        yield Finding(Severity.WARNING, "no-headlines", _no_headlines(verifiers, task.steps))


def _no_headlines(verifiers: list[TaskStep], steps: Sequence[TaskStep]) -> str:
    """Several verifiers and no headline: every grade is top-line. Name the aggregate that covers the others, if one
    does."""
    graders = {step.id for step in verifiers
               if isinstance(step, VerifierTaskStep) or isinstance(getattr(step, "verifier_id", None), str)}
    covering = [step.id for step in verifiers if isinstance(step, AggregateVerifiersTaskStep)
                and graders - {step.id} <= set(step.source_steps(steps))]
    suggestion = (f"; {covering[0]!r} aggregates the others, so name it in headline_verifiers" if covering
                  else "; name the one that is its reward in headline_verifiers")
    return (f"{len(verifiers)} verifiers and no headline_verifiers, so every grade is top-line and anything that needs "
            f"one number must name one{suggestion}")
