"""The statistics behind an eval's results: the task is the unit, Student's t over per-task means, Wilson for values
that are all 0 or 1. Standard library only, so it runs in core and in any service that reads stored runs; a change to
what any of these functions returns is a new STATS_VERSION."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from math import fsum, isfinite, lgamma, log, log1p, sqrt
from statistics import NormalDist
from typing import Mapping, Optional, Sequence, Tuple, Union

STATS_VERSION = 1

_Z975 = 1.959963984540054
_NO_SPREAD = 1e-12
_CORNISH_FISHER_DF = 10_000
_NEWTON_STEPS = 200
_NEWTON_TOLERANCE = 1e-13
_FRACTION_TERMS = 10_000
_FRACTION_TOLERANCE = 1e-16
_TINY = 1e-300
_HALF_LN_PI = 0.5 * log(math.pi)

TaskKey = Tuple[str, int]


class RunState(StrEnum):
    GRADED = "graded"
    UNGRADED = "ungraded"
    NOT_RUN = "not_run"
    PENDING = "pending"


class GradeStatus(StrEnum):
    GRADED = "graded"
    SKIPPED = "skipped"
    FAILED = "failed"


class Method(StrEnum):
    T = "t"
    WILSON = "wilson"
    NONE = "none"


class Basis(StrEnum):
    TASKS = "tasks"
    RUNS = "runs"


@dataclass(frozen=True)
class GradeValue:
    """One named grade of a run as its verifier stored it: score is None unless the status is graded."""

    status: GradeStatus
    score: Optional[float] = None
    passed: bool = False
    min_passing_score: float = 1.0


@dataclass(frozen=True)
class RunValue:
    """One run as Results reads it, built from an eval slot or a stored run. `cause` is opaque and only counted;
    `agent_failed` makes a skipped grade count as 0; `passed` is the run's own result, which a pass rule over every
    grade reads."""

    task_id: str
    task_version: int
    state: RunState
    grades: Mapping[str, GradeValue] = field(default_factory=dict)
    cause: Optional[str] = None
    passed: Optional[bool] = None
    agent_failed: bool = False
    replica: Optional[int] = None
    run_id: Optional[str] = None
    agent: Optional[str] = None
    models: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskValues:
    """The values one task's runs give one grade, sorted, so the order runs arrive in changes nothing."""

    task_id: str
    task_version: int
    scores: Tuple[float, ...]
    passes: Tuple[bool, ...]

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return fsum(self.scores) / len(self.scores)

    @property
    def passed_runs(self) -> int:
        return sum(self.passes)


@dataclass(frozen=True)
class Counts:
    """Why each run did or didn't give a grade a value. `graded` counts the values, the zeros after an agent failure
    (`zero_after_agent_failure`) included; `invalid_score` counts graded grades whose score isn't a finite number in
    [0, 1]."""

    graded: int = 0
    zero_after_agent_failure: int = 0
    ungraded: Mapping[str, int] = field(default_factory=dict)
    not_run: Mapping[str, int] = field(default_factory=dict)
    pending: int = 0
    skipped: int = 0
    failed: int = 0
    invalid_score: int = 0
    no_such_grade: int = 0


@dataclass(frozen=True)
class PerTaskValues:
    """A grade's values task by task (ordered by task id and version), with what was left out."""

    tasks: Tuple[TaskValues, ...]
    counts: Counts
    no_graded_run: Tuple[TaskKey, ...]
    no_such_grade: Tuple[TaskKey, ...]


@dataclass(frozen=True)
class Estimate:
    """A mean with its 95% interval. `basis` is tasks, or runs for an instance of one task (run to run); `n` counts
    the values the interval is over. A single value has no interval."""

    value: float
    lo: Optional[float]
    hi: Optional[float]
    method: Method
    basis: Basis
    n: int


def per_task_values(
    runs: Sequence[RunValue],
    grade: Union[str, Mapping[TaskKey, str]],
    *,
    min_passing_score: Optional[float] = None,
) -> PerTaskValues:
    """Each task's values of `grade`, a name or each task's own grade by (task id, version). Only graded runs give
    values; a skipped grade gives 0 after an agent failure, and `min_passing_score` re-derives `passed` from the
    score."""
    if min_passing_score is not None and not 0.0 < min_passing_score <= 1.0:
        raise ValueError(f"min_passing_score must be above 0 and at most 1, got {min_passing_score!r}")
    by_task: dict[TaskKey, list[RunValue]] = {}
    for run in runs:
        by_task.setdefault((run.task_id, run.task_version), []).append(run)
    tally: Counter[str] = Counter()
    ungraded: Counter[str] = Counter()
    not_run: Counter[str] = Counter()
    tasks: list[TaskValues] = []
    no_graded_run: list[TaskKey] = []
    no_such_grade: list[TaskKey] = []
    for key in sorted(by_task):
        name = grade if isinstance(grade, str) else grade.get(key)
        pairs: list[Tuple[float, bool]] = []
        graded_runs = runs_with_grade = 0
        for run in by_task[key]:
            state = RunState(run.state)
            if state is RunState.UNGRADED:
                ungraded[run.cause or "unknown"] += 1
                continue
            if state is RunState.NOT_RUN:
                not_run[run.cause or "unknown"] += 1
                continue
            if state is RunState.PENDING:
                tally["pending"] += 1
                continue
            graded_runs += 1
            value = run.grades.get(name) if name is not None else None
            if value is None:
                tally["no_such_grade"] += 1
                continue
            runs_with_grade += 1
            reason, pair = _read(value, run.agent_failed, min_passing_score)
            if reason:
                tally[reason] += 1
            if pair:
                pairs.append(pair)
        if pairs:
            pairs.sort()
            tasks.append(TaskValues(key[0], key[1], tuple(s for s, _ in pairs), tuple(p for _, p in pairs)))
        elif graded_runs and not runs_with_grade:
            no_such_grade.append(key)
        else:
            no_graded_run.append(key)
    counts = Counts(
        graded=sum(task.n for task in tasks),
        zero_after_agent_failure=tally["zero_after_agent_failure"],
        ungraded=dict(sorted(ungraded.items())),
        not_run=dict(sorted(not_run.items())),
        pending=tally["pending"],
        skipped=tally["skipped"],
        failed=tally["failed"],
        invalid_score=tally["invalid_score"],
        no_such_grade=tally["no_such_grade"],
    )
    return PerTaskValues(tuple(tasks), counts, tuple(no_graded_run), tuple(no_such_grade))


def _read(
    value: GradeValue, agent_failed: bool, min_passing_score: Optional[float]
) -> Tuple[Optional[str], Optional[Tuple[float, bool]]]:
    """The grade's (score, passed), or None, with the count it adds to."""
    status = GradeStatus(value.status)
    if status is GradeStatus.SKIPPED:
        return ("zero_after_agent_failure", (0.0, False)) if agent_failed else ("skipped", None)
    if status is GradeStatus.FAILED:
        return "failed", None
    score = value.score
    if score is None or not isfinite(score) or not 0.0 <= score <= 1.0:
        return "invalid_score", None
    passed = score >= min_passing_score if min_passing_score is not None else value.passed
    return None, (float(score), bool(passed))


def mean_over_tasks(values: PerTaskValues) -> Optional[Estimate]:
    """The grade's mean score with the task as the unit."""
    return estimate([task.scores for task in values.tasks])


def pass_rate(values: PerTaskValues) -> Optional[Estimate]:
    """The grade's pass rate, each task's own rate first, then the same estimate as the mean."""
    return estimate([[1.0 if passed else 0.0 for passed in task.passes] for task in values.tasks])


def estimate(samples: Sequence[Sequence[float]]) -> Optional[Estimate]:
    """The mean of per-task means with its interval, from each task's run values in [0, 1]. No values gives None;
    one task gives the interval over its runs."""
    tasks = [list(sample) for sample in samples if len(sample)]
    for sample in tasks:
        for x in sample:
            if not isfinite(x) or not 0.0 <= x <= 1.0:
                raise ValueError(f"values must be finite and in [0, 1], got {x!r}")
    if not tasks:
        return None
    if len(tasks) == 1:
        return _interval(tasks[0], Basis.RUNS)
    return _interval([fsum(sample) / len(sample) for sample in tasks], Basis.TASKS)


def _interval(values: list[float], basis: Basis) -> Estimate:
    n = len(values)
    mean = fsum(values) / n
    if n == 1:
        return Estimate(mean, None, None, Method.NONE, basis, 1)
    if all(x == 0.0 or x == 1.0 for x in values):
        lo, hi = wilson(fsum(values), n)
        return Estimate(mean, lo, hi, Method.WILSON, basis, n)
    if max(values) - min(values) <= _NO_SPREAD:
        lo, hi = wilson(mean * n, n)
        return Estimate(mean, lo, hi, Method.WILSON, basis, n)
    sd = sqrt(fsum((x - mean) ** 2 for x in values) / (n - 1))
    half = t_quantile(0.975, n - 1) * sd / sqrt(n)
    return Estimate(mean, max(mean - half, 0.0), min(mean + half, 1.0), Method.T, basis, n)


def wilson(successes: float, n: int) -> Tuple[float, float]:
    """The 95% Wilson score interval for `successes` of `n`, without continuity correction, as scipy's
    binomtest().proportion_ci(method='wilson'); `successes` may be fractional (a mean times its count)."""
    if n < 1 or not 0.0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n >= 1, got {successes!r} of {n!r}")
    z2 = _Z975 * _Z975
    center = (successes + z2 / 2.0) / (n + z2)
    half = _Z975 * sqrt(successes * (n - successes) / n + z2 / 4.0) / (n + z2)
    lo = 0.0 if successes == 0 else max(center - half, 0.0)
    hi = 1.0 if successes == n else min(center + half, 1.0)
    return lo, hi


def t_quantile(p: float, df: int) -> float:
    """Student's t quantile: exact at 1 and 2 degrees of freedom, the Cornish-Fisher expansion (Abramowitz and Stegun
    26.7.5) from 10,000 up, and Newton's method on the t tail otherwise, each within about 1e-13 of scipy."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df!r}")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    if p == 0.5:
        return 0.0
    tail = 1.0 - p
    if df == 1:
        return 1.0 / math.tan(math.pi * tail)
    if df == 2:
        return (2.0 * p - 1.0) / sqrt(2.0 * p * tail)
    start = _cornish_fisher(p, df)
    if df >= _CORNISH_FISHER_DF:
        return start
    return _t_solve(start, tail, df)


def _t_solve(start: float, tail: float, df: int) -> float:
    """The t whose upper tail is `tail`: Newton's method kept inside a bracket of the root, halving the bracket when a
    step would leave it, so a poor start or an extreme tail still converges."""
    lo, hi = 0.0, max(start, 1.0)
    while _t_tail(hi, df) > tail:
        lo, hi = hi, 2.0 * hi
    t = start if lo < start < hi else (lo + hi) / 2.0
    for _ in range(_NEWTON_STEPS):
        excess = _t_tail(t, df) - tail
        if excess == 0.0:
            return t
        if excess > 0.0:
            lo = t
        else:
            hi = t
        nxt = t + excess / _t_density(t, df)
        if not lo < nxt < hi:
            nxt = (lo + hi) / 2.0
        if abs(nxt - t) <= _NEWTON_TOLERANCE * nxt:
            return nxt
        t = nxt
    raise ArithmeticError(f"t quantile did not converge at tail={tail!r}, df={df!r}")


def _cornish_fisher(p: float, df: int) -> float:
    z = NormalDist().inv_cdf(p)
    z2 = z * z
    g1 = (z2 + 1.0) * z / 4.0
    g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0
    g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0
    g4 = ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) * z / 92160.0
    v = float(df)
    return z + g1 / v + g2 / v**2 + g3 / v**3 + g4 / v**4


def _t_tail(t: float, df: int) -> float:
    """P(T > t) for t > 0, as half the regularized incomplete beta I_x(df/2, 1/2) at x = df/(df + t^2). x and 1 - x
    are each formed directly, and ln x is taken from whichever keeps its precision."""
    a = df / 2.0
    x = df / (df + t * t)
    y = t * t / (df + t * t)
    ln_x = log1p(-y) if y < 0.5 else log(x)
    front = math.exp(_ln_gamma_ratio(a) - _HALF_LN_PI + a * ln_x + 0.5 * log(y))
    if x < (a + 1.0) / (a + 2.5):
        return 0.5 * front * _beta_fraction(a, 0.5, x) / a
    return 0.5 * (1.0 - front * _beta_fraction(0.5, a, y) / 0.5)


def _t_density(t: float, df: int) -> float:
    a = df / 2.0
    return math.exp(_ln_gamma_ratio(a) - 0.5 * log(df * math.pi) - (a + 0.5) * log1p(t * t / df))


def _ln_gamma_ratio(a: float) -> float:
    """ln Gamma(a + 1/2) - ln Gamma(a), by Stirling's series above 30, where two large lgammas would cancel."""
    if a < 30.0:
        return lgamma(a + 0.5) - lgamma(a)

    def correction(z: float) -> float:
        z2 = z * z
        return (1 / 12 - (1 / 360 - (1 / 1260 - (1 / 1680 - 1 / (1188 * z2)) / z2) / z2) / z2) / z

    return 0.5 * log(a) + (a * log1p(0.5 / a) - 0.5) + (correction(a + 0.5) - correction(a))


def _beta_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction of the regularized incomplete beta, by the modified Lentz method."""

    def clamp(v: float) -> float:
        return v if abs(v) > _TINY else _TINY

    c = 1.0
    d = 1.0 / clamp(1.0 - (a + b) * x / (a + 1.0))
    h = d
    for m in range(1, _FRACTION_TERMS):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0)),
        ):
            d = 1.0 / clamp(1.0 + numerator * d)
            c = clamp(1.0 + numerator / c)
            h *= d * c
        if abs(d * c - 1.0) < _FRACTION_TOLERANCE:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")
