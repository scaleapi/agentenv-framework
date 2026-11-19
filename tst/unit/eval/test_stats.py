"""agent_env.eval.stats: the task as the unit, what enters a grade's numbers, and the interval each case gets."""
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_env.eval import stats
from agent_env.eval.stats import Basis, GradeStatus, GradeValue, Method, RunState, RunValue

GRADED, SKIPPED, FAILED = GradeStatus.GRADED, GradeStatus.SKIPPED, GradeStatus.FAILED


def graded(task, score, passed=None, *, grade="check", version=1, threshold=1.0, **run):
    passed = score >= threshold if passed is None else passed
    return RunValue(task, version, RunState.GRADED, {grade: GradeValue(GRADED, score, passed, threshold)}, **run)


def with_grade(task, value, *, grade="check", **run):
    return RunValue(task, 1, RunState.GRADED, {grade: value}, **run)


def test_a_task_with_ten_runs_weighs_the_same_as_a_task_with_one():
    values = stats.per_task_values([graded("a", 1.0) for _ in range(10)] + [graded("b", 0.0)], "check")
    mean = stats.mean_over_tasks(values)
    assert mean.value == 0.5 and mean.basis is Basis.TASKS and mean.n == 2


def test_runs_that_give_no_value_are_counted_by_cause_never_averaged_as_zero():
    runs = [
        graded("a", 0.8),
        RunValue("a", 1, RunState.UNGRADED, cause="infra_environment"),
        RunValue("a", 1, RunState.UNGRADED, cause="infra_environment"),
        RunValue("a", 1, RunState.UNGRADED),
        RunValue("a", 1, RunState.NOT_RUN, cause="not_run_budget"),
        RunValue("a", 1, RunState.PENDING),
        graded("b", 0.4),
    ]
    values = stats.per_task_values(runs, "check")
    assert stats.mean_over_tasks(values).value == pytest.approx(0.6)
    assert values.counts.graded == 2
    assert values.counts.ungraded == {"infra_environment": 2, "unknown": 1}
    assert values.counts.not_run == {"not_run_budget": 1}
    assert values.counts.pending == 1


def test_a_skipped_grade_counts_as_zero_only_after_an_agent_failure():
    runs = [
        graded("a", 1.0),
        with_grade("a", GradeValue(SKIPPED), agent_failed=True, cause="agent_timeout"),
        with_grade("a", GradeValue(SKIPPED)),
    ]
    values = stats.per_task_values(runs, "check")
    (task,) = values.tasks
    assert task.scores == (0.0, 1.0) and task.passes == (False, True)
    assert values.counts.zero_after_agent_failure == 1 and values.counts.skipped == 1 and values.counts.graded == 2


@pytest.mark.parametrize("score", [None, 1.7, -0.2, math.nan, math.inf])
def test_a_score_outside_zero_to_one_is_left_out_and_counted(score):
    values = stats.per_task_values([graded("a", 0.5), with_grade("a", GradeValue(GRADED, score, True))], "check")
    assert values.tasks[0].scores == (0.5,) and values.counts.invalid_score == 1


def test_a_failed_grade_is_left_out_of_its_row_and_counted():
    values = stats.per_task_values([graded("a", 0.5), with_grade("a", GradeValue(FAILED))], "check")
    assert values.tasks[0].scores == (0.5,) and values.counts.failed == 1


def test_the_stored_pass_is_trusted_and_an_override_re_derives_it_from_the_score():
    runs = [graded("a", 0.85, threshold=0.8), graded("a", 0.85, threshold=0.9), graded("b", 0.7, threshold=0.6)]
    assert [t.passes for t in stats.per_task_values(runs, "check").tasks] == [(False, True), (True,)]
    overridden = stats.per_task_values(runs, "check", min_passing_score=0.75)
    assert [t.passes for t in overridden.tasks] == [(True, True), (False,)]


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_an_override_outside_the_pass_rule_range_is_refused(threshold):
    with pytest.raises(ValueError, match="min_passing_score"):
        stats.per_task_values([graded("a", 1.0)], "check", min_passing_score=threshold)


def test_a_partial_credit_pass_rate_is_each_tasks_rate_first():
    runs = [graded("a", s, threshold=0.8) for s in (0.9, 0.85, 0.5, 0.2)] + [graded("b", 0.95, threshold=0.8)]
    rate = stats.pass_rate(stats.per_task_values(runs, "check"))
    assert rate.value == pytest.approx((0.5 + 1.0) / 2) and rate.basis is Basis.TASKS


def test_each_task_can_name_its_own_grade():
    runs = [graded("a", 1.0, grade="verify"), graded("b", 0.0, grade="judge"), graded("c", 0.5, grade="judge")]
    values = stats.per_task_values(runs, {("a", 1): "verify", ("b", 1): "judge"})
    assert [(t.task_id, t.scores) for t in values.tasks] == [("a", (1.0,)), ("b", (0.0,))]
    assert values.no_such_grade == (("c", 1),)


def test_coverage_says_which_tasks_gave_no_value_and_which_lack_the_grade():
    runs = [
        graded("a", 1.0),
        RunValue("b", 1, RunState.UNGRADED, cause="verifier_failed"),
        graded("c", 1.0, grade="other"),
        with_grade("d", GradeValue(FAILED)),
    ]
    values = stats.per_task_values(runs, "check")
    assert [t.task_id for t in values.tasks] == ["a"]
    assert values.no_graded_run == (("b", 1), ("d", 1))
    assert values.no_such_grade == (("c", 1),)
    assert values.counts.no_such_grade == 1


def test_two_versions_of_a_task_are_two_tasks():
    values = stats.per_task_values([graded("a", 1.0, version=1), graded("a", 0.0, version=2)], "check")
    assert [(t.task_version, t.scores) for t in values.tasks] == [(1, (1.0,)), (2, (0.0,))]


def test_an_unknown_run_state_or_grade_status_is_refused():
    with pytest.raises(ValueError):
        stats.per_task_values([RunValue("a", 1, "finished")], "check")
    with pytest.raises(ValueError):
        stats.per_task_values([with_grade("a", GradeValue("passed"))], "check")


def test_no_graded_task_gives_none_not_zero():
    values = stats.per_task_values([RunValue("a", 1, RunState.UNGRADED, cause="cancelled")], "check")
    assert stats.mean_over_tasks(values) is None and stats.pass_rate(values) is None
    assert stats.estimate([]) is None and stats.estimate([[], []]) is None


@pytest.mark.parametrize("tasks,expected", [(20, (0.8389, 1.0)), (8, (0.6756, 1.0))])
def test_every_task_at_one_gets_wilson_over_the_task_count(tasks, expected):
    estimate = stats.estimate([[1.0]] * tasks)
    assert estimate.method is Method.WILSON and estimate.value == 1.0
    assert (estimate.lo, estimate.hi) == pytest.approx(expected, abs=5e-5)


def test_every_task_at_zero_gets_wilson_over_the_task_count():
    estimate = stats.estimate([[0.0]] * 20)
    assert estimate.method is Method.WILSON and (estimate.lo, estimate.hi) == pytest.approx((0.0, 0.1611), abs=5e-5)


def test_values_that_are_all_zero_or_one_get_wilson_so_failing_a_task_never_raises_the_lower_bound():
    nineteen = stats.estimate([[1.0]] * 19 + [[0.0]])
    twenty = stats.estimate([[1.0]] * 20)
    assert nineteen.method is Method.WILSON
    assert (nineteen.lo, nineteen.hi) == pytest.approx((0.7639, 0.9911), abs=5e-5)
    assert nineteen.lo < twenty.lo


def test_fractional_task_means_get_students_t():
    estimate = stats.estimate([[1.0, 0.0, 1.0]] + [[1.0]] * 9)
    assert estimate.method is Method.T and estimate.n == 10


def test_values_with_no_spread_get_wilson_at_their_mean():
    estimate = stats.estimate([[0.5]] * 20)
    assert estimate.method is Method.WILSON and estimate.value == 0.5
    assert (estimate.lo, estimate.hi) == pytest.approx(stats.wilson(10, 20), abs=1e-15)


def test_float_noise_is_no_spread():
    estimate = stats.estimate([[0.30000000000000004], [0.3], [0.29999999999999993]])
    assert estimate.method is Method.WILSON and estimate.hi - estimate.lo > 0.5


def test_one_task_gets_its_interval_over_its_runs():
    binary = stats.estimate([[1.0, 1.0, 1.0, 0.0, 0.0]])
    assert binary.basis is Basis.RUNS and binary.method is Method.WILSON and binary.n == 5
    assert (binary.lo, binary.hi) == pytest.approx((0.2307, 0.8824), abs=5e-5)
    fractional = stats.estimate([[0.5, 0.6, 0.7, 0.8, 0.9]])
    assert fractional.method is Method.T and (fractional.lo, fractional.hi) == pytest.approx((0.5037, 0.8963), abs=5e-5)


def test_one_task_with_three_of_five_runs_passing_reads_the_same_as_its_pass_rate():
    values = stats.per_task_values([graded("a", s) for s in (1.0, 1.0, 1.0, 0.0, 0.0)], "check")
    assert stats.mean_over_tasks(values) == stats.pass_rate(values)


@pytest.mark.parametrize("successes,n,expected", [(3, 8, (0.1368, 0.6943)), (8, 8, (0.6756, 1.0))])
def test_a_single_tasks_pass_rate_over_its_runs(successes, n, expected):
    runs = [graded("a", 1.0)] * successes + [graded("a", 0.0)] * (n - successes)
    rate = stats.pass_rate(stats.per_task_values(runs, "check"))
    assert rate.basis is Basis.RUNS and (rate.lo, rate.hi) == pytest.approx(expected, abs=5e-5)


def test_a_single_value_has_no_interval():
    estimate = stats.estimate([[0.7]])
    assert (estimate.value, estimate.lo, estimate.hi, estimate.method, estimate.n) == (0.7, None, None, Method.NONE, 1)


def test_a_t_interval_that_would_leave_zero_to_one_is_clipped():
    estimate = stats.estimate([[0.95], [0.99], [0.4]])
    assert estimate.method is Method.T and estimate.lo == 0.0 and estimate.hi == 1.0


@pytest.mark.parametrize("bad", [1.5, -0.1, math.nan])
def test_estimate_refuses_values_outside_zero_to_one(bad):
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        stats.estimate([[0.5], [bad]])


def test_the_order_runs_arrive_in_changes_nothing():
    rng = random.Random(7)
    runs = [graded(f"t{i % 37}", round(rng.random(), 3), replica=i) for i in range(500)]
    expected = stats.per_task_values(runs, "check")
    for _ in range(5):
        rng.shuffle(runs)
        values = stats.per_task_values(runs, "check")
        assert values == expected
        assert stats.mean_over_tasks(values) == stats.mean_over_tasks(expected)


unit = st.one_of(st.sampled_from([0.0, 1.0, 0.5]), st.floats(0.0, 1.0))


@settings(max_examples=400, deadline=None)
@given(st.lists(st.lists(unit, min_size=1, max_size=6), min_size=1, max_size=25))
def test_no_bound_ever_leaves_zero_to_one_or_misses_the_value(samples):
    estimate = stats.estimate(samples)
    assert 0.0 <= estimate.value <= 1.0
    if estimate.lo is not None:
        assert 0.0 <= estimate.lo <= estimate.value + 1e-12
        assert estimate.value - 1e-12 <= estimate.hi <= 1.0


@settings(max_examples=200, deadline=None)
@given(st.lists(st.lists(st.booleans(), min_size=1, max_size=6), min_size=1, max_size=25))
def test_pass_rates_follow_the_same_bounds(passes):
    runs = [graded(f"t{i}", 1.0 if p else 0.0) for i, task in enumerate(passes) for p in task]
    rate = stats.pass_rate(stats.per_task_values(runs, "check"))
    assert 0.0 <= rate.value <= 1.0 and (rate.lo is None or 0.0 <= rate.lo <= rate.value <= rate.hi <= 1.0)


def test_t_quantile_is_exact_at_one_and_two_degrees_of_freedom():
    assert stats.t_quantile(0.975, 1) == pytest.approx(math.tan(0.475 * math.pi), rel=1e-14)
    assert stats.t_quantile(0.975, 2) == pytest.approx(0.95 / math.sqrt(2 * 0.975 * 0.025), rel=1e-14)


def test_t_quantile_is_symmetric_and_tends_to_the_normal():
    assert stats.t_quantile(0.025, 7) == pytest.approx(-stats.t_quantile(0.975, 7), rel=1e-14)
    assert stats.t_quantile(0.5, 7) == 0.0
    assert stats.t_quantile(0.975, 10**8) == pytest.approx(1.959963984540054, rel=1e-7)


@pytest.mark.parametrize("df", [3, 10, 499, 500, 9_999, 10_000])
def test_t_quantile_converges_at_extreme_probabilities(df):
    assert stats.t_quantile(1 - 1e-12, df) > stats.t_quantile(0.9999999, df) > stats.t_quantile(0.999, df) > 0


@pytest.mark.parametrize("p,df", [(0.0, 5), (1.0, 5), (0.975, 0)])
def test_t_quantile_refuses_arguments_outside_its_domain(p, df):
    with pytest.raises(ValueError):
        stats.t_quantile(p, df)


@pytest.mark.parametrize("successes,n", [(-1, 5), (6, 5), (0, 0)])
def test_wilson_refuses_impossible_counts(successes, n):
    with pytest.raises(ValueError):
        stats.wilson(successes, n)
