"""agent_env.eval.stats against reference values scipy and Inspect computed (vectors/stats_v1.json, written by
vectors/generate.py), so CI checks them with neither installed."""
import json
import math
from pathlib import Path

from agent_env.eval import stats

VECTORS = json.loads((Path(__file__).parent / "vectors" / "stats_v1.json").read_text())


def close(a, b, *, rel=0.0, abs_=0.0):
    return a is not None and math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def test_vectors_record_the_versions_they_came_from():
    assert VECTORS["generated_with"] == {"scipy": "1.18.1", "inspect-ai": "0.3.288"}


def test_t_quantile_matches_scipy():
    misses = [(p, df, t, stats.t_quantile(p, df)) for p, df, t in VECTORS["t_quantile"]
              if not close(stats.t_quantile(p, df), t, rel=1e-12)]
    assert not misses, misses[:5]
    assert len(VECTORS["t_quantile"]) > 700


def test_wilson_matches_scipy_and_inspect():
    misses = [(k, n, lo, hi) for k, n, lo, hi in VECTORS["wilson"]
              if not all(close(a, b, abs_=1e-12) for a, b in zip(stats.wilson(k, n), (lo, hi)))]
    assert not misses, misses[:5]


def test_task_interval_matches_scipy_and_inspect():
    for case in VECTORS["task_interval"]:
        estimate = stats.estimate([[v] for v in case["values"]])
        assert estimate.method is stats.Method.T and estimate.basis is stats.Basis.TASKS
        assert close(estimate.lo, case["lo"], abs_=1e-12) and close(estimate.hi, case["hi"], abs_=1e-12), case


def test_runs_reduce_per_task_first_as_inspects_epochs_do():
    for case in VECTORS["run_level_interval"]:
        estimate = stats.estimate(case["runs"])
        assert estimate.n == len(case["runs"])
        assert close(estimate.lo, case["lo"], abs_=1e-12) and close(estimate.hi, case["hi"], abs_=1e-12), case
