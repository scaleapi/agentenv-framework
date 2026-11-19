"""Writes stats_v1.json, the reference values agent_env.eval.stats is checked against, from scipy and Inspect. CI runs
the checks without either; run this only to change the vectors, from the repository root:

    uv run --no-project --python 3.12 --with scipy==1.18.1 --with inspect-ai==0.3.288 \\
        python tst/unit/eval/vectors/generate.py
"""

import json
import random
from importlib.metadata import version
from pathlib import Path

from inspect_ai.scorer import SampleScore, Score, ci, ci_wilson, mean_score, stderr
from scipy import stats

OUT = Path(__file__).with_name("stats_v1.json")

T_PS = (0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999, 0.999999, 0.9999999)
T_DFS = [*range(1, 301), 400, 499, 500, 501, 750, 1000, 2000, 5000, 9999, 10_000, 100_000, 1_000_000, 100_000_000]
T_DFS_SPARSE = [*range(1, 31), 50, 100, 200, 499, 500, 1000, 5000, 9999, 10_000, 100_000, 100_000_000]


def t_quantiles():
    rows = []
    for p in T_PS:
        dfs = T_DFS if p in (0.8, 0.975) else T_DFS_SPARSE
        rows += [[p, df, float(stats.t.ppf(p, df))] for df in dfs]
    return rows


def wilson_intervals():
    cases = [(k, n) for n in range(1, 41) for k in range(n + 1)] + [(k, n) for n in (100, 200, 500) for k in (0, 1, n // 2, n - 1, n)]
    rows = []
    for k, n in cases:
        interval = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
        inspect = ci_wilson()([_sample(1.0, i) for i in range(k)] + [_sample(0.0, k + i) for i in range(n - k)])
        assert abs(inspect["lower"] - interval.low) < 1e-12 and abs(inspect["upper"] - interval.high) < 1e-12, (k, n)
        rows.append([k, n, float(interval.low), float(interval.high)])
    return rows


def task_intervals():
    """Per-task means that are fractional and spread out, so the t interval applies unclipped, against scipy's and
    Inspect's t interval over one value per task."""
    rng = random.Random(5998)
    rows = []
    for tasks in (2, 3, 5, 10, 20, 50, 200):
        for _ in range(4):
            values, interval = _unclipped(lambda: [round(rng.uniform(0.3, 0.7), 6) for _ in range(tasks)])
            samples = [_sample(v, i) for i, v in enumerate(values)]
            inspect, se = ci()(samples), stderr()(samples)
            assert abs(inspect["lower"] - interval.low) < 1e-12 and abs(inspect["upper"] - interval.high) < 1e-12
            rows.append({"values": values, "lo": float(interval.low), "hi": float(interval.high), "se": float(se)})
    return rows


def run_level_intervals():
    """Runs per task from 1 to 10, reduced per task by Inspect's epoch mean before its t interval, so a task's run
    count never weighs in the mean."""
    rng = random.Random(15998)
    rows = []
    for tasks in (3, 8, 30):
        for _ in range(3):
            runs, interval = _unclipped(
                lambda: [[round(rng.uniform(0.1, 0.9), 6) for _ in range(rng.choice((1, 2, 3, 10)))] for _ in range(tasks)],
                lambda runs: [sum(task) / len(task) for task in runs],
            )
            inspect = ci()([_reduced(task, i) for i, task in enumerate(runs)])
            assert abs(inspect["lower"] - interval.low) < 1e-12 and abs(inspect["upper"] - interval.high) < 1e-12
            rows.append({"runs": runs, "lo": float(inspect["lower"]), "hi": float(inspect["upper"])})
    return rows


def _unclipped(draw, means=lambda values: values):
    """Draws until scipy's t interval over the means lies inside (0, 1), where agentenv clips nothing."""
    while True:
        drawn = draw()
        interval = stats.ttest_1samp(means(drawn), 0.0).confidence_interval(0.95)
        if 0.0 < interval.low and interval.high < 1.0:
            return drawn, interval


def _sample(value, sample_id):
    return SampleScore(score=Score(value=value), sample_id=sample_id)


def _reduced(values, sample_id):
    return SampleScore(score=mean_score()([Score(value=v) for v in values]), sample_id=sample_id)


def main():
    vectors = {
        "generated_with": {"scipy": version("scipy"), "inspect-ai": version("inspect-ai")},
        "t_quantile": t_quantiles(),
        "wilson": wilson_intervals(),
        "task_interval": task_intervals(),
        "run_level_interval": run_level_intervals(),
    }
    _write(vectors)


def _write(vectors):
    """One row per line, so a regeneration diffs row by row."""
    parts = []
    for key, value in vectors.items():
        if isinstance(value, list):
            body = "[\n" + ",\n".join(f"  {json.dumps(row)}" for row in value) + "\n ]"
        else:
            body = json.dumps(value)
        parts.append(f" {json.dumps(key)}: {body}")
    OUT.write_text("{\n" + ",\n".join(parts) + "\n}\n")


if __name__ == "__main__":
    main()
