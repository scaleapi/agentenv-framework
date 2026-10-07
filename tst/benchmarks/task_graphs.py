"""Compare task validation and scheduler graph construction with a Git ref.

From the repository root, run:

    PYTHONPATH=src:packages/agentenv-protocol/src .venv/bin/python tst/benchmarks/task_graphs.py --base origin/main

Baseline methods are extracted from ``git show <base>:src/agent_env/task/task.py``;
the current methods are imported from the working tree. Timing and allocation
measurements run separately. Synthetic steps keep the workload focused on graph work.
"""

from __future__ import annotations

import argparse
import ast
import statistics
import subprocess
import time
import tracemalloc
from dataclasses import dataclass
from types import SimpleNamespace

from agent_env.task.task import Task


def _baseline_methods(base_ref: str):
    source = subprocess.check_output(
        ["git", "show", f"{base_ref}:src/agent_env/task/task.py"], text=True,
    )
    module = ast.parse(source)
    dependencies = {"_SchedulerState", "_ancestor_ids", "_dependency_ids"}
    baseline_dependencies = [
        node for node in module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in dependencies
    ]
    if {node.name for node in baseline_dependencies} != dependencies:
        raise RuntimeError(f"{base_ref} does not contain the expected scheduler dependencies")
    task_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Task"
    )
    methods = {
        node.name: node for node in task_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_validate_dag", "_build_scheduler_state"}
    }
    if set(methods) != {"_validate_dag", "_build_scheduler_state"}:
        raise RuntimeError(f"{base_ref} does not contain the expected Task methods")
    baseline_class = ast.ClassDef(
        name="BaselineTask", bases=[], keywords=[],
        body=[methods["_validate_dag"], methods["_build_scheduler_state"]],
        decorator_list=[],
    )
    future_annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )
    extracted = ast.fix_missing_locations(
        ast.Module(body=[future_annotations, *baseline_dependencies, baseline_class], type_ignores=[])
    )
    namespace = {"dataclass": dataclass, "__name__": __name__}
    exec(compile(extracted, f"{base_ref}:src/agent_env/task/task.py", "exec"), namespace)
    return namespace["BaselineTask"]


def _steps(count: int, self_retry: bool = False):
    return [
        SimpleNamespace(
            id=f"s{i}",
            depends_on=None,
            retry_config=(SimpleNamespace(retry_from_step_id=f"s{i}") if self_retry else None),
        )
        for i in range(count)
    ]


def _measure(fn, repetitions: int) -> tuple[float, int]:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)

    tracemalloc.start()
    fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(samples), peak_bytes


def _row(name: str, base, count: int, repetitions: int) -> str:
    steps = _steps(count, self_retry=(name == "self-retry validation"))
    baseline = getattr(base, "_validate_dag") if name.endswith("validation") else None
    if name == "scheduler graph":
        baseline_fn = lambda: base._build_scheduler_state(object.__new__(base), steps, 0)
        current_fn = lambda: Task._build_scheduler_state(object.__new__(Task), steps, 0)
    else:
        baseline_fn = lambda: baseline(steps)
        current_fn = lambda: Task._validate_dag(steps)
    base_s, base_peak = _measure(baseline_fn, repetitions)
    current_s, current_peak = _measure(current_fn, repetitions)
    return (
        f"{name:<23} {count:>6}  {base_s * 1000:>10.3f}  {current_s * 1000:>10.3f}  "
        f"{base_peak:>12,}  {current_peak:>12,}  {base_s / current_s:>7.1f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git ref to compare against")
    parser.add_argument("--steps", nargs="+", type=int, default=[300, 1000])
    parser.add_argument("--retry-steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    base = _baseline_methods(args.base)

    print(f"Base ref: {args.base}; timing: median of {args.repeats}; allocation: one separate traced run")
    print(f"{'workload':<23} {'steps':>6}  {'base ms':>10}  {'current ms':>10}  {'base peak B':>12}  {'current B':>12}  {'speedup':>8}")
    for count in args.steps:
        print(_row("implicit validation", base, count, args.repeats))
        print(_row("scheduler graph", base, count, args.repeats))
    print(_row("self-retry validation", base, args.retry_steps, args.repeats))
    print("Baseline methods, dependency helpers and state class: extracted together from the selected Git ref.")


if __name__ == "__main__":
    main()
