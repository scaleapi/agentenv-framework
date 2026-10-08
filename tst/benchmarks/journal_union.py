"""Compare journal list union with a git revision using the production function bodies.

Run from the repository root, for example:
    python tst/benchmarks/journal_union.py --base origin/main --repeats 5
"""

from __future__ import annotations

import argparse
import ast
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "src/agent_env/task/step_journal.py"


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def load_union(source: str):
    tree = ast.parse(source)
    wanted = {"_union", "_item_key"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("source does not define both _union and _item_key")
    namespace = {}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<step_journal>", "exec"), namespace)
    return namespace["_union"]


def median_ms(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="git revision to compare against (default: origin/main)")
    parser.add_argument("--repeats", type=positive_int, default=5, help="timing samples per implementation and size (default: 5)")
    args = parser.parse_args()

    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.base}:{TARGET}"], cwd=ROOT, text=True
    )
    candidate_source = (ROOT / TARGET).read_text()
    baseline, candidate = load_union(baseline_source), load_union(candidate_source)

    print(f"Median runtime; production _union/_item_key function bodies extracted from {args.base} and the worktree")
    print("items  baseline_ms  worktree_ms  speedup")
    for size in (100, 1000, 3000):
        items = [
            {"instance_id": str(i), "sandbox_id": str(i), "card": {"tools": ["a", "b"]}}
            for i in range(size)
        ]
        expected = baseline([], items, path="context.deployed_envs")
        actual = candidate([], items, path="context.deployed_envs")
        if actual != expected:
            raise AssertionError(f"union output differs at {size} items")
        before = median_ms(lambda: baseline([], items, path="context.deployed_envs"), args.repeats)
        after = median_ms(lambda: candidate([], items, path="context.deployed_envs"), args.repeats)
        print(f"{size:>5}  {before:>11.3f}  {after:>11.3f}  {before / after:>6.2f}x")


if __name__ == "__main__":
    main()
