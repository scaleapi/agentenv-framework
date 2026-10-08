"""Compare Explorer run-group and entity-page reads with a Git baseline.

Run from a checkout with the development dependencies installed:

    PYTHONPATH=src:packages/agentenv-protocol/src python tst/benchmarks/explorer_reads.py --base origin/main

The script makes a temporary detached worktree for ``--base``, seeds both versions
with the same 1,000 runs in 100 groups and 2,000 entities with five versions each
in temporary SQLite stores. It reports the median of three endpoint calls plus
document-store query counts. It measures Python
and SQLite endpoint work only; it makes no HTTP or network latency claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_env.config import configure
from agent_env.explorer.routers.common import versioned_router
from agent_env.explorer.routers.runs import list_run_groups
from agent_env.runner import store as run_store
from agent_env.runner.runner import RunRecord, RunStatus
from agent_env.store import LocalSqliteDocumentStore


def _worker() -> None:
    with tempfile.TemporaryDirectory(prefix="agentenv-explorer-bench-") as temp:
        root = Path(temp)
        config_file = root / "config.toml"
        config_file.write_text("")
        os.environ["AGENT_ENV_CONFIG"] = str(config_file)
        store = LocalSqliteDocumentStore(str(root / "documents.db"))
        configure(document_store=store)
        run_store.ensure_indexes()
        store.ensure_index("task_instances", ["instance_id"], unique=True)
        for i in range(1000):
            group = i // 10
            run_store.insert_run(RunRecord(
                run_id=f"bench-run-{i}", runner="local", task_id="bench-task", task_version=1,
                instance_id=f"bench-instance-{i}", status=RunStatus.COMPLETED,
                created_at_utc=f"2026-01-{group // 30 + 1:02d}T{group % 24:02d}:{i % 60:02d}:00Z",
                overrides={"metadata": {"run_group_id": f"bench-group-{group}"}},
            ))
            store.insert("task_instances", {
                "instance_id": f"bench-instance-{i}", "completed_steps": [],
            })
        for entity in range(2000):
            for version in range(1, 6):
                store.insert("bench_entities", {
                    "id": f"entity-{entity:04d}", "version": version,
                    "type": "benchmark", "created_at_utc": f"2026-01-{version:02d}T00:00:00Z",
                })

        entity_list = versioned_router(
            prefix="/bench", tag="bench", collection="bench_entities", noun="entity",
        ).routes[0].endpoint

        counts = {"query": 0, "find_one": 0, "id_lookup": 0}
        original_query, original_find_one = store.query, store.find_one

        def counted_query(*args, **kwargs):
            counts["query"] += 1
            return original_query(*args, **kwargs)

        def counted_find_one(*args, **kwargs):
            counts["find_one"] += 1
            return original_find_one(*args, **kwargs)

        store.query, store.find_one = counted_query, counted_find_one
        if hasattr(store, "find_many_by_id"):
            original_lookup = store.find_many_by_id

            def counted_lookup(*args, **kwargs):
                counts["id_lookup"] += 1
                return original_lookup(*args, **kwargs)

            store.find_many_by_id = counted_lookup

        def measure(endpoint, *args):
            elapsed, query_counts, find_one_counts, lookup_counts, digests = [], [], [], [], []
            for _ in range(3):
                counts.update(query=0, find_one=0, id_lookup=0)
                started = time.perf_counter()
                result = endpoint(*args)
                elapsed.append(time.perf_counter() - started)
                payload = result.model_dump(mode="json")
                digests.append(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
                query_counts.append(counts["query"])
                find_one_counts.append(counts["find_one"])
                lookup_counts.append(counts["id_lookup"])
            assert len(set(digests)) == 1
            return {
                "median_seconds": statistics.median(elapsed),
                "query_calls_per_call": statistics.median(query_counts),
                "find_one_calls_per_call": statistics.median(find_one_counts),
                "id_lookup_calls_per_call": statistics.median(lookup_counts),
                "response_sha256": digests[0],
                "total": result.total,
                "page_items": len(result.items),
            }

        run_metrics = measure(list_run_groups, "bench-task", None, 20, 0)
        entity_metrics = measure(entity_list, 50, 0, None, None, "created_at_utc", True)
        assert (run_metrics["total"], run_metrics["page_items"]) == (100, 20)
        assert (entity_metrics["total"], entity_metrics["page_items"]) == (2000, 50)
        print(json.dumps({
            "run_groups": {**run_metrics, "runs": 1000, "groups": 100, "page_groups": 20},
            "entity_list": {**entity_metrics, "entities": 2000, "versions_each": 5, "page_size": 50},
        }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git ref to benchmark as the baseline")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker()
        return

    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    script = Path(__file__).resolve()
    base_sha = subprocess.check_output(["git", "rev-parse", args.base], cwd=repo, text=True).strip()

    def run(checkout: Path) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(checkout / "src"), str(checkout / "packages/agentenv-protocol/src"),
        ])
        output = subprocess.check_output(
            [sys.executable, str(script), "--worker"], cwd=checkout, env=env, text=True,
        )
        return json.loads(output)

    current = run(repo)
    with tempfile.TemporaryDirectory(prefix="agentenv-explorer-base-") as temp:
        baseline = Path(temp) / "checkout"
        subprocess.run(["git", "worktree", "add", "--detach", str(baseline), base_sha],
                       cwd=repo, check=True, stdout=subprocess.DEVNULL)
        try:
            before = run(baseline)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)],
                           cwd=repo, check=True, stdout=subprocess.DEVNULL)
    for workload in ("run_groups", "entity_list"):
        if before[workload]["response_sha256"] != current[workload]["response_sha256"]:
            raise SystemExit(f"{workload} response differs between {args.base} and current")
    print(json.dumps({"base": args.base, "base_sha": base_sha, "baseline": before, "current": current}, indent=2))


if __name__ == "__main__":
    main()
