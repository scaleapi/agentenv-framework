"""Compare full VersionedEntityStore operations before and after indexed latest reads.

Run from the repository root with the shared environment:
``PYTHONPATH=src:packages/agentenv-protocol/src .venv/bin/python tst/benchmarks/version_lookup.py``
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, TypeVar

from agent_env.store.document_store.document_store import DuplicateKeyError, Filter, Sort, VersionedEntityStore
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore

HISTORY_SIZES = (1, 100, 1000, 5000)
BASELINE_REVISION = "05b3310d5fd2c0f210b22750fd46f8c586ab6732"
UNRELATED_IDS = 1000
PAYLOAD_BYTES = 128
GET_ROUNDS = 9
PUT_ROUNDS = 7
GETS_PER_SAMPLE = 30
PUTS_PER_ROUND = 12


@dataclass(eq=True)
class Entity:
    id: str
    version: int
    payload: str


def _serialize(entity: Entity) -> dict:
    return asdict(entity)


def _deserialize(doc: dict) -> Entity:
    return Entity(**doc)


class CountingDocumentStore(LocalSqliteDocumentStore):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.decoded_rows = 0

    def _load(self, collection, filter):
        rows = super()._load(collection, filter)
        self.decoded_rows += len(rows)
        return rows

    def latest_version(self, collection: str, entity_id: str) -> dict | None:
        before = self.decoded_rows
        result = super().latest_version(collection, entity_id)
        if self.decoded_rows == before and result is not None:
            self.decoded_rows += 1
        return result


def _baseline_versioned_store() -> tuple[type[VersionedEntityStore], str]:
    """Load the version-store methods directly from the pinned pre-change source."""
    command = ["git", "show", f"{BASELINE_REVISION}:src/agent_env/store/document_store/document_store.py"]
    try:
        source = subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"Benchmark baseline {BASELINE_REVISION} is unavailable in this checkout. "
            f"Fetch it with `git fetch origin {BASELINE_REVISION}` and rerun the benchmark."
        ) from None
    module = ast.parse(source)
    original = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "VersionedEntityStore"
    )
    methods = [
        node for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"get", "next_version", "put"}
    ]
    if {node.name for node in methods} != {"get", "next_version", "put"}:
        raise RuntimeError(f"{BASELINE_REVISION} does not contain the expected version-store methods")
    legacy = ast.ClassDef(
        name="BaselineVersionedEntityStore",
        bases=[ast.Name(id="VersionedEntityStore", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace = {
        "VersionedEntityStore": VersionedEntityStore,
        "Filter": Filter,
        "Sort": Sort,
        "Optional": Optional,
        "T": TypeVar("T"),
        "DuplicateKeyError": DuplicateKeyError,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[legacy], type_ignores=[])), "baseline_versioned_store", "exec"), namespace)
    return namespace["BaselineVersionedEntityStore"], hashlib.sha256(source.encode()).hexdigest()


def _seed(
    path: Path,
    backend: type[LocalSqliteDocumentStore],
    history: int,
    versioned_store: type[VersionedEntityStore],
):
    docs = backend(str(path))
    view = versioned_store(docs, "entities", _serialize, _deserialize)
    for version in range(1, history + 1):
        docs.insert("entities", {"id": "target", "version": version, "payload": "x" * PAYLOAD_BYTES})
    for index in range(UNRELATED_IDS):
        docs.insert("entities", {"id": f"other-{index}", "version": index + 1, "payload": "x" * PAYLOAD_BYTES})
    return docs, view


def _timed(call, repetitions: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(repetitions):
        call()
    return (time.perf_counter_ns() - started) / repetitions / 1000


def _plan_and_rows(store: LocalSqliteDocumentStore) -> tuple[str, int]:
    tbl = store._table("entities")
    plan = store._conn.execute(
        f"EXPLAIN QUERY PLAN SELECT doc FROM \"{tbl}\" "
        "WHERE json_extract(doc, '$.id') = json_extract(?, '$') "
        "ORDER BY json_extract(doc, '$.version') DESC LIMIT 1",
        (json.dumps("target"),),
    ).fetchall()
    candidates = store._conn.execute(
        f"SELECT count(*) FROM \"{tbl}\" WHERE json_extract(doc, '$.id') = json_extract(?, '$')",
        (json.dumps("target"),),
    ).fetchone()[0]
    return "; ".join(row[3] for row in plan), candidates


def main() -> None:
    baseline_store, baseline_source_hash = _baseline_versioned_store()
    result = {
        "baseline_revision": BASELINE_REVISION,
        "baseline_methods": ["get", "next_version", "put"],
        "baseline_source_sha256": baseline_source_hash,
        "final_worktree_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "final_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": sys.version,
        "platform": platform.platform(),
        "rows": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for history in HISTORY_SIZES:
            base_docs, base = _seed(root / f"base-{history}.db", CountingDocumentStore, history, baseline_store)
            fast_docs, fast = _seed(root / f"fast-{history}.db", CountingDocumentStore, history, VersionedEntityStore)
            get_base, get_fast = [], []
            for round_number in range(GET_ROUNDS):
                order = [("base", base), ("fast", fast)]
                if round_number % 2:
                    order.reverse()
                for name, view in order:
                    duration = _timed(lambda: view.get("target"), GETS_PER_SAMPLE)
                    (get_base if name == "base" else get_fast).append(duration)
            put_base, put_fast, versions_equal = [], [], True
            for round_number in range(PUT_ROUNDS):
                for offset in range(PUTS_PER_ROUND):
                    first, second = (base, fast) if (round_number + offset) % 2 == 0 else (fast, base)
                    start = time.perf_counter_ns()
                    first_version = first.put(Entity("target", 0, "p" * PAYLOAD_BYTES))
                    first_us = (time.perf_counter_ns() - start) / 1000
                    start = time.perf_counter_ns()
                    second_version = second.put(Entity("target", 0, "p" * PAYLOAD_BYTES))
                    second_us = (time.perf_counter_ns() - start) / 1000
                    versions_equal &= first_version == second_version
                    if first is base:
                        put_base.append(first_us)
                        put_fast.append(second_us)
                    else:
                        put_fast.append(first_us)
                        put_base.append(second_us)
            latest_equal = base.get("target") == fast.get("target")
            exact_equal = base.get("target", version=history) == fast.get("target", version=history)
            plan, candidate_rows = _plan_and_rows(fast_docs)
            base_docs.decoded_rows = fast_docs.decoded_rows = 0
            base.get("target")
            baseline_decode_count = base_docs.decoded_rows
            fast_docs.decoded_rows = 0
            fast.get("target")
            final_decode_count = fast_docs.decoded_rows
            result["rows"].append({
                "history_seed": history,
                "unrelated_ids": UNRELATED_IDS,
                "get_baseline_us_median": round(statistics.median(get_base), 2),
                "get_final_us_median": round(statistics.median(get_fast), 2),
                "get_speedup": round(statistics.median(get_base) / statistics.median(get_fast), 1),
                "put_baseline_us_median": round(statistics.median(put_base), 2),
                "put_final_us_median": round(statistics.median(put_fast), 2),
                "put_speedup": round(statistics.median(put_base) / statistics.median(put_fast), 1),
                "put_versions_equal": versions_equal,
                "latest_equal": latest_equal,
                "exact_equal": exact_equal,
                "indexed_candidate_rows": candidate_rows,
                "latest_get_decoded_rows_baseline": baseline_decode_count,
                "latest_get_decoded_rows_final": final_decode_count,
                "top_one_query_plan": plan,
            })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
