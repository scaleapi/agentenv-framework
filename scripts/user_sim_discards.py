"""How often the user sim judge discards runs, per assistant model: read-only, over run records.

    python scripts/user_sim_discards.py RECORD_OR_FOLDER [...] [--json]

A record is a task instance as a run stores it (``{"context": {...}}``), its context alone, or a JSON list of them; a
folder is searched for ``*.json`` records. Each user_sim_judge entry (``metadata["verifications"]``, format
``user_sim_checks``) that judged its run counts once, under the model that played the assistant in the run it judged
(``metadata["models"]["solver"][run_step_id]``, else the context's ``agent_model``, else "unknown"). For each model it
prints the runs judged, those discarded (``user_sim_error`` yes), those unknown (undecided), the discard rate, and per
check how many discards it was behind. Skipped entries (the run ended in an error, no graded turn) aren't judged runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

FORMAT = "user_sim_checks"


def contexts(value: Any) -> Iterator[dict]:
    """The run contexts in a loaded record: an instance's ``context``, a context, or a list of either."""
    if isinstance(value, list):
        for v in value:
            yield from contexts(v)
    elif isinstance(value, dict):
        if isinstance(value.get("context"), dict):
            yield value["context"]
        elif isinstance(value.get("metadata"), dict):
            yield value


def records(paths: list[str]) -> Iterator[tuple[str, dict]]:
    for p in map(Path, paths):
        for f in sorted(p.rglob("*.json")) if p.is_dir() else [p]:
            try:
                loaded = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for ctx in contexts(loaded):
                yield str(f), ctx


def judged_runs(ctx: dict) -> Iterator[tuple[str, dict]]:
    """``(assistant model, entry)`` for each user sim judge entry in a context that judged its run."""
    md = ctx.get("metadata") or {}
    solvers = (md.get("models") or {}).get("solver") or {}
    for entry in (md.get("verifications") or {}).values():
        if not isinstance(entry, dict) or entry.get("format") != FORMAT or entry.get("skipped"):
            continue
        model = solvers.get(entry.get("run_step_id")) or ctx.get("agent_model") or "unknown"
        yield str(model), entry


def tally(paths: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"judged": 0, "discarded": 0, "unknown": 0, "checks": Counter()})
    seen = set()
    for _, ctx in records(paths):
        key = ctx.get("instance_id") or (ctx.get("metadata") or {}).get("instance_id") or id(ctx)
        if key in seen:  # the same run exported twice
            continue
        seen.add(key)
        for model, entry in judged_runs(ctx):
            error = entry.get("user_sim_error") or {}
            row = out[model]
            row["judged"] += 1
            if error.get("value") == "yes":
                row["discarded"] += 1
                row["checks"].update(error.get("because") or [])
            elif error.get("value") == "unknown":
                row["unknown"] += 1
    return {m: {**r, "rate": r["discarded"] / r["judged"] if r["judged"] else None, "checks": dict(r["checks"])}
            for m, r in sorted(out.items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", action="store_true", help="print the tally as JSON")
    a = p.parse_args(argv)
    got = tally(a.paths)
    if a.json:
        print(json.dumps(got, indent=1))
        return 0
    if not got:
        print("no judged runs found", file=sys.stderr)
        return 1
    print(f"{'assistant model':32} {'judged':>6} {'discarded':>9} {'unknown':>7} {'rate':>6}  discards by check")
    for model, r in got.items():
        checks = ", ".join(f"{c} {n}" for c, n in sorted(r["checks"].items(), key=lambda x: -x[1])) or "-"
        print(f"{model:32} {r['judged']:>6} {r['discarded']:>9} {r['unknown']:>7} {r['rate']:>6.0%}  {checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
