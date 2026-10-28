"""Split the selected tests across CI runners: ``AGENTENV_TEST_SHARD=i/N`` keeps the i-th of N groups.

A file's tests stay together, so its module fixtures (image builds, deployed envs) run once per runner;
a file the durations table names in ``split_by_test`` is split by test instead. The groups are balanced
by the seconds the table records (``tst/integration/slow_durations.json``), a unit it doesn't know
weighs the median of the ones it does, and the same table and selection always give the same groups.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ENV_VAR = "AGENTENV_TEST_SHARD"
DURATIONS = Path(__file__).resolve().parents[1] / "integration" / "slow_durations.json"


def parse(value: str) -> tuple[int, int]:
    """``"i/N"`` as ``(i, N)``, with ``1 <= i <= N``."""
    try:
        index, count = (int(part) for part in value.split("/"))
    except ValueError:
        raise ValueError(f"{ENV_VAR} must look like 2/4, not {value!r}") from None
    if not 1 <= index <= count:
        raise ValueError(f"{ENV_VAR}={value}: the shard must be between 1 and {count}")
    return index, count


def select(nodeids: list[str], index: int, count: int, table: dict) -> list[str]:
    """The nodeids that shard ``index`` of ``count`` runs, in their collected order."""
    split = set(table.get("split_by_test", []))
    seconds = table.get("seconds", {})
    units = {_unit(nodeid, split) for nodeid in nodeids}
    known = [seconds[unit] for unit in units if seconds.get(unit, 0) > 0]
    default = statistics.median(known) if known else 1.0
    shard = _assign({unit: seconds.get(unit, default) for unit in units}, count)
    return [nodeid for nodeid in nodeids if shard[_unit(nodeid, split)] == index]


def load_table(path: Path = DURATIONS) -> dict:
    return json.loads(path.read_text())


def _unit(nodeid: str, split_by_test: set[str]) -> str:
    path = nodeid.split("::", 1)[0]
    return nodeid if path in split_by_test else path


def _assign(weights: dict[str, float], count: int) -> dict[str, int]:
    """Each unit's shard, from 1: the heaviest unit first, onto the lightest shard so far."""
    loads = [0.0] * count
    shard = {}
    for unit in sorted(weights, key=lambda name: (-weights[name], name)):
        lightest = min(range(count), key=lambda i: (loads[i], i))
        shard[unit] = lightest + 1
        loads[lightest] += weights[unit]
    return shard
