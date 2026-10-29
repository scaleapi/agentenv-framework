"""Split the selected tests across CI runners: ``AGENTENV_TEST_SHARD=i/N`` keeps the i-th of N groups.

A file's tests stay together, so its module fixtures (image builds, deployed envs) run once per runner;
a file the durations table names in ``split_by_test`` is split by test instead, except that the tests
``pytest.mark.dependency`` links stay on one runner. The groups are balanced by the seconds the table
records (``tst/integration/slow_durations.json``), a unit it doesn't know weighs the median of the ones it
does, and the same table and selection always give the same groups.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
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


def select(nodeids: list[str], index: int, count: int, table: dict, together: dict[str, str] | None = None) -> list[str]:
    """The nodeids that shard ``index`` of ``count`` runs, in their collected order. ``together`` maps a nodeid
    to a unit it shares with others (``dependency_groups``); that unit weighs the sum of its tests."""
    split = set(table.get("split_by_test", []))
    seconds = table.get("seconds", {})
    base = {nodeid: _unit(nodeid, split) for nodeid in nodeids}
    unit = {nodeid: (together or {}).get(nodeid, base[nodeid]) for nodeid in nodeids}
    known = [seconds[name] for name in set(base.values()) if seconds.get(name, 0) > 0]
    default = statistics.median(known) if known else 1.0
    weights: dict[str, float] = defaultdict(float)
    for base_unit, shared in {base[nodeid]: unit[nodeid] for nodeid in nodeids}.items():
        weights[shared] += seconds.get(base_unit, default)
    shard = _assign(weights, count)
    return [nodeid for nodeid in nodeids if shard[unit[nodeid]] == index]


def dependency_groups(tests: list[tuple[str, str, list[str]]], table: dict) -> dict[str, str]:
    """The unit each test of a ``split_by_test`` file shares with the tests ``pytest.mark.dependency`` links
    it to, as ``select``'s ``together``. ``tests`` are (nodeid, the name its marker gives it, the names it
    depends on), for the tests with the marker; a dependency is looked up in the test's own file."""
    split = set(table.get("split_by_test", []))
    parent: dict[str, str] = {}

    def root(nodeid: str) -> str:
        while parent.get(nodeid, nodeid) != nodeid:
            nodeid = parent[nodeid]
        return nodeid

    named = {(nodeid.split("::", 1)[0], name): nodeid for nodeid, name, _ in tests}
    for nodeid, _, depends in tests:
        path = nodeid.split("::", 1)[0]
        if path not in split:
            continue
        for name in depends:
            other = named.get((path, name))
            if other is not None:
                parent[root(nodeid)] = root(other)
    linked = {nodeid for nodeid, _, _ in tests if nodeid.split("::", 1)[0] in split}
    groups = defaultdict(list)
    for nodeid in linked:
        groups[root(nodeid)].append(nodeid)
    return {nodeid: min(members) for members in groups.values() if len(members) > 1 for nodeid in members}


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
