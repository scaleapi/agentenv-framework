"""AGENTENV_TEST_SHARD (``tst/util/shards.py`` and its hook in ``tst/conftest.py``): the CI slow-tier shards."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tst.util import shards

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "tst/integration/env/gateway/gateway_test.py"


def _table(seconds: dict[str, float]) -> dict:
    return {"split_by_test": [SPLIT], "seconds": seconds}


def _groups(nodeids: list[str], count: int, table: dict) -> list[list[str]]:
    return [shards.select(nodeids, index, count, table) for index in range(1, count + 1)]


@pytest.mark.parametrize("value,expected", [("1/4", (1, 4)), ("4/4", (4, 4)), ("1/1", (1, 1))])
def test_parse_reads_the_shard_and_the_count(value, expected):
    assert shards.parse(value) == expected


@pytest.mark.parametrize("value", ["", "2", "0/4", "5/4", "a/4", "1/4/2"])
def test_parse_refuses_anything_but_i_of_n(value):
    with pytest.raises(ValueError, match=shards.ENV_VAR):
        shards.parse(value)


def test_every_test_runs_in_exactly_one_shard_in_its_collected_order():
    nodeids = [f"tst/integration/f{n}_test.py::test_{t}" for n in range(7) for t in range(3)]
    groups = _groups(nodeids, 3, _table({f"tst/integration/f{n}_test.py": 10 * n for n in range(7)}))
    assert sorted(sum(groups, [])) == sorted(nodeids)
    for group in groups:
        assert group == [nodeid for nodeid in nodeids if nodeid in group]


def test_a_files_tests_stay_together_and_the_heaviest_files_go_to_different_shards():
    seconds = {"tst/integration/big_test.py": 600, "tst/integration/mid_test.py": 500, "tst/integration/small_test.py": 100}
    nodeids = [f"{path}::test_{t}" for path in seconds for t in range(4)]
    groups = _groups(nodeids, 2, _table(seconds))
    assert {nodeid.split("::")[0] for nodeid in groups[0]} == {"tst/integration/big_test.py"}
    assert {nodeid.split("::")[0] for nodeid in groups[1]} == {"tst/integration/mid_test.py", "tst/integration/small_test.py"}


def test_a_file_named_in_split_by_test_is_split_by_test():
    nodeids = [f"{SPLIT}::test_{t}" for t in range(4)]
    groups = _groups(nodeids, 2, _table({nodeid: 60 for nodeid in nodeids}))
    assert [len(group) for group in groups] == [2, 2]


def test_tests_a_dependency_links_stay_on_one_shard_and_weigh_their_sum():
    nodeids = [f"{SPLIT}::test_{name}" for name in ("save", "load", "a", "b")]
    table = _table({f"{SPLIT}::test_save": 50, f"{SPLIT}::test_load": 50, f"{SPLIT}::test_a": 60, f"{SPLIT}::test_b": 40})
    linked = [(f"{SPLIT}::test_save", "save", []), (f"{SPLIT}::test_load", "test_load", ["save"])]
    together = shards.dependency_groups(linked, table)
    assert together == {f"{SPLIT}::test_save": f"{SPLIT}::test_load", f"{SPLIT}::test_load": f"{SPLIT}::test_load"}
    groups = [shards.select(nodeids, index, 2, table, together) for index in (1, 2)]
    # save+load (100) on one shard, a (60) and b (40) on the other
    assert sorted(groups, key=len) == [[f"{SPLIT}::test_save", f"{SPLIT}::test_load"],
                                       [f"{SPLIT}::test_a", f"{SPLIT}::test_b"]]


def test_a_dependency_outside_a_split_file_or_on_an_unknown_name_links_nothing():
    table = _table({})
    linked = [("tst/integration/whole_test.py::test_load", "test_load", ["save"]),
              ("tst/integration/whole_test.py::test_save", "save", []),
              (f"{SPLIT}::test_load", "test_load", ["missing"])]
    assert shards.dependency_groups(linked, table) == {}


def test_a_unit_the_table_does_not_know_weighs_the_median_of_the_known_ones():
    seconds = {"tst/integration/a_test.py": 10, "tst/integration/b_test.py": 20, "tst/integration/c_test.py": 900,
               "tst/integration/skipped_test.py": 0}
    nodeids = [f"{path}::test" for path in [*seconds, "tst/integration/new_test.py"]]
    groups = _groups(nodeids, 2, _table(seconds))
    # c (900) alone on one shard; the rest, the new file weighing the median of 10, 20 and 900, on the other
    assert groups[0] == ["tst/integration/c_test.py::test"]


def test_the_same_table_and_tests_always_give_the_same_shards():
    nodeids = [f"tst/integration/f{n}_test.py::test" for n in range(12)]
    table = _table({f"tst/integration/f{n}_test.py": 30 for n in range(12)})
    assert _groups(nodeids, 4, table) == _groups(list(nodeids), 4, table)


def test_the_committed_table_names_a_file_for_every_slow_tier_unit_it_splits():
    table = shards.load_table()
    assert SPLIT in table["split_by_test"]
    assert all((ROOT / path).is_file() for path in table["split_by_test"]), "split_by_test names a test file that is gone"
    assert all((ROOT / unit.split("::")[0]).is_file() for unit in table["seconds"]), "the table names a test file that is gone"


def _collect(*paths: str, shard: str | None) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != shards.ENV_VAR}
    if shard is not None:
        env[shards.ENV_VAR] = shard
    return subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *paths],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def _collected(result: subprocess.CompletedProcess) -> set[str]:
    return {line for line in result.stdout.splitlines() if "::" in line}


def test_the_hook_splits_a_collection_into_shards_that_together_are_all_of_it():
    paths = ("tst/unit/test_check_skip_policy.py", "tst/unit/test_tls_verification.py")
    everything = _collected(_collect(*paths, shard=None))
    first, second = (_collected(_collect(*paths, shard=f"{index}/2")) for index in (1, 2))
    assert first and second and not first & second
    assert first | second == everything


def test_the_hook_refuses_a_malformed_shard():
    result = _collect("tst/unit/test_tls_verification.py", shard="3/2")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert shards.ENV_VAR in result.stderr
