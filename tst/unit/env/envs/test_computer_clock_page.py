"""The clock page's format (version 3, utils/faketime_clock/clockpage.py): the universe at the anchor as whole seconds
and nanoseconds, so a page holds any date the gateway's clock takes, from before 1970 to 9999-12-31, where version 2's
signed 64-bit count of nanoseconds ran out on 2262-04-11; the specification libfaketime reads from the clock file is
exact at any of them; and a page of another version is never read as this one's."""

from __future__ import annotations

import getpass
import importlib.util
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import agent_env

CLOCK = Path(agent_env.__file__).parent / "utils" / "faketime_clock"
NS = 10**9
INT64_MAX_NS = 2**63 - 1
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def clockpage(tmp_path):
    spec = importlib.util.spec_from_file_location("clockpage_v3", CLOCK / "clockpage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PAGE, module.FILE = str(tmp_path / "agentenv-clock"), str(tmp_path / "run" / "clock")
    return module


def _ns(text: str) -> int:
    whole, _, part = text.partition(".")
    seconds = (datetime.fromisoformat(whole).replace(tzinfo=timezone.utc) - EPOCH) // timedelta(seconds=1)
    return seconds * NS + int((part or "0").ljust(9, "0"))


@pytest.mark.parametrize("universe_ns", [
    pytest.param(0, id="1970"),
    pytest.param(-1, id="a-nanosecond-before-1970"),
    pytest.param(_ns("1949-12-31T23:59:59.75"), id="1949-fractional"),
    pytest.param(_ns("1900-01-01T00:00:00"), id="1900"),
    pytest.param(INT64_MAX_NS, id="version-2s-last-nanosecond"),
    pytest.param(INT64_MAX_NS + 1, id="past-2262"),
    pytest.param(_ns("3026-11-23T12:00:00"), id="3026"),
    pytest.param(_ns("9999-12-31T23:59:59.999999999"), id="9999-12-31"),
])
def test_a_page_holds_the_universe_exactly_at_any_date(clockpage, universe_ns):
    page = clockpage.encode(4, universe_ns, 1_794_000_000 * NS, 3600.0, 1, 9, 5 * NS, 5 * NS)

    assert len(page) == clockpage.LAYOUT.size == 80
    assert clockpage.read(page) == (4, 1, universe_ns, 1_794_000_000 * NS, 3600.0, 9)
    assert 0 <= struct.unpack_from("<I", page, 44)[0] < NS


@pytest.mark.parametrize("change", [
    pytest.param(lambda p: p[:8] + (5).to_bytes(8, "little") + p[16:], id="being-written"),
    pytest.param(lambda p: p[:16] + bytes([p[16] ^ 1]) + p[17:], id="torn-seconds"),
    pytest.param(lambda p: p[:44] + bytes([p[44] ^ 1]) + p[45:], id="torn-nanoseconds"),
    pytest.param(lambda p: p[:4] + (2).to_bytes(4, "little") + p[8:], id="version-2"),
    pytest.param(lambda p: p[:40], id="short"),
])
def test_a_page_being_written_torn_short_or_of_another_version_isnt_read(clockpage, change):
    assert clockpage.read(change(clockpage.encode(4, _ns("9999-01-01T00:00:00"), 0, 1.0, 1, 2, 0, 0))) is None


@pytest.mark.parametrize("universe_ns, written", [
    (_ns("2026-11-23T12:00:00.5"), "1795435200.500000000"),
    (_ns("9999-12-31T23:59:59.999999999"), "253402300799.999999999"),
    (INT64_MAX_NS + 1, "9223372036.854775808"),
    # Before 1970 libfaketime reads "-S.N" as -(S + N/1e9): the sign written once, never "-S.-N".
    (_ns("1949-12-31T23:59:59.75"), "-631152000.250000000"),
    (-1, "-0.000000001"),
    (_ns("1900-01-01T00:00:00"), "-2208988800.000000000"),
])
def test_the_clock_files_specification_is_exact_at_any_date(clockpage, universe_ns, written):
    assert clockpage.spec(1, universe_ns, 2.0, 1_794_000_000 * NS + 5) == f"@{written} x2 ^1794000000.000000005"
    assert clockpage.spec(0, universe_ns, 2.0, 0) == "+0 x1"


def test_a_write_past_2262_is_what_the_page_and_the_clock_file_hold(clockpage):
    universe = _ns("9999-06-01T12:00:00") / NS
    written = clockpage.write(1, universe, 60.0, time.time())

    with open(clockpage.PAGE, "rb") as f:
        _, armed, universe_ns, anchor_ns, rate, generation = clockpage.read(f.read(clockpage.LAYOUT.size))
    assert (armed, universe_ns, rate, generation) == (1, _ns("9999-06-01T12:00:00"), 60.0, written["generation"])
    assert universe_ns == written["universe_ns"] and anchor_ns == written["anchor_ns"]
    assert Path(clockpage.FILE).read_text() == written["spec"] + "\n"
    assert written["spec"].startswith(f"@{_ns('9999-06-01T12:00:00') // NS}.000000000 x60 ^")


def test_a_rate_change_far_in_the_future_goes_on_from_where_the_universe_is(clockpage):
    """The continuity a write keeps (a rate change from a universe read a moment before) holds past 2262 too."""
    start = _ns("3026-01-01T00:00:00") / NS
    clockpage.write(1, start, 1.0, time.time())
    time.sleep(0.05)
    with open(clockpage.PAGE, "rb") as f:
        _, _, u, a, rate, _ = clockpage.read(f.read(clockpage.LAYOUT.size))
    now = clockpage._universe_at(1, u, a, rate, time.time_ns())
    written = clockpage.write(1, (now - 2_000_000) / NS, 1.0, time.time())

    assert written["continued"] and written["universe_ns"] >= now


def test_a_page_of_another_version_is_refused_by_the_writer_and_replaced_at_boot(clockpage):
    old = clockpage.encode(2, 0, 0, 1.0, 0, 1, 0, 0)
    Path(clockpage.PAGE).write_bytes(old[:4] + (2).to_bytes(4, "little") + old[8:])
    with pytest.raises(SystemExit, match="older page"):
        clockpage.open_page()

    clockpage.create(getpass.getuser())
    with open(clockpage.PAGE, "rb") as f:
        assert clockpage.read(f.read(clockpage.LAYOUT.size)) == (0, 0, 0, 0, 1.0, 0)
