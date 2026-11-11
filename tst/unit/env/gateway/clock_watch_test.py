"""The clock's version, the /clock/watch long poll computers follow it through, and clock writes that wait for them."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

import httpx
import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock, ClockError
from agent_env.env.gateway.clock_followers import ClockFollowers
from agent_env.env.gateway.gateway import Gateway
from tst.unit.env.gateway.triggers_time_test import FakeNow

T0 = "2026-01-01T00:00:00Z"
TOKENS = {"desktop": "token-desktop", "laptop": "token-laptop"}


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_every_write_changes_the_version_and_reads_leave_it():
    clock = Clock(now_fn=FakeNow())
    seen = [clock.version]
    clock.set_time(T0, rate=1.0)
    seen.append(clock.version)
    clock.read(), clock.state(), clock.now()
    assert clock.version == seen[-1]
    clock.advance("2026-01-02T00:00:00Z")
    seen.append(clock.version)
    clock.set_time(None, rate=60)
    seen.append(clock.version)
    clock.clear()
    seen.append(clock.version)
    clock.clear()
    seen.append(clock.version)
    assert len(set(seen)) == len(seen)
    assert [clock.count_of(version) for version in seen] == [0, 1, 2, 3, 4, 5]


def test_a_version_never_repeats_across_clocks_and_names_only_what_its_clock_had():
    first, second = Clock(now_fn=FakeNow()), Clock(now_fn=FakeNow())
    assert first.version != second.version
    first.set_time(T0)
    boot = first.version.split(".")[0]
    assert first.count_of(first.version) == 1
    assert first.count_of(second.version) is None
    assert [first.count_of(v) for v in (f"{boot}.2", f"{boot}.-1", f"{boot}.²", f"{boot}", "", None, 3)] == [None] * 7


def test_a_rate_change_without_a_time_continues_from_the_clock_as_of_the_write():
    """Taking the new time from a state read a moment earlier stepped the universe back by the gap, which a computer,
    never stepping back, then ran ahead of."""
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(T0, rate=1.0)
    now.tick(10)
    read = clock.state()
    now.tick(0.07)
    state = clock.set_time(None, rate=60)
    assert _utc(state["virtual_time"]) - _utc(read["virtual_time"]) == timedelta(milliseconds=70)
    assert state["virtual_seconds_per_real_second"] == 60
    now.tick(1)
    assert clock.read()["virtual_time"] == "2026-01-01T00:01:10.070000Z"


def test_a_rate_change_hands_followers_a_clock_continuous_with_the_one_before():
    """A follower extrapolating the snapshot before the change to the moment of the write is where the one after starts."""
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(T0, rate=1.0)
    before, at = clock.state(), now()
    now.tick(3.25)
    after = clock.set_time(None, rate=2)
    extrapolated = _utc(before["virtual_time"]) + timedelta(seconds=(now() - at) * before["virtual_seconds_per_real_second"])
    assert abs((_utc(after["virtual_time"]) - extrapolated).total_seconds()) < 0.001


def test_a_time_is_needed_to_arm_the_clock():
    with pytest.raises(ClockError, match="required to arm"):
        Clock(now_fn=FakeNow()).set_time(None, rate=2)


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    monkeypatch.setattr(ClockFollowers, "WATCH_S", 0.3)
    made = []

    def make(followers=TOKENS) -> Gateway:
        made.append(Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[],
                            clock_followers=followers))
        return made[-1]

    yield make
    for gw in made:
        gw._close_trajectory_file()


def _client(gw: Gateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway")


async def _watch(client, after: str, follower: str | None = None, token: str | None = None) -> httpx.Response:
    params = {"after": after, **({"follower": follower} if follower else {})}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return await client.get("/clock/watch", params=params, headers=headers)


async def _follow(client, name: str, token: str, *, applies_after_s: float = 0.0):
    """A computer's follower: each answer shows a version, applied after ``applies_after_s``, and the next watch names it."""
    version = ""
    while True:
        answer = (await _watch(client, version, name, token)).json()
        if answer["version"] != version:
            await asyncio.sleep(applies_after_s)
            version = answer["version"]


@pytest.mark.asyncio
async def test_a_watch_naming_another_version_answers_at_once_with_the_clock(gateway):
    gw = gateway()
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
        started = time.monotonic()
        answer = await _watch(client, "some-earlier-version")
    body = answer.json()
    assert answer.status_code == 200 and time.monotonic() - started < 0.2
    assert body["version"] == gw._clock.version
    assert (body["armed"], body["t0"], body["virtual_time"], body["virtual_seconds_per_real_second"]) == (True, T0, T0, 0)
    assert body["received"] <= body["sent"] <= time.time()


@pytest.mark.asyncio
async def test_a_watch_naming_the_current_version_answers_at_the_next_change(gateway):
    gw = gateway()
    async with _client(gw) as client:
        held = asyncio.create_task(_watch(client, gw._clock.version))
        await asyncio.sleep(0.1)
        assert not held.done()
        await client.put("/clock/set-time", json={"virtual_time": T0})
        answer = (await held).json()
    assert answer["armed"] and answer["version"] == gw._clock.version
    assert answer["received"] < answer["sent"] - 0.09


@pytest.mark.asyncio
async def test_a_watch_with_no_change_answers_after_its_timeout_with_the_same_version(gateway):
    gw = gateway()
    async with _client(gw) as client:
        started = time.monotonic()
        answer = (await _watch(client, gw._clock.version)).json()
    assert time.monotonic() - started >= ClockFollowers.WATCH_S
    assert answer == {**answer, "armed": False, "version": gw._clock.version}


@pytest.mark.asyncio
async def test_a_write_waits_until_every_declared_computer_shows_its_version(gateway):
    gw = gateway()
    async with _client(gw) as client:
        followers = [asyncio.create_task(_follow(client, name, token, applies_after_s=0.05)) for name, token in TOKENS.items()]
        try:
            await asyncio.sleep(0.1)
            started = time.monotonic()
            written = await client.put("/clock/set-time", json={"virtual_time": T0, "wait_followers_s": 2})
            waited = time.monotonic() - started
            advanced = await client.put("/clock/advance", json={"virtual_time": "2026-01-02T00:00:00Z", "wait_followers_s": 2})
            cleared = await client.post("/clock/clear", json={"wait_followers_s": 2})
            state = (await client.get("/clock/state")).json()
        finally:
            for follower in followers:
                follower.cancel()
    assert [r.status_code for r in (written, advanced, cleared)] == [200, 200, 200]
    assert 0.05 <= waited < 1.0
    assert {name: f["version"] for name, f in state["followers"].items()} == {"desktop": gw._clock.version,
                                                                              "laptop": gw._clock.version}


@pytest.mark.asyncio
async def test_a_write_answers_504_naming_the_computers_still_behind_and_the_change_stands(gateway):
    gw = gateway()
    async with _client(gw) as client:
        follower = asyncio.create_task(_follow(client, "desktop", TOKENS["desktop"]))
        try:
            await asyncio.sleep(0.1)
            started = time.monotonic()
            answer = await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0,
                                                               "wait_followers_s": 0.4})
            waited = time.monotonic() - started
            read = (await client.get("/clock/time")).json()
        finally:
            follower.cancel()
    body = answer.json()
    assert answer.status_code == 504 and 0.4 <= waited < 1.0
    assert (body["ok"], body["stale"], body["version"]) == (False, ["laptop"], gw._clock.version)
    assert "laptop didn't show it within 0.4 s" in body["error"]
    assert read == {"virtual_time": T0}


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "token-desktop", "wrong"], ids=["none", "another-computers", "unknown"])
async def test_a_watch_acknowledges_only_with_its_computers_token(gateway, token):
    gw = gateway({"laptop": TOKENS["laptop"]})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        held = asyncio.create_task(_watch(client, gw._clock.version, "laptop", token))
        await asyncio.sleep(0.05)
        state = (await client.get("/clock/state")).json()
        held.cancel()
    assert state["followers"] == {"laptop": {"version": None, "seen_s_ago": None}}


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["later", "another-clocks"])
async def test_an_acknowledgement_cant_claim_a_version_the_clock_hasnt_had(gateway, claim):
    gw = gateway({"laptop": TOKENS["laptop"]})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        boot, count = gw._clock.version.split(".")
        claimed = f"{boot}.{int(count) + 5}" if claim == "later" else Clock().version
        await _watch(client, claimed, "laptop", TOKENS["laptop"])
        answer = await client.put("/clock/advance", json={"virtual_time": "2026-02-01T00:00:00Z", "wait_followers_s": 0.1})
    assert answer.status_code == 504 and answer.json()["stale"] == ["laptop"]


@pytest.mark.asyncio
async def test_a_write_without_a_wait_or_without_computers_answers_at_once(gateway):
    with_computers, without = gateway(), gateway({})
    for gw, body in ((with_computers, {"virtual_time": T0}), (without, {"virtual_time": T0, "wait_followers_s": 5})):
        async with _client(gw) as client:
            started = time.monotonic()
            answer = await client.put("/clock/set-time", json=body)
        assert answer.status_code == 200 and time.monotonic() - started < 0.2
        assert "stale" not in answer.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [-1, "2", True, 31, None], ids=["negative", "string", "bool", "over-30", "null"])
async def test_a_wait_is_a_number_of_seconds_up_to_30(gateway, wait):
    gw = gateway({})
    async with _client(gw) as client:
        answers = [await client.put("/clock/set-time", json={"virtual_time": T0, "wait_followers_s": wait}),
                   await client.put("/clock/advance", json={"virtual_time": "2026-02-01T00:00:00Z", "wait_followers_s": wait}),
                   await client.post("/clock/clear", json={"wait_followers_s": wait})]
    if wait is None:
        assert [a.status_code for a in answers] == [200, 200, 200]
    else:
        assert [a.status_code for a in answers] == [400, 400, 400]
        assert all("wait_followers_s" in a.json()["error"] for a in answers)
        assert gw._clock.count_of(gw._clock.version) == 0


@pytest.mark.asyncio
async def test_clear_still_takes_no_body(gateway):
    gw = gateway({})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        answer = await client.post("/clock/clear")
    assert answer.status_code == 200 and answer.json() == {"ok": True, "armed": False, "version": gw._clock.version}


@pytest.mark.asyncio
async def test_a_restarted_gateway_answers_a_watch_from_before_it_at_once_and_unarmed(gateway):
    before = gateway()
    async with _client(before) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
    version = before._clock.version
    restarted = gateway()
    async with _client(restarted) as client:
        started = time.monotonic()
        answer = (await _watch(client, version, "desktop", TOKENS["desktop"])).json()
        state = (await client.get("/clock/state")).json()
    assert time.monotonic() - started < 0.2
    assert answer["armed"] is False and answer["version"] != version
    assert state["followers"]["desktop"]["version"] is None


@pytest.mark.asyncio
async def test_the_route_changes_the_rate_at_the_time_of_the_write(gateway):
    gw = gateway({})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        read = (await client.get("/clock/state")).json()
        await asyncio.sleep(0.07)
        changed = await client.put("/clock/set-time", json={"virtual_seconds_per_real_second": 2})
    body = changed.json()
    gap = (_utc(body["virtual_time"]) - _utc(read["virtual_time"])).total_seconds()
    assert changed.status_code == 200 and body["virtual_seconds_per_real_second"] == 2
    assert 0.07 <= gap < 0.2


@pytest.mark.asyncio
async def test_a_rate_change_of_an_unarmed_clock_is_refused(gateway):
    gw = gateway({})
    async with _client(gw) as client:
        answer = await client.put("/clock/set-time", json={"virtual_seconds_per_real_second": 2})
    assert answer.status_code == 400 and "required to arm" in answer.json()["error"]


@pytest.mark.asyncio
async def test_a_hundred_advances_are_each_acknowledged_in_order(gateway):
    gw = gateway({"desktop": TOKENS["desktop"]})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
        follower = asyncio.create_task(_follow(client, "desktop", TOKENS["desktop"]))
        try:
            await asyncio.sleep(0.05)
            answers, waits = [], []
            for hour in range(1, 101):
                started = time.monotonic()
                answers.append(await client.put("/clock/advance", json={
                    "virtual_time": f"2026-01-{1 + hour // 24:02d}T{hour % 24:02d}:00:00Z", "wait_followers_s": 2}))
                waits.append(time.monotonic() - started)
        finally:
            follower.cancel()
    assert [a.status_code for a in answers] == [200] * 100
    counts = [gw._clock.count_of(a.json()["version"]) for a in answers]
    assert counts == sorted(counts) and len(set(counts)) == 100
    assert sorted(waits)[50] < 0.05
