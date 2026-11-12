"""The /clock/watch long poll computers follow the gateway's clock through, and clock writes that wait for them."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

import httpx
import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.clock_followers import ClockFollowers
from agent_env.env.gateway.constants import (CLOCK_MAX_WAIT_FOLLOWERS_S, CLOCK_WATCH_HOLD_S, EXT_CLOCK_URI,
                                             GATEWAY_EXTENSIONS)
from agent_env.env.gateway.entrypoint import parse_clock_followers
from agent_env.env.gateway.gateway import Gateway

T0 = "2026-01-01T00:00:00Z"
TOKENS = {"desktop": "token-desktop", "laptop": "token-laptop"}
HELD_S = 5.0  # long enough that an answer "at once" can't be a held watch timing out, on a loaded runner too
SHORT_HOLD_S = 0.3


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    monkeypatch.setattr(ClockFollowers, "WATCH_S", HELD_S)
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


async def _until_shown(gw: Gateway, *names: str) -> None:
    """Until each of ``names`` has shown the clock's current version: its follower is up and holding a watch."""
    for _ in range(500):
        state = gw._clock_followers.state()
        if all(state[name]["version"] == gw._clock.version for name in names):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{names} never showed {gw._clock.version}: {gw._clock_followers.state()}")


def test_the_card_offers_every_clock_method_with_the_waits_bound_and_the_watchs_hold():
    methods = next(e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_CLOCK_URI)["params"]["methods"]
    wait = {"type": "number", "minimum": 0, "maximum": CLOCK_MAX_WAIT_FOLLOWERS_S}
    assert {name: (m["method"], m["endpoint"]) for name, m in methods.items()} == {
        "get_time": ("GET", "/clock/time"), "set_time": ("PUT", "/clock/set-time"), "set_rate": ("PUT", "/clock/set-rate"),
        "advance": ("PUT", "/clock/advance"), "clear": ("POST", "/clock/clear"), "state": ("GET", "/clock/state"),
        "watch": ("GET", "/clock/watch")}
    assert all(methods[name]["request"]["properties"]["wait_followers_s"] == wait
               for name in ("set_time", "set_rate", "advance", "clear"))
    assert methods["set_rate"]["request"]["required"] == ["virtual_seconds_per_real_second"]
    assert methods["set_time"]["request"]["required"] == ["virtual_time"]
    assert set(methods["watch"]["request"]["properties"]) == {"after", "follower"}
    assert methods["watch"]["hold_s"] == CLOCK_WATCH_HOLD_S == ClockFollowers.WATCH_S == 25.0


@pytest.mark.asyncio
async def test_a_watch_naming_another_version_answers_at_once_with_the_clock(gateway):
    gw = gateway()
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
        started = time.monotonic()
        answer = await _watch(client, "some-earlier-version")
    body = answer.json()
    assert answer.status_code == 200 and time.monotonic() - started < HELD_S / 2
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
async def test_a_watch_with_no_change_answers_after_its_hold_with_the_same_version(gateway, monkeypatch):
    monkeypatch.setattr(ClockFollowers, "WATCH_S", SHORT_HOLD_S)
    gw = gateway()
    async with _client(gw) as client:
        started = time.monotonic()
        answer = (await _watch(client, gw._clock.version)).json()
    assert time.monotonic() - started >= SHORT_HOLD_S
    assert answer == {**answer, "armed": False, "version": gw._clock.version}


@pytest.mark.asyncio
async def test_a_write_waits_until_every_declared_computer_shows_its_version(gateway):
    gw = gateway()
    async with _client(gw) as client:
        followers = [asyncio.create_task(_follow(client, name, token, applies_after_s=0.05)) for name, token in TOKENS.items()]
        try:
            await _until_shown(gw, *TOKENS)
            started = time.monotonic()
            written = await client.put("/clock/set-time", json={"virtual_time": T0, "wait_followers_s": 2})
            waited = time.monotonic() - started
            rated = await client.put("/clock/set-rate", json={"virtual_seconds_per_real_second": 60, "wait_followers_s": 2})
            advanced = await client.put("/clock/advance", json={"virtual_time": "2026-01-02T00:00:00Z", "wait_followers_s": 2})
            cleared = await client.post("/clock/clear", json={"wait_followers_s": 2})
            state = (await client.get("/clock/state")).json()
        finally:
            for follower in followers:
                follower.cancel()
    assert [r.status_code for r in (written, rated, advanced, cleared)] == [200] * 4
    assert 0.05 <= waited < 2
    assert {name: f["version"] for name, f in state["followers"].items()} == {"desktop": gw._clock.version,
                                                                              "laptop": gw._clock.version}


@pytest.mark.asyncio
async def test_a_write_answers_504_naming_the_computers_still_behind_and_the_change_stands(gateway):
    gw = gateway()
    async with _client(gw) as client:
        follower = asyncio.create_task(_follow(client, "desktop", TOKENS["desktop"]))
        try:
            await _until_shown(gw, "desktop")
            started = time.monotonic()
            answer = await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0,
                                                               "wait_followers_s": 0.4})
            waited = time.monotonic() - started
            read = (await client.get("/clock/time")).json()
        finally:
            follower.cancel()
    body = answer.json()
    assert answer.status_code == 504 and waited >= 0.4
    assert (body["ok"], body["stale"], body["version"]) == (False, ["laptop"], gw._clock.version)
    assert "laptop didn't show it within 0.4 s" in body["error"]
    assert read == {"virtual_time": T0}


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "token-desktop", "wrong"], ids=["none", "another-computers", "unknown"])
async def test_a_watch_acknowledges_only_with_its_computers_token(gateway, monkeypatch, token):
    monkeypatch.setattr(ClockFollowers, "WATCH_S", SHORT_HOLD_S)
    gw = gateway({"laptop": TOKENS["laptop"]})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        answered = await _watch(client, gw._clock.version, "laptop", token)
        state = (await client.get("/clock/state")).json()
    assert answered.status_code == 200
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
async def test_a_late_watch_naming_an_earlier_version_doesnt_move_its_computer_back(gateway):
    gw = gateway({"laptop": TOKENS["laptop"]})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        earlier = gw._clock.version
        await client.put("/clock/advance", json={"virtual_time": "2026-01-02T00:00:00Z"})
        latest = gw._clock.version
        held = asyncio.create_task(_watch(client, latest, "laptop", TOKENS["laptop"]))
        await _until_shown(gw, "laptop")
        await _watch(client, earlier, "laptop", TOKENS["laptop"])
        long = await _watch(client, f"{latest.split('.')[0]}.{'9' * 5000}", "laptop", TOKENS["laptop"])
        state = (await client.get("/clock/state")).json()
        waited = await client.put("/clock/set-time", json={"virtual_time": T0, "wait_followers_s": 0.5})
        held.cancel()
    assert state["followers"]["laptop"]["version"] == latest
    assert long.status_code == 200 and long.json()["version"] == latest
    assert waited.status_code == 504


@pytest.mark.asyncio
async def test_a_write_without_a_wait_or_without_computers_answers_at_once(gateway):
    with_computers, without = gateway(), gateway({})
    for gw, body in ((with_computers, {"virtual_time": T0}), (without, {"virtual_time": T0, "wait_followers_s": 5})):
        async with _client(gw) as client:
            started = time.monotonic()
            answer = await client.put("/clock/set-time", json=body)
        assert answer.status_code == 200 and time.monotonic() - started < 2.5
        assert "stale" not in answer.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [-1, "2", True, 31, 10**309, None],
                         ids=["negative", "string", "bool", "over-30", "past-a-floats-range", "null"])
async def test_a_wait_is_a_number_of_seconds_up_to_30(gateway, wait):
    gw = gateway({})
    async with _client(gw) as client:
        answers = [await client.put("/clock/set-time", json={"virtual_time": T0, "wait_followers_s": wait}),
                   await client.put("/clock/set-rate", json={"virtual_seconds_per_real_second": 2, "wait_followers_s": wait}),
                   await client.put("/clock/advance", json={"virtual_time": "2026-02-01T00:00:00Z", "wait_followers_s": wait}),
                   await client.post("/clock/clear", json={"wait_followers_s": wait})]
    if wait is None:
        assert [a.status_code for a in answers] == [200] * 4
    else:
        assert [a.status_code for a in answers] == [400] * 4
        assert all("wait_followers_s" in a.json()["error"] for a in answers)
        assert gw._clock.count_of(gw._clock.version) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path,method", [("/clock/set-time", "PUT"), ("/clock/set-rate", "PUT"),
                                         ("/clock/advance", "PUT"), ("/clock/clear", "POST")])
@pytest.mark.parametrize("body,error", [("{not json", "Invalid JSON body"), ("[]", "must be a JSON object"),
                                        ("null", "must be a JSON object")])
async def test_every_clock_write_refuses_a_body_that_isnt_a_json_object(gateway, path, method, body, error):
    gw = gateway({})
    async with _client(gw) as client:
        answer = await client.request(method, path, content=body)
    assert answer.status_code == 400 and error in answer.json()["error"]
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
    assert time.monotonic() - started < HELD_S / 2
    assert answer["armed"] is False and answer["version"] != version
    assert state["followers"]["desktop"]["version"] is None


@pytest.mark.asyncio
async def test_set_rate_changes_the_rate_at_the_time_of_the_write_and_keeps_t0(gateway):
    gw = gateway({})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0})
        read = (await client.get("/clock/state")).json()
        await asyncio.sleep(0.07)
        changed = await client.put("/clock/set-rate", json={"virtual_seconds_per_real_second": 2})
    body = changed.json()
    gap = (_utc(body["virtual_time"]) - _utc(read["virtual_time"])).total_seconds()
    assert changed.status_code == 200 and (body["virtual_seconds_per_real_second"], body["t0"], body["due"]) == (2, T0, [])
    assert gap >= 0.07


@pytest.mark.asyncio
@pytest.mark.parametrize("path,body,error", [
    ("/clock/set-time", {}, "virtual_time must be an RFC3339 string"),
    ("/clock/set-time", {"virtual_seconds_per_real_second": 2}, "virtual_time must be an RFC3339 string"),
    ("/clock/set-rate", {}, "rate must be a number"),
    ("/clock/set-rate", {"virtual_time": T0}, "rate must be a number"),
    ("/clock/set-rate", {"virtual_seconds_per_real_second": 10**309}, "exceeds max"),
    ("/clock/set-time", {"virtual_time": T0, "virtual_seconds_per_real_second": 10**309}, "exceeds max"),
])
async def test_set_time_needs_a_time_and_set_rate_a_rate_it_can_run_at(gateway, path, body, error):
    gw = gateway({})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
        answer = await client.put(path, json=body)
        state = (await client.get("/clock/state")).json()
    assert answer.status_code == 400 and error in answer.json()["error"]
    assert (state["virtual_seconds_per_real_second"], state["virtual_time"]) == (0, T0)


@pytest.mark.asyncio
async def test_set_rate_refuses_an_unarmed_clock(gateway):
    gw = gateway({})
    async with _client(gw) as client:
        answer = await client.put("/clock/set-rate", json={"virtual_seconds_per_real_second": 2})
    assert answer.status_code == 400 and "not armed" in answer.json()["error"]


@pytest.mark.asyncio
async def test_a_hundred_advances_are_each_acknowledged_in_order(gateway):
    gw = gateway({"desktop": TOKENS["desktop"]})
    async with _client(gw) as client:
        await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
        follower = asyncio.create_task(_follow(client, "desktop", TOKENS["desktop"]))
        try:
            await _until_shown(gw, "desktop")
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
    assert max(waits) < 2


def test_the_gateway_reads_its_declared_computers_from_its_environment():
    assert parse_clock_followers("") == {}
    assert parse_clock_followers(" desktop=read-d ,laptop=read-l,") == {"desktop": "read-d", "laptop": "read-l"}
    for spec, fragment in (("desktop", "expected 'name=read token'"), ("=tok", "expected"), ("desktop=", "expected"),
                           ("desktop=a,desktop=b", "twice")):
        with pytest.raises(ValueError, match=fragment):
            parse_clock_followers(spec)


@pytest.mark.asyncio
async def test_a_change_on_the_arm_wakes_the_triggers_and_a_re_arm_leaves_them_to_its_generation(gateway):
    gw = gateway({})
    async with _client(gw) as client:
        armed = (await client.put("/clock/set-time", json={"virtual_time": T0})).json()
        rated = (await client.put("/clock/set-rate", json={"virtual_seconds_per_real_second": 2})).json()
        advanced = (await client.put("/clock/advance", json={"virtual_time": "2026-02-01T00:00:00Z"})).json()
        cleared = (await client.post("/clock/clear")).json()
    assert ["due" in answer for answer in (armed, rated, advanced, cleared)] == [False, True, True, False]
