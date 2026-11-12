"""A computer's side of clock/v1: the driver serves sync_time where the base runs the clock follower and it isn't a disk
guest, the method every server that follows the clock serves; it checks the task's zone against the one the computer
booted with and answers once the follower shows the gateway's current version, or says why it doesn't."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver, http_client  # noqa: E402

from agentenv_protocol.computer import clock  # noqa: E402
from agentenv_protocol.computer.errors import InvalidArgument, TargetUnavailable  # noqa: E402

CLOCK = "urn:agentenv:clock/v1"


@pytest.fixture
def base(tmp_path, monkeypatch):
    """A base that runs the follower, its status at ``tmp_path/status.json``, the session in Europe/Paris."""
    follower = tmp_path / "clock-follower"
    follower.write_text("")
    monkeypatch.setattr(clock, "FOLLOWER", str(follower))
    monkeypatch.setattr(clock, "DISK_GUEST", str(tmp_path / "image.env"))
    monkeypatch.setattr(clock, "STATUS", str(tmp_path / "status.json"))
    monkeypatch.setattr(clock, "CONNECT_WAIT_S", 0.3)
    monkeypatch.setenv("TZ", "Europe/Paris")
    monkeypatch.setattr(clock, "gateway_version", lambda url: "b:3")
    return tmp_path


async def _clock_extension(driver) -> dict | None:
    async with http_client(driver) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
    return next((e for e in card["capabilities"]["extensions"] if e["uri"] == CLOCK), None)


@pytest.mark.asyncio
async def test_the_card_serves_sync_time_where_the_base_runs_the_follower_and_it_isnt_a_disk_guest(base, monkeypatch):
    served = await _clock_extension(FakeDriver())
    (base / "image.env").write_text("user=root\n")
    on_a_disk = await _clock_extension(FakeDriver())
    monkeypatch.setattr(clock, "FOLLOWER", str(base / "absent"))

    assert set(served["params"]["methods"]) == {"sync_time"}
    assert set(served["params"]["methods"]["sync_time"]["request"]["properties"]) == {"env_get_time_url", "timezone"}
    assert on_a_disk is None and await _clock_extension(FakeDriver()) is None


@pytest.mark.asyncio
async def test_sync_time_answers_once_the_follower_shows_the_gateways_current_version(base):
    (base / "status.json").write_text(json.dumps({"connected": True, "url": "http://gw/clock/watch", "name": "desktop",
                                                  "version": "b:3", "armed": True, "token": "never read"}))

    synced = await clock.sync_time("http://gw/clock/time", " Europe/Paris ")

    assert synced == {"synced": True, "timezone": "Europe/Paris", "env_get_time_url": "http://gw/clock/time",
                      "follower": {"name": "desktop", "url": "http://gw/clock/watch", "version": "b:3", "armed": True}}


@pytest.mark.parametrize("version, why", [
    ("b:3", "it shows the gateway's version 'b:2', and the gateway is at 'b:3'"),
    (OSError("refused"), "this computer can't read the gateway's version at http://gw/clock/watch: OSError: refused"),
])
@pytest.mark.asyncio
async def test_a_follower_behind_the_gateways_version_or_a_gateway_it_cant_read_is_named(base, monkeypatch, version,
                                                                                        why):
    def answer(url):
        assert url == "http://gw/clock/watch"
        if isinstance(version, Exception):
            raise version
        return version

    monkeypatch.setattr(clock, "gateway_version", answer)
    (base / "status.json").write_text(json.dumps({"connected": True, "url": "http://gw/clock/watch", "version": "b:2"}))

    with pytest.raises(TargetUnavailable, match="isn't following the gateway: " + why.replace("(", r"\(")):
        await clock.sync_time("http://gw/clock/time")


@pytest.mark.asyncio
async def test_a_task_in_another_zone_is_refused_naming_the_setting_to_change(base):
    with pytest.raises(InvalidArgument, match="deploy it with computer_settings.timezone = 'America/Chicago'"):
        await clock.sync_time("http://gw/clock/time", "America/Chicago")


@pytest.mark.parametrize("status, why", [
    (None, "core hasn't written its config"),
    ({"connected": False, "url": "http://gw/clock/watch", "error": "URLError: refused"},
     "it can't reach the gateway at http://gw/clock/watch: URLError: refused"),
    ({"connected": False, "url": "http://gw/clock/watch", "error": None}, "it hasn't heard from the gateway"),
])
@pytest.mark.asyncio
async def test_a_follower_that_isnt_connected_is_named_with_why(base, status, why):
    if status is not None:
        (base / "status.json").write_text(json.dumps(status))

    with pytest.raises(TargetUnavailable, match="this computer's clock follower isn't following the gateway: " + why):
        await clock.sync_time("http://gw/clock/time")


@pytest.mark.asyncio
async def test_a_computer_whose_base_doesnt_follow_the_clock_says_so(base, monkeypatch):
    monkeypatch.setattr(clock, "FOLLOWER", str(base / "absent"))

    with pytest.raises(TargetUnavailable, match="doesn't follow the universe clock"):
        await clock.sync_time("http://gw/clock/time")
