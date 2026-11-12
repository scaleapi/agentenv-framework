"""Core's side of a computer's clock: which computers run the clock follower, and its config written and the follower
started as root through the sandbox's root channel."""
import json
from types import SimpleNamespace

import pytest

from agent_env.env.envs.computer.utils import clock


def test_a_computer_runs_the_follower_when_it_composes_the_clock_app():
    method = {"method": "POST", "request": {"properties": {"follower_config": {"type": "string"}}}}
    clock_app = {"name": "clock", "capabilities": {"extensions": [
        {"uri": "urn:agentenv:clock/v1", "params": {"methods": {"sync_time": method}}}]}}
    driver_clock = {"name": "desktop", "capabilities": {"extensions": [
        {"uri": "urn:agentenv:clock/v1", "params": {"methods": {"sync_time": {"method": "POST"}}}}]}}

    assert clock.runs_follower({"name": "desktop", "children_environments": [clock_app]})
    assert not clock.runs_follower(driver_clock) and clock.follows_clock(driver_clock)
    assert clock.takes_config(clock_app) and not clock.takes_config(driver_clock)


@pytest.mark.asyncio
async def test_start_follower_writes_its_config_as_root_for_its_user_alone_then_starts_it(monkeypatch):
    runs, secrets = [], {}

    class Channel:
        def __init__(self, sandbox):
            pass

        async def run(self, argv):
            runs.append(argv)

        async def write_secret(self, path, text):
            secrets[str(path)] = json.loads(text)

    monkeypatch.setattr(clock, "RootChannel", Channel)
    config_id = await clock.start_follower(SimpleNamespace(), watch_url="http://h/clock/watch", token="t", follower="d")

    assert secrets == {str(clock.CONFIG): {"watch_url": "http://h/clock/watch", "token": "t", "follower": "d",
                                           "config_id": config_id}}
    assert ["chmod", "0600", str(clock.CONFIG)] in runs
    assert runs[0][:4] == ["install", "-d", "-m", "0700"]
    assert runs[-2] == ["python3", "/opt/agentenv/clock/clockpage.py", "create", "agentenv-clock"]
    assert "setsid /opt/agentenv/clock/follow" in runs[-1][-1]


@pytest.mark.asyncio
async def test_wait_shown_reads_the_followers_status_as_root_until_it_shows_the_config(monkeypatch):
    statuses = ["", json.dumps({"config_id": "c0", "version": "b.1"}), json.dumps({"config_id": "c1", "version": "b.2"})]

    class Channel:
        def __init__(self, sandbox):
            pass

        async def run(self, argv):
            return statuses.pop(0) if len(statuses) > 1 else statuses[0]

    monkeypatch.setattr(clock, "RootChannel", Channel)
    monkeypatch.setattr(clock.asyncio, "sleep", lambda s: _done())
    assert (await clock.wait_shown(SimpleNamespace(), "c1"))["version"] == "b.2"
    with pytest.raises(TimeoutError, match="didn't show config 'c9'"):
        await clock.wait_shown(SimpleNamespace(), "c9", timeout_s=0)


async def _done():
    return None
