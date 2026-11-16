"""An agent's side of clock/v1: in an image that runs the clock follower, every AgentEnvAgent serves sync_time with no
code of its own; called by sync_env_clock, it writes the follower's config from the request, sets the task's zone for
what the server starts, and answers once the follower shows the gateway's current version, or says why it doesn't."""
from __future__ import annotations

import json
import os
import stat
import time

import pytest
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import (
    CLOCK_V1,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    a2a_agent,
)
from agentenv_protocol.computer import clock as follower

REQUEST = {"env_get_time_url": "http://gw/clock/time", "watch_url": "http://gw/clock/watch", "name": "agent-solver",
           "read_token": "token"}


@pytest.fixture
def image(tmp_path, monkeypatch):
    """An image that runs the follower: its config and status under ``tmp_path``, the gateway at version b:3."""
    (tmp_path / "clock-follower").write_text("")
    monkeypatch.setattr(follower, "FOLLOWER", str(tmp_path / "clock-follower"))
    monkeypatch.setattr(follower, "DISK_GUEST", str(tmp_path / "image.env"))
    monkeypatch.setattr(follower, "CONFIG", str(tmp_path / "follower.json"))
    monkeypatch.setattr(follower, "STATUS", str(tmp_path / "status.json"))
    monkeypatch.setattr(follower, "CONNECT_WAIT_S", 0.3)
    monkeypatch.setattr(follower, "gateway_version", lambda url: "b:3")
    monkeypatch.setenv("TZ", "UTC")
    yield tmp_path
    time.tzset()


def _status(image, **status) -> None:
    (image / "status.json").write_text(json.dumps(status))


def _app(**declared):
    @a2a_agent(identity=AgentIdentity(name="clocked", description="test", version="1"), **declared)
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    return TestClient(Agent().create_app())


def _clock(client) -> dict | None:
    card = client.get("/.well-known/agent.json").json()
    return next((e for e in card["capabilities"]["extensions"] if e["uri"] == CLOCK_V1.uri), None)


def test_every_agent_in_an_image_with_the_follower_serves_sync_time(image):
    with _app() as client:
        served = _clock(client)

    assert served is not None
    request = served["params"]["methods"]["sync_time"]["request"]
    assert request == {"required": ["env_get_time_url"], "optional": ["timezone", "watch_url", "name", "read_token"]}


def test_an_agent_in_an_image_without_the_follower_serves_no_clock(image, monkeypatch):
    monkeypatch.setattr(follower, "FOLLOWER", str(image / "absent"))
    with _app() as client:
        assert _clock(client) is None


def test_sync_time_writes_the_followers_config_sets_the_zone_and_answers_once_it_follows(image):
    _status(image, connected=True, url="http://gw/clock/watch", name="agent-solver", version="b:3", armed=True)
    with _app() as client:
        answer = client.post("/ext/clock/sync-time", json={**REQUEST, "timezone": "America/Chicago"})

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"synced": True, "timezone": "America/Chicago", "env_get_time_url": "http://gw/clock/time",
                             "follower": {"name": "agent-solver", "url": "http://gw/clock/watch", "version": "b:3",
                                          "armed": True}}
    config = image / "follower.json"
    assert json.loads(config.read_text()) == {"watch_url": "http://gw/clock/watch", "name": "agent-solver",
                                              "token": "token"}
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    assert os.environ["TZ"] == "America/Chicago" and time.strftime("%Z") in {"CST", "CDT"}
    assert [p.name for p in image.iterdir() if p.name.startswith("follower.json.")] == []


def test_a_follower_behind_the_gateway_is_named(image):
    _status(image, connected=True, url="http://gw/clock/watch", version="b:2")
    with _app() as client:
        answer = client.post("/ext/clock/sync-time", json=REQUEST)

    assert answer.status_code == 404
    assert "this agent's clock follower isn't following the gateway: it shows the gateway's version 'b:2'" in answer.text


@pytest.mark.parametrize("missing", ["watch_url", "name", "read_token"])
def test_a_request_without_the_followers_config_is_refused_and_nothing_is_written(image, missing):
    with _app() as client:
        answer = client.post("/ext/clock/sync-time", json={k: v for k, v in REQUEST.items() if k != missing})

    assert answer.status_code == 400 and "watch_url, name and read_token" in answer.text
    assert not (image / "follower.json").exists()


def test_a_zone_that_isnt_one_is_refused(image):
    with _app() as client:
        answer = client.post("/ext/clock/sync-time", json={**REQUEST, "timezone": "Mars/Olympus"})

    assert answer.status_code == 400 and "IANA" in answer.text and os.environ["TZ"] == "UTC"
