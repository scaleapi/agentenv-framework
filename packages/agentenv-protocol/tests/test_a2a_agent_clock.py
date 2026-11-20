"""An agent's side of clock/v1: in an image that runs the clock follower, every AgentEnvAgent serves sync_time with no
code of its own; called by sync_env_clock, it writes the follower's config from the request, on real time, sets the
task's zone for what the server starts, and answers once the follower shows the gateway's current version, or says why
it doesn't; use_env_time, the opt-in, then has the follower arm the page as the gateway's clock is."""
from __future__ import annotations

import json
import os
import stat
import threading
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
from agentenv_protocol.a2a_agent import extension
from agentenv_protocol.a2a_agent.extensions import ClockSyncTimeRequest
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


EARLIER_STATUS_AGE_NS = 1_000_000_000


def _status_from_before(image, **status) -> None:
    """A status the follower wrote for an earlier config, ``EARLIER_STATUS_AGE_NS`` before this one: a file's mtime on
    Linux moves in ticks of a few milliseconds, so a config written right after a status can share its mtime."""
    _status(image, **status)
    before = time.time_ns() - EARLIER_STATUS_AGE_NS
    os.utime(image / "status.json", ns=(before, before))


def _follows_the_next_config(image, **status) -> threading.Thread:
    """A follower that, as the real one, writes its status once it reads a config other than the one it had."""
    config = image / "follower.json"
    before = config.read_bytes() if config.exists() else None

    def run() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if config.exists() and config.read_bytes() != before:
                _status(image, **status)
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


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
    _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver", version="b:3",
                             armed=True)
    with _app() as client:
        answer = client.post("/ext/clock/sync-time", json={**REQUEST, "timezone": "America/Chicago"})

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"synced": True, "timezone": "America/Chicago", "env_get_time_url": "http://gw/clock/time",
                             "follower": {"name": "agent-solver", "url": "http://gw/clock/watch", "version": "b:3",
                                          "armed": True}}
    config = image / "follower.json"
    assert json.loads(config.read_text()) == {"watch_url": "http://gw/clock/watch", "name": "agent-solver",
                                              "token": "token", "timezone": "America/Chicago", "env_time": False}
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    assert os.environ["TZ"] == "America/Chicago" and time.strftime("%Z") in {"CST", "CDT"}
    assert [p.name for p in image.iterdir() if p.name.startswith("follower.json.")] == []


def test_a_follower_behind_the_gateway_is_named(image):
    _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver", version="b:2")
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


@pytest.mark.parametrize("before", [
    {"url": "http://old-gw/clock/watch", "name": "agent-solver"},  # another gateway: it may answer the same version
    {"url": "http://gw/clock/watch", "name": "agent-solver"},  # this one, under the token the config replaced
])
def test_what_the_follower_showed_of_its_config_before_never_passes_for_the_new_one(image, before):
    _status_from_before(image, connected=True, version="b:3", armed=True, **before)
    with _app() as client:
        answer = client.post("/ext/clock/sync-time", json=REQUEST)

    assert answer.status_code == 404
    assert "still shows the config it had" in answer.text


def test_the_config_it_already_follows_resent_answers_from_its_status(image):
    with _app() as client:
        _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver",
                                 version="b:3")
        assert client.post("/ext/clock/sync-time", json=REQUEST).status_code == 200
        answer = client.post("/ext/clock/sync-time", json=REQUEST)  # the follower writes nothing anew

    assert answer.status_code == 200, answer.text


def test_a_server_restarted_mid_run_takes_back_the_tasks_zone(image, monkeypatch):
    _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver", version="b:3")
    with _app() as client:
        assert client.post("/ext/clock/sync-time", json={**REQUEST, "timezone": "America/Chicago"}).status_code == 200
    monkeypatch.setenv("TZ", "UTC")  # the restarted server's environment: the image's, without the task's zone
    time.tzset()

    with _app() as client:  # starting it again, as supervisord does after the server dies
        assert client.get("/.well-known/agent.json").status_code == 200

    assert os.environ["TZ"] == "America/Chicago" and time.strftime("%Z") in {"CST", "CDT"}


def _synced(image, client, request=REQUEST) -> None:
    _follows_the_next_config(image, connected=True, url=request["watch_url"], name=request["name"], version="b:3")
    assert client.post("/ext/clock/sync-time", json=request).status_code == 200


def _env_time(image):
    return json.loads((image / "follower.json").read_text())["env_time"]


def test_the_card_serves_use_env_time_beside_sync_time(image):
    with _app() as client:
        methods = _clock(client)["params"]["methods"]

    assert set(methods) == {"sync_time", "use_env_time"}
    assert methods["use_env_time"]["method"] == "POST"


def test_an_agent_that_serves_its_own_sync_time_still_starts_without_use_env_time(image):
    @a2a_agent(identity=AgentIdentity(name="own-clock", description="test", version="1"), extensions=(CLOCK_V1,))
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(CLOCK_V1.sync_time)
        async def sync_time(self, request: ClockSyncTimeRequest) -> dict:
            return {"synced": True, "timezone": "UTC", "follower": {}}

    with TestClient(Agent().create_app()) as client:
        assert set(_clock(client)["params"]["methods"]) == {"sync_time"}
        assert client.post("/ext/clock/use-env-time").status_code == 404


def test_an_agent_that_declares_the_clock_and_leaves_it_to_the_sdk_serves_use_env_time_too(image):
    """Its sync_time is the SDK's, which keeps real time until use_env_time, so it must serve that too."""
    with _app(extensions=(CLOCK_V1,)) as client:
        assert set(_clock(client)["params"]["methods"]) == {"sync_time", "use_env_time"}
        _synced(image, client)
        _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver", version="b:3")
        assert client.post("/ext/clock/use-env-time").status_code == 200

    assert _env_time(image) is True


def test_sync_time_alone_has_the_follower_follow_on_real_time(image):
    with _app() as client:
        _synced(image, client)

    assert _env_time(image) is False


def test_use_env_time_has_the_follower_arm_the_page_and_answers_once_it_shows_that_config(image):
    with _app() as client:
        _synced(image, client)
        _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver",
                                 version="b:3", armed=True)
        answer = client.post("/ext/clock/use-env-time")

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"env_time": True, "follower": {"name": "agent-solver", "url": "http://gw/clock/watch",
                                                            "version": "b:3", "armed": True}}
    assert json.loads((image / "follower.json").read_text()) == {
        "watch_url": "http://gw/clock/watch", "name": "agent-solver", "token": "token", "timezone": "UTC",
        "env_time": True}


def test_sync_time_again_keeps_env_time_for_the_same_gateway_and_name_and_not_for_another(image):
    with _app() as client:
        _synced(image, client)
        _follows_the_next_config(image, connected=True, url="http://gw/clock/watch", name="agent-solver", version="b:3")
        assert client.post("/ext/clock/use-env-time").status_code == 200
        _synced(image, client, {**REQUEST, "read_token": "another-token"})
        assert _env_time(image) is True
        _synced(image, client, {**REQUEST, "watch_url": "http://other-gw/clock/watch"})

    assert _env_time(image) is False


def test_use_env_time_before_sync_time_is_refused_and_writes_nothing(image):
    with _app() as client:
        answer = client.post("/ext/clock/use-env-time")

    assert answer.status_code == 400 and "sync_time" in answer.text
    assert not (image / "follower.json").exists()


def test_use_env_time_takes_no_request_fields(image):
    with _app() as client:
        _synced(image, client)
        assert client.post("/ext/clock/use-env-time", json={"x": 1}).status_code == 400


@pytest.mark.parametrize("config", ['{"watch_url": "u", "name": "n", "token": "t"}', '{"timezone": "Mars/Olympus"}',
                                    "not json", None])
def test_a_server_starting_without_a_saved_zone_keeps_its_own(image, monkeypatch, config):
    if config is not None:
        (image / "follower.json").write_text(config)
    with _app() as client:
        assert client.get("/.well-known/agent.json").status_code == 200
    assert os.environ["TZ"] == "UTC"


@pytest.mark.parametrize("spec, expected", [
    (None, None),
    ("+0 x1\n", None),
    ("@1748768400.000000000 x2 ^{past}\n", 1748768400.0 + 20),
    ("@1748768400.000000000 x1 ^{future}\n", 1748768400.0),
    ("@1748768400.000000000 x0 ^{past}\n", 1748768400.0),
])
def test_the_session_time_is_the_universe_while_the_clock_is_armed_and_real_time_otherwise(
        tmp_path, monkeypatch, spec, expected):
    clock_file = tmp_path / "clock"
    if spec is not None:
        clock_file.write_text(spec.format(past=time.time() - 10, future=time.time() + 3600))
    monkeypatch.setattr(follower, "CLOCK_FILE", str(clock_file))
    assert follower.session_time() == pytest.approx(time.time() if expected is None else expected, abs=1)
