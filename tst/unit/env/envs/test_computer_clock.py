"""The computer's clock follower and the clock app's sync_time, run here against stand-ins for the gateway's watch and
the clock page; the real thing is in tst/integration/computer/clock_follower_test.py."""
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from agentenv_protocol.agent_env_environment import _EXT_ATTR
from agentenv_protocol.computer.errors import InvalidArgument, Timeout

import agent_env

CLOCK = Path(agent_env.__file__).parent / "computers" / "clock"
APP = Path(agent_env.__file__).parent / "computers" / "envs" / "clock"


@pytest.fixture
def follower(monkeypatch, tmp_path):
    """The follower's module, its clock page writes recorded instead of made, its config and status in ``tmp_path``."""
    monkeypatch.syspath_prepend(str(CLOCK))
    loader = importlib.machinery.SourceFileLoader("clock_follower", str(CLOCK / "clock-follower"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("clock_follower", loader))
    loader.exec_module(module)
    writes = []
    monkeypatch.setattr(module.clockpage, "write",
                        lambda armed, universe, rate, anchor: writes.append((armed, universe, rate, anchor)) or
                        {"spec": f"@{universe} x{rate:g}" if armed else "+0 x1"})
    monkeypatch.setattr(module, "CONFIG", str(tmp_path / "follower.json"))
    monkeypatch.setattr(module, "STATUS", str(tmp_path / "run" / "status.json"))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    module.writes = writes
    yield module
    sys.modules.pop("clock_follower", None)


def _configure(module, config_id="c1"):
    Path(module.CONFIG).write_text(json.dumps({"watch_url": "http://gw/clock/watch", "token": "t", "follower": "desktop",
                                               "config_id": config_id}))


def _armed(version, virtual_time="2026-05-01T09:00:00Z", rate=1.0, received=1000.05, sent=1000.15):
    return {"armed": True, "version": version, "virtual_time": virtual_time, "virtual_seconds_per_real_second": rate,
            "received": received, "sent": sent}


def _run(module, answers):
    """The follower's loop over ``answers``, each a watch's (state, local sent, local received) or an exception; the
    ``after`` each watch carried."""
    afters, answers = [], list(answers)

    def watch(config, after):
        afters.append(after)
        if not answers:
            raise SystemExit
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    module.watch = watch
    with pytest.raises(SystemExit):
        module.main()
    return afters


def test_the_page_is_written_from_the_local_time_the_gateways_sent_stamp_corresponds_to(follower):
    # The gateway's clock is 900 s ahead of ours: it received at 1000.05 what we sent at 100, and sent at 1000.15 what
    # we received at 100.2; so its sent stamp is our 100.15.
    follower.show(_armed("b.1", rate=60), 100.0, 100.2)
    (armed, universe, rate, anchor), = follower.writes
    assert (armed, rate) == (1, 60.0) and anchor == pytest.approx(100.15)
    assert universe == follower.epoch("2026-05-01T09:00:00Z")


def test_each_new_version_is_written_once_acknowledged_by_the_next_watch_and_recorded(follower):
    _configure(follower)
    held = _armed("b.2")  # the same version again: a watch that ran out its hold
    afters = _run(follower, [({"armed": False, "version": "b.1", "received": 1, "sent": 1}, 1, 1),
                             (_armed("b.2"), 1, 1), (held, 1, 1)])

    assert afters == ["", "b.1", "b.2", "b.2"]
    assert [w[0] for w in follower.writes] == [0, 1]
    status = json.loads(Path(follower.STATUS).read_text())
    assert (status["config_id"], status["version"], status["armed"]) == ("c1", "b.2", True)


def test_a_new_config_starts_over_from_no_version(follower):
    _configure(follower, "c1")

    def watch(config, after):
        if config["config_id"] == "c1" and after == "b.1":
            _configure(follower, "c2")  # core re-arms with a new config while a watch is held
        if len(follower.writes) == 2:
            raise SystemExit
        return _armed("b.1"), 1, 1

    follower.watch = watch
    with pytest.raises(SystemExit):
        follower.main()
    assert json.loads(Path(follower.STATUS).read_text())["config_id"] == "c2"


def test_a_failed_watch_is_recorded_and_tried_again(follower):
    _configure(follower)
    afters = _run(follower, [OSError("connection refused"), (_armed("b.1"), 1, 1)])

    assert afters == ["", "", "b.1"]
    assert json.loads(Path(follower.STATUS).read_text())["version"] == "b.1"


def test_without_a_config_the_follower_waits_for_one(follower, monkeypatch):
    slept = []
    monkeypatch.setattr(follower.time, "sleep", lambda s: slept.append(s) or (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        follower.main()
    assert slept == [1]


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(APP))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    monkeypatch.setattr(module, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(module, "SYNC_WAIT_S", 0.5)
    yield module
    sys.modules.pop("app", None)


@pytest.mark.asyncio
async def test_the_clock_apps_sync_time_reports_what_the_follower_showed_for_the_config_core_wrote(app):
    app.STATUS.write_text(json.dumps({"config_id": "c1", "version": "b.2", "armed": True}))
    result = await app.Clock.sync_time(None, env_get_time_url="http://gw/clock/time", follower_config="c1")
    assert result["follower"]["version"] == "b.2"


@pytest.mark.asyncio
async def test_the_clock_apps_sync_time_times_out_on_another_config_or_a_failing_follower(app):
    app.STATUS.write_text(json.dumps({"config_id": "c0", "version": "b.2"}))
    with pytest.raises(Timeout, match="didn't show config 'c1'"):
        await app.Clock.sync_time(None, env_get_time_url="u", follower_config="c1")
    app.STATUS.write_text(json.dumps({"config_id": "c1", "version": "b.2", "error": "URLError: refused"}))
    with pytest.raises(Timeout, match="refused"):
        await app.Clock.sync_time(None, env_get_time_url="u", follower_config="c1")
    with pytest.raises(InvalidArgument):
        await app.Clock.sync_time(None, env_get_time_url="u")


def test_the_clock_app_advertises_a_sync_time_that_takes_the_followers_config(app):
    extension = getattr(app.Clock.sync_time, _EXT_ATTR)
    request = extension.params["methods"]["sync_time"]["request"]
    assert extension.uri == "urn:agentenv:clock/v1"
    assert {"env_get_time_url", "follower_config"} <= set(request["properties"])


def test_the_follower_writes_nothing_another_user_may_change_whatever_umask_it_started_with(follower, monkeypatch):
    masks = []
    monkeypatch.setattr(follower.os, "umask", lambda mask: masks.append(mask) or 0)
    _configure(follower)
    _run(follower, [])
    assert masks == [0o022]
