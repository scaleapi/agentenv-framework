"""Upstream's setup and checks, run in the worker: a check is upstream's metric over what its result getter reads, against
its expected getter's when it has one, with its options; a result file the guest doesn't have scores 0; a func or getter
upstream doesn't have is an UnknownCheck; setup is upstream's SetupController at the guest's screen size; and every
call reaches the guest only through its loopback forwarder. Upstream's modules are stood in for (the integration test
runs the real ones)."""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import requests
from pytest_socket import enable_socket, socket_allow_hosts

from agent_env.task_step.task_steps.computer.osworld import runtime
from agent_env.task_step.task_steps.computer.osworld.forwarder import LOOPBACK

GUEST = "https://computer.example/svc/osworld-server/osworld"


@pytest.fixture(autouse=True)
def _loopback():
    """The forwarders listen on loopback; nothing else is reached."""
    enable_socket()
    socket_allow_hosts(["127.0.0.1"], allow_unix_socket=True)


class _Python:
    def __init__(self, vm_ip, server_port):
        self.vm_ip, self.server_port = vm_ip, server_port

    def get_vm_screen_size(self):
        return {"width": 1280, "height": 800}


class _Setup:
    made: list[dict] = []
    answer = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _Setup.made.append(kwargs)

    def setup(self, config):
        self.kwargs["config"] = config
        return _Setup.answer


@pytest.fixture
def upstream(monkeypatch):
    seen = []

    def get_vm_file(env, config):
        seen.append((env, config))
        if config["path"] == "/missing":
            raise FileNotFoundError(config["path"])
        return f"read {config['path']}"

    def exact(result, expected=None, **options):
        seen.append((result, expected, options))
        return 1.0 if expected is None or result == expected else 0.0

    modules = SimpleNamespace(setup=SimpleNamespace(SetupController=_Setup, MAX_RETRIES=20),
                              python=SimpleNamespace(PythonController=_Python),
                              getters=SimpleNamespace(get_vm_file=get_vm_file),
                              metrics=SimpleNamespace(exact=exact, _private=exact))
    monkeypatch.setattr(runtime.upstream, "load", lambda: modules)
    _Setup.made, _Setup.answer = [], True
    return SimpleNamespace(modules=modules, seen=seen)


def test_a_check_is_the_metric_over_its_getters_with_its_options(upstream):
    assert runtime.evaluate(GUEST, {}, "exact", {"type": "vm_file", "path": "/a"},
                            {"type": "vm_file", "path": "/a"}, {"strict": True}) == 1.0
    env, _ = upstream.seen[0]
    assert (env.vm_ip, env.cache_dir.startswith("/") and "osworld-" in env.cache_dir) == (LOOPBACK, True)
    assert len({env.server_port, env.chromium_port, env.vlc_port}) == 3
    assert upstream.seen[-1] == ("read /a", "read /a", {"strict": True})
    assert runtime.evaluate(GUEST, {}, "exact", {"type": "vm_file", "path": "/a"}, None, {}) == 1.0
    assert upstream.seen[-1] == ("read /a", None, {})


def test_a_result_file_the_guest_doesnt_have_scores_0(upstream):
    assert runtime.evaluate(GUEST, {}, "exact", {"type": "vm_file", "path": "/missing"}, None, {}) == 0.0


@pytest.mark.parametrize("func, result, match", [("nope", {"type": "vm_file"}, "no metric 'nope'"),
                                                 ("_private", {"type": "vm_file"}, "no metric '_private'"),
                                                 ("exact", {"type": "nope"}, "no getter 'get_nope'")])
def test_a_func_or_getter_upstream_doesnt_have_is_unknown(upstream, func, result, match):
    with pytest.raises(runtime.UnknownCheck, match=match):
        runtime.evaluate(GUEST, {}, func, result, None, {})


def test_a_check_with_no_result_getter_fails(upstream):
    with pytest.raises(ValueError, match="no result getter"):
        runtime.evaluate(GUEST, {}, "exact", None, None, {})


def test_setup_is_upstreams_setup_controller_at_the_guests_screen_size(upstream):
    runtime.setup(GUEST, {}, [{"type": "sleep", "parameters": {"seconds": 0}}])
    [made] = _Setup.made
    assert (made["vm_ip"], made["screen_width"], made["screen_height"], made["os_type"]) == (LOOPBACK, 1280, 800,
                                                                                            "Ubuntu")
    assert made["config"] == [{"type": "sleep", "parameters": {"seconds": 0}}]
    assert upstream.modules.setup.MAX_RETRIES == runtime.SETUP_RETRIES


def test_a_setup_upstream_reports_failed_raises(upstream):
    _Setup.answer = False
    with pytest.raises(RuntimeError, match="didn't answer /terminal"):
        runtime.setup(GUEST, {}, [])


def test_a_failure_that_called_an_unexposed_port_says_which(upstream, monkeypatch):
    def get_vm_file(env, config):
        requests.get(f"http://{env.vm_ip}:{env.chromium_port}/json", timeout=5)
        raise ConnectionError("no browser")

    upstream.modules.getters.get_vm_file = get_vm_file
    with pytest.raises(RuntimeError, match="ConnectionError: no browser; it needs the computer's Chrome DevTools port"):
        runtime.evaluate(GUEST, {}, "exact", {"type": "vm_file", "path": "/a"}, None, {})


@pytest.mark.asyncio
async def test_a_cancelled_call_stops_its_forwarder_and_waits_for_upstream_to_end(upstream):
    """A thread can't be cancelled: upstream's next call after the cancel is refused, and the step's cancellation
    waits for upstream to end, so nothing it does lands on the computer after."""
    started, answers = threading.Event(), []

    class Blocking(_Setup):
        def setup(self, config):
            started.set()
            time.sleep(0.3)
            answers.append(requests.post(f"http://{self.kwargs['vm_ip']}:{self.kwargs['server_port']}/execute",
                                         json={"command": "launch"}, timeout=5).status_code)
            return True

    upstream.modules.setup.SetupController = Blocking
    call = asyncio.create_task(runtime.run_setup(GUEST, {}, []))
    await asyncio.to_thread(started.wait, 5)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert answers == [503]
