"""Upstream OSWorld's setup and checks, run unmodified in the worker against a computer's guest server through loopback
forwarders (:mod:`.forwarder`): ``SetupController.setup`` for a task's config, and for one func the single-metric branch
of ``DesktopEnv.evaluate``, whose result and expected getters take the env ``GuestEnv`` stands in for, with exactly the
attributes upstream's getters read. ``DesktopEnv`` itself isn't imported: it builds a VM provider and needs gymnasium.

Every call here blocks, as upstream does, and is made off the event loop (``asyncio.to_thread``); each builds its own
controllers, forwarders and cache folder, so calls for different computers run side by side. A thread can't be
cancelled, so a cancelled call stops its forwarder, which refuses upstream's next call, and waits for the thread to end
before it is cancelled itself: nothing a setup does lands on the computer after its step is."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from agent_env.task_step.task_steps.computer.osworld import upstream
from agent_env.task_step.task_steps.computer.osworld.forwarder import CHROMIUM, LOOPBACK, VLC, Guest

logger = logging.getLogger(__name__)

#: How often upstream's setup polls the guest server's /terminal before giving up, 5 s apart; its own 20 holds a call
#: on a dead guest for minutes past what a step waits for.
SETUP_RETRIES = 3
#: The screen a task's ``{SCREEN_WIDTH}``-style placeholders read when the guest won't say its own.
DEFAULT_SCREEN = (1920, 1080)
OS_TYPE = "Ubuntu"


class UnknownCheck(LookupError):
    """A func or getter upstream OSWorld doesn't have."""


class GuestEnv:
    """What upstream's getters read of a ``DesktopEnv``: the guest's address and ports, a cache folder, its controllers
    and, through them, its platform."""

    def __init__(self, modules: upstream.Upstream, guest: Guest, *, cache_dir: str, os_type: str = OS_TYPE,
                 client_password: str = ""):
        self.vm_ip, self.server_port = LOOPBACK, guest.port
        self.chromium_port, self.vlc_port = guest.ports[CHROMIUM], guest.ports[VLC]
        self.cache_dir = cache_dir
        self.os_type = os_type
        self.client_password = client_password
        self.current_use_proxy = False
        self.is_environment_used = True
        self.controller = modules.python.PythonController(vm_ip=self.vm_ip, server_port=self.server_port)

    @property
    def vm_platform(self) -> str:
        return self.controller.get_vm_platform()

    @property
    def vm_machine(self) -> str:
        return self.controller.get_vm_machine()


def _session(guest_url: str, headers: dict[str, str], call: Callable[[upstream.Upstream, Guest, GuestEnv], Any],
             stop: threading.Event | None) -> Any:
    """``call`` with upstream's modules, the guest on loopback and an env over it, its cache folder removed after; an
    error that called a port the app doesn't expose says which."""
    modules = upstream.load()
    modules.setup.MAX_RETRIES = SETUP_RETRIES
    with tempfile.TemporaryDirectory(prefix="osworld-") as cache, Guest(guest_url, headers, stop) as guest:
        env = GuestEnv(modules, guest, cache_dir=cache)
        try:
            return call(modules, guest, env)
        except UnknownCheck:
            raise
        except Exception as e:
            if why := guest.needed_since(0):
                raise RuntimeError(f"{type(e).__name__}: {e}; {why}") from e
            raise


def setup(guest_url: str, headers: dict[str, str], config: list[dict], stop: threading.Event | None = None) -> None:
    """``config``, OSWorld's ``[{type, parameters}]``, run on the guest by upstream's ``SetupController``; raises with
    what failed."""

    def run(modules: upstream.Upstream, guest: Guest, env: GuestEnv) -> None:
        try:
            size = env.controller.get_vm_screen_size() or {}
            width, height = int(size["width"]), int(size["height"])
        except Exception as e:
            logger.warning(f"the guest's screen size is unknown ({type(e).__name__}); using {DEFAULT_SCREEN}")
            width, height = DEFAULT_SCREEN
        controller = modules.setup.SetupController(
            vm_ip=env.vm_ip, server_port=env.server_port, chromium_port=env.chromium_port, vlc_port=env.vlc_port,
            cache_dir=env.cache_dir, client_password=env.client_password, screen_width=width, screen_height=height,
            os_type=env.os_type)
        if controller.setup(config) is False:
            raise RuntimeError(f"the guest server didn't answer /terminal after {SETUP_RETRIES} tries")

    _session(guest_url, headers, run, stop)


def _named(module: Any, name: Any, what: str) -> Any:
    found = getattr(module, name, None) if isinstance(name, str) and not name.startswith("_") else None
    if not callable(found):
        raise UnknownCheck(f"upstream OSWorld has no {what} {name!r}")
    return found


def evaluate(guest_url: str, headers: dict[str, str], func: str, result: Any, expected: Any, options: dict,
             stop: threading.Event | None = None) -> float:
    """The score upstream's metric ``func`` gives what its ``result`` getter reads off the guest, against what its
    ``expected`` getter reads when it has one, with ``options``. A result file the guest doesn't have scores 0. Raises
    UnknownCheck for a func or getter upstream doesn't have."""

    def run(modules: upstream.Upstream, guest: Guest, env: GuestEnv) -> float:
        metric = _named(modules.metrics, func, "metric")
        if not isinstance(result, dict):
            raise ValueError(f"{func}: the evaluator gives no result getter")
        result_getter = _named(modules.getters, f"get_{result.get('type')}", "getter")
        expected_getter = (_named(modules.getters, f"get_{expected.get('type')}", "getter")
                           if isinstance(expected, dict) and expected else None)
        try:
            got = result_getter(env, result)
        except FileNotFoundError:
            logger.warning(f"{func}: the result file wasn't found; it scores 0")
            return 0.0
        if expected_getter is not None:
            return float(metric(got, expected_getter(env, expected), **options))
        return float(metric(got, **options))

    return _session(guest_url, headers, run, stop)


async def _in_thread(call: Callable[..., Any], *args: Any) -> Any:
    """``call(*args, stop)`` in a thread. Cancelled, it sets ``stop`` and waits for the thread to end, however often
    it's cancelled meanwhile, before it raises CancelledError."""
    stop = threading.Event()
    done = asyncio.ensure_future(asyncio.to_thread(call, *args, stop))
    try:
        return await asyncio.shield(done)
    except asyncio.CancelledError:
        stop.set()
        while not done.done():
            try:
                await asyncio.wait({done})
            except asyncio.CancelledError:
                pass
        if not done.cancelled() and done.exception() is not None:
            logger.info(f"a cancelled OSWorld call ended with {type(done.exception()).__name__}")
        raise


async def run_setup(guest_url: str, headers: dict[str, str], config: list[dict]) -> None:
    await _in_thread(setup, guest_url, headers, config)


async def run_evaluate(guest_url: str, headers: dict[str, str], func: str, result: Any, expected: Any,
                       options: dict) -> float:
    return await _in_thread(evaluate, guest_url, headers, func, result, expected, options)
