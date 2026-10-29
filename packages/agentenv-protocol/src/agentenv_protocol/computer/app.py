"""ComputerAppEnvironment: software on a computer, an AgentEnvEnvironment whose class also declares what it is for a
computer: the platforms it supports and the permissions it needs (``@computer_app``), how each OS installs it
(``linux = Linux(...)``), how it launches (``launch``), its install steps and ready checks, its skill, and its typed
``Settings``. A ``computer_app`` env's put reads that declaration from the class; its data hooks are the base class's.
Its name is its env's ``environment_name``. On the computer the class runs as the app's own server (``serve_app``),
which launches the app, adds its launcher and says whether it's ready (``computer-app/v1``)."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agent_env_environment import _EXT_ATTR, _OP_ATTR, AgentEnvEnvironment, extension
from ..types import METHOD_ADD, METHOD_GET, METHOD_RESET, SKILL_EXTENSION_URI
from . import session
from .errors import InvalidArgument

#: The environment variable a computer delivers its apps' settings in: JSON of each app's values by its name.
APP_SETTINGS_ENV = "AGENTENV_APP_SETTINGS"
_INSTALL_STEP_ATTR = "_aee_install_step"
_READY_CHECK_ATTR = "_aee_ready_check"
_DATA_METHODS = {"reset_data": METHOD_RESET, "add_data": METHOD_ADD, "get_data": METHOD_GET}
#: What a launch's window can be matched by, as the session lists each window.
WINDOW_KEYS = frozenset({"class", "instance", "title"})
#: The framework extension an app's own server serves: its launch as params, and its ``launch`` and ``ready`` methods.
COMPUTER_APP = "urn:agentenv:computer-app/v1"
#: Where an app's own server listens, in the app's folder, and where it says why it last failed to start.
SOCKET = Path(".agentenv/server.sock")
FAILURE = Path(".agentenv/server.failure")
#: Where an app's server marks that it launched the app since the computer booted, at boot or on demand.
LAUNCHED = Path("/run/agentenv/apps")
#: How long one readiness check may take; an app's checks run at once.
CHECK_TIMEOUT_S = 10.0


class Service(BaseModel):
    """A program the computer's init runs from the app's folder as the session user; a relative program names one
    there."""

    model_config = ConfigDict(extra="forbid")

    name: str
    command: str


class Linux(BaseModel):
    """How a Linux computer installs the app: ``files``, a folder of the app's copied under its prefix; ``python``, a
    requirements file in that folder installed into a venv beside them; ``services``, programs its init runs."""

    model_config = ConfigDict(extra="forbid")

    files: str = "."
    python: Optional[str] = None
    services: list[Service] = Field(default_factory=list)


class Ready(BaseModel):
    """When the app is ready: its window shown, a port answering or a command exiting 0, within ``timeout_s``."""

    model_config = ConfigDict(extra="forbid")

    window: bool = False
    port: Optional[int] = None
    command: Optional[list[str]] = None
    timeout_s: float = 60

    @model_validator(mode="after")
    def _waits_on_something(self) -> Ready:
        if not self.window and self.port is None and not self.command:
            raise ValueError("Ready waits on nothing: set window, port or command")
        return self


class Launch(BaseModel):
    """How the app starts in the desktop session: with it (``boot``) or when the agent opens it (``demand``)."""

    model_config = ConfigDict(extra="forbid")

    at: Literal["boot", "demand"] = "boot"
    command: list[str]
    #: Matches the app's window by its WM_CLASS ``class`` or ``instance`` and its ``title``, as given.
    window: Optional[dict[str, str]] = None
    ready: Optional[Ready] = None

    @field_validator("window")
    @classmethod
    def _matchable(cls, window: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        if window is not None and (not window or set(window) - WINDOW_KEYS):
            raise ValueError(f"window matches by {', '.join(sorted(WINDOW_KEYS))}, not {window!r}")
        return window

    @model_validator(mode="after")
    def _window_to_wait_for(self) -> Launch:
        if self.ready is not None and self.ready.window and self.window is None:
            raise ValueError("ready waits on a window: name it with Launch(window={'class': ...})")
        return self


class AppSkill(BaseModel):
    """Where an app's skill for agents comes from: a ``file`` of its folder, or the ``@agent_skill`` ``method`` that
    writes it."""

    model_config = ConfigDict(extra="forbid")

    file: Optional[str] = None
    method: Optional[str] = None


class InstallStep(BaseModel):
    """An ``@install_step`` method, by name, and whom it runs as."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scope: Literal["system", "session"]


class AppDeclaration(BaseModel):
    """What a ComputerAppEnvironment class declares, as a ``computer_app`` env stores it: the platforms it ``supports``,
    as a driver names them; its ``install`` by OS type; ``settings``, the JSON schema of its ``Settings``; and the
    ``data`` methods, ``install_steps`` and ``ready_checks`` it serves, by name."""

    model_config = ConfigDict(extra="forbid")

    supports: list[dict[str, Any]]
    permissions: list[str]
    install: dict[str, Linux]
    launch: Optional[Launch] = None
    settings: dict[str, Any]
    skill: Optional[AppSkill] = None
    data: list[str]
    install_steps: list[InstallStep]
    ready_checks: list[str]


def computer_app(*, supports: Optional[list] = None, permissions: Optional[list[str]] = None) -> Callable:
    """Declare what an app is: the platforms it supports, as a driver names them, and the permissions it needs of the
    computer. Its env.toml names it and the computer and driver it deploys alone on."""
    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, ComputerAppEnvironment)):
            raise TypeError("@computer_app decorates a ComputerAppEnvironment subclass")
        cls.supports, cls.permissions = list(supports or []), list(permissions or [])
        return cls
    return deco


def install_step(scope: Literal["system", "session"] = "system") -> Callable:
    """Mark a method a computer runs as it installs the app, after its files, in the order the class defines them: as
    root, or for ``session``, as the session user."""
    def deco(fn: Callable) -> Callable:
        setattr(fn, _INSTALL_STEP_ATTR, scope)
        return fn
    return deco


def ready_check(fn: Callable) -> Callable:
    """Mark a method that says whether the app is ready, when ``launch``'s ``ready`` can't."""
    setattr(fn, _READY_CHECK_ATTR, True)
    return fn


class ComputerAppEnvironment(AgentEnvEnvironment):
    """Subclass it, declare it with ``@computer_app``, and set ``linux``, ``launch`` and ``skill`` as class attributes;
    a deploy's settings arrive as ``self.settings``."""

    supports: ClassVar[list] = []
    permissions: ClassVar[list[str]] = []
    linux: ClassVar[Optional[Linux]] = None
    launch: ClassVar[Optional[Launch]] = None
    #: A file of the app's folder holding its skill for agents, when no ``@agent_skill`` method writes it.
    skill: ClassVar[Optional[str]] = None

    class Settings(BaseModel):
        """The app's settings: an app declares its own; a ``SecretStr`` field is a secret."""

    def __init__(self) -> None:
        values = json.loads(os.environ.get(APP_SETTINGS_ENV) or "{}").get(os.environ.get("ENVIRONMENT_NAME"), {})
        self.settings = self.Settings.model_validate(values)
        self._computer_app_started_at = time.monotonic()
        self._computer_app_launched_at: Optional[float] = None
        self._computer_app_launch_failed: Optional[str] = None
        #: Each sync ``@ready_check`` still running in its thread, by name, which a later poll waits on.
        self._computer_app_checking: dict[str, asyncio.Future] = {}

    async def _computer_app_start(self) -> None:
        """Once the desktop session is up, add the app's menu launcher and launch a boot app, once per boot: a server
        that restarts finds the app launched, at boot or on demand, and leaves it be."""
        if self.launch is None:
            return
        await session.wait_until_up()
        session.add_launcher(os.environ["ENVIRONMENT_NAME"], Path.cwd())
        if _launched().exists():
            self._computer_app_launched_at = time.monotonic()
        elif self.launch.at == "boot":
            try:
                await self._computer_app_launch()
            except Exception as e:
                self._computer_app_launch_failed = f"{type(e).__name__}: {e}"

    @extension(COMPUTER_APP, op="launch", description=(
        "The app's own server: launch launches the app from its folder, as its launcher and a boot do; ready says "
        "whether it's ready, by its launch's ready and its @ready_check methods."))
    async def _computer_app_launch(self) -> dict:
        if self.launch is None:
            raise InvalidArgument(f"{type(self).__name__} declares no launch")
        pid = await session.launch(self.launch.command, Path.cwd(),
                                   {"ENVIRONMENT_NAME": os.environ["ENVIRONMENT_NAME"]})
        self._computer_app_launched_at, self._computer_app_launch_failed = time.monotonic(), None
        _launched().parent.mkdir(parents=True, exist_ok=True)
        _launched().touch()
        return {"pid": pid}

    @extension(COMPUTER_APP, op="ready", method="GET")
    async def _computer_app_ready(self) -> dict:
        ready = self.launch.ready if self.launch is not None else None
        checks: list[tuple[str, Callable[[], Awaitable[Optional[str]]]]] = []
        if ready is not None and ready.window:
            checks.append(("window", lambda: _window(self.launch.window)))
        if ready is not None and ready.port is not None:
            checks.append(("port", lambda: _port(ready.port)))
        if ready is not None and ready.command:
            checks.append(("command", lambda: _command(ready.command)))
        checks += [(f"@ready_check {fn.__name__}", lambda fn=fn: _ready_check(fn, self))
                   for fn in _methods(type(self)) if hasattr(fn, _READY_CHECK_ATTR)]
        found = await asyncio.gather(*(_bounded(name, check) for name, check in checks))
        problems = [problem for problem in found if problem is not None]
        if self.launch is not None and self._computer_app_launched_at is None:
            problems.insert(0, f"its launch failed: {self._computer_app_launch_failed}"
                            if self._computer_app_launch_failed else "not launched yet")
        since = self._computer_app_launched_at or self._computer_app_started_at
        return {"ready": not problems, "problems": problems, "launched": self._computer_app_launched_at is not None,
                "waited_s": round(time.monotonic() - since, 1),
                "timeout_s": ready.timeout_s if ready is not None else Ready.model_fields["timeout_s"].default}

    def extension_params(self, uri: str) -> dict:
        if uri == COMPUTER_APP:
            return {"launch": self.launch.model_dump(mode="json", exclude_none=True) if self.launch else None}
        return super().extension_params(uri)

    @classmethod
    def declaration(cls) -> AppDeclaration:
        """What the class declares. Raises ValueError for a skill declared twice or an install that isn't a
        ``Linux``."""
        if cls.linux is not None and not isinstance(cls.linux, Linux):
            raise ValueError(f"{cls.__name__}.linux is a Linux(...), not {cls.linux!r}")
        methods = _methods(cls)
        skill_methods = [fn.__name__ for fn in methods
                         if getattr(getattr(fn, _EXT_ATTR, None), "uri", None) == SKILL_EXTENSION_URI]
        if cls.skill is not None and skill_methods:
            raise ValueError(f"{cls.__name__} declares its skill twice: skill = {cls.skill!r} and @agent_skill "
                             f"{skill_methods[0]}")
        return AppDeclaration(
            supports=cls.supports, permissions=cls.permissions,
            install={"linux": cls.linux} if cls.linux is not None else {},
            launch=cls.launch, settings=cls.Settings.model_json_schema(),
            skill=AppSkill(file=cls.skill) if cls.skill is not None else
                  AppSkill(method=skill_methods[0]) if skill_methods else None,
            data=sorted({_DATA_METHODS[op] for fn in methods if (op := getattr(fn, _OP_ATTR, None)) in _DATA_METHODS}),
            install_steps=[InstallStep(name=fn.__name__, scope=getattr(fn, _INSTALL_STEP_ATTR)) for fn in methods
                           if hasattr(fn, _INSTALL_STEP_ATTR)],
            ready_checks=[fn.__name__ for fn in methods if hasattr(fn, _READY_CHECK_ATTR)],
        )


def _launched() -> Path:
    return LAUNCHED / Path.cwd().name


def _methods(cls: type) -> list[Callable]:
    """The class's methods, each as the class resolves its name, in the order they were first defined."""
    names = dict.fromkeys(name for klass in reversed(cls.__mro__) for name, member in vars(klass).items()
                          if inspect.isfunction(member))
    return [method for name in names if inspect.isfunction(method := getattr(cls, name))]


async def _bounded(name: str, check: Callable[[], Awaitable[Optional[str]]]) -> Optional[str]:
    """What ``check`` finds wrong, within ``CHECK_TIMEOUT_S``; a check that runs past it or raises says so."""
    try:
        return await asyncio.wait_for(check(), CHECK_TIMEOUT_S)
    except TimeoutError:
        return f"{name} ran past {CHECK_TIMEOUT_S:g} s"
    except Exception as e:
        return f"{name}: {type(e).__name__}: {e}"


async def _window(match: dict[str, str]) -> Optional[str]:
    if any(all(window.get(key) == value for key, value in match.items()) for window in await session.windows()):
        return None
    return f"no window with {', '.join(f'{key} {value!r}' for key, value in match.items())}"


async def _port(port: int) -> Optional[str]:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return f"nothing answers on port {port}"
    writer.close()
    return None


async def _command(argv: list[str]) -> Optional[str]:
    process = await asyncio.create_subprocess_exec(*argv, cwd=Path.cwd(), stdout=asyncio.subprocess.DEVNULL,
                                                   stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        raise
    if process.returncode == 0:
        return None
    reason = (err.decode(errors="replace").strip().splitlines() or [""])[-1]
    return f"{shlex.join(argv)} exited {process.returncode}" + (f": {reason}" if reason else "")


async def _ready_check(fn: Callable, app: ComputerAppEnvironment) -> Optional[str]:
    """A sync check runs in a thread, so it can't hold up the server, and once at a time: a thread can't be stopped, so
    a poll while one runs past its bound waits on that same call."""
    if inspect.iscoroutinefunction(fn):
        passed = await fn(app)
    else:
        running = app._computer_app_checking.get(fn.__name__)
        if running is None or running.done():
            running = app._computer_app_checking[fn.__name__] = asyncio.ensure_future(asyncio.to_thread(fn, app))
            running.add_done_callback(lambda done: done.cancelled() or done.exception())
        passed = await asyncio.shield(running)
    return None if passed else f"@ready_check {fn.__name__} returned {passed!r}"
