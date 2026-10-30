"""ComputerAppEnvironment: software on a computer, an AgentEnvEnvironment whose class also declares what it is for a
computer: the platforms it supports and the permissions it needs (``@computer_app``), how each OS installs it
(``linux = Linux(...)``), how it launches (``launch``), its install steps and ready checks, its skill, and its typed
``Settings``. A ``computer_app`` env's put reads that declaration from the class; its data hooks are the base class's.
Its name is its env's ``environment_name``. On the computer the class runs as the app's own server (``serve_app``),
which launches the app, adds its launcher and says whether it's ready (``computer-app/v1``)."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agent_env_environment import _EXT_ATTR, _OP_ATTR, AgentEnvEnvironment, extension
from ..types import METHOD_ADD, METHOD_GET, METHOD_RESET, SKILL_EXTENSION_URI
from .app_runtime import AppRuntime
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
#: How long an app may take to be ready when its launch doesn't say.
DEFAULT_TIMEOUT_S = 60.0


class Service(BaseModel):
    """A program the computer's init runs from the app's folder as the session user; a relative program names one
    there."""

    model_config = ConfigDict(extra="forbid")

    name: str
    command: str


class Packages(BaseModel):
    """The OS's packages, each as its package manager takes it: for apt, ``name``, ``name=version`` or
    ``name/suite``."""

    model_config = ConfigDict(extra="forbid")

    apt: list[str] = Field(default_factory=list)


class AptSource(BaseModel):
    """An apt repository resolved from besides the OS's own: its URL, suite and components, and its signing key, a file
    of the app's folder."""

    model_config = ConfigDict(extra="forbid")

    apt: str
    suite: str
    components: list[str] = Field(default_factory=list)
    key: str


class Download(BaseModel):
    """A file fetched by its URL and checked against its sha256, put at ``to`` under the app's prefix; for one arch
    only, when ``arch`` names it."""

    model_config = ConfigDict(extra="forbid")

    url: str
    sha256: str
    to: str
    arch: Optional[Literal["amd64", "arm64"]] = None


class RunScript(BaseModel):
    """A script of the app's folder run as root, or for ``session``, as the session user, before its install steps."""

    model_config = ConfigDict(extra="forbid")

    script: str
    scope: Literal["system", "session"] = "system"


class Linux(BaseModel):
    """How a Linux computer installs the app: ``files``, a folder of the app's copied under its prefix; ``packages``,
    the OS's; ``python``, a requirements file in that folder installed into a venv beside them; ``sources``, extra apt
    repositories; ``downloads``, files fetched by URL; ``run``, scripts run once those are in; ``services``, programs
    its init runs; ``ports``, those it listens on. Everything is fetched once at put, and builds install only that."""

    model_config = ConfigDict(extra="forbid")

    files: str = "."
    packages: Optional[Packages] = None
    python: Optional[str] = None
    sources: Optional[list[AptSource]] = None
    downloads: Optional[list[Download]] = None
    run: Optional[list[RunScript]] = None
    services: list[Service] = Field(default_factory=list)
    ports: Optional[list[int]] = None


class Ready(BaseModel):
    """When the app is ready: its window shown, a port answering or a command exiting 0, within ``timeout_s``."""

    model_config = ConfigDict(extra="forbid")

    window: bool = False
    port: Optional[int] = None
    command: Optional[list[str]] = None
    timeout_s: float = DEFAULT_TIMEOUT_S

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
    """Mark a method a computer runs as it installs the app, after its files, packages, venv and run scripts, in the
    order the class defines them: as root, or for ``session``, as the session user. It runs offline, at build, without
    the class's ``__init__``: ``self.settings`` holds only the Settings' defaults, since no deploy's are known yet."""
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
    _computer_app: Optional[AppRuntime] = None
    linux: ClassVar[Optional[Linux]] = None
    launch: ClassVar[Optional[Launch]] = None
    #: A file of the app's folder holding its skill for agents, when no ``@agent_skill`` method writes it.
    skill: ClassVar[Optional[str]] = None

    class Settings(BaseModel):
        """The app's settings: an app declares its own; a ``SecretStr`` field is a secret."""

    def __init__(self) -> None:
        values = json.loads(os.environ.get(APP_SETTINGS_ENV) or "{}").get(os.environ.get("ENVIRONMENT_NAME"), {})
        self.settings = self.Settings.model_validate(values)
        ready = self.launch.ready if self.launch is not None else None
        self._computer_app = AppRuntime(
            self.launch, {fn.__name__: getattr(self, fn.__name__) for fn in _methods(type(self))
                          if hasattr(fn, _READY_CHECK_ATTR)},
            ready.timeout_s if ready is not None else DEFAULT_TIMEOUT_S)

    @extension(COMPUTER_APP, op="launch", description=(
        "The app's own server: launch launches the app from its folder, as its launcher and a boot do; ready says "
        "whether it's ready, by its launch's ready and its @ready_check methods."))
    async def _computer_app_launch(self) -> dict:
        if self.launch is None:
            raise InvalidArgument(f"{type(self).__name__} declares no launch")
        return {"pid": await self._computer_app_runtime().launch_app()}

    @extension(COMPUTER_APP, op="ready", method="GET")
    async def _computer_app_ready(self) -> dict:
        return (await self._computer_app_runtime().readiness()).model_dump()

    def _computer_app_runtime(self) -> AppRuntime:
        if self._computer_app is None:
            raise TypeError(f"{type(self).__qualname__}.__init__ must call super().__init__()")
        return self._computer_app

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


def _methods(cls: type) -> list[Callable]:
    """The class's methods, each as the class resolves its name, in the order they were first defined."""
    names = dict.fromkeys(name for klass in reversed(cls.__mro__) for name, member in vars(klass).items()
                          if inspect.isfunction(member))
    return [method for name in names if inspect.isfunction(method := getattr(cls, name))]

