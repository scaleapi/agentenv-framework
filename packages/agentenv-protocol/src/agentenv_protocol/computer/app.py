"""ComputerAppEnvironment: software on a computer, an AgentEnvEnvironment whose class also declares what it is for a
computer: the platforms it supports and the permissions it needs (``@computer_app``), how each OS installs it
(``linux = Linux(...)``), how it launches (``launch``), its install steps and ready checks, its skill, and its typed
``Settings``. A ``computer_app`` env's put reads that declaration from the class; its data hooks are the base class's.
Its name is its env's ``environment_name``."""
from __future__ import annotations

import inspect
import json
import os
from typing import Any, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agent_env_environment import _EXT_ATTR, _OP_ATTR, AgentEnvEnvironment
from ..types import METHOD_ADD, METHOD_GET, METHOD_RESET, SKILL_EXTENSION_URI

#: The environment variable a computer delivers its apps' settings in: JSON of each app's values by its name.
APP_SETTINGS_ENV = "AGENTENV_APP_SETTINGS"
_INSTALL_STEP_ATTR = "_aee_install_step"
_READY_CHECK_ATTR = "_aee_ready_check"
_DATA_METHODS = {"reset_data": METHOD_RESET, "add_data": METHOD_ADD, "get_data": METHOD_GET}
#: What a launch's window can be matched by, as a driver's list_windows describes each window.
WINDOW_KEYS = frozenset({"class", "instance", "title"})


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

    @model_validator(mode="after")
    def _venv_for_ready_checks(self) -> AppDeclaration:
        if self.ready_checks and not any(install.python for install in self.install.values()):
            raise ValueError(f"@ready_check {', '.join(self.ready_checks)} runs in the app's own venv: name its "
                             "requirements with Linux(python=...)")
        return self


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
