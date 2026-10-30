"""ComputerAppEnvironment: software on a computer, an AgentEnvEnvironment whose class also declares what it is for a
computer: the platforms it supports and the permissions it needs (``@computer_app``); for each OS, how it installs and
launches there and what differs there (``@linux``, or ``@install`` for any OS type); its install steps and ready checks,
each on every OS or on those it names; its skill, its typed ``Settings`` and its ``BuildSettings``. A ``computer_app``
env's put reads that declaration from the class; its data hooks are the base class's. Its name is its env's
``environment_name``. On the computer the class runs as the app's own server (``serve_app``), which launches the app,
adds its launcher and says whether it's ready (``computer-app/v1``), as declared for the OS it runs on."""
from __future__ import annotations

import functools
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Callable, ClassVar, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from ..agent_env_environment import _EXT_ATTR, _OP_ATTR, AgentEnvEnvironment, extension
from ..types import METHOD_ADD, METHOD_GET, METHOD_RESET, SKILL_EXTENSION_URI
from .app_runtime import AppRuntime
from .errors import InvalidArgument
from .platform import detect

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
#: The values of an app's build settings its env fixes in its computer's image, in the app's folder.
BUILD_SETTINGS = Path(".agentenv/build_settings.json")
# The JSON schema keywords that hold a subschema, a list of them, or a map of them by name.
_SUBSCHEMA = ("items", "additionalProperties", "not")
_SUBSCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")
_SUBSCHEMA_MAPS = ("properties", "$defs", "patternProperties")
#: How long an app may take to be ready when its launch doesn't say.
DEFAULT_TIMEOUT_S = 60.0


def _inside(path: str) -> str:
    if not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError(f"{path!r} must be a path inside the layer's folder")
    return path


#: A path relative to the folder an app or a driver installs from, which never leaves it.
LayerPath = Annotated[str, AfterValidator(_inside)]


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

    @field_validator("apt")
    @classmethod
    def _one_word_each(cls, value: list[str]) -> list[str]:
        if bad := [name for name in value if not name or name.split() != [name] or name.startswith("-")]:
            raise ValueError(f"each package is one word, as apt takes it, not {bad}")
        return value


class AptSource(BaseModel):
    """An apt repository resolved from besides the OS's own: its URL, suite and components, and its signing key, a file
    of the folder it installs from."""

    model_config = ConfigDict(extra="forbid")

    apt: str = Field(pattern=r"^https?://")
    suite: str = Field(min_length=1)
    components: list[str] = Field(default_factory=list)
    key: LayerPath


class Download(BaseModel):
    """A file fetched by its URL and checked against its sha256, put at ``to`` under the prefix it installs to; for one
    arch only, when ``arch`` names it."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(pattern=r"^https?://")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    to: LayerPath
    arch: Optional[Literal["amd64", "arm64"]] = None


class RunScript(BaseModel):
    """A script of the folder it installs from, run as root, or for ``session``, as the session user, once its files,
    packages and venv are in."""

    model_config = ConfigDict(extra="forbid")

    script: LayerPath
    scope: Literal["system", "session"] = "system"


class Linux(BaseModel):
    """How a Linux computer installs the app: ``files``, a folder of the app's copied under its prefix; ``packages``,
    the OS's; ``python``, a requirements file in that folder installed into a venv beside them; ``sources``, extra apt
    repositories; ``downloads``, files fetched by URL; ``run``, scripts run once those are in; ``services``, programs
    its init runs; ``ports``, those it listens on. Everything is fetched once at put, and builds install only that."""

    model_config = ConfigDict(extra="forbid")

    files: LayerPath = "."
    packages: Optional[Packages] = None
    python: Optional[LayerPath] = None
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
    """An ``@install_step`` method, by name, whom it runs as, and the OS types it runs on: every one when ``os`` is
    None."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scope: Literal["system", "session"]
    os: Optional[list[str]] = None


class AppReadyCheck(BaseModel):
    """A ``@ready_check`` method, by name, and the OS types it runs on: every one when ``os`` is None."""

    model_config = ConfigDict(extra="forbid")

    name: str
    os: Optional[list[str]] = None


class AppOs(BaseModel):
    """What an app declares for one OS type: ``install``, how that OS installs it, as the OS type's install model gives
    it; its ``launch`` there; and its ``skill``, ``settings`` and ``build_settings`` there when they differ from the
    class's, the schemas with the class's fields added."""

    model_config = ConfigDict(extra="forbid")

    install: dict[str, Any]
    launch: Optional[Launch] = None
    skill: Optional[AppSkill] = None
    settings: Optional[dict[str, Any]] = None
    build_settings: Optional[dict[str, Any]] = None


class AppDeclaration(BaseModel):
    """What a ComputerAppEnvironment class declares, as a ``computer_app`` env stores it: the platforms it ``supports``,
    as a driver names them; ``os``, what it declares for each OS type; ``settings`` and ``build_settings``, the JSON
    schemas of its ``Settings`` and of its ``BuildSettings`` when it declares any; its ``skill``; and the ``data``
    methods, ``install_steps`` and ``ready_checks`` it serves, by name."""

    model_config = ConfigDict(extra="forbid")

    supports: list[dict[str, Any]]
    permissions: list[str]
    os: dict[str, AppOs]
    settings: dict[str, Any]
    build_settings: Optional[dict[str, Any]] = None
    skill: Optional[AppSkill] = None
    data: list[str]
    install_steps: list[InstallStep]
    ready_checks: list[AppReadyCheck]

    @model_validator(mode="before")
    @classmethod
    def _stored_before_each_os(cls, data: Any) -> Any:
        """A declaration stored before each OS had its own part: its ``install`` by OS type and one ``launch`` for
        them, which become ``os``, and its ready checks by name."""
        if not isinstance(data, dict) or "install" not in data:
            return data
        data = dict(data)
        installs, launch = data.pop("install"), data.pop("launch", None)
        data["os"] = {os_type: {"install": install, "launch": launch} for os_type, install in installs.items()}
        data["ready_checks"] = [{"name": check} if isinstance(check, str) else check
                                for check in data.get("ready_checks", [])]
        return data


@dataclass(frozen=True)
class _OsDeclared:
    """What ``@install`` declares for one OS type, as the class keeps it."""

    install: BaseModel
    launch: Optional[Launch]
    skill: Optional[str]
    settings: Optional[type[BaseModel]]
    build_settings: Optional[type[BaseModel]]


def computer_app(*, supports: Optional[list] = None, permissions: Optional[list[str]] = None) -> Callable:
    """Declare what an app is: the platforms it supports, as a driver names them, and the permissions it needs of the
    computer. Its env.toml names it and the computer and driver it deploys alone on."""
    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, ComputerAppEnvironment)):
            raise TypeError("@computer_app decorates a ComputerAppEnvironment subclass")
        cls.supports, cls.permissions = list(supports or []), list(permissions or [])
        return cls
    return deco


def install(os_type: str, spec: BaseModel, *, launch: Optional[Launch] = None, skill: Optional[str] = None,
            settings: Optional[type[BaseModel]] = None,
            build_settings: Optional[type[BaseModel]] = None) -> Callable:
    """Declare how a computer of ``os_type`` installs the app, ``spec`` being that OS type's install model, and what is
    the app's own there: its ``launch``; its ``skill``, a file of its folder, instead of the class's; and ``settings``
    and ``build_settings``, models whose fields add to the class's ``Settings`` and ``BuildSettings`` there. ``@linux``
    is this for Linux; an OS type a plugin adds is declared with it."""
    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, ComputerAppEnvironment)):
            raise TypeError("@install decorates a ComputerAppEnvironment subclass")
        cls._installs = {**cls._installs, os_type: _OsDeclared(spec, launch, skill, settings, build_settings)}
        return cls
    return deco


def linux(*, files: str = ".", packages: Optional[Packages] = None, python: Optional[str] = None,
          sources: Optional[list[AptSource]] = None, downloads: Optional[list[Download]] = None,
          run: Optional[list[RunScript]] = None, services: Optional[list[Service]] = None,
          ports: Optional[list[int]] = None, launch: Optional[Launch] = None, skill: Optional[str] = None,
          settings: Optional[type[BaseModel]] = None, build_settings: Optional[type[BaseModel]] = None) -> Callable:
    """``@install`` for Linux: how a Linux computer installs the app, ``Linux``'s fields, and what is its own there."""
    return install("linux", Linux(files=files, packages=packages, python=python, sources=sources, downloads=downloads,
                                  run=run, services=services or [], ports=ports),
                   launch=launch, skill=skill, settings=settings, build_settings=build_settings)


def install_step(scope: Literal["system", "session"] = "system", *,
                 os: Union[str, list[str], None] = None) -> Callable:
    """Mark a method a computer runs as it installs the app, after its files, packages, venv and run scripts, in the
    order the class defines them: as root, or for ``session``, as the session user; on every OS the app installs on,
    or on the ``os`` types named. It runs offline, at build, without the class's ``__init__``, on
    ``self.build_settings``; a deploy's ``self.settings`` aren't known yet."""
    def deco(fn: Callable) -> Callable:
        setattr(fn, _INSTALL_STEP_ATTR, (scope, _named(os)))
        return fn
    return deco


def ready_check(fn: Optional[Callable] = None, *, os: Union[str, list[str], None] = None) -> Callable:
    """Mark a method that says whether the app is ready, when its launch's ``ready`` can't: on every OS the app
    installs on, or on the ``os`` types named; ``@ready_check`` or ``@ready_check(os=...)``."""
    def mark(fn: Callable) -> Callable:
        setattr(fn, _READY_CHECK_ATTR, _named(os))
        return fn
    return mark(fn) if fn is not None else mark


def _named(os_types: Union[str, list[str], None]) -> Optional[list[str]]:
    return None if os_types is None else [os_types] if isinstance(os_types, str) else list(os_types)


class ComputerAppEnvironment(AgentEnvEnvironment):
    """Subclass it, declare it with ``@computer_app`` and an OS decorator for each OS (``@linux``), and set ``skill`` as
    a class attribute; a deploy's settings arrive as ``self.settings``, those its env fixes at build as
    ``self.build_settings``, and its launch as ``self.launch``, each as declared for the OS it runs on."""

    supports: ClassVar[list] = []
    permissions: ClassVar[list[str]] = []
    _installs: ClassVar[dict[str, _OsDeclared]] = {}
    _computer_app: Optional[AppRuntime] = None
    launch: Optional[Launch] = None
    #: A file of the app's folder holding its skill for agents, when no ``@agent_skill`` method writes it.
    skill: ClassVar[Optional[str]] = None

    class Settings(BaseModel):
        """The app's settings: an app declares its own; a ``SecretStr`` field is a secret."""

    class BuildSettings(BaseModel):
        """The app's build settings, which its env fixes in its computer's image and only another image changes: an
        app declares its own, which its install steps read too. None is a secret, since the image keeps them."""

    def __init__(self) -> None:
        cls, on = type(self), detect().os
        values = json.loads(os.environ.get(APP_SETTINGS_ENV) or "{}").get(os.environ.get("ENVIRONMENT_NAME"), {})
        self.settings = cls.settings_on(on).model_validate(values)
        self.build_settings = _built_settings(cls, on)
        self.launch = declared.launch if (declared := cls._installs.get(on)) is not None else None
        ready = self.launch.ready if self.launch is not None else None
        self._computer_app = AppRuntime(
            self.launch, {fn.__name__: getattr(self, fn.__name__) for fn in _methods(cls)
                          if hasattr(fn, _READY_CHECK_ATTR) and _runs_on(getattr(fn, _READY_CHECK_ATTR), on)},
            ready.timeout_s if ready is not None else DEFAULT_TIMEOUT_S)

    @classmethod
    def settings_on(cls, os_type: Optional[str]) -> type[BaseModel]:
        """Its ``Settings`` on ``os_type``, with the fields its OS adds."""
        added = declared.settings if (declared := cls._installs.get(os_type)) is not None else None
        return _added(added, cls.Settings) if added is not None else cls.Settings

    @classmethod
    def build_settings_on(cls, os_type: Optional[str]) -> type[BaseModel]:
        """Its ``BuildSettings`` on ``os_type``, with the fields its OS adds."""
        added = declared.build_settings if (declared := cls._installs.get(os_type)) is not None else None
        return _added(added, cls.BuildSettings) if added is not None else cls.BuildSettings

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
        """What the class declares. Raises ValueError for a skill declared twice, a build setting that's a secret, or
        an install step or ready check on an OS type the class declares no install for."""
        build = cls.BuildSettings.model_json_schema()
        for name, schema in [(f"{cls.__name__}.BuildSettings", build),
                             *((declared.build_settings.__name__, declared.build_settings.model_json_schema())
                               for declared in cls._installs.values() if declared.build_settings is not None)]:
            if (secret := _secret_at(schema)) is not None:
                raise ValueError(f"{name} holds a secret at {secret}, which its computer's image would keep; declare "
                                 "it in Settings")
        methods = _methods(cls)
        skill_methods = [fn.__name__ for fn in methods
                         if getattr(getattr(fn, _EXT_ATTR, None), "uri", None) == SKILL_EXTENSION_URI]
        if cls.skill is not None and skill_methods:
            raise ValueError(f"{cls.__name__} declares its skill twice: skill = {cls.skill!r} and @agent_skill "
                             f"{skill_methods[0]}")
        if skill_methods and (os_type := next((os_type for os_type, declared in cls._installs.items()
                                               if declared.skill is not None), None)):
            raise ValueError(f"{cls.__name__} declares its skill twice on {os_type}: skill = "
                             f"{cls._installs[os_type].skill!r} there and @agent_skill {skill_methods[0]}, which its "
                             "server serves on every OS")
        steps = [InstallStep(name=fn.__name__, scope=scope, os=os_types) for fn in methods
                 if hasattr(fn, _INSTALL_STEP_ATTR) for scope, os_types in [getattr(fn, _INSTALL_STEP_ATTR)]]
        checks = [AppReadyCheck(name=fn.__name__, os=getattr(fn, _READY_CHECK_ATTR)) for fn in methods
                  if hasattr(fn, _READY_CHECK_ATTR)]
        for method in [*steps, *checks]:
            if unknown := [os_type for os_type in method.os or [] if os_type not in cls._installs]:
                raise ValueError(f"{cls.__name__}.{method.name} runs on {unknown[0]!r}, which {cls.__name__} declares "
                                 f"no install for; it declares {', '.join(cls._installs) or 'none'}")
        return AppDeclaration(
            supports=cls.supports, permissions=cls.permissions,
            os={os_type: AppOs(
                install=declared.install.model_dump(mode="json", exclude_none=True), launch=declared.launch,
                skill=AppSkill(file=declared.skill) if declared.skill is not None else None,
                settings=cls.settings_on(os_type).model_json_schema() if declared.settings is not None else None,
                build_settings=(cls.build_settings_on(os_type).model_json_schema()
                                if declared.build_settings is not None else None))
                for os_type, declared in cls._installs.items()},
            settings=cls.Settings.model_json_schema(), build_settings=build if build.get("properties") else None,
            skill=AppSkill(file=cls.skill) if cls.skill is not None else
                  AppSkill(method=skill_methods[0]) if skill_methods else None,
            data=sorted({_DATA_METHODS[op] for fn in methods if (op := getattr(fn, _OP_ATTR, None)) in _DATA_METHODS}),
            install_steps=steps, ready_checks=checks,
        )


def _secret_at(schema: dict, path: str = "") -> Optional[str]:
    """Where in ``schema``, a JSON schema, a secret is, a ``SecretStr``'s ``writeOnly``: in it or any of its subschemas,
    a field's, a list's items or a nested model's definition; None when it holds none. A default or a field's name
    is data, never a marker."""
    if schema.get("writeOnly") is True:
        return path
    subschemas = [*((_joined(path, key), schema[key]) for key in _SUBSCHEMA if key in schema),
                  *((_joined(path, f"{key}[{index}]"), sub) for key in _SUBSCHEMA_LISTS
                    for index, sub in enumerate(schema.get(key, []))),
                  *((_joined(path, f"{key}.{name}"), sub) for key in _SUBSCHEMA_MAPS
                    for name, sub in schema.get(key, {}).items())]
    return next((at for at in (_secret_at(sub, at) for at, sub in subschemas if isinstance(sub, dict))
                 if at is not None), None)


def _joined(path: str, part: str) -> str:
    return f"{path}.{part}" if path else part


def _built_settings(cls: type[ComputerAppEnvironment], os_type: Optional[str]) -> BaseModel:
    """The app's build settings on ``os_type``, as its env fixed them in its computer's image, read in the app's folder:
    the values the env gives, over the defaults."""
    values = json.loads(BUILD_SETTINGS.read_text()) if BUILD_SETTINGS.is_file() else {}
    return cls.build_settings_on(os_type).model_validate(values)


def _runs_on(os_types: Optional[list[str]], os_type: Optional[str]) -> bool:
    return os_types is None or os_type in os_types


@functools.cache
def _added(added: type[BaseModel], base: type[BaseModel]) -> type[BaseModel]:
    """``base`` with the fields of ``added``, whose win where both declare one."""
    return create_model(added.__name__, __base__=(added, base))


def _methods(cls: type) -> list[Callable]:
    """The class's methods, each as the class resolves its name, in the order they were first defined."""
    names = dict.fromkeys(name for klass in reversed(cls.__mro__) for name, member in vars(klass).items()
                          if inspect.isfunction(member))
    return [method for name in names if inspect.isfunction(method := getattr(cls, name))]

