"""What a computer's layer installs on one OS, as a driver's env.toml declares it under ``[install.<os>]`` or an app's
class with ``@linux(...)``: ``files``, a folder of its own copied under the layer's prefix; ``packages``, the OS's
packages, as apt takes them; ``python``, a requirements file in that folder installed into a venv beside them;
``sources``, extra apt repositories to resolve from; ``downloads``, files fetched by URL and checked by sha256;
``run``, scripts of the folder run once those are installed; ``services``, the programs the base's init runs from the
prefix; and ``ports``, those it listens on. An app's Linux install also takes ``shm_mb``, the shared memory it needs in
``/dev/shm``, in MiB. Everything but the files is resolved and fetched once at put, and every build installs only what
was stored then (``packages`` module). The models of packages, sources, downloads and run
scripts are the protocol's, which an app's ``@linux(...)`` uses too. ``[install.macos]`` takes the same, but apt's
packages and sources; its services are LaunchAgents of the login session, and its ``grants`` the TCC permissions an app
bundle or an executable it installs needs. ``[install.windows]`` takes the same as macOS but grants: its run scripts are
PowerShell, run elevated, and its services run windowless in the session user's sign-in. ``[install."linux/omarchy"]``
takes Linux's but apt's packages and sources, Omarchy being Arch."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Optional, TypeVar

from agentenv_protocol.computer.app import AptSource, Download, LayerPath, Packages, RunScript
from agentenv_protocol.computer.platform import entry_for, platform_key, supported
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent_env.artifact.artifacts.computer import DISTRIBUTION, OS_TYPES

#: The highest port a layer can listen on.
MAX_PORT = 65535
#: The most shared memory, in MiB, an app can ask its computer's ``/dev/shm`` for.
MAX_SHM_MB = 65536
# A requirements line that includes another requirements or constraints file, or installs one editable.
_INCLUDES = re.compile(r"-[rce]|--(requirement|constraint|editable)(=|$)")
_FIND_LINKS = re.compile(r"-f|--find-links(=|$)")


class Service(BaseModel):
    """A program the base's init runs: as the session user, or for ``system``, as root. A ``command`` whose program is a
    relative path names one under the layer's prefix."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    command: str = Field(min_length=1)
    scope: Literal["session", "system"] = "session"


class MacService(Service):
    """A program launchd runs in the login session, as a LaunchAgent: through agent-env's service runner, on the layer's
    own Python, which gives it the deploy's settings, or, when ``direct``, as its own command, with no settings. macOS
    gives a process the TCC permissions of the program launchd started, so a service an app bundle's permissions are
    for is direct, and one its layer's Python's are for runs through the runner."""

    scope: Literal["session"] = "session"
    direct: bool = False
    environment: dict[str, str] | None = Field(default=None, description="a direct service's variables")

    @model_validator(mode="after")
    def _environment_is_direct(self) -> MacService:
        if self.environment and not self.direct:
            raise ValueError("a service gets the deploy's environment, so only a direct one names its own")
        return self


class Grant(BaseModel):
    """The TCC permissions macOS gives a program the layer installs, bound to the code signature it has when it's
    composed: an ``app`` bundle, by its path on the computer, granted to the bundle's identifier; or an ``executable``,
    by its path under the layer's prefix, granted to that path. macOS charges what a process starts to the process
    launchd started, so an executable's permissions cover the programs it runs, and none it didn't."""

    model_config = ConfigDict(extra="forbid")

    app: str | None = Field(default=None, pattern=r"^/.+\.app$")
    executable: LayerPath | None = None
    services: list[Literal["accessibility", "post_event", "screen_capture"]] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_program(self) -> Grant:
        if (self.app is None) == (self.executable is None):
            raise ValueError("a grant names one program: an app bundle (app) or an executable under the prefix "
                             "(executable)")
        return self


class Install(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: LayerPath = "."
    packages: Packages | None = None
    python: LayerPath | None = None
    sources: list[AptSource] | None = None
    downloads: list[Download] | None = None
    run: list[RunScript] | None = None
    services: list[Service] = Field(min_length=1)
    ports: list[int] | None = None

    @field_validator("ports")
    @classmethod
    def _ports(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (bad := [port for port in value if not 1 <= port <= MAX_PORT]):
            raise ValueError(f"a port is 1 to {MAX_PORT}, not {bad}")
        if value is not None and len(set(value)) != len(value):
            raise ValueError(f"names a port twice: {value}")
        return value

    @model_validator(mode="after")
    def _sources_need_packages(self) -> Install:
        if self.sources and not (self.packages and self.packages.apt):
            raise ValueError("sources are apt repositories to resolve packages from, and packages names none")
        return self

    def fetches(self) -> bool:
        """Whether it has anything to resolve and fetch at put."""
        return bool((self.packages and self.packages.apt) or self.python is not None or self.downloads)

    @field_validator("services")
    @classmethod
    def _named_once(cls, value: list[Service]) -> list[Service]:
        names = [service.name for service in value]
        if len(set(names)) != len(names):
            raise ValueError(f"names a service twice: {names}")
        return value


class AppInstall(Install):
    """An app's install, which may run no service: a desktop app runs when it's launched. ``shm_mb`` is the shared
    memory it needs in ``/dev/shm``, a Linux computer's alone."""

    services: list[Service] = Field(default_factory=list)
    shm_mb: int | None = None

    @field_validator("shm_mb")
    @classmethod
    def _shm_mb(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= MAX_SHM_MB:
            raise ValueError(f"shm_mb is 1 to {MAX_SHM_MB} MiB, not {value}")
        return value


class MacInstall(Install):
    """How a macOS computer installs a layer: as on Linux, but with no apt, its services LaunchAgents and its
    ``grants`` the TCC permissions of the app bundles and executables it installs."""

    services: list[MacService] = Field(min_length=1)
    grants: list[Grant] | None = None

    @model_validator(mode="after")
    def _no_apt(self) -> MacInstall:
        if self.packages is not None or self.sources:
            raise ValueError("macOS has no apt: fetch what the layer needs with downloads, checked by sha256")
        return self


class MacAppInstall(MacInstall):
    services: list[MacService] = Field(default_factory=list)


def _without_apt(install: Install, os: str) -> None:
    if install.packages is not None or install.sources:
        raise ValueError(f"{os} has no apt: fetch what the layer needs with downloads, checked by sha256")


class OmarchyInstall(Install):
    """How an Omarchy computer installs a layer: as on Linux, but with no apt."""

    @model_validator(mode="after")
    def _no_apt(self) -> OmarchyInstall:
        _without_apt(self, "Omarchy")
        return self


class OmarchyAppInstall(AppInstall):
    @model_validator(mode="after")
    def _no_apt(self) -> OmarchyAppInstall:
        _without_apt(self, "Omarchy")
        return self


class WindowsService(Service):
    """A program agentenv-session starts, windowless, in the session user's sign-in, with the deploy's settings."""

    scope: Literal["session"] = "session"


class WindowsInstall(Install):
    """How a Windows computer installs a layer: as on Linux, but with no apt; its run scripts are PowerShell, run
    elevated over ssh as it's composed, and its services run in the session user's sign-in."""

    services: list[WindowsService] = Field(min_length=1)

    @model_validator(mode="after")
    def _windows_steps(self) -> WindowsInstall:
        if self.packages is not None or self.sources:
            raise ValueError("Windows has no apt: fetch what the layer needs with downloads, checked by sha256")
        if bad := [run.script for run in self.run or [] if not run.script.endswith(".ps1") or run.scope != "system"]:
            raise ValueError(f"a Windows layer's run scripts are PowerShell (.ps1), run elevated (scope system): {bad}")
        return self


class WindowsAppInstall(WindowsInstall):
    services: list[WindowsService] = Field(default_factory=list)


#: The install model of each OS type, or of a distribution whose differs from its OS type's, for a driver and for an
#: app.
DRIVER_INSTALLS: dict[str, type[Install]] = {"linux": Install, "linux/omarchy": OmarchyInstall, "macos": MacInstall,
                                             "windows": WindowsInstall}
APP_INSTALLS: dict[str, type[Install]] = {"linux": AppInstall, "linux/omarchy": OmarchyAppInstall,
                                          "macos": MacAppInstall, "windows": WindowsAppInstall}


_Entry = TypeVar("_Entry")


def platform_problem(key: str) -> Optional[str]:
    """What's wrong with ``key`` as what an install or a known failure is for: an OS type agent-env composes, or one
    of its distributions as ``<os>/<distribution>`` (``linux/omarchy``), whose entry replaces its OS type's there."""
    os_type, slash, distribution = key.partition("/")
    if os_type in OS_TYPES and (not slash or re.fullmatch(DISTRIBUTION, distribution)):
        return None
    return (f"an OS type agent-env composes: {', '.join(OS_TYPES)}, or one of its distributions as "
            "<os>/<distribution>, such as linux/omarchy")


def os_type_of(key: str) -> str:
    return key.partition("/")[0]


def platform_of(os: Any) -> str:
    """The key a base's OS, ``os``, takes its installs, launches and known failures by: ``<os>/<distribution>``, or its
    OS type when it names no distribution."""
    return platform_key(os.type, getattr(os, "distribution", None))


def unsupported(keys: Iterable[str], supports: list[dict], at: str) -> list[str]:
    """Each distribution of ``keys``, named from ``at``, that no ``supports`` entry names: an install or known failures
    that would never apply, as for a misspelled distribution."""
    named = {platform_key(platform.os, platform.distribution) for entry in supports for platform in supported(entry)
             if platform.distribution}
    return [f"{at}.{key}: no supports entry names {key.partition('/')[2]!r} on {os_type_of(key)}, so this never "
            "applies" for key in keys if "/" in key and key not in named]


def for_os(by_platform: dict[str, _Entry], os: Any) -> Optional[_Entry]:
    """``by_platform``'s entry for ``os``, a base's OS: its distribution's when there's one, else its OS type's."""
    return entry_for(by_platform, platform_of(os))


def installs(data: Any, models: dict[str, type[Install]] = DRIVER_INSTALLS) -> tuple[dict[str, Install], list[str]]:
    """The installs ``data``, an ``install`` table, declares by OS type or distribution (``platform_problem``), each
    its platform's model of ``models`` (``entry_for``), and the problems with it, each named by its key."""
    if not isinstance(data, dict):
        return {}, [f"install must be a table of OS types ({', '.join(OS_TYPES)}), not {data!r}"]
    found, problems = {}, []
    for key, table in data.items():
        if problem := platform_problem(key):
            problems.append(f"install.{key}: an install is for {problem}")
            continue
        try:
            found[key] = entry_for(models, key).model_validate(table)
        except ValidationError as e:
            problems.extend(f"install.{'.'.join(map(str, (key, *error['loc'])))}: {error['msg']}"
                            for error in e.errors())
    return found, problems


def folder_problems(os_type: str, install: Install, files: dict[str, Path]) -> list[str]:
    """What's wrong with ``install`` against the layer's folder, ``files`` by path: what it names that the folder
    doesn't hold (its folder, its requirements, its sources' keys and its scripts, each inside the folder it installs),
    and each line of its requirements file that names another file: an included requirements or constraints file, an
    editable install, a local path or ``file:`` URL, or a local folder of wheels. A lock's inputs are the requirements
    file alone, so it would be reused when one of those changed."""
    folder = PurePosixPath(install.files)
    problems = []
    if folder != PurePosixPath(".") and not any(PurePosixPath(path).is_relative_to(folder) for path in files):
        problems.append(f"install.{os_type}.files: this folder has no {install.files!r} folder")
    named = [("python", install.python), *(("sources", source.key) for source in install.sources or []),
             *(("run", run.script) for run in install.run or [])]
    problems += [f"install.{os_type}.{field}: {(folder / path).as_posix()!r} isn't in this folder"
                 for field, path in named if path is not None and (folder / path).as_posix() not in files]
    requirements = (folder / install.python).as_posix() if install.python is not None else None
    if requirements in files:
        problems += [f"install.{os_type}.python: {requirements!r} line {number}, {line.strip()!r}, names another file, "
                     "which the lock of a requirements file doesn't cover; list what it installs in the file itself"
                     for number, line in _joined(files[requirements].read_text()) if _names_a_file(line)]
    return problems


def _joined(text: str) -> list[tuple[int, str]]:
    """``text``'s lines as pip reads them, a line ending in a backslash joined to the next, each with the number of the
    line it starts on."""
    lines, start, parts = [], None, []
    for number, line in enumerate(text.splitlines(), 1):
        start = start or number
        if line.endswith("\\") and not line.lstrip().startswith("#"):
            parts.append(line[:-1])
            continue
        lines.append((start, "".join(parts) + line))
        start, parts = None, []
    return lines + ([(start, "".join(parts))] if parts else [])


def _names_a_file(line: str) -> bool:
    words = re.sub(r"(^|\s)#.*", "", line).split()
    if not words:
        return False
    if _INCLUDES.match(words[0]):
        return True
    if _FIND_LINKS.match(words[0]):
        return "://" not in line
    return words[0].startswith((".", "/", "~", "file:")) or re.search(r"@\s*file:", line) is not None
