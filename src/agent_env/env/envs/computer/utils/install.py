"""What a computer's layer installs on one OS, as a driver's env.toml declares it under ``[install.<os>]`` or an app's
class with ``@linux(...)``: ``files``, a folder of its own copied under the layer's prefix; ``packages``, the OS's
packages, as apt takes them; ``python``, a requirements file in that folder installed into a venv beside them;
``sources``, extra apt repositories to resolve from; ``downloads``, files fetched by URL and checked by sha256;
``run``, scripts of the folder run once those are installed; ``services``, the programs the base's init runs from the
prefix; and ``ports``, those it listens on. Everything but the files is resolved and fetched once at put, and every
build installs only what was stored then (``packages`` module). The models of packages, sources, downloads and run
scripts are the protocol's, which an app's ``@linux(...)`` uses too. ``[install.macos]`` takes the same, but apt's
packages and sources; its services are LaunchAgents of the login session, and its ``grants`` the TCC permissions an app
bundle or an executable it installs needs."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from agentenv_protocol.computer.app import AptSource, Download, LayerPath, Packages, RunScript
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent_env.artifact.artifacts.computer import OS_TYPES

#: The highest port a layer can listen on.
MAX_PORT = 65535
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
    """An app's install, which may run no service: a desktop app runs when it's launched."""

    services: list[Service] = Field(default_factory=list)


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


#: The install model of each OS type, for a driver and for an app.
DRIVER_INSTALLS: dict[str, type[Install]] = {"linux": Install, "macos": MacInstall}
APP_INSTALLS: dict[str, type[Install]] = {"linux": AppInstall, "macos": MacAppInstall}


def installs(data: Any, models: dict[str, type[Install]] = DRIVER_INSTALLS) -> tuple[dict[str, Install], list[str]]:
    """The installs ``data``, an ``install`` table, declares by OS type, each its type's model of ``models``, and the
    problems with it, each named by its key."""
    if not isinstance(data, dict):
        return {}, [f"install must be a table of OS types ({', '.join(OS_TYPES)}), not {data!r}"]
    found, problems = {}, []
    for os_type, table in data.items():
        if os_type not in OS_TYPES:
            problems.append(f"install.{os_type}: an install is for an OS type agent-env composes: {', '.join(OS_TYPES)}")
            continue
        try:
            found[os_type] = models[os_type].model_validate(table)
        except ValidationError as e:
            problems.extend(f"install.{'.'.join(map(str, (os_type, *error['loc'])))}: {error['msg']}"
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
