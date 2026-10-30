"""What a computer's layer installs on one OS, as a driver's env.toml declares it under ``[install.<os>]`` or an app's
class with ``@linux(...)``: ``files``, a folder of its own copied under the layer's prefix; ``packages``, the OS's
packages, as apt takes them; ``python``, a requirements file in that folder installed into a venv beside them;
``sources``, extra apt repositories to resolve from; ``downloads``, files fetched by URL and checked by sha256;
``run``, scripts of the folder run once those are installed; ``services``, the programs the base's init runs from the
prefix; and ``ports``, those it listens on. Everything but the files is resolved and fetched once at put, and every
build installs only what was stored then (``packages`` module). The models of packages, sources, downloads and run
scripts are the protocol's, which an app's ``@linux(...)`` uses too."""

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


def installs(data: Any, model: type[Install] = Install) -> tuple[dict[str, Install], list[str]]:
    """The installs ``data``, an ``install`` table, declares by OS type, each a ``model``, and the problems with it,
    each named by its key."""
    if not isinstance(data, dict):
        return {}, [f"install must be a table of OS types ({', '.join(OS_TYPES)}), not {data!r}"]
    found, problems = {}, []
    for os_type, table in data.items():
        if os_type not in OS_TYPES:
            problems.append(f"install.{os_type}: an install is for an OS type agent-env composes: {', '.join(OS_TYPES)}")
            continue
        try:
            found[os_type] = model.model_validate(table)
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
