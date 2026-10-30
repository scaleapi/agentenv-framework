"""What a computer's layer installs on one OS, as a driver's env.toml declares it under ``[install.<os>]`` or an app's
class as ``linux = Linux(...)``: ``files``, a folder of its own copied under the layer's prefix; ``packages``, the OS's
packages, as apt takes them; ``python``, a requirements file in that folder installed into a venv beside them;
``sources``, extra apt repositories to resolve from; ``downloads``, files fetched by URL and checked by sha256;
``run``, scripts of the folder run once those are installed; ``services``, the programs the base's init runs from the
prefix; and ``ports``, those it listens on. Everything but the files is resolved and fetched once at put, and every
build installs only what was stored then (``packages`` module)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent_env.artifact.artifacts.computer import OS_TYPES

#: The highest port a layer can listen on.
MAX_PORT = 65535


class Service(BaseModel):
    """A program the base's init runs: as the session user, or for ``system``, as root. A ``command`` whose program is a
    relative path names one under the layer's prefix."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    command: str = Field(min_length=1)
    scope: Literal["session", "system"] = "session"


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
    of the layer's folder."""

    model_config = ConfigDict(extra="forbid")

    apt: str = Field(pattern=r"^https?://")
    suite: str = Field(min_length=1)
    components: list[str] = Field(default_factory=list)
    key: str

    @field_validator("key")
    @classmethod
    def _relative(cls, value: str) -> str:
        return _inside(value)


class Download(BaseModel):
    """A file fetched by its URL and checked against its sha256, put at ``to`` under the layer's prefix; for one arch
    only, when ``arch`` names it."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(pattern=r"^https?://")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    to: str
    arch: Literal["amd64", "arm64"] | None = None

    @field_validator("to")
    @classmethod
    def _relative(cls, value: str) -> str:
        return _inside(value)


class RunScript(BaseModel):
    """A script of the layer's folder run as root, or for ``session``, as the session user, once files, packages and the
    venv are in."""

    model_config = ConfigDict(extra="forbid")

    script: str
    scope: Literal["system", "session"] = "system"

    @field_validator("script")
    @classmethod
    def _relative(cls, value: str) -> str:
        return _inside(value)


class Install(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: str = "."
    packages: Packages | None = None
    python: str | None = None
    sources: list[AptSource] | None = None
    downloads: list[Download] | None = None
    run: list[RunScript] | None = None
    services: list[Service] = Field(min_length=1)
    ports: list[int] | None = None

    @field_validator("files", "python")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        return value if value is None else _inside(value)

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


def missing(os_type: str, install: Install, files: dict) -> list[str]:
    """What ``install`` names that the layer's folder, ``files`` by path, doesn't hold: its folder, its requirements, its
    sources' keys and its scripts, each inside the folder it installs."""
    folder = PurePosixPath(install.files)
    problems = []
    if folder != PurePosixPath(".") and not any(PurePosixPath(path).is_relative_to(folder) for path in files):
        problems.append(f"install.{os_type}.files: this folder has no {install.files!r} folder")
    named = [("python", install.python), *(("sources", source.key) for source in install.sources or []),
             *(("run", run.script) for run in install.run or [])]
    problems += [f"install.{os_type}.{field}: {(folder / path).as_posix()!r} isn't in this folder"
                 for field, path in named if path is not None and (folder / path).as_posix() not in files]
    return problems


def _inside(path: str) -> str:
    if not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError(f"{path!r} must be a path inside the layer's folder")
    return path
