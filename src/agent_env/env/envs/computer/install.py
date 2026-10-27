"""What a computer's layer installs on one OS, as a driver's env.toml declares it under ``[install.<os>]``: ``files``, a
folder of its own copied under the layer's prefix; ``python``, a requirements file in that folder installed into a
venv beside them; and ``services``, the programs the base's init runs from the prefix."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_env.artifact.artifacts.computer import OS_TYPES


class Service(BaseModel):
    """A program the base's init runs: as the session user, or for ``system``, as root. A ``command`` whose program is a
    relative path names one under the layer's prefix."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    command: str = Field(min_length=1)
    scope: Literal["session", "system"] = "session"


class Install(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: str = "."
    python: str | None = None
    services: list[Service] = Field(min_length=1)

    @field_validator("files", "python")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        if value is not None and (PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts):
            raise ValueError(f"{value!r} must be a path inside the layer's folder")
        return value

    @field_validator("services")
    @classmethod
    def _named_once(cls, value: list[Service]) -> list[Service]:
        names = [service.name for service in value]
        if len(set(names)) != len(names):
            raise ValueError(f"names a service twice: {names}")
        return value


def installs(data: Any) -> tuple[dict[str, Install], list[str]]:
    """The installs ``data``, an env.toml's ``install`` table, declares by OS type, and the problems with it, each
    named by its key."""
    if not isinstance(data, dict):
        return {}, [f"install must be a table of OS types ({', '.join(OS_TYPES)}), not {data!r}"]
    found, problems = {}, []
    for os_type, table in data.items():
        if os_type not in OS_TYPES:
            problems.append(f"install.{os_type}: an install is for an OS type agent-env composes: {', '.join(OS_TYPES)}")
            continue
        try:
            found[os_type] = Install.model_validate(table)
        except ValidationError as e:
            problems.extend(f"install.{'.'.join(map(str, (os_type, *error['loc'])))}: {error['msg']}"
                            for error in e.errors())
    return found, problems
