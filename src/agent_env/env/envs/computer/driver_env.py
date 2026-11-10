"""Computer driver env: the code that serves a computer's tools, a ``ComputerDriverEnvironment`` subclass, with its
files, what it declares for each platform it supports, read from the class at put, how a computer installs it and
reaches it, and the probe checks it's known to fail on an OS, which gate 3 records without failing the build."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Optional

from agentenv_protocol.computer.driver import DriverDeclaration
from agentenv_protocol.computer.platform import entry_for
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.env.envs.computer.utils.install import (DRIVER_INSTALLS, Install, folder_problems, installs,
                                                        platform_problem, unsupported)
from agent_env.store.ids import derive_id

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.env.env import DeployedEnv

#: How long reading a driver's class at put may take.
DECLARE_TIMEOUT_S = 120
#: The port the SDK serves a driver on when MCP_PORT doesn't name one.
DEFAULT_PORT = 18765
MAX_PORT = 65535


class KnownFailure(BaseModel):
    """A probe check the driver fails on an OS for a defect of its own, by the check's id and name, and why. Gate 3
    records it in the build's probe and doesn't fail the build for it; any other failure still fails it."""

    model_config = ConfigDict(extra="forbid")

    check: str = Field(pattern=r"^(P[0-9]+|X)$")
    name: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ComputerDriverEnv(Env):
    type: ClassVar[str] = "computer_driver"
    description = "A computer driver: the code that serves a computer's tools, and what it declares"

    def __init__(self, id: str, version: Optional[int], module: str, files: FileArtifactUniverse,
                 declaration: DriverDeclaration, install: Optional[dict[str, Install]] = None, port: int = DEFAULT_PORT,
                 metadata: Optional[dict[str, Any]] = None,
                 known_failures: Optional[dict[str, list[KnownFailure]]] = None):
        super().__init__(id, version, metadata=metadata)
        self.module = module
        self.files = files
        self.declaration = declaration
        self.install = install or {}
        self.port = port
        self.known_failures = known_failures or {}

    def to_dict(self) -> dict:
        return {**super().to_dict(), "module": self.module,
                "declaration": self.declaration.model_dump(mode="json", exclude_none=True),
                "files": {"id": self.files.id, "version": self.files.version, "type": self.files.type},
                "install": {os_type: install.model_dump(exclude_none=True) for os_type, install in self.install.items()},
                "port": self.port,
                **({"known_failures": {os_type: [failure.model_dump() for failure in failures]
                                       for os_type, failures in self.known_failures.items()}}
                   if self.known_failures else {})}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerDriverEnv:
        files = Artifact.get(data["files"]["id"], version=data["files"]["version"])
        install = {key: entry_for(DRIVER_INSTALLS, key).model_validate(table)
                   for key, table in (data.get("install") or {}).items()}
        return cls(id=data["id"], version=data.get("version"), module=data["module"], files=files,
                   declaration=DriverDeclaration.model_validate(data["declaration"]), install=install,
                   port=data.get("port", DEFAULT_PORT), metadata=data.get("metadata") or {},
                   known_failures=_known_failures(data.get("known_failures") or {})[0])

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerDriverEnv:
        """The driver's class is read in a process of its own (``declared``)."""
        fields = cls.accept_toml(data, ctx)
        declaration = DriverDeclaration.model_validate(declared(ctx, fields["module"], cls.type))
        if problems := [*unsupported(fields["install"], declaration.supports, "install"),
                        *unsupported(fields.get("known_failures") or {}, declaration.supports, "known_failures")]:
            ctx.refuse(list(map(ctx.config_problem, problems)))
        universe = FileArtifactUniverse.put_bundled(derive_id(ctx.id, "files"), files=ctx.files())
        return cls.put(id=ctx.id, module=fields["module"], files=universe, declaration=declaration,
                       install=fields["install"], port=fields.get("port", DEFAULT_PORT),
                       known_failures=fields.get("known_failures"))

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a computer driver env takes: ``module = "<file>:<class>"``, naming a file
        of its folder; ``port``, where it serves; ``install``, by OS type, whose files and requirements are in its
        folder; and ``known_failures``, by OS type, the probe checks it fails there. An install or known failures for
        a distribution, ``<os>/<distribution>``, replace its OS type's there. Raises BundleError listing every
        problem."""
        fields, refused = ctx.accepted(data, module=str, port=int, install=dict, known_failures=dict)
        files = ctx.files()
        module, problems = fields.get("module"), []
        if module is None and "module" not in data:
            problems.append('module is required: the driver\'s file and class, such as "driver.py:ReferenceDriver"')
        elif module is not None and (not module.partition(":")[2] or module.partition(":")[0] not in files):
            problems.append(f'module {module!r} names no "<file>:<class>" in this folder; it has '
                            f"{', '.join(sorted(files))}")
        if not 1 <= fields.get("port", DEFAULT_PORT) <= MAX_PORT:
            problems.append(f"port must be 1 to {MAX_PORT}, not {fields['port']}")
        fields["known_failures"], unknown = _known_failures(fields.get("known_failures", {}))
        problems += unknown
        fields["install"], invalid = installs(fields.get("install", {}))
        problems += invalid + [problem for os_type, install in fields["install"].items()
                               for problem in folder_problems(os_type, install, files)]
        if refused or problems:
            ctx.refuse([*refused, *map(ctx.config_problem, problems)])
        return fields

    async def deploy(self, **kwargs) -> DeployedEnv:
        raise NotImplementedError(f"{self.id} is a computer_driver env, which deploys inside a computer, never on its "
                                  "own: name it as a computer env's driver and deploy that")


def declared(ctx: AuthoringContext, module: str, env_type: str) -> dict:
    """What the class ``module`` names in the folder declares, read in a process of its own
    (``agentenv_protocol.computer._declare``), so its own dependencies need not be installed here; refused unless an
    env of ``env_type`` stores it."""
    file, _, name = module.partition(":")
    try:
        done = subprocess.run([sys.executable, "-m", "agentenv_protocol.computer._declare", str(ctx.dir), file, name],
                              capture_output=True, text=True, timeout=DECLARE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        _refuse(ctx, f"reading {module} took over {DECLARE_TIMEOUT_S} s")
    if done.returncode != 0:
        _refuse(ctx, f"{module} can't be read: {(done.stderr.strip().splitlines() or ['no output'])[-1]}")
    declaration = json.loads(done.stdout)
    if (declares := declaration.pop("type")) != env_type:
        _refuse(ctx, f"{module} is a class a {declares} env declares, not a {env_type} env")
    return declaration


def _refuse(ctx: AuthoringContext, problem: str) -> NoReturn:
    ctx.refuse([ctx.config_problem(problem)])


def _known_failures(data: dict) -> tuple[dict[str, list[KnownFailure]], list[str]]:
    """The known failures ``data`` declares by OS type or distribution, and the problems with it, each named by its
    key."""
    found, problems = {}, []
    for key, failures in data.items():
        if problem := platform_problem(key):
            problems.append(f"known_failures.{key}: a known failure is for {problem}")
            continue
        try:
            found[key] = [KnownFailure.model_validate(failure) for failure in failures]
        except (ValidationError, TypeError) as e:
            problems.append(f"known_failures.{key}: {e}")
    return found, problems
