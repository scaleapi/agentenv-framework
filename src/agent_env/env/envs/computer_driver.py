"""Computer driver env: the code that serves a computer's tools, a ``ComputerDriverEnvironment`` subclass, with its
files and what it declares for each platform it supports, read from the class at put."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Optional

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.store.ids import derive_id

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.env.env import DeployedEnv

#: How long reading a driver's class at put may take.
DECLARE_TIMEOUT_S = 120


class ComputerDriverEnv(Env):
    type: ClassVar[str] = "computer_driver"
    description = "A computer driver: the code that serves a computer's tools, and what it declares"

    def __init__(self, id: str, version: Optional[int], module: str, files: FileArtifactUniverse, declaration: dict,
                 metadata: Optional[dict[str, Any]] = None):
        super().__init__(id, version, metadata=metadata)
        self.module = module
        self.files = files
        #: The card's ``name``, the class's ``supports``, and per platform those name the extensions it ``provides``,
        #: its ``tools`` and its computer/v1 ``features``.
        self.declaration = declaration

    def to_dict(self) -> dict:
        return {**super().to_dict(), "module": self.module, "declaration": self.declaration,
                "files": {"id": self.files.id, "version": self.files.version, "type": self.files.type}}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerDriverEnv:
        files = Artifact.get(data["files"]["id"], version=data["files"]["version"])
        return cls(id=data["id"], version=data.get("version"), module=data["module"], files=files,
                   declaration=data["declaration"], metadata=data.get("metadata") or {})

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerDriverEnv:
        """The driver's class is read in a process of its own (``agentenv_protocol.computer._declare``), so the
        driver's own dependencies need not be installed here."""
        module = cls.accept_toml(data, ctx)["module"]
        file, _, name = module.partition(":")
        declaration = _declare(ctx, file, name)
        universe = FileArtifactUniverse.put_bundled(derive_id(ctx.id, "files"), files=ctx.files())
        return cls.put(id=ctx.id, module=module, files=universe, declaration=declaration)

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a computer driver env takes: ``module = "<file>:<class>"``, naming a file
        of its folder. Raises BundleError listing every problem."""
        fields = ctx.accept(data, module=str)
        module = fields.get("module")
        if module is None:
            ctx.refuse([ctx.config_problem('module is required: the driver\'s file and class, such as '
                                           '"driver.py:ReferenceDriver"')])
        file, _, name = module.partition(":")
        files = ctx.files()
        if not name or file not in files:
            ctx.refuse([ctx.config_problem(f'module {module!r} names no "<file>:<class>" in this folder; it has '
                                           f"{', '.join(sorted(files))}")])
        return fields

    async def deploy(self, **kwargs) -> DeployedEnv:
        raise NotImplementedError(f"{self.id} is a computer_driver env, which deploys inside a computer, and "
                                  "agent-env can't create a computer yet; serve the driver from an image of its own "
                                  "as an mcp_server env meanwhile")


def _declare(ctx: AuthoringContext, file: str, name: str) -> dict:
    try:
        done = subprocess.run([sys.executable, "-m", "agentenv_protocol.computer._declare", str(ctx.dir), file, name],
                              capture_output=True, text=True, timeout=DECLARE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        _refuse(ctx, f"reading {file}:{name} took over {DECLARE_TIMEOUT_S} s")
    if done.returncode != 0:
        _refuse(ctx, f"{file}:{name} can't be read: {(done.stderr.strip().splitlines() or ['no output'])[-1]}")
    return json.loads(done.stdout)


def _refuse(ctx: AuthoringContext, problem: str) -> NoReturn:
    ctx.refuse([ctx.config_problem(problem)])
