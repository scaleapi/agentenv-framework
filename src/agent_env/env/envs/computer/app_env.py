"""Computer app env: software on a computer, a ``ComputerAppEnvironment`` subclass, with its files and what it declares,
read from the class at put. Deployed alone it is a one-app computer env: the computer and driver its env.toml names,
composed with the app installed under its prefix, its card named after it."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agentenv_protocol.computer import AppDeclaration

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.env.envs.computer.utils import composer, gate
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils.install import AppInstall, installs, missing
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv, declared
from agent_env.env.store import get_env_store
from agent_env.store.ids import derive_id

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_env.bundle.authoring import AuthoringContext


class ComputerAppEnv(ComputerEnv):
    type: ClassVar[str] = "computer_app"
    description = ("A computer app: software and its settings on a computer, read from its ComputerAppEnvironment "
                   "class at put; deployed alone, a computer of its own")

    def __init__(self, id: str, version: Optional[int], environment_name: str, module: str, files: FileArtifactUniverse,
                 declaration: AppDeclaration, computer: Optional[ComputerArtifact] = None,
                 driver: Optional[ComputerDriverEnv] = None, metadata: Optional[dict[str, Any]] = None):
        super().__init__(id, version, environment_name, computer, driver, metadata=metadata)
        self.module = module
        self.files = files
        self.declaration = declaration

    @property
    def install(self) -> dict[str, AppInstall]:
        return _installs(self.declaration)[0]

    def composed_apps(self) -> list[ComputerAppEnv]:
        return [self]

    def secret_settings(self) -> set[str]:
        """The names of its settings that are secrets: a ``SecretStr``, optional or not, which its schema marks
        ``writeOnly``."""
        return {name for name, schema in self.declaration.settings.get("properties", {}).items()
                if any(option.get("writeOnly") for option in [schema, *schema.get("anyOf", [])])}

    def to_dict(self) -> dict:
        return {**Env.to_dict(self), "service_name": self.environment_name,
                "environment_name": self.environment_name,  # dual-write with service_name during the rename
                "module": self.module,
                "files": {"id": self.files.id, "version": self.files.version, "type": self.files.type},
                "declaration": self.declaration.model_dump(mode="json", exclude_none=True),
                "computer": {"id": self.computer.id, "version": self.computer.version} if self.computer else None,
                "driver": {"id": self.driver.id, "version": self.driver.version} if self.driver else None}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerAppEnv:
        driver = Env.get(data["driver"]["id"], data["driver"]["version"]) if data.get("driver") else None
        if driver is not None and not isinstance(driver, ComputerDriverEnv):
            raise TypeError(f"{data['driver']['id']} is a {driver.type} env, not a computer_driver env")
        computer = ComputerArtifact.get(data["computer"]["id"], data["computer"]["version"]) if data.get("computer") else None
        return cls(id=data["id"], version=data.get("version"), environment_name=data["environment_name"],
                   module=data["module"], files=Artifact.get(data["files"]["id"], version=data["files"]["version"]),
                   declaration=AppDeclaration.model_validate(data["declaration"]), computer=computer, driver=driver,
                   metadata=data.get("metadata") or {})

    @classmethod
    def put(cls, *, id: str, environment_name: str, module: str, files: FileArtifactUniverse,
            declaration: AppDeclaration, computer: Optional[ComputerArtifact] = None,
            driver: Optional[ComputerDriverEnv] = None, compose: bool = True,
            on_build: Optional[Callable[[], None]] = None) -> Self:
        """Write the app once what it declares checks out, and with a computer and a driver, once it fits them (gate
        1); then compose it with them unless ``compose`` is False. Refused, naming every problem, before anything is
        stored."""
        app = cls(id=id, version=None, environment_name=environment_name, module=module, files=files,
                  declaration=declaration, computer=computer, driver=driver)
        _, problems = _installs(declaration)
        problems += _ready_checks_beyond_reach(module, declaration)
        if (computer is None) != (driver is None):
            problems.append("a computer app deploys alone on a computer and a driver; name both or neither")
        elif computer is not None and not problems:
            _, refused = gate.check(computer, driver, apps=[], display=None, network=None, computer_settings={})
            problems += refused + gate.check_app(computer, driver, app)
        if problems:
            raise ValueError(f"computer app env {id!r}: " + "; ".join(problems))
        app = get_env_store().put_document(app)
        if computer is not None and compose:
            composer.compose(app, on_build=on_build)
        return app

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerAppEnv:
        """Write the app an env.toml declares, its class read in a process of its own (``declared``), its files stored
        as a file universe, with the computer and driver env.toml names; its environment_name the folder's when it sets
        none."""
        fields = cls.accept_toml(data, ctx)
        declaration = AppDeclaration.model_validate(declared(ctx, fields["module"], cls.type))
        files = ctx.files()
        found, problems = _installs(declaration)
        problems += [problem for os_type, install in found.items() for problem in missing(os_type, install, files)]
        if (skill := declaration.skill and declaration.skill.file) is not None and skill not in files:
            problems.append(f"skill: {skill!r} isn't in this folder")
        if problems:
            ctx.refuse(list(map(ctx.config_problem, problems)))
        computer = ctx.artifact(fields["computer"], ComputerArtifact) if "computer" in fields else None
        driver = ctx.env(fields["driver"], ComputerDriverEnv) if "driver" in fields else None
        universe = FileArtifactUniverse.put_bundled(derive_id(ctx.id, "files"), files=files)
        try:
            return cls.put(id=ctx.id, environment_name=fields.get("environment_name", ctx.name),
                           module=fields["module"], files=universe, declaration=declaration, computer=computer,
                           driver=driver, on_build=ctx.on_build)
        except ValueError as e:
            ctx.refuse([str(e)])

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a computer app env takes: ``module = "<file>:<class>"``, naming a file of
        its folder; ``environment_name``; and the ``computer`` and ``driver`` it deploys alone on. Raises BundleError
        listing every problem."""
        fields, refused = ctx.accepted(data, environment_name=str, module=str, computer=object, driver=object)
        files, module, problems = ctx.files(), fields.get("module"), []
        if module is None and "module" not in data:
            problems.append('module is required: the app\'s file and class, such as "app.py:MailApp"')
        elif module is not None and (not module.partition(":")[2] or module.partition(":")[0] not in files):
            problems.append(f'module {module!r} names no "<file>:<class>" in this folder; it has '
                            f"{', '.join(sorted(files))}")
        if refused or problems:
            ctx.refuse([*refused, *map(ctx.config_problem, problems)])
        return fields

    def deploy_refusal(self, *, sandbox_type: str | None = None, env_params: Optional[dict] = None,
                       **options) -> str | None:
        if self.computer is None:
            return (f"computer app env {self.id!r} names no computer and driver to deploy alone on; name them in its "
                    "env.toml")
        return super().deploy_refusal(sandbox_type=sandbox_type, env_params=env_params, **options)


def _installs(declaration: AppDeclaration) -> tuple[dict[str, AppInstall], list[str]]:
    """Its installs by OS type, each checked as the composer installs it, and the problems with them."""
    return installs({os_type: linux.model_dump() for os_type, linux in declaration.install.items()}, AppInstall)


def _ready_checks_beyond_reach(module: str, declaration: AppDeclaration) -> list[str]:
    """An install that leaves out the class's file, from which the computer runs its ``@ready_check`` methods."""
    file = PurePosixPath(module.partition(":")[0])
    return [f"install.{os_type}.files: @ready_check runs from the app's folder, which needs {str(file)!r} in "
            f"{linux.files!r}" for os_type, linux in declaration.install.items()
            if declaration.ready_checks and linux.files != "." and not file.is_relative_to(linux.files)]
