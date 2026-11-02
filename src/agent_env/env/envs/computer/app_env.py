"""Computer app env: software on a computer, a ``ComputerAppEnvironment`` subclass, with its files, what it declares,
read from the class at put, and the values of its build settings the env fixes in the image. Deployed alone it is a
one-app computer env: the computer and driver its env.toml names, composed with the app installed under its prefix, its
card named after it."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agentenv_protocol.computer import AppDeclaration, Launch
from agentenv_protocol.computer.app import SOCKET

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.env.envs.computer.utils import composer, gate
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.env import APP_TYPE, ComputerEnv
from agent_env.env.envs.computer.utils.install import AppInstall, folder_problems, installs
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv, declared
from agent_env.env.store import get_env_store
from agent_env.store.ids import derive_id

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_env.bundle.authoring import AuthoringContext


class ComputerAppEnv(ComputerEnv):
    type: ClassVar[str] = APP_TYPE
    description = ("A computer app: software and its settings on a computer, read from its ComputerAppEnvironment "
                   "class at put; deployed alone, a computer of its own")

    def __init__(self, id: str, version: Optional[int], environment_name: str, module: str, files: FileArtifactUniverse,
                 declaration: AppDeclaration, computer: Optional[ComputerArtifact] = None,
                 driver: Optional[ComputerDriverEnv] = None, build_settings: Optional[dict[str, Any]] = None,
                 metadata: Optional[dict[str, Any]] = None):
        super().__init__(id, version, environment_name, computer, driver, metadata=metadata)
        self.module = module
        self.files = files
        self.declaration = declaration
        self.build_settings = build_settings or {}

    @property
    def install(self) -> dict[str, AppInstall]:
        return _installs(self.declaration)[0]

    def composed_apps(self) -> list[ComputerAppEnv]:
        return [self]

    def settings_schema(self, os_type: str | None) -> dict[str, Any]:
        """The JSON schema of its settings on ``os_type``: its class's, with the fields that OS adds; with no OS type,
        its class's."""
        on = self.declaration.os.get(os_type)
        own = on.settings if on is not None else None
        return own if own is not None else self.declaration.settings

    def build_settings_schema(self, os_type: str | None) -> dict[str, Any] | None:
        """The JSON schema of its build settings on ``os_type``, as ``settings_schema``; None when it declares none."""
        on = self.declaration.os.get(os_type)
        own = on.build_settings if on is not None else None
        return own if own is not None else self.declaration.build_settings

    def secret_settings(self, os_type: str | None) -> set[str]:
        """The names of its settings on ``os_type`` that are secrets: a ``SecretStr``, optional or not, which its schema
        marks ``writeOnly``."""
        return {name for name, schema in self.settings_schema(os_type).get("properties", {}).items()
                if any(option.get("writeOnly") for option in [schema, *schema.get("anyOf", [])])}

    def launch(self, os_type: str) -> Launch | None:
        """How it launches on ``os_type``."""
        on = self.declaration.os.get(os_type)
        return on.launch if on is not None else None

    def to_dict(self) -> dict:
        return {**Env.to_dict(self), "service_name": self.environment_name,
                "environment_name": self.environment_name,  # dual-write with service_name during the rename
                "module": self.module,
                "files": {"id": self.files.id, "version": self.files.version, "type": self.files.type},
                "declaration": self.declaration.model_dump(mode="json", exclude_none=True),
                "computer": {"id": self.computer.id, "version": self.computer.version} if self.computer else None,
                "driver": {"id": self.driver.id, "version": self.driver.version} if self.driver else None,
                **({"build_settings": self.build_settings} if self.build_settings else {})}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerAppEnv:
        driver = Env.get(data["driver"]["id"], data["driver"]["version"]) if data.get("driver") else None
        if driver is not None and not isinstance(driver, ComputerDriverEnv):
            raise TypeError(f"{data['driver']['id']} is a {driver.type} env, not a computer_driver env")
        computer = ComputerArtifact.get(data["computer"]["id"], data["computer"]["version"]) if data.get("computer") else None
        return cls(id=data["id"], version=data.get("version"), environment_name=data["environment_name"],
                   module=data["module"], files=Artifact.get(data["files"]["id"], version=data["files"]["version"]),
                   declaration=AppDeclaration.model_validate(data["declaration"]), computer=computer, driver=driver,
                   build_settings=data.get("build_settings"), metadata=data.get("metadata") or {})

    @classmethod
    def put(cls, *, id: str, environment_name: str, module: str, files: FileArtifactUniverse,
            declaration: AppDeclaration, computer: Optional[ComputerArtifact] = None,
            driver: Optional[ComputerDriverEnv] = None, build_settings: Optional[dict[str, Any]] = None,
            compose: bool = True, composing: Composing = Composing()) -> Self:
        """Write the app once what it declares and ``build_settings``, its build settings' values, check out, and with
        a computer and a driver, once it fits them (gate 1); then compose it with them as ``composing`` says unless
        ``compose`` is False. Refused, naming every problem, before anything is stored."""
        app = cls(id=id, version=None, environment_name=environment_name, module=module, files=files,
                  declaration=declaration, computer=computer, driver=driver, build_settings=build_settings)
        _, problems = _installs(declaration)
        problems += _class_beyond_reach(module, declaration)
        problems += _in_server_folder(declaration, files.get_file_artifacts())
        problems += gate.check_app_build_settings(app, app.build_settings, "build_settings",
                                                  computer.os.type if computer is not None else None)
        if (computer is None) != (driver is None):
            problems.append("a computer app deploys alone on a computer and a driver; name both or neither")
        elif computer is not None and not problems:
            _, refused = gate.check(computer, driver, environment_name=environment_name, apps=[], display=None,
                                    network=None, computer_settings={})
            problems += refused + gate.check_app(computer, driver, app)
        if problems:
            raise ValueError(f"computer app env {id!r}: " + "; ".join(problems))
        app = get_env_store().put_document(app)
        if computer is not None and compose:
            composer.compose(app, composing=composing)
        return app

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerAppEnv:
        """Write the app an env.toml declares, its class read in a process of its own (``declared``), its files stored
        as a file universe, with the computer and driver and the build settings' values env.toml names; its
        environment_name the folder's when it sets none."""
        fields = cls.accept_toml(data, ctx)
        declaration = AppDeclaration.model_validate(declared(ctx, fields["module"], cls.type))
        files = ctx.files()
        found, problems = _installs(declaration)
        problems += [problem for os_type, install in found.items()
                     for problem in folder_problems(os_type, install, files)]
        skills = [("skill", declaration.skill),
                  *((f"{os_type}.skill", on.skill) for os_type, on in declaration.os.items())]
        problems += [f"{where}: {skill.file!r} isn't in this folder" for where, skill in skills
                     if skill is not None and skill.file is not None and skill.file not in files]
        if problems:
            ctx.refuse(list(map(ctx.config_problem, problems)))
        computer = ctx.artifact(fields["computer"], ComputerArtifact) if "computer" in fields else None
        driver = ctx.env(fields["driver"], ComputerDriverEnv) if "driver" in fields else None
        universe = FileArtifactUniverse.put_bundled(derive_id(ctx.id, "files"), files=files)
        try:
            return cls.put(id=ctx.id, environment_name=fields["environment_name"], module=fields["module"],
                           files=universe, declaration=declaration, computer=computer, driver=driver,
                           build_settings=fields.get("build_settings"), composing=ctx.composing)
        except ValueError as e:
            ctx.refuse([str(e)])

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a computer app env takes: ``module = "<file>:<class>"``, naming a file of
        its folder; ``environment_name``, the folder's when it sets none; the ``computer`` and ``driver`` it deploys
        alone on; and ``build_settings``, the values of its build settings. Raises BundleError listing every
        problem."""
        fields, refused = ctx.accepted(data, environment_name=str, module=str, computer=object, driver=object,
                                       build_settings=dict)
        files, module, problems = ctx.files(), fields.get("module"), []
        if module is None and "module" not in data:
            problems.append('module is required: the app\'s file and class, such as "app.py:MailApp"')
        elif module is not None and (not module.partition(":")[2] or module.partition(":")[0] not in files):
            problems.append(f'module {module!r} names no "<file>:<class>" in this folder; it has '
                            f"{', '.join(sorted(files))}")
        if refused or problems:
            ctx.refuse([*refused, *map(ctx.config_problem, problems)])
        return {"environment_name": ctx.name, **fields}

    def deploy_refusal(self, *, sandbox_type: str | None = None, env_params: Optional[dict] = None,
                       **options) -> str | None:
        if self.computer is None:
            return (f"computer app env {self.id!r} names no computer and driver to deploy alone on; name them in its "
                    "env.toml")
        return super().deploy_refusal(sandbox_type=sandbox_type, env_params=env_params, **options)


def _installs(declaration: AppDeclaration) -> tuple[dict[str, AppInstall], list[str]]:
    """Its installs by OS type, each checked as the composer installs it, and the problems with them."""
    return installs({os_type: on.install for os_type, on in declaration.os.items()}, AppInstall)


def _in_server_folder(declaration: AppDeclaration, names: Iterable[str]) -> list[str]:
    """The app's files its install would put in the folder its own server keeps its socket and build settings in,
    which is the framework's: a build settings file there would reach the app unchecked."""
    reserved, problems = SOCKET.parent.as_posix(), []
    for os_type, on in declaration.os.items():
        folder = PurePosixPath(on.install.get("files", "."))
        problems += [f"install.{os_type}.files: {name!r} is in {reserved}/, where the app's own server keeps its socket "
                     "and build settings; leave it out of the app's folder"
                     for name in sorted(names) if PurePosixPath(name).is_relative_to(folder / reserved)]
    return problems


def _class_beyond_reach(module: str, declaration: AppDeclaration) -> list[str]:
    """An install that leaves out the class's file, which the app's own server serves from the app's folder."""
    file = PurePosixPath(module.partition(":")[0])
    folders = {os_type: on.install.get("files", ".") for os_type, on in declaration.os.items()}
    return [f"install.{os_type}.files: the app's server serves its class from the app's folder, which needs "
            f"{str(file)!r} in {folder!r}" for os_type, folder in folders.items()
            if folder != "." and not file.is_relative_to(folder)]
