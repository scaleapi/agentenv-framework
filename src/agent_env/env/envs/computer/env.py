"""The computer env: a base computer and a driver, named by the env with what makes it itself, checked against each
other at put (gate 1) and composed into an image for each form the base has and each arch, each recorded in its
metadata as it builds (``composed``), and the arches built written as its computer (``computer``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agent_env.artifact.artifacts.computer import ComputerArtifact, SettingValue
from agent_env.entity.refs import EntityRef
from agent_env.env.env import Env
from agent_env.env.envs.computer import composer, gate
from agent_env.env.envs.computer_driver import ComputerDriverEnv
from agent_env.env.store import get_env_store

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.env.env import DeployedEnv


class ComputerEnv(Env):
    type: ClassVar[str] = "computer"
    description = "A computer: a base computer and a driver, composed at put into an image for each form and arch"
    builds_at_put: ClassVar[bool] = True
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("computer", artifact_type="computer"),
                                                  EntityRef.env("driver", env_type=ComputerDriverEnv.type))

    def __init__(self, id: str, version: Optional[int], environment_name: str, computer: ComputerArtifact,
                 driver: ComputerDriverEnv, display: Optional[dict] = None, network: Optional[dict] = None,
                 computer_settings: Optional[dict[str, SettingValue]] = None, metadata: Optional[dict[str, Any]] = None):
        super().__init__(id, version, metadata=metadata)
        self.environment_name = environment_name
        self.computer, self.driver = computer, driver
        self.display, self.network = display, network
        self.computer_settings = computer_settings or {}

    def to_dict(self) -> dict:
        return {**super().to_dict(), "service_name": self.environment_name,
                "environment_name": self.environment_name,  # dual-write with service_name during the rename
                "computer": {"id": self.computer.id, "version": self.computer.version},
                "driver": {"id": self.driver.id, "version": self.driver.version}, "display": self.display,
                "network": self.network, "computer_settings": self.computer_settings}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerEnv:
        driver = Env.get(data["driver"]["id"], data["driver"]["version"])
        if not isinstance(driver, ComputerDriverEnv):
            raise TypeError(f"{data['driver']['id']} is a {driver.type} env, not a computer_driver env")
        return cls(id=data["id"], version=data.get("version"),
                   environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
                   computer=ComputerArtifact.get(data["computer"]["id"], data["computer"]["version"]), driver=driver,
                   display=data.get("display"), network=data.get("network"),
                   computer_settings=data.get("computer_settings"), metadata=data.get("metadata") or {})

    @classmethod
    def put(cls, *, id: str, computer: ComputerArtifact, driver: ComputerDriverEnv, environment_name: str,
            apps: Any = (), display: Any = None, network: Any = None, computer_settings: Any = None,
            compose: bool = True) -> Self:
        """Write a computer env of ``computer`` and ``driver`` once gate 1 passes, then compose it unless ``compose`` is
        False. Refused, naming every mismatch, before anything is stored."""
        values, problems = gate.check(computer, driver, apps=list(apps), display=display, network=network,
                                      computer_settings=computer_settings or {})
        if problems:
            raise ValueError(f"computer env {id!r}: " + "; ".join(problems))
        env = get_env_store().put_document(cls(
            id=id, version=None, environment_name=environment_name,
            computer=computer, driver=driver, display=display, network=network, computer_settings=values))
        if compose:
            composer.compose(env)
        return env

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerEnv:
        """Write the computer env an env.toml declares, its base and driver as the bundle wrote them or the ones they
        name, its environment_name the folder's when it sets none."""
        fields, refused = ctx.accepted(data, environment_name=str, computer=object, driver=object, apps=list,
                                       display=dict, network=dict, computer_settings=dict)
        missing = [ctx.config_problem(f"{key} is required: the {what}, such as {example}") for key, what, example in (
            ("computer", "base computer", '"ubuntu-24.04"'), ("driver", "driver", '"reference"')) if key not in data]
        if refused or missing:
            ctx.refuse([*refused, *missing])
        base, driver = ctx.artifact(fields.pop("computer"), ComputerArtifact), ctx.env(fields.pop("driver"),
                                                                                       ComputerDriverEnv)
        try:
            return cls.put(id=ctx.id, computer=base, driver=driver,
                           environment_name=fields.pop("environment_name", ctx.name), **fields)
        except ValueError as e:
            ctx.refuse([str(e)])

    async def deploy(self, **kwargs) -> DeployedEnv:
        statuses = {f"{form}/{arch}": result.get("status") for form, by_arch in self.metadata.get("composed", {}).items()
                    for arch, result in by_arch.items()}
        raise NotImplementedError(f"{self.id} is a computer env, which agent-env can't boot yet; its builds are "
                                  f"{statuses or 'none yet'}")
