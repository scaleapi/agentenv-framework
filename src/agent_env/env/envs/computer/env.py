"""The computer env: a base computer and a driver, named by the env with what makes it itself, checked against each
other at put (gate 1) and composed into an image for each form and arch, each recorded in its metadata as it builds
(``composed``), and the arches built written as its computer (``computer``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agent_env.artifact.artifacts.computer import FORMS, ComputerArtifact
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

    def __init__(self, id: str, version: Optional[int], environment_name: str, computer: dict, driver: dict,
                 forms: list[str], display: Optional[dict] = None, network: Optional[dict] = None,
                 hidden_tools: Optional[list[str]] = None, settings: Optional[dict] = None,
                 data_paths: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None):
        super().__init__(id, version, metadata=metadata)
        self.environment_name = environment_name
        self.computer, self.driver = computer, driver  # each {"id": ..., "version": ...}
        self.forms = forms
        self.display, self.network = display, network
        self.hidden_tools = hidden_tools or []
        self.settings = settings or {}  # setting values by layer id
        self.data_paths = data_paths or []

    def to_dict(self) -> dict:
        return {**super().to_dict(), "service_name": self.environment_name,
                "environment_name": self.environment_name,  # dual-write with service_name during the rename
                "computer": self.computer,
                "driver": self.driver, "forms": self.forms, "display": self.display, "network": self.network,
                "hidden_tools": self.hidden_tools, "settings": self.settings, "data_paths": self.data_paths}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerEnv:
        return cls(id=data["id"], version=data.get("version"),
                   environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
                   computer=data["computer"], driver=data["driver"], forms=data["forms"], display=data.get("display"),
                   network=data.get("network"), hidden_tools=data.get("hidden_tools"), settings=data.get("settings"),
                   data_paths=data.get("data_paths"), metadata=data.get("metadata") or {})

    @classmethod
    def put(cls, *, id: str, computer: ComputerArtifact, driver: ComputerDriverEnv, environment_name: str,
            forms: Optional[list[str]] = None, apps: Any = (), display: Any = None, network: Any = None,
            tools: Any = None, settings: Any = None, data_paths: Any = (), compose: bool = True) -> Self:
        """Write a computer env of ``computer`` and ``driver`` once gate 1 passes, then compose it unless ``compose`` is
        False. Refused, naming every mismatch, before anything is stored."""
        forms = list(forms) if forms is not None else [form for form in FORMS if getattr(computer.forms, form, None)]
        tools = tools if tools is not None else {}
        problems = [] if isinstance(tools, dict) and tools.keys() <= {"hide"} else [
            f"tools is {{ hide = [<pattern>, ...] }}, not {tools!r}"]
        hidden = list(tools.get("hide", [])) if isinstance(tools, dict) else []
        by_layer, mismatched = gate.check(computer, driver, forms=forms, apps=list(apps), display=display,
                                          network=network, hidden_tools=hidden, settings=settings or {},
                                          data_paths=list(data_paths))
        if problems := problems + mismatched:
            raise ValueError(f"computer env {id!r}: " + "; ".join(problems))
        env = get_env_store().put_document(cls(
            id=id, version=None, environment_name=environment_name,
            computer={"id": computer.id, "version": computer.version}, driver={"id": driver.id, "version": driver.version},
            forms=forms, display=display, network=network, hidden_tools=hidden, settings=by_layer,
            data_paths=list(data_paths)))
        if compose:
            composer.compose(env, computer, driver)
        return env

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerEnv:
        """Write the computer env an env.toml declares, its base and driver as the bundle wrote them or the ones they
        name, its environment_name the folder's when it sets none."""
        fields, refused = ctx.accepted(data, environment_name=str, computer=object, driver=object, forms=list,
                                       apps=list, display=dict, network=dict, tools=dict, settings=dict,
                                       data_paths=list)
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

    def base(self) -> ComputerArtifact:
        return ComputerArtifact.get(self.computer["id"], self.computer["version"])

    def driver_env(self) -> ComputerDriverEnv:
        driver = Env.get(self.driver["id"], self.driver["version"])
        if not isinstance(driver, ComputerDriverEnv):
            raise TypeError(f"{self.driver['id']} is a {driver.type} env, not a computer_driver env")
        return driver

    async def deploy(self, **kwargs) -> DeployedEnv:
        statuses = {f"{form}/{arch}": result.get("status") for form, by_arch in self.metadata.get("composed", {}).items()
                    for arch, result in by_arch.items()}
        raise NotImplementedError(f"{self.id} is a computer env, which agent-env can't boot yet; its builds are "
                                  f"{statuses or 'none yet'}")
