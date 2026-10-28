"""What a deploy of a computer env takes as ``deploy_env`` ``env_params``: the form to boot, deploy-phase settings by
layer, and the default coordinate space the driver's tools describe. The env's display and network are its own, which
no deploy overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_env.env.envs.computer import gate

if TYPE_CHECKING:
    from agent_env.env.envs.computer.env import ComputerEnv


class Spaces(BaseModel):
    """The coordinate space the driver's tools take points in unless a call names one, as ``computer-spaces/v1`` sets it."""

    model_config = ConfigDict(extra="forbid")

    default_space: Literal["frame", "norm"]
    default_norm_scale: int = Field(default=1000, ge=1, le=1_000_000)


class ComputerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form: Optional[str] = None
    settings: dict[str, Any] = Field(default_factory=dict)
    spaces: Optional[Spaces] = None


def checked(env: ComputerEnv, env_params: Optional[dict]) -> tuple[ComputerParams, dict[str, dict], list[str]]:
    """``env_params`` read for ``env``, its settings by layer id, and every problem with them."""
    try:
        params = ComputerParams.model_validate(env_params or {})
    except ValidationError as e:
        return ComputerParams(), {}, [f"env_params.{'.'.join(map(str, error['loc']))}: {error['msg']}"
                                      for error in e.errors()]
    problems = []
    if params.form is not None and params.form not in (forms := gate.forms(env.computer)):
        problems.append(f"env_params.form: {params.form!r} isn't one of {env.computer.id}'s forms ({', '.join(forms)})")
    base = env.computer
    by_layer, layered = gate.layered(base, env.driver, params.settings)
    problems += [f"env_params.{problem}" for problem in layered]
    for name in by_layer.get(base.id, {}):
        if base.config[name].phase == "build":
            problems.append(f"env_params.settings: {name} is a build-phase setting, which only another image changes")
        if name == "resolution" and env.display is not None:
            problems.append("env_params.settings: the env's display sets the screen's size, which no deploy changes")
    return params, by_layer, problems
