"""What a deploy of a computer env takes as ``deploy_env`` ``env_params``: the form to boot, ``computer_settings``,
values of the base's deploy-phase settings, ``driver_settings``, the driver's ``ComputerSettings``, such as the default
coordinate space its tools describe, and ``app_settings``, values of each app's ``Settings`` by its name. The env's
display and network are its own, which no deploy overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agentenv_protocol.computer import ComputerSettings
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_env.artifact.artifacts.computer import SettingValue
from agent_env.env.envs.computer import gate

if TYPE_CHECKING:
    from agent_env.env.envs.computer.env import ComputerEnv


class ComputerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form: Optional[str] = None
    computer_settings: dict[str, Any] = Field(default_factory=dict)
    driver_settings: dict[str, Any] = Field(default_factory=dict)
    app_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)


def checked(env: ComputerEnv, env_params: Optional[dict]) -> tuple[ComputerParams, dict[str, SettingValue], list[str]]:
    """``env_params`` read for ``env``, its values of the base's settings, and every problem with them."""
    try:
        params = ComputerParams.model_validate(env_params or {})
    except ValidationError as e:
        return ComputerParams(), {}, _problems("env_params", e)
    problems = []
    if params.form is not None and params.form not in (forms := gate.forms(env.computer)):
        problems.append(f"env_params.form: {params.form!r} isn't one of {env.computer.id}'s forms ({', '.join(forms)})")
    base = env.computer
    values, invalid = gate.check_computer_settings(base, params.computer_settings)
    problems += [f"env_params.{problem}" for problem in invalid]
    for name in values:
        if base.config[name].phase == "build":
            problems.append(f"env_params.computer_settings: {name} is a build-phase setting, which only another image "
                            "changes")
        if name == "resolution" and env.display is not None:
            problems.append("env_params.computer_settings: the env's display sets the screen's size, which no deploy "
                            "changes")
    try:
        ComputerSettings.model_validate(params.driver_settings)
    except ValidationError as e:
        problems += _problems("env_params.driver_settings", e)
    apps = {app.environment_name: app for app in env.composed_apps()}
    for name, app_values in params.app_settings.items():
        if (app := apps.get(name)) is None:
            problems.append(f"env_params.app_settings.{name}: {env.id} has no app {name!r}"
                            + (f"; its apps are {', '.join(apps)}" if apps else ""))
        else:
            problems += [f"env_params.app_settings.{name}.{problem}" for problem in gate.check_app_settings(app, app_values)]
    return params, values, problems


def _problems(at: str, error: ValidationError) -> list[str]:
    return [f"{at}.{'.'.join(map(str, problem['loc']))}: {problem['msg']}" for problem in error.errors()]
