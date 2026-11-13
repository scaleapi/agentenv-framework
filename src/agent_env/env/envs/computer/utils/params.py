"""What a deploy of a computer env takes as ``deploy_env`` ``env_params``: the form to boot, ``computer_settings``,
values of the base's deploy-phase settings, ``driver_settings``, values of the driver's ``Settings``, such as the
default coordinate space its tools describe and any its driver adds, ``app_settings``, values of each app's
``Settings`` by its name, and ``site_routing``, opt-in, to reach the run's websites by their names from the computer:
its ``sites``, which deploy_env fills in from the websites the run deployed before the computer, all of them or those
of the env ids it lists. The env's display and network are its own, which no deploy overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from agentenv_protocol.computer import ComputerSettings
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from agent_env.artifact.artifacts.computer import SettingValue
from agent_env.env.envs.computer.utils import gate, sites
from agent_env.env.envs.computer.utils.install import platform_of

if TYPE_CHECKING:
    from agent_env.env.envs.computer.env import ComputerEnv


class Site(BaseModel):
    """A website the computer reaches by its names: its route on the env gateway, as this process reaches it, the
    names its card's website/v1 declares, and the request headers it names as its sign-in headers, whose values the
    browser's network log takes out."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    domains: list[str] = Field(default_factory=list)
    sign_in_headers: list[str] = Field(default_factory=list)


class ComputerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form: Optional[str] = None
    computer_settings: dict[str, Any] = Field(default_factory=dict)
    driver_settings: dict[str, Any] = Field(default_factory=dict)
    app_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    site_routing: Union[StrictBool, list[str]] = False
    sites: list[Site] = Field(default_factory=list)


def driver_problems(env: ComputerEnv, values: dict[str, Any]) -> list[str]:
    """What's wrong with ``values`` as the settings of ``env``'s driver: what the schema of its own ``Settings`` refuses,
    when it declares one, else what ComputerSettings does."""
    if (schema := env.driver.declaration.settings) is None:
        try:
            ComputerSettings.model_validate(values)
        except ValidationError as e:
            return _problems("env_params.driver_settings", e)
        return []
    return [f"{'.'.join(['env_params.driver_settings', *map(str, error.absolute_path)])}: {error.message}"
            for error in Draft202012Validator(schema).iter_errors(values)]


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
    problems += driver_problems(env, params.driver_settings)
    apps = {app.environment_name: app for app in env.composed_apps()}
    problems += [f"env_params.app_settings.{name}: {env.id} has no app {name!r}"
                 + (f"; its apps are {', '.join(apps)}" if apps else "") for name in params.app_settings if name not in apps]
    merged = app_settings(env, params.app_settings)
    for name, app in apps.items():
        problems += gate.check_app_settings(app, merged[name], f"env_params.app_settings.{name}",
                                            platform_of(env.computer.os))
    if params.sites and not params.site_routing:
        problems.append("env_params.sites: a computer reaches sites only with site_routing")
    if params.site_routing and (os_type := env.computer.os.type) not in sites.TRUST_ENVIRONMENTS:
        problems.append(f"env_params.site_routing: site routing reaches Linux computers alone, and this computer runs "
                        f"{os_type}")
    allow_hosts = (env.network or {}).get("allow_hosts") or ()
    problems += sites.table_problems(params.sites, allow_hosts) if params.sites else []
    return params, values, problems


def app_settings(env: ComputerEnv, given: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The settings each of ``env``'s apps deploys with, by name: the env's defaults for it under ``given``, a deploy's
    ``app_settings``, field by field."""
    return {name: {**defaults, **given.get(name, {})} for name, defaults in env.app_defaults().items()}


def _problems(at: str, error: ValidationError) -> list[str]:
    return [f"{at}.{'.'.join(map(str, problem['loc']))}: {problem['msg']}" for problem in error.errors()]
