"""Gate 1, run at a computer env's put before anything is stored or built: its base, driver and apps are matched against
each other and against what the env declares, and every mismatch is named."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agentenv_protocol.computer.platform import Platform, supported
from jsonschema import Draft202012Validator

from agent_env.artifact.artifacts.computer import FORMS, ComputerArtifact, SettingValue
from agent_env.config.loader import ENV_REF_PREFIX, SECRET_REF_PREFIX
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.store.image_store.oci_registry_credentials import names_registry

if TYPE_CHECKING:
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.env.envs.computer.driver_env import ComputerDriverEnv

#: The largest screen side X can address: its coordinates are signed 16-bit.
MAX_SIDE = 32767


def check(base: ComputerArtifact, driver: ComputerDriverEnv, *, apps: list, display: Any, network: Any,
          computer_settings: Any) -> tuple[dict[str, SettingValue], list[str]]:
    """The env's values of the base's settings, and every way ``base``, ``driver`` and what the env declares don't
    compose."""
    problems = []
    if apps:
        problems.append("apps: computer app envs aren't supported yet")
    problems += _platform(base, driver)
    problems += _image(base)
    problems += _display(base, display, computer_settings)
    problems += _network(network)
    values, invalid = check_computer_settings(base, computer_settings)
    return values, problems + invalid


def forms(base: ComputerArtifact) -> list[str]:
    """The forms a composition of ``base`` builds: every one the base has."""
    return [form for form in FORMS if getattr(base.forms, form, None)]


def arches(base: ComputerArtifact, driver: ComputerDriverEnv, apps: Sequence[ComputerAppEnv] = ()) -> list[str]:
    """The architectures a composition of ``base``, ``driver`` and ``apps`` runs on: the base's that a platform of the
    driver's and of each app's matching the base's OS names, in the base's order."""
    matching = {platform.arch for platform in _matching(base, driver.declaration["supports"])}
    for app in apps:
        matching &= {platform.arch for platform in _matching(base, app.declaration["supports"])}
    return [arch for arch in base.arch if arch in matching]


def check_app(base: ComputerArtifact, driver: ComputerDriverEnv, app: ComputerAppEnv) -> list[str]:
    """Every way ``app`` doesn't fit a composition of ``base`` and ``driver``: a platform it doesn't support, no
    architecture in common, or no install for the base's OS."""
    os = base.os
    if not _matching(base, app.declaration["supports"]):
        named = ", ".join(_shown(platform) for entry in app.declaration["supports"] for platform in supported(entry))
        return [f"app: {app.id} supports no platform {base.id} is ({os.type} {os.distribution} {os.version} "
                f"{os.display}); it supports {named or 'none'}"]
    problems = []
    if not arches(base, driver, [app]):
        problems.append(f"app: no architecture {base.id}, {driver.id} and {app.id} all run")
    if os.type not in app.install:
        problems.append(f"app: {app.id} declares no install for {os.type}")
    return problems


def check_app_settings(app: ComputerAppEnv, values: dict[str, Any], at: str) -> list[str]:
    """The problems with ``values``, of ``app``'s settings by name, each named from ``at``: a setting it doesn't declare,
    a secret's given as anything but a bare reference (or null, where it's optional), a required one missing, and what
    the schema of its ``Settings`` refuses of the rest. A secret's reference never meets the constraints on its value,
    and its value never reaches a message."""
    schema, secrets = app.declaration["settings"], app.secret_settings()
    properties, required = schema.get("properties", {}), schema.get("required", [])
    problems = [f"{at}.{name}: {app.id} declares no setting {name!r}" for name in values if name not in properties]
    problems += [f"{at}.{name} is a secret, so it takes a {SECRET_REF_PREFIX}KEY or {ENV_REF_PREFIX}NAME reference "
                 "with no ?default, not its value" for name in values if name in secrets
                 and not (_names_a_secret(values[name]) or values[name] is None and _optional(properties[name]))]
    problems += [f"{at}: {name!r} is a required property" for name in required if name in secrets and name not in values]
    plain = {**schema, "properties": {name: setting for name, setting in properties.items() if name not in secrets},
             "required": [name for name in required if name not in secrets]}
    return problems + [f"{'.'.join([at, *map(str, error.absolute_path)])}: {error.message}" for error in
                       Draft202012Validator(plain).iter_errors({k: v for k, v in values.items() if k not in secrets})]


def _optional(setting: dict) -> bool:
    return any(option.get("type") == "null" for option in setting.get("anyOf", []))


def check_computer_settings(base: ComputerArtifact, values: Any) -> tuple[dict[str, SettingValue], list[str]]:
    """``values``, of the base's settings by name, and the problems with them: a setting the base doesn't declare, a
    value of another type than its default's, or a secret's given as anything but a bare reference, since the env, the
    task and the deploy record all keep what's given; a secret's reference is a string whatever its default is."""
    if not isinstance(values, dict):
        return {}, [f"computer_settings must be a table of the computer's setting values, not {values!r}"]
    checked, problems = {}, []
    for name, value in values.items():
        if (setting := base.config.get(name)) is None:
            problems.append(f"computer_settings.{name}: {base.id} declares no setting {name!r}")
        elif setting.secret:
            if _names_a_secret(value):
                checked[name] = value
            else:
                problems.append(f"computer_settings.{name} is a secret, so it takes a {SECRET_REF_PREFIX}KEY or "
                                f"{ENV_REF_PREFIX}NAME reference with no ?default, not its value")
        elif type(value) is not type(setting.default):
            problems.append(f"computer_settings.{name} must be a {type(setting.default).__name__}, like its default "
                            f"{setting.default!r}, not {value!r}")
        else:
            checked[name] = value
    return checked, problems


def _names_a_secret(value: Any) -> bool:
    """Whether ``value`` names a secret rather than holding it: a bare reference, as a stored context keeps one."""
    return (isinstance(value, str) and value.startswith((ENV_REF_PREFIX, SECRET_REF_PREFIX)) and "?" not in value
            and value not in (ENV_REF_PREFIX, SECRET_REF_PREFIX))


def _platform(base: ComputerArtifact, driver: ComputerDriverEnv) -> list[str]:
    os = base.os
    if not _matching(base, driver.declaration["supports"]):
        named = ", ".join(_shown(platform) for entry in driver.declaration["supports"] for platform in supported(entry))
        return [f"driver: {driver.id} supports no platform {base.id} is ({os.type} {os.distribution} {os.version} "
                f"{os.display}); it supports {named}"]
    problems = []
    if not arches(base, driver):
        problems.append(f"driver: no architecture both run: {base.id} runs {', '.join(base.arch)}, and {driver.id} "
                        f"{', '.join(sorted({p.arch for p in _matching(base, driver.declaration['supports'])}))}")
    if (install := driver.install.get(os.type)) is None:
        problems.append(f"driver: {driver.id} declares no install for {os.type}")
    else:
        problems += [f"driver: its service {service.name!r} runs as root, and {base.id}'s "
                     f"{base.forms.container.init} runs every program as {base.session.user}"
                     for service in install.services if service.scope == "system"]
    return problems


def _matching(base: ComputerArtifact, supports: list) -> list[Platform]:
    """The platforms of ``supports``, as a driver or app declares them, that the base's OS is."""
    os = base.os
    return [platform for entry in supports for platform in supported(entry)
            if platform.os == os.type and platform.display == os.display
            and platform.distribution in (None, os.distribution) and platform.version in (None, os.version)]


def _shown(platform: Platform) -> str:
    return " ".join(part for part in (platform.os, platform.distribution, platform.version, platform.display,
                                      platform.arch) if part)


def _image(base: ComputerArtifact) -> list[str]:
    image = base.image()
    if names_registry(image.image_name) or image.buildable:
        return []
    return [f"computer: {base.id}'s container image {image.image_name!r} is in no registry and has no build context "
            "to build it from"]


def _display(base: ComputerArtifact, display: Any, computer_settings: Any) -> list[str]:
    if display is None:
        return []
    if not isinstance(display, dict) or display.keys() != {"width", "height"} or not all(
            isinstance(side, int) and not isinstance(side, bool) and 1 <= side <= MAX_SIDE for side in display.values()):
        return [f"display is {{ width = <n>, height = <n> }}, each 1 to {MAX_SIDE} pixels, not {display!r}"]
    if "resolution" not in base.config:
        return [f"display: {base.id} declares no resolution setting for it to set"]
    if isinstance(computer_settings, dict) and "resolution" in computer_settings:
        return ["display and the base's resolution setting both set the screen's size; set display alone"]
    return []


def _network(network: Any) -> list[str]:
    if network is None:
        return []
    if not isinstance(network, dict) or not network.keys() <= {"mode", "allow_hosts", "allow_cidrs"}:
        return [f"network is {{ mode, allow_hosts, allow_cidrs }}, not {network!r}"]
    try:
        NetworkPolicy(**network)
    except (TypeError, ValueError) as e:
        return [f"network: {e}"]
    return []
