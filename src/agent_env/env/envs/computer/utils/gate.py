"""Gate 1, run at a computer env's put before anything is stored or built: its base, driver and apps are matched against
each other and against what the env declares, and every mismatch is named."""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from agentenv_protocol.computer.apps import APPS
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


def check(base: ComputerArtifact, driver: ComputerDriverEnv, *, environment_name: str, apps: list, display: Any,
          network: Any, computer_settings: Any) -> tuple[dict[str, SettingValue], list[str]]:
    """The env's values of the base's settings, and every way ``base``, ``driver`` and what the env declares, named
    ``environment_name``, don't compose."""
    problems = _platform(base, driver) + _apps(base, driver, apps, environment_name)
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
    matching = {platform.arch for platform in _matching(base, driver.declaration.supports)}
    for app in apps:
        matching &= {platform.arch for platform in _matching(base, app.declaration.supports)}
    return [arch for arch in base.arch if arch in matching]


def check_app(base: ComputerArtifact, driver: ComputerDriverEnv, app: ComputerAppEnv, at: str = "app") -> list[str]:
    """Every way ``app`` doesn't fit a composition of ``base`` and ``driver``, each named from ``at``: a platform it
    doesn't support, no architecture in common, no install for the base's OS, or a port the driver listens on too."""
    os = base.os
    if not _matching(base, app.declaration.supports):
        named = ", ".join(_shown(platform) for entry in app.declaration.supports for platform in supported(entry))
        return [f"{at}: {app.id} supports no platform {base.id} is ({os.type} {os.distribution} {os.version} "
                f"{os.display}); it supports {named or 'none'}"]
    problems = []
    if not arches(base, driver, [app]):
        problems.append(f"{at}: no architecture {base.id}, {driver.id} and {app.id} all run")
    if os.type not in app.install:
        problems.append(f"{at}: {app.id} declares no install for {os.type}")
    elif taken := sorted(set(_ports(app, os.type)) & set(_driver_ports(driver, os.type))):
        problems.append(f"{at}: {app.id} listens on {', '.join(map(str, taken))}, as {driver.id} does")
    return problems


def folder_name(name: str, fallback: str) -> str:
    """``name``'s last part, lowercased, as a slug: the name of a folder a part is installed in."""
    return re.sub(r"[^a-z0-9-]+", "-", name.rsplit("/", 1)[-1].lower()).strip("-") or fallback


def app_folder(app: ComputerAppEnv) -> str:
    """The name of the folder ``app`` is installed in, under ``APPS``, and of its programs: its environment name's, the
    one name a computer knows it by."""
    return folder_name(app.environment_name, "app")


def _apps(base: ComputerArtifact, driver: ComputerDriverEnv, apps: list[ComputerAppEnv],
          environment_name: str) -> list[str]:
    """Every way ``apps`` don't fit together on ``base`` and ``driver`` in a computer env named ``environment_name``:
    each one's own (``check_app``), no architecture they all run, one listed twice, one named as the computer is, two
    under one environment name, in one folder, with one program name or listening on one port, each pair of apps named
    for the first they'd share, and two keeping their data in one path or one inside the other."""
    problems = [problem for i, app in enumerate(apps) for problem in check_app(base, driver, app, f"apps[{i}]")]
    if not problems and apps and not arches(base, driver, apps):
        problems.append(f"apps: no architecture {base.id}, {driver.id} and every app all run")
    ids = [app.id for app in apps]
    problems += [f"apps: {app_id} is listed twice" for app_id in sorted({i for i in ids if ids.count(i) > 1})]
    distinct, named = list({app.id: app for app in apps}.values()), set()
    problems += [f"apps: {app.id} is named {environment_name!r}, as the computer env is, which names the computer's "
                 "own files" for app in distinct if app.environment_name == environment_name]
    for kind, keys in (("environment name", lambda app: [app.environment_name]),
                       ("folder", lambda app: [f"{APPS}/{app_folder(app)}"]),
                       ("program", lambda app: programs(app, base.os.type)),
                       ("port", lambda app: _ports(app, base.os.type))):
        claims: dict[object, list[str]] = {}
        for app in distinct:
            for key in keys(app):
                claims.setdefault(key, []).append(app.id)
        for key, owners in claims.items():
            if len(owners) > 1 and frozenset(owners) not in named:
                named.add(frozenset(owners))
                problems.append(f"apps: {' and '.join(owners)} would share the {kind} {key!r}")
    owned = [(app, PurePosixPath(path)) for app in distinct for path in _paths(app, base.os.type)]
    problems += [f"apps: {one.id} keeps its data in {str(a)!r} and {other.id} in {str(b)!r}, so they'd share it"
                 for (one, a), (other, b) in itertools.combinations(owned, 2)
                 if one is not other and (a.is_relative_to(b) or b.is_relative_to(a))]
    return problems


def programs(app: ComputerAppEnv, os_type: str) -> list[str]:
    """The names of the supervisord programs ``app`` runs on ``os_type``: its own server's, then its services'."""
    server = f"app-{app_folder(app)}"
    install = app.install.get(os_type)
    return [server, *(f"{server}-{service.name}" for service in (install.services if install is not None else []))]


def _paths(app: ComputerAppEnv, os_type: str) -> list[str]:
    declared = app.declaration.os.get(os_type)
    return list(declared.paths or []) if declared is not None else []


def _ports(app: ComputerAppEnv, os_type: str) -> list[int]:
    install = app.install.get(os_type)
    return list(install.ports or []) if install is not None else []


def _driver_ports(driver: ComputerDriverEnv, os_type: str) -> list[int]:
    install = driver.install.get(os_type)
    return [driver.port, *(install.ports or [] if install is not None else [])]


def check_app_settings(app: ComputerAppEnv, values: dict[str, Any], at: str, os_type: str) -> list[str]:
    """The problems with ``values``, of ``app``'s settings on ``os_type`` by name, each named from ``at``: a setting it
    doesn't declare, a secret's given as anything but a bare reference (or null, where it's optional), a required one
    missing, and what the schema of its ``Settings`` refuses of the rest. A secret's reference never meets the
    constraints on its value, and its value never reaches a message."""
    schema, secrets = app.settings_schema(os_type), app.secret_settings(os_type)
    properties, required = schema.get("properties", {}), schema.get("required", [])
    build = (app.build_settings_schema(os_type) or {}).get("properties", {})
    problems = [f"{at}.{name} is one of {app.id}'s build settings, which only another image changes" if name in build
                else f"{at}.{name}: {app.id} declares no setting {name!r}" for name in values if name not in properties]
    problems += [f"{at}.{name} is a secret, so it takes a {SECRET_REF_PREFIX}KEY or {ENV_REF_PREFIX}NAME reference "
                 "with no ?default, not its value" for name in values if name in secrets
                 and not (_names_a_secret(values[name]) or values[name] is None and _optional(properties[name]))]
    problems += [f"{at}: {name!r} is a required property" for name in required if name in secrets and name not in values]
    plain = {**schema, "properties": {name: setting for name, setting in properties.items() if name not in secrets},
             "required": [name for name in required if name not in secrets]}
    return problems + [f"{'.'.join([at, *map(str, error.absolute_path)])}: {error.message}" for error in
                       Draft202012Validator(plain).iter_errors({k: v for k, v in values.items() if k not in secrets})]


def check_app_build_settings(app: ComputerAppEnv, values: dict[str, Any], at: str, os_type: str | None) -> list[str]:
    """The problems with ``values``, of ``app``'s build settings on ``os_type`` by name, each named from ``at``: a
    setting it doesn't declare as one, a required one missing, and what the schema of its ``BuildSettings`` refuses of
    the rest. With no OS type, its class's."""
    schema = app.build_settings_schema(os_type) or {"type": "object", "properties": {}}
    problems = [f"{at}.{name}: {app.id} declares no build setting {name!r}" for name in values
                if name not in schema["properties"]]
    return problems + [f"{'.'.join([at, *map(str, error.absolute_path)])}: {error.message}"
                       for error in Draft202012Validator(schema).iter_errors(values)]


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
    if not _matching(base, driver.declaration.supports):
        named = ", ".join(_shown(platform) for entry in driver.declaration.supports for platform in supported(entry))
        return [f"driver: {driver.id} supports no platform {base.id} is ({os.type} {os.distribution} {os.version} "
                f"{os.display}); it supports {named}"]
    problems = []
    if not arches(base, driver):
        problems.append(f"driver: no architecture both run: {base.id} runs {', '.join(base.arch)}, and {driver.id} "
                        f"{', '.join(sorted({p.arch for p in _matching(base, driver.declaration.supports)}))}")
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
