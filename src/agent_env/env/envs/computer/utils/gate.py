"""Gate 1, run at a computer env's put before anything is stored or built: its base, driver and apps are matched against
each other and against what the env declares, and every mismatch is named."""

from __future__ import annotations

import functools
import itertools
import re
import zoneinfo
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from agentenv_protocol.computer.apps import apps_folder
from agentenv_protocol.computer.platform import Platform, entry_for, supported
from jsonschema import Draft202012Validator

from agent_env.artifact.artifacts.computer import FORMS, ComputerArtifact, SettingValue
from agent_env.config.loader import ENV_REF_PREFIX, SECRET_REF_PREFIX, is_bare_reference
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.env.envs.computer.utils.install import Install, for_os, platform_of
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


def check_screen(driver: ComputerDriverEnv, apps: Sequence[ComputerAppEnv], at: str = "apps") -> list[str]:
    """Why ``apps`` don't fit ``driver`` when its screen is a website browser's page: that needs exactly one website
    browser among them."""
    if not any(platform.features.get("screen") == "page" for platform in driver.declaration.platforms):
        return []
    browsers = [app.id for app in apps if app.declaration.browser]
    if len(browsers) == 1:
        return []
    return [f"{at}: {driver.id}'s screen is a website browser's page, so its computer takes exactly one website "
            f"browser; it has {len(browsers)}{': ' + ', '.join(browsers) if browsers else ''}"]


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
    doesn't support, no architecture in common, no install for the base's OS, a port the driver listens on too, on
    Omarchy apt's packages (``_apt_on_omarchy``), shared memory, whose size its disk doesn't take yet, or a service as
    root, as its systemd runs every service as the session user, or on Windows an install step for the session user,
    as a Windows computer is composed over ssh as its admin."""
    os = base.os
    if not _matching(base, app.declaration.supports):
        named = ", ".join(_shown(platform) for entry in app.declaration.supports for platform in supported(entry))
        return [f"{at}: {app.id} supports no platform {base.id} is ({_os_shown(base)}); it supports {named or 'none'}"]
    problems = []
    if not arches(base, driver, [app]):
        problems.append(f"{at}: no architecture {base.id}, {driver.id} and {app.id} all run")
    if for_os(app.install, os) is None:
        problems.append(f"{at}: {app.id} declares no install for {_os_shown(base)}")
    elif taken := sorted(set(_ports(app, platform_of(os))) & set(_driver_ports(driver, os))):
        problems.append(f"{at}: {app.id} listens on {', '.join(map(str, taken))}, as {driver.id} does")
    if platform_of(os) == "linux/omarchy":
        problems += _apt_on_omarchy(app.install, os, at, app.id)
        install = for_os(app.install, os)
        if install is not None and install.shm_mb:
            problems.append(f"{at}: {app.id} asks for {install.shm_mb} MiB of /dev/shm, which an Omarchy computer "
                            "doesn't size yet: its /dev/shm is systemd's, sized at boot")
        problems += [f"{at}: {app.id}'s service {service.name!r} runs as root, and {base.id}'s systemd runs every "
                     f"program as {base.session.user}" for service in (install.services if install else [])
                     if service.scope == "system"]
    if os.type == "windows" and (session := [step.name for step in app.declaration.install_steps
                                             if step.scope == "session" and (step.os is None or os.type in step.os)]):
        problems.append(f"{at}: {app.id}'s install steps {', '.join(session)} run as the session user, which a Windows "
                        "computer can't as it's composed, over ssh as its admin; give them scope \"system\" for "
                        "Windows")
    return problems


def folder_name(name: str, fallback: str) -> str:
    """``name``'s last part, lowercased, as a slug: the name of a folder a part is installed in."""
    return re.sub(r"[^a-z0-9-]+", "-", name.rsplit("/", 1)[-1].lower()).strip("-") or fallback


def app_folder(app: ComputerAppEnv) -> str:
    """The name of the folder ``app`` is installed in, under its computer's apps folder (``apps_folder``), and of its
    programs: its environment name's, the one name a computer knows it by."""
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
                       ("folder", lambda app: [str(apps_folder(base.os.type) / app_folder(app))]),
                       ("program", lambda app: programs(app, platform_of(base.os))),
                       ("port", lambda app: _ports(app, platform_of(base.os)))):
        claims: dict[object, list[str]] = {}
        for app in distinct:
            for key in keys(app):
                claims.setdefault(key, []).append(app.id)
        for key, owners in claims.items():
            if len(owners) > 1 and frozenset(owners) not in named:
                named.add(frozenset(owners))
                problems.append(f"apps: {' and '.join(owners)} would share the {kind} {key!r}")
    owned = [(app, PurePosixPath(path)) for app in distinct for path in _paths(app, platform_of(base.os))]
    problems += [f"apps: {one.id} keeps its data in {str(a)!r} and {other.id} in {str(b)!r}, so they'd share it"
                 for (one, a), (other, b) in itertools.combinations(owned, 2)
                 if one is not other and (a.is_relative_to(b) or b.is_relative_to(a))]
    return problems


def programs(app: ComputerAppEnv, platform: str) -> list[str]:
    """The names of the supervisord programs ``app`` runs on ``platform``, an OS type or ``<os>/<distribution>``
    (``install.platform_of``): its own server's, then its services'."""
    server = f"app-{app_folder(app)}"
    install = entry_for(app.install, platform)
    return [server, *(f"{server}-{service.name}" for service in (install.services if install is not None else []))]


def _paths(app: ComputerAppEnv, platform: str) -> list[str]:
    declared = entry_for(app.declaration.os, platform)
    return list(declared.paths or []) if declared is not None else []


def _ports(app: ComputerAppEnv, platform: str) -> list[int]:
    install = entry_for(app.install, platform)
    return list(install.ports or []) if install is not None else []


def _driver_ports(driver: ComputerDriverEnv, os: Any) -> list[int]:
    install = for_os(driver.install, os)
    return [driver.port, *(install.ports or [] if install is not None else [])]


def check_app_settings(app: ComputerAppEnv, values: dict[str, Any], at: str, os_type: str | None,
                       partial: bool = False) -> list[str]:
    """The problems with ``values``, of ``app``'s settings on ``os_type`` by name, each named from ``at``: a setting it
    doesn't declare, a secret's given as anything but a bare reference (or null, where it's optional), a required one
    missing unless ``values`` are ``partial``, defaults a deploy completes, and what the schema of its ``Settings``
    refuses of the rest. A secret's reference never meets the constraints on its value, and its value never reaches a
    message."""
    schema, secrets = app.settings_schema(os_type), app.secret_settings(os_type)
    properties, required = schema.get("properties", {}), [] if partial else schema.get("required", [])
    build = (app.build_settings_schema(os_type) or {}).get("properties", {})
    problems = [f"{at}.{name} is one of {app.id}'s build settings, which only another image changes" if name in build
                else f"{at}.{name}: {app.id} declares no setting {name!r}" for name in values if name not in properties]
    problems += [f"{at}.{name} is a secret, so it takes a {SECRET_REF_PREFIX}KEY or {ENV_REF_PREFIX}NAME reference "
                 "with no ?default, not its value" for name in values if name in secrets
                 and not (is_bare_reference(values[name]) or values[name] is None and _optional(properties[name]))]
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


#: A Windows computer's display: Windows keeps the mode it was installed at, whatever size the launcher's VGA offers.
WINDOWS_DISPLAY = "1920x1080"


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
            if is_bare_reference(value):
                checked[name] = value
            else:
                problems.append(f"computer_settings.{name} is a secret, so it takes a {SECRET_REF_PREFIX}KEY or "
                                f"{ENV_REF_PREFIX}NAME reference with no ?default, not its value")
        elif type(value) is not type(setting.default):
            problems.append(f"computer_settings.{name} must be a {type(setting.default).__name__}, like its default "
                            f"{setting.default!r}, not {value!r}")
        elif base.os.type == "windows" and name == "resolution" and value != WINDOWS_DISPLAY:
            problems.append(f"computer_settings.resolution: a Windows computer's display is {WINDOWS_DISPLAY} for now, "
                            f"not {value}")
        elif name == "timezone" and value not in _time_zones():
            problems.append(f"computer_settings.timezone: {value!r} isn't an IANA time zone, such as UTC or "
                            "America/Chicago")
        else:
            checked[name] = value
    return checked, problems


@functools.cache
def _time_zones() -> frozenset[str]:
    return frozenset(zoneinfo.available_timezones())


def _platform(base: ComputerArtifact, driver: ComputerDriverEnv) -> list[str]:
    os = base.os
    if not _matching(base, driver.declaration.supports):
        named = ", ".join(_shown(platform) for entry in driver.declaration.supports for platform in supported(entry))
        return [f"driver: {driver.id} supports no platform {base.id} is ({_os_shown(base)}); it supports {named}"]
    problems = []
    if not arches(base, driver):
        problems.append(f"driver: no architecture both run: {base.id} runs {', '.join(base.arch)}, and {driver.id} "
                        f"{', '.join(sorted({p.arch for p in _matching(base, driver.declaration.supports)}))}")
    if (install := for_os(driver.install, os)) is None:
        problems.append(f"driver: {driver.id} declares no install for {_os_shown(base)}")
    else:
        form = base.forms.container or base.forms.disk
        init = form.init if form is not None else "launchd"
        problems += [f"driver: its service {service.name!r} runs as root, and {base.id}'s {init} runs every program as "
                     f"{base.session.user}" for service in install.services if service.scope == "system"]
        problems += _apt_on_omarchy(driver.install, os, "driver", driver.id)
    return problems


def _apt_on_omarchy(installs: dict[str, Install], os: Any, at: str, who: str) -> list[str]:
    """Why ``who``'s install, of ``installs``, doesn't fit an Omarchy computer: it's its Linux one, as it declares none
    for linux/omarchy, and names apt's packages or sources, which Omarchy has no apt for, so they'd be left out."""
    install = for_os(installs, os)
    if (platform_of(os) != "linux/omarchy" or "linux/omarchy" in installs or install is None
            or (install.packages is None and not install.sources)):
        return []
    return [f"{at}: {who}'s Linux install names apt packages, which an Omarchy computer has no apt for; declare its "
            "install for linux/omarchy"]


def _matching(base: ComputerArtifact, supports: list) -> list[Platform]:
    """The platforms of ``supports``, as a driver or app declares them, that the base's OS is."""
    os = base.os
    return [platform for entry in supports for platform in supported(entry)
            if platform.os == os.type and platform.display == os.display
            and platform.distribution in (None, getattr(os, "distribution", None))
            and platform.version in (None, os.version)]


def _os_shown(base: ComputerArtifact) -> str:
    return " ".join(part for part in (base.os.type, getattr(base.os, "distribution", None), base.os.version,
                                      base.os.display) if part)


def _shown(platform: Platform) -> str:
    return " ".join(part for part in (platform.os, platform.distribution, platform.version, platform.display,
                                      platform.arch) if part)


def _image(base: ComputerArtifact) -> list[str]:
    if base.forms.container is None:
        return []
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
    if base.os.type == "windows" and f"{display['width']}x{display['height']}" != WINDOWS_DISPLAY:
        return [f"display: a Windows computer's display is {WINDOWS_DISPLAY} for now"]
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
