"""Booting a composed computer: one build of it in a container, on a sandbox provider that runs its arch natively, with
its settings in the environment the base and the driver read them from, until its driver says it's ready. A deploy and
gate 3 boot computers this way."""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import APP_SETTINGS_ENV, COMPUTER_READY, SETTINGS_ENV
from agentenv_protocol.computer.apps import ANSWER_TIMEOUT_S

from agent_env.config import get_config
from agent_env.providers.env_providers.env_provider import _size_kwargs
from agent_env.providers.sandbox_providers.local_docker.provider import remove_local_work_dir

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.computer import ComputerArtifact
    from agent_env.attribution import Attribution
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, Sandbox
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

#: How long a computer may take from its container starting to its driver saying it's ready.
READY_TIMEOUT_S = 180.0
#: How much longer than the slowest starting app's own readiness timeout a computer may take, for the rest of it first.
APP_READY_MARGIN_S = 30.0
#: How long one computer-ready status may take: the driver's own checks, then every app's server at once.
_STATUS_TIMEOUT_S = int(ANSWER_TIMEOUT_S) + 15
_POLL_S = 0.5
#: How a setting a record keeps shows a secret's or a generated one's value, which it never holds.
WITHHELD = "<withheld>"


@dataclass
class Booted:
    sandbox: Sandbox
    url: str  # the driver's address, which its card's paths join onto
    card: dict


def native_links(sandbox_provider: SandboxProvider, arch: str) -> list[SandboxProvider]:
    """The providers a boot of ``arch`` may run on, in the order a chain tries them: those that run it natively."""
    return [link for link in sandbox_provider.links if f"linux/{arch}" in link.native_platforms()]


def settings_environment(computer: ComputerArtifact, values: dict[str, Any], *, environment_name: str,
                         driver_settings: Optional[dict] = None) -> dict[str, str]:
    """The environment a boot of ``computer`` gives it: each deploy-phase setting as ``AGENTENV_SETTING_<NAME>``, from
    ``values`` over its default, minted afresh when it's generated and resolved through the secret store when it's a
    secret; ``driver_settings``, when there are any, in ``AGENTENV_COMPUTER_SETTINGS``; and the name its card is served
    under."""
    environment = {"ENVIRONMENT_NAME": environment_name}
    for name, setting in computer.config.items():
        if setting.phase != "deploy":
            continue
        value = secrets.token_urlsafe(24) if setting.generate else values.get(name, setting.default)
        if setting.secret and not setting.generate:
            value = get_config().resolve_reference(value)
        environment[f"AGENTENV_SETTING_{name.upper()}"] = str(value)
    if driver_settings:
        environment[SETTINGS_ENV] = json.dumps(driver_settings)
    return environment


def app_settings_environment(apps: list[ComputerAppEnv], app_settings: dict[str, dict],
                             os_type: str) -> dict[str, str]:
    """The environment that hands each of ``apps``, on a computer of ``os_type``, the values ``app_settings`` gives it,
    by its name, in ``APP_SETTINGS_ENV``, a secret's resolved through the secret store; nothing when none has any."""
    given = [(app, app_settings[app.environment_name]) for app in apps if app_settings.get(app.environment_name)]
    if not given:
        return {}
    return {APP_SETTINGS_ENV: json.dumps({app.environment_name: {
        name: get_config().resolve_reference(value) if name in app.secret_settings(os_type) and value is not None
        else value
        for name, value in values.items()} for app, values in given})}


def recorded_settings(computer: ComputerArtifact, values: dict[str, Any]) -> dict[str, Any]:
    """The deploy-phase settings a boot applies as a record keeps them, a secret's and a generated one's withheld."""
    return {name: WITHHELD if setting.secret or setting.generate else values.get(name, setting.default)
            for name, setting in computer.config.items() if setting.phase == "deploy"}


async def boot(sandbox_provider: SandboxProvider, *, image: str, port: int, environment: dict[str, str],
               network_policy: Optional[NetworkPolicy], timeout: int, attribution: Optional[Attribution],
               disk_size_gb: float = 10, cpu: Optional[float] = None, memory_mb: Optional[int] = None,
               ready_s: float = READY_TIMEOUT_S) -> Booted:
    """``image`` booted with ``environment`` on ``sandbox_provider``, which the caller chose to run its arch natively,
    its driver's ``port`` reached on the sandbox, once the driver says the computer is ready; a boot that fails is torn
    down."""
    sandbox = await sandbox_provider.create_container(
        image_name=image, port=port, env=environment, network_policy=network_policy,
        disk_size_gb=disk_size_gb, timeout=timeout, attribution=dict(attribution or {}), **_size_kwargs(cpu, memory_mb))
    try:
        url = sandbox.tunnel_urls[port]
        return Booted(sandbox, url, await ready(url, ready_s))
    except BaseException:
        await discard(sandbox)
        raise


async def discard(sandbox: Sandbox) -> None:
    """Tear down a sandbox no run records, as a run's teardown would: terminate it, then remove its work folder when it
    ran on this machine."""
    try:
        await sandbox.terminate()
    except Exception:
        return
    if sandbox.ON_THIS_MACHINE:
        with contextlib.suppress(OSError):
            await asyncio.to_thread(remove_local_work_dir, sandbox.sandbox_id)


def ready_timeout(apps: list[ComputerAppEnv], os_type: str) -> float:
    """How long a computer of ``os_type`` and ``apps`` may take to be ready: ``READY_TIMEOUT_S``, or longer when an app
    that starts with it there gives itself longer."""
    return max([READY_TIMEOUT_S, *(launch.ready.timeout_s + APP_READY_MARGIN_S for app in apps
                                   if (launch := app.launch(os_type)) and launch.at == "boot" and launch.ready)])


async def ready(url: str, timeout: float) -> dict:
    """The card of the driver at ``url``, read once its ``computer-ready`` status says it's ready, so each app it fronts
    is on it as the app's own server serves it; raises TimeoutError, naming what it last saw, when it isn't within
    ``timeout`` seconds."""
    last = "nothing answered"
    try:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    card = await protocol_v1.get_card(url, timeout=5)
                    status = await protocol_v1.invoke_extension(url, card, COMPUTER_READY, None, _STATUS_TIMEOUT_S,
                                                                method="status")
                    if status.get("ready"):
                        return await protocol_v1.get_card(url, timeout=5)
                    last = "not ready: " + "; ".join(f"{check['name']}: {check.get('detail')}"
                                                     for check in status.get("checks", []) if not check.get("ok"))
                except Exception as e:
                    last = f"{type(e).__name__}: {e}"
                await asyncio.sleep(_POLL_S)
    except TimeoutError:
        raise TimeoutError(f"the computer at {url} wasn't ready within {timeout:.0f} s ({last})") from None
