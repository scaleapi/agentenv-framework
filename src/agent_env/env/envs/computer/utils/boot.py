"""Booting a composed computer: one build of it, its container in a container or its disk in a VM, on a sandbox provider
that runs its arch natively, with its settings in the environment the base and the driver read them from (a disk's in
its NoCloud seed), until its driver says it's ready. A deploy and gate 3 boot computers this way."""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agentenv_protocol import CONTROL_TOKEN_ENV
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import APP_SETTINGS_ENV, COMPUTER_READY, SETTINGS_ENV
from agentenv_protocol.computer.app import COMPUTER_APP
from agentenv_protocol.computer.apps import ANSWER_TIMEOUT_S

from agent_env.config import get_config
from agent_env.env.envs.computer.utils.composed import REPOSITORY
from agent_env.providers.env_providers.env_provider import _size_kwargs
from agent_env.providers.sandbox_providers.local_docker.provider import remove_local_work_dir
from agent_env.store.image_store.oci_registry_credentials import is_loopback_host, registry_host_from_ref

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
#: How long one read of the driver's card may take.
_CARD_TIMEOUT_S = 5
#: How a setting a record keeps shows a secret's or a generated one's value, which it never holds.
WITHHELD = "<withheld>"


class ComputerFailed(RuntimeError):
    """The computer's ``computer-ready`` status says it failed for good, a check that can't pass again."""
#: How many random bytes a computer's control token carries: 256 bits, past guessing.
CONTROL_TOKEN_BYTES = 32
#: The size of a disk's VM when a deploy sets none: what the desktop was measured booting and passing the probe in.
DISK_CPU, DISK_MEMORY_MB = 2.0, 4096


@dataclass
class Booted:
    sandbox: Sandbox
    url: str  # the driver's address, which its card's paths join onto
    card: dict


def native_links(sandbox_provider: SandboxProvider, arch: str, form: str, store_id: str) -> list[SandboxProvider]:
    """The providers a boot of ``form`` for ``arch`` of the env stored as ``store_id`` may run on, in the order a chain
    tries them: those that boot its disk natively, which they carry themselves, or run its container natively and pull
    it from the env's image store."""
    if form == "disk":
        return [link for link in sandbox_provider.links if f"linux/{arch}" in link.disk_image_platforms()]
    if form == "vm_bundle":
        return [link for link in sandbox_provider.links if f"macos/{arch}" in link.vm_bundle_platforms()]
    return [link for link in sandbox_provider.links if f"linux/{arch}" in link.native_platforms()
            and pulls_containers(link, store_id)]


def first_native(link: SandboxProvider, pairs: Iterable[tuple[str, str]], store_id: str) -> tuple[str, str] | None:
    """The first of ``pairs``, forms and arches in the base's order, that ``link`` boots natively: the one a deploy on it
    boots, and so the one a run composes for it."""
    return next(((form, arch) for form, arch in pairs if native_links(link, arch, form, store_id)), None)


def pulls_containers(link: SandboxProvider, store_id: str) -> bool:
    """Whether ``link`` pulls the containers the env stored as ``store_id`` composes: from a registry on this machine's
    loopback, only a link on this machine does."""
    ref = get_config().get_image_store_for(store_id).image_ref(REPOSITORY, "key")
    return link.ON_THIS_MACHINE or not is_loopback_host(registry_host_from_ref(ref))


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


def recorded_app_settings(apps: list[ComputerAppEnv], app_settings: dict[str, dict],
                          os_type: str) -> dict[str, dict[str, Any]]:
    """The values ``app_settings`` hands each of ``apps`` on a computer of ``os_type``, by its name, as a record keeps
    them, a secret's withheld; its class's defaults fill in the rest."""
    return {app.environment_name: {name: WITHHELD if name in app.secret_settings(os_type) else value
                                   for name, value in values.items()}
            for app in apps if (values := app_settings.get(app.environment_name))}


def recorded_settings(computer: ComputerArtifact, values: dict[str, Any]) -> dict[str, Any]:
    """The deploy-phase settings a boot applies as a record keeps them, a secret's and a generated one's withheld."""
    return {name: WITHHELD if setting.secret or setting.generate else values.get(name, setting.default)
            for name, setting in computer.config.items() if setting.phase == "deploy"}


def control_token() -> str:
    """A token for one computer's framework-only routes (``CONTROL_TOKEN_ENV``), which agent-env holds and gives no
    agent."""
    return secrets.token_urlsafe(CONTROL_TOKEN_BYTES)


def control_environment(token: Optional[str]) -> dict[str, str]:
    """What hands a computer its control ``token`` in its boot environment; nothing without one, as for a computer an
    older agent-env booted."""
    return {CONTROL_TOKEN_ENV: token} if token else {}


async def boot(sandbox_provider: SandboxProvider, *, image: str, port: int, environment: dict[str, str], token: str,
               network_policy: Optional[NetworkPolicy], timeout: int, attribution: Optional[Attribution],
               disk_size_gb: float = 10, cpu: Optional[float] = None, memory_mb: Optional[int] = None,
               ready_s: float = READY_TIMEOUT_S, form: str = "container", held: bool = False) -> Booted:
    """``image``, of ``form``, booted with ``environment`` and its control ``token`` on ``sandbox_provider``, which the
    caller chose to run its arch natively, its driver's ``port`` reached on the sandbox, once the driver, asked with that
    token, says the computer is ready, or with ``held``, its boot apps' launch held, once it answers (``ready``); a boot
    that fails is torn down."""
    environment = {**environment, **control_environment(token)}
    if form == "vm_bundle":
        sandbox = await sandbox_provider.create_vm(
            image=image, env=environment, exposed_ports=[port], network_policy=network_policy,
            disk_size_gb=disk_size_gb, timeout=timeout, attribution=dict(attribution or {}), cpu=cpu, memory=memory_mb)
    elif form == "disk":
        sandbox = await sandbox_provider.create_vm(
            image=image, env=environment, exposed_ports=[port], network_policy=network_policy,
            disk_size_gb=disk_size_gb, timeout=timeout, attribution=dict(attribution or {}),
            cpu=cpu or DISK_CPU, memory=memory_mb or DISK_MEMORY_MB)
    else:
        sandbox = await sandbox_provider.create_container(
            image_name=image, port=port, env=environment, network_policy=network_policy, disk_size_gb=disk_size_gb,
            timeout=timeout, attribution=dict(attribution or {}), **_size_kwargs(cpu, memory_mb))
    try:
        url = sandbox.tunnel_urls[port]
        return Booted(sandbox, url, await ready(url, ready_s, token, answering=held))
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


async def ready(url: str, timeout: float, token: Optional[str] = None, *, answering: bool = False) -> dict:
    """The card of the driver at ``url``, read once its ``computer-ready`` status, asked with its control ``token``,
    says it's ready, so each app it fronts is on it as the app's own server serves it, or with ``answering``, once it
    answers at all: a computer whose boot apps' launch is held can't be ready before they launch, as a page driver's
    screen is its browser's page. Raises ComputerFailed as soon as the status says it failed, and TimeoutError, naming
    what it last saw, when it isn't ready within ``timeout`` seconds."""
    last, failed = "nothing answered", None
    try:
        async with asyncio.timeout(timeout):
            while failed is None:
                try:
                    card = await protocol_v1.get_card(url, timeout=_CARD_TIMEOUT_S)
                    status = await protocol_v1.invoke_extension(url, card, COMPUTER_READY, None, _STATUS_TIMEOUT_S,
                                                                method="status",
                                                                headers=protocol_v1.control_headers(token))
                    if status.get("ready") or (answering and not status.get("failed")):
                        return await protocol_v1.get_card(url, timeout=_CARD_TIMEOUT_S)
                    failed = status.get("failed")
                    last = "not ready: " + "; ".join(f"{check['name']}: {check.get('detail')}"
                                                     for check in status.get("checks", []) if not check.get("ok"))
                except Exception as e:
                    last = f"{type(e).__name__}: {e}"
                if failed is None:
                    await asyncio.sleep(_POLL_S)
    except TimeoutError:
        raise TimeoutError(f"the computer at {url} wasn't ready within {timeout:.0f} s ({last})") from None
    raise ComputerFailed(f"the computer at {url} failed: {failed}")


async def listed(url: str, apps: list[str], timeout: float) -> dict:
    """The card of the driver at ``url`` once it lists each of ``apps`` with its own server's card, which offers
    ``computer-app/v1`` ``launch``: a driver names an app whose server isn't answering yet by its name alone. Raises
    TimeoutError, naming the apps it last lacked, when that takes longer than ``timeout`` seconds."""
    lacking = list(apps)
    try:
        async with asyncio.timeout(timeout):
            while True:
                with contextlib.suppress(Exception):
                    card = await protocol_v1.get_card(url, timeout=_CARD_TIMEOUT_S)
                    offered = {child.name for child in protocol_v1.children(card)
                               if protocol_v1.find_extension_method(child.card, COMPUTER_APP, "launch") is not None}
                    lacking = [app for app in apps if app not in offered]
                    if not lacking:
                        return card
                await asyncio.sleep(_POLL_S)
    except TimeoutError:
        raise TimeoutError(f"the computer at {url} didn't list {', '.join(lacking)} with its server within "
                           f"{timeout:.0f} s") from None
