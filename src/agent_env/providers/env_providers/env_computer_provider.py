"""The computer environment provider: a computer env's computer, one ready build of it, booted with no gateway in front
on a sandbox provider that runs its arch natively; the driver's card is the env card."""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, ClassVar, Optional
from urllib.parse import urlsplit

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import COMPUTER_DISPLAY, COMPUTER_RECORDING, tool_prefix_problem
from agentenv_protocol.computer.app import COMPUTER_APP
from agentenv_protocol.computer.app_runtime import HOLD_BOOT_LAUNCH_ENV
from agentenv_protocol.website_browser import BROWSER, BROWSER_RECORDING, LOADED, start_tab_why
from agentenv_protocol.website_browser.recording import END_S as BROWSER_RECORDING_END_S

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.attribution import Attribution
from agent_env.env.env import DeployedSandboxEnv, EnvCapabilityUnsupported
from agent_env.env.envs.computer.utils import composer
from agent_env.env.envs.computer.utils import params as computer_params
from agent_env.env.envs.computer.utils.install import platform_of
from agent_env.env.envs.computer.utils.probing import probe_first_boot
from agent_env.env.envs.computer.utils import sites
from agent_env.env.envs.computer.utils.boot import (
    app_settings_environment,
    boot,
    control_environment,
    control_token,
    discard,
    first_native,
    listed,
    native_links,
    ready,
    ready_timeout,
    recorded_app_settings,
    recorded_settings,
    settings_environment,
    shm_environment,
    shm_kwargs,
)
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.providers.env_providers.env_provider import DeployRefused, _SandboxEnvironmentProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from agent_env.utils.progress import current_progress

if TYPE_CHECKING:
    from agent_env.env.env import Env
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.providers.sandbox_providers.sandbox import Sandbox
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

#: How long a driver may take to end its recordings, uploading the segments still to go.
_END_RECORDINGS_TIMEOUT_S = 300


class EnvironmentComputerProvider(_SandboxEnvironmentProvider):
    """Deploys a computer env with no gateway in front; the driver's own card is the env card."""

    type: ClassVar[str] = "computer"
    record_class = DeployedSandboxEnv

    def __init__(self) -> None:
        super().__init__()
        self._later: dict[str, tuple[Sandbox, int]] = {}  # each computer booted with sites_later: its sandbox and CA lifetime

    async def deploy(
        self,
        env: Env,
        sandbox_provider: SandboxProvider,
        *,
        ttl_seconds: int = 10800,
        disk_size_gb: float = 10,
        cpu: float | None = None,
        memory_mb: int | None = None,
        attribution: Optional[Attribution] = None,
        env_params: Optional[dict] = None,
        tool_prefix: str = "",
        reach: str = "",
        sites_later: bool = False,
        live_hosts: tuple[str, ...] = (),
    ) -> DeployedSandboxEnv:
        """The env's first ready build, in its base's order of forms and arches (or of ``env_params.form``), that the sandbox
        provider runs natively, booted, once what it boots that no compose started is composed; returns the record once
        the driver is ready with the env's display. With ``tool_prefix``, its driver serves its tools under it; with ``reach``,
        a URL this process reaches, a network that restricts the computer's egress allows its host too, and the
        ``live_hosts`` of the run's live websites. With
        ``sites_later``, a computer routing sites, whose sites aren't up yet, boots with its boot apps held and returns
        once its driver answers; ``route_sites`` routes them and finishes its boot."""
        from agent_env.env.envs.computer.env import ComputerEnv

        if not isinstance(env, ComputerEnv):
            raise TypeError(f"env_provider_type '{self.type}' deploys a computer env, not a {env.type} env")
        if env.computer is None:  # an app that names no computer to deploy alone on
            raise DeployRefused(env.deploy_refusal())
        params, values, problems = computer_params.checked(env, env_params)
        if problem := tool_prefix_problem(tool_prefix):
            problems.append(problem)
        if problems:
            raise DeployRefused(f"{env.id}: " + "; ".join(problems))
        if params.site_routing and not params.sites and not sites_later:
            raise DeployRefused(f"{env.id}: env_params.site_routing reaches the run's websites, which a run's "
                                "deploy_env names in env_params.sites; give them there to deploy it any other way")
        await env.compose_unstarted(sandbox_provider, params.form)
        builds = env.deployable_builds(params.form)
        unprobed = composer.unprobed(env, sandbox_provider, params.form)
        # A ready build only of an arch this provider can't run mustn't hide why its native one is still unprobed.
        if unprobed and not any(native_links(sandbox_provider, arch, form, env.stored_id, env.computer.os.type)
                                for form, arch, _ in builds):
            builds = []
        if not builds:
            why = "".join(f"; {name}/{arch}: {record.get('reason', 'not probed')}" for name, arch, record in unprobed)
            raise DeployRefused(f"{env.id} has no ready build to deploy; its builds are "
                                f"{env.build_statuses() or 'none yet'}{why}; compose it with: agent-env env build --id "
                                f"{env.id}")
        if params.site_routing and not (builds := [build for build in builds if build[0] != "disk"]):
            raise DeployRefused(f"{env.id}: env_params.site_routing installs into the computer's container, and a disk "
                                "computer runs inside a launcher container it can't reach; deploy a container form")
        attempt = functools.partial(self._boot, env, builds, params, values, ttl_seconds=ttl_seconds,
                                    disk_size_gb=disk_size_gb, cpu=cpu, memory_mb=memory_mb, attribution=attribution,
                                    tool_prefix=tool_prefix, reach=reach, sites_later=sites_later,
                                    live_hosts=tuple(live_hosts))
        return await self._run(sandbox_provider, env.id, attempt)

    async def route_sites(self, env: ComputerEnv, record: DeployedSandboxEnv, sites: list[dict]) -> None:
        """Route ``sites`` on the computer ``record`` names, which this provider booted with ``sites_later``, launch its
        held apps and finish its boot as a deploy does; its record then keeps them, so a reset routes them again."""
        sandbox, valid_for_s = self._later.pop(record.sandbox_id)
        computer = record.metadata["computer"]
        params, _, problems = computer_params.checked(env, {**computer["env_params"], "sites": sites})
        if problems:
            raise ValueError(f"{env.id}: " + "; ".join(problems))
        computer["env_params"] = params.model_dump(exclude_defaults=True)
        if routed := await _finish_boot(env, record, sandbox, params, valid_for_s, computer.get("display")):
            computer["sites"] = routed

    async def install_changelog_triggers(self, environment_name: str) -> None:
        """No state store, so no changelog to install."""

    async def _boot(self, env: ComputerEnv, builds: list[tuple[str, str, dict]], params: computer_params.ComputerParams,
                    values: dict, sandbox_provider: SandboxProvider, *, ttl_seconds: int,
                    disk_size_gb: float, cpu: float | None, memory_mb: int | None,
                    attribution: Optional[Attribution], tool_prefix: str = "", reach: str = "",
                    sites_later: bool = False, live_hosts: tuple[str, ...] = ()) -> DeployedSandboxEnv:
        """One attempt on one sandbox provider: the first build it runs natively, booted and readied; its record."""
        from agent_env.env.envs.computer.env import ComputerEnv

        pairs = [(form, arch) for form, arch, _ in builds]
        if (picked := first_native(sandbox_provider, pairs, env.stored_id, env.computer.os.type)) is None:
            raise ValueError(f"{type(sandbox_provider).__name__} runs none of {env.id}'s ready builds natively "
                             f"({', '.join(f'{form}/{arch}' for form, arch, _ in builds)})")
        if env.serves_website_browser() and not sandbox_provider.WEBSOCKETS:
            raise ValueError(f"{type(sandbox_provider).__name__} carries no WebSockets, which {env.id}'s website "
                             "browser's CDP address needs")
        form, arch = picked
        build = next(build for name, at, build in builds if (name, at) == picked)
        if refusal := env.disk_size_refusal(form, disk_size_gb):
            raise ValueError(refusal)
        if form == "vm_bundle":
            await asyncio.to_thread(composer.held_or_composed_here, env, arch, sandbox_provider, current_progress())
        if build["status"] == "unverified":
            build = await probe_first_boot(env, form, arch, sandbox_provider)
            if build["status"] != "ready":
                raise RuntimeError(f"{env.id}'s {form}/{arch} failed its probe on its first boot: {build['reason']}")
        computer, name, token = env.computer_artifact(), env.environment_name, control_token()
        values = _values(computer, values)
        app_settings = recorded_app_settings(env.composed_apps(), computer_params.app_settings(env, params.app_settings),
                                             platform_of(env.computer.os))
        policy = _reaching(NetworkPolicy.from_dict(env.network) if env.network else None, sandbox_provider, reach,
                           live_hosts)
        shm_mb = env.shm_mb()
        booted = await boot(
            sandbox_provider, image=build["image"], port=env.driver.port, network_policy=policy, token=token,
            environment=_environment(env, computer, values, params, tool_prefix, sites_later=sites_later),
            timeout=ttl_seconds,
            attribution=attribution,
            disk_size_gb=disk_size_gb, cpu=cpu, memory_mb=memory_mb,
            ready_s=ready_timeout(env.composed_apps(), platform_of(env.computer.os)), form=form,
            held=bool(params.site_routing), platform=platform_of(env.computer.os), shm_mb=shm_mb)
        self._container_sandboxes.append(booted.sandbox)
        self._environment_sandboxes[name] = booted.sandbox
        try:
            _require_name(env, booted.card)
            record = DeployedSandboxEnv(
                env_id=env.id, env_version=env.version, env_provider_type=self.type,
                environment_card_url=f"{booted.url}{WELL_KNOWN_PATH}", environment_card=booted.card,
                environment_card_read_at_utc=datetime.now(timezone.utc).isoformat(),
                sandbox_id=booted.sandbox.sandbox_id, sandbox_type=booted.sandbox.type,
                sandbox_ids={ComputerEnv.type: {name: booted.sandbox.sandbox_id}}, control_token=token,
                metadata={"computer": {
                    "form": form, "arch": arch, "provider": type(sandbox_provider).__name__, "image": build["image"],
                    "computer": {"id": env.metadata["computer"]["id"], "version": env.metadata["computer"]["version"]},
                    "port": env.driver.port, "display": values.get("resolution"), "network": env.network,
                    "computer_settings": recorded_settings(computer, values),
                    **({"live_hosts": list(live_hosts)} if live_hosts else {}),
                    **({"shm_mb": shm_mb} if shm_mb else {}),
                    **({"app_settings": app_settings} if app_settings else {}),
                    **({"tool_prefix": tool_prefix} if tool_prefix else {}),
                    "env_params": params.model_dump(exclude_defaults=True)}},
            )
            if sites_later:
                self._later[record.sandbox_id] = (booted.sandbox, ttl_seconds)
            elif routed := await _finish_boot(env, record, booted.sandbox, params, ttl_seconds, values.get("resolution")):
                record.metadata["computer"]["sites"] = routed
        except BaseException:
            await discard(booted.sandbox)
            raise
        return record

    async def reboot(self, env: ComputerEnv, deployed: DeployedSandboxEnv) -> None:
        """Replace the computer ``deployed`` records with a fresh container of the image it booted, or its disk booted
        afresh, once its recordings ended, its settings applied as that deploy applied them (a generated one minted
        afresh) and its control token the record's, on the same sandbox and port, readied as a deploy readies it."""
        record = deployed.metadata["computer"]
        params, values, _ = computer_params.checked(env, record["env_params"])
        computer = ComputerArtifact.get(record["computer"]["id"], record["computer"]["version"])
        values = _values(computer, values)
        await end_recordings(deployed, "reset")
        sandbox_provider = build_sandbox_provider(deployed.sandbox_type)
        sandbox = await sandbox_provider.get_sandbox(deployed.sandbox_id)
        shm_mb = record.get("shm_mb")
        environment = {**_environment(env, computer, values, params, record.get("tool_prefix", "")),
                       **control_environment(deployed.control_token),
                       **shm_environment(record.get("form", "container"), env.computer.os.type, shm_mb)}
        try:
            if record.get("form") in ("disk", "vm_bundle"):
                await sandbox_provider.reset_vm(sandbox, image=record["image"], env=environment)
            else:
                await sandbox_provider.reset_container(sandbox, image_name=record["image"], port=env.driver.port,
                                                       env=environment, **shm_kwargs(sandbox_provider, shm_mb))
        except NotImplementedError as e:  # a reset the env supports, which this provider can't carry out
            raise RuntimeError(f"'{env.id}' can't be reset on {type(sandbox_provider).__name__}: {e}") from None
        card = await ready(deployed.environment_url, ready_timeout(env.composed_apps(), platform_of(env.computer.os)),
                           deployed.control_token, answering=bool(params.site_routing), sandbox=sandbox)
        _require_name(env, card)
        valid_for_s = (record.get("sites") or {}).get("valid_for_s", 0)
        if routed := await _finish_boot(env, deployed, sandbox, params, valid_for_s, values.get("resolution")):
            record["sites"] = routed


async def _finish_boot(env: ComputerEnv, record: DeployedSandboxEnv, sandbox: Sandbox,
                       params: computer_params.ComputerParams, valid_for_s: int,
                       resolution: Optional[str]) -> Optional[dict]:
    """What a deploy and a reset do once the computer's driver answers, in order: with ``site_routing``, the sites
    routed, each browser handed their names and the held boot apps launched, the computer then ready; then its display
    checked; then each browser's start tabs read into the record, the boot failing on one on a site of the env's that
    didn't load. What the record keeps of the routing, or None without it."""
    routed = None
    if params.site_routing:
        routed = await _route_sites(env, sandbox, params, valid_for_s,
                                    tuple(record.metadata["computer"].get("live_hosts") or ()))
        await _launch_held_apps(env, record, record.environment_url, params.sites)
    await _require_display(record, resolution)
    if start_tabs := await _start_tabs(record):
        record.metadata["computer"]["start_tabs"] = start_tabs
        _refuse_failed_start_tabs(record, start_tabs)
    return routed


async def _route_sites(env: ComputerEnv, sandbox: Sandbox, params: computer_params.ComputerParams,
                       valid_for_s: int, live_hosts: tuple[str, ...] = ()) -> dict:
    """The computer on ``sandbox`` routed to ``params.sites`` under a CA of its own, valid ``valid_for_s``, a browser
    alone's forward port taking ``live_hosts`` straight to them; what its record keeps of it."""
    routed = await sites.route(sandbox, params.sites, user=env.computer.session.user, network=env.network,
                               valid_for=timedelta(seconds=valid_for_s), label=env.environment_name,
                               os_type=env.computer.os.type, display=env.computer.os.display, live_hosts=live_hosts)
    return {**routed, "valid_for_s": valid_for_s}


async def _launch_held_apps(env: ComputerEnv, record: DeployedSandboxEnv, url: str,
                            routed: list[computer_params.Site]) -> None:
    """Launch each app that starts with the computer, whose launch a deploy routing sites holds until the sites' names,
    CA and proxy are in place, through its ``computer-app/v1`` ``launch`` once the computer lists it with its server, a
    website browser first handed the ``routed`` sites, then wait until the computer is ready, which its boot only
    waited to answer; the record keeps the card the computer answers then, as its boot's card may list an app by its
    name alone. Each is launched as the child its card lists, as a computer named for its app, a browser alone, has
    that name too."""
    platform = platform_of(env.computer.os)
    timeout = ready_timeout(env.composed_apps(), platform)
    held = [app for app in env.composed_apps() if (launch := app.launch(platform)) is not None and launch.at == "boot"]
    launchers: dict[str, protocol_v1.ChildEnv] = {}
    if held:
        record.environment_card = await listed(url, [app.environment_name for app in held], timeout)
        launchers = {child.name: child for child in protocol_v1.children(record.environment_card)
                     if protocol_v1.find_extension_method(child.card, COMPUTER_APP, "launch") is not None}
    for app in held:
        await _hand_sites(record, launchers[app.environment_name], routed)
        await record.invoke(COMPUTER_APP, "launch", child=launchers[app.environment_name])
    record.environment_card = await ready(url, timeout, record.control_token)
    record.environment_card_read_at_utc = datetime.now(timezone.utc).isoformat()


async def _hand_sites(record: DeployedSandboxEnv, child: protocol_v1.ChildEnv,
                      routed: list[computer_params.Site]) -> None:
    """The ``routed`` sites, by name and domains, handed to the website browser ``child`` is, through ``browser/v1``'s
    ``sites/set``, so its start tabs on them must load; a browser built before it took them is told nothing, its start
    tabs on them waited on as any live one's."""
    if child.find_extension(BROWSER) is None:
        return
    try:
        await record.invoke(BROWSER, "sites/set", {"sites": [{"name": site.name, "domains": site.domains}
                                                            for site in routed if site.domains]}, child=child)
    except EnvCapabilityUnsupported:
        logger.warning(f"the website browser {child.name} in {record.env_id} serves no browser/v1 sites/set, so its "
                       "start tabs on the env's sites are waited on as live ones")


async def _start_tabs(record: DeployedSandboxEnv) -> dict[str, list[dict]]:
    """Each website browser's start tabs as its latest launch loaded them, by its name: the computer's own card's, for
    a browser alone, or a child's; a browser that keeps none, built before it did, left out."""
    card = record.environment_card or {}
    browsers = ([(card.get("name") or record.env_id, None)] if protocol_v1.find_extension(card, BROWSER) else []) \
        + [(child.name, child) for child in protocol_v1.children(card) if child.find_extension(BROWSER) is not None]
    kept: dict[str, list[dict]] = {}
    for name, child in browsers:
        try:
            launches = (await record.invoke(BROWSER, "record", child=child)).get("start_tabs") or []
        except EnvCapabilityUnsupported:
            continue
        if launches:
            kept[name] = launches[-1]["tabs"]
    return kept


def _refuse_failed_start_tabs(record: DeployedSandboxEnv, start_tabs: dict[str, list[dict]]) -> None:
    """RuntimeError naming each of ``start_tabs``, by browser, on one of the env's sites that didn't load."""
    if failed := [f"{name}'s start tab {tab['url']} on the env's site {tab['site']} {start_tab_why(tab)}"
                  for name, tabs in start_tabs.items() for tab in tabs
                  if tab["site"] is not None and tab["outcome"] != LOADED]:
        raise RuntimeError(f"computer env {record.env_id!r}: " + "; ".join(failed))


async def end_recordings(record: DeployedSandboxEnv, reason: str) -> None:
    """Have the computer's driver, and each browser app on it that records, end every recording not yet ended, each
    uploading what's left and its manifest under the grant it was started with, before the computer goes; a driver or
    app that doesn't record has none."""
    try:
        await record.invoke(COMPUTER_RECORDING, "end_all", {"reason": reason}, timeout=_END_RECORDINGS_TIMEOUT_S)
    except EnvCapabilityUnsupported:
        pass
    except Exception as e:
        logger.warning(f"'{record.env_id}': its recordings weren't ended before its computer went: {e}")
    for child in (record.environment_card or {}).get("children_environments") or []:
        if protocol_v1.find_extension(child, BROWSER_RECORDING) is None:
            continue
        try:
            await record.invoke(BROWSER_RECORDING, "end_all", {"reason": reason}, environment_name=child.get("name"),
                                timeout=max(_END_RECORDINGS_TIMEOUT_S, BROWSER_RECORDING_END_S))
        except Exception as e:
            logger.warning(f"'{record.env_id}': {child.get('name')}'s browser recordings weren't ended before its "
                           f"computer went: {e}")


def _reaching(policy: Optional[NetworkPolicy], sandbox_provider: SandboxProvider, url: str,
              live_hosts: tuple[str, ...] = ()) -> Optional[NetworkPolicy]:
    """``policy``, also allowing ``url``'s host as the sandbox provider's sandboxes reach it, and ``live_hosts``, when
    it lists the hosts allowed; a network that allows every host is left as it is."""
    host = urlsplit(sandbox_provider.url_from_sandbox(url)).hostname if url else None
    hosts = [*([host] if host else []), *live_hosts]
    return policy.with_hosts(hosts) if policy is not None and hosts else policy


def _values(computer: ComputerArtifact, values: dict) -> dict:
    """The computer's settings a boot applies: each one's default, which carries the env's value, under what
    ``env_params`` set."""
    return {**{key: setting.default for key, setting in computer.config.items()}, **values}


def _environment(env: ComputerEnv, computer: ComputerArtifact, values: dict,
                 params: computer_params.ComputerParams, tool_prefix: str = "", *,
                 sites_later: bool = False) -> dict[str, str]:
    return {**settings_environment(computer, values, environment_name=env.computer_name,
                                   driver_settings=params.driver_settings, tool_prefix=tool_prefix),
            **app_settings_environment(env.composed_apps(), computer_params.app_settings(env, params.app_settings),
                                       platform_of(env.computer.os)),
            **({HOLD_BOOT_LAUNCH_ENV: "1", **sites.trust_environment(env.computer.os.type, params.sites,
                                                                     later=sites_later),
                **sites.browser_environment(env.computer.os.display)} if params.site_routing else {})}


def _require_name(env: ComputerEnv, card: dict) -> None:
    if card.get("name") != env.computer_name:
        raise RuntimeError(f"'{env.environment_name}' serves a card named '{card.get('name')}'; a computer env's driver "
                           f"must serve its card as {env.computer_name!r}")


async def _require_display(record: DeployedSandboxEnv, resolution: Optional[str]) -> None:
    """Fail unless the driver reports the screen size the boot set, as ``WIDTHxHEIGHT``."""
    if resolution is None:
        return
    display = await record.invoke(COMPUTER_DISPLAY, "get")
    if f"{display['width']}x{display['height']}" != resolution:
        raise RuntimeError(f"'{record.env_id}' booted at {display['width']}x{display['height']}, not the {resolution} "
                           "its display sets")
