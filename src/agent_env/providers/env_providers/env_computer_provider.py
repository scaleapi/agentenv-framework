"""The computer environment provider: a computer env's computer, one ready build of it, booted with no gateway in front
on a sandbox provider that runs its arch natively; the driver's card is the env card."""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar, Optional

from agentenv_protocol.computer import COMPUTER_DISPLAY, COMPUTER_RECORDING

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.attribution import Attribution
from agent_env.env.env import DeployedSandboxEnv, EnvCapabilityUnsupported
from agent_env.env.envs.computer.utils import params as computer_params
from agent_env.env.envs.computer.utils.boot import (
    app_settings_environment,
    boot,
    control_environment,
    control_token,
    discard,
    ready,
    ready_timeout,
    recorded_settings,
    settings_environment,
)
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.providers.env_providers.env_provider import DeployRefused, _SandboxEnvironmentProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

if TYPE_CHECKING:
    from agent_env.env.env import Env
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

#: How long a driver may take to end its recordings, uploading the segments still to go.
_END_RECORDINGS_TIMEOUT_S = 300


class EnvironmentComputerProvider(_SandboxEnvironmentProvider):
    """Deploys a computer env with no gateway in front; the driver's own card is the env card."""

    type: ClassVar[str] = "computer"
    record_class = DeployedSandboxEnv

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
    ) -> DeployedSandboxEnv:
        """The env's first ready build, in its base's order of forms and arches (or of ``env_params.form``), that the sandbox
        provider runs natively, booted; returns the record once the driver is ready with the env's display."""
        from agent_env.env.envs.computer.env import ComputerEnv

        if not isinstance(env, ComputerEnv):
            raise TypeError(f"env_provider_type '{self.type}' deploys a computer env, not a {env.type} env")
        if env.computer is None:  # an app that names no computer to deploy alone on
            raise DeployRefused(env.deploy_refusal())
        params, values, problems = computer_params.checked(env, env_params)
        if problems:
            raise DeployRefused(f"{env.id}: " + "; ".join(problems))
        if not (builds := env.ready_builds(params.form)):
            raise DeployRefused(f"{env.id} has no ready build to deploy; its builds are {env.build_statuses() or 'none yet'}")
        attempt = functools.partial(self._boot, env, builds, params, values, ttl_seconds=ttl_seconds,
                                    disk_size_gb=disk_size_gb, cpu=cpu, memory_mb=memory_mb, attribution=attribution)
        return await self._run(sandbox_provider, env.id, attempt)

    async def install_changelog_triggers(self, environment_name: str) -> None:
        """No state store, so no changelog to install."""

    async def _boot(self, env: ComputerEnv, builds: list[tuple[str, str, dict]], params: computer_params.ComputerParams,
                    values: dict, sandbox_provider: SandboxProvider, *, ttl_seconds: int,
                    disk_size_gb: float, cpu: float | None, memory_mb: int | None,
                    attribution: Optional[Attribution]) -> DeployedSandboxEnv:
        """One attempt on one sandbox provider: the first build it runs natively, booted and readied; its record."""
        from agent_env.env.envs.computer.env import ComputerEnv

        native = [(form, arch, build) for form, arch, build in builds
                  if f"linux/{arch}" in sandbox_provider.native_platforms()]
        if not native:
            raise ValueError(f"{type(sandbox_provider).__name__} runs none of {env.id}'s ready builds natively "
                             f"({', '.join(f'{form}/{arch}' for form, arch, _ in builds)})")
        form, arch, build = native[0]
        computer, name, token = env.computer_artifact(), env.environment_name, control_token()
        values = _values(computer, values)
        policy = NetworkPolicy.from_dict(env.network) if env.network else None
        booted = await boot(
            sandbox_provider, image=build["image"], port=env.driver.port, network_policy=policy, token=token,
            environment=_environment(env, computer, values, params), timeout=ttl_seconds, attribution=attribution,
            disk_size_gb=disk_size_gb, cpu=cpu, memory_mb=memory_mb,
            ready_s=ready_timeout(env.composed_apps(), env.computer.os.type))
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
                    "env_params": params.model_dump(exclude_defaults=True)}},
            )
            await _require_display(record, values.get("resolution"))
        except BaseException:
            await discard(booted.sandbox)
            raise
        return record

    async def reboot(self, env: ComputerEnv, deployed: DeployedSandboxEnv) -> None:
        """Replace the computer ``deployed`` records with a fresh container of the image it booted, once its recordings
        ended, its settings applied as that deploy applied them (a generated one minted afresh) and its control token
        the record's, on the same sandbox and port, readied as a deploy readies it."""
        record = deployed.metadata["computer"]
        params, values, _ = computer_params.checked(env, record["env_params"])
        computer = ComputerArtifact.get(record["computer"]["id"], record["computer"]["version"])
        values = _values(computer, values)
        await end_recordings(deployed, "reset")
        sandbox_provider = build_sandbox_provider(deployed.sandbox_type)
        sandbox = await sandbox_provider.get_sandbox(deployed.sandbox_id)
        try:
            await sandbox_provider.reset_container(sandbox, image_name=record["image"], port=env.driver.port,
                                                      env={**_environment(env, computer, values, params),
                                                           **control_environment(deployed.control_token)})
        except NotImplementedError as e:  # a reset the env supports, which this provider can't carry out
            raise RuntimeError(f"'{env.id}' can't be reset on {type(sandbox_provider).__name__}: {e}") from None
        card = await ready(deployed.environment_url, ready_timeout(env.composed_apps(), env.computer.os.type),
                           deployed.control_token)
        _require_name(env, card)
        await _require_display(deployed, values.get("resolution"))


async def end_recordings(record: DeployedSandboxEnv, reason: str) -> None:
    """Have the computer's driver end every recording not yet ended, each uploading what's left and its manifest under
    the grant it was started with, before the computer goes; a driver that doesn't record has none."""
    try:
        await record.invoke(COMPUTER_RECORDING, "end_all", {"reason": reason}, timeout=_END_RECORDINGS_TIMEOUT_S)
    except EnvCapabilityUnsupported:
        return
    except Exception as e:
        logger.warning(f"'{record.env_id}': its recordings weren't ended before its computer went: {e}")


def _values(computer: ComputerArtifact, values: dict) -> dict:
    """The computer's settings a boot applies: each one's default, which carries the env's value, under what
    ``env_params`` set."""
    return {**{key: setting.default for key, setting in computer.config.items()}, **values}


def _environment(env: ComputerEnv, computer: ComputerArtifact, values: dict,
                 params: computer_params.ComputerParams) -> dict[str, str]:
    return {**settings_environment(computer, values, environment_name=env.environment_name,
                                   driver_settings=params.driver_settings),
            **app_settings_environment(env.composed_apps(), params.app_settings, env.computer.os.type)}


def _require_name(env: ComputerEnv, card: dict) -> None:
    if card.get("name") != env.environment_name:
        raise RuntimeError(f"'{env.environment_name}' serves a card named '{card.get('name')}'; a computer env's driver "
                           "must serve its card under the env's environment_name")


async def _require_display(record: DeployedSandboxEnv, resolution: Optional[str]) -> None:
    """Fail unless the driver reports the screen size the boot set, as ``WIDTHxHEIGHT``."""
    if resolution is None:
        return
    display = await record.invoke(COMPUTER_DISPLAY, "get")
    if f"{display['width']}x{display['height']}" != resolution:
        raise RuntimeError(f"'{record.env_id}' booted at {display['width']}x{display['height']}, not the {resolution} "
                           "its display sets")
