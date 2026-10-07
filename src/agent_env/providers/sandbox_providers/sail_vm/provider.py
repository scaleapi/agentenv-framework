"""Sail Research Sailbox VM sandbox provider."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import uuid
from typing import Any, Awaitable, Callable, ClassVar, Self

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, Attribution
from agent_env.config import get_config
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sail_vm import _sdk
from agent_env.providers.sandbox_providers.sail_vm.model_key import ModelKeyInjection
from agent_env.providers.sandbox_providers.sail_vm.sandbox import (
    MAX_ALLOWLIST_ENTRIES,
    SailVmSandbox,
    create_saved_policy,
    egress_document,
    policy_name,
    release_injection,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_VM,
    SandboxProvider,
    apply_default_attribution,
)

logger = logging.getLogger(__name__)

SANDBOX_STARTED_EVENT = "agent_env.sail_vm_sandbox_started"

#: (size, vCPU, (min, max) memory GiB, (min, max) disk GiB), smallest first. Memory and disk are
#: ceilings, not reservations: Sail bills observed usage.
_SIZES: tuple[tuple[str, int, tuple[int, int], tuple[int, int]], ...] = (
    ("s", 1, (2, 64), (8, 128)),
    ("m", 4, (8, 128), (32, 512)),
    ("l", 8, (16, 256), (64, 1024)),
)
_SIZE_NAMES = tuple(size[0] for size in _SIZES)
_LISTENER_TIMEOUT = 60
_LISTENER_POLL_INTERVAL = 1
_MAX_NAME_LENGTH = 128
_REAP_ATTEMPTS = 3

_reapers: set[asyncio.Task] = set()


def sailbox_shape(cpu: float, memory_mb: int, disk_size_gb: float, *, min_size: str = "s") -> tuple[str, int, int]:
    """The smallest size at or above ``min_size`` covering the request, and its memory and disk ceilings
    in GiB, each rounded up to whole GiB and into the size's range."""
    memory_gib = math.ceil(memory_mb / 1024)
    disk_gib = math.ceil(disk_size_gb)
    for name, vcpu, (memory_min, memory_max), (disk_min, disk_max) in _SIZES[_SIZE_NAMES.index(min_size):]:
        if cpu <= vcpu and memory_gib <= memory_max and disk_gib <= disk_max:
            return name, max(memory_gib, memory_min), max(disk_gib, disk_min)
    raise ValueError(
        f"no Sailbox size fits cpu={cpu}, memory={memory_mb}MiB, disk={disk_size_gb}GB "
        "(largest is l: 8 vCPU, 256 GiB memory, 1024 GiB disk)"
    )


def sailbox_name(attribution: Attribution) -> str:
    """``ae-<random>`` plus the attribution values in key order, slugged: for people and ``list(search=)``."""
    slugs = [re.sub(r"[^A-Za-z0-9]+", "-", str(attribution[key])).strip("-") for key in sorted(attribution)]
    return "-".join(["ae", uuid.uuid4().hex[:8], *filter(None, slugs)])[:_MAX_NAME_LENGTH].rstrip("-")


async def _reap(sandbox: SailVmSandbox) -> None:
    for attempt in range(_REAP_ATTEMPTS):
        try:
            await sandbox.terminate()
            logger.info("Terminated Sailbox %s, created after its caller was cancelled", sandbox.sandbox_id)
            return
        except Exception as exc:  # noqa: BLE001 - every failure is retried, then reported
            logger.warning("Terminating orphaned Sailbox %s failed (attempt %s): %s", sandbox.sandbox_id, attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    logger.error(
        "Orphaned Sailbox %s is still running after %s termination attempts; it stops at its max lifetime",
        sandbox.sandbox_id, _REAP_ATTEMPTS,
    )


async def _terminate_named(sdk: Any, app: Any, name: str) -> bool:
    """Terminate every live Sailbox in ``app`` called ``name`` (one a create made though it reported failing),
    retrying through a brief outage; False when that couldn't be confirmed."""
    for attempt in range(_REAP_ATTEMPTS):
        try:
            for box in await sdk.Sailbox.list.aio(app_id=app, search=name):
                if box.name == name and box.status not in ("terminated", "terminating"):
                    await box.terminate.aio()
                    logger.info("Terminated Sailbox %s, created though its create reported a failure", box.sailbox_id)
            return True
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            logger.warning("Checking for a Sailbox %s left by a failed create failed (attempt %s): %s", name, attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    logger.error("A Sailbox %s may be left by a failed create; it stops at its max lifetime", name)
    return False


async def _create_or_reclaim(
    create: Any, wrap: Callable[[Any], SailVmSandbox], release: Callable[[], Awaitable[None]] | None = None,
) -> Any:
    """Await a Sailbox create. If the caller is cancelled first, terminate (``wrap``ped, so its model-key
    policy and secret go too) the Sailbox it yields, which would otherwise keep running with no handle; if
    that create then fails, ``release`` what it was given."""
    task = asyncio.ensure_future(create)

    def terminate_orphan(done: asyncio.Future) -> None:
        if done.cancelled() or done.exception() is not None:
            cleanup = release() if release is not None else None
        else:
            cleanup = _reap(wrap(done.result()))
        if cleanup is None:
            return
        reaper = asyncio.ensure_future(cleanup)
        _reapers.add(reaper)
        reaper.add_done_callback(_reapers.discard)

    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(terminate_orphan)
        raise


class SailVmSandboxProvider(SandboxProvider):
    """Docker-capable Sailboxes. ``api_key`` comes from resolved provider config (a ``secret:`` reference)
    and never reaches a workload. With ``inject_model_key`` (the default) neither does an agent's model key:
    Sail adds it to the agent's requests to the model endpoint (see ``model_key``)."""

    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.sail.box",)

    def __init__(
        self,
        *,
        api_key: str,
        app: str = "agent-env",
        min_size: str = "s",
        auto_sleep: bool = False,
        auto_sleep_min_idle_seconds: int | None = None,
        runtime_threads: int | None = None,
        inject_model_key: bool = True,
        sdk: Any | None = None,
    ):
        self._api_key = api_key
        self._app_name = app
        self._min_size = min_size
        self._auto_sleep = auto_sleep or auto_sleep_min_idle_seconds is not None
        self._auto_sleep_min_idle_seconds = auto_sleep_min_idle_seconds
        self._runtime_threads = runtime_threads
        self._inject_model_key = inject_model_key
        self._sdk = sdk
        self._app: Any | None = None

    def __repr__(self) -> str:
        return f"SailVmSandboxProvider(app={self._app_name!r})"

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        section = "[sandbox.providers.sail_vm.config]"
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigError(f"{section} requires a non-empty 'api_key' (e.g. \"secret:sail_api_key\")")
        app = config.get("app", "agent-env")
        if not isinstance(app, str) or not app.strip():
            raise ConfigError(f"{section} 'app' must be a non-empty string")
        if config.get("min_size", "s") not in _SIZE_NAMES:
            raise ConfigError(f"{section} 'min_size' must be one of {list(_SIZE_NAMES)}")
        for key in ("auto_sleep", "inject_model_key"):
            if not isinstance(config.get(key, False), bool):
                raise ConfigError(f"{section} '{key}' must be true or false")
        for key, (low, high) in {"auto_sleep_min_idle_seconds": (1, 3600), "runtime_threads": (1, 256)}.items():
            value = config.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high):
                raise ConfigError(f"{section} '{key}' must be an integer from {low} to {high}")
        unknown = set(config) - {
            "api_key", "app", "min_size", "auto_sleep", "auto_sleep_min_idle_seconds", "runtime_threads", "inject_model_key",
        }
        if unknown:
            raise ConfigError(f"{section} has unknown key(s): {sorted(unknown)}")
        return cls(**config)

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allowlist of hostnames, IPv4 addresses and IPv4 CIDRs within Sail's entry limit."""
        if not policy.restricts_egress:
            return True
        return (
            len(policy.allow_hosts) + len(policy.allow_cidrs) <= MAX_ALLOWLIST_ENTRIES
            and not any(":" in entry for entry in (*policy.allow_hosts, *policy.allow_cidrs))
        )

    async def _connect(self) -> tuple[Any, Any]:
        if self._app is None:
            self._sdk, self._app = await asyncio.to_thread(
                _sdk.connect, self._api_key, self._app_name, runtime_threads=self._runtime_threads, sdk=self._sdk,
            )
        return self._sdk, self._app

    def _auto_sleep_setting(self, sdk: Any) -> Any:
        if not self._auto_sleep:
            return sdk.AutoSleep.never()
        if self._auto_sleep_min_idle_seconds is not None:
            return sdk.AutoSleep.not_before(self._auto_sleep_min_idle_seconds)
        return sdk.AutoSleep.default()

    async def create_vm(
        self,
        *,
        image: str | None = None,
        boot_mode: str | None = None,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: list[int] | None = None,
        setup_for_gateway: bool = True,
        attribution: Attribution | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> SailVmSandbox:
        """Create a Sailbox from the devbox image; ``timeout`` is its hard maximum lifetime."""
        if image is not None:
            raise ValueError("the Sail provider boots its own Docker-capable image; image overrides are unsupported")
        del boot_mode
        return await self._create(
            cpu=cpu, memory=memory, disk_size_gb=disk_size_gb, timeout=timeout, exposed_ports=exposed_ports,
            setup_for_gateway=setup_for_gateway, attribution=attribution, network_policy=network_policy, injection=None,
        )

    async def _create(
        self,
        *,
        cpu: float,
        memory: int,
        disk_size_gb: float,
        timeout: int,
        exposed_ports: list[int] | None,
        setup_for_gateway: bool,
        attribution: Attribution | None,
        network_policy: NetworkPolicy | None,
        injection: ModelKeyInjection | None,
    ) -> SailVmSandbox:
        size, memory_gib, disk_gib = sailbox_shape(cpu, memory, disk_size_gb, min_size=self._min_size)
        effective_policy = self.effective_network_policy(network_policy)
        if injection is not None:
            effective_policy = effective_policy.with_hosts([injection.host])
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                f"Sail enforces allow-all or up to {MAX_ALLOWLIST_ENTRIES} hostname/IPv4 allowlist entries, "
                f"not {effective_policy.to_dict()}"
            )
        resolved_attribution = {
            key: str(value) for key, value in apply_default_attribution(dict(attribution or {})).items() if value is not None
        }
        ports = list(dict.fromkeys(exposed_ports or []))
        sdk, app = await self._connect()
        egress: Any = egress_document(effective_policy)

        launch = f"launch-{uuid.uuid4().hex}"

        def wrap(raw: Any) -> SailVmSandbox:
            sandbox = SailVmSandbox(
                raw, sdk=sdk, tunnel_urls={}, network_policy=effective_policy, injection=injection,
                refuse_model_keys=self._inject_model_key,
            )
            if injection is not None:
                injection.release(launch)
            return sandbox

        name = sailbox_name(resolved_attribution)

        async def release() -> None:
            """Undo a create that failed or was abandoned: terminate any Sailbox Sail made under ``name`` (its
            response may have been lost), then release the injection's policy."""
            terminated = await _terminate_named(sdk, app, name)
            if injection is not None:
                if terminated:
                    await release_injection(sdk, injection, launch)
                else:
                    injection.release(launch)
                    logger.error("Egress policy %s stays with the unconfirmed Sailbox %s; delete it by name", injection.policy_id, name)

        creating = False
        try:
            if injection is not None:
                injection.hold(launch)
                await sdk.Secret.set.aio(injection.secret, injection.key)
                egress = await create_saved_policy(sdk, policy_name(injection), egress_document(effective_policy, injection))
                injection.policy_id = egress.id
            creating = True
            raw = await _create_or_reclaim(sdk.Sailbox.create.aio(
                app=app,
                image=sdk.Image.devbox("amd64"),
                name=name,
                size=size,
                memory_limit_gib=memory_gib,
                disk_limit_gib=disk_gib,
                max_lifetime_seconds=timeout,
                ingress_ports=ports,
                auto_sleep=self._auto_sleep_setting(sdk),
                egress_policy=egress,
            ), wrap, release)
        except BaseException as exc:
            # A create cancelled in flight is _create_or_reclaim's to clean up once it settles.
            if not (creating and isinstance(exc, asyncio.CancelledError)):
                await release()
            raise
        sandbox = wrap(raw)
        try:
            if raw.status in ("failed", "create_failed"):
                raise RuntimeError(f"Sailbox {raw.sailbox_id} failed to start: {raw.error_message}")
            sandbox.tunnel_urls = await self._tunnel_urls(raw, ports)
            sandbox.mode = SANDBOX_MODE_VM
            logger.info(
                "Sail VM sandbox started: sailbox_id=%s app=%s size=%s memory=%sGiB disk=%sGiB model_key_injected=%s attribution=%s",
                raw.sailbox_id, self._app_name, size, memory_gib, disk_gib, injection is not None, resolved_attribution,
                extra={
                    "event": SANDBOX_STARTED_EVENT,
                    "sail_sailbox_id": raw.sailbox_id,
                    "sail_app_name": self._app_name,
                    "sail_attribution": resolved_attribution,
                    PIPELINE_STEP_KEY: resolved_attribution.get(PIPELINE_STEP_KEY),
                    RUN_ID_KEY: resolved_attribution.get(RUN_ID_KEY),
                    "size": size,
                    "memory_limit_gib": memory_gib,
                    "disk_limit_gib": disk_gib,
                },
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            if injection is not None:
                await sandbox.install_container_trust()
            return sandbox
        except BaseException:
            try:
                await sandbox.terminate()
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not mask the create failure
                logger.warning("Failed to terminate Sailbox %s after setup failure: %s", raw.sailbox_id, cleanup_error)
            raise

    @staticmethod
    async def _tunnel_urls(raw: Any, ports: list[int]) -> dict[int, str]:
        """Each exposed port's public URL, polling until Sail has routed all of them."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _LISTENER_TIMEOUT
        while True:
            urls = {
                listener.guest_port: listener.endpoint.url
                for listener in await raw.listeners.aio()
                if listener.endpoint is not None and getattr(listener.endpoint, "url", None)
            }
            missing = [port for port in ports if port not in urls]
            if not missing:
                return {port: urls[port] for port in ports}
            if loop.time() >= deadline:
                raise RuntimeError(f"Sailbox {raw.sailbox_id} has no public URL for port(s) {missing} after {_LISTENER_TIMEOUT}s")
            await asyncio.sleep(_LISTENER_POLL_INTERVAL)

    async def create_sandbox(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Attribution | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> SailVmSandbox:
        """A bare VM the caller loads and starts ``image_name`` in, as on the other VM providers. A model key in
        ``env`` is injected by Sail rather than passed in (unless ``inject_model_key`` is off)."""
        del image_name
        injection = ModelKeyInjection.for_env(env) if self._inject_model_key else None
        return await self._create(
            cpu=cpu, memory=memory, disk_size_gb=disk_size_gb, timeout=timeout, exposed_ports=[port],
            setup_for_gateway=True, attribution=attribution, network_policy=network_policy, injection=injection,
        )

    async def create_container(self, **kwargs: Any) -> SailVmSandbox:
        """The inherited login-pull-run, then the registry credentials removed from the VM disk, which Sail
        checkpoints for host-failure recovery."""
        sandbox = await super().create_container(**kwargs)
        try:
            await sandbox.exec_script("rm -f /root/.docker/config.json")
        except BaseException:
            await sandbox.terminate()
            raise
        return sandbox

    async def get_sandbox(self, sandbox_id: str) -> SailVmSandbox:
        """Reconnect, restoring ports, the applied egress policy and any model-key injection. The key itself
        is recovered only when it is the configured ``[model]`` key, so the reconnected handle scrubs it too."""
        sdk, _ = await self._connect()
        raw = await sdk.Sailbox.get.aio(sandbox_id)
        tunnel_urls = {
            listener.guest_port: listener.endpoint.url
            for listener in await raw.listeners.aio()
            if listener.endpoint is not None and getattr(listener.endpoint, "url", None)
        }
        applied = getattr(raw, "egress_policy", None)
        injection = ModelKeyInjection.from_document(getattr(applied, "document", None), getattr(applied, "policy_id", None))
        if injection is not None:
            try:
                injection.recover_key(get_config().get_litellm_api_key())
            except Exception:  # noqa: BLE001 - no configured key to recover; the handle just can't scrub it
                pass
        sandbox = SailVmSandbox(
            raw, sdk=sdk, tunnel_urls=tunnel_urls, network_policy=None, injection=injection,
            refuse_model_keys=self._inject_model_key,
        )
        sandbox.network_policy = sandbox.adopt_applied_policy(applied)
        if sandbox.network_policy is None:
            logger.warning(
                "Sailbox %s has an egress policy agent-env can't represent (%r); image loading will fail closed",
                sandbox_id, applied,
            )
        return sandbox
