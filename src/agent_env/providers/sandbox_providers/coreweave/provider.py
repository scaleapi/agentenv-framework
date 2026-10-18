"""CoreWeave Sandbox provider.

Each sandbox is one CoreWeave Sandbox. A container (``create_container``, ``create_sandbox``) runs the workload's image
itself. A VM (``create_vm``) is a privileged container in a Kata VM, booted from a Docker host image, where the gateway's
compose, and the containers steps start, run under Docker through the inherited ``VmSandbox`` helpers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Self
from urllib.parse import urlparse

import cwsandbox
from cwsandbox import (
    AuthHeaders,
    Container,
    Endpoint,
    EndpointAuth,
    EndpointKind,
    ResourceOptions,
    SandboxStatus,
    SecurityContext,
    Service,
    ServiceVisibility,
)

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.attribution import Attribution
from agent_env.config import get_config
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.coreweave.sandbox import (
    DOCKER_STORAGE_LOOP,
    DOCKER_STORAGE_MEMORY,
    Client,
    CoreWeaveContainerSandbox,
    CoreWeaveSandbox,
    host_setup_script,
    network_options,
    policy_from_egress,
    runs_docker_host,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import (
    Accepts,
    SandboxNotFoundError,
    SandboxProvider,
    apply_default_attribution,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "public.ecr.aws/docker/library/docker:dind"
DEFAULT_RUNTIME_CLASS = "kata-qemu"
SANDBOX_TAG = "agentenv"
_SECTION = "[sandbox.providers.coreweave.config]"
_STRING_KEYS = ("api_key", "wandb_api_key", "wandb_entity", "wandb_project", "base_url", "image", "runtime_class",
                "placement_mode")
_CONFIG_KEYS = frozenset({*_STRING_KEYS, "docker_storage", "runner_ids"})

# Docker keeps both the compressed and the extracted layers, so 10 GB runs out once a gateway and a few MCP server
# images are on the sandbox. A loop-mounted disk is a sparse file, so only what Docker writes is used.
MIN_DOCKER_DISK_GB = 20
# The platform closes an HTTPS request to an exposed port after this many seconds, its maximum; the default, 15, is
# shorter than an agent's turn.
ENDPOINT_REQUEST_TIMEOUT_SEC = 900
# A sandbox reconnected to has no lifetime agent-env can read, so its commands get this budget.
_RECONNECT_COMMAND_TIMEOUT_SEC = 2 * 3600
_MAX_TAG_LENGTH = 59
_REAP_ATTEMPTS = 3
# How long a failed or cancelled create waits for its start request to answer, so it learns the sandbox's id, before it
# looks for the sandbox by its tag instead.
_START_SETTLE_SEC = 30
# The Sandbox API keeps a deleted sandbox's record, so one draining or ended is as gone as one it no longer knows.
_GONE_STATUSES = frozenset({SandboxStatus.TERMINATING, SandboxStatus.COMPLETED, SandboxStatus.FAILED,
                            SandboxStatus.TERMINATED})

# An exposed port's URL is <port>-<sandbox id>.<runner label>.<domain>, and a CoreWeave wildcard covers exactly one
# label, so no one entry covers every runner. The parents seen on exposed ports are added to later allowlists, so a
# restricted agent reaches environments served there.
_endpoint_parents: set[str] = set()
_MAX_ENDPOINT_PARENTS = 16


def resources(cpu: float, memory_mb: int) -> ResourceOptions:
    """Requests and equal limits, so the sandbox gets exactly ``cpu`` (in millicores, rounded up) and ``memory_mb``."""
    if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0:
        raise ValueError("cpu must be a finite number greater than zero")
    if isinstance(memory_mb, bool) or not isinstance(memory_mb, int) or memory_mb <= 0:
        raise ValueError("memory must be a positive integer number of MB")
    shape = {"cpu": f"{math.ceil(cpu * 1000)}m", "memory": f"{memory_mb}Mi"}
    return ResourceOptions(requests=shape, limits=dict(shape))


def docker_disk_gb(disk_size_gb: float) -> int:
    if (isinstance(disk_size_gb, bool) or not isinstance(disk_size_gb, (int, float))
            or not math.isfinite(disk_size_gb) or disk_size_gb <= 0):
        raise ValueError("disk_size_gb must be a finite number greater than zero")
    return max(MIN_DOCKER_DISK_GB, math.ceil(disk_size_gb))


def _tag(text: str) -> str:
    """``text`` as a CoreWeave tag: letters, digits, ``.``, ``_`` and ``-``, at most 59, ending with a letter or digit."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)[:_MAX_TAG_LENGTH].rstrip("._-")


def sandbox_tags(attribution: Attribution, create_tag: str) -> list[str]:
    """``agentenv``, the create's unique tag, and one ``key.value`` tag per attribution entry."""
    tags = [SANDBOX_TAG, create_tag]
    tags += [tag for key in sorted(attribution) if (tag := _tag(f"{key}.{attribution[key]}"))]
    return list(dict.fromkeys(tags))


def services(ports: list[int]) -> list[Service]:
    endpoint = Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN,
                        request_timeout_seconds=ENDPOINT_REQUEST_TIMEOUT_SEC)
    return [Service(port=port, visibility=ServiceVisibility.PUBLIC, endpoint=endpoint) for port in ports]


def tunnel_urls(raw: Any) -> dict[int, str]:
    urls = {endpoint.port: endpoint.url for endpoint in raw.service_endpoints if endpoint.url}
    for url in urls.values():
        parent = (urlparse(url).hostname or "").partition(".")[2]
        if parent and len(_endpoint_parents) < _MAX_ENDPOINT_PARENTS:
            _endpoint_parents.add(f"*.{parent}")
    return urls


def auth_headers(config: dict[str, Any]) -> AuthHeaders:
    """The headers the Sandbox API authenticates: a CoreWeave API key, or a W&B API key with an optional entity and
    project. Built here rather than by the SDK, which would read the process environment or W&B's settings files."""
    if config.get("api_key"):
        return AuthHeaders(headers={"Authorization": f"Bearer {config['api_key']}"}, strategy="api_key")
    headers = {"x-wandb-api-key": config["wandb_api_key"]}
    if config.get("wandb_entity"):
        headers["x-entity-id"] = config["wandb_entity"]
    if config.get("wandb_project"):
        headers["x-project-name"] = config["wandb_project"]
    return AuthHeaders(headers=headers, strategy="wandb_api_key")


async def _finish(cleanup: Any) -> None:
    """Run ``cleanup`` to its end in a task of its own though this one is cancelled meanwhile, and then honor the
    cancellation. ``asyncio.run`` cancels every task left when it ends and closes the loop once they are done, so
    cleanup left to a later callback would not run before the loop closed."""
    task = asyncio.ensure_future(cleanup)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


async def _reap(client: Client, sandbox_id: str) -> None:
    """Delete ``sandbox_id``, retrying."""
    for attempt in range(_REAP_ATTEMPTS):
        try:
            await client.delete(sandbox_id)
            logger.info("Deleted CoreWeave sandbox %s after its create failed or was cancelled", sandbox_id)
            return
        except Exception as exc:  # noqa: BLE001 - retried, then logged
            logger.warning("Deleting CoreWeave sandbox %s failed (attempt %s): %s", sandbox_id, attempt + 1, exc)
            if attempt + 1 < _REAP_ATTEMPTS:
                await asyncio.sleep(2**attempt)
    logger.error("CoreWeave sandbox %s from a failed create may still be running; it stops at its max lifetime",
                 sandbox_id)


async def _reclaim(client: Client, raw: Any, create_tag: str, starting: asyncio.Future) -> None:
    """Delete the sandbox a failed or cancelled create made, or may have made with its response lost: the one with the
    id its start request answered, or else the one tagged with the create's unique tag."""
    if not starting.done():
        await asyncio.wait({starting}, timeout=_START_SETTLE_SEC)
    if not starting.done():
        logger.warning("A cancelled CoreWeave create's start request is still unanswered after %ss; looking for its "
                       "sandbox by its tag %s", _START_SETTLE_SEC, create_tag)
    elif not starting.cancelled():
        starting.exception()
    if raw.sandbox_id:
        await _reap(client, raw.sandbox_id)
        return
    try:
        found = await client.list([create_tag])
    except Exception as exc:  # noqa: BLE001 - best effort; the sandbox still stops at its max lifetime
        logger.warning("Could not look for a CoreWeave sandbox tagged %s left by a failed create: %s", create_tag, exc)
        return
    for sandbox in found:
        await _reap(client, sandbox.sandbox_id)


class CoreWeaveSandboxProvider(SandboxProvider):
    """Provision CoreWeave Sandboxes: containers that run a workload's image, and Docker-capable VMs.

    Credentials come from resolved provider config, never from the process environment: ``api_key`` (a CoreWeave
    Sandbox API key) or ``wandb_api_key``, each an ``env:`` or ``secret:`` reference in
    ``[sandbox.providers.coreweave.config]``. Workloads never see them. ``image``, ``docker_storage`` and
    ``runtime_class`` configure the VM's Docker host only.
    """

    # The platform pulls a container's image by its name; only a VM loads a tarball or builds a context.
    CONTAINER_ACCEPTS = Accepts.NAME
    SANDBOX_ACCEPTS = Accepts.NAME

    def __init__(
        self,
        *,
        api_key: str | None = None,
        wandb_api_key: str | None = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        base_url: str | None = None,
        image: str = DEFAULT_IMAGE,
        docker_storage: str = DOCKER_STORAGE_LOOP,
        runtime_class: str = DEFAULT_RUNTIME_CLASS,
        placement_mode: str | None = None,
        runner_ids: list[str] | None = None,
        sandbox_cls: Any = cwsandbox.Sandbox,
    ):
        if bool(api_key) == bool(wandb_api_key):
            raise ValueError("Pass exactly one of api_key and wandb_api_key")
        self._image = image
        self._docker_storage = docker_storage
        self._runtime_class = runtime_class
        self._placement_mode = placement_mode
        self._runner_ids = runner_ids
        self._pinned_images: dict[str, str] = {}
        # sandbox_cls is injectable so unit tests never reach the Sandbox API.
        self._client = Client(
            sandbox_cls=sandbox_cls,
            auth=auth_headers({"api_key": api_key, "wandb_api_key": wandb_api_key, "wandb_entity": wandb_entity,
                               "wandb_project": wandb_project}),
            base_url=base_url,
        )
        # The SDK's SIGINT/SIGTERM handlers would stop sandboxes a later run reconnects to; agent-env tears down its own.
        # Raises only when another user of the SDK in this process installed them first.
        with contextlib.suppress(RuntimeError):
            cwsandbox.disable_signal_handlers()

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"{_SECTION} has unknown key(s): {sorted(unknown)}")
        for key in _STRING_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{_SECTION} '{key}' must be a non-empty string")
        if bool(config.get("api_key")) == bool(config.get("wandb_api_key")):
            raise ConfigError(f"{_SECTION} requires exactly one of 'api_key' and 'wandb_api_key'")
        if config.get("api_key") and (config.get("wandb_entity") or config.get("wandb_project")):
            raise ConfigError(f"{_SECTION} takes 'wandb_entity' and 'wandb_project' only with 'wandb_api_key'")
        if config.get("docker_storage", DOCKER_STORAGE_LOOP) not in (DOCKER_STORAGE_LOOP, DOCKER_STORAGE_MEMORY):
            raise ConfigError(f"{_SECTION} 'docker_storage' must be {DOCKER_STORAGE_LOOP!r} or "
                              f"{DOCKER_STORAGE_MEMORY!r}")
        runner_ids = config.get("runner_ids")
        if runner_ids is not None and (not isinstance(runner_ids, list) or not runner_ids or not all(
                isinstance(runner, str) and runner.strip() for runner in runner_ids)):
            raise ConfigError(f"{_SECTION} 'runner_ids' must be a non-empty list of non-empty strings")
        return cls(**config)

    async def prepare_image(self, image: DockerImageArtifact, *, attribution: Attribution | None = None) -> None:
        """Remember the artifact's digest so native containers pull that version rather than its mutable tag."""
        if image.image_digest:
            self._pinned_images[image.image_name] = image.pinned_name
        else:
            self._pinned_images.pop(image.image_name, None)

    @classmethod
    def effective_network_policy(cls, policy: NetworkPolicy | None) -> NetworkPolicy:
        """Every platform's hosts, as on the base, and the exposed-port domains seen so far."""
        return super().effective_network_policy(policy).with_hosts(sorted(_endpoint_parents))

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allowlist of hostnames, IP addresses and CIDRs. A hostname entry allows HTTPS on port 443
        only. The runner's policy may still refuse hostname rules, or ranges outside the public internet, when the
        sandbox is created."""
        try:
            network_options(policy)
        except ValueError:
            return False
        return True

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
    ) -> CoreWeaveSandbox:
        """Create a CoreWeave Sandbox that is a Docker host and expose ``exposed_ports`` as public HTTPS endpoints.
        ``timeout`` is the sandbox's hard maximum lifetime. It boots the configured image, so a per-call ``image`` is
        refused rather than ignored; ``boot_mode`` has no CoreWeave equivalent and is ignored."""
        if image is not None:
            raise ValueError(
                "CoreWeave sandboxes boot the image in [sandbox.providers.coreweave.config]; a per-call image is "
                "unsupported"
            )
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: CoreWeave sandboxes have no boot modes", boot_mode)
        ports = list(dict.fromkeys(exposed_ports or []))
        return await self._launch(
            CoreWeaveSandbox,
            ports=ports,
            timeout=timeout,
            attribution=attribution,
            network_policy=network_policy,
            ready=(lambda sandbox: sandbox.setup_vm_for_gateway(ports)) if setup_for_gateway else None,
            command="/bin/sh",
            args=["-c", host_setup_script(self._docker_storage, docker_disk_gb(disk_size_gb))],
            container_image=self._image,
            resources=resources(cpu, memory),
            runtime_class=self._runtime_class,
            security_context=SecurityContext(privileged=True),
        )

    async def create_container(
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
        private_network: bool = False,
    ) -> CoreWeaveContainerSandbox:
        """Create a CoreWeave Sandbox that runs ``image_name`` itself, unprivileged, with its own entrypoint and
        command and ``env``, and expose ``port`` as a public HTTPS endpoint. ``timeout`` is its hard maximum lifetime;
        ``disk_size_gb`` has no CoreWeave equivalent and is logged as ignored. The platform pulls the image without the
        image store's credentials, so an image in a registry they are for is refused."""
        if private_network:
            raise ValueError("CoreWeave sandboxes share no private network, so containers can't be put on one")
        # Every caller passes a size, so this is info rather than a warning on each deploy.
        logger.info("Ignoring disk_size_gb=%s: a CoreWeave container sandbox has no disk size to set", disk_size_gb)
        store = get_config().get_image_store_at(image_name)
        if (auth := await asyncio.to_thread(store.auth, image_name)) is not None:
            raise ValueError(
                f"CoreWeave pulls {image_name} itself, without the credentials agent-env's image store holds for "
                f"{auth.registry}; run it from a registry CoreWeave can pull from without them"
            )
        return await self._launch(
            CoreWeaveContainerSandbox,
            ports=[port],
            timeout=timeout,
            attribution=attribution,
            network_policy=network_policy,
            # A single container's spec, not the SDK's single-container arguments, which would replace an unset command
            # with its own default: the image's entrypoint and command must run.
            containers=[Container(image=self._pinned_images.get(image_name, image_name),
                                  environment_variables=dict(env), resources=resources(cpu, memory))],
        )

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
    ) -> CoreWeaveContainerSandbox:
        return await self.create_container(
            image_name=image_name, port=port, env=env, cpu=cpu, memory=memory, disk_size_gb=disk_size_gb,
            timeout=timeout, attribution=attribution, network_policy=network_policy,
        )

    async def _launch(
        self,
        adapter: Any,
        *,
        ports: list[int],
        timeout: int,
        attribution: Attribution | None,
        network_policy: NetworkPolicy | None,
        ready: Callable[[Any], Awaitable[None]] | None = None,
        **spec: Any,
    ) -> Any:
        """Create a sandbox from ``spec``, wait for it to run and for ``ready``, and wrap it in ``adapter``; one whose
        create fails or is cancelled is deleted before this ends."""
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer number of seconds")
        effective_policy = self.effective_network_policy(network_policy)
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                f"CoreWeave enforces allow-all or an allowlist of hostnames, IP addresses and CIDRs, not "
                f"{effective_policy.to_dict()}"
            )
        metadata = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        create_tag = f"{SANDBOX_TAG}-{uuid.uuid4().hex[:12]}"
        raw = self._client.sandbox_cls(
            auth=self._client.auth,
            base_url=self._client.base_url,
            max_lifetime_seconds=timeout,
            tags=sandbox_tags(metadata, create_tag),
            runner_ids=self._runner_ids,
            placement_mode=self._placement_mode,
            network=network_options(effective_policy),
            services=services(ports),
            **spec,
        )
        await self._start(raw, create_tag)
        applied = policy_from_egress(raw.effective_egress)
        if applied is None:
            logger.warning("CoreWeave sandbox %s reports egress rules agent-env can't represent; its network policy "
                           "is recorded as unknown", raw.sandbox_id)
        sandbox = adapter(raw, client=self._client, tunnel_urls=tunnel_urls(raw), network_policy=applied,
                          command_timeout=timeout)
        try:
            if missing := [port for port in ports if port not in sandbox.tunnel_urls]:
                raise RuntimeError(f"CoreWeave sandbox {sandbox.sandbox_id} has no endpoint for port(s) {missing}")
            logger.info("CoreWeave sandbox started: sandbox_id=%s mode=%s runner=%s ports=%s attribution=%s",
                        sandbox.sandbox_id, sandbox.mode, raw.runner_id, ports, metadata)
            if ready is not None:
                await ready(sandbox)
            return sandbox
        except BaseException:
            await _finish(_reap(self._client, sandbox.sandbox_id))
            raise

    async def _start(self, raw: Any, create_tag: str) -> None:
        """Start ``raw`` and wait for it to run. The SDK runs the create on its own event loop, so cancelling this task
        doesn't stop a create the server has taken; a sandbox that would otherwise run with no handle until its max
        lifetime is deleted before this ends: the one a failed or cancelled create makes, and the one it may have made
        with its response lost."""
        starting = asyncio.ensure_future(raw.start())
        try:
            await asyncio.shield(starting)
            await raw
        except BaseException:
            await _finish(_reclaim(self._client, raw, create_tag, starting))
            raise

    async def get_sandbox(self, sandbox_id: str) -> CoreWeaveContainerSandbox | CoreWeaveSandbox:
        """Reconnect, restoring the mode, the exposed ports and the applied egress policy. The mode is read from the
        sandbox's own spec: a VM is the one started by the Docker host's setup script."""
        try:
            raw = await self._client.get(sandbox_id)
        except cwsandbox.SandboxNotFoundError as exc:
            raise SandboxNotFoundError(f"CoreWeave sandbox {sandbox_id!r} was not found") from exc
        if raw.status in _GONE_STATUSES:
            raise SandboxNotFoundError(f"CoreWeave sandbox {sandbox_id!r} has ended ({raw.status})")
        docker_host = runs_docker_host(raw)
        if docker_host is None:
            raise RuntimeError(f"CoreWeave sandbox {sandbox_id!r} reports no container spec, so whether it is a "
                               "container or a VM is unknown")
        adapter = CoreWeaveSandbox if docker_host else CoreWeaveContainerSandbox
        return adapter(
            raw,
            client=self._client,
            tunnel_urls=tunnel_urls(raw),
            network_policy=policy_from_egress(raw.effective_egress),
            command_timeout=_RECONNECT_COMMAND_TIMEOUT_SEC,
        )

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        await self._client.delete(sandbox_id)
