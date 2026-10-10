"""Modal sandbox provider — runs agent containers as Modal sandboxes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import logging
import os
import re
import shlex
from typing import TYPE_CHECKING, AsyncIterator, ClassVar, Optional

import httpx
import modal

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, Attribution
from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER,
    Accepts,
    SandboxProvider,
    apply_default_attribution,
)
from agent_env.providers.sandbox_providers.modal_image_build import context_image, fmt_exc

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "agent-env"
# Lets a dedicated service (e.g. the sandbox proxy service) attribute its Modal apps under its own
# base name instead of the shared default. Resolved per-construction, so it's read
# from the env at deploy time, not import time.
MODAL_APP_NAME_ENV_VAR = "AGENT_ENV_MODAL_APP_NAME"

# Public: out-of-tree Modal providers (a GPU one, for instance) import it too.
ECR_READER_SECRET_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]

_MODAL_TAG_MAX_LEN = 63
_MODAL_TAG_KEY = re.compile(r"[a-zA-Z0-9._-]{1,63}")


def _resolve_app_base_name(app_name: Optional[str]) -> str:
    return app_name or os.getenv(MODAL_APP_NAME_ENV_VAR) or DEFAULT_APP_NAME

_I6PN_RESOLVE_CMD = "getent ahostsv6 i6pn.modal.local 2>/dev/null | awk 'NR==1 {print $1}'"
# 8 MB chunks of raw bytes — base64 inflates to ~10.7 MB, comfortably under
# Modal's TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE (16 MB) per write. Keep non-final
# chunks divisible by 3 so concatenated base64 has padding only at EOF (mid-stream
# '=' would make `base64 -d` truncate the file).
_STDIN_CHUNK_BYTES = 8 * 1024 * 1024 - ((8 * 1024 * 1024) % 3)


def _tunnel_url(t) -> str:
    return f"https://{t.host}" + (f":{t.port}" if t.port != 443 else "")


def _modal_network_kwargs(policy: NetworkPolicy) -> dict:
    """``policy`` as Modal ``_experimental_create`` kwargs.

    ALLOW_ALL contributes nothing: passing ``outbound_domain_allowlist`` at all flips Modal
    OPEN -> ALLOWLIST and drops raw-IP egress, even for ``["*"]``.
    """
    if policy.mode is NetworkMode.ALLOW_ALL:
        return {}
    return {
        "outbound_domain_allowlist": list(policy.allow_hosts),
        "outbound_cidr_allowlist": list(policy.allow_cidrs),
    }


async def _resolve_i6pn_address(sb: modal.Sandbox) -> Optional[str]:
    try:
        p = await sb.exec.aio("sh", "-c", _I6PN_RESOLVE_CMD)
        raw = await p.stdout.read.aio()
        out = raw.decode().strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
        rc = await p.wait.aio()
        if rc == 0 and out and ":" in out:
            return out
        logger.warning(f"i6pn address resolution returned rc={rc} stdout={out!r}")
    except Exception as e:
        logger.warning(f"i6pn address resolution failed: {type(e).__name__}: {e}")
    return None


class _ModalStreamAdapter:
    def __init__(self, stream):
        self._stream = stream

    async def read(self) -> bytes:
        return await self._stream.read.aio()


class _ModalProcessAdapter:
    def __init__(self, process):
        self._process = process
        self.stdout = _ModalStreamAdapter(process.stdout)
        self.stderr = _ModalStreamAdapter(process.stderr)

    async def wait(self) -> int:
        return await self._process.wait.aio()


class ModalSandbox(Sandbox):
    type = "modal"

    def __init__(self, sb: modal.Sandbox, tunnel_urls: dict[int, str], i6pn_address: Optional[str] = None,
                 network_policy: NetworkPolicy | None = None):
        self._sb = sb
        self.sandbox_id = sb.object_id
        self.tunnel_urls = tunnel_urls
        self.i6pn_address = i6pn_address
        self.vnc_url = None
        self.mode = SANDBOX_MODE_CONTAINER
        self.network_policy = network_policy

    @property
    def private_host(self) -> str | None:
        """Its address on i6pn, Modal's private network between sandboxes, bracketed for a URL."""
        return f"[{self.i6pn_address}]" if self.i6pn_address else None

    async def terminate(self) -> None:
        await self._sb.terminate.aio()

    async def exec(self, *command: str):
        process = await self._sb.exec.aio(*command, text=False)
        return _ModalProcessAdapter(process)

    async def _write_stream_via_exec(
        self, stream: AsyncIterator[bytes], destination_path: str,
    ) -> None:
        # V2 has no filesystem API and our container images don't reliably ship curl —
        # bytes flow through `base64 -d` over exec stdin instead.
        parent = "/".join(destination_path.split("/")[:-1])
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        cmd = f"{mkdir}base64 -d > {shlex.quote(destination_path)}"
        proc = await self._sb.exec.aio("sh", "-c", cmd, text=False)
        buf = bytearray()
        async for chunk in stream:
            buf.extend(chunk)
            while len(buf) >= _STDIN_CHUNK_BYTES:
                proc.stdin.write(base64.b64encode(bytes(buf[:_STDIN_CHUNK_BYTES])))
                del buf[:_STDIN_CHUNK_BYTES]
                await proc.stdin.drain.aio()
        if buf:
            proc.stdin.write(base64.b64encode(bytes(buf)))
            await proc.stdin.drain.aio()
        proc.stdin.write_eof()
        await proc.stdin.drain.aio()
        rc = await proc.wait.aio()
        if rc != 0:
            stderr = await proc.stderr.read.aio()
            raise RuntimeError(f"write to {destination_path} failed (exit={rc}): {stderr!r}")

    async def write_file_from_object(self, object_url: str, destination_path: str) -> None:
        body = await asyncio.to_thread(get_config().get_object_store_at(object_url).open, object_url)
        with contextlib.closing(body):

            async def _aiter() -> AsyncIterator[bytes]:
                while chunk := await asyncio.to_thread(body.read, _STDIN_CHUNK_BYTES):
                    yield chunk

            await self._write_stream_via_exec(_aiter(), destination_path)

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        async def _aiter() -> AsyncIterator[bytes]:
            yield content.encode()

        await self._write_stream_via_exec(_aiter(), destination_path)

    async def write_file_from_url(self, url: str, destination_path: str) -> None:
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                await self._write_stream_via_exec(
                    r.aiter_bytes(_STDIN_CHUNK_BYTES), destination_path,
                )


class ModalSandboxProvider(SandboxProvider):
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.modal.host", "*.w.modal.host")
    # Sandbox class this provider produces. A subclass or instance registered under a different
    # ``[sandbox.providers.<name>]`` key can set it to a ``ModalSandbox`` subclass whose
    # ``type`` equals that name — the registry's type-guard requires the produced
    # sandbox's ``.type`` to match the config key.
    _sandbox_cls: type[ModalSandbox] = ModalSandbox

    # Each image in a container of its own, built first when it is only a build context (prepare_image); its
    # containers share i6pn, Modal's private network.
    SANDBOX_ACCEPTS = CONTAINER_ACCEPTS = Accepts.NAME_OR_CONTEXT
    PRIVATE_NETWORK = True

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        return True

    def __init__(
        self,
        app_name: Optional[str] = None,
        gpu: Optional[str] = None,
        ecr_pull_secret_name: Optional[str] = None,
    ):
        self._app_name = _resolve_app_base_name(app_name)
        self._ecr_pull_secret_name = ecr_pull_secret_name
        self._client: Optional[modal.Client] = None
        # Optional Modal GPU spec (e.g. "H100", "A100", "H100:2"). When set, create_container
        # provisions the sandbox with the V1 ``modal.Sandbox.create`` factory (which accepts
        # ``gpu=`` + docker-in-gVisor) instead of the V2 ``_experimental_create``, which
        # cannot attach a GPU. V1 has no ``i6pn``, so a GPU sandbox has no private-IPv6
        # east-west networking (fine for a single sandbox; not for an i6pn mesh).
        self._gpu = gpu
        # Looked up on demand and cached by name.
        self._apps: dict[str, modal.App] = {}
        self._token_id = ""
        # The context-only images prepare_image was given, by name, so create_container builds them rather than pull.
        self._context_images: dict[str, DockerImageArtifact] = {}

    async def _get_client(self) -> modal.Client:
        if self._client is None:
            from agent_env.config import get_config
            token_id, token_secret = get_config().get_modal_credentials()
            self._client = await modal.Client.from_credentials.aio(token_id, token_secret)
            self._token_id = token_id
        return self._client

    async def _get_app(self, app_name: str, app_tags: Optional[dict[str, str]] = None) -> modal.App:
        app = self._apps.get(app_name)
        if app is None:
            client = await self._get_client()
            app = await modal.App.lookup.aio(app_name, create_if_missing=True, client=client)
            if app_tags:
                # App tags are the dimension Modal's billing report breaks down by; set them
                # once per app (cached below) rather than on every sandbox.
                try:
                    await app.set_tags.aio(app_tags)
                except Exception as e:
                    logger.warning(f"Failed to set tags on Modal app {app_name}: {type(e).__name__}: {e}")
            self._apps[app_name] = app
        return app

    async def prepare_image(self, image: DockerImageArtifact, *, attribution: Optional[Attribution] = None) -> None:
        """Build ``image`` when it's only a build context, unless this process has already built its sources."""
        if image.context_only:
            self._context_images[image.image_name] = image
            await self._context_image(image)

    async def _context_image(self, image: DockerImageArtifact, *,
                             stale: str | None = None) -> tuple[modal.Image, str]:
        """The Modal image built from ``image``'s build context, and its id (``modal_image_build.context_image``)."""
        client = await self._get_client()
        app = await self._get_app(self._app_name, _attribution_tags({}))
        return await context_image(image, client=client, app=app, token_id=self._token_id, app_name=self._app_name,
                                   stale=stale)

    async def create_container(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        # Modal's minimum; cpu is a burstable reservation here, not a cap.
        cpu: float = 0.125,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
        command: list[str] | None = None,
        expose_externally: bool = True,
        vnc_port: Optional[int] = None,
        private_network: bool = False,
    ) -> Sandbox:
        # The private network is i6pn, Modal's network between sandboxes, in the configured region.
        i6pn = private_network
        region = get_config().modal_default_region if private_network else None
        # GPU sandboxes need the V1 ``Sandbox.create`` factory: the V2
        # ``_experimental_create`` has no ``gpu`` parameter and cannot attach one. V1 in
        # turn has no ``i6pn``, so reject that combination up front — before any Modal
        # call — rather than silently returning one whose i6pn_address is None
        # and failing downstream (e.g. the gateway's "service-db has no i6pn address").
        if self._gpu and i6pn:
            raise ValueError(
                "i6pn is unavailable on GPU sandboxes: the GPU-capable factory "
                "(modal.Sandbox.create) has no i6pn parameter."
            )
        effective = self.effective_network_policy(network_policy)
        attribution = dict(attribution or {})

        # Modal bills by App tags, and the one shared App is tagged once, so it carries only the
        # deployment's [sandbox.attribution]; the run's own attribution goes on its sandbox.
        app_tags = _attribution_tags({})
        sandbox_tags = _attribution_tags(attribution)
        app_name = self._app_name
        app = await self._get_app(app_name, app_tags)

        built_id = None
        if (context_only := self._context_images.get(image_name)) is not None:
            image, built_id = await self._context_image(context_only)
        else:
            image = await self._registry_image(image_name)

        logger.info(
            f"Creating Modal sandbox from image {image_name} "
            f"(app={app_name}, tags={app_tags}, sandbox_tags={sandbox_tags}, port={port}, "
            f"cpu={cpu}, memory={memory}MB, "
            f"gpu={self._gpu}, i6pn={i6pn}, region={region}, expose_externally={expose_externally})"
        )
        client = await self._get_client()
        ports = [port] if vnc_port is None else [port, vnc_port]
        port_kwargs = {"encrypted_ports": ports} if expose_externally else {}

        call_context = (
            f"image={image_name} port={port} cpu={cpu} memory={memory}MB "
            f"app={app_name} region={region} i6pn={i6pn}"
        )

        # ``i6pn`` is threaded per-factory below (V2 only); everything else is shared.
        create_kwargs = dict(
            app=app, cpu=cpu, memory=memory, timeout=timeout,
            env=env or {}, client=client, **port_kwargs,
            tags=sandbox_tags or None,
            readiness_probe=modal.Probe.with_exec("true"),
            **_modal_network_kwargs(effective),
        )
        if region is not None:
            create_kwargs["region"] = region

        async def create(image: modal.Image) -> modal.Sandbox:
            if self._gpu:
                return await modal.Sandbox.create.aio(
                    *(command or []),
                    gpu=self._gpu,
                    experimental_options={"enable_docker_in_gvisor": True},
                    image=image,
                    **create_kwargs,
                )
            return await modal.Sandbox._experimental_create.aio(
                *(command or []), i6pn=i6pn, image=image, **create_kwargs
            )

        try:
            try:
                sb = await create(image)
            except modal.exception.NotFoundError:
                if built_id is None:
                    raise
                # Modal no longer holds the image this process built from the context, so it's built again, once.
                image, built_id = await self._context_image(context_only, stale=built_id)
                sb = await create(image)
        except Exception as e:
            raise RuntimeError(
                f"Modal sandbox create failed [{fmt_exc(e)}]: {call_context}"
            ) from e

        try:
            await sb.wait_until_ready.aio(timeout=150)
            if expose_externally:
                tunnels = await sb.tunnels.aio()
                tunnel_urls = {p: _tunnel_url(t) for p, t in tunnels.items()}
                if port not in tunnel_urls:
                    raise RuntimeError(f"No Modal tunnel for port {port}; got {list(tunnel_urls)}")
            else:
                tunnel_urls = {}
            i6pn_address = await _resolve_i6pn_address(sb) if i6pn else None
            endpoint = tunnel_urls.get(port, "i6pn-only" if i6pn_address else "no-tunnel")
            logger.info(
                f"Modal sandbox ready: {sb.object_id} ({endpoint})"
                + (f" i6pn={i6pn_address}" if i6pn_address else "")
            )
            await _log_sandbox_started(
                sb, app_name=app_name, sandbox_tags=sandbox_tags,
                cpu=cpu, memory=memory, gpu=self._gpu,
            )
            sandbox = self._sandbox_cls(
                sb,
                tunnel_urls,
                i6pn_address=i6pn_address,
                network_policy=effective,
            )
            if vnc_port in tunnel_urls:
                sandbox.vnc_url = f"{tunnel_urls[vnc_port]}/vnc.html"
                logger.info(f"Modal sandbox vnc_url: {sandbox.vnc_url}")
            return sandbox
        except BaseException as e:  # a cancelled create, at a chain's deadline say, terminates its sandbox too
            try:
                await sb.terminate.aio()
            except Exception:
                pass
            if not isinstance(e, Exception):
                raise
            raise RuntimeError(
                f"Modal sandbox post-create failed [{fmt_exc(e)}]: "
                f"sb_id={sb.object_id} {call_context}"
            ) from e

    async def create_sandbox(
        self, *, image_name: str, port: int, env: dict[str, str],
        cpu: float = 0.125, memory: int = 8192, disk_size_gb: float = 10, timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> Sandbox:
        return await self.create_container(
            image_name=image_name, port=port, env=env,
            cpu=cpu, memory=memory, disk_size_gb=disk_size_gb, timeout=timeout,
            attribution=attribution,
            network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        client = await self._get_client()
        sb = await modal.Sandbox.from_id.aio(sandbox_id, client=client)
        tunnels = await sb.tunnels.aio()
        tunnel_urls = {p: _tunnel_url(t) for p, t in tunnels.items()}
        i6pn_address = await _resolve_i6pn_address(sb)
        return self._sandbox_cls(sb, tunnel_urls, i6pn_address=i6pn_address)

    async def _registry_image(self, image_name: str) -> modal.Image:
        """``image_name`` pulled from its registry, with the image store's credentials for it, if any."""
        image_store = get_config().get_image_store_at(image_name)
        from agent_env.store.image_store import (
            EcrCredentials,
            OciRegistryImageStore,
            registry_host_from_ref,
        )

        uses_ecr_pull_secret = (
            self._ecr_pull_secret_name is not None
            and isinstance(image_store, OciRegistryImageStore)
            and isinstance(image_store.credentials, EcrCredentials)
            and registry_host_from_ref(image_name) == image_store.registry_host
        )
        auth = None if uses_ecr_pull_secret else await asyncio.to_thread(image_store.auth, image_name)
        if uses_ecr_pull_secret:
            return modal.Image.from_aws_ecr(image_name, secret=modal.Secret.from_name(
                self._ecr_pull_secret_name, required_keys=ECR_READER_SECRET_KEYS))
        if auth is not None:
            return modal.Image.from_registry(image_name, secret=modal.Secret.from_dict({
                "REGISTRY_USERNAME": auth.username,
                "REGISTRY_PASSWORD": auth.password,
            }))
        return modal.Image.from_registry(image_name)


def _attribution_tags(attribution: Attribution) -> dict[str, str]:
    """``attribution`` as Modal tags, any keys: unset ones filled from config.toml
    ``[sandbox.attribution]``, None omitted. A key Modal can't take is refused, not rewritten."""
    tags = {}
    for key, value in apply_default_attribution(attribution).items():
        if value is None:
            continue
        if not _MODAL_TAG_KEY.fullmatch(key):
            raise ValueError(f"attribution key {key!r} can't be a Modal tag: use 1-63 of a-z A-Z 0-9 . _ -")
        tags[key] = _modal_tag_value(value)
    return tags


SANDBOX_STARTED_EVENT = "agent_env.modal_sandbox_started"


async def _log_sandbox_started(
    sb, *, app_name: str, sandbox_tags: dict[str, str],
    cpu: float, memory: int, gpu: Optional[str],
) -> None:
    """Log one structured line mapping Modal's container id to the sandbox and its tags.

    Modal's exported metrics carry only ``container_id`` (the ``ta-`` task id), not sandbox tags, so
    this line is the join key from a container's usage to its run and pipeline step. Never raises:
    the task id comes from a private Modal method.
    """
    try:
        container_id = await sb._get_task_id.aio()
    except Exception as e:
        logger.warning(f"Could not read the Modal container id of {sb.object_id}: {type(e).__name__}: {e}")
        container_id = None
    logger.info(
        f"Modal sandbox started: sandbox_id={sb.object_id} container_id={container_id} "
        f"app={app_name} tags={sandbox_tags}",
        extra={
            "event": SANDBOX_STARTED_EVENT,
            "modal_sandbox_id": sb.object_id,
            "modal_container_id": container_id,
            "modal_app_name": app_name,
            "modal_sandbox_tags": sandbox_tags,
            PIPELINE_STEP_KEY: sandbox_tags.get(PIPELINE_STEP_KEY),
            RUN_ID_KEY: sandbox_tags.get(RUN_ID_KEY),
            "cpu": cpu,
            "memory_mb": memory,
            "gpu": gpu,
        },
    )


def _modal_tag_value(value: str) -> str:
    """Modal rejects the whole create on an invalid tag, so ``value`` is cut to ``[a-zA-Z0-9._-]``;
    past 63 chars it keeps a prefix plus a hash of the whole, so long values stay distinct."""
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", value)
    if len(slug) > _MODAL_TAG_MAX_LEN:
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        slug = f"{slug[:_MODAL_TAG_MAX_LEN - len(digest) - 1]}-{digest}"
    return slug
