"""Modal sandbox provider — runs agent containers as Modal sandboxes."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shlex
from typing import AsyncIterator, ClassVar, Optional
from urllib.parse import urlparse

import boto3
import httpx
import modal

from agent_env.attribution import Attribution
from agent_env.providers.sandbox import NetworkMode, NetworkPolicy, Sandbox
from agent_env.providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER,
    SandboxProvider,
    apply_default_attribution,
)

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "agent-env"
# Lets a dedicated service (e.g. the sandbox proxy service) attribute its Modal apps under its own
# base name instead of the shared default. Resolved per-construction, so it's read
# from the env at deploy time, not import time.
MODAL_APP_NAME_ENV_VAR = "AGENT_ENV_MODAL_APP_NAME"

# Public: out-of-tree Modal providers (a GPU one, for instance) import it too.
ECR_READER_SECRET_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]


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


def _fmt_exc(e: BaseException) -> str:
    type_name = f"{type(e).__module__}.{type(e).__qualname__}"
    msg = str(e).strip()
    return f"{type_name}: {msg}" if msg else type_name


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

    async def write_file_from_s3(self, s3_url: str, destination_path: str) -> None:
        parsed = urlparse(s3_url)
        body = boto3.client("s3").get_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip("/"),
        )["Body"]
        loop = asyncio.get_event_loop()

        async def _aiter() -> AsyncIterator[bytes]:
            while True:
                # boto3's StreamingBody is sync; offload each read so we don't block
                # the event loop on a multi-hundred-MB download.
                chunk = await loop.run_in_executor(None, body.read, _STDIN_CHUNK_BYTES)
                if not chunk:
                    return
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
    # Sandbox class this provider produces. A subclass registered under a different
    # ``[sandbox.providers.<name>]`` key can override it with a ``ModalSandbox`` subclass
    # whose ``type`` equals that name — the registry's type-guard requires the produced
    # sandbox's ``.type`` to match the config key.
    _sandbox_cls: ClassVar[type[ModalSandbox]] = ModalSandbox

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
        # One Modal App per attribution scope (keyed by project_id), looked up on demand.
        # Billing aggregates per App and the dashboard is browsable by App name, so a
        # scope-named, scope-tagged App is what makes cost both attributable and findable.
        self._apps: dict[str, modal.App] = {}

    async def _get_client(self) -> modal.Client:
        if self._client is None:
            from agent_env.config import get_config
            token_id, token_secret = get_config().get_modal_credentials()
            self._client = await modal.Client.from_credentials.aio(token_id, token_secret)
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
        priority: Optional[int] = None,
        network_policy: Optional[NetworkPolicy] = None,
        command: list[str] | None = None,
        i6pn: bool = False,
        region: Optional[str] = None,
        expose_externally: bool = True,
        vnc_port: Optional[int] = None,
    ) -> Sandbox:
        from agent_env.config import get_config

        effective = self.effective_network_policy(network_policy)
        attribution = dict(attribution or {})

        # Place the sandbox under a per-project App: Modal billing aggregates cost per App and
        # the dashboard is browsable by App name, so a scope-named, scope-tagged App is what
        # makes cost both attributable and findable. We attribute at the App level only —
        # sandbox-level tags don't show up in billing.
        app_tags = _build_cost_attribution_tags(attribution)
        app_name = _app_name_for_project(self._app_name, app_tags.get("project_id"))
        app = await self._get_app(app_name, app_tags)

        image_store = get_config().get_image_store()
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
        if uses_ecr_pull_secret:
            image = modal.Image.from_aws_ecr(image_name, secret=modal.Secret.from_name(
                self._ecr_pull_secret_name, required_keys=ECR_READER_SECRET_KEYS))
        elif (auth := image_store.auth(image_name)) is not None:
            image = modal.Image.from_registry(image_name, secret=modal.Secret.from_dict({
                "REGISTRY_USERNAME": auth.username,
                "REGISTRY_PASSWORD": auth.password,
            }))
        else:
            image = modal.Image.from_registry(image_name)

        logger.info(
            f"Creating Modal sandbox from image {image_name} "
            f"(app={app_name}, tags={app_tags}, port={port}, cpu={cpu}, memory={memory}MB, "
            f"gpu={self._gpu}, i6pn={i6pn}, region={region}, expose_externally={expose_externally})"
        )
        client = await self._get_client()
        ports = [port] if vnc_port is None else [port, vnc_port]
        port_kwargs = {"encrypted_ports": ports} if expose_externally else {}

        call_context = (
            f"image={image_name} port={port} cpu={cpu} memory={memory}MB "
            f"app={app_name} region={region} i6pn={i6pn}"
        )

        # GPU sandboxes need the V1 ``Sandbox.create`` factory: the V2
        # ``_experimental_create`` has no ``gpu`` parameter and cannot attach one. V1 in
        # turn has no ``i6pn``, so reject that combination up front — before a billed
        # sandbox exists — rather than silently returning one whose i6pn_address is None
        # and failing downstream (e.g. the gateway's "service-db has no i6pn address").
        if self._gpu and i6pn:
            raise ValueError(
                "i6pn is unavailable on GPU sandboxes: the GPU-capable factory "
                "(modal.Sandbox.create) has no i6pn parameter."
            )

        # ``i6pn`` is threaded per-factory below (V2 only); everything else is shared.
        create_kwargs = dict(
            app=app, image=image, cpu=cpu, memory=memory, timeout=timeout,
            env=env or {}, client=client, **port_kwargs,
            readiness_probe=modal.Probe.with_exec("true"),
            **_modal_network_kwargs(effective),
        )
        if region is not None:
            create_kwargs["region"] = region

        try:
            if self._gpu:
                sb = await modal.Sandbox.create.aio(
                    *(command or []),
                    gpu=self._gpu,
                    experimental_options={"enable_docker_in_gvisor": True},
                    **create_kwargs,
                )
            else:
                sb = await modal.Sandbox._experimental_create.aio(
                    *(command or []), i6pn=i6pn, **create_kwargs
                )
        except Exception as e:
            raise RuntimeError(
                f"Modal sandbox create failed [{_fmt_exc(e)}]: {call_context}"
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
        except Exception as e:
            try:
                await sb.terminate.aio()
            except Exception:
                pass
            raise RuntimeError(
                f"Modal sandbox post-create failed [{_fmt_exc(e)}]: "
                f"sb_id={sb.object_id} {call_context}"
            ) from e

    async def create_sandbox(
        self, *, image_name: str, port: int, env: dict[str, str],
        cpu: float = 0.125, memory: int = 8192, disk_size_gb: float = 10, timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        priority: Optional[int] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> Sandbox:
        return await self.create_container(
            image_name=image_name, port=port, env=env,
            cpu=cpu, memory=memory, disk_size_gb=disk_size_gb, timeout=timeout,
            attribution=attribution,
            priority=priority, network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        client = await self._get_client()
        sb = await modal.Sandbox.from_id.aio(sandbox_id, client=client)
        tunnels = await sb.tunnels.aio()
        tunnel_urls = {p: _tunnel_url(t) for p, t in tunnels.items()}
        i6pn_address = await _resolve_i6pn_address(sb)
        return self._sandbox_cls(sb, tunnel_urls, i6pn_address=i6pn_address)


def _app_name_for_project(base: str, project_id: Optional[str]) -> str:
    """Map a project_id to a stable Modal App name for per-project cost attribution.

    The app name is what shows up (and is searchable) in the Modal dashboard, so it encodes
    the project. Names are sanitized to ``[a-zA-Z0-9._-]``, kept under Modal's 64-char limit,
    and not double-prefixed when ``project_id`` already starts with ``base``. Falls back to
    ``base`` when no project_id is available.
    """
    if not project_id:
        return base
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", project_id).strip("-") or base
    name = slug if (slug == base or slug.startswith(f"{base}-")) else f"{base}-{slug}"
    if len(name) > 64:
        logger.warning(
            f"Modal app name for project_id={project_id!r} exceeds 64 chars; truncating to "
            f"{name[:64]!r}. Distinct project_ids that share this prefix will be cost-attributed "
            f"together."
        )
        name = name[:64]
    return name


def _build_cost_attribution_tags(attribution: Attribution) -> dict[str, str]:
    """Build the Modal App tag set used for cost attribution.

    Unset dimensions fall back to config.toml ``[sandbox.attribution]``; a dimension with
    no value anywhere is omitted rather than emitted as a null tag. Returns a flat
    ``dict[str, str]`` for ``modal.App.set_tags``. (priority is a scheduling concern,
    not attribution, so it is not included here.)
    """
    resolved = apply_default_attribution(attribution)
    return {
        name: resolved[name]
        for name in ("product", "customer", "team", "project_id")
        if resolved.get(name) is not None
    }

