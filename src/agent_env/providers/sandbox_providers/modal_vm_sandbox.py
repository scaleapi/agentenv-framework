"""Modal **VM-mode** sandbox provider — runs an entire env on ONE Modal VM sandbox.

This is the Modal analogue of a KubeVirt-style one-VM provider rather than of the
container-per-service ModalSandboxProvider. It provisions a single Modal VM sandbox
(`experimental_options={"vm_runtime": True}` → real Linux kernel + nested Docker),
so the gateway's VM path (`EnvironmentGatewayProvider._deploy_via_vm`) can `docker compose up`
the whole stack (postgres + N MCP servers + gateway) inside one box behind one tunnel.

Deliberately a *separate* class from ModalSandboxProvider (no shared base): the
container provider stays byte-for-byte unchanged, and because this is NOT an instance
of ModalSandboxProvider, `EnvironmentGatewayProvider._deploy_gateway` routes it to `_deploy_via_vm`
(not `_deploy_via_containers`) and the i6pn gate never fires — i6pn is a container-mode
east-west concern and is irrelevant here (one sandbox, one docker bridge).

Only the pure, stateless helpers are reused from modal_sandbox (no class coupling).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import time
from typing import Awaitable, Callable, ClassVar, Optional

import modal

from agent_env.providers.sandbox_providers.modal_sandbox import (
    _ModalProcessAdapter,
    _app_name_for_project,
    _build_cost_attribution_tags,
    _build_sandbox_tags,
    _log_sandbox_started,
    _resolve_app_base_name,
    _tunnel_url,
    _modal_network_kwargs,
)
from agent_env.attribution import Attribution
from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM, SandboxProvider

logger = logging.getLogger(__name__)

# Ubuntu 22.04 + Docker engine + the compose v2 plugin baked in — the Modal-native
# equivalent of a KubeVirt containerdisk image with Docker pre-installed,
# matching its OS baseline so the two backends behave alike. Modal builds/caches this
# image (no ECR push / no auth — public base). `docker.io` is the engine; the compose v2
# plugin isn't in Ubuntu's repos, so we drop the release binary into the CLI-plugins dir.
# amd64: Modal VM sandboxes run x86_64, matching the linux-x86_64 compose binary.
_COMPOSE_VERSION = "v2.29.7"
_VM_IMAGE = (
    modal.Image.from_registry("ubuntu:22.04")
    .apt_install("docker.io", "curl", "aria2")  # aria2: object downloads, see _DL_*
    .run_commands(
        "mkdir -p /usr/local/lib/docker/cli-plugins",
        f"curl -sSL https://github.com/docker/compose/releases/download/{_COMPOSE_VERSION}/"
        "docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose",
        "chmod +x /usr/local/lib/docker/cli-plugins/docker-compose",
    )
)

# The sandbox MAIN process. Two decoupled parts, both for robustness:
#  1. A best-effort log forwarder run as a *child* of this process — NOT a separate `exec`.
#     Modal reaps an exec'd process tree when the exec call returns (that killed an earlier
#     nohup'd forwarder); a child of the everlasting main process is not. It polls
#     `docker ps` and, per container, streams `docker logs -f` (name-prefixed, `sed -u`
#     unbuffered) to THIS process's stdout. Because the forwarder is a child it inherits
#     that stdout, which Modal captures into the dashboard — so logs reach Modal with no
#     intermediate file. There is therefore no second on-disk copy: Docker's own json-file
#     logs (what `docker logs` reads) are the single on-disk copy, and they are size-capped
#     at dockerd start (see _ensure_dockerd_running) so they can't fill the disk. Uses
#     `docker ps -aq` (not `-q`) so containers that have already exited are also followed:
#     `docker logs` persists after exit, and `docker logs -f` on an exited container just
#     dumps its logs and returns. That keeps startup/crash logs of a short-lived or failed
#     service (often the ones you need) instead of dropping them. New containers (compose
#     services / the agent-api container) are picked up by the poll loop as they appear.
#  2. A keepalive that never exits, so liveness is fully decoupled from logging: if the
#     forwarder child dies, this loop keeps the sandbox alive — equivalent to the previous
#     `sleep infinity` main process. Logging can never take the env down.
_MAIN_SCRIPT = r'''
(
  seen=""
  while true; do
    for c in $(docker ps -aq 2>/dev/null); do
      case " $seen " in
        *" $c "*) ;;
        *)
          seen="$seen $c"
          n=$(docker inspect -f '{{.Name}}' "$c" 2>/dev/null | sed 's#^/##')
          ( docker logs -f --tail 200 --timestamps "$c" 2>&1 | sed -u "s/^/[$n] /" ) &
          ;;
      esac
    done
    sleep 3
  done
) &
while true; do sleep 3600; done
'''
_KEEPALIVE_CMD = ("sh", "-c", _MAIN_SCRIPT)


class ModalVmSandbox(VmSandbox):
    """A Modal VM sandbox (vm_runtime) presented through the VmSandbox contract.

    Subclasses VmSandbox so it inherits exec_script / load_docker_images / load_s3_file /
    write_file_* unchanged — they are built on exec_with_output ->
    exec + an in-VM Docker daemon, which this class provides. Only the Modal-specific
    pieces (raw exec, terminate, dockerd startup) are implemented here.
    """

    type = "modal_vm"
    # Modal rejects an exec whose command exceeds ~48 KB, well under Linux's MAX_ARG_STRLEN.
    _WFT_CHUNK_BYTES = 32 * 1024

    def __init__(self, sb: modal.Sandbox, tunnel_urls: dict[int, str],
                 network_policy: NetworkPolicy | None = None):
        self._sb = sb
        self.sandbox_id = sb.object_id
        self.tunnel_urls = tunnel_urls
        self.vnc_url = None
        self.mode = SANDBOX_MODE_VM
        self.network_policy = network_policy
        self._download_slots = asyncio.Semaphore(max(1, _dl_setting("PARALLEL_FILES", _DL_PARALLEL_FILES)))

    async def terminate(self) -> None:
        await self._sb.terminate.aio()

    async def exec(self, *command: str):
        # Modal VM sandboxes run as root and have no `sudo`; the inherited VmSandbox
        # helpers prefix commands with "sudo" (sandbox.py). Strip it, exactly like
        # LocalSandbox.exec does, so those helpers work unchanged.
        cmd = [c for c in command if c != "sudo"]
        process = await self._sb.exec.aio(*cmd, text=False)
        return _ModalProcessAdapter(process)

    async def setup_vm_for_gateway(self, exposed_ports: Optional[list[int]] = None) -> None:
        """Make the VM ready for the gateway deploy.

        Unlike a containerdisk image that auto-starts dockerd via systemd, a Modal VM
        sandbox does not start dockerd on its own, so we start it here. No iptables step:
        ports are exposed via Modal `encrypted_ports` at create time, and Modal has no
        host INPUT chain to open (contrast VmSandbox.setup_vm_for_gateway).

        The container-log forwarder is NOT started here — it runs as a child of the
        sandbox main process (see `_MAIN_SCRIPT`) so Modal can't reap it the way it reaps
        a transient `exec`'d process when the exec call returns.
        """
        await self._ensure_dockerd_running()

    async def wait_for_vm(self) -> None:
        # For a Modal VM, "ready" means the Docker daemon is reachable.
        await self._ensure_dockerd_running()

    async def _ensure_dockerd_running(self) -> None:
        """Start dockerd if needed, then poll `docker info` until ready (non-blocking).

        Idempotent (safe on reconnect). Bounded by the inherited VmSandbox wall-clock
        budget; never blocks the event loop (background-launch + asyncio.sleep poll).
        """
        # Launch dockerd detached only if not already up. `nohup ... &` so it survives
        # the exec session; the `if` keeps this idempotent across reconnects. The
        # --log-driver/--log-opt daemon flags cap the json-file driver via rotation
        # (max-size 30m x max-file 3 => ~90 MB/container, oldest dropped first) so a
        # long-running env's container logs can't fill the 512 GiB rootfs. max-file>=2
        # keeps a buffer of already-written log behind the live `docker logs -f` follower,
        # so a follower briefly behind under load doesn't lose lines at a rotation. Passed
        # as daemon args (not a daemon.json file) so nothing extra is written to disk —
        # this is the single on-disk log copy (`docker logs` reads it; the forwarder
        # streams from it to stdout without a second copy).
        await self.exec_script(
            "if ! docker info >/dev/null 2>&1; then "
            "nohup dockerd "
            "--log-driver json-file --log-opt max-size=30m --log-opt max-file=3 "
            ">/var/log/dockerd.log 2>&1 & fi"
        )
        deadline = time.monotonic() + self._VM_READY_TIMEOUT
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            exit_code, _, _ = await self.exec_with_output("docker", "info")
            if exit_code == 0:
                elapsed = self._VM_READY_TIMEOUT - max(0.0, deadline - time.monotonic())
                logger.info(f"dockerd ready after {elapsed:.0f}s ({attempts} polls)")
                return
            await asyncio.sleep(self._VM_READY_POLL_INTERVAL)
        _, log_tail, _ = await self.exec_with_output("tail", "-n", "40", "/var/log/dockerd.log")
        raise RuntimeError(
            f"dockerd not ready in Modal VM {self.sandbox_id} after {self._VM_READY_TIMEOUT}s "
            f"({attempts} polls); /var/log/dockerd.log tail:\n{log_tail}"
        )

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Presigned objects go through aria2c (see _DL_*); a store that cannot presign keeps the base path."""
        store = get_config().get_object_store()
        signed = await asyncio.to_thread(store.signed_get_url, object_url)
        signed_at = time.monotonic()
        if signed is None:
            await self._write_unsigned_object(store, object_url, vm_path)
            return

        async def url(attempt: int) -> str:
            if attempt == 1 and time.monotonic() - signed_at < _FRESH_URL_SECONDS:
                return signed
            return await asyncio.to_thread(store.signed_get_url, object_url)

        await self._download_ranged(url, vm_path, label=object_url.rsplit("/", 1)[-1])

    async def _download_ranged(self, url_factory: Callable[[int], Awaitable[str]], vm_path: str, *, label: str) -> None:
        """Run the download script per attempt with the URL ``url_factory`` gives for it; 404 raises,
        anything else retries. The first script to run clears any stale file at ``vm_path``."""
        attempts = max(1, _dl_setting("ATTEMPTS", _DL_ATTEMPTS))
        deadline = _dl_setting("DEADLINE_S", _DL_DEADLINE_S)
        last = "no attempt made"
        ran = False
        async with self._download_slots:
            for attempt in range(1, attempts + 1):
                try:
                    url = await url_factory(attempt)
                except Exception as e:  # a failed re-sign costs this attempt, not the download
                    last = f"attempt {attempt}: signing failed: {type(e).__name__}: {str(e)[:300]}"
                    logger.warning(f"download {label}: {last}")
                    continue
                script = _render_download_script(url, vm_path, fresh=not ran)
                ran = True
                try:
                    stdout = await asyncio.wait_for(self.exec_script(script, max_retries=1), timeout=deadline + 120)
                except (asyncio.TimeoutError, RuntimeError) as e:  # host-side wait expired / exec transport failed
                    last = f"attempt {attempt}: {type(e).__name__}: {str(e)[:300]}"
                    logger.warning(f"download {label}: {last}")
                    continue
                parsed = _parse_download_summary(stdout)
                if parsed is None:
                    last = f"attempt {attempt}: no summary line; output tail {stdout[-300:]!r}"
                    logger.warning(f"download {label}: {last}")
                    continue
                rc, nbytes, seconds, msg = parsed
                if rc == 0:
                    line = (f"download result=ok bytes={nbytes} seconds={seconds} "
                            f"mib_s={nbytes / max(seconds, 1) / 1024 ** 2:.1f} attempts={attempt} label={label}")
                    logger.info(line)
                    print(line, flush=True)
                    return
                if _not_found(rc, msg):
                    raise RuntimeError(f"download {label}: object not found (aria2c rc={rc} {msg[:200]})")
                last = f"attempt {attempt}: aria2c rc={rc}; {msg[:300]}"
                logger.warning(f"download {label}: {last}")
        logger.error(f"download result=failed attempts={attempts} label={label}")
        await self._remove_vm_temp_file(vm_path)
        await self._remove_vm_temp_file(vm_path + ".aria2")
        raise RuntimeError(f"download {label} failed after {attempts} attempts: {last}")


class ModalVmSandboxProvider(SandboxProvider):
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.modal.host", "*.w.modal.host")

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        return True

    """Provisions a single Modal VM sandbox per env (the VM-mode Modal backend)."""

    def __init__(self, app_name: Optional[str] = None):
        self._app_name = _resolve_app_base_name(app_name)
        self._client: Optional[modal.Client] = None
        # Per-attribution-scope Modal App, looked up on demand (billing aggregates per App).
        # Self-contained copy of ModalSandboxProvider's client/app handling — intentionally
        # NOT shared, so the container provider is untouched.
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
                try:
                    await app.set_tags.aio(app_tags)
                except Exception as e:
                    logger.warning(f"Failed to set tags on Modal app {app_name}: {type(e).__name__}: {e}")
            self._apps[app_name] = app
        return app

    async def create_vm(
        self,
        *,
        image: Optional[str] = None,
        boot_mode: Optional[str] = None,
        cpu: float = 0.5,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: Optional[list[int]] = None,
        setup_for_gateway: bool = True,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> ModalVmSandbox:
        # image/boot_mode/disk_size_gb accepted for interface parity but inert here.
        # disk_size_gb: Modal has no disk-size knob on Sandbox.create; a VM sandbox's rootfs is
        # fixed at the 512 GiB max. https://modal.com/docs/guide/vm-sandboxes
        attribution = dict(attribution or {})
        app_tags = _build_cost_attribution_tags(attribution)
        app_name = _app_name_for_project(self._app_name, app_tags.get("project_id"))
        app = await self._get_app(app_name, app_tags)
        sandbox_tags = _build_sandbox_tags(attribution)
        client = await self._get_client()
        effective = self.effective_network_policy(network_policy)
        ports = list(exposed_ports or [])
        logger.info(
            f"Creating Modal VM sandbox (vm_runtime; app={app_name}, tags={app_tags}, "
            f"sandbox_tags={sandbox_tags}, "
            f"ports={ports}, cpu={cpu}, memory={memory}MB, timeout={timeout}s, "
            f"policy={effective.mode.value})"
        )

        # No retry around create: retrying a create can orphan a half-provisioned sandbox
        # (retry_transient is documented for read-only calls only). Modal's client already
        # retries RESOURCE_EXHAUSTED / queueing internally, which is the real high-load case.
        try:
            sb = await modal.Sandbox._experimental_create.aio(
                *_KEEPALIVE_CMD,
                app=app, image=_VM_IMAGE, cpu=cpu, memory=memory, timeout=timeout,
                encrypted_ports=ports, experimental_options={"vm_runtime": True}, client=client,
                tags=sandbox_tags or None,
                **_modal_network_kwargs(effective),
            )
        except Exception as e:
            raise RuntimeError(f"Modal VM sandbox create failed [{type(e).__name__}: {e}]") from e

        sandbox: ModalVmSandbox | None = None
        try:
            tunnels = await sb.tunnels.aio(timeout=150)
            tunnel_urls = {p: _tunnel_url(t) for p, t in tunnels.items()}
            for p in ports:
                if p not in tunnel_urls:
                    raise RuntimeError(f"No Modal tunnel for exposed port {p}; got {sorted(tunnel_urls)}")
            sandbox = ModalVmSandbox(sb, tunnel_urls, network_policy=effective)
            logger.info(f"Modal VM sandbox created: {sb.object_id} (tunnels={tunnel_urls})")
            await _log_sandbox_started(
                sb, app_name=app_name, sandbox_tags=sandbox_tags,
                cpu=cpu, memory=memory, gpu=None,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            try:
                await sb.terminate.aio()
            except Exception as e:
                logger.warning(f"Failed to terminate Modal VM sandbox {sb.object_id} after failure: {e}")
            raise

    async def create_sandbox(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 0.5,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> ModalVmSandbox:
        # Like the other VM providers' create_sandbox: return a bare VM (image_name/env are
        # ignored at create). The caller sees mode == "vm" and loads + runs the image
        # itself (A2AAgent.deploy).
        return await self.create_vm(
            exposed_ports=[port], cpu=cpu, memory=memory, disk_size_gb=disk_size_gb,
            timeout=timeout,
            attribution=attribution,
            network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> ModalVmSandbox:
        client = await self._get_client()
        sb = await modal.Sandbox.from_id.aio(sandbox_id, client=client)
        tunnels = await sb.tunnels.aio()
        tunnel_urls = {p: _tunnel_url(t) for p, t in tunnels.items()}
        # dockerd is already running on a live sandbox; do not re-run setup here.
        return ModalVmSandbox(sb, tunnel_urls)


# ── Object downloads (internal) ────────────────────────────────────────────────────
# Object downloads run as one aria2c per object: a pool of HTTP range connections with small pieces,
# so a single slow TCP flow only ever holds one piece instead of the whole object; resume via the
# .aria2 control file; bounded by `timeout`. Modal-only: this class owns _VM_IMAGE, so aria2c is
# present by construction. Defaults are overridable as AGENT_ENV_MODAL_VM_DOWNLOAD_<NAME>.
_DL_CONNECTIONS = 16         # range connections per object (aria2c's cap); throughput under loss scales with it
_DL_MIN_SPLIT_MIB = 4        # piece size: a slow connection's tail is one piece
_DL_SPEED_FLOOR_KIB_S = 0    # aria2c's --lowest-speed-limit aborts the whole download, not one connection; keep off
_FRESH_URL_SECONDS = 60      # a signed URL older than this is re-signed before its first attempt
_DL_ATTEMPTS = 3             # each with a freshly minted URL, resuming
_DL_PARALLEL_FILES = 4       # concurrent downloads per VM
_DL_DEADLINE_S = 1800        # wall clock per attempt
_DL_ENV_PREFIX = "AGENT_ENV_MODAL_VM_DOWNLOAD_"
_DL_SUMMARY = "AGENTENV_DOWNLOAD"
_DL_SUMMARY_RE = re.compile(rf"^{_DL_SUMMARY} rc=(-?\d+) bytes=(\d+) seconds=(\d+)(?: msg=(.*))?$")


def _dl_setting(name: str, default):
    """`default`, or its AGENT_ENV_MODAL_VM_DOWNLOAD_<name> override coerced to the same type."""
    raw = os.environ.get(_DL_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return type(default)(raw)
    except ValueError as e:
        raise ValueError(f"{_DL_ENV_PREFIX}{name}={raw!r} is not a valid {type(default).__name__}") from e


def _render_download_script(url: str, vm_path: str, *, fresh: bool) -> str:
    """One `bash -c` script: aria2c under `timeout`; always exits 0 and reports through its last line."""
    connections = max(1, min(16, _dl_setting("CONNECTIONS", _DL_CONNECTIONS)))
    aria2 = (
        f"timeout -k 15 {_dl_setting('DEADLINE_S', _DL_DEADLINE_S)} aria2c --console-log-level=warn"
        " --summary-interval=0 --show-console-readout=false --download-result=hide --allow-overwrite=true --auto-file-renaming=false"
        " --continue=true --file-allocation=none --remote-time=false --async-dns=false"
        " --http-accept-gzip=false --max-file-not-found=2"
        f" --max-connection-per-server={connections} --split={connections}"
        f" --min-split-size={_dl_setting('MIN_SPLIT_MIB', _DL_MIN_SPLIT_MIB)}M"
        f" --lowest-speed-limit={_dl_setting('SPEED_FLOOR_KIB_S', _DL_SPEED_FLOOR_KIB_S) * 1024}"
        " --max-tries=8 --retry-wait=1 --timeout=60 --connect-timeout=10"
        ' --dir="$dir" --out="$name" "$url"'
    )
    return "\n".join([
        "set +e",
        "url=" + shlex.quote(url),
        "out=" + shlex.quote(vm_path),
        'dir=$(dirname "$out"); name=$(basename "$out"); log="$dir/.$name.download.log"',
        'mkdir -p "$dir"; t0=$(date +%s)',
        # The first attempt for an object must not --continue onto another object's bytes at the same path.
        *(['rm -f "$out" "$out.aria2"'] if fresh else []),
        f'{aria2} >"$log" 2>&1; rc=$?',
        'bytes=$(wc -c <"$out" 2>/dev/null | tr -d " "); [ -n "$bytes" ] || bytes=0',
        # Mask URLs and long tokens before truncating, or a tail starting mid-URL keeps a credential fragment.
        "msg=$(tr '\\n\\r' '  ' <\"$log\" 2>/dev/null | sed -E 's#https?://[^[:space:]]+#<url>#g; s#[A-Za-z0-9%+/=_-]{40,}#<token>#g' | tail -c 400)",
        'rm -f "$log"',
        f'echo "{_DL_SUMMARY} rc=$rc bytes=$bytes seconds=$(( $(date +%s) - t0 )) msg=$msg"',
        "exit 0",
    ])


def _parse_download_summary(stdout: str) -> Optional[tuple[int, int, int, str]]:
    """(rc, bytes, seconds, msg) from the script's last summary line; None if it never got there."""
    for line in reversed(stdout.splitlines()):
        if m := _DL_SUMMARY_RE.match(line.strip()):
            return int(m[1]), int(m[2]), int(m[3]), (m[4] or "").strip()
    return None


def _not_found(rc: int, msg: str) -> bool:
    """aria2c exit 3 = resource not found, 4 = --max-file-not-found reached."""
    return rc in (3, 4) or "status=404" in msg
