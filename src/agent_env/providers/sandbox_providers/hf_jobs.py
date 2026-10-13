"""Hugging Face Jobs sandbox provider: each container an HF Job, on a network group the provider's other private
containers share, its port exposed publicly, and commands run in it through HF's sandbox server, which each job starts
in place of its image's own command; the image's command then runs beside it, in the background."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import re
import secrets
import tempfile
import threading
import time
from typing import Any, ClassVar, Optional, Self

import httpx
import huggingface_hub
from huggingface_hub import HfApi, get_token
from huggingface_hub import Sandbox as HubSandbox
from huggingface_hub import _sandbox as hub_sandbox

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, Attribution
from agent_env.config import get_config
from agent_env.config.errors import ConfigError
from agent_env.env.gateway.constants import PRIVATE_HOST_ENV
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER,
    Accepts,
    SandboxProvider,
    apply_default_attribution,
    refuse_unenforceable_policy,
)
from agent_env.store.image_store.registry_api import image_config

logger = logging.getLogger(__name__)

_CONFIG_KEYS = ("token", "namespace", "flavor", "network_group")
_SERVER_PORT = hub_sandbox.SANDBOX_SERVER_PORT
_AGENTENV_LABEL = "agentenv"
_ALIAS_LABEL = "agentenv-alias"
# The image's own command writes here, which ``exec`` can read when a container doesn't come up.
_MAIN_LOG = "/tmp/agentenv-main.log"
_TERMINAL_STAGES = ("COMPLETED", "ERROR", "DELETED", "CANCELED")
# Pulling a large image dominates a job's start; HF caps no job's queueing, so this bounds it.
_RUNNING_TIMEOUT_SECONDS = 600
_EXEC_READY_TIMEOUT_SECONDS = 120
_EXEC_PROBE_TIMEOUT_SECONDS = 10
_POLL_SECONDS = 2
_POST_TIMEOUT_SECONDS = 60
_DOWNLOAD_TIMEOUT_SECONDS = 120
_ERROR_TEXT_MAX_CHARS = 500
# Random bytes, as hex: in the provider's network group's name, in each container's alias in it, and in each job's
# sandbox-token nonce.
_GROUP_ID_BYTES = 8
_ALIAS_BYTES = 6
_NONCE_BYTES = 16
_LABEL_MAX_LEN = 63
_LABEL_KEY = re.compile(rf"[a-zA-Z0-9._-]{{1,{_LABEL_MAX_LEN}}}")
_LABEL_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]")
# The private huggingface_hub helpers the provider calls, with the keywords it passes them; checked as it's built, so a
# release that changes them fails there rather than mid-deploy.
_HUB_HELPERS = {
    "_bootstrap_job_spec": {"env", "secrets", "volumes", "idle_timeout", "forward_hf_token", "sandbox_token"},
    "_derive_sandbox_token": set(),
    "_tail_job_logs": {"namespace"},
}
SANDBOX_STARTED_EVENT = "agent_env.hf_job_started"


class _Finished:
    """A command that has run to its end, as the process ``Sandbox.exec`` returns."""

    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self._exit_code = exit_code
        self.stdout = _Bytes(stdout)
        self.stderr = _Bytes(stderr)

    async def wait(self) -> int:
        return self._exit_code


class _Bytes:
    def __init__(self, text: str):
        self._data = text.encode()

    async def read(self) -> bytes:
        return self._data


class HfJobsSandbox(Sandbox):
    type = "hf_jobs"

    def __init__(self, provider: HfJobsSandboxProvider, job_id: str, tunnel_urls: dict[int, str],
                 box: HubSandbox | None = None, private_host: str | None = None,
                 network_policy: NetworkPolicy | None = None):
        self._provider = provider
        self.network_policy = network_policy
        self._box = box
        self.sandbox_id = job_id
        self.tunnel_urls = tunnel_urls
        self.private_host = private_host
        self.vnc_url = None
        self.mode = SANDBOX_MODE_CONTAINER

    async def terminate(self) -> None:
        await self._provider._cancel(self.sandbox_id)
        if self._box is not None:
            await asyncio.to_thread(self._box.close)

    async def exec(self, *command: str) -> _Finished:
        result = await asyncio.to_thread(self._connected().run, list(command), check=False)
        return _Finished(-1 if result.exit_code is None else result.exit_code, result.stdout, result.stderr)

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        await asyncio.to_thread(self._connected().files.write, destination_path, content)

    async def write_file_from_object(self, object_url: str, destination_path: str) -> None:
        def write() -> None:
            with contextlib.closing(get_config().get_object_store_at(object_url).open(object_url)) as body:
                self._connected().files.write(destination_path, body)

        await asyncio.to_thread(write)

    async def write_file_from_url(self, url: str, destination_path: str) -> None:
        with tempfile.TemporaryFile() as spool:
            async with httpx.AsyncClient(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        spool.write(chunk)
            spool.seek(0)
            await asyncio.to_thread(self._connected().files.write, destination_path, spool)

    def _connected(self) -> HubSandbox:
        if self._box is None:
            raise RuntimeError(f"HF job {self.sandbox_id} isn't running, so nothing can run in it")
        return self._box


class HfJobsSandboxProvider(SandboxProvider):
    """Runs each container as a Hugging Face Job, through the Jobs API with the configured token and billed to its
    namespace. The containers it puts on a private network join one network group, one per provider, where each is
    reached at its own alias; a container's port is exposed publicly, as Modal's tunnels are. ``cpu``, ``memory`` and
    ``disk_size_gb`` aren't per container: every job runs on the configured ``flavor``."""

    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.hf.jobs",)
    # Each image in a job of its own, pulled by its name: HF Jobs build nothing and take no registry credentials, so
    # its registry must serve it publicly.
    CONTAINER_ACCEPTS = Accepts.NAME
    PRIVATE_NETWORK = True
    # A network group routes IPv4, which a "::" listener misses: asyncio binds it to IPv6 alone.
    PRIVATE_LISTEN_HOST = "0.0.0.0"

    def __init__(self, token: str, namespace: str | None = None, flavor: str = "cpu-basic",
                 network_group: str | None = None):
        self._token = token
        self._api = HfApi(token=token)
        self._namespace = namespace
        self._flavor = flavor
        self._network_group = network_group or f"agentenv-{secrets.token_hex(_GROUP_ID_BYTES)}"

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        section = "[sandbox.providers.hf_jobs.config]"
        if unknown := set(config) - set(_CONFIG_KEYS):
            raise ConfigError(f"{section} has unknown key(s): {sorted(unknown)}")
        for key in _CONFIG_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{section} '{key}' must be a non-empty string")
        token = config.get("token") or get_token()
        if not token:
            raise ConfigError(f"{section} needs a 'token' (a secret: or env: reference), or a Hugging Face login: "
                              "`hf auth login`, or HF_TOKEN")
        if changed := _hub_helpers_changed():
            raise ConfigError(f"The hf_jobs provider runs on huggingface_hub's private sandbox helpers, which "
                              f"huggingface_hub {huggingface_hub.__version__} changed ({'; '.join(changed)}); install "
                              "one it runs on, such as 1.33")
        return cls(**{**config, "token": token})

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
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
        command: list[str] | None = None,
        expose_externally: bool = True,
        private_network: bool = False,
    ) -> Sandbox:
        refuse_unenforceable_policy(self, network_policy)
        effective = self.effective_network_policy(network_policy)
        launch = await asyncio.to_thread(_image_command, image_name, command)
        alias = f"c{secrets.token_hex(_ALIAS_BYTES)}" if private_network else None
        namespace = await self._resolved_namespace()
        spec = self._job_spec(image_name, port=port, env=env, timeout=timeout, attribution=attribution, alias=alias,
                              exposed=expose_externally)
        job_id = await _created(functools.partial(self._post_job, namespace, spec), self._cancel_job)
        try:
            return await self._started(job_id, namespace, port=port, launch=launch, alias=alias,
                                       exposed=expose_externally, image_name=image_name, spec=spec,
                                       network_policy=effective)
        except BaseException as e:  # a failed or cancelled start cancels its job, which would run to its timeout
            await self._cancel(job_id)
            if not isinstance(e, Exception):
                raise
            raise RuntimeError(f"HF job {job_id} for {image_name} didn't start: {type(e).__name__}: {e}") from e

    async def create_sandbox(
        self, *, image_name: str, port: int, env: dict[str, str],
        cpu: float = 1.0, memory: int = 8192, disk_size_gb: float = 10, timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> Sandbox:
        return await self.create_container(
            image_name=image_name, port=port, env=env, cpu=cpu, memory=memory, disk_size_gb=disk_size_gb,
            timeout=timeout, attribution=attribution, network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        namespace = await self._resolved_namespace()
        job = await asyncio.to_thread(self._api.inspect_job, job_id=sandbox_id, namespace=namespace)
        tunnel_urls = _tunnel_urls(job.status.expose_urls)
        if job.status.stage != "RUNNING":
            return HfJobsSandbox(self, sandbox_id, tunnel_urls)
        box = await asyncio.to_thread(HubSandbox.connect, sandbox_id, namespace=namespace, token=self._token)
        alias = (job.labels or {}).get(_ALIAS_LABEL)
        private_host = await _private_host(box, alias) if alias else None
        return HfJobsSandbox(self, sandbox_id, tunnel_urls, box=box, private_host=private_host)

    async def _started(self, job_id: str, namespace: str, *, port: int, launch: list[str], alias: str | None,
                       exposed: bool, image_name: str, spec: dict, network_policy: NetworkPolicy) -> HfJobsSandbox:
        job = await self._running(job_id, namespace)
        box = await asyncio.to_thread(HubSandbox.connect, job_id, namespace=namespace, token=self._token)
        await _exec_ready(box, job_id)
        private_host = await _private_host(box, alias) if alias else None
        await asyncio.to_thread(
            box.run, ["/bin/sh", "-c", f'exec "$@" >>{_MAIN_LOG} 2>&1', "sh", *launch], background=True,
            env={PRIVATE_HOST_ENV: private_host} if private_host else None,
        )
        tunnel_urls = _tunnel_urls(job.status.expose_urls)
        if exposed and port not in tunnel_urls:
            raise RuntimeError(f"HF exposes no URL for port {port}; it exposes {sorted(tunnel_urls)}")
        _log_started(job_id, image_name, spec, tunnel_urls, private_host)
        return HfJobsSandbox(self, job_id, tunnel_urls, box=box, private_host=private_host,
                             network_policy=network_policy)

    def _job_spec(self, image_name: str, *, port: int, env: dict[str, str], timeout: int,
                  attribution: Optional[Attribution], alias: str | None, exposed: bool) -> dict:
        """The Jobs API spec of a job that runs HF's sandbox server, whose bootstrap huggingface_hub builds (it pins the
        server's digest). The run's env goes in as secrets, kept out of the job's spec; every process gets it all the
        same."""
        nonce = secrets.token_hex(_NONCE_BYTES)
        command, job_env, job_secrets, volumes = hub_sandbox._bootstrap_job_spec(
            self._api, self._token, env=None, secrets=env, volumes=None, idle_timeout=None, forward_hf_token=False,
            sandbox_token=hub_sandbox._derive_sandbox_token(self._token, nonce),
        )
        labels = {
            **_attribution_labels(attribution),
            _AGENTENV_LABEL: "1",
            hub_sandbox.SANDBOX_LABEL: "1",
            hub_sandbox.MODE_LABEL: hub_sandbox.MODE_DEDICATED,
            hub_sandbox.NONCE_LABEL: nonce,
        }
        spec: dict[str, Any] = {
            "dockerImage": image_name,
            "command": command,
            "arguments": [],
            "environment": job_env,
            "secrets": job_secrets,
            "flavor": self._flavor,
            "timeoutSeconds": int(timeout),
            "labels": labels,
            "volumes": [volume.to_dict() for volume in volumes],
            "expose": {"ports": [_SERVER_PORT, port], "portsPublic": [port]} if exposed else {"ports": [_SERVER_PORT]},
        }
        if alias:
            labels[_ALIAS_LABEL] = alias
            spec["network"] = {"group": self._network_group, "aliases": [alias]}
        return spec

    def _post_job(self, namespace: str, spec: dict) -> str:
        response = httpx.post(f"{self._api.endpoint}/api/jobs/{namespace}", json=spec,
                              headers={"Authorization": f"Bearer {self._token}"}, timeout=_POST_TIMEOUT_SECONDS)
        if response.status_code >= 400:
            raise RuntimeError(f"HF Jobs refused the job for {spec['dockerImage']}: HTTP {response.status_code}: "
                               f"{response.text[:_ERROR_TEXT_MAX_CHARS]}")
        return response.json()["id"]

    async def _running(self, job_id: str, namespace: str):
        deadline = time.monotonic() + _RUNNING_TIMEOUT_SECONDS
        while True:
            job = await asyncio.to_thread(self._api.inspect_job, job_id=job_id, namespace=namespace)
            if job.status.stage == "RUNNING":
                return job
            if job.status.stage in _TERMINAL_STAGES:
                logs = await asyncio.to_thread(hub_sandbox._tail_job_logs, self._api, job_id, namespace=namespace)
                raise RuntimeError(f"it ended {job.status.stage} before running: {job.status.message or ''}{logs}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"it wasn't running after {_RUNNING_TIMEOUT_SECONDS} s (stage {job.status.stage})")
            await asyncio.sleep(_POLL_SECONDS)

    async def _cancel(self, job_id: str) -> None:
        await asyncio.to_thread(self._cancel_job, job_id)

    def _cancel_job(self, job_id: str) -> None:
        """Cancel ``job_id``; one already ended is left as it is."""
        try:
            self._api.cancel_job(job_id=job_id, namespace=self._namespace)
        except Exception as e:
            logger.warning(f"Couldn't cancel HF job {job_id}; it runs until its timeout unless it has ended: {e}")

    async def _resolved_namespace(self) -> str:
        if self._namespace is None:
            self._namespace = (await asyncio.to_thread(self._api.whoami))["name"]
        return self._namespace


async def _created(post, cancel_job) -> str:
    """The id of the job ``post`` creates. HF may create it even when this is cancelled, so the request runs on, and
    whichever ends second, the request or the cancel, cancels its job: when the cancel came first, the request's own
    thread does as it ends, with no event loop to outlive, so a loop's shutdown leaves nothing running."""
    lock = threading.Lock()
    state: dict[str, Any] = {"job_id": None, "abandoned": False}

    def post_and_record() -> str:
        job_id = post()
        with lock:
            state["job_id"] = job_id
            abandoned = state["abandoned"]
        if abandoned:
            cancel_job(job_id)
        return job_id

    try:
        return await asyncio.to_thread(post_and_record)
    except asyncio.CancelledError:
        with lock:
            state["abandoned"] = True
            job_id = state["job_id"]
        if job_id is not None:
            await asyncio.to_thread(cancel_job, job_id)
        raise


async def _exec_ready(box: HubSandbox, job_id: str) -> None:
    deadline = time.monotonic() + _EXEC_READY_TIMEOUT_SECONDS
    while True:
        try:
            await asyncio.to_thread(box.run, ["true"], timeout=_EXEC_PROBE_TIMEOUT_SECONDS)
            return
        except Exception as e:  # the server isn't up yet, or not yet reachable
            if time.monotonic() > deadline:
                raise TimeoutError(f"its sandbox server didn't answer within {_EXEC_READY_TIMEOUT_SECONDS} s: "
                                   f"{e}") from e
            await asyncio.sleep(_POLL_SECONDS)


async def _private_host(box: HubSandbox, alias: str) -> str:
    """The host ``alias`` is reached at in the job's network group: the group's prefix, which only its members are
    told, then the alias."""
    result = await asyncio.to_thread(box.run, ["sh", "-c", 'printf %s "$HF_NETWORK_GROUP_PREFIX"'], check=False)
    prefix = result.stdout.strip()
    if not prefix:
        raise RuntimeError("it's in no network group, so the provider's other containers can't reach it")
    return f"{prefix}{alias}"


def _hub_helpers_changed() -> list[str]:
    """How this huggingface_hub's private sandbox helpers differ from what the provider calls them with."""
    changed = []
    for name, keywords in _HUB_HELPERS.items():
        helper = getattr(hub_sandbox, name, None)
        if helper is None:
            changed.append(f"no {name}")
        elif missing := keywords - set(inspect.signature(helper).parameters):
            changed.append(f"{name} takes no {', '.join(sorted(missing))}")
    return changed


def _image_command(image_name: str, args: list[str] | None) -> list[str]:
    """The command ``image_name`` runs as Docker runs it, read from its registry: its ENTRYPOINT, then ``args`` in place
    of its CMD. The job's own command starts HF's sandbox server instead, which then starts this one."""
    auth = get_config().get_image_store_at(image_name).auth(image_name)
    config = image_config(image_name, auth)
    command = [*(config.get("Entrypoint") or []), *(args or config.get("Cmd") or [])]
    if not command:
        raise ValueError(f"{image_name} has no ENTRYPOINT or CMD to run")
    return command


def _tunnel_urls(expose_urls: list[str] | None) -> dict[int, str]:
    """The public URL of each port a job exposes, by port, but HF's sandbox server's."""
    urls = {}
    for url in expose_urls or []:
        host = url.split("://", 1)[-1].split("/", 1)[0].split(".", 1)[0]
        _, sep, port = host.rpartition("--")
        if sep and port.isdigit() and int(port) != _SERVER_PORT:
            urls[int(port)] = url.rstrip("/")
    return urls


def _attribution_labels(attribution: Optional[Attribution]) -> dict[str, str]:
    """``attribution``, with config.toml ``[sandbox.attribution]``'s defaults, as job labels: values cut to
    ``[a-zA-Z0-9._-]`` and 63 characters."""
    labels = {}
    for key, value in apply_default_attribution(dict(attribution or {})).items():
        if value is None or not _LABEL_KEY.fullmatch(key):
            continue
        labels[key] = _LABEL_UNSAFE.sub("-", str(value))[:_LABEL_MAX_LEN]
    return labels


def _log_started(job_id: str, image_name: str, spec: dict, tunnel_urls: dict[int, str],
                 private_host: str | None) -> None:
    """One structured line mapping the job to the run and step it's billed to."""
    labels = spec["labels"]
    logger.info(
        f"HF job started: job_id={job_id} image={image_name} flavor={spec['flavor']} urls={tunnel_urls}"
        + (f" private_host={private_host}" if private_host else ""),
        extra={
            "event": SANDBOX_STARTED_EVENT,
            "hf_job_id": job_id,
            "hf_flavor": spec["flavor"],
            PIPELINE_STEP_KEY: labels.get(PIPELINE_STEP_KEY),
            RUN_ID_KEY: labels.get(RUN_ID_KEY),
        },
    )
