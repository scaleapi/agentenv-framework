"""``lume``: macOS computers on this Mac, as VM bundles Lume runs. A deploy's VM is an APFS clone of its computer's VM
bundle, pulled once when that's a registry image. The deploy's environment reaches it as the protocol's seed, in a
folder the VM shares; each exposed port is fronted by a relay on this Mac, run by Apple's ``/usr/bin/python3``, since
macOS's Local Network privacy lets only Apple's own binaries reach a VM's address; and a reset clones it afresh under the
same MAC address, so its address and its URLs stay. Virtualization.framework runs a VM only in a logged-in session, so
agent-env runs in one, and runs two macOS VMs on a Mac, as Apple's licence allows. Composing boots a builder: a clone
under the VM bundle's name with a build folder shared read-write, its commands run over ``lume ssh``, then shut down and
kept (``open_builder``, ``keep_builder``)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, ClassVar, Optional

from agentenv_protocol.computer import seed

from agent_env.artifact.artifacts.computer import LOCAL_IMAGE_PREFIX
from agent_env.attribution import Attribution
from agent_env.config import get_config
from agent_env.config.paths import state_root
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
    refuse_unenforceable_policy,
)
from agent_env.store.image_store.registry_api import pin_digest
from agent_env.store.local_state import awaiting_lock

logger = logging.getLogger(__name__)

#: Apple's macOS licence allows two macOS VMs on a Mac, and Virtualization.framework runs no more.
MAX_RUNNING = 2
#: Where a VM mounts each folder it shares, by the folder's name: the seed's is SEED_FOLDER, a builder's BUILD_FOLDER.
SHARED_FILES = "/Volumes/My Shared Files"
SEED_FOLDER = "agentenv-seed"
BUILD_FOLDER = "agentenv-build"
#: The relay that fronts a VM's ports: the local provider's, which Apple's Python runs as it is.
RELAY = Path(__file__).resolve().parent.parent / "local_docker" / "relay" / "relay.py"
#: The prefix of the names of the VMs it clones for deploys, and of the bases it pulls.
SANDBOX_PREFIX, BASE_PREFIX = "agentenv-", "agentenv-base-"
_POLL_S = 2.0
#: What Virtualization.framework says when it's asked to run a VM outside a logged-in session.
_NO_SESSION = "security information"
#: The variable the ssh password reaches ssh's askpass in.
ASKPASS_VARIABLE = "AGENTENV_LUME_SSH_PASSWORD"
#: A log line lume prints among its output.
_LOG_LINE = re.compile(r"^\[\d{4}-\d\d-\d\dT[^\]]*\] [A-Z]+: ")


class LumeSandbox(VmSandbox):
    """One macOS VM: ``exec`` runs a command in it over ``lume ssh``, as the computer's session user."""

    type = "lume"
    mode = "vm"
    vnc_url = None
    platform = None
    ON_THIS_MACHINE: ClassVar[bool] = True

    def __init__(self, provider: LumeSandboxProvider, sandbox_id: str, tunnel_urls: dict[int, str]) -> None:
        self._provider, self.sandbox_id, self.tunnel_urls = provider, sandbox_id, tunnel_urls

    async def terminate(self) -> None:
        await self._provider.terminate_sandbox(self.sandbox_id)

    async def exec(self, *command: str) -> asyncio.subprocess.Process:
        return await self._provider._ssh(self.sandbox_id, list(command))

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        """``script`` run by bash as the session user, whose sudo asks for the password the provider holds."""
        code, stdout, stderr = await self.exec_with_output("bash", "-c", script)
        if code != 0:
            raise RuntimeError(f"Script failed (exit {code}):\nstdout: {stdout[-1500:]}\nstderr: {stderr[-1500:]}")
        return stdout


class LumeSandboxProvider(SandboxProvider):
    """VM bundles of macOS on this Mac, run by Lume, configured under ``[sandbox.providers.lume.config]``: the session
    user's ``ssh_user`` and ``ssh_password`` (an ``env:`` or ``secret:`` reference), which composing and exec log in
    with, the password handed to ssh through its environment, never on a command line; ``max_running``, at most
    ``MAX_RUNNING``; ``relay_host``, the address the relays listen on, this Mac's loopback unless something off it must
    reach the computers, since a computer's MCP endpoint takes no token; and the ``lume``, ``ssh`` and ``relay_python``
    executables, Apple's own ssh and Python, which Local Network privacy lets reach a VM."""

    ON_THIS_MACHINE: ClassVar[bool] = True

    def __init__(self, *, ssh_user: str = "lume", ssh_password: str = "lume", max_running: int = MAX_RUNNING,
                 relay_host: str = "127.0.0.1", lume: str = "lume", ssh: str = "/usr/bin/ssh",
                 relay_python: str = "/usr/bin/python3", boot_timeout_s: float = 600) -> None:
        if not 1 <= max_running <= MAX_RUNNING:
            raise ValueError(f"max_running is 1 to {MAX_RUNNING}, the macOS VMs Apple's licence allows on a Mac, not "
                             f"{max_running}")
        self.ssh_user, self.ssh_password, self.max_running = ssh_user, ssh_password, max_running
        self.relay_host = relay_host
        self.lume, self.ssh, self.relay_python, self.boot_timeout_s = lume, ssh, relay_python, boot_timeout_s

    @classmethod
    def from_config(cls, **config: Any) -> LumeSandboxProvider:
        if "ssh_password" in config:
            config["ssh_password"] = get_config().resolve_reference(config["ssh_password"])
        return cls(**config)

    @classmethod
    def native_platforms(cls) -> frozenset[str]:
        """None: it runs VM bundles, not containers."""
        return frozenset()

    @classmethod
    def vm_bundle_platforms(cls) -> frozenset[str]:
        """``macos/arm64`` on a Mac with Apple silicon, the one machine a macOS VM bundle boots on."""
        return frozenset({"macos/arm64"} if sys.platform == "darwin" and platform.machine() == "arm64" else ())

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        return not policy.restricts_egress()

    async def create_vm(self, *, image: Optional[str] = None, boot_mode: Optional[str] = None,
                        cpu: Optional[float] = None, memory: Optional[int] = None, disk_size_gb: float = 10,
                        timeout: int = 3600 * 2, exposed_ports: Optional[list[int]] = None,
                        setup_for_gateway: bool = True, attribution: Optional[Attribution] = None,
                        network_policy: Optional[NetworkPolicy] = None,
                        env: Optional[dict[str, str]] = None) -> LumeSandbox:
        """A VM of the VM bundle ``image``, with ``env`` in its seed and each of ``exposed_ports`` relayed from this
        Mac's loopback, sized ``cpu`` and ``memory`` (MB) when given, else as the bundle is."""
        refuse_unenforceable_policy(self, network_policy)
        if image is None:
            raise ValueError("lume boots a computer's VM bundle, and was given none")
        source = await self._source(image)
        sandbox_id = f"{SANDBOX_PREFIX}{uuid.uuid4().hex[:12]}"
        async with self._admitting():
            await self._refuse_over_limit()
            await self._lume("clone", source, sandbox_id)
            try:
                state = {"source": source, "image": image, "mac": (await self._get(sandbox_id))["macAddress"],
                         "cpu": cpu, "memory": memory, "ports": {str(port): _free_port() for port in exposed_ports or []}}
                self._save(sandbox_id, state)
                await self._started(sandbox_id, state, env or {})
            except BaseException:
                await self.terminate_sandbox(sandbox_id)
                raise
        sandbox = self._sandbox(sandbox_id, state)
        sandbox.network_policy = self.effective_network_policy(network_policy)
        return sandbox

    async def reset_vm(self, sandbox: Any, *, image: str, env: dict[str, str]) -> None:
        """The VM cloned afresh from the bundle it was cloned from, under its MAC address, so it takes the same
        address, with ``env`` in its seed, its relays pointed at it on the same ports."""
        state = self._state(sandbox.sandbox_id)
        await self._stopped(sandbox.sandbox_id, state)
        await self._lume("delete", sandbox.sandbox_id, "--force")
        await self._lume("clone", state["source"], sandbox.sandbox_id)
        await self._lume("set", sandbox.sandbox_id, "--mac-address", state["mac"])
        await self._started(sandbox.sandbox_id, state, env)

    async def get_sandbox(self, sandbox_id: str) -> LumeSandbox:
        return self._sandbox(sandbox_id, self._state(sandbox_id))

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        """The VM stopped and deleted, and its relays and its folder gone; one that's gone already is terminated."""
        try:
            state = self._state(sandbox_id)
        except SandboxNotFoundError:
            state = {}
        await self._stopped(sandbox_id, state)
        with contextlib.suppress(RuntimeError):
            await self._lume("delete", sandbox_id, "--force")
        shutil.rmtree(self._folder(sandbox_id), ignore_errors=True)

    async def open_builder(self, image: str, name: str, build: Path) -> LumeSandbox:
        """A VM to compose the VM bundle ``name`` in: a clone of ``image`` under that name, any earlier one of that name
        deleted, with ``build``, a folder named ``BUILD_FOLDER``, shared read-write at ``SHARED_FILES/BUILD_FOLDER``,
        booted and answering ssh."""
        if build.name != BUILD_FOLDER:
            raise ValueError(f"a builder's folder is named {BUILD_FOLDER}, which the VM mounts it under, not {build.name}")
        source = await self._source(image)
        with contextlib.suppress(RuntimeError):
            await self._lume("delete", name, "--force")
        state = {"source": source, "image": image, "ports": {}, "build": str(build)}
        async with self._admitting():
            await self._refuse_over_limit()
            await self._lume("clone", source, name)
            try:
                self._save(name, state)
                await self._started(name, state, {})
            except BaseException:
                await self.terminate_sandbox(name)
                raise
        try:
            await self._answering(name)
        except BaseException:
            await self.terminate_sandbox(name)
            raise
        return self._sandbox(name, state)

    async def keep_builder(self, builder: LumeSandbox) -> str:
        """``builder`` shut down from inside, once what it wrote is on its disk, and kept as the VM bundle of its name:
        its ``local:`` image. A VM stopped from outside can lose its last writes, as it was before it shut down."""
        name = builder.sandbox_id
        password = f"{SHARED_FILES}/{BUILD_FOLDER}/.password"
        shutdown = await self._ssh(name, ["sh", "-c", f"sync; sudo -S -p '' shutdown -h now < {shlex.quote(password)}"],
                                   timeout=60)
        await shutdown.wait()
        deadline = time.monotonic() + self.boot_timeout_s
        while (await self._get(name)).get("status") != "stopped":
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name} didn't shut down within {self.boot_timeout_s:.0f} s, so it isn't kept")
            await asyncio.sleep(_POLL_S)
        await self._stopped(name, self._state(name))
        shutil.rmtree(self._folder(name), ignore_errors=True)
        return f"{LOCAL_IMAGE_PREFIX}{name}"

    async def delete_vm_bundle(self, image: str) -> None:
        """``image``'s VM bundle deleted from this Mac, as one that failed its probe is."""
        with contextlib.suppress(RuntimeError):
            await self._lume("delete", image.removeprefix(LOCAL_IMAGE_PREFIX), "--force")

    async def has_vm_bundle(self, image: str) -> bool:
        """Whether this Mac holds ``image``'s VM bundle, as a composed one is reused by its name."""
        name = image.removeprefix(LOCAL_IMAGE_PREFIX)
        try:
            return (await self._get(name))["status"] == "stopped"
        except RuntimeError:
            return False

    # Internals.

    async def _source(self, image: str) -> str:
        """The VM on this Mac a sandbox of ``image`` is cloned from: the local one it names, or the registry image it
        pins, pulled once under a name of its digest, once its tag is checked to name that digest still."""
        if image.startswith(LOCAL_IMAGE_PREFIX):
            name = image.removeprefix(LOCAL_IMAGE_PREFIX)
            try:
                await self._get(name)
            except RuntimeError:
                raise ValueError(f"this Mac has no VM bundle named {name!r}; compose it here with agent-env env build "
                                 "or copy it under ~/.lume") from None
            return name
        repository, _, digest = image.partition("@")
        name = f"{BASE_PREFIX}{digest.removeprefix('sha256:')[:12]}"
        async with self._admitting():
            try:
                await self._get(name)
                return name
            except RuntimeError:
                pass
            if ":" not in repository.rsplit("/", 1)[-1]:
                raise ValueError(f"{image}: Lume pulls by tag, so a VM bundle's registry image names its tag beside its "
                                 "digest (<repository>:<tag>@sha256:<digest>)")
            await asyncio.to_thread(pin_digest, image, None)
            host, path = repository.split("/", 1)
            organization, _, tagged = path.rpartition("/")
            logger.info("lume: pulling %s into %s, once", image, name)
            await self._lume("pull", tagged, name, "--registry", host, "--organization", organization, timeout=None)
            try:
                await asyncio.to_thread(pin_digest, image, None)
            except ValueError as e:
                await self._lume("delete", name, "--force")
                raise ValueError(f"{image}: its tag moved while it was pulled, so what was pulled is deleted: {e}") from None
        return name

    async def _started(self, name: str, state: dict, env: dict[str, str]) -> None:
        """``name`` sized as ``state`` says, its seed written from ``env``, booted, and relayed once it has an address."""
        sizes = [*(("--cpu", str(max(1, round(state["cpu"])))) if state.get("cpu") else ()),
                 *(("--memory", f"{int(state['memory'])}MB") if state.get("memory") else ())]
        display = env.get("AGENTENV_SETTING_RESOLUTION")
        if display:
            sizes += ["--display", display]
        if sizes:
            await self._lume("set", name, *[part for pair in zip(sizes[::2], sizes[1::2]) for part in pair])
        folder = self._folder(name)
        share = f"{state['build']}:rw" if state.get("build") else f"{self._seed(folder, env)}:ro"
        log = folder / "run.log"
        with open(log, "ab") as out:
            run = subprocess.Popen([self.lume, "run", name, "--display", "none", "--vnc", "disabled",
                                    f"--shared-dir={share}"],
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        state["run"] = run.pid
        self._save(name, state)
        address = await self._address(name, run, log)
        relays = {int(port): local for port, local in state["ports"].items()}
        if relays:
            relay = subprocess.Popen([self.relay_python, "-I", str(RELAY),
                                      *(f"{self.relay_host}:{local}={address}:{port}" for port, local in relays.items())],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
            state["relay"] = relay.pid
        state["address"] = address
        self._save(name, state)

    async def _address(self, name: str, run: subprocess.Popen, log: Path) -> str:
        """``name``'s address once Lume says it's running with one, within ``boot_timeout_s``; a run that exits first
        fails, with the end of its log."""
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if run.poll() is not None:
                said = log.read_text(errors="replace")[-1500:]
                hint = (" Virtualization.framework runs a VM only in a logged-in session: run agent-env in one, such "
                        "as through launchctl asuser" if _NO_SESSION in said else "")
                raise RuntimeError(f"lume run {name} exited {run.returncode}:{hint}\n{said}")
            info = await self._get(name)
            if info.get("status") == "running" and info.get("ipAddress"):
                return info["ipAddress"]
            await asyncio.sleep(_POLL_S)
        raise TimeoutError(f"{name} had no address within {self.boot_timeout_s:.0f} s")

    async def _answering(self, name: str) -> None:
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            process = await self._ssh(name, ["true"], timeout=15)
            if await process.wait() == 0:
                return
            await asyncio.sleep(_POLL_S)
        raise TimeoutError(f"{name} didn't answer ssh within {self.boot_timeout_s:.0f} s")

    async def _stopped(self, name: str, state: dict) -> None:
        """``name``'s relays ended and the VM stopped."""
        if pid := state.get("relay"):
            _kill(pid)
        with contextlib.suppress(RuntimeError):
            await self._lume("stop", name, "--force")
        if pid := state.get("run"):
            _kill(pid)

    async def _refuse_over_limit(self) -> None:
        listed = _json(await self._lume("ls", "--format", "json")) or []
        running = [vm["name"] for vm in listed if vm.get("status") not in ("stopped", None)]
        if len(running) >= self.max_running:
            raise RuntimeError(f"this Mac runs {len(running)} macOS VMs already ({', '.join(running)}), and lume is "
                               f"configured to run at most {self.max_running} (Apple's licence allows {MAX_RUNNING})")

    async def _get(self, name: str) -> dict:
        found = _json(await self._lume("get", name, "--format", "json"))
        return found[0] if isinstance(found, list) else found

    async def _lume(self, *args: str, timeout: Optional[float] = 300) -> str:
        """``lume`` run with ``args``; its output, or a RuntimeError with what it said."""
        process = await asyncio.create_subprocess_exec(self.lume, *args, stdin=subprocess.DEVNULL,
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(f"lume {args[0]} took over {timeout:.0f} s") from None
        if process.returncode != 0:
            raise RuntimeError(f"lume {' '.join(args[:2])} exited {process.returncode}: "
                               f"{(stderr or stdout).decode(errors='replace').strip()[-1000:]}")
        return stdout.decode()

    async def _ssh(self, name: str, command: list[str], timeout: int = 0) -> asyncio.subprocess.Process:
        """``command`` run in ``name`` over ssh as the session user, its output piped; the password reaches ssh through
        ``ASKPASS``, which reads it from the environment. ``timeout`` bounds the connection, when it's given."""
        address = (await self._get(name)).get("ipAddress") or "0.0.0.0"
        options = ["StrictHostKeyChecking=no", "UserKnownHostsFile=/dev/null", "LogLevel=ERROR",
                   "NumberOfPasswordPrompts=1", *((f"ConnectTimeout={timeout}",) if timeout else ())]
        return await asyncio.create_subprocess_exec(
            self.ssh, *(part for option in options for part in ("-o", option)), f"{self.ssh_user}@{address}",
            shlex.join(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env={**os.environ, "SSH_ASKPASS": str(_askpass()), "SSH_ASKPASS_REQUIRE": "force",
                                         ASKPASS_VARIABLE: self.ssh_password})

    def _sandbox(self, name: str, state: dict) -> LumeSandbox:
        host = "127.0.0.1" if self.relay_host in ("0.0.0.0", "::") else self.relay_host
        return LumeSandbox(self, name, {int(port): f"http://{host}:{local}" for port, local in state["ports"].items()})

    @staticmethod
    def _seed(folder: Path, env: dict[str, str]) -> Path:
        """The folder of ``env``'s seed, as the protocol writes it: each version this agent-env writes, and the oldest
        of them."""
        shared = folder / SEED_FOLDER
        shutil.rmtree(shared, ignore_errors=True)
        shared.mkdir(mode=0o700)
        for version in seed.WRITTEN:
            _private(shared / seed.user_data_file(version), seed.user_data(env))
        _private(shared / seed.OLDEST, f"{seed.WRITTEN[0]}\n")
        return shared

    @staticmethod
    def _folder(name: str) -> Path:
        folder = state_root() / "lume" / name
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        return folder

    def _state(self, name: str) -> dict:
        try:
            return json.loads((state_root() / "lume" / name / "state.json").read_text())
        except FileNotFoundError:
            raise SandboxNotFoundError(f"lume made no sandbox {name!r} on this Mac") from None

    def _save(self, name: str, state: dict) -> None:
        _private(self._folder(name) / "state.json", json.dumps(state))

    def _admitting(self) -> AbstractAsyncContextManager[None]:
        """Held while a VM is counted, cloned and started, or a base pulled, by every agent-env on this Mac, so two
        deploys don't both take its last slot or pull one base twice; waited for on the event loop."""
        return awaiting_lock("lume-admitting")


def _json(text: str) -> Any:
    """The JSON ``lume`` printed, past the log lines it prints on the same stream (``[<time>] INFO: ...``)."""
    return json.loads("\n".join(line for line in text.splitlines() if not _LOG_LINE.match(line)) or "null")


def _askpass() -> Path:
    """The program ssh asks for the password, which prints it from the environment."""
    path = state_root() / "lume" / "askpass"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _private(path, f'#!/bin/sh\nprintf \'%s\\n\' "${ASKPASS_VARIABLE}"\n')
        path.chmod(0o700)
    return path


def _private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(text)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _kill(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGTERM)
