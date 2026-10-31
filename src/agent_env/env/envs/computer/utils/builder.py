"""Where a computer env's packages are resolved and its images built and pushed (``composer``): this machine's Docker
for the local sandbox provider, else a VM of the put's sandbox provider, which fetches the files the put stored by
signed URL, takes the rest inline, and pushes straight to the image store. A builder has a work folder of its own, which
holds the build context in ``CONTEXT`` and a resolve in ``RESOLVED``; the paths it takes are relative to it."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import tarfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import httpx

from agent_env.artifact.artifacts.docker_image import docker_login, push_local_image
from agent_env.config import get_config
from agent_env.env.envs.computer.utils import layers_in_image
from agent_env.providers.sandbox_providers.host_images import build_for, host_platform, platform_tag
from agent_env.providers.sandbox_providers.sandbox import upload_vm_file
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM
from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.image_store.oci_registry_credentials import (
    is_loopback_host, names_registry, registry_host_from_ref)
from agent_env.utils.docker.build import DockerBuildError, build_image

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
    from agent_env.providers.sandbox_providers.sandbox import VmSandbox
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

CONTEXT, RESOLVED = "context", "resolve"
RESOLVER = Path(__file__).with_name("resolve_in_base.py")
LAYERS = Path(__file__).with_name("layers_in_image.py")
#: The shell side of a builder VM, its subcommands run there (``VmBuilder``).
IN_VM = Path(__file__).with_name("builder_in_vm.sh")
#: What a file starts with when it's only ever run: a script or a Linux binary, made executable where it's staged,
#: since the stores keep no file modes.
RUN_ONLY = (b"#!", b"\x7fELF")
EXECUTABLE = 0o755
#: What registers QEMU in a builder VM's kernel, for a platform no link runs natively: Docker Hub's image through
#: Google's mirror of it, which doesn't limit anonymous pulls by IP as Docker Hub does.
BINFMT = ("mirror.gcr.io/tonistiigi/binfmt:qemu-v9.2.2"
          "@sha256:1b804311fe87047a4c96d38b4b3ef6f62fca8cd125265917a9e3dc3c996c39e6")
#: The BuildKit client a builder VM's Docker builds with, installed where it has none, by the VM's platform.
BUILDX = "https://github.com/docker/buildx/releases/download/v0.38.0/buildx-v0.38.0.{platform}"
BUILDX_SHA256 = {"linux/amd64": "4fe4cc38adf48169132749b6ca22a990928db0118e3407584ee553723115d287",
                 "linux/arm64": "38e890e1a162bfdbf32fbe91404991dc4fc28e55b0de901b34374a061bab4d2c"}
_PLUGINS = "/usr/local/lib/docker/cli-plugins"
#: A builder VM: a build runs apt, pip and an app's install steps, and holds the base, the lock's files and the image.
_VM = {"cpu": 4.0, "memory": 8192, "disk_size_gb": 40}
#: How long a builder VM may run, natively and when it emulates the arch, which runs some 10-20 times slower.
VM_TIMEOUT_S, EMULATED_VM_TIMEOUT_S = 7200, 21600
_DOWNLOAD_TIMEOUT_S = 600
_FETCHES_IN_FLIGHT = 8
_CHUNK = 1 << 20


class ResolveError(RuntimeError):
    """Resolving or fetching failed; ``log`` holds what the resolver printed."""

    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log


class Unbuildable(RuntimeError):
    """No link of the put's sandbox provider builds a platform; the message says why for each."""


class Skipped(Exception):
    """No link of the put's sandbox provider runs a platform natively, and the put turned emulation off."""


@dataclass(frozen=True)
class Hashed:
    sha256: str
    size: int


class Builder(ABC):
    #: The provider it's from, as a pair's record names it, and the platform its Docker runs natively.
    name: str
    platform: str

    @abstractmethod
    def resolve(self, request: dict, source: str, arch: str, timeout: int) -> dict:
        """Run the resolver on ``request`` in a container of ``source`` for ``arch``, as root, with ``CONTEXT``
        read-only at /ctx, writing into ``RESOLVED``/out; what it says it wrote. Raises ResolveError."""

    @abstractmethod
    def download(self, url: str, sha256: str, path: str) -> None:
        """Download ``url`` to ``path``, refused (ResolveError) unless its sha256 is ``sha256``."""

    @abstractmethod
    def files(self, folder: str) -> dict[str, Hashed]:
        """Each plain file under ``folder``, never a link, by its path there: a dependency's build code, run in the
        resolver's container, could leave a link to one of the builder's own files instead."""

    @abstractmethod
    def store(self, path: str, store, key: str) -> None:
        """Put the file at ``path`` in the object store ``store`` at ``key``."""

    @abstractmethod
    def place(self, copies: dict[str, str], fetches: dict[str, tuple[str, str]]) -> None:
        """Put at each path of ``copies`` a link to the file it names, and at each of ``fetches`` the object store's
        object at its URL, refused unless its sha256 is the one given."""

    @abstractmethod
    def mark_runnable(self, paths: list[str]) -> None:
        """Make each of ``paths`` executable that starts as a script or a Linux binary does."""

    @abstractmethod
    def build(self, dockerfile: str, tag: str, platform: str, labels: dict[str, str], context: str = CONTEXT) -> None:
        """Build ``context`` with ``dockerfile``, written into it, as the image ``tag`` for ``platform``. Raises
        DockerBuildError."""

    @abstractmethod
    def installed(self, tag: str, prefix: str) -> dict[str, str]:
        """The packages ``tag``'s build installed into the venv of the driver at ``prefix``, by name, read from the
        image without running it, so on any arch; none for a driver with no venv."""

    @abstractmethod
    def written(self, tag: str, commands: list[str]) -> list[dict[str, str]]:
        """What each of ``commands``, a RUN of ``tag``'s build, wrote in its layer of ``tag``, read from a save of the
        image (``layers_in_image``)."""

    @abstractmethod
    def push(self, tag: str, ref: str, store) -> None:
        """Push ``tag`` to ``ref`` in the image store ``store``, signed in with its credentials."""

    def close(self) -> None:
        pass


class Builds:
    """The builders the put's sandbox provider gives: for each platform, from the first of its links that runs the
    platform natively, else, unless ``emulate`` is False, from the first that emulates it. A link builds when it's this
    machine's Docker, or gives VMs that reach the stores ``store_id`` routes to and the registry ``repository`` is in."""

    def __init__(self, provider: SandboxProvider, store_id: str, repository: str, emulate: bool = True):
        self.provider, self.emulate = provider, emulate
        self._refused = [(link, _refused(link, store_id, repository)) for link in provider.links]

    def link(self, platform: str) -> SandboxProvider:
        """The link that builds ``platform``. Raises Unbuildable, or Skipped when only emulation would build it and the
        put turned emulation off."""
        links = [link for link, problem in self._refused if problem is None]
        if native := [link for link in links if platform in link.native_platforms()]:
            return native[0]
        if not self.emulate:
            if refused := [problem for link, problem in self._refused
                           if problem is not None and platform in link.native_platforms()]:
                raise Unbuildable("; ".join(refused))
            raise Skipped("emulation is off, and " + "; ".join(
                f"{type(link).__name__} doesn't run {platform} natively" for link, _ in self._refused))
        for link in links:
            if not _docker_here(link) or platform in _emulated_platforms():
                return link
        unemulated = f"this machine can't build {platform}: its Docker has no emulation for it"
        raise Unbuildable("; ".join(problem or unemulated for _, problem in self._refused))

    @contextlib.contextmanager
    def open(self, link: SandboxProvider, platform: str, work: Path, base: DockerImageArtifact,
             from_store: dict[str, tuple[str, str]]) -> Iterator[Builder]:
        """A builder from ``link`` of ``platform``, able to build ``FROM`` ``base``, holding the build context the
        composer staged in ``work``, the files of ``from_store`` (by path, their object URL and sha256) fetched from
        the store rather than sent; torn down when the block ends."""
        if _docker_here(link):
            yield LocalBuilder(type(link).__name__, work, base, platform)
            return
        vm = VmBuilder(link, emulating=platform not in link.native_platforms())
        try:
            vm.start(work, base, platform, from_store)
            yield vm
        finally:
            vm.close()


def _refused(link: SandboxProvider, store_id: str, repository: str) -> str | None:
    """Why ``link`` can't build, or None when it can."""
    name = type(link).__name__
    if _docker_here(link):
        return None
    if SANDBOX_MODE_VM not in link.SUPPORTED_SANDBOX_MODES:
        return f"{name} gives no VM to build in"
    if link.ON_THIS_MACHINE:
        return None
    config = get_config()
    objects = config.get_object_store_for(store_id)
    if objects.signed_get_url(objects.object_url(f"{config.get_artifact_key_prefix()}agentenv-packages")) is None:
        return f"{name}'s VMs can't fetch from the object store, which signs no URLs; build on the local provider"
    if is_loopback_host(registry_host_from_ref(config.get_image_store_for(store_id).image_ref(repository, "key"))):
        return f"{name}'s VMs can't push to the image store's registry on this machine; build on the local provider"
    return None


def where(link: SandboxProvider) -> str:
    """Where ``link`` builds, for a person to read."""
    return "with this machine's Docker" if _docker_here(link) else f"in a {type(link).__name__} VM"


def instead_of_emulating(provider: SandboxProvider, platform: str) -> str:
    """What a person can do rather than wait for ``provider`` to emulate ``platform``: skip it, or, when this machine
    runs it and the chain has no link on this machine's Docker, add one."""
    local = "add local to --sandbox to build it natively with this machine's Docker, or " if (
        platform == host_platform() and not any(_docker_here(link) for link in provider.links)) else ""
    return f"{local}pass --no-emulate to skip it"


def _docker_here(link: SandboxProvider) -> bool:
    """Whether ``link``'s sandboxes are containers on this machine's Docker, which then builds itself."""
    return link.ON_THIS_MACHINE and link.REACHES_HOST_AT is None


def _emulated_platforms() -> set[str]:
    """The platforms this machine's Docker builds, its own and those it emulates."""
    inspected = subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], capture_output=True, text=True)
    platforms = set()
    for line in inspected.stdout.splitlines():
        if line.strip().startswith("Platforms:"):
            platforms |= {item.strip().rstrip("*") for item in line.split(":", 1)[1].split(",")}
    return platforms


def _resolver(source: str, arch: str, context: str, out: str) -> list[str]:
    return ["docker", "run", "--rm", "--platform", f"linux/{arch}", "--user", "root", "--entrypoint", "python3",
            "-v", f"{context}:/ctx:ro", "-v", f"{out}:/out", source, "/out/resolve.py", "/out/request.json", "/out"]


class LocalBuilder(Builder):
    """This machine's Docker, in the folder the composer staged the build context in."""

    def __init__(self, name: str, work: Path, base: DockerImageArtifact, platform: str):
        self.name, self.platform, self.work = name, host_platform(), work
        if names_registry(base.image_name):
            docker_login(get_config().get_image_store().auth(base.pinned_name))
        else:
            build_for(base, platform)

    def resolve(self, request: dict, source: str, arch: str, timeout: int) -> dict:
        out = self.work / RESOLVED / "out"
        out.mkdir(parents=True)
        (out / "request.json").write_text(json.dumps({**request, "owner": [os.getuid(), os.getgid()]}))
        shutil.copy(RESOLVER, out / "resolve.py")
        try:
            ran = subprocess.run(_resolver(source, arch, str(self.work / CONTEXT), str(out)), capture_output=True,
                                 text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ResolveError(f"resolving took over {timeout} s", str(e.stdout or "")) from None
        if ran.returncode != 0:
            raise ResolveError("resolving the packages failed", ran.stdout + ran.stderr)
        return json.loads((out / "result.json").read_text())

    def download(self, url: str, sha256: str, path: str) -> None:
        target, hashed = self.work / path, hashlib.sha256()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
                response.raise_for_status()
                with target.open("wb") as file:
                    for chunk in response.iter_bytes(_CHUNK):
                        hashed.update(chunk)
                        file.write(chunk)
        except httpx.HTTPError as e:
            raise ResolveError(f"downloading {url} failed: {e}") from None
        if hashed.hexdigest() != sha256:
            raise ResolveError(f"{url} has the sha256 {hashed.hexdigest()}, not the {sha256} its download names")

    def files(self, folder: str) -> dict[str, Hashed]:
        root, found = self.work / folder, {}
        for directory, _, names in os.walk(root):
            for name in names:
                path = Path(directory) / name
                if path.is_file() and not path.is_symlink():
                    found[path.relative_to(root).as_posix()] = Hashed(sha256_of(path), path.stat().st_size)
        return found

    def store(self, path: str, store, key: str) -> None:
        store.put_file(key, str(self.work / path))

    def place(self, copies: dict[str, str], fetches: dict[str, tuple[str, str]]) -> None:
        for target, source in copies.items():
            (self.work / target).parent.mkdir(parents=True, exist_ok=True)
            (self.work / target).unlink(missing_ok=True)
            os.link(self.work / source, self.work / target)
        for target, (url, sha256) in fetches.items():
            (self.work / target).parent.mkdir(parents=True, exist_ok=True)
            get_config().get_object_store_at(url).download_to_file(url, str(self.work / target))
            if (found := sha256_of(self.work / target)) != sha256:
                raise RuntimeError(f"the store's {url} isn't the file this build stages: its sha256 is {found}")

    def mark_runnable(self, paths: list[str]) -> None:
        for path in paths:
            with (self.work / path).open("rb") as file:
                if file.read(4).startswith(RUN_ONLY):
                    (self.work / path).chmod(EXECUTABLE)

    def build(self, dockerfile: str, tag: str, platform: str, labels: dict[str, str], context: str = CONTEXT) -> None:
        (self.work / context).mkdir(parents=True, exist_ok=True)
        (self.work / context / "Dockerfile").write_text(dockerfile)
        build_image(self.work / context / "Dockerfile", self.work / context, tag, platform=platform, labels=labels)

    def installed(self, tag: str, prefix: str) -> dict[str, str]:
        created = subprocess.run(["docker", "create", tag], capture_output=True, text=True, check=True).stdout.strip()
        try:
            copied = subprocess.run(["docker", "cp", f"{created}:{prefix}/installed.json", "-"], capture_output=True)
        finally:
            subprocess.run(["docker", "rm", created], capture_output=True)
        if copied.returncode != 0:
            return {}
        with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
            return _packages(archive.extractfile(archive.getmembers()[0]).read())

    def written(self, tag: str, commands: list[str]) -> list[dict[str, str]]:
        saved = self.work / "image.tar"
        save_image(tag, saved)
        try:
            return layers_in_image.written(str(saved), commands)
        finally:
            saved.unlink(missing_ok=True)

    def push(self, tag: str, ref: str, store) -> None:
        push_local_image(tag, ref, store)


class VmBuilder(Builder):
    """A VM of a sandbox provider with Docker in it, building in a folder of its own there. Its calls run on an event
    loop of its own, in the thread that made it."""

    def __init__(self, link: SandboxProvider, emulating: bool):
        self.name, self._runner = type(link).__name__, asyncio.Runner()
        self.work = f"/tmp/agentenv-compose-{uuid.uuid4().hex[:12]}"
        timeout = EMULATED_VM_TIMEOUT_S if emulating else VM_TIMEOUT_S
        try:
            self.sandbox: VmSandbox = self._run(link.create_vm(exposed_ports=[], timeout=timeout, **_VM))
        except BaseException:
            self._runner.close()
            raise
        self.platform, self.helper = self.sandbox.platform, f"bash {self.work}/{IN_VM.name}"

    def start(self, work: Path, base: DockerImageArtifact, platform: str,
              from_store: dict[str, tuple[str, str]]) -> None:
        """Give the VM's Docker BuildKit when it has none, register emulation for ``platform`` when it isn't the VM's
        own, sign in to the base's registry or build the base from its build context, and put the build context here:
        the files of ``from_store`` from the store, and the rest, which only the putting machine holds, inline."""
        self._write(f"{self.work}/{IN_VM.name}", IN_VM.read_bytes())
        self._exec(f"{self.helper} buildx {BUILDX.format(platform=self.platform.replace('/', '-'))} "
                   f"{BUILDX_SHA256[self.platform]} {_PLUGINS}")
        if platform != self.platform:
            try:
                self._exec(f"{self.helper} emulate {platform.rsplit('/', 1)[-1]} {BINFMT}")
            except RuntimeError as e:
                raise Unbuildable(f"{self.name}'s VM couldn't register emulation for {platform}: {e}") from None
        if names_registry(base.image_name):
            self._login(get_config().get_image_store().auth(base.pinned_name))
        else:
            self._base(base, platform)
        self._write(f"{self.work}/{CONTEXT}.tar.gz", _inline(work / CONTEXT, set(from_store)))
        self._exec(f"mkdir -p {self.work}/{CONTEXT} && tar -xzf {self.work}/{CONTEXT}.tar.gz --no-same-owner "
                   f"-C {self.work}/{CONTEXT} && rm {self.work}/{CONTEXT}.tar.gz")
        self.place({}, {f"{CONTEXT}/{path}": fetch for path, fetch in from_store.items()})
        self._chmod([f"{CONTEXT}/{path}" for path in from_store if os.access(work / CONTEXT / path, os.X_OK)])

    def resolve(self, request: dict, source: str, arch: str, timeout: int) -> dict:
        out = f"{self.work}/{RESOLVED}/out"
        self._write(f"{out}/request.json", json.dumps({**request, "owner": [0, 0]}).encode())
        self._write(f"{out}/resolve.py", RESOLVER.read_bytes())
        code, stdout, stderr = self._output(
            f"timeout {timeout} {shlex.join(_resolver(source, arch, f'{self.work}/{CONTEXT}', out))}")
        if code == 124:
            raise ResolveError(f"resolving took over {timeout} s", stdout + stderr)
        if code != 0:
            raise ResolveError("resolving the packages failed", stdout + stderr)
        return json.loads(self._exec(f"cat {out}/result.json"))

    def download(self, url: str, sha256: str, path: str) -> None:
        try:
            found = self._exec(f"{self.helper} download {shlex.quote(url)} {self._at(path)}").strip()
        except RuntimeError as e:
            raise ResolveError(f"downloading {url} failed: {e}") from None
        if found != sha256:
            raise ResolveError(f"{url} has the sha256 {found}, not the {sha256} its download names")

    def files(self, folder: str) -> dict[str, Hashed]:
        listed = self._exec(f"{self.helper} files {self._at(folder)}")
        found = {}
        for entry in filter(None, listed.split("\0")):
            sha256, size, path = entry.split(" ", 2)
            found[path] = Hashed(sha256, int(size))
        return found

    def store(self, path: str, store, key: str) -> None:
        self._run(upload_vm_file(self.sandbox, f"{self.work}/{path}", store, store.object_url(key)))

    def place(self, copies: dict[str, str], fetches: dict[str, tuple[str, str]]) -> None:
        self._each([f"{self.helper} link {self._at(source)} {self._at(target)}" for target, source in copies.items()])
        if not fetches:
            return
        self._each([f"mkdir -p {shlex.quote(str(PurePosixPath(self.work, target).parent))}" for target in fetches])
        self._run(self._fetch({f"{self.work}/{target}": url for target, (url, _) in fetches.items()}))
        listed = "".join(f"{sha256}  {target}\n" for target, (_, sha256) in fetches.items())
        self._write(f"{self.work}/fetched.sha256", listed.encode())
        try:
            self._exec(f"{self.helper} check {self.work} {self.work}/fetched.sha256")
        except RuntimeError as e:
            raise RuntimeError(f"a file the store served isn't the one this build stages: {e}") from None

    def mark_runnable(self, paths: list[str]) -> None:
        self._each([f"{self.helper} runnable {self._at(path)}" for path in paths])

    def build(self, dockerfile: str, tag: str, platform: str, labels: dict[str, str], context: str = CONTEXT) -> None:
        folder = f"{self.work}/{context}"
        self._write(f"{folder}/Dockerfile", dockerfile.encode())
        options = " ".join(f"--label {shlex.quote(f'{name}={value}')}" for name, value in labels.items())
        code, stdout, stderr = self._output(f"cd {folder} && docker build --platform {shlex.quote(platform)} {options} "
                                            f"-f Dockerfile -t {shlex.quote(tag)} . 2>&1")
        if code != 0:
            raise DockerBuildError(f"docker build of {tag} failed (exit {code}):\n{(stdout + stderr).rstrip()}")

    def installed(self, tag: str, prefix: str) -> dict[str, str]:
        listed = self._exec(f"{self.helper} installed {shlex.quote(tag)} {shlex.quote(f'{prefix}/installed.json')}")
        return _packages(listed.encode()) if listed.strip() else {}

    def written(self, tag: str, commands: list[str]) -> list[dict[str, str]]:
        """Read in a container of ``tag``, on its own Python, so the VM needs none."""
        image, script = f"{self.work}/image.tar", f"{self.work}/{LAYERS.name}"
        self._write(script, LAYERS.read_bytes())
        return json.loads(self._exec(
            f"docker save -o {image} {shlex.quote(tag)} && docker run --rm --network none --user root "
            f"-v {image}:/image.tar:ro -v {script}:/layers.py:ro --entrypoint python3 {shlex.quote(tag)} /layers.py "
            f"/image.tar {shlex.quote(json.dumps(commands))} && rm -f {image}"))

    def push(self, tag: str, ref: str, store) -> None:
        self._login(store.auth(ref))
        self._exec(f"docker tag {shlex.quote(tag)} {shlex.quote(ref)} && docker push {shlex.quote(ref)}")

    def close(self) -> None:
        """Tear the VM down; one that won't go is left to its provider's timeout, since what it built is pushed."""
        try:
            self._run(self.sandbox.terminate())
        except Exception as e:
            logger.warning(f"{self.name} VM {self.sandbox.sandbox_id} wasn't torn down, so its provider stops it at its "
                           f"timeout: {type(e).__name__}: {e}")
        finally:
            self._runner.close()

    def _login(self, auth) -> None:
        if auth is not None:
            self._exec(f"echo {shlex.quote(auth.password)} | docker login --username {shlex.quote(auth.username)} "
                       f"--password-stdin {shlex.quote(auth.registry)}")

    def _base(self, base: DockerImageArtifact, platform: str) -> None:
        """Build ``base`` for ``platform`` from its stored build context, under the tag a local build gives it."""
        folder = f"{self.work}/base"
        self._exec(f"mkdir -p {folder}")
        self._run(self.sandbox.load_object_file(base.build_context_object_url, f"{folder}.tar.gz"))
        code, stdout, stderr = self._output(
            f"tar -xzf {folder}.tar.gz -C {folder} && cd {folder} && docker build --platform {shlex.quote(platform)} "
            f"-f {shlex.quote(base.dockerfile_path)} -t {shlex.quote(platform_tag(base, platform))} . 2>&1")
        if code != 0:
            raise DockerBuildError(f"docker build of the base {base.image_name} failed (exit {code}):\n"
                                   f"{(stdout + stderr).rstrip()}")

    async def _fetch(self, urls: dict[str, str]) -> None:
        slots = asyncio.Semaphore(_FETCHES_IN_FLIGHT)

        async def fetch(path: str, url: str) -> None:
            async with slots:
                await self.sandbox.load_object_file(url, path)

        await asyncio.gather(*(fetch(path, url) for path, url in urls.items()))

    def _chmod(self, paths: list[str]) -> None:
        self._each([f"chmod {EXECUTABLE:o} {self._at(path)}" for path in paths])

    def _at(self, path: str) -> str:
        """``path``, in the builder's work folder, quoted for a script."""
        return shlex.quote(f"{self.work}/{path}")

    def _each(self, commands: list[str]) -> None:
        """Run ``commands`` in as few scripts as fit a script's limit."""
        script = ""
        for command in commands:
            if script and len(script) + len(command) > self.sandbox._WFT_CHUNK_BYTES:
                self._exec(script)
                script = ""
            script += f"{command}\n"
        if script:
            self._exec(script)

    def _write(self, path: str, data: bytes) -> None:
        self._run(self.sandbox.write_host_file(data, path))

    def _exec(self, script: str) -> str:
        return self._run(self.sandbox.exec_script(f"set -e\n{script}"))

    def _output(self, script: str) -> tuple[int, str, str]:
        return self._run(self.sandbox.exec_with_output("sudo", "bash", "-c", script))

    def _run(self, coroutine):
        """``coroutine`` run on the builder's loop; what its provider's client raises, as a RuntimeError naming the VM."""
        try:
            return self._runner.run(coroutine)
        except (RuntimeError, ValueError, OSError, ObjectAlreadyExistsError):
            raise
        except Exception as e:
            raise RuntimeError(f"{self.name}'s VM failed: {type(e).__name__}: {e}") from e


def _inline(context: Path, skip: set[str]) -> bytes:
    """A tar.gz of the folders and files under ``context`` but those of ``skip``, with their modes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(context.rglob("*")):
            if (name := path.relative_to(context).as_posix()) not in skip:
                archive.add(path, arcname=name, recursive=False)
    return buffer.getvalue()


def _packages(listed: bytes) -> dict[str, str]:
    return {package["name"]: package["version"] for package in json.loads(listed)}


def save_image(tag: str, path: Path) -> None:
    subprocess.run(["docker", "save", "-o", str(path), tag], check=True, capture_output=True, text=True)


def sha256_of(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            hashed.update(chunk)
    return hashed.hexdigest()
