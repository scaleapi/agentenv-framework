"""Windows computers' disks, made with this machine's Docker and KVM, which keeps them, as a Mac keeps the VM bundles
composed on it: Windows isn't agent-env's to publish. A base is Windows installed from Microsoft's evaluation media,
fetched once into this machine's cache, by the QEMU launcher with the answers and setup of ``computers/disk/windows``,
then packed as a thin disk's base: ``disk/base.img`` in a layer of its own, an empty overlay of it as ``disk/disk.img``
and the guest file beside. A composed disk is a layer on its base: the base booted on an overlay kept on this machine,
given each part over ssh as the base's admin, by the key agent-env made for that base, then shut down and packed onto
the base's image, so every disk composed on a base shares one copy of it. An evaluation runs for 90 days from its
install, so a base records when it expires, and one with fewer than ``RENEW_DAYS`` left is installed afresh."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NamedTuple

from agentenv_protocol.computer.app import ADDRESS
from agentenv_protocol.computer.apps import apps_folder

from agent_env.artifact.artifacts.computer import DISK_BASE_FILE, DISK_FILE, DISK_OWNER
from agent_env.artifact.artifacts.docker_image import docker_login, push_local_image
from agent_env.config.paths import cache_root
from agent_env.env.envs.computer.utils.install import WindowsInstall
from agent_env.providers.sandbox_providers.host_images import present
from agent_env.providers.sandbox_providers.local_docker.cli import docker
from agent_env.providers.sandbox_providers.local_docker.disk import KVM, launcher, machine_image
from agent_env.store.image_store.image_contents import labels as image_labels
from agent_env.store.image_store.image_store import ImageStore
from agent_env.store.image_store.registry_api import pin_digest, pin_if_served
from agent_env.store.local_state import holding_locks
from agent_env.utils.docker.build import build_image

#: Microsoft's link to the Windows 11 Enterprise evaluation ISO, English (United States), x64.
ISO_URL = "https://go.microsoft.com/fwlink/?linkid=2334167&clcid=0x409"
#: What a base is installed with: the answers setup reads, the setup it runs, and agentenv-session.
RECIPE = Path(__file__).resolve().parents[4] / "computers" / "disk" / "windows"
ADMIN = "agentadmin"
#: Where agent-env's own files are on a Windows computer, and each layer's folder under them.
ROOT = PureWindowsPath("C:/agentenv")
DRIVERS, APPS = ROOT / "drivers", apps_folder("windows")
PYTHON = ROOT / "python" / "python.exe"
BUILD = ROOT / "build"
#: The fewest days of its evaluation a base may have left, as a deploy needs some.
RENEW_DAYS = 7
EXPIRES_LABEL, BUILD_LABEL, KEY_LABEL = "dev.agentenv.windows.expires", "dev.agentenv.windows.build", "dev.agentenv.key"
#: Where a base's image keeps the key to its admin, which composing a layer on it takes over ssh. The image is in the
#: user's own image store, and the guest's ssh is reached only on this machine's loopback while a layer is composed.
ADMIN_KEY = PurePosixPath("/disk/admin.key")
#: How long an install, and a compose, may take, and how long the session user's first sign-in is left to settle.
INSTALL_TIMEOUT_S, COMPOSE_TIMEOUT_S, SETTLE_S = 3600, 3600, 60
#: How long a guest may take to power off once told.
SHUTDOWN_TIMEOUT_S = 300
#: How long trimming a disk's free space may take.
COMPACT_TIMEOUT_S = 1800
#: How often an install looks for setup's end, a compose for the base's ssh, and a shutdown for the guest's.
INSTALL_POLL_S, BOOT_POLL_S, SHUTDOWN_POLL_S = 15, 5, 2
#: How many times a compose tries a layer's download, and how long it waits between, as the base's setup does: a
#: server's errors can last minutes, as GitHub's release downloads' 504s have.
FETCH_TRIES, FETCH_PAUSE_S = 30, 10
CPUS, MEMORY_MB = 4, 8192
#: What every service runs with: Python reading and writing files as UTF-8, as on Linux, not in Windows' code page.
UTF8 = {"PYTHONUTF8": "1"}
#: What a base's install prints once, as it fetches Microsoft's media.
NOTICE = ("windows: fetching the Windows 11 Enterprise evaluation from Microsoft ({url}, about 7 GB) into {path}, "
          "once; it's installed for evaluation under Microsoft's terms and runs 90 days from each install")


@dataclass(frozen=True)
class Registry:
    """The env's image store as Windows disks are kept in it: by tag in ``repository``, pinned by digest, as Linux
    disks are."""

    store: ImageStore
    repository: str

    def served(self, tag: str) -> str | None:
        """``tag``'s ref, pinned to the digest the store serves for it now; None when it serves none."""
        ref = self.store.image_ref(self.repository, tag)
        return pin_if_served(ref, self.store.auth(ref))

    def labels(self, pinned: str) -> dict[str, str]:
        return image_labels(pinned, "amd64", self.store.auth(pinned))

    def push(self, local: str, tag: str) -> str:
        """``local`` pushed as ``tag``; its ref there, pinned."""
        ref = self.store.image_ref(self.repository, tag)
        push_local_image(local, ref, self.store)
        return pin_digest(ref, self.store.auth(ref))

    def pull(self, pinned: str) -> None:
        docker_login(self.store.auth(pinned))
        docker("pull", "-q", pinned)


class Names(NamedTuple):
    """What a base or a disk is called: ``local`` in this machine's Docker, ``tag`` in the image store."""

    local: str
    tag: str


def base_names(key: str) -> Names:
    return Names(f"agentenv-windows-base:{_short(key)}", f"windows-base-{_short(key)}")


def disk_names(key: str) -> Names:
    """A disk's names, its tag in the image store being its key, as a Linux disk's is."""
    return Names(f"agentenv-windows:{_short(key)}", key.removeprefix("sha256:"))


def _short(key: str) -> str:
    return key.removeprefix("sha256:")[:16]


@dataclass(frozen=True)
class Base:
    """A base: its image's tag on this machine, the folder of its admin's key, when its evaluation expires, and, for one
    the image store holds that this machine doesn't yet, its ref there, pulled when a layer is composed on it."""

    image: str
    keys: Path
    expires: datetime
    pinned: str | None = None


@dataclass(frozen=True)
class Part:
    """A layer as a Windows disk installs it: its folder under ``context/``, its prefix, its install, its services'
    argv by name, the environment they're given, what its venv installs beside its requirements, and for an app, the
    ``<file>:<Class>`` its own server serves and its install steps by name, each run elevated."""

    folder: str
    prefix: PureWindowsPath
    install: WindowsInstall
    services: dict[str, list[str]]
    environment: dict[str, Any]
    wants: tuple[str, ...] = ()
    served: str | None = None
    steps: tuple[str, ...] = ()


def media_path() -> Path:
    """Where this machine's cache keeps the evaluation ISO."""
    return cache_root() / "windows" / "windows-11-enterprise-evaluation-x64.iso"


def media(say: Callable[[str], None]) -> Path:
    """The evaluation ISO in this machine's cache, fetched from Microsoft first when it isn't there."""
    iso = media_path()
    with holding_locks([f"windows-media:{iso}"]):
        if not iso.exists():
            iso.parent.mkdir(parents=True, exist_ok=True)
            say(NOTICE.format(url=ISO_URL, path=iso))
            partial = iso.with_suffix(".part")
            with urllib.request.urlopen(ISO_URL, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            partial.rename(iso)
    return iso


def base(key: str, iso: Callable[[], Path], disk_gb: int, user: str, say: Callable[[str], None],
         registry: Registry) -> Base:
    """The base of ``key``: this machine's, pushed to the image store unless it holds one of ``key``; else the one the
    store holds, pulled, with the key to its admin it carries, when a layer is composed on it; else Windows installed
    from the ISO ``iso`` gives, and pushed there. One with fewer than ``RENEW_DAYS`` of its evaluation left is installed
    afresh. One install of a key runs at once on a machine, and another waits for it."""
    (image, tag), keys = base_names(key), cache_root() / "windows" / "bases" / _short(key)
    with holding_locks([_base_lock(image)], lambda: say("waiting for the install of the same base under way")):
        if present(image) and (keys / "id").exists():
            expires = datetime.fromisoformat(_label(image, EXPIRES_LABEL))
            if not expiring(expires.isoformat()):
                if registry.served(tag) is None:
                    say("pushing the base to the image store, which doesn't hold it")
                    registry.push(image, tag)
                return Base(image, keys, expires)
            say(f"its base's evaluation expires {expires:%Y-%m-%d}; installing it afresh")
        elif (pinned := registry.served(tag)) is not None and not expiring(
                stored := registry.labels(pinned).get(EXPIRES_LABEL)):
            return Base(image, keys, datetime.fromisoformat(stored), pinned)
        shutil.rmtree(keys, ignore_errors=True)
        keys.mkdir(parents=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"agentenv {image}", "-f",
                        str(keys / "id")], check=True)
        media = iso()
        with _working(f"base-{_short(key)}") as work:
            facts = _installed(work, media, keys, disk_gb, user, say)
            say("packing the base")
            expires = datetime.now(timezone.utc) + timedelta(minutes=facts["grace_minutes"])
            shutil.copy(keys / "id", work / ADMIN_KEY.name)
            _pack(work, image, {KEY_LABEL: key, EXPIRES_LABEL: expires.isoformat(timespec="seconds"),
                                BUILD_LABEL: facts["build"]}, base_image=None)
        say("pushing the base to the image store")
        registry.push(image, tag)
        return Base(image, keys, expires)


def _here(base_: Base, registry: Registry, say: Callable[[str], None]) -> None:
    """``base_`` on this machine, with the key to its admin: pulled from the image store when it's the store's, under
    the base's lock, so a compose of another disk on it waits, then finds it."""
    if base_.pinned is None:
        return
    with holding_locks([_base_lock(base_.image)], lambda: say("waiting for the pull of the same base under way")):
        if present(base_.image) and (base_.keys / "id").exists():
            return
        say(f"pulling its base from the image store, {base_.pinned}")
        registry.pull(base_.pinned)
        docker("tag", base_.pinned, base_.image)
        _take_admin_key(base_.image, base_.keys)


def _base_lock(image: str) -> str:
    return f"windows-base:{image}"


def _take_admin_key(image: str, keys: Path) -> None:
    """The key to the admin of the base ``image`` holds, written to ``keys``, as composing a layer on it needs."""
    shutil.rmtree(keys, ignore_errors=True)
    keys.mkdir(parents=True)
    container = docker("create", image, "none")
    try:
        docker("cp", f"{container}:{ADMIN_KEY}", str(keys / "id"))
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True)
    (keys / "id").chmod(0o600)


def _installed(work: Path, iso: Path, keys: Path, disk_gb: int, user: str, say: Callable[[str], None]) -> dict:
    """Windows installed from ``iso`` onto ``work/disk.img`` by the launcher, answered as the recipe says: then booted
    until its session user has signed in and settled, and shut down; what it reports of itself."""
    answers, out = work / "answers", work / "out"
    answers.mkdir()
    out.mkdir()
    admin_password, user_password = secrets.token_urlsafe(18), secrets.token_urlsafe(18)
    text = (RECIPE / "autounattend.xml").read_text()
    for name, value in (("USER", user), ("ADMIN_PASSWORD", admin_password), ("USER_PASSWORD", user_password)):
        text = text.replace(f"@{name}@", value)
    (answers / "autounattend.xml").write_text(text)
    for name in ("setup.ps1", "agentenv-session.py"):
        shutil.copy(RECIPE / name, answers / name)
    shutil.copy(keys / "id.pub", answers / "authorized_keys")
    (out / "guest").write_text("windows")
    (keys / "passwords.json").write_text(json.dumps({ADMIN: admin_password, user: user_password}))
    (keys / "passwords.json").chmod(0o600)
    name = f"agentenv-windows-install-{keys.name}"
    say(f"installing Windows from {iso.name} under QEMU in {name}; this takes about 15 minutes")
    started = time.monotonic()
    container = _launched(name, {"VM_KEEP": "/out/disk.img", "VM_DISK_GB": str(disk_gb), "VM_MEDIA": "/in/media.iso",
                                 "VM_FILES": "/in/answers"},
                          ["-v", f"{out}:/out", "-v", f"{iso}:/in/media.iso:ro", "-v", f"{answers}:/in/answers:ro"],
                          launcher())
    try:
        port = _ssh_port(container)
        deadline = started + INSTALL_TIMEOUT_S
        while not _signed_in(port, keys, user):
            _alive(container, "installing", deadline)
            time.sleep(INSTALL_POLL_S)
        say(f"installed and signed in as {user} after {time.monotonic() - started:.0f} s; settling")
        time.sleep(SETTLE_S)
        facts = json.loads(ssh(port, keys, powershell(_FACTS)).stdout.strip().splitlines()[-1])
        _compacted(port, keys, say)
        _shut_down(container, port, keys)
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    _converted(out, launcher())
    shutil.move(out / "disk.img", work / "base.img")
    return facts


def compose(base_: Base, build: Path, key: str, labels: dict[str, str], say: Callable[[str], None],
            registry: Registry) -> str:
    """The disk of ``key`` composed on ``base_``: the base booted on an overlay kept in a folder of this machine,
    ``build`` copied in and its ``provision.ps1`` run as the admin over ssh, then shut down, packed onto the base's
    image with ``labels`` and pushed to the image store; its ref there, pinned. A base the store holds is pulled
    first."""
    image, tag = disk_names(key)
    _here(base_, registry, say)
    with _working(f"disk-{_short(key)}") as work:
        out = work / "out"
        out.mkdir()
        name = f"agentenv-windows-compose-{_short(key)}"
        say(f"booting its base in {name}")
        machine = asyncio.run(machine_image(base_.image))
        container = _launched(name, {"VM_KEEP": "/out/disk.img"}, ["-v", f"{out}:/out"], machine)
        try:
            port, deadline = _ssh_port(container), time.monotonic() + COMPOSE_TIMEOUT_S
            while ssh(port, base_.keys, "echo ready", check=False, timeout=60).returncode != 0:
                _alive(container, "booting", deadline)
                time.sleep(BOOT_POLL_S)
            say("installing its layers over ssh")
            ssh(port, base_.keys, f'(if exist "{BUILD}" rmdir /s /q "{BUILD}") & mkdir "{BUILD}"')
            ssh(port, base_.keys, f'tar -xf - -C "{BUILD}"', input=_tarred(build), timeout=600)
            done = ssh(port, base_.keys, f'powershell -NoProfile -ExecutionPolicy Bypass -File "{BUILD}\\provision.ps1"',
                       check=False, timeout=COMPOSE_TIMEOUT_S)
            if done.returncode != 0:
                raise RuntimeError(f"provisioning exited {done.returncode}: {(done.stderr + done.stdout).strip()[-3000:]}")
            ssh(port, base_.keys, f'rmdir /s /q "{BUILD}"')
            _compacted(port, base_.keys, say)
            _shut_down(container, port, base_.keys)
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        _converted(out, machine, backing=f"/{DISK_BASE_FILE}")
        subprocess.run(["docker", "run", "--rm", "-v", f"{out}:/out", "--entrypoint", "qemu-img", launcher(), "rebase",
                        "-u", "-b", PureWindowsPath(DISK_BASE_FILE).name, "-F", "qcow2", "/out/disk.img"],
                       check=True, capture_output=True)
        (out / "guest").write_text("windows")
        say("packing the disk")
        _pack(out, image, labels, base_image=base_.image)
    say("pushing the disk to the image store")
    return registry.push(image, tag)


def expiring(expires: str | None) -> bool:
    """Whether an evaluation that ends at ``expires`` has fewer than ``RENEW_DAYS`` left, as a disk with none recorded
    is taken to."""
    return expires is None or datetime.fromisoformat(expires) - datetime.now(timezone.utc) < timedelta(days=RENEW_DAYS)


def write(build: Path, parts: list[Part], user: str, build_settings: dict[str, Any], image: dict) -> str:
    """Write ``build``'s ``provision.ps1``, which installs ``parts`` from ``build/context``, each app's own folder the
    session user ``user``'s, their services for agentenv-session, ``build_settings`` as every service's
    ``AGENTENV_SETTING_<NAME>`` and ``image`` as what the disk says it holds, then leaves ``ROOT`` the admins'
    (``_locked``); its text, which a composed disk's key covers."""
    lines = ["$ErrorActionPreference = 'Stop'", "$ProgressPreference = 'SilentlyContinue'",
             f"$build = '{BUILD}'", _CHECKED, _FETCH]
    for part in parts:
        lines += [f"# {part.folder}", *_installed_part(part, user)]
    settings = {f"AGENTENV_SETTING_{name.upper()}": str(value) for name, value in build_settings.items()}
    lines += [_written(ROOT / "build-settings.json", json.dumps(settings)),
              _written(ROOT / "image.json", json.dumps(image, sort_keys=True)), *_locked(user)]
    script = "\r\n".join(lines) + "\r\n"
    (build / "provision.ps1").write_text(script)
    return script


def _locked(user: str) -> list[str]:
    """The lines that leave ``ROOT`` the admins', as on Linux ``/opt/agentenv`` is root's: every user reads and runs
    what's there, none of the access ``C:\\`` gives every user to change it is left, and the session user ``user``
    writes only the services' logs, and each app's own folder, which keeps its own access."""
    return [f"Checked {{ icacls '{ROOT}' /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' "
            "'*S-1-5-32-545:(OI)(CI)RX' | Out-Null }",
            f"Checked {{ icacls '{ROOT / 'logs'}' /grant {_ps(f'{user}:(OI)(CI)M')} | Out-Null }}"]


def _installed_part(part: Part, user: str) -> list[str]:
    """The lines that install ``part``: its files, downloads, venv, an app's own folder, its run scripts and install
    steps, and its services. An app's own folder, where its server writes its address and token, holds none of the
    access its parents give: only ``user``, the session user its server runs as, SYSTEM and Administrators have it, as
    on Linux the session user alone reaches the server's socket."""
    prefix, install = part.prefix, part.install
    lines = [f"New-Item -ItemType Directory -Force '{prefix}' | Out-Null",
             f"Copy-Item -Recurse -Force \"$build\\context\\{PureWindowsPath(part.folder)}\\*\" '{prefix}'"]
    for download in install.downloads or []:
        if download.arch in (None, "amd64"):
            to = prefix / PureWindowsPath(download.to)
            lines += [f"New-Item -ItemType Directory -Force '{to.parent}' | Out-Null",
                      f"Fetch '{download.url}' '{download.sha256}' '{to}'"]
    if install.python is not None or part.wants:
        venv = prefix / "venv"
        lines.append(f"Checked {{ & '{PYTHON}' -m venv '{venv}' }}")
        wants = [*(("-r", str(prefix / PureWindowsPath(install.python))) if install.python else ()), *part.wants]
        lines.append(f"Checked {{ & '{venv / 'Scripts' / 'python.exe'}' -m pip install -q --disable-pip-version-check "
                     + " ".join(_ps(want) for want in wants) + " }")
        lines.append(f"Checked {{ & '{venv / 'Scripts' / 'python.exe'}' -m pip list --format=json | "
                     f"Set-Content -Encoding utf8 '{prefix / 'installed.json'}' }}")
    if part.served is not None:
        own = prefix / ADDRESS.parent.as_posix()
        lines += [f"New-Item -ItemType Directory -Force '{own}' | Out-Null",
                  f"Checked {{ icacls '{own}' /inheritance:r /grant:r {_ps(f'{user}:(OI)(CI)F')} '*S-1-5-18:(OI)(CI)F' "
                  "'*S-1-5-32-544:(OI)(CI)F' | Out-Null }"]
    for run in install.run or []:
        lines.append(f"$env:PREFIX = '{prefix}'; Push-Location '{prefix}'; Checked {{ & powershell -NoProfile "
                     f"-ExecutionPolicy Bypass -File '{prefix / PureWindowsPath(run.script)}' }}; Pop-Location")
    for step in part.steps:
        lines.append(f"Push-Location '{prefix}'; Checked {{ & '{prefix / 'venv' / 'Scripts' / 'python.exe'}' -m "
                     f"agentenv_protocol.computer.run_install_step {_ps(part.served)} {_ps(step)} }}; Pop-Location")
    for name, argv in part.services.items():
        spec = {"name": name, "prefix": str(prefix), "command": argv,
                "env": {**UTF8, **{key: str(value) for key, value in part.environment.items()}}}
        lines.append(_written(ROOT / "services" / f"{name}.json", json.dumps(spec)))
    return lines


def argv(command: str, prefix: PureWindowsPath) -> list[str]:
    """A service's ``command`` as the argv agentenv-session runs, split as Windows quotes it, a word in double quotes
    one word without them, and a relative program named under ``prefix``."""
    words = [word[1:-1] if len(word) > 1 and word[0] == word[-1] == '"' else word
             for word in shlex.split(command, posix=False)]
    if "/" in words[0] and not PureWindowsPath(words[0]).is_absolute():
        words[0] = str(prefix / PureWindowsPath(words[0]))
    return words


def ssh(port: int, keys: Path, command: str, *, check: bool = True, timeout: float = 120,
        input: bytes | None = None) -> subprocess.CompletedProcess:
    """``command`` run by cmd on the guest as its admin, over ssh to ``port`` here with the base's key, its output
    decoded; refused when it fails unless not ``check``, as one that took over ``timeout`` has."""
    argv = ["ssh", "-p", str(port), "-i", str(keys / "id"), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o",
            "ConnectTimeout=10", f"{ADMIN}@127.0.0.1", command]
    try:
        done = subprocess.run(argv, input=input, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        done = subprocess.CompletedProcess(argv, 255, b"", f"it took over {timeout:.0f} s".encode())
    done = subprocess.CompletedProcess(argv, done.returncode, done.stdout.decode(errors="replace"),
                                       done.stderr.decode(errors="replace"))
    if check and done.returncode != 0:
        raise RuntimeError(f"ssh {command[:200]!r} exited {done.returncode}: "
                           f"{(done.stderr + done.stdout).strip()[-2000:]}")
    return done


def powershell(script: str) -> str:
    """The command that runs ``script`` in PowerShell, encoded, so cmd's quoting never touches it."""
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + base64.b64encode(
        script.encode("utf-16-le")).decode()


def _signed_in(port: int, keys: Path, user: str) -> bool:
    """Whether setup has finished and ``user`` has an active session on the console; refused, with setup's error, when
    setup failed."""
    done = ssh(port, keys, f'if exist "{ROOT}\\setup-failed" (type "{ROOT}\\setup-failed" & exit 3) else '
                           f'if exist "{ROOT}\\setup-done" (quser) else (echo setting-up)', check=False, timeout=60)
    if done.returncode == 3:
        raise RuntimeError(f"windows: setup failed in the guest: {done.stdout.strip()[-2000:]}")
    return re.search(rf"^\s*>?{re.escape(user)}\s+console\s+\d+\s+Active", done.stdout,
                     re.MULTILINE | re.IGNORECASE) is not None


def _compacted(port: int, keys: Path, say: Callable[[str], None]) -> None:
    """What the guest keeps that a computer doesn't need removed and its free space trimmed (``compact.ps1``), so its
    disk's image, rewritten after it shuts down, keeps only what Windows uses."""
    say("trimming its free space")
    done = ssh(port, keys, powershell((RECIPE / "compact.ps1").read_text()), check=False, timeout=COMPACT_TIMEOUT_S)
    if done.returncode != 0:
        say(f"trimming exited {done.returncode}, so its image keeps what it held: {(done.stderr + done.stdout)[-300:]}")


def _converted(folder: Path, image: str, backing: str | None = None) -> None:
    """``folder``'s ``disk.img`` rewritten by ``qemu-img convert`` in ``image``, which holds its backing file, keeping
    only the clusters it uses: a base's whole, a layer's over ``backing``, the backing file ``image`` holds."""
    over = ["-B", backing, "-F", "qcow2"] if backing else []
    subprocess.run(["docker", "run", "--rm", "-v", f"{folder}:/out", "--entrypoint", "qemu-img", image, "convert", "-O",
                    "qcow2", *over, "/out/disk.img", "/out/compact.img"], check=True, capture_output=True)
    (folder / "compact.img").replace(folder / "disk.img")


def _launched(name: str, variables: dict[str, str], mounts: list[str], image: str) -> str:
    """A launcher container of ``image``, named ``name``, booting with ``variables`` and KVM, its guest's ssh published
    on a port of this machine's loopback; any earlier one of the name removed first."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    environment = {"VM_PORTS": "22", "VM_CPUS": str(CPUS), "VM_MEMORY_MB": str(MEMORY_MB), "VM_REQUIRE_KVM": "1",
                   **variables}
    return docker("run", "-d", "--name", name, "--device", KVM, "-p", "127.0.0.1::22", *mounts,
                  *(arg for key, value in environment.items() for arg in ("-e", f"{key}={value}")), image)


def _ssh_port(container: str) -> int:
    return int(docker("port", container, "22/tcp").splitlines()[0].rpartition(":")[2])


def _alive(container: str, doing: str, deadline: float) -> None:
    """Raise when ``container`` has stopped or ``deadline`` has passed, with the end of its output."""
    running = docker("inspect", "-f", "{{.State.Running}}", container) == "true"
    if running and time.monotonic() < deadline:
        return
    tail = subprocess.run(["docker", "logs", "--tail", "20", container], capture_output=True, text=True)
    why = "its QEMU exited" if not running else f"it was still {doing} after the time it's given"
    raise RuntimeError(f"windows: {why}: {(tail.stdout + tail.stderr).strip()[-2000:]}")


def _shut_down(container: str, port: int, keys: Path) -> None:
    """The guest powered off, and so its launcher's container exited, which keeps its overlay."""
    ssh(port, keys, "shutdown /s /t 0", check=False, timeout=30)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    while docker("inspect", "-f", "{{.State.Running}}", container) == "true":
        if time.monotonic() > deadline:
            raise RuntimeError(f"windows: the guest didn't power off within {SHUTDOWN_TIMEOUT_S} s")
        time.sleep(SHUTDOWN_POLL_S)
    if (code := docker("inspect", "-f", "{{.State.ExitCode}}", container)) != "0":
        raise RuntimeError(f"windows: its launcher exited {code} as the guest powered off")


def _pack(folder: Path, image: str, labels: dict[str, str], base_image: str | None) -> None:
    """``image`` of ``folder``'s disk: a base's, ``base.img`` in a layer of its own under an empty overlay of it, or a
    composed one's ``disk.img`` over ``base_image``; the guest file beside it."""
    owner = f"--chown={DISK_OWNER}:{DISK_OWNER}"
    if base_image is None:
        subprocess.run(["docker", "run", "--rm", "-v", f"{folder}:/out", "--entrypoint", "qemu-img", launcher(), "create",
                        "-q", "-f", "qcow2", "-b", Path(DISK_BASE_FILE).name, "-F", "qcow2", "/out/disk.img"],
                       check=True, capture_output=True)
        (folder / "guest").write_text("windows")
        dockerfile = (f"FROM scratch\nCOPY {owner} base.img /{DISK_BASE_FILE}\n"
                      f"COPY --chmod=600 {ADMIN_KEY.name} {ADMIN_KEY}\n"
                      f"COPY {owner} disk.img guest /{Path(DISK_FILE).parent}/\n")
    else:
        dockerfile = f"FROM {base_image}\nCOPY {owner} disk.img guest /{Path(DISK_FILE).parent}/\n"
    for name in ("answers", "out"):
        shutil.rmtree(folder / name, ignore_errors=True)
    (folder / "Dockerfile").write_text(dockerfile)
    build_image(folder / "Dockerfile", folder, image, platform="linux/amd64", labels=labels)


@contextlib.contextmanager
def _working(name: str) -> Iterator[Path]:
    """A work folder of ``name`` in this machine's cache, emptied first of what a build killed in it left, and removed
    after; its caller holds the lock of its build, so one build works in it at a time."""
    work = cache_root() / "windows" / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _tarred(folder: Path) -> bytes:
    """``folder``'s files as a tar, which Windows' own tar unpacks."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(folder.rglob("*")):
            tar.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)
    return buffer.getvalue()


def _label(image: str, name: str) -> str:
    return json.loads(docker("image", "inspect", "-f", "{{json .Config.Labels}}", image) or "{}").get(name, "")


def _written(path: PureWindowsPath, text: str) -> str:
    """The line that writes ``text`` to ``path`` as UTF-8 with no BOM, which PowerShell 5's Set-Content would add."""
    return f"[IO.File]::WriteAllText('{path}', {_ps(text)})"


def _ps(value: str) -> str:
    """``value`` as a PowerShell literal string."""
    return "'" + value.replace("'", "''") + "'"


#: What a base reports of itself once installed: its build and the minutes left of its evaluation.
_FACTS = """$v = Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'
$g = (Get-CimInstance SoftwareLicensingProduct -Filter 'PartialProductKey IS NOT NULL' |
      Where-Object Name -like 'Windows*' | Select-Object -First 1).GracePeriodRemaining
@{build = "$($v.CurrentBuild).$($v.UBR)"; grace_minutes = [int]$g} | ConvertTo-Json -Compress"""
_CHECKED = ("function Checked([scriptblock]$Run) { & $Run; if ($LASTEXITCODE) { throw \"provision: $Run exited "
            "$LASTEXITCODE\" } }")
_FETCH = ("function Fetch([string]$Url, [string]$Sha256, [string]$To) { for ($try = 1; ; $try++) { try { "
          "Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $To; break } catch { if ($try -ge "
          f"{FETCH_TRIES}) {{ throw }}; Start-Sleep {FETCH_PAUSE_S} }} }}; $got = (Get-FileHash -Algorithm SHA256 $To)"
          ".Hash.ToLower(); if ($got -ne $Sha256) { throw \"provision: $Url came with sha256 $got, not $Sha256\" } }")
