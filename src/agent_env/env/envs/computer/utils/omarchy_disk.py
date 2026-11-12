"""Omarchy computers' disks, thin disks (``thin_disk``) made with this machine's Docker and KVM. A base is Omarchy's own
VM image, fetched once into this machine's cache from its artifact's source and checked by sha256, booted once by the
QEMU launcher as a linux-desktop guest, with root's ssh key passed as a systemd credential, made a computer's base by
the recipe of ``computers/disk/omarchy`` run as root over ssh, then packed whole as a thin disk's base. A composed
disk's parts are installed by a bash script, ``provision.sh``, run as root: each layer's files, downloads, venv and run
scripts, an app's own folder, the session user's alone, and its install steps, and each layer's services as systemd user
units of the graphical session, which start through ``with-seed``, so each boot's deploy environment is theirs."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import shutil
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agentenv_protocol.computer.app import SOCKET
from agentenv_protocol.computer.platform import PLATFORM_ENV

from agent_env.artifact.artifacts.computer import DiskSource
from agent_env.config.paths import cache_root
from agent_env.env.envs.computer.utils import thin_disk
from agent_env.env.envs.computer.utils.install import Install
from agent_env.env.envs.computer.utils.thin_disk import Base, Registry
from agent_env.providers.sandbox_providers.local_docker.disk import launcher
from agent_env.store.local_state import holding_locks

_DISK = Path(__file__).resolve().parents[4] / "computers" / "disk"
#: What a base is made with: its recipe, run as root in its builder, and the seed's reader, the Linux disk's own.
RECIPE, SEED_ENV = _DISK / "omarchy", _DISK / "linux" / "boot" / "seed-env"
ADMIN = "root"
#: Where agent-env's own files are on an Omarchy computer, each layer's folder under them, and its services' units.
ROOT = PurePosixPath("/opt/agentenv")
DRIVERS = ROOT / "drivers"
#: Where an app's folder goes, where the driver finds the apps composed onto its computer
#: (``agentenv_protocol.computer.apps``).
APPS = ROOT / "apps"
BUILD = ROOT / "build"
UNITS = PurePosixPath("/etc/systemd/user")
#: What starts every service, with the deploy's environment (``computers/disk/omarchy/with-seed``).
WITH_SEED = ROOT / "with-seed"
OS_ID_LABEL, OS_VERSION_LABEL = "dev.agentenv.os-release.id", "dev.agentenv.os-release.version-id"
#: How long a builder may take to answer over ssh, and its recipe to run.
BOOT_TIMEOUT_S, RECIPE_TIMEOUT_S = 600, 1800
#: The exit code of a recipe that found its image isn't what its artifact declares.
MISLABELLED = 3
#: How many times a provision tries a layer's download, and how long it waits between.
FETCH_TRIES, FETCH_PAUSE_S = 30, 10
#: What a base's build prints once, as it fetches Omarchy's image.
NOTICE = "omarchy: fetching Omarchy's image from {url} (about 4 GB) into {path}, once"

#: Omarchy as a thin disk's guest: root over ssh runs bash, which a compose has unpack its build into ``BUILD`` and run
#: its ``provision.sh``, then trim with the recipe's ``compact.sh``, and power off.
GUEST = thin_disk.Guest(
    name="omarchy", file="linux-desktop", user=ADMIN,
    prepare=f"rm -rf {BUILD} && mkdir -p {BUILD}",
    unpack=f"tar -xf - -C {BUILD}",
    provision=f"bash {BUILD}/provision.sh",
    cleanup=f"rm -rf {BUILD}",
    compact="bash -c " + shlex.quote((RECIPE / "compact.sh").read_text()),
    shutdown="systemctl poweroff")


@dataclass(frozen=True)
class Part:
    """A layer as an Omarchy disk installs it: its folder under ``context/``, its prefix, its install, its services'
    argv by name, the environment they're given, and what its venv installs beside its requirements; for an app, the
    ``<file>:<Class>`` its own server serves, its install steps by name and scope, and the platform they're run for."""

    folder: str
    prefix: PurePosixPath
    install: Install
    services: dict[str, list[str]]
    environment: dict[str, Any]
    wants: tuple[str, ...] = ()
    served: str | None = None
    steps: tuple[tuple[str, str], ...] = ()
    platform: str | None = None


def source_path(sha256: str) -> Path:
    """Where this machine's cache keeps the image of ``sha256``."""
    return cache_root() / "omarchy" / "images" / f"{sha256}.qcow2"


def source(disk_source: DiskSource, say: Callable[[str], None]) -> Path:
    """The image ``disk_source`` names in this machine's cache, fetched first when it isn't there and refused when it
    doesn't come with its sha256."""
    path = source_path(disk_source.sha256)
    with holding_locks([f"omarchy-source:{path}"]):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            say(NOTICE.format(url=disk_source.url, path=path))
            partial, digest = path.with_suffix(".part"), hashlib.sha256()
            with urllib.request.urlopen(disk_source.url, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != disk_source.sha256:
                partial.unlink()
                raise RuntimeError(f"omarchy: {disk_source.url} came with sha256 {digest.hexdigest()}, not "
                                   f"{disk_source.sha256}")
            partial.rename(path)
    return path


def base(key: str, image: Callable[[], Path], user: str, distribution: str, version: str,
         say: Callable[[str], None], registry: Registry) -> Base:
    """The base of ``key``, as ``thin_disk.base`` finds or builds one: built from the image ``image`` gives, its session
    user ``user``, refused unless it's the ``distribution`` and ``version`` its artifact declares, which it records."""
    def built(work: Path, keys: Path) -> dict[str, str]:
        _built(work, image(), keys, user, distribution, version, say)
        return {OS_ID_LABEL: distribution, OS_VERSION_LABEL: version}

    return thin_disk.base(GUEST, key, built, say, registry)


def _built(work: Path, source_image: Path, keys: Path, user: str, distribution: str, version: str,
           say: Callable[[str], None]) -> None:
    """``source_image`` made a base, ``work/base.img``: booted on an overlay by the launcher as a linux-desktop guest,
    root's key in a systemd credential, the recipe run over ssh, then trimmed, powered off and folded into one image."""
    out, recipe = work / "out", work / "recipe"
    out.mkdir()
    shutil.copytree(RECIPE, recipe)
    shutil.copy(SEED_ENV, recipe / "seed-env")
    (out / "guest").write_text(GUEST.file)
    name = f"agentenv-omarchy-build-{keys.name}"
    say(f"booting Omarchy's image in {name}")
    mounts = ("-v", f"{source_image}:/in/source.qcow2:ro")
    with thin_disk.running(name, {"VM_KEEP": "/out/disk.img", "VM_DISK": "/in/source.qcow2",
                                  "VM_CREDENTIALS": f"tmpfiles.extra={_root_key(keys)}"},
                           ["-v", f"{out}:/out", *mounts], launcher()) as (container, port):
        thin_disk.answering(GUEST, container, port, keys, time.monotonic() + BOOT_TIMEOUT_S)
        say("making it a base over ssh")
        thin_disk.ssh(port, keys, ADMIN, "rm -rf /root/agentenv-recipe && mkdir /root/agentenv-recipe")
        thin_disk.ssh(port, keys, ADMIN, "tar -xf - -C /root/agentenv-recipe", input=thin_disk.tarred(recipe))
        done = thin_disk.ssh(port, keys, ADMIN, " ".join(map(shlex.quote, (
            "bash", "/root/agentenv-recipe/recipe.sh", user, distribution, version))), check=False,
            timeout=RECIPE_TIMEOUT_S)
        if done.returncode == MISLABELLED:
            raise RuntimeError(f"omarchy: {done.stderr.strip() or done.stdout.strip()}")
        if done.returncode != 0:
            raise RuntimeError(f"omarchy: its recipe exited {done.returncode}: "
                               f"{(done.stderr + done.stdout).strip()[-3000:]}")
        thin_disk.ssh(port, keys, ADMIN, "rm -rf /root/agentenv-recipe")
        thin_disk.compacted(GUEST, port, keys, say)
        thin_disk.shut_down(GUEST, container, port, keys)
    thin_disk.converted(out, launcher(), args=mounts)
    shutil.move(out / "disk.img", work / "base.img")


def _root_key(keys: Path) -> str:
    """The systemd credential, base64'd, whose tmpfiles lines give root the base's key, as Omarchy applies at boot."""
    lines = (f"d /root/.ssh 0700 root root -\n"
             f"f+ /root/.ssh/authorized_keys 0600 root root - {(keys / 'id.pub').read_text().strip()}\n")
    return base64.b64encode(lines.encode()).decode()


def write(build: Path, parts: list[Part], *, user: str, build_settings: dict[str, Any], image: dict) -> str:
    """Write ``build``'s ``provision.sh``, which installs ``parts`` from ``build/context``, their session-scoped run
    scripts as the session user ``user``, and enables their services for the session, ``build_settings`` as every
    service's ``AGENTENV_SETTING_<NAME>`` and ``image`` as what the disk says it holds, then leaves the computer no ssh:
    only a compose logs in, and it boots the base. Its text, which a composed disk's key covers."""
    lines = ["#!/bin/bash", "set -euo pipefail", f"build={BUILD}", f"user={shlex.quote(user)}",
             'home=$(getent passwd "$user" | cut -d: -f6)', _FETCH]
    for part in parts:
        lines += [f"# {part.folder}", *_installed_part(part)]
    settings = "".join(f"export AGENTENV_SETTING_{name.upper()}={shlex.quote(str(value))}\n"
                       for name, value in build_settings.items())
    lines += [_written(ROOT / "build-environment", settings),
              _written(ROOT / "image.json", json.dumps(image, sort_keys=True)),
              "systemctl mask --quiet sshd.service sshd.socket"]
    script = "\n".join(lines) + "\n"
    (build / "provision.sh").write_text(script)
    return script


def _installed_part(part: Part) -> list[str]:
    """The lines that install ``part``: its files, downloads, venv, an app's own folder, where its server listens, the
    session user's alone, then its run scripts and install steps, each as root or, for ``session``, as the session user
    in their home, and its services as user units of the graphical session, each enabled for every user."""
    prefix, install = part.prefix, part.install
    lines = [f"install -d {shlex.quote(str(prefix))}",
             f'cp -a "$build/context/{part.folder}/." {shlex.quote(str(prefix))}/']
    for download in install.downloads or []:
        if download.arch in (None, "amd64"):
            to = prefix / download.to
            lines += [f"install -d {shlex.quote(str(to.parent))}",
                      f"fetch {shlex.quote(download.url)} {download.sha256} {shlex.quote(str(to))}"]
    if install.python is not None or part.wants:
        venv = prefix / "venv"
        wants = [*(("-r", str(prefix / install.python)) if install.python else ()), *part.wants]
        lines += [f"python3 -m venv {shlex.quote(str(venv))}",
                  f"{venv}/bin/pip install -q --disable-pip-version-check {' '.join(map(shlex.quote, wants))}",
                  f"{venv}/bin/pip list --format=json > {shlex.quote(str(prefix / 'installed.json'))}"]
    if part.served is not None:
        own = shlex.quote(str(prefix / SOCKET.parent))
        lines.append(f'install -d {own} && chown "$user": {own} && chmod 700 {own}')
    for run in install.run or []:
        lines.append(f"(cd {shlex.quote(str(prefix))} && {_as(run.scope)}PREFIX={shlex.quote(str(prefix))} "
                     f"bash {shlex.quote(str(prefix / run.script))})")
    for step, scope in part.steps:
        platform = f"{PLATFORM_ENV}={shlex.quote(part.platform)} " if part.platform else ""
        lines.append(f"(cd {shlex.quote(str(prefix))} && {_as(scope)}{platform}{prefix / 'venv' / 'bin' / 'python'} -m "
                     f"agentenv_protocol.computer.run_install_step {shlex.quote(part.served)} {shlex.quote(step)})")
    for name, argv in part.services.items():
        lines += [_written(UNITS / f"agentenv-{name}.service", unit(name, prefix, argv, part.environment)),
                  f"systemctl --global enable --quiet agentenv-{name}.service"]
    return lines


def _as(scope: str) -> str:
    """What a command starts with to run as ``scope`` says: as root, provision's own user, or for ``session``, as the
    session user in their home."""
    return 'runuser -u "$user" -- env HOME="$home" ' if scope == "session" else ""


def unit(name: str, prefix: PurePosixPath, argv: list[str], environment: dict[str, Any]) -> str:
    """The systemd user unit of the service ``name``: started with the graphical session, which gives it Hyprland's
    instance and Wayland's display, and again with it, as Hyprland restarts under a new instance signature; through
    ``with-seed``, in its ``prefix``, with ``environment`` set after the deploy's, so a layer's own variables win, as on
    every OS; again whenever it exits."""
    own = ["/usr/bin/env", *(f"{key}={value}" for key, value in environment.items())] if environment else []
    lines = ["[Unit]", f"Description=agent-env: {name}", "PartOf=graphical-session.target",
             "After=graphical-session.target", "ConditionEnvironment=WAYLAND_DISPLAY", "",
             "[Service]", f"WorkingDirectory={prefix}",
             "ExecStart=" + " ".join(_unit_word(word) for word in (str(WITH_SEED), *own, *argv)),
             "Restart=always", "RestartSec=1", "",
             "[Install]", "WantedBy=graphical-session.target"]
    return "\n".join(lines) + "\n"


def argv(command: str, prefix: PurePosixPath) -> list[str]:
    """A service's ``command`` as the argv its unit runs, split as a shell would, a relative program named under
    ``prefix``."""
    words = shlex.split(command)
    if "/" in words[0] and not PurePosixPath(words[0]).is_absolute():
        words[0] = str(prefix / words[0])
    return words


def _unit_word(word: str) -> str:
    """``word`` as one word of a unit's line: quoted when it holds more than plain characters, its specifiers and
    variables escaped, as systemd would otherwise expand them."""
    escaped = word.replace("%", "%%").replace("$", "$$")
    if escaped and all(ch.isalnum() or ch in "_./:=+-,@" for ch in escaped):
        return escaped
    return '"' + escaped.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _written(path: PurePosixPath, text: str) -> str:
    """The line that writes ``text`` to ``path``."""
    return f"install -d {shlex.quote(str(path.parent))} && printf %s {shlex.quote(text)} > {shlex.quote(str(path))}"


#: A layer's download, fetched again on a server's error, as the base's recipe does, and refused unless it comes with
#: its sha256.
_FETCH = (f"fetch() {{ for try in $(seq {FETCH_TRIES}); do curl -fsSL -o \"$3\" \"$1\" && break; "
          f"[ \"$try\" -lt {FETCH_TRIES} ] || {{ echo \"provision: $1 couldn't be fetched\" >&2; exit 1; }}; "
          f"sleep {FETCH_PAUSE_S}; done; echo \"$2  $3\" | sha256sum --check --quiet || "
          "{ echo \"provision: $1 came with another sha256 than $2\" >&2; exit 1; }; }")
