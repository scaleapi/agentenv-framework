"""What composing a macOS computer's VM bundle writes into the build folder the VM it's composed in shares read-write:
the base's readying (``computers/vm_bundle/macos``), each layer's files under ``context/``, its services as LaunchAgents
under ``agents/``, and ``provision.sh``, which the VM runs as its session user, root through sudo with the password
beside them. A layer installs as its ``[install.macos]`` says: its files under its prefix, its downloads checked by
sha256, its venv on the pinned Python ``prepare`` installs, from the wheels of its lock, staged under ``packages/``
and installed offline, its run scripts and install steps, its grants, and its services, each through
``agentenv-service``, which gives it the deploy's environment, or, when direct, as its own command. ``resolver`` is the
script a VM of the base runs to resolve those wheels, once for given inputs."""

from __future__ import annotations

import json
import plistlib
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentenv_protocol.computer.app import SOCKET

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.env.envs.computer.utils.install import MacInstall
from agent_env.env.envs.computer.utils.packages import staged_wheels
from agent_env.providers.sandbox_providers.lume.provider import BUILD_FOLDER, SHARED_FILES

#: The base's readying, copied into each build folder.
PREPARE = Path(__file__).resolve().parents[4] / "computers" / "vm_bundle" / "macos"
#: Where the base's Python and agent-env's own programs are in a composed VM bundle.
BIN = "/opt/agentenv/bin"
#: The architecture a macOS VM bundle runs on.
ARCH = "arm64"


@dataclass(frozen=True)
class Part:
    """A layer as a VM bundle installs it: its folder under ``context/``, its prefix, its install, its programs by name
    (each a command, and when launchd runs it directly, the variables it has), the environment the others are given,
    what its venv adds to its requirements (the protocol), the wheels of its lock its venv installs, and for an app,
    the class its server serves and its install steps."""

    folder: str
    prefix: str
    install: MacInstall
    programs: dict[str, tuple[str, dict[str, str] | None]]
    environment: dict[str, Any]
    venv: bool
    protocol: str | None = None
    served: str | None = None
    steps: tuple[tuple[str, str], ...] = ()
    wheels: tuple[str, ...] = ()


def write(build: Path, base: ComputerArtifact, parts: list[Part], password: str,
          build_settings: dict[str, Any]) -> str:
    """Write ``build`` for ``parts`` on ``base``, with the values of the base's build-phase settings the VM bundle fixes,
    which every service gets as ``AGENTENV_SETTING_<NAME>``, and the password its sudo reads; the text of its provision
    script, which, with the context and the LaunchAgents, a VM bundle's key covers."""
    shutil.copytree(PREPARE, build / "prepare")
    (build / "agents").mkdir()
    home, lines = f"/Users/{base.session.user}", ["#!/bin/bash", "set -euo pipefail",
                                                   f'. "{SHARED_FILES}/{BUILD_FOLDER}/prepare/lib.sh"',
                                                   'bash "$build/prepare/prepare.sh"']
    (build / "agents" / "dev.agentenv.set-display.plist").write_bytes(plistlib.dumps(
        _agent("dev.agentenv.set-display", [f"{BIN}/set-display"], "/", home, keep_alive=False)))
    for part in parts:
        lines += [f"# {part.folder}", *_installed(part)]
        for name, (command, direct) in part.programs.items():
            label = f"dev.agentenv.{name}"
            words = shlex.split(command)
            if "/" in words[0] and not words[0].startswith("/"):
                words[0] = f"{part.prefix}/{words[0]}"
            arguments = words if direct is not None else [
                f"{BIN}/python3", f"{BIN}/agentenv-service", part.prefix,
                *(f"{key}={value}" for key, value in part.environment.items()), "--", *words]
            (build / "agents" / f"{label}.plist").write_bytes(plistlib.dumps(
                _agent(label, arguments, part.prefix, home, environment=direct)))
    lines += ['cp "$build"/agents/*.plist "$HOME/Library/LaunchAgents/"',
              'as_root cp "$build/image.json" "$build/build-settings.json" /etc/agentenv/', "sync"]
    (build / "build-settings.json").write_text(json.dumps(
        {f"AGENTENV_SETTING_{name.upper()}": str(value) for name, value in build_settings.items()}, sort_keys=True))
    (build / "image.json").write_text(json.dumps({"base": base.forms.vm_bundle.image,
                                                  "layers": [part.folder for part in parts]}, sort_keys=True))
    script = "\n".join(lines) + "\n"
    (build / "provision.sh").write_text(script)
    secret = build / ".password"
    secret.write_text(password + "\n")
    secret.chmod(0o600)
    return script


def _installed(part: Part) -> list[str]:
    """The lines that install ``part``: its files, downloads, venv, an app's socket folder, its run scripts and install
    steps, as root or the session user, and its grants."""
    prefix, install = part.prefix, part.install
    quoted = shlex.quote(prefix)
    lines = [f"as_root mkdir -p {quoted}", f'as_root chown "$(id -un):staff" {quoted}',
             f'ditto "$build/context/{part.folder}" {quoted}']
    for download in install.downloads or []:
        if download.arch in (None, ARCH):
            to = shlex.quote(f"{prefix}/{download.to}")
            lines += [f"mkdir -p \"$(dirname {to})\"", f"curl -fsSL --retry 3 -o {to} {shlex.quote(download.url)}",
                      f'echo "{download.sha256}  "{to} | shasum -a 256 -c - > /dev/null']
    if part.venv:
        staged = staged_wheels(part.folder)
        lines.append(f"{BIN}/python3 -m venv {quoted}/venv")
        if part.wheels:
            lines.append(f"{prefix}/venv/bin/pip install -q --disable-pip-version-check --no-index --no-deps "
                         + " ".join(f'"$build/{staged}/"{shlex.quote(wheel)}' for wheel in part.wheels))
        lines.append(f"{prefix}/venv/bin/pip list --format=json > {quoted}/installed.json")
    if part.served is not None:
        lines.append(f"mkdir -p {shlex.quote(f'{prefix}/{SOCKET.parent}')}")
    commands = [*((run.scope, shlex.quote(f"./{run.script}")) for run in install.run or []),
                *((scope, f"venv/bin/python -m agentenv_protocol.computer.run_install_step {part.served} {step}")
                  for step, scope in part.steps)]
    for scope, command in commands:
        run = f"cd {quoted} && {command}"
        lines.append(f"as_root env PREFIX={quoted} bash -c {shlex.quote(run)}" if scope == "system" else f"({run})")
    lines += [f"grant {shlex.quote(grant.app)} {' '.join(grant.services)}" for grant in install.grants or []]
    return lines


def resolver(parts: list[Part]) -> str:
    """The script a VM of the base runs, as its session user, to resolve the wheels of ``parts``' venvs: it installs the
    pinned Python they run on, then builds each venv's wheels from its requirements and what it adds (the protocol)
    into ``staged_wheels`` of the folder the VM shares, one resolve of them all, as a lock's is on Linux."""
    lines = ["#!/bin/bash", "set -euo pipefail", f'. "{SHARED_FILES}/{BUILD_FOLDER}/prepare/lib.sh"', "pinned_python"]
    for part in parts:
        if not part.venv:
            continue
        staged, venv = staged_wheels(part.folder), shlex.quote(f"/tmp/agentenv-resolve/{part.folder}")
        wants = [*(("-r", f'"$build/context/{part.folder}/"{shlex.quote(part.install.python)}')
                   if part.install.python else ()), *((part.protocol,) if part.protocol else ())]
        lines += [f'mkdir -p "$build/{staged}"', f"{BIN}/python3 -m venv {venv}"]
        if wants:
            lines.append(f"{venv}/bin/pip wheel -q --disable-pip-version-check --no-cache-dir --wheel-dir "
                         f'"$build/{staged}" {" ".join(wants)}')
    return "\n".join(lines) + "\n"


def _agent(label: str, arguments: list[str], directory: str, home: str, keep_alive: bool = True,
           environment: dict[str, str] | None = None) -> dict:
    """A LaunchAgent of the login session that runs ``arguments`` in ``directory`` from login, with ``environment``,
    again whenever it exits when ``keep_alive``, logging to ``~/Library/Logs/agentenv``."""
    log = f"{home}/Library/Logs/agentenv/{label}.log"
    return {"Label": label, "ProgramArguments": arguments, "WorkingDirectory": directory, "RunAtLoad": True,
            "KeepAlive": keep_alive, "LimitLoadToSessionType": "Aqua", "StandardOutPath": log, "StandardErrorPath": log,
            "EnvironmentVariables": {"PATH": f"{BIN}:/usr/bin:/bin:/usr/sbin:/sbin", **(environment or {})}}
