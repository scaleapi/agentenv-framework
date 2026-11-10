"""Windows computers' disks, thin disks (``thin_disk``) made with this machine's Docker and KVM, which keeps them, as a
Mac keeps the VM bundles composed on it: Windows isn't agent-env's to publish. A base is Windows installed from
Microsoft's evaluation media, fetched once into this machine's cache, by the QEMU launcher with the answers and setup of
``computers/disk/windows``; a composed disk's parts are installed by a PowerShell script, ``provision.ps1``, run as the
base's admin. An evaluation runs for 90 days from its install, so a base records when it expires, and one with fewer
than ``RENEW_DAYS`` left is installed afresh."""

from __future__ import annotations

import base64
import json
import re
import secrets
import shlex
import shutil
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from agentenv_protocol.computer.app import ADDRESS

from agent_env.config.paths import cache_root
from agent_env.env.envs.computer.utils import thin_disk
from agent_env.env.envs.computer.utils.install import WindowsInstall
from agent_env.env.envs.computer.utils.thin_disk import Base, Registry
from agent_env.providers.sandbox_providers.local_docker.disk import launcher
from agent_env.store.local_state import holding_locks

#: Microsoft's link to the Windows 11 Enterprise evaluation ISO, English (United States), x64.
ISO_URL = "https://go.microsoft.com/fwlink/?linkid=2334167&clcid=0x409"
#: What a base is installed with: the answers setup reads, the setup it runs, and agentenv-session.
RECIPE = Path(__file__).resolve().parents[4] / "computers" / "disk" / "windows"
ADMIN = "agentadmin"
#: Where agent-env's own files are on a Windows computer, and each layer's folder under them.
ROOT = PureWindowsPath("C:/agentenv")
DRIVERS = ROOT / "drivers"
PYTHON = ROOT / "python" / "python.exe"
BUILD = ROOT / "build"
#: The fewest days of its evaluation a base may have left, as a deploy needs some.
RENEW_DAYS = 7
EXPIRES_LABEL, BUILD_LABEL = "dev.agentenv.windows.expires", "dev.agentenv.windows.build"
#: How long an install may take, and how long the session user's first sign-in is left to settle.
INSTALL_TIMEOUT_S, SETTLE_S = 3600, 60
#: How often an install looks for setup's end.
INSTALL_POLL_S = 15
#: How many times a compose tries a layer's download, and how long it waits between, as the base's setup does: a
#: server's errors can last minutes, as GitHub's release downloads' 504s have. A client error, 4xx, but a timeout or
#: a rate limit, fails at once, as no retry mends a wrong URL.
FETCH_TRIES, FETCH_PAUSE_S = 30, 10
#: The groups and the account the folders under ``ROOT`` grant access to, by their well-known SIDs, as icacls takes
#: them.
ADMINISTRATORS, SYSTEM, USERS = "*S-1-5-32-544", "*S-1-5-18", "*S-1-5-32-545"
#: ALL APPLICATION PACKAGES and ALL RESTRICTED APPLICATION PACKAGES: the identities a sandboxed process runs as, Chrome's
#: and its kind's, which read and run their own program where they're installed, as Program Files lets them.
APP_PACKAGES, RESTRICTED_APP_PACKAGES = "*S-1-15-2-1", "*S-1-15-2-2"
#: What every service runs with: Python reading and writing files as UTF-8, as on Linux, not in Windows' code page.
UTF8 = {"PYTHONUTF8": "1"}
#: What a base's install prints once, as it fetches Microsoft's media.
NOTICE = ("windows: fetching the Windows 11 Enterprise evaluation from Microsoft ({url}, about 7 GB) into {path}, "
          "once; it's installed for evaluation under Microsoft's terms and runs 90 days from each install")


def powershell(script: str) -> str:
    """The command that runs ``script`` in PowerShell, encoded, so cmd's quoting never touches it."""
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + base64.b64encode(
        script.encode("utf-16-le")).decode()


#: Windows as a thin disk's guest: its admin over ssh runs cmd, which a compose has unpack its build into ``BUILD`` and
#: run its ``provision.ps1``, and trims its free space with the recipe's ``compact.ps1``.
GUEST = thin_disk.Guest(
    name="windows", file="windows", user=ADMIN,
    prepare=f'(if exist "{BUILD}" rmdir /s /q "{BUILD}") & mkdir "{BUILD}"',
    unpack=f'tar -xf - -C "{BUILD}"',
    provision=f'powershell -NoProfile -ExecutionPolicy Bypass -File "{BUILD}\\provision.ps1"',
    cleanup=f'rmdir /s /q "{BUILD}"',
    compact=powershell((RECIPE / "compact.ps1").read_text()),
    shutdown="shutdown /s /t 0")


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
    """The base of ``key``, as ``thin_disk.base`` finds or builds one: Windows installed from the ISO ``iso`` gives, its
    session user ``user``, on a disk of ``disk_gb``, recording when its evaluation expires and its build. One with fewer
    than ``RENEW_DAYS`` of its evaluation left is installed afresh."""
    def installed(work: Path, keys: Path) -> dict[str, str]:
        facts = _installed(work, iso(), keys, disk_gb, user, say)
        expires = datetime.now(timezone.utc) + timedelta(minutes=facts["grace_minutes"])
        return {EXPIRES_LABEL: expires.isoformat(timespec="seconds"), BUILD_LABEL: facts["build"]}

    return thin_disk.base(GUEST, key, installed, say, registry, stale=_expiring)


def _expiring(labels: dict[str, str]) -> str | None:
    """Why a base of ``labels`` is installed afresh: its evaluation nears its end; None while it doesn't."""
    if not expiring(expires := labels.get(EXPIRES_LABEL)):
        return None
    if expires is None:
        return "its base doesn't record when its evaluation expires"
    return f"its base's evaluation expires {datetime.fromisoformat(expires):%Y-%m-%d}"


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
    (out / "guest").write_text(GUEST.file)
    (keys / "passwords.json").write_text(json.dumps({ADMIN: admin_password, user: user_password}))
    (keys / "passwords.json").chmod(0o600)
    name = f"agentenv-windows-install-{keys.name}"
    say(f"installing Windows from {iso.name} under QEMU in {name}; this takes about 15 minutes")
    started = time.monotonic()
    with thin_disk.running(
            name, {"VM_KEEP": "/out/disk.img", "VM_DISK_GB": str(disk_gb), "VM_MEDIA": "/in/media.iso",
                   "VM_FILES": "/in/answers"},
            ["-v", f"{out}:/out", "-v", f"{iso}:/in/media.iso:ro", "-v", f"{answers}:/in/answers:ro"],
            launcher()) as (container, port):
        deadline = started + INSTALL_TIMEOUT_S
        while not _signed_in(port, keys, user):
            thin_disk.alive(GUEST, container, "installing", deadline)
            time.sleep(INSTALL_POLL_S)
        say(f"installed and signed in as {user} after {time.monotonic() - started:.0f} s; settling")
        time.sleep(SETTLE_S)
        facts = json.loads(thin_disk.ssh(port, keys, ADMIN, powershell(_FACTS)).stdout.strip().splitlines()[-1])
        thin_disk.compacted(GUEST, port, keys, say)
        thin_disk.shut_down(GUEST, container, port, keys)
    thin_disk.converted(out, launcher())
    shutil.move(out / "disk.img", work / "base.img")
    return facts


def expiring(expires: str | None) -> bool:
    """Whether an evaluation that ends at ``expires`` has fewer than ``RENEW_DAYS`` left, as a disk with none recorded
    is taken to."""
    return expires is None or datetime.fromisoformat(expires) - datetime.now(timezone.utc) < timedelta(days=RENEW_DAYS)


def write(build: Path, parts: list[Part], *, user: str, build_settings: dict[str, Any], image: dict) -> str:
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
    """The lines that leave ``ROOT`` the admins', as on Linux ``/opt/agentenv`` is root's: every user, and every
    sandboxed process, reads and runs what's there, none of the access ``C:\\`` gives every user to change it is left,
    and the session user ``user`` writes only the services' logs, and each app's own folder, which keeps its own
    access."""
    return [_granted_alone(ROOT, {ADMINISTRATORS: "F", SYSTEM: "F", USERS: "RX", APP_PACKAGES: "RX",
                                  RESTRICTED_APP_PACKAGES: "RX"}),
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
                  _granted_alone(own, {user: "F", SYSTEM: "F", ADMINISTRATORS: "F"})]
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


def _granted_alone(folder: PureWindowsPath, access: dict[str, str]) -> str:
    """The line that leaves ``folder`` with only ``access``, rights by account, which what's under it inherits, and
    none of the access its parents give."""
    granted = " ".join(_ps(f"{account}:(OI)(CI){rights}") for account, rights in access.items())
    return f"Checked {{ icacls '{folder}' /inheritance:r /grant:r {granted} | Out-Null }}"


def argv(command: str, prefix: PureWindowsPath) -> list[str]:
    """A service's ``command`` as the argv agentenv-session runs, split as Windows quotes it, a word in double quotes
    one word without them, and a relative program named under ``prefix``."""
    words = [word[1:-1] if len(word) > 1 and word[0] == word[-1] == '"' else word
             for word in shlex.split(command, posix=False)]
    if "/" in words[0] and not PureWindowsPath(words[0]).is_absolute():
        words[0] = str(prefix / PureWindowsPath(words[0]))
    return words




def _signed_in(port: int, keys: Path, user: str) -> bool:
    """Whether setup has finished and ``user`` has an active session on the console; refused, with setup's error, when
    setup failed."""
    done = thin_disk.ssh(port, keys, ADMIN, f'if exist "{ROOT}\\setup-failed" (type "{ROOT}\\setup-failed" & exit 3) else '
                           f'if exist "{ROOT}\\setup-done" (quser) else (echo setting-up)', check=False, timeout=60)
    if done.returncode == 3:
        raise RuntimeError(f"windows: setup failed in the guest: {done.stdout.strip()[-2000:]}")
    return re.search(rf"^\s*>?{re.escape(user)}\s+console\s+\d+\s+Active", done.stdout,
                     re.MULTILINE | re.IGNORECASE) is not None


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
          "Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $To; break } catch { "
          "$status = [int]$_.Exception.Response.StatusCode; "
          f"if ($try -ge {FETCH_TRIES} -or ($status -ge 400 -and $status -lt 500 -and $status -notin 408, 429)) "
          f"{{ throw }}; Start-Sleep {FETCH_PAUSE_S} }} }}; $got = (Get-FileHash -Algorithm SHA256 $To).Hash.ToLower(); "
          "if ($got -ne $Sha256) { throw \"provision: $Url came with sha256 $got, not $Sha256\" } }")
