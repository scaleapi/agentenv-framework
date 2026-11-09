"""agentenv-session: the session's services on a Windows computer, run windowless by pythonw at the session user's sign-in.
Each service a composed layer declares (C:\\agentenv\\services\\<name>.json: its prefix, command and environment) starts
in its prefix with the build-phase settings the disk fixes, the deploy's environment over them, read from the NoCloud
seed the launcher attaches (the CIDATA volume; agentenv_protocol.computer.seed, version 1), and its own environment over
that; one that exits, or couldn't start, starts again after a pause, what kept it from starting written to its log. It
waits for the seed, and refuses one that carries only versions it doesn't read, as an init does. A seed's trust folder,
a local transfer CA's, is what TLS clients are pointed at, as a container's are."""

import base64
import ctypes
import json
import os
import string
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("C:/agentenv")
SERVICES, LOGS = ROOT / "services", ROOT / "logs"
BUILD_SETTINGS = ROOT / "build-settings.json"
#: The seed versions this reads, oldest first.
READS = (1,)
HEADER = "#agentenv-environment"
WAIT_S = 120
RESTART_S = 2
NO_WINDOW = 0x08000000


def seed_folder() -> Path | None:
    """The root of the volume labelled CIDATA, if one is mounted."""
    label = ctypes.create_unicode_buffer(64)
    for letter in string.ascii_uppercase[3:]:
        root = f"{letter}:\\"
        if ctypes.windll.kernel32.GetVolumeInformationW(root, label, 64, None, None, None, None, 0) and \
                label.value.upper() == "CIDATA":
            return Path(root)
    return None


def environment(folder: Path) -> dict[str, str]:
    """The variables of the newest version the seed in ``folder`` carries that this reads, from the one its
    agentenv-seed-oldest names (1 without it); raises SystemExit when it carries none."""
    try:
        oldest = int((folder / "agentenv-seed-oldest").read_text().strip() or 1)
    except (FileNotFoundError, ValueError):
        oldest = 1
    if oldest > READS[-1]:
        raise SystemExit(f"agentenv-session: this computer reads seed v{READS[-1]}; this agent-env writes v{oldest} or "
                         "later; recompose it with agent-env env build")
    newest = max(version for version in READS if version >= oldest)
    found = {}
    for line in (folder / ("user-data" if newest == 1 else f"user-data.v{newest}")).read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.startswith("#") and key.isidentifier():
            found[key] = base64.b64decode(value).decode()
    return found


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + WAIT_S
    while (folder := seed_folder()) is None or not (folder / "user-data").is_file():
        if time.monotonic() > deadline:
            raise SystemExit(f"agentenv-session: no CIDATA seed after {WAIT_S} s")
        time.sleep(1)
    fixed = json.loads(BUILD_SETTINGS.read_text()) if BUILD_SETTINGS.is_file() else {}
    base = {**os.environ, **fixed, **trust(folder / "trust"), **environment(folder)}
    specs = [json.loads(path.read_text()) for path in sorted(SERVICES.glob("*.json"))]
    running = {spec["name"]: start(spec, base) for spec in specs}
    while True:
        time.sleep(RESTART_S)
        for spec in specs:
            if (process := running[spec["name"]]) is None or process.poll() is not None:
                running[spec["name"]] = start(spec, base)


def trust(folder: Path) -> dict[str, str]:
    """The variables that point TLS clients at the trust files in ``folder``, none when it has none."""
    if not (folder / "ca-bundle.pem").is_file():
        return {}
    return {"SSL_CERT_FILE": str(folder / "ca-bundle.pem"), "REQUESTS_CA_BUNDLE": str(folder / "ca-bundle.pem"),
            "NODE_EXTRA_CA_CERTS": str(folder / "ca.pem")}


def start(spec: dict, base: dict[str, str]) -> subprocess.Popen | None:
    """``spec``'s service started, its output to its log; None when it couldn't be, the reason in its log if that
    opens."""
    try:
        with open(LOGS / f"{spec['name']}.log", "ab") as log:
            try:
                return subprocess.Popen(spec["command"], cwd=spec["prefix"], env={**base, **spec.get("env", {})},
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        creationflags=NO_WINDOW)
            except OSError as e:
                log.write(f"agentenv-session: couldn't start {spec['command']}: {e}\n".encode())
    except OSError:
        pass
    return None


if __name__ == "__main__":
    main()
    sys.exit(0)
