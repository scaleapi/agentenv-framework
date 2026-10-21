"""Pull the Docker official images that Dockerfiles build from on public.ecr.aws from mirror.gcr.io instead, under the
names the Dockerfiles use, so their builds don't pull them from public.ecr.aws.

ECR Public rate-limits anonymous pulls by source IP, and hosted runners share IPs, so its pulls fail for hours at a time.
mirror.gcr.io serves the same images under the same digests, but keeps one only while it's asked for, so a base it
doesn't give is left to its build to pull from public.ecr.aws.

Docker 28 resolves a digest-pinned FROM only from an image under that exact name, which ``docker tag`` refuses, so those
are also named in containerd, whose images Docker reads only on the containerd image store. Switching dockerd to it
restarts dockerd, which stops the containers it runs, so this runs before any start.

``mirror_base_images.py PATH...`` does this for the Dockerfiles at or under each PATH.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ECR = "public.ecr.aws/docker/library/"
MIRROR = "mirror.gcr.io/library/"
_FROM = re.compile(rf"^FROM\s+(?:--\S+\s+)*({re.escape(ECR)}\S+)", re.IGNORECASE | re.MULTILINE)
_PULL_WAITS_SECONDS = (10, 20)
_DAEMON_CONFIG = Path("/etc/docker/daemon.json")
_CONTAINERD_STORE = "io.containerd.snapshotter"


def main(argv: list[str]) -> int:
    bases = _bases([Path(arg) for arg in argv])
    if any("@" in base for base in bases):
        _use_containerd_store()
    for base in bases:
        _pull(base)
    return 0


def _bases(paths: list[Path]) -> list[str]:
    """The public.ecr.aws images the Dockerfiles at or under ``paths`` build from."""
    dockerfiles = [found for path in paths for found in ([path] if path.is_file() else sorted(path.rglob("Dockerfile*")))]
    return sorted({base for dockerfile in dockerfiles if dockerfile.is_file()
                   for base in _FROM.findall(dockerfile.read_text())})


def _mirrored(base: str) -> str:
    """What mirror.gcr.io serves for ``base``: the image under its digest when it's pinned to one, else under its tag."""
    name, _, digest = base.removeprefix(ECR).partition("@")
    return f"{MIRROR}{name.split(':')[0]}@{digest}" if digest else f"{MIRROR}{name}"


def _pull(base: str) -> None:
    source = _mirrored(base)
    for wait in (*_PULL_WAITS_SECONDS, None):
        if subprocess.run(["docker", "pull", "--quiet", source]).returncode == 0:
            break
        if wait is None:
            print(f"::warning::couldn't pull {source}, so {base}'s build will pull it", flush=True)
            return
        time.sleep(wait)
    name, _, digest = base.partition("@")
    if not digest or ":" in name.rsplit("/", 1)[-1]:
        _run(["docker", "tag", source, name])
    if digest:
        _run(["sudo", "ctr", "--namespace", "moby", "images", "tag", "--force", source, base])
    print(f"{base} is {source}", flush=True)


def _use_containerd_store() -> None:
    """Switch dockerd to the containerd image store, which restarts it, unless it's on that store already."""
    if _CONTAINERD_STORE in _driver_status():
        return
    config = json.loads(_DAEMON_CONFIG.read_text().strip() or "{}") if _DAEMON_CONFIG.exists() else {}
    config.setdefault("features", {})["containerd-snapshotter"] = True
    subprocess.run(["sudo", "tee", str(_DAEMON_CONFIG)], input=json.dumps(config, indent=2), text=True,
                   stdout=subprocess.DEVNULL, check=True)
    _run(["sudo", "systemctl", "restart", "docker"])
    if _CONTAINERD_STORE not in _driver_status():
        raise SystemExit("dockerd isn't on the containerd image store after restarting")


def _driver_status() -> str:
    return subprocess.run(["docker", "info", "--format", "{{json .DriverStatus}}"], capture_output=True, text=True,
                          check=True).stdout


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
