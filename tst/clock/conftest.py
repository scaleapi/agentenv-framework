"""The clock tier: agent-env's libfaketime (src/agent_env/computers/clock) built in an Ubuntu 24.04 image with stock
0.9.12 beside it and the gate programs, each gate run in a container of that image by harness.py. Needs Docker; the
image is for the Docker daemon's own arch, and Chrome is the browsers bundle's pinned build for it. Chrome's archive and
upstream libfaketime's tarball are fetched once into AGENTENV_TEST_CLOCK_CACHE (~/.cache/agentenv-clock by default),
checked by sha256, which CI keeps between runs."""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECIPE = REPO / "src" / "agent_env" / "computers" / "clock"
CHROMIUM_APP = REPO / "src" / "agent_env" / "browsers" / "envs" / "chromium" / "chromium.py"
IMAGE = "agentenv-clock-gates:test"
#: Tries at the downloads and the image build (its apt-get), which fail now and then on a network's hiccup.
TRIES = 3
CACHE = Path(os.environ.get("AGENTENV_TEST_CLOCK_CACHE") or Path.home() / ".cache" / "agentenv-clock")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tried(attempt):
    """``attempt()``'s result, tried up to TRIES times, 10 s apart, while it raises."""
    for n in range(1, TRIES + 1):
        try:
            return attempt()
        except Exception:
            if n == TRIES:
                raise
            time.sleep(10)


def _fetched(url: str, sha256: str, name: str) -> Path:
    """``url`` in the cache as ``name``, downloaded once and checked against ``sha256``."""
    path = CACHE / name
    if path.exists() and _sha256(path) == sha256:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    part = path.with_name(name + ".part")
    _tried(lambda: urllib.request.urlretrieve(url, part))
    assert _sha256(part) == sha256, f"{url} doesn't have the pinned sha256 {sha256}"
    part.replace(path)
    return path


def _built(context: Path, build_args: dict[str, str]) -> None:
    """The gate image built from ``context``."""
    args = [a for k, v in build_args.items() for a in ("--build-arg", f"{k}={v}")]

    def build():
        built = subprocess.run(["docker", "build", "-q", "-t", IMAGE, *args, str(context)], capture_output=True,
                               text=True)
        if built.returncode:
            raise RuntimeError(f"the gate image didn't build:\n{built.stderr[-4000:]}")

    _tried(build)


def _libfaketime_pin() -> tuple[str, str]:
    """The upstream tarball and sha256 build.sh pins, so the two can't drift apart."""
    text = (RECIPE / "build.sh").read_text()
    return re.search(r"https://\S+\.tar\.gz", text).group(0), re.search(r"\b[0-9a-f]{64}\b", text).group(0)


def _chrome_pin(arch: str) -> tuple[str, str, str]:
    """The browsers bundle's Chrome for Testing archive for ``arch``, its sha256, and the packages it links against."""
    spec = importlib.util.spec_from_file_location("clock_tier_chromium_app", CHROMIUM_APP)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    path, sha256 = app._ARCHIVES[arch]
    url = f"https://storage.googleapis.com/chrome-for-testing-public/{app.VERSION}/{path}"
    return url, sha256, " ".join(p for p in app.PACKAGES if p != "unzip")


@pytest.fixture(scope="session")
def gate(tmp_path_factory):
    """Run one of harness.py's gates in a fresh container: gate(NAME, *ARGS, docker_args=()) -> its JSON."""
    arch = subprocess.run(["docker", "version", "--format", "{{.Server.Arch}}"], capture_output=True, text=True,
                          check=True).stdout.strip()
    url, sha256, packages = _chrome_pin(arch)
    context = tmp_path_factory.mktemp("clock-gates")
    shutil.copytree(RECIPE, context / "clock", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(HERE / "programs", context / "programs")
    (context / "downloads").mkdir()
    shutil.copy(_fetched(url, sha256, f"chrome-{arch}-{sha256[:12]}.zip"), context / "downloads" / "chrome.zip")
    shutil.copy(_fetched(*_libfaketime_pin(), "libfaketime-0.9.12.tar.gz"), context / "downloads" / "libfaketime.tar.gz")
    for name in ("Dockerfile", "harness.py"):
        shutil.copy(HERE / name, context / name)
    _built(context, {"CHROME_SHA256": sha256, "CHROME_PACKAGES": packages})

    def run(name: str, *args, timeout: float = 600, docker_args=()):
        r = subprocess.run(["docker", "run", "--rm", "--shm-size=1g", *docker_args, IMAGE, "python3",
                            "/opt/clock/harness.py", name, *map(str, args)], capture_output=True, text=True,
                           timeout=timeout)
        assert r.returncode == 0, f"gate {name} failed:\n{r.stderr[-3000:]}"
        return json.loads(r.stdout)

    return run


#: futex_waitv answered ENOSYS, as a kernel before 5.16 or gVisor answers it; everything else allowed.
NO_FUTEX_WAITV = {"defaultAction": "SCMP_ACT_ALLOW",
                  "syscalls": [{"names": ["futex_waitv"], "action": "SCMP_ACT_ERRNO", "errnoRet": 38}]}


@pytest.fixture(scope="session")
def gate_without_waitv(gate, tmp_path_factory):
    """``gate`` in containers whose seccomp profile refuses futex_waitv, checked to refuse it."""
    profile = tmp_path_factory.mktemp("seccomp") / "no-futex-waitv.json"
    profile.write_text(json.dumps(NO_FUTEX_WAITV))

    def run(name: str, *args, timeout: float = 600):
        return gate(name, *args, timeout=timeout, docker_args=["--security-opt", f"seccomp={profile}"])

    assert run("probe") == {"futex_waitv_errno": 38}
    return run


@pytest.fixture(scope="module", params=["futex_waitv", "without futex_waitv"])
def waits_gate(request, gate, gate_without_waitv):
    """The gates for waits that use futex_waitv, run with it and without it (where they wait in slices instead)."""
    return gate if request.param == "futex_waitv" else gate_without_waitv
