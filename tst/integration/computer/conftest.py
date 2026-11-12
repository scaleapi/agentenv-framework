"""The ubuntu-24.04 base computer, built from its folder once for the tests here and tagged for the local registry; the
shipped desktop example's computer env, the cua-driver test bundle's, the chromium test bundle's, the website-browser
test bundle's, the TLS clients test bundle's, the clock desktop test bundle's and agentenv's chromium deployed alone, each
composed once for a module's tests;
``state``, local stores for a test that runs a bundle; each of these stores starts with the packages the run has resolved
(``tst.util.package_cache``); ``docker``, which runs the docker CLI for them; and
``cdp_clients``, Playwright and Puppeteer installed to connect to a CDP address. A compose's build of an image waits for
another test worker's build of the same tag."""

import fcntl
import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath

import pytest

from agentenv_protocol.computer.clock import FOLLOWER, FOLLOWER_USER

import agent_env
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import find_bundle, installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import BuiltImage, resolve_bundle
from agent_env.config import configure, reset_config
from agent_env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.computer.utils import builder
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from agent_env.task.store import get_task_instance_store
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs
from tst.util import package_cache
from tst.util.published_image import pull_published_else_build

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"
CUA_DRIVER = Path(__file__).parent / "cua-driver"
CHROMIUM = Path(__file__).parent / "chromium"
WEBSITE_BROWSER = Path(__file__).parent / "website-browser"
BROWSER_COMPUTER = Path(__file__).parent / "browser-computer"
TLS = Path(__file__).parent / "tls"
CLOCK_DESKTOP = Path(__file__).parent / "clock-desktop"
#: Playwright and Puppeteer, which connect to a cdp address from Node.
CDP_CLIENTS = Path(__file__).parent / "cdp_clients"
CHROMIUM_ALONE = "@local/agentenv-framework/browsers/chromium"


@pytest.fixture(scope="session", autouse=True)
def one_build_of_a_tag_at_a_time():
    """Test workers share one Docker daemon, and two composes of the same computer in stores of their own build the
    same tag, which fails with "image ... already exists" when both builds finish at once, so they take turns."""
    built = builder.build_image

    def build_image(dockerfile, context, tag, **kwargs):
        lock = Path(tempfile.gettempdir()) / f"agentenv-test-build-{hashlib.sha256(tag.encode()).hexdigest()[:16]}.lock"
        with open(lock, "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            return built(dockerfile, context, tag, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "build_image", build_image)
        yield


@pytest.fixture(scope="session")
def base():
    """The ubuntu-24.04 base for this machine's platform: the image agent-env published from these sources, pulled, or
    built here when they've changed, so a run doesn't install its desktop from Ubuntu's archive."""
    tag = f"localhost:5000/agentenv-test-ubuntu-24.04:{uuid.uuid4().hex[:8]}"
    pull_published_else_build("ubuntu-24.04", BASE / "Dockerfile", BASE, tag, platform=host_platform())
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.fixture(scope="module")
def desktop(tmp_path_factory):
    """The shipped desktop example's computer env, composed for the module in stores of the module's own: an image for
    each arch this machine's Docker builds, the native one probed by gate 3."""
    bundle = find_bundle("desktop")
    yield from _composed(tmp_path_factory.mktemp("desktop"), parse_bundle(bundle.root, id_root=bundle.id_root))


@pytest.fixture(scope="module")
def office_desktop(tmp_path_factory):
    """The shipped office example's computer env, the ubuntu-24.04 base and the reference driver with the apps bundle's
    libreoffice, composed as ``desktop`` is."""
    bundle = find_bundle("office")
    yield from _composed(tmp_path_factory.mktemp("office"), parse_bundle(bundle.root, id_root=bundle.id_root))


@pytest.fixture(scope="module")
def cua_desktop(tmp_path_factory):
    """The cua-driver test bundle's desktop, the ubuntu-24.04 base with the cua-driver driver, composed as ``desktop``
    is."""
    yield from _composed(tmp_path_factory.mktemp("cua-desktop"), parse_bundle(CUA_DRIVER))


@pytest.fixture(scope="module")
def chromium_desktop(tmp_path_factory):
    """The chromium test bundle's desktop, the ubuntu-24.04 base and the reference driver with agentenv's chromium app,
    composed as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("chromium-desktop"), parse_bundle(CHROMIUM))


@pytest.fixture(scope="module")
def website_browser_desktop(tmp_path_factory):
    """The website-browser test bundle's desktop, the ubuntu-24.04 base and the reference driver with its fixture
    website browser, composed as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("website-browser-desktop"), parse_bundle(WEBSITE_BROWSER))


@pytest.fixture(scope="module")
def browser_computer(tmp_path_factory):
    """agentenv's chromium, which deploys alone as a browser computer: the ubuntu-24.04-browser base, with no desktop, and the
    page driver, whose screen is the browser's page; composed, through the browser-computer test bundle's task that
    deploys it, as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("browser-computer"), parse_bundle(BROWSER_COMPUTER), CHROMIUM_ALONE)


@pytest.fixture(scope="module")
def tls_desktop(tmp_path_factory):
    """The TLS clients test bundle's desktop, the ubuntu-24.04 base and the reference driver with Node, an apt JDK, a
    copy of it under /opt and curl, composed as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("tls-desktop"), parse_bundle(TLS))


@pytest.fixture(scope="module")
def clock_desktop(tmp_path_factory):
    """The clock desktop test bundle's desktop, the ubuntu-24.04 base, whose every process loads agent-env's
    libfaketime, and the reference driver with agentenv's chromium and curl; composed as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("clock-desktop"), parse_bundle(CLOCK_DESKTOP))


@pytest.fixture(scope="module")
def cdp_clients(tmp_path_factory) -> Path:
    """The CDP clients installed in a folder of their own, or the test skipped without Node 20 or later and npm."""
    node, npm = shutil.which("node"), shutil.which("npm")
    version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout if node else ""
    if not npm or not (match := re.match(r"v(\d+)", version)) or int(match.group(1)) < 20:
        pytest.skip("the CDP clients need Node 20 or later and npm")
    folder = tmp_path_factory.mktemp("cdp-clients")
    for name in ("package.json", "connect.mjs"):
        shutil.copy(CDP_CLIENTS / name, folder / name)
    installed = subprocess.run([npm, "install", "--no-audit", "--no-fund", "--loglevel=error"], cwd=folder,
                               capture_output=True, text=True, timeout=600)
    assert installed.returncode == 0, installed.stderr
    return folder


def _composed(root: Path, bundle, env_id: str | None = None):
    patch = pytest.MonkeyPatch()
    patch.chdir(root)
    patch.setenv("XDG_STATE_HOME", str(root / "state"))
    patch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    patch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    configure()
    reset_artifact_store()
    package_cache.seed()
    with namespace_routing():
        plan = plan_bundle(resolve_bundle(bundle), installed=installed_bundles)
        try:
            materialize(plan, build_here=frozenset(w.id for w in plan.writes if isinstance(w.source, BuiltImage)))
        finally:
            package_cache.keep()
        yield Env.get(env_id or f"{bundle.id_root}/desktop")
    reset_artifact_store()
    reset_config()
    patch.undo()


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=True, capture_output=True)


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    package_cache.seed()
    logging.disable(logging.CRITICAL)
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        package_cache.keep()
        logging.disable(logging.NOTSET)
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def run_context(output: str) -> dict:
    """The context of the run whose ``agent-env run`` output ``output`` is."""
    with namespace_routing():
        return get_task_instance_store().get(re.search(r"instance (@local/.+)$", output, re.M).group(1)).context


#: What root runs: tini, the init it started, which is root's supervisord, and the ps that lists them.
ROOTS = ("/usr/bin/tini -- /usr/local/bin/start-desktop init",
         "/usr/bin/python3 /usr/bin/supervisord -c /etc/supervisor/root.conf", "ps -eo user:32=,args=")
#: The clock page's writer, which the base installs beside the follower.
PAGE_WRITER = str(PurePosixPath(FOLLOWER).parent / "clockpage.py")


def process_owners(name: str) -> dict[str, list[str]]:
    """Each process's command line in the container ``name``, by the user that runs it."""
    owners: dict[str, list[str]] = {}
    for line in _exec_text(name, "ps", "-eo", "user:32=,args=").splitlines():
        user, _, args = line.strip().partition(" ")
        owners.setdefault(user, []).append(args.strip())
    return owners


def only_session_and_clock(owners: dict[str, list[str]]) -> bool:
    """Whether root runs nothing but its init and supervisord, the clock's user nothing but the follower, and every
    other process is the session user's."""
    assert set(owners) == {"root", FOLLOWER_USER, "user"}, owners
    assert all(args in ROOTS for args in owners["root"]), owners["root"]
    assert owners[FOLLOWER_USER] == [f"/usr/bin/python3 -I {FOLLOWER}"], owners[FOLLOWER_USER]
    return True


def _exec_text(name: str, *argv: str) -> str:
    return subprocess.run(["docker", "exec", name, *argv], check=True, capture_output=True, text=True).stdout


#: Kills the follower and times it until another runs, inside the container, as root, polling every 20 ms: prints the
#: pid killed, the pid back, and the seconds between.
_KILL_FOLLOWER = """import os, signal, subprocess, sys, time
def running():
    return subprocess.run(["pgrep", "-u", sys.argv[2], "-f", sys.argv[1]], capture_output=True, text=True).stdout.strip()
killed = running()
os.kill(int(killed), signal.SIGKILL)
start = time.monotonic()
while (back := running()) in ("", killed) and time.monotonic() - start < 10:
    time.sleep(0.02)
print(killed, back or "-", time.monotonic() - start)
"""


#: How soon supervisord has a killed follower back: it notices the exit when it next wakes, up to a second later,
#: and the follower starts Python; on a CI runner busy with other tests that has taken 2.03 s.
FOLLOWER_BACK_S = 3


def kill_follower(name: str) -> tuple[str, str, float]:
    """The clock follower in the container ``name`` killed with SIGKILL: its pid, the pid of the one back ("-" if
    none within 10 s), and how long that took, timed in the container."""
    killed, back, took = subprocess.run(["docker", "exec", "-u", "0", name, "python3", "-c", _KILL_FOLLOWER, FOLLOWER,
                                         FOLLOWER_USER],
                                        check=True, capture_output=True, text=True).stdout.split()
    return killed, back, float(took)


#: A multi env's computer on the local backend in its container form, as its child_envs entry gives it.
CONTAINER = {"sandbox_type": "local", "env_params": {"form": "container"}}


@pytest.fixture(scope="module")
def items(desktop):
    """The gateway's and service-db's envs, and the items MCP server under a local id, since a multi env naming the
    desktop's local id must be local too and a store's id can't reference a local one."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return MCPServerEnv.put(id=f"@local/~/office-{uid}/items", environment_name="items",
                                docker_image_artifact=_put_items_image(f"office-items-{uid}"))


def multi_env(items, *computers, websites=()) -> MultiEnv:
    """A multi env named office of the ``items`` MCP server, ``websites`` and ``computers``, under a local id."""
    return MultiEnv.put(id=f"@local/~/office-{uuid.uuid4().hex[:8]}/office", mcp_server_envs=[items],
                        website_envs=list(websites), computer_envs=list(computers), name="office")
