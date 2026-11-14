"""The ubuntu-24.04 base computer, built from its folder once for the tests here and tagged for the local registry; the
shipped desktop example's computer env, the cua-driver test bundle's, the chromium test bundle's, the website-browser
test bundle's, the TLS clients test bundle's, the clock desktop test bundle's and agentenv's chromium deployed alone, each
composed once for a module's tests;
``state``, local stores for a test that runs a bundle; each of these stores starts with the packages the run has resolved
(``tst.util.package_cache``); ``docker``, which runs the docker CLI for them; and
``cdp_clients``, Playwright and Puppeteer installed to connect to a CDP address. A compose's build of an image waits for
another test worker's build of the same tag."""

import ast
import fcntl
import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
import time
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
from agent_env.env.envs.computer.utils import builder, inside
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
EGRESS_CLIENTS = Path(__file__).parent / "egress"
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
def egress_desktop(tmp_path_factory):
    """The egress clients test bundle's desktop, the ubuntu-24.04 base and the reference driver with an apt JDK, an
    official Node build and curl, composed as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("egress-desktop"), parse_bundle(EGRESS_CLIENTS))


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
    for name in ("package.json", "connect.mjs", "scoped.mjs"):
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
#: The base's egress proxy, the user it runs as, and its address.
EGRESS, EGRESS_USER, EGRESS_PROXY = "/usr/local/lib/agentenv-egress/egress", "agentenv-egress", "http://127.0.0.1:3128"


def process_owners(name: str) -> dict[str, list[str]]:
    """Each process's command line in the container ``name``, by the user that runs it."""
    owners: dict[str, list[str]] = {}
    for line in _exec_text(name, "ps", "-eo", "user:32=,args=").splitlines():
        user, _, args = line.strip().partition(" ")
        owners.setdefault(user, []).append(args.strip())
    return owners


def only_session_clock_and_egress(owners: dict[str, list[str]]) -> bool:
    """Whether root runs nothing but its init and supervisord, the clock's user nothing but the follower, the egress
    user nothing but the egress proxy, and every other process is the session user's."""
    assert set(owners) == {"root", FOLLOWER_USER, EGRESS_USER, "user"}, owners
    assert all(args in ROOTS for args in owners["root"]), owners["root"]
    assert owners[FOLLOWER_USER] == [f"/usr/bin/python3 -I {FOLLOWER}"], owners[FOLLOWER_USER]
    assert owners[EGRESS_USER] == [f"{EGRESS} serve"], owners[EGRESS_USER]
    return True


def _exec_text(name: str, *argv: str) -> str:
    return subprocess.run(["docker", "exec", name, *argv], check=True, capture_output=True, text=True).stdout


#: Kills a program of root's supervisord (by its command and user) and times it until another runs, inside the
#: container, as root, polling every 20 ms: prints the pid killed, the pid back, and the seconds between.
_KILL_PROGRAM = """import os, signal, subprocess, sys, time
def running():
    return subprocess.run(["pgrep", "-u", sys.argv[2], "-f", sys.argv[1]], capture_output=True, text=True).stdout.strip()
killed = running()
os.kill(int(killed), signal.SIGKILL)
start = time.monotonic()
while (back := running()) in ("", killed) and time.monotonic() - start < 10:
    time.sleep(0.02)
print(killed, back or "-", time.monotonic() - start)
"""


#: How soon supervisord has a killed program of its own back: it notices the exit when it next wakes, up to a second
#: later, and the program starts; on a CI runner busy with other tests a follower took 2.03 s.
FOLLOWER_BACK_S = 3


def kill_program(name: str, command: str, user: str) -> tuple[str, str, float]:
    """The program running ``command`` as ``user`` in the container ``name`` killed with SIGKILL: its pid, the pid of
    the one back ("-" if none within 10 s), and how long that took, timed in the container."""
    killed, back, took = subprocess.run(["docker", "exec", "-u", "0", name, "python3", "-c", _KILL_PROGRAM, command,
                                         user], check=True, capture_output=True, text=True).stdout.split()
    return killed, back, float(took)


def kill_follower(name: str) -> tuple[str, str, float]:
    return kill_program(name, FOLLOWER, FOLLOWER_USER)


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


class UniverseClock:
    """A deployed computer's clock page, written as its follower's user, as its follower writes it, and what the
    session reads of it."""

    def __init__(self, deployed):
        self.container = computer_container(deployed.sandbox_id)
        self.page = None

    def set(self, universe: float, rate: float) -> dict:
        """The page written to ``universe`` at ``rate``: the write's own answer, its rate, and when it started."""
        started = time.time()
        out = subprocess.run(["docker", "exec", "-u", FOLLOWER_USER, self.container, "python3", PAGE_WRITER, "set",
                              repr(universe), repr(rate)], capture_output=True, text=True, check=True).stdout
        self.page = ast.literal_eval(out.strip())
        self.page.update(rate=rate, started=started)
        return self.page

    def advance(self, seconds: float) -> dict:
        return self.set(self.now() + seconds, self.page["rate"])

    def rate(self, rate: float) -> dict:
        return self.set(self.now(), rate)

    def now(self, real: float | None = None) -> float:
        """The universe at real time ``real`` (now by default), as the page has it."""
        real_ns = (time.time() if real is None else real) * 1e9
        return (self.page["universe_ns"] + max(0.0, real_ns - self.page["anchor_ns"]) * self.page["rate"]) / 1e9

    async def read(self, deployed) -> float | bool:
        """The session's ``date``, when it's the universe between the moments before and after it ran (a few ms either
        side for the two machines' reads of the clock); False otherwise."""
        start = time.time()
        value = float(await session_run(deployed, "date +%s.%N"))
        end = time.time()
        rate = self.page["rate"]
        return value if self.now(start) - 0.01 * max(rate, 1) <= value <= self.now(end) + 0.01 * max(rate, 1) else False


def computer_container(sandbox_id: str) -> str:
    """A deployed computer's container, the one of its sandbox's (each named for the sandbox) holding the clock
    page's writer."""
    listed = subprocess.run(["docker", "ps", "-q", "--filter", f"name={sandbox_id}"],
                            capture_output=True, text=True, check=True).stdout.split()
    for container in listed:
        if subprocess.run(["docker", "exec", container, "test", "-e", PAGE_WRITER]).returncode == 0:
            return container
    raise AssertionError(f"no container of sandbox {sandbox_id} has {PAGE_WRITER}: {listed}")


async def session_run(deployed, command: str) -> str:
    """What ``command`` prints, run in the deployed computer's session through its driver; it must succeed."""
    ran = await inside.run(deployed, ["sh", "-c", command], timeout_s=30)
    assert ran["exit_code"] == 0, ran
    return ran["stdout"].strip()
