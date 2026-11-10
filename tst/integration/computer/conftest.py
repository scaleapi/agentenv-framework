"""The ubuntu-24.04 base computer, built from its folder once for the tests here and tagged for the local registry; the
shipped desktop example's computer env, the cua-driver test bundle's, the chromium test bundle's, the website-browser
test bundle's, the TLS clients test bundle's and agentenv's chromium deployed alone, each composed once for a module's
tests;
``state``, local stores for a test that runs a bundle; ``docker``, which runs the docker CLI for them; and
``cdp_clients``, Playwright and Puppeteer installed to connect to a CDP address."""

import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

import agent_env
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import find_bundle, installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import BuiltImage, resolve_bundle
from agent_env.config import configure, reset_config
from agent_env.env import Env
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from agent_env.task.store import get_task_instance_store
from tst.util.published_image import pull_published_else_build

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"
CUA_DRIVER = Path(__file__).parent / "cua-driver"
CHROMIUM = Path(__file__).parent / "chromium"
WEBSITE_BROWSER = Path(__file__).parent / "website-browser"
BROWSER_COMPUTER = Path(__file__).parent / "browser-computer"
TLS = Path(__file__).parent / "tls"
#: Playwright and Puppeteer, which connect to a cdp address from Node.
CDP_CLIENTS = Path(__file__).parent / "cdp_clients"
CHROMIUM_ALONE = "@local/agentenv-framework/browsers/chromium"


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
    """agentenv's chromium, which deploys alone as a browser computer: the ubuntu-24.04-headless base, with no display, and the
    page driver, whose screen is the browser's page; composed, through the browser-computer test bundle's task that
    deploys it, as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("browser-computer"), parse_bundle(BROWSER_COMPUTER), CHROMIUM_ALONE)


@pytest.fixture(scope="module")
def tls_desktop(tmp_path_factory):
    """The TLS clients test bundle's desktop, the ubuntu-24.04 base and the reference driver with Node, an apt JDK, a
    copy of it under /opt and curl, composed as ``desktop`` is."""
    yield from _composed(tmp_path_factory.mktemp("tls-desktop"), parse_bundle(TLS))


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
    with namespace_routing():
        plan = plan_bundle(resolve_bundle(bundle), installed=installed_bundles)
        materialize(plan, build_here=frozenset(w.id for w in plan.writes if isinstance(w.source, BuiltImage)))
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
    logging.disable(logging.CRITICAL)
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def run_context(output: str) -> dict:
    """The context of the run whose ``agent-env run`` output ``output`` is."""
    with namespace_routing():
        return get_task_instance_store().get(re.search(r"instance (@local/.+)$", output, re.M).group(1)).context
