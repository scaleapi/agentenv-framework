"""The ubuntu-24.04 base computer, built from its folder once for the tests here and tagged for the local registry; the
shipped desktop example's computer env, composed once for a module's tests; and ``docker``, which runs the docker CLI
for them."""

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
from agent_env.store.routing import namespace_routing

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"


@pytest.fixture(scope="session")
def base():
    tag = f"localhost:5000/agentenv-test-ubuntu-24.04:{uuid.uuid4().hex[:8]}"
    docker("build", "-q", "-t", tag, str(BASE))
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.fixture(scope="module")
def desktop(tmp_path_factory):
    """The shipped desktop example's computer env, composed for the module in stores of the module's own: an image for
    each arch this machine's Docker builds, the native one probed by gate 3."""
    root = tmp_path_factory.mktemp("desktop")
    patch = pytest.MonkeyPatch()
    patch.chdir(root)
    patch.setenv("XDG_STATE_HOME", str(root / "state"))
    patch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    patch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    configure()
    reset_artifact_store()
    bundle = find_bundle("desktop")
    with namespace_routing():
        plan = plan_bundle(resolve_bundle(parse_bundle(bundle.root, id_root=bundle.id_root)), installed=installed_bundles)
        materialize(plan, build_here=frozenset(w.id for w in plan.writes if isinstance(w.source, BuiltImage)))
        yield Env.get(f"{bundle.id_root}/desktop")
    reset_artifact_store()
    reset_config()
    patch.undo()


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=True, capture_output=True)
