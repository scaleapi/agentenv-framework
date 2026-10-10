"""The gateway server module is container code. Importing the library must not load it, the
library wheel must not require its Postgres driver, and a copy of this directory vendored as a
top-level ``gateway`` package (how standalone delivery bundles ship it) must serve the server
class from ``gateway.gateway`` and the vocabulary from its root."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import agent_env
import agent_env.env.gateway as gateway_pkg
from agent_env.env.gateway import (
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,
    AGENT_ENV_SESSION_META_KEY,
    DEFAULT_ROLE,
    GATEWAY_TRAJECTORY_FILE,
    TOOL_DISABLE_ACTION,
    TOOL_ENABLE_ACTION,
    WILDCARD,
    GatewayMode,
)
from agent_env.env.gateway import gateway as server_module

PYPROJECT = Path(agent_env.__file__).resolve().parents[2] / "pyproject.toml"


def _run(code: str, cwd: Path | None = None) -> str:
    proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_importing_the_library_loads_neither_the_server_module_nor_its_driver():
    out = _run(
        "import sys, agent_env.env, agent_env.task_step.task_steps.deploy_env; "
        "print(sorted(m for m in sys.modules if m in ('psycopg2', 'agent_env.env.gateway.gateway')))"
    )
    assert out == "[]", out


def test_the_package_vocabulary_matches_the_server_module():
    assert GatewayMode is server_module.GatewayMode
    assert (AGENT_ENV_ROLE_HEADER, AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY, DEFAULT_ROLE, WILDCARD,
            TOOL_DISABLE_ACTION, TOOL_ENABLE_ACTION) == (
        server_module.AGENT_ENV_ROLE_HEADER,
        server_module.AGENT_ENV_ROLE_META_KEY,
        server_module.AGENT_ENV_SESSION_META_KEY,
        server_module.DEFAULT_ROLE,
        server_module.WILDCARD,
        server_module.TOOL_DISABLE_ACTION,
        server_module.TOOL_ENABLE_ACTION,
    )
    assert GATEWAY_TRAJECTORY_FILE == server_module.GATEWAY_TRAJECTORY_FILE
    # the literals a child with no agent_env dependency mirrors by hand
    assert (AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY) == ("agentenv.io/role", "agentenv.io/session")


def test_the_server_class_is_not_a_package_export():
    assert not hasattr(gateway_pkg, "Gateway")
    assert "Gateway" not in gateway_pkg.__all__


def test_a_vendored_copy_of_the_package_serves_the_bundle_imports(tmp_path):
    source = Path(gateway_pkg.__file__).parent
    vendored = tmp_path / "gateway"
    vendored.mkdir()
    for path in source.glob("*.py"):
        shutil.copy(path, vendored / path.name)
    out = _run(
        "from gateway import GatewayMode, InternalMCPServer; from gateway.gateway import Gateway; "
        "import gateway.gateway as server; print(Gateway is server.Gateway, GatewayMode is server.GatewayMode)",
        cwd=tmp_path,
    )
    assert out == "True True", out


def test_the_postgres_driver_is_not_a_runtime_dependency():
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    assert not [d for d in project["dependencies"] if d.lower().startswith("psycopg2")], project["dependencies"]
    assert any(d.lower().startswith("psycopg2") for d in project["optional-dependencies"]["dev"])
