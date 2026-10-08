"""Disable real network access for the unit test suite.

Unit tests must not touch the network: anything that does is either missing a
mock or belongs in tst/integration. This autouse fixture (scoped to tst/unit
via this conftest's location) blocks IP sockets through pytest-socket. AF_UNIX
sockets are allowed (asyncio's self-pipe / local IPC), and subprocesses run in
their own interpreter so they are unaffected (e.g. generated CLI scripts).

A test that genuinely needs a socket can opt out with @pytest.mark.enable_socket.

Unit tests that touch AWS run against moto, which still makes botocore resolve credentials;
without any, botocore probes the instance-metadata endpoint and trips the socket guard. So
placeholder credentials are set for the session when the environment has none.
"""

import os

import pytest
from pytest_socket import disable_socket, enable_socket

from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, reset_config
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore
from agent_env.store.routing import enable_namespace_routing
from agent_env.utils.deprecation import reset_deprecation_state

_PLACEHOLDER_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": "us-west-2",
    "AWS_EC2_METADATA_DISABLED": "true",
}


@pytest.fixture(scope="session", autouse=True)
def _placeholder_aws_credentials():
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        for name, value in _PLACEHOLDER_AWS_ENV.items():
            os.environ.setdefault(name, value)


@pytest.fixture(autouse=True)
def _disable_network():
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()


@pytest.fixture(autouse=True)
def _docker_answers(monkeypatch):
    """A bundle run asks whether Docker answers before it deploys containers locally or builds an infra env; unit
    tests never reach a daemon, so it answers. Tests of the probe itself patch ``subprocess.run``. A local sandbox
    asks the engine how many CPUs it has; here it can't tell, so nothing is held below what was asked for."""
    monkeypatch.setattr("agent_env.bundle.preflight.docker_unreachable", lambda: None)
    monkeypatch.setattr("agent_env.env.bootstrap.docker_unreachable", lambda: None)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.local_sandbox._engine_cpus", lambda: None)


@pytest.fixture(autouse=True)
def deprecations_fire_again():
    """``warn_deprecated`` warns once per process; each test sees its own first use."""
    reset_deprecation_state()


@pytest.fixture(autouse=True)
def isolated_state_root(tmp_path_factory, monkeypatch):
    """A fresh per-user state root for every unit test, overriding the run's."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


@pytest.fixture
def cli_routing():
    """Namespace routing on, as the CLI runs, so @local ids are written to the @local namespace's store."""
    enable_namespace_routing()


@pytest.fixture
def local_stores(tmp_path, monkeypatch):
    """Local document and object stores under this test's state root, with no infra."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    configure()
    reset_artifact_store()  # rebind the cached store to this test's fresh local config
    cfg = get_config()
    assert isinstance(cfg.get_object_store(), LocalFilesystemObjectStore)
    assert isinstance(cfg.get_document_store(), LocalSqliteDocumentStore)
    yield cfg
    reset_config()
    reset_artifact_store()
