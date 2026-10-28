"""Shared pytest fixtures for all tests."""

import os
import shutil
import signal
import sys
import tempfile

import pytest

from agent_env.config import reset_config
from agent_env.store.document_store import document_store as document_store_module
from agent_env.store.object_store.local import grant_server as grant_server_module
from agent_env.store.routing import disable_namespace_routing
from tst.util import shards


def _reset_store_singletons() -> None:
    """Drop every cached store singleton (``reset_*_store``) so the next access rebuilds it
    under the current config rather than the one a previous module ran with."""
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("agent_env."):
            continue
        for name, value in list(vars(module).items()):
            if name.startswith("reset_") and name.endswith("_store") and callable(value):
                value()


_STATE_DIR = pytest.StashKey[str]()
_state_home = pytest.MonkeyPatch()


def pytest_configure(config):
    """Point the per-user state root at a fresh directory for the run, so nothing reads or writes
    the real one. Set here rather than in a fixture because collection already reads the stores
    (the capability checks behind skip marks)."""
    config.stash[_STATE_DIR] = tempfile.mkdtemp(prefix="agent-env-state-")
    _state_home.setenv("XDG_STATE_HOME", config.stash[_STATE_DIR])


def pytest_unconfigure(config):
    _state_home.undo()
    shutil.rmtree(config.stash[_STATE_DIR], ignore_errors=True)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """With ``AGENTENV_TEST_SHARD=i/N``, run only the i-th of N groups of the selected tests (``tst/util/shards.py``).
    Last, so it splits what ``-m`` and the other filters left."""
    value = os.environ.get(shards.ENV_VAR)
    if not value:
        return
    try:
        index, count = shards.parse(value)
    except ValueError as e:
        raise pytest.UsageError(str(e)) from None
    table = shards.load_table()
    marked = [(item, item.get_closest_marker("dependency")) for item in items]
    linked = [(item.nodeid, mark.kwargs.get("name") or item.name, list(mark.kwargs.get("depends") or []))
              for item, mark in marked if mark is not None]
    together = shards.dependency_groups(linked, table)
    keep = set(shards.select([item.nodeid for item in items], index, count, table, together))
    deselected = [item for item in items if item.nodeid not in keep]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item.nodeid in keep]


@pytest.fixture
def sigint_handled():
    """SIGINT handled as Python handles it by default, for a test that sends it: a CI job started in the background
    runs with it ignored, and agent-env keeps an ignore it inherits."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.fixture(autouse=True)
def entity_collections_restored(monkeypatch):
    """A VersionedEntityStore registers its collection for the whole process; one a test builds on a
    scratch collection must not make that collection an entity collection for the next test."""
    monkeypatch.setattr(document_store_module, "_entity_collections", set(document_store_module._entity_collections))


@pytest.fixture(autouse=True)
def namespace_routing_off():
    """The CLI turns namespace routing on for its whole process; a test that invokes it must not
    leave it on for the next test."""
    yield
    disable_namespace_routing()


@pytest.fixture(autouse=True)
def fresh_config_document():
    """Re-resolve the config document for every test.

    The document is resolved once per process, so a test pointing AGENT_ENV_CONFIG at its own
    file would otherwise read whatever a previous test in the same module resolved. Resetting
    the whole config rather than only the document keeps the Config, its stores and the five
    registries on the same file as the document — they are all built from it, and the ones
    built before a test re-pointed would otherwise disagree with the ones built after. The
    store-singleton sweep below stays module-scoped because it walks sys.modules.
    """
    reset_config()
    yield
    reset_config()


@pytest.fixture(scope="module", autouse=True)
def setup_config():
    """Per-module store-singleton sweep; the config itself is reset per test above. The grant servers close too:
    one a module started under its own state root presents that root's CA, which the next module's containers
    don't trust, and the next grant starts a server afresh."""
    _reset_store_singletons()
    yield
    reset_config()
    _reset_store_singletons()
    for server in list(grant_server_module._servers.values()):
        server.close()
