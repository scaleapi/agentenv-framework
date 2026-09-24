"""Shared pytest fixtures for all tests."""

import sys

import pytest

from agent_env.config import reset_config


def _reset_store_singletons() -> None:
    """Drop every cached store singleton (``reset_*_store``) so the next access rebuilds it
    under the current config rather than the one a previous module ran with."""
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("agent_env."):
            continue
        for name, value in list(vars(module).items()):
            if name.startswith("reset_") and name.endswith("_store") and callable(value):
                value()


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
    """Per-module store-singleton sweep; the config itself is reset per test above."""
    _reset_store_singletons()
    yield
    reset_config()
    _reset_store_singletons()
