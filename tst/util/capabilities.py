"""Test-environment capabilities: what the resolved config and the test host can do.

Backends are config-selected, so only the resolved config can say whether a test's
backend exists; these probes let a profile without one skip the test instead of failing
inside the code under test. The gate is a collection-time ``skipif`` mark, not a
fixture, so a module-scoped fixture (a Docker build, say) never runs for a test that is
going to skip. Reasons are exactly ``agentenv-capability-missing: <name>`` so a CI job
can assert the skips taken equal the skips declared for its profile.
"""

from __future__ import annotations

import subprocess

import pytest

MISSING_CAPABILITY_PREFIX = "agentenv-capability-missing: "

#: Only the agent-driving paths need a model endpoint.
MODEL_ENDPOINT = "model_endpoint_configured"

#: The remote sandbox providers (``modal``, ``modal_vm``, ``e2b``) need credentials the resolved
#: config may not carry; the local default never does.
REMOTE_SANDBOX = "remote_sandbox"

#: The A2A agent a ``deploy_agent`` step without an id resolves to (``default_a2a_agent_id``); the
#: model-driven suites deploy it, so it must be registered in the configured store.
DEFAULT_A2A_AGENT = "default_a2a_agent"

#: The gateway's virtual-clock tests build real MCP servers from sources outside this repository,
#: found through ``AGENT_ENV_TEST_MCP_SERVERS_DIR`` (one ``<server>/Dockerfile`` per server).
MCP_SERVER_SOURCES = "mcp_server_sources"

#: ``collect_artifacts`` sizes a file with ``stat -c %s``, which only GNU stat takes (macOS ships BSD stat),
#: so collecting off a local sandbox needs GNU stat first on PATH.
GNU_STAT = "gnu_stat"


def missing_capability_reason(capability: str) -> str:
    return f"{MISSING_CAPABILITY_PREFIX}{capability}"


def model_endpoint_is_configured() -> bool:
    """Whether the resolved config gives the agent-driving paths a model endpoint.

    Runs at collection time, possibly against a config that reaches a secret store, so it
    must neither break collection nor turn a broken config into a silent skip: only the
    ``ConfigError`` for an absent endpoint answers False; any other failure answers True
    and lets the test fail on the real problem.
    """
    from agent_env.config import get_config
    from agent_env.config.errors import ConfigError

    # A malformed [model] section also raises ConfigError; parsing it first separates
    # "broken, let the test fail" from "absent, skip".
    try:
        get_config().get_default_model()
    except Exception:
        return True

    try:
        return bool(get_config().get_litellm_base_url())
    except ConfigError:
        return False
    except Exception:
        return True


def skip_without_model_endpoint() -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that need a real model endpoint; use in ``pytestmark`` or as a decorator."""
    return pytest.mark.skipif(
        not model_endpoint_is_configured(),
        reason=missing_capability_reason(MODEL_ENDPOINT),
    )


def remote_sandbox_is_available(provider: str) -> bool:
    """Whether the resolved config can build the ``modal`` / ``modal_vm`` / ``e2b`` sandbox
    provider, credentials included.

    Modal: any failure to resolve the credentials answers False. E2B: building the provider
    resolves ``[sandbox.providers.e2b.config]`` and its ``secret:`` references; only a
    ``ConfigError`` (absent or incomplete config) answers False, any other failure answers True and
    lets the test fail on the real problem, as ``model_endpoint_is_configured`` does."""
    if provider not in ("modal", "modal_vm", "e2b"):
        raise ValueError(f"unknown remote sandbox provider {provider!r}")
    if provider == "e2b":
        from agent_env.config.errors import ConfigError
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

        try:
            build_sandbox_provider("e2b")
        except ConfigError:
            return False
        except Exception:
            return True
        return True
    from agent_env.config import get_config

    try:
        get_config().get_modal_credentials()
    except Exception:
        return False
    return True


def skip_without_remote_sandbox(provider: str) -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that deploy on a remote sandbox provider."""
    return pytest.mark.skipif(
        not remote_sandbox_is_available(provider),
        reason=missing_capability_reason(REMOTE_SANDBOX),
    )


def default_a2a_agent_is_registered() -> bool:
    """Whether ``default_a2a_agent_id`` names an A2A agent in the configured store.

    Like ``model_endpoint_is_configured``: only ``NotFoundError`` answers False; any other
    failure answers True and lets the test fail on the real problem.
    """
    from agent_env.a2a_agent import A2AAgent
    from agent_env.config import get_config
    from agent_env.store.base import NotFoundError

    try:
        A2AAgent.get(get_config().get_default_a2a_agent_id())
    except NotFoundError:
        return False
    except Exception:
        return True
    return True


def skip_without_default_a2a_agent() -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that deploy the configured default A2A agent."""
    return pytest.mark.skipif(
        not default_a2a_agent_is_registered(),
        reason=missing_capability_reason(DEFAULT_A2A_AGENT),
    )


def gnu_stat_is_available() -> bool:
    """Whether ``stat -c %s`` prints a file's size on this host."""
    try:
        return subprocess.run(["stat", "-c", "%s", __file__], capture_output=True).returncode == 0
    except OSError:
        return False


def skip_without_gnu_stat() -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that collect artifacts off a local sandbox."""
    return pytest.mark.skipif(
        not gnu_stat_is_available(),
        reason=missing_capability_reason(GNU_STAT),
    )
