"""Test-environment capabilities: what the resolved config and the test host can do.

Backends are config-selected, so only the resolved config can say whether a test's
backend exists; these probes let a profile without one skip the test instead of failing
inside the code under test. The gate is a collection-time ``skipif`` mark, not a
fixture, so a module-scoped fixture (a Docker build, say) never runs for a test that is
going to skip. Reasons are exactly ``agentenv-capability-missing: <name>`` so a CI job
can assert the skips taken equal the skips declared for its profile.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

import pytest

MISSING_CAPABILITY_PREFIX = "agentenv-capability-missing: "

#: Only the agent-driving paths need a model endpoint.
MODEL_ENDPOINT = "model_endpoint_configured"

#: The remote sandbox providers (``modal``, ``modal_vm``, ``e2b``, ``sail_vm``, ``tensorlake``, ``tenki``,
#: ``vercel``, ``boxd``, ``coreweave``, ``daytona``, ``boat``, ``agent37``) need credentials the resolved config may not
#: carry; the local default never does.
REMOTE_SANDBOX = "remote_sandbox"

#: The A2A agent a ``deploy_agent`` step without an id resolves to (``default_a2a_agent_id``); the
#: model-driven suites deploy it, so it must be registered in the configured store.
DEFAULT_A2A_AGENT = "default_a2a_agent"

#: The gateway's virtual-clock tests build real MCP servers from sources outside this repository,
#: found through ``AGENT_ENV_TEST_MCP_SERVERS_DIR`` (one ``<server>/Dockerfile`` per server).
MCP_SERVER_SOURCES = "mcp_server_sources"

#: The TRL contract tier (``tst/trl``) trains with TRL on CPU; trl and torch aren't in the dev extra, and only that
#: tier's CI job installs them.
TRL = "trl_installed"

#: The NeMo Gym tier (``tst/nemo_gym``) runs the resources server in-process; nemo-gym needs Python 3.13.14 or later and
#: isn't in the dev extra, and only that tier's CI job installs it.
NEMO_GYM = "nemo_gym_installed"

#: The verifiers tier (``tst/verifiers``) runs ``vf-eval`` from verifiers' own venv, since verifiers pins mcp 2 and the
#: framework mcp 1; only that tier's CI job installs it and puts ``vf-eval`` on PATH.
VERIFIERS = "verifiers_installed"

#: The verl tier (``tst/verl``) runs ``AgentEnvAgentLoop`` on verl itself; verl and torch aren't in the dev extra, and
#: only that tier's CI job installs them.
VERL = "verl_installed"

#: The SkyRL tier (``tst/skyrl``) runs ``AgentEnvGenerator`` and ``AgentEnvTextEnv`` on SkyRL itself; SkyRL, skyrl-gym
#: and torch aren't in the dev extra, and only that tier's CI job installs them.
SKYRL = "skyrl_installed"

#: The Miles tier (``tst/miles``) plays rows through Miles' own rollout; Miles, its test fixtures and CPU torch aren't in
#: the dev extra, and only that tier's CI job puts a Miles checkout on PYTHONPATH and installs its requirements.
MILES = "miles_installed"

#: The slime tier (``tst/slime``) plays rows through slime's own OpenAI adapter with slime's own arguments; slime, its
#: SGLang and Megatron-LM and CPU torch aren't in the dev extra, and only that tier's CI job installs them.
SLIME = "slime_installed"

#: ``collect_artifacts`` sizes a file with ``stat -c %s``, which only GNU stat takes (macOS ships BSD stat),
#: so collecting off a local sandbox needs GNU stat first on PATH.
GNU_STAT = "gnu_stat"

#: Booting a computer's disk form on local Docker needs KVM: /dev/kvm on this host, which Docker passes to the launcher.
#: Docker Desktop on a Mac has none, nor has a host without nested virtualization.
KVM = "kvm"

#: Composing a Windows computer installs Windows from Microsoft's evaluation ISO, which a run fetches into this machine's
#: cache once, about 7 GB; a test that composes one needs it there already, and KVM, which boots windows/amd64 disks.
WINDOWS_MEDIA = "windows_media"

#: Composing an Omarchy computer builds its base from Omarchy's own VM image, which a run fetches into this machine's
#: cache once, about 4 GB; a test that composes one needs it there already, and KVM, which boots linux/amd64 disks.
OMARCHY_IMAGE = "omarchy_image"
#: The shipped Omarchy base, whose source the capability looks for.
OMARCHY_ARTIFACT = Path(__file__).parents[2] / "src" / "agent_env" / "computers" / "artifacts" / "omarchy-4.5" / "artifact.toml"


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
    """Whether the resolved config can build the ``modal`` / ``modal_vm`` / ``e2b`` / ``sail_vm`` /
    ``tensorlake`` / ``tenki`` / ``vercel`` / ``boxd`` / ``coreweave`` / ``daytona`` / ``boat`` / ``agent37`` sandbox provider, credentials included.

    Modal: any failure to resolve the credentials answers False. E2B, Sail, Tensorlake, Tenki, Vercel, boxd,
    CoreWeave, Daytona, boat and Agent37: building the provider resolves ``[sandbox.providers.<name>.config]`` and its ``secret:`` references;
    only a ``ConfigError`` (absent or incomplete config, or the provider's extra not installed) answers
    False, any other failure answers True and lets the test fail on the real problem, as
    ``model_endpoint_is_configured`` does. Vercel, which builds without credentials, also needs them in
    its config or the environment."""
    if provider not in ("modal", "modal_vm", "e2b", "sail_vm", "tensorlake", "tenki", "vercel", "boxd", "coreweave", "daytona", "boat", "agent37"):
        raise ValueError(f"unknown remote sandbox provider {provider!r}")
    if provider in ("e2b", "sail_vm", "tensorlake", "tenki", "vercel", "boxd", "coreweave", "daytona", "boat", "agent37"):
        from agent_env.config.errors import ConfigError
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

        try:
            build_sandbox_provider(provider)
        except ConfigError:
            return False
        except Exception:
            return True
        if provider != "vercel":
            return True
        from agent_env.config import get_config

        vercel = get_config().section("sandbox").get("providers", {}).get("vercel", {})
        config = vercel.get("config", {}) if isinstance(vercel, dict) else {}
        if isinstance(config, dict) and all(
            key in config for key in ("token", "team_id", "project_id")
        ):
            return True
        return bool(os.getenv("VERCEL_OIDC_TOKEN")) or all(
            os.getenv(name)
            for name in ("VERCEL_TOKEN", "VERCEL_TEAM_ID", "VERCEL_PROJECT_ID")
        )
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


def kvm_is_available() -> bool:
    """Whether the local sandbox provider boots disks here: this host's /dev/kvm, readable and writable."""
    from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider

    return bool(LocalSandboxProvider.disk_image_platforms())


def windows_media_is_available() -> bool:
    """Whether this machine boots Windows disks and holds Microsoft's evaluation ISO in its cache."""
    from agent_env.env.envs.computer.utils.windows_disk import media_path
    from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider

    return "windows/amd64" in LocalSandboxProvider.disk_image_platforms() and media_path().exists()


def omarchy_image_is_available() -> bool:
    """Whether this machine boots Omarchy's disks and holds the image the shipped Omarchy base is built from."""
    from agent_env.env.envs.computer.utils.omarchy_disk import source_path
    from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider

    artifact = OMARCHY_ARTIFACT.read_text()
    sha256 = tomllib.loads(artifact)["forms"]["disk"]["source"]["sha256"]
    return "linux/amd64" in LocalSandboxProvider.disk_image_platforms() and source_path(sha256).exists()


def skip_without_omarchy_image() -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that compose an Omarchy computer on local Docker."""
    return pytest.mark.skipif(not omarchy_image_is_available(), reason=missing_capability_reason(OMARCHY_IMAGE))


def skip_without_windows_media() -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that compose a Windows computer on local Docker."""
    return pytest.mark.skipif(not windows_media_is_available(), reason=missing_capability_reason(WINDOWS_MEDIA))


def skip_without_kvm() -> pytest.MarkDecorator:
    """Collection-time ``skipif`` for tests that boot a computer's disk form on local Docker."""
    return pytest.mark.skipif(not kvm_is_available(), reason=missing_capability_reason(KVM))
