"""Sandbox providers: the compute an environment or agent runs on."""

from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER, SANDBOX_MODE_VM,
    SandboxProvider,
    build_sandbox_provider,
    get_sandbox_provider, set_sandbox_provider, reset_sandbox_provider,
    get_env_sandbox_provider, set_env_sandbox_provider, reset_env_sandbox_provider,
    get_agent_sandbox_provider, set_agent_sandbox_provider, reset_agent_sandbox_provider,
)
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox, ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.e2b import E2BSandbox, E2BSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm import SailVmSandbox, SailVmSandboxProvider

__all__ = [
    "ChainedSandboxProvider", "Sandbox", "VmSandbox",
    "SANDBOX_MODE_CONTAINER", "SANDBOX_MODE_VM",
    "SandboxProvider", "LocalSandbox", "LocalSandboxProvider",
    "ModalSandbox", "ModalSandboxProvider",
    "ModalVmSandbox", "ModalVmSandboxProvider",
    "E2BSandbox", "E2BSandboxProvider",
    "SailVmSandbox", "SailVmSandboxProvider",
    "build_sandbox_provider",
    "get_sandbox_provider", "set_sandbox_provider", "reset_sandbox_provider",
    "get_env_sandbox_provider", "set_env_sandbox_provider", "reset_env_sandbox_provider",
    "get_agent_sandbox_provider", "set_agent_sandbox_provider", "reset_agent_sandbox_provider",
]
