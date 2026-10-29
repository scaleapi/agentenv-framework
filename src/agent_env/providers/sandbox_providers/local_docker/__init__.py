"""The local Docker sandbox provider: this machine's Docker, and the relay a container closed off from every network is
reached through."""

from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox, LocalSandboxProvider

__all__ = ["LocalSandbox", "LocalSandboxProvider"]
