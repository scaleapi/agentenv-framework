from __future__ import annotations

import asyncio
import logging

from agent_env.attribution import Attribution
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError, Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

_PROVISION_DEADLINE_SECONDS = 180


class ChainedSandboxProvider(SandboxProvider):
    def __init__(self, providers: list[SandboxProvider]):
        if not providers:
            raise ValueError("ChainedSandboxProvider requires at least one provider")
        self._providers = providers

    @property
    def providers(self) -> tuple[SandboxProvider, ...]:
        """The providers tried, in order."""
        return tuple(self._providers)

    @staticmethod
    def filter_sandbox_providers(
        providers: list[SandboxProvider], policy: NetworkPolicy | None
    ) -> list[SandboxProvider]:
        """``providers`` that can enforce ``policy``, so an incapable one is never called."""
        if policy is None or not policy.restricts_egress:
            return providers
        eligible: list[SandboxProvider] = []
        for p in providers:
            if p.supports_network_policy(policy):
                eligible.append(p)
            else:
                log = logger.warning if p is providers[0] else logger.info
                log(f"sandbox_provider_skipped provider={type(p).__name__} mode={policy.mode.value}")
        if not eligible:
            raise NetworkPolicyUnsupportedError(
                f"No sandbox backend can enforce network policy mode {policy.mode.value!r} "
                f"(providers: {[type(p).__name__ for p in providers]})"
            )
        return eligible

    def supports_network_policy(self, policy: NetworkPolicy) -> bool:
        return any(p.supports_network_policy(policy) for p in self._providers)

    async def create_sandbox(
        self,
        *,
        attribution: Attribution | None = None,
        **kwargs,
    ) -> Sandbox:
        sandbox_providers = self.filter_sandbox_providers(self._providers, kwargs.get("network_policy"))
        errors: list[tuple[str, Exception]] = []
        for p in sandbox_providers:
            name = type(p).__name__
            try:
                async with asyncio.timeout(_PROVISION_DEADLINE_SECONDS):
                    sandbox = await p.create_sandbox(attribution=attribution, **kwargs)
            except Exception as e:
                logger.warning(f"{name}.create_sandbox failed: {e}; trying next provider")
                errors.append((name, e))
                continue
            logger.info(
                f"chain_provider_attempt provider={name} status=success sandbox_type={sandbox.type} "
                f"network_policy={sandbox.network_policy.mode.value if sandbox.network_policy else 'unknown'}"
            )
            return sandbox
        detail = "; ".join(f"{n}: {e!r}" for n, e in errors)
        raise RuntimeError(f"All {len(sandbox_providers)} providers failed: {detail}")

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        errors: list[tuple[str, Exception]] = []
        for p in self._providers:
            name = type(p).__name__
            try:
                return await p.get_sandbox(sandbox_id)
            except NotImplementedError:
                continue
            except Exception as e:
                logger.warning(
                    f"{name}.get_sandbox({sandbox_id}) failed: {e}; trying next provider"
                )
                errors.append((name, e))
        detail = "; ".join(f"{n}: {e!r}" for n, e in errors)
        raise RuntimeError(
            f"All {len(self._providers)} providers failed get_sandbox({sandbox_id}): {detail}"
        )

    async def close(self) -> None:
        for p in self._providers:
            try:
                await p.close()
            except Exception as e:
                logger.warning(f"close() failed on {type(p).__name__}: {e}")
