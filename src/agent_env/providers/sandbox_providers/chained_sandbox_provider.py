from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent_env.attribution import Attribution
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError, Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

logger = logging.getLogger(__name__)

_PROVISION_DEADLINE_SECONDS = 180


class ChainedSandboxProvider(SandboxProvider):
    def __init__(self, providers: list[SandboxProvider]):
        if not providers:
            raise ValueError("ChainedSandboxProvider requires at least one provider")
        self._providers = providers
        self._images: dict[str, DockerImageArtifact] = {}

    @property
    def providers(self) -> tuple[SandboxProvider, ...]:
        """The providers tried, in order."""
        return tuple(self._providers)

    @property
    def links(self) -> tuple[SandboxProvider, ...]:
        return self.providers

    @staticmethod
    def filter_sandbox_providers(
        providers: list[SandboxProvider], policy: NetworkPolicy | None
    ) -> list[SandboxProvider]:
        """``providers`` that can enforce ``policy``, so an incapable one is never called."""
        if policy is None or not policy.restricts_egress:
            return providers
        return _without(
            providers,
            lambda p: None if p.supports_network_policy(policy) else f"can't enforce network policy mode {policy.mode.value!r}",
            lambda refused: NetworkPolicyUnsupportedError(
                f"No sandbox backend can enforce network policy mode {policy.mode.value!r} "
                f"(providers: {[type(p).__name__ for p in providers]})"),
        )

    def sandbox_problem(self, sandbox: Sandbox, image: DockerImageArtifact) -> str | None:
        """None: create_sandbox checks each provider's sandbox as it tries it, and gives only one that runs
        the image."""
        return None

    def supports_network_policy(self, policy: NetworkPolicy) -> bool:
        return any(p.supports_network_policy(policy) for p in self._providers)

    async def prepare_image(self, image: DockerImageArtifact, *, attribution: Attribution | None = None) -> None:
        """Remember ``image``, so each provider is prepared for it just before it's tried, and only that one."""
        self._images[image.image_name] = image

    async def create_sandbox(
        self,
        *,
        attribution: Attribution | None = None,
        **kwargs,
    ) -> Sandbox:
        return await self._create("create_sandbox", attribution=attribution, **kwargs)

    async def create_container(
        self,
        *,
        attribution: Attribution | None = None,
        **kwargs,
    ) -> Sandbox:
        return await self._create("create_container", attribution=attribution, **kwargs)

    async def _create(self, method: str, *, attribution: Attribution | None = None, **kwargs) -> Sandbox:
        sandbox_providers = self.filter_sandbox_providers(self._providers, kwargs.get("network_policy"))
        image = self._images.get(kwargs.get("image_name"))
        errors: list[tuple[str, Exception]] = []
        for p in sandbox_providers:
            name = type(p).__name__
            try:
                if image is not None:
                    # outside the deadline, which bounds creating the sandbox, not a build
                    await p.prepare_image(image, attribution=attribution)
                async with asyncio.timeout(_PROVISION_DEADLINE_SECONDS):
                    sandbox = await getattr(p, method)(attribution=attribution, **kwargs)
            except Exception as e:
                logger.warning(f"{name}.{method} failed: {e}; trying next provider")
                errors.append((name, e))
                continue
            # A VM refuses an image it can't get or run, and a container runs only what its provider's containers do;
            # the next provider's sandbox may run it.
            if image is not None and (problem := p.sandbox_problem(sandbox, image)):
                logger.warning(f"{problem}; trying next provider")
                errors.append((name, ValueError(problem)))
                try:
                    await sandbox.terminate()
                except Exception as e:
                    logger.warning(f"{name}: terminating sandbox {sandbox.sandbox_id} failed: {e}")
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


def skip_refusing(provider: SandboxProvider, problem: Callable[[SandboxProvider], str | None], *, what: str
                  ) -> SandboxProvider:
    """``provider``, less each of its links ``problem`` refuses, as a deploy of ``what`` runs on any link that takes it.
    ``problem`` says why a link refuses as a clause on its name ("which ..."); a ValueError naming each is raised when
    it refuses every link."""
    links = list(provider.links)
    running = _without(links, problem, lambda refused: ValueError(f"Can't deploy {what} " + "; nor ".join(
        f"with {type(link).__name__}, {reason}" for link, reason in refused)))
    return provider if len(running) == len(links) else ChainedSandboxProvider(running)


def _without(providers: list[SandboxProvider], problem: Callable[[SandboxProvider], str | None],
             refuse: Callable[[list[tuple[SandboxProvider, str]]], Exception]) -> list[SandboxProvider]:
    """``providers`` less each one ``problem`` refuses, skipped with a warning when it's the first, the one preferred,
    and noted otherwise; ``refuse(refusals)`` is raised when it refuses them all."""
    refused = [(p, reason) for p in providers if (reason := problem(p))]
    if refused and len(refused) == len(providers):
        raise refuse(refused)
    for p, reason in refused:
        log = logger.warning if p is providers[0] else logger.info
        log(f"sandbox_provider_skipped provider={type(p).__name__}: {reason}")
    return [p for p in providers if all(p is not refusing for refusing, _ in refused)]
