"""Unit tests for E2B's deterministic size-specific template resolver."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import ClassVar

import pytest

from agent_env.providers.e2b.template import (
    E2BTemplateResolver,
    normalize_e2b_cpu,
    normalize_e2b_memory_mb,
)


class _FakeE2B:
    existing: ClassVar[set[str]] = set()
    exists_calls: ClassVar[list[str]] = []
    builds: ClassVar[list[tuple[object, str, int, int]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.existing = set()
        cls.exists_calls = []
        cls.builds = []

    @classmethod
    async def exists(cls, name: str) -> bool:
        cls.exists_calls.append(name)
        return name in cls.existing

    @classmethod
    async def build(
        cls,
        template: object,
        name: str,
        *,
        cpu_count: int,
        memory_mb: int,
        on_build_logs=None,
    ) -> None:
        cls.builds.append((template, name, cpu_count, memory_mb))
        if on_build_logs:
            on_build_logs(SimpleNamespace(level="info", message="fake build"))
        cls.existing.add(name)


class _CredentialRecordingE2B(_FakeE2B):
    api_keys: ClassVar[list[str | None]] = []

    @classmethod
    async def exists(cls, name: str, *, api_key: str | None = None) -> bool:
        cls.api_keys.append(api_key)
        return await super().exists(name)

    @classmethod
    async def build(cls, *args, api_key: str | None = None, **kwargs) -> None:
        cls.api_keys.append(api_key)
        await super().build(*args, **kwargs)


class _FakeTemplate:
    created: ClassVar[list[_FakeTemplate]] = []

    def __init__(self) -> None:
        self.base_template: str | None = None
        type(self).created.append(self)

    def from_template(self, base_template: str) -> _FakeTemplate:
        self.base_template = base_template
        return self


@pytest.fixture(autouse=True)
def _reset_fake_sdk() -> None:
    _FakeE2B.reset()
    _CredentialRecordingE2B.api_keys = []
    _FakeTemplate.created = []


def _resolver() -> E2BTemplateResolver:
    return E2BTemplateResolver(
        async_template_cls=_FakeE2B,
        template_cls=_FakeTemplate,
    )


@pytest.mark.asyncio
async def test_resolve_reuses_existing_deterministic_template() -> None:
    _FakeE2B.existing.add("base-1c-2048m")

    name = await _resolver().resolve("base", cpu=1.0, memory_mb=2048)

    assert name == "base-1c-2048m"
    assert _FakeE2B.exists_calls == ["base-1c-2048m"]
    assert _FakeTemplate.created == []
    assert _FakeE2B.builds == []


@pytest.mark.asyncio
async def test_resolve_builds_missing_template_from_base_with_requested_resources() -> None:
    name = await _resolver().resolve("base", cpu=3.0, memory_mb=6000)

    assert name == "base-3c-6000m"
    assert len(_FakeTemplate.created) == 1
    assert _FakeTemplate.created[0].base_template == "base"
    assert _FakeE2B.builds == [(_FakeTemplate.created[0], name, 3, 6000)]


@pytest.mark.asyncio
async def test_resolve_forwards_configured_api_key_to_template_calls() -> None:
    resolver = E2BTemplateResolver(
        api_key="resolved-secret",
        async_template_cls=_CredentialRecordingE2B,
        template_cls=_FakeTemplate,
    )

    await resolver.resolve("base", cpu=2, memory_mb=4096)

    assert _CredentialRecordingE2B.api_keys == ["resolved-secret", "resolved-secret"]


@pytest.mark.asyncio
async def test_resolve_single_flight_builds_once_for_concurrent_requests() -> None:
    build_started = asyncio.Event()
    allow_build = asyncio.Event()

    class _SlowFakeE2B(_FakeE2B):
        @classmethod
        async def build(cls, *args, **kwargs) -> None:
            cls.builds.append((args[0], args[1], kwargs["cpu_count"], kwargs["memory_mb"]))
            build_started.set()
            await allow_build.wait()
            cls.existing.add(args[1])

    resolver = E2BTemplateResolver(
        async_template_cls=_SlowFakeE2B,
        template_cls=_FakeTemplate,
    )
    first = asyncio.create_task(resolver.resolve("base", cpu=1, memory_mb=2048))
    await build_started.wait()
    second = asyncio.create_task(resolver.resolve("base", cpu=1.0, memory_mb=2048))
    await asyncio.sleep(0)
    allow_build.set()

    assert await asyncio.gather(first, second) == ["base-1c-2048m", "base-1c-2048m"]
    assert len(_SlowFakeE2B.builds) == 1


@pytest.mark.asyncio
async def test_resolve_accepts_template_created_by_another_worker_during_build() -> None:
    class _RaceWinnerFakeE2B(_FakeE2B):
        @classmethod
        async def build(cls, *args, **kwargs) -> None:
            cls.builds.append((args[0], args[1], kwargs["cpu_count"], kwargs["memory_mb"]))
            cls.existing.add(args[1])
            raise RuntimeError("template alias already exists")

    name = await E2BTemplateResolver(
        async_template_cls=_RaceWinnerFakeE2B,
        template_cls=_FakeTemplate,
    ).resolve("base", cpu=1, memory_mb=2048)

    assert name == "base-1c-2048m"
    assert _RaceWinnerFakeE2B.exists_calls == [name, name]
    assert len(_RaceWinnerFakeE2B.builds) == 1


@pytest.mark.asyncio
async def test_build_is_bounded_and_checks_for_a_race_after_timeout() -> None:
    class _HungFakeE2B(_FakeE2B):
        @classmethod
        async def build(cls, *args, **kwargs) -> None:
            await asyncio.Event().wait()

    resolver = E2BTemplateResolver(
        async_template_cls=_HungFakeE2B,
        template_cls=_FakeTemplate,
        build_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await resolver.resolve("timeout-base", cpu=1, memory_mb=2048)

    assert _HungFakeE2B.exists_calls == ["timeout-base-1c-2048m"] * 2


def test_single_flight_lock_is_safe_across_event_loops() -> None:
    class _ThreadedFakeE2B(_FakeE2B):
        @classmethod
        async def build(cls, *args, **kwargs) -> None:
            cls.builds.append((args[0], args[1], kwargs["cpu_count"], kwargs["memory_mb"]))
            await asyncio.sleep(0.02)
            cls.existing.add(args[1])

    resolver = E2BTemplateResolver(
        async_template_cls=_ThreadedFakeE2B,
        template_cls=_FakeTemplate,
    )

    def resolve() -> str:
        return asyncio.run(resolver.resolve("threaded-base", cpu=2, memory_mb=4096))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(), range(2)))

    assert results == ["threaded-base-2c-4096m"] * 2
    assert len(_ThreadedFakeE2B.builds) == 1


@pytest.mark.parametrize("cpu", [1, 2, 3, 9])
def test_cpu_requests_preserve_exact_integer_value(cpu: int) -> None:
    assert normalize_e2b_cpu(cpu) == cpu


@pytest.mark.parametrize("memory_mb", [1, 2048, 6000, 32769])
def test_memory_requests_preserve_exact_integer_value(memory_mb: int) -> None:
    assert normalize_e2b_memory_mb(memory_mb) == memory_mb


@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), float("nan"), True, "1"])
def test_normalize_e2b_cpu_rejects_non_positive_or_fractional_values(value: object) -> None:
    with pytest.raises(ValueError, match="cpu must be a positive integer"):
        normalize_e2b_cpu(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1024, 1.5, float("inf"), float("nan"), True, "2048"])
def test_normalize_e2b_memory_rejects_non_positive_or_fractional_values(value: object) -> None:
    with pytest.raises(ValueError, match="memory_mb must be a positive integer"):
        normalize_e2b_memory_mb(value)  # type: ignore[arg-type]
