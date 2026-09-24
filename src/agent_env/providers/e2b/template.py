"""Deterministic E2B template aliases for requested sandbox sizing.

E2B fixes a sandbox's CPU and memory at template-build time.  This resolver
derives a template from a configured base template and gives that derivative a
stable alias, so equivalent requests reuse the same template.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from numbers import Real
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

_DEFAULT_BUILD_TIMEOUT_SECONDS = 15 * 60
_LOCK_POLL_INTERVAL_SECONDS = 0.05


def _normalize_positive_integer(value: Real, *, field: str) -> int:
    """Return an integral positive resource value, rejecting ambiguous sizes."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        or int(value) != value
    ):
        raise ValueError(f"{field} must be a positive integer; got {value!r}")
    return int(value)


def normalize_e2b_cpu(cpu: float) -> int:
    """Validate an exact CPU request accepted by E2B's integer API."""
    return _normalize_positive_integer(cpu, field="cpu")


def normalize_e2b_memory_mb(memory_mb: int) -> int:
    """Validate an exact memory request accepted by E2B's integer API."""
    return _normalize_positive_integer(memory_mb, field="memory_mb")


class E2BTemplateResolver:
    """Resolve a size-specific E2B template, building it at most once locally.

    Locks are shared by all resolver instances because providers may be
    constructed more than once in the same process.  E2B itself remains the
    authority across processes: a failed build is accepted if a follow-up
    existence check shows another worker created the same alias.
    """

    _locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        api_key: str | None = None,
        async_template_cls: type[Any] | None = None,
        template_cls: type[Any] | None = None,
        build_timeout_seconds: float = _DEFAULT_BUILD_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._async_template_cls = async_template_cls
        self._template_cls = template_cls
        if build_timeout_seconds <= 0:
            raise ValueError("build_timeout_seconds must be positive")
        self._build_timeout_seconds = build_timeout_seconds

    @classmethod
    def _lock_for(cls, name: str) -> threading.Lock:
        # A threading lock is safe when separate worker threads each run their
        # own event loop. ``resolve`` acquires it with non-blocking polls so it
        # neither blocks the event loop nor leaks an acquired lock if cancelled.
        with cls._locks_guard:
            return cls._locks.setdefault(name, threading.Lock())

    @staticmethod
    async def _acquire_lock(lock: threading.Lock) -> None:
        while not lock.acquire(blocking=False):
            await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _log_build(name: str, entry: Any) -> None:
        level = getattr(entry, "level", "info")
        message = getattr(entry, "message", str(entry))
        log = logger.warning if level in {"warn", "error"} else logger.info
        log("E2B template build %s: %s", name, message)

    def _sdk(self) -> tuple[type[Any], type[Any]]:
        if self._async_template_cls is None or self._template_cls is None:
            # Defer SDK initialization until an E2B backend is actually used.
            from e2b import AsyncTemplate, Template

            return AsyncTemplate, Template
        return self._async_template_cls, self._template_cls

    async def resolve(self, base_template: str, *, cpu: float, memory_mb: int) -> str:
        """Return the E2B alias for ``base_template`` at the requested size."""
        exact_cpu = normalize_e2b_cpu(cpu)
        exact_memory_mb = normalize_e2b_memory_mb(memory_mb)
        name = f"{base_template}-{exact_cpu}c-{exact_memory_mb}m"
        api_params = {"api_key": self._api_key} if self._api_key is not None else {}

        async_template_cls, template_cls = self._sdk()
        lock = self._lock_for(name)
        await self._acquire_lock(lock)
        try:
            if await async_template_cls.exists(name, **api_params):
                return name

            template = template_cls().from_template(base_template)
            logger.info(
                "Building E2B template %s from %s (cpu=%s, memory=%sMB, timeout=%ss)",
                name,
                base_template,
                exact_cpu,
                exact_memory_mb,
                self._build_timeout_seconds,
            )
            try:
                async with asyncio.timeout(self._build_timeout_seconds):
                    await async_template_cls.build(
                        template,
                        name,
                        cpu_count=exact_cpu,
                        memory_mb=exact_memory_mb,
                        on_build_logs=lambda entry: self._log_build(name, entry),
                        **api_params,
                    )
            except Exception:
                # A separate process may have created the deterministic alias
                # after our initial exists check.  In that case its template is
                # exactly the resource variant we needed, so reuse it.
                if await async_template_cls.exists(name, **api_params):
                    return name
                raise
            return name
        finally:
            lock.release()


async def resolve_e2b_template(
    base_template: str,
    *,
    api_key: str | None = None,
    cpu: float,
    memory_mb: int,
) -> str:
    """Resolve a size-specific E2B template using the default SDK classes."""
    return await E2BTemplateResolver(api_key=api_key).resolve(
        base_template,
        cpu=cpu,
        memory_mb=memory_mb,
    )
