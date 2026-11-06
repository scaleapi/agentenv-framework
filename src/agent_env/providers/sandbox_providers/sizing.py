"""The cpu and memory a deploy step reserves for whatever size it leaves unset.

A step's own value always wins. For each one it leaves unset, the sizer config.toml's ``[sandbox.sizer]`` names
(``impl`` and ``config``, as for ``[runner]``, or the ``impl`` alone as a string) may choose one; otherwise the step's
default applies, or, with none, the provider's. Where each value came from is added to the sandbox's attribution, so
it reaches the provider with the rest.

The sizer is told whether the step may deploy onto a VM, whose memory is a hard limit on everything in it, so it can
decline to size one. A sizer that fails never fails the deploy: its error is logged and it is taken to choose nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from agent_env.attribution import CPU_SOURCE_KEY, MEMORY_SOURCE_KEY, Attribution
from agent_env.config import loader, runtime

if TYPE_CHECKING:
    from agent_env.config.runtime import Config

logger = logging.getLogger(__name__)

__all__ = ["CPU_SOURCE_KEY", "MEMORY_SOURCE_KEY", "SandboxSize", "SandboxSizer", "resolve", "with_default_memory"]

# ``provider``: the sandbox doesn't take the step's value at all (a gateway's containers share its cpu but not its
# memory), so it gets its provider's, and a size chosen for the step says nothing about it.
Source = Literal["step", "sizer", "default", "provider"]


@dataclass(frozen=True)
class SandboxSize:
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None


class SandboxSizer:
    """Chooses a size for what a deploy step leaves unset. This base chooses nothing, so the step's or provider's
    default applies; a subclass returns a value for each field it sizes and None for the rest.

    ``vm`` is True when the step may deploy onto a VM (its sandbox mode is ``vm``, or a backend its sandbox type names
    creates VMs), where the memory chosen is the limit on everything the VM runs."""

    @classmethod
    def from_config(cls, **config: Any) -> SandboxSizer:
        return cls(**config)

    def size(self, step: Any, attribution: Attribution, *, vm: bool) -> SandboxSize:
        return SandboxSize()


def _build_sizer(source: Config) -> SandboxSizer:
    raw = source.section("sandbox").get("sizer")
    section = {"impl": raw} if isinstance(raw, str) else dict(raw or {})
    if not section:
        return SandboxSizer()
    return loader.build_store(section, SandboxSizer, secret_resolver=source._resolve_secret)


def _pick(own: Any, chosen: Any, default: Any) -> tuple[Any, Source]:
    if own is not None:
        return own, "step"
    if chosen is not None:
        return chosen, "sizer"
    return default, "default"


def with_default_memory(attribution: Optional[Attribution]) -> Optional[Attribution]:
    """``attribution`` for a sandbox created without the step's memory, which gets its provider's: its memory source
    is ``provider`` whatever the step's was."""
    if not attribution or MEMORY_SOURCE_KEY not in attribution:
        return attribution
    return {**attribution, MEMORY_SOURCE_KEY: "provider"}


def _chosen(step: Any, attribution: Attribution, vm: bool) -> SandboxSize:
    sizer = runtime.get_config().sandbox_sizer()  # a misconfigured sizer is a config error, raised as one
    try:
        chosen = sizer.size(step, dict(attribution), vm=vm)
        if not isinstance(chosen, SandboxSize):
            raise TypeError(f"returned {type(chosen).__name__}, not a SandboxSize")
        return chosen
    except Exception as e:
        logger.warning(f"{type(sizer).__name__} could not size step {getattr(step, 'id', None)!r}, so it gets its "
                       f"default: {type(e).__name__}: {e}")
        return SandboxSize()


def resolve(step: Any, attribution: Attribution, default: SandboxSize = SandboxSize(), *,
            vm: bool) -> tuple[SandboxSize, Attribution]:
    """The size ``step`` reserves, from its ``cpu`` and ``memory_mb``, and ``attribution`` with where each came from.
    ``vm``: whether the step may deploy onto a VM, which the sizer is told."""
    own = SandboxSize(step.cpu, step.memory_mb)
    chosen = SandboxSize() if None not in (own.cpu, own.memory_mb) else _chosen(step, attribution, vm)
    cpu, cpu_source = _pick(own.cpu, chosen.cpu, default.cpu)
    memory_mb, memory_source = _pick(own.memory_mb, chosen.memory_mb, default.memory_mb)
    return SandboxSize(cpu, memory_mb), {**attribution, CPU_SOURCE_KEY: cpu_source, MEMORY_SOURCE_KEY: memory_source}
