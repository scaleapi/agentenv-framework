"""The cpu and memory a deploy step reserves for whatever size it leaves unset.

A step's own value always wins. For each one it leaves unset, the sizer config.toml's ``[sandbox.sizer]`` names
(``impl`` and ``config``, as for ``[runner]``) may choose one; otherwise the step's default applies, or, with none, the
provider's. Where each value came from is added to the sandbox's attribution, so it reaches the provider with the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from agent_env.attribution import Attribution
from agent_env.config import loader, runtime

if TYPE_CHECKING:
    from agent_env.config.runtime import Config

CPU_SOURCE_KEY = "cpu_source"
MEMORY_SOURCE_KEY = "memory_source"
Source = Literal["step", "sizer", "default"]


@dataclass(frozen=True)
class SandboxSize:
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None


class SandboxSizer:
    """Chooses a size for what a deploy step leaves unset. This base chooses nothing, so the step's or provider's
    default applies; a subclass returns a value for each field it sizes and None for the rest."""

    @classmethod
    def from_config(cls, **config: Any) -> SandboxSizer:
        return cls(**config)

    def size(self, step: Any, attribution: Attribution) -> SandboxSize:
        return SandboxSize()


def _build_sizer(source: Config) -> SandboxSizer:
    section = dict(source.section("sandbox", "sizer"))
    if not section:
        return SandboxSizer()
    return loader.build_store(section, SandboxSizer, secret_resolver=source._resolve_secret)


def _pick(own: Any, chosen: Any, default: Any) -> tuple[Any, Source]:
    if own is not None:
        return own, "step"
    if chosen is not None:
        return chosen, "sizer"
    return default, "default"


def resolve(step: Any, attribution: Attribution, default: SandboxSize = SandboxSize()) -> tuple[SandboxSize, Attribution]:
    """The size ``step`` reserves, from its ``cpu`` and ``memory_mb``, and ``attribution`` with where each came from."""
    own = SandboxSize(step.cpu, step.memory_mb)
    chosen = SandboxSize() if None not in (own.cpu, own.memory_mb) else runtime.get_config().sandbox_sizer().size(
        step, attribution)
    cpu, cpu_source = _pick(own.cpu, chosen.cpu, default.cpu)
    memory_mb, memory_source = _pick(own.memory_mb, chosen.memory_mb, default.memory_mb)
    return SandboxSize(cpu, memory_mb), {**attribution, CPU_SOURCE_KEY: cpu_source, MEMORY_SOURCE_KEY: memory_source}
