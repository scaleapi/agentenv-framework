"""Reading installed entry points: which distribution claims which name in each plugin group.

Internal: the public contract is ``agent_env.plugins``. Discovery reads installed metadata only
and imports no plugin code; ``_registration`` imports a group's plugins when its registry is built.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import EntryPoint, distributions, entry_points

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Plugin:
    """Where an entry-point registration came from."""

    name: str
    value: str
    dist: str
    version: str

    def __str__(self) -> str:
        source = f"{self.dist} {self.version}".strip() or "an unreadable distribution"
        return f"{self.name!r} from {source} ({self.value})"


def claims(group: str) -> dict[str, list[tuple[Plugin, EntryPoint]]]:
    """Every distinct claim per entry-point name in ``group``, in a deterministic order."""
    eps = sorted(entry_points(group=group), key=sort_key)
    found: dict[str, list[tuple[Plugin, EntryPoint]]] = {}
    for ep in eps:
        plugin = Plugin(ep.name, getattr(ep, "value", "") or "", dist_name(ep), dist_version(ep))
        seen = found.setdefault(ep.name, [])
        # The same distribution found twice on the path is one registration, not a conflict.
        if all((p.dist, p.value) != (plugin.dist, plugin.value) for p, _ in seen):
            seen.append((plugin, ep))
    return found


def discover(group: str) -> dict[str, list[tuple[Plugin, EntryPoint]]]:
    """``claims(group)``, or none with a warning when installed metadata cannot be read."""
    try:
        return claims(group)
    except Exception:
        logger.warning("Entry-point discovery for %s failed; no plugins loaded", group, exc_info=True)
        return {}


def discovery_error(exc: Exception) -> str:
    """``exc`` from reading installed entry points, naming the distributions that cannot be read."""
    unreadable = []
    with suppress(Exception):
        for dist in distributions():
            try:
                _ = dist.entry_points  # parses entry_points.txt
            except Exception:
                unreadable.append(str(getattr(dist, "_path", None) or "a distribution with no path"))
    where = f" (unreadable entry points: {', '.join(unreadable)})" if unreadable else ""
    return f"{type(exc).__name__}: {exc}{where}"


def sort_key(ep: EntryPoint) -> tuple[str, str, str]:
    # ep.value tie-breaks even when both colliding dists have unreadable metadata.
    return (ep.name, dist_name(ep), getattr(ep, "value", "") or "")


def dist_name(ep: EntryPoint) -> str:
    # Tolerates corrupt/missing distribution metadata — the name is only a label.
    try:
        return getattr(getattr(ep, "dist", None), "name", "") or ""
    except Exception:
        return ""


def dist_version(ep: EntryPoint) -> str:
    try:
        return getattr(getattr(ep, "dist", None), "version", "") or ""
    except Exception:
        return ""
