"""The platform a driver runs on, and the predicates (``os``, ``distribution``, ``version``, ``arch``, ``display``)
that choose one method per operation for it."""
from __future__ import annotations

import os
import platform as _platform
import sys
from dataclasses import dataclass
from pathlib import Path

PREDICATE_KEYS = ("os", "distribution", "version", "arch", "display")

_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
_OS = {"linux": "linux", "darwin": "macos", "win32": "windows"}


@dataclass(frozen=True)
class Platform:
    os: str | None = None
    distribution: str | None = None
    version: str | None = None
    arch: str | None = None
    display: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in PREDICATE_KEYS if getattr(self, k) is not None}


def detect(environ: dict | None = None, os_release: Path = Path("/etc/os-release")) -> Platform:
    """The running platform: the OS, ``/etc/os-release``'s ID and VERSION_ID on Linux, the machine, and the
    display from ``XDG_SESSION_TYPE``, else ``DISPLAY`` (x11) or ``WAYLAND_DISPLAY`` (wayland)."""
    environ = os.environ if environ is None else environ
    name = _OS.get(sys.platform, sys.platform)
    release = _os_release(os_release) if name == "linux" else {}
    display = None
    if name == "linux":
        session = (environ.get("XDG_SESSION_TYPE") or "").strip().lower()
        if session in ("x11", "wayland"):
            display = session
        elif environ.get("DISPLAY"):
            display = "x11"
        elif environ.get("WAYLAND_DISPLAY"):
            display = "wayland"
    machine = _platform.machine().lower()
    return Platform(os=name, distribution=release.get("ID"), version=release.get("VERSION_ID"),
                    arch=_ARCH.get(machine, machine or None), display=display)


def _os_release(path: Path) -> dict:
    try:
        text = path.read_text()
    except OSError:
        return {}
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip():
            fields[key.strip()] = value.strip().strip("\"'")
    return fields


def normalize(predicates: dict) -> dict[str, frozenset]:
    """Predicates as sets of allowed values; an unknown key is refused."""
    unknown = sorted(set(predicates) - set(PREDICATE_KEYS))
    if unknown:
        raise TypeError(f"unknown predicate(s) {unknown}; predicates are {list(PREDICATE_KEYS)}")
    return {k: frozenset([v] if isinstance(v, str) else v) for k, v in predicates.items()}


def matches(predicates: dict[str, frozenset], running: Platform) -> bool:
    return all(getattr(running, key) in allowed for key, allowed in predicates.items())


def overlap(a: dict[str, frozenset], b: dict[str, frozenset]) -> bool:
    """Whether some platform could satisfy both: no key both constrain to disjoint values."""
    return all(a[k] & b[k] for k in set(a) & set(b))
