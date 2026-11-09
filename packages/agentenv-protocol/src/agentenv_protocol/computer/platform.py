"""The platform a driver runs on, and the predicates (``os``, ``distribution``, ``version``, ``arch``, ``display``)
that choose one method per operation for it."""
from __future__ import annotations

import os
import platform as _platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, TypeVar

PREDICATE_KEYS = ("os", "distribution", "version", "arch", "display")

_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
_Entry = TypeVar("_Entry")
#: The platform (``platform_key``) the computer's composer built an app for, which its server and install steps read in
#: place of what /etc/os-release says, so they act on the install the image holds; unset outside a composed computer.
PLATFORM_ENV = "AGENTENV_PLATFORM"
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
    """The running platform: the OS, ``/etc/os-release``'s ID and VERSION_ID on Linux and the macOS version on macOS,
    Windows' marketing version (``11`` from build 22000) on Windows, the machine, and the display: on Linux from
    ``XDG_SESSION_TYPE``, ``none`` for a computer with no display, else ``DISPLAY`` (x11) or ``WAYLAND_DISPLAY``
    (wayland); on macOS Aqua; on Windows the Desktop Window Manager (dwm)."""
    environ = os.environ if environ is None else environ
    name = _OS.get(sys.platform, sys.platform)
    release = _os_release(os_release) if name == "linux" else {}
    if name == "macos":
        release = {"VERSION_ID": _platform.mac_ver()[0] or None}
    if name == "windows":
        release = {"VERSION_ID": "11" if sys.getwindowsversion().build >= 22000 else "10"}
    display = {"macos": "aqua", "windows": "dwm"}.get(name)
    if name == "linux":
        session = (environ.get("XDG_SESSION_TYPE") or "").strip().lower()
        if session in ("x11", "wayland", "none"):
            display = session
        elif environ.get("DISPLAY"):
            display = "x11"
        elif environ.get("WAYLAND_DISPLAY"):
            display = "wayland"
    machine = _platform.machine().lower()
    return Platform(os=name, distribution=release.get("ID"), version=release.get("VERSION_ID"),
                    arch=_ARCH.get(machine, machine or None), display=display)


def platform_key(os_type: Optional[str], distribution: Optional[str] = None) -> Optional[str]:
    """What a platform's own install, launch or known failures are keyed by: ``<os>/<distribution>``, its distribution
    being /etc/os-release's ID, or its OS type when it has none."""
    if os_type is None:
        return None
    return f"{os_type}/{distribution}" if distribution else os_type


def entry_for(by_platform: Mapping[str, _Entry], key: Optional[str]) -> Optional[_Entry]:
    """``by_platform``'s entry for the platform ``key`` names: its distribution's, which replaces its OS type's there,
    else its OS type's."""
    if key is None:
        return None
    return by_platform[key] if key in by_platform else by_platform.get(key.partition("/")[0])


def applies(keys: Optional[Iterable[str]], key: Optional[str]) -> bool:
    """Whether what's declared for ``keys``, OS types or distributions, or for every OS when None, applies on the
    platform ``key`` names: one names its OS type or its distribution."""
    if keys is None:
        return True
    return key is not None and (key in keys or key.partition("/")[0] in keys)


def supported(entry: dict) -> list[Platform]:
    """The platforms a ``supports`` entry names: its ``os`` type, with each of its displays and arches, and its
    distribution and version when it names one. ``{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]}``
    names one platform."""
    os_ = entry.get("os") or {}
    if not isinstance(os_, dict) or not isinstance(os_.get("type"), str):
        raise ValueError(f"a supports entry names its os as {{'type': ...}}, got {entry!r}")
    return [Platform(os=os_["type"], distribution=distribution, version=version, arch=arch, display=display)
            for distribution in _values(os_.get("distribution")) for version in _values(os_.get("version"))
            for display in _values(os_.get("display")) for arch in _values(entry.get("arch"))]


def _values(value) -> list:
    return [None] if value is None else [value] if isinstance(value, str) else list(value)


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
