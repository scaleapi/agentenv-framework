"""Whether an installed plugin's requirements admit the agent-env installed next to it, and which of a list of
requirements this environment doesn't meet.

Internal. Reads installed metadata only,
so it runs before a plugin is imported, and for ``inventory(load=False)``.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, distribution, version
from typing import Any, Optional

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

# What a plugin may constrain: agent-env, and the protocol package it pins exactly.
CORE = ("agentenv-framework", "agentenv-framework-protocol")


@functools.cache
def installed(name: str) -> Optional[str]:
    """The installed version of ``name``, or None without metadata, as for a source tree on sys.path."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def incompatibility(dist: Any) -> Optional[str]:
    """Why ``dist``'s requirements on agent-env exclude the installed versions, or None.

    Only unconditional requirements count: one under an extra, or whose marker is false here, does not."""
    try:
        requires = list(getattr(dist, "requires", None) or ())
    except Exception:
        return None  # the metadata is reported as unreadable where it is listed
    unmet = []
    for line in requires:
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        name = canonicalize_name(requirement.name)
        if name not in CORE or (requirement.marker is not None and not requirement.marker.evaluate({"extra": ""})):
            continue
        have = installed(name)
        if have is not None and not _allows(requirement, have):
            unmet.append(f"{name}{requirement.specifier} (installed: {have})")
    return f"needs {'; '.join(unmet)}" if unmet else None


def unmet(requires: Iterable[str]) -> list[tuple[str, str]]:
    """Each of ``requires`` that isn't met here, with why: its package isn't installed, is at a version it excludes, or
    is without what the extras it names need. One whose marker is false here doesn't apply."""
    found = []
    for line in requires:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        have = installed(canonicalize_name(requirement.name))
        if have is None:
            found.append((line, "which isn't installed"))
        elif not _allows(requirement, have):
            found.append((line, f"and {requirement.name} {have} is installed"))
        elif missing := _missing_for_extras(requirement, set()):
            found.append((line, f"and {requirement.name} {have} is installed without {', '.join(missing)}, which its "
                                f"extras need"))
    return found


def _allows(requirement: Requirement, version: str) -> bool:
    """Whether ``requirement`` admits the installed ``version``, prereleases included as pip check counts them; one pip
    can't compare it admits only when it names no versions."""
    try:
        return requirement.specifier.contains(version, prereleases=True)
    except InvalidVersion:
        return not requirement.specifier


def _missing_for_extras(requirement: Requirement, seen: set[tuple[str, str]]) -> list[str]:
    """What the extras ``requirement`` names need of its installed package that isn't installed, or not at a version
    it allows, following an extra that names another of the package's own."""
    name = canonicalize_name(requirement.name)
    missing = []
    for extra in sorted(requirement.extras):
        if (name, extra) in seen:
            continue
        seen.add((name, extra))
        try:
            lines = distribution(name).requires or ()
        except PackageNotFoundError:
            return []
        for line in lines:
            try:
                needed = Requirement(line)
            except InvalidRequirement:
                continue
            if needed.marker is None or not needed.marker.evaluate({"extra": extra}) \
                    or needed.marker.evaluate({"extra": ""}):
                continue
            have = installed(canonicalize_name(needed.name))
            if have is None or not _allows(needed, have):
                missing.append(f"{needed.name}{needed.specifier}")
            else:
                missing.extend(_missing_for_extras(needed, seen))
    return list(dict.fromkeys(missing))
