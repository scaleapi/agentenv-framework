"""Whether an installed plugin's requirements admit the agent-env installed next to it.

Internal. Reads installed metadata only,
so it runs before a plugin is imported, and for ``inventory(load=False)``.
"""

from __future__ import annotations

import functools
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

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
        if have is None:
            continue
        try:
            met = requirement.specifier.contains(have, prereleases=True)
        except Exception:
            continue
        if not met:
            unmet.append(f"{name}{requirement.specifier} (installed: {have})")
    return f"needs {'; '.join(unmet)}" if unmet else None
