"""Bundles installed with a package, found through the ``agent_env.bundles`` entry points (``agent_env.plugins._bundles``
finds them), and run by name: a name several distributions install is run as ``<dist>/<name>``, except that agent-env's
own bundles keep their bare names.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Callable
from pathlib import Path

from packaging.utils import canonicalize_name

from agent_env.plugins._bundles import BUNDLES, CORE, InstalledBundle, installed_bundles

from .materialize import materialize
from .parse import Bundle, BundleError, BundleKind, parse_bundle
from .plan import check_bundle, plan_entities
from .resolve import resolve_bundle

__all__ = ["BUNDLES", "CORE", "InstalledBundle", "checked", "find_bundle", "installed_bundles", "run_name",
           "write_installed"]


def locate_bundle(
    text: str, installed: Callable[[], tuple[InstalledBundle, ...]] = installed_bundles,
) -> tuple[Path, str | None]:
    """The folder ``text`` names, and the id root of an installed bundle's, as ``agent-env run BUNDLE`` reads it: an
    existing folder wins, and text that starts like a path is always one; anything else names a bundle ``installed``
    lists. Raises BundleError."""
    if path_like(text) or os.path.isdir(text):
        return Path(text), None
    found = find_bundle(text, installed())
    return found.root, found.id_root


def path_like(text: str) -> bool:
    """Text that names a folder even when there is none there."""
    return text.startswith((".", "/", "~"))


def find_bundle(text: str, bundles: tuple[InstalledBundle, ...] | None = None) -> InstalledBundle:
    """The installed bundle ``text`` names, as ``<name>`` or ``<dist>/<name>``. Raises BundleError."""
    bundles = installed_bundles() if bundles is None else bundles
    dist, _, name = text.rpartition("/")
    matches = [bundle for bundle in bundles if bundle.name == name and (not dist or bundle.dist == canonicalize_name(dist))]
    if not dist and len({bundle.dist for bundle in matches}) > 1:
        matches = [bundle for bundle in matches if bundle.dist == CORE]
        if not matches:
            names = ", ".join(sorted(repr(bundle.qualified) for bundle in bundles if bundle.name == name))
            raise BundleError([f"{text!r}: several packages install a bundle of that name; run one of {names}"])
    if not matches:
        raise BundleError([_unknown(text, bundles)])
    bundle = matches[0]
    if bundle.problem:
        raise BundleError([f"{bundle.qualified}: {bundle.problem}"])
    return bundle


def run_name(bundle: InstalledBundle, bundles: tuple[InstalledBundle, ...]) -> str:
    """The name ``bundle`` runs by: bare, unless another distribution installs that name and ``bundle`` isn't
    agent-env's own."""
    shared = any(other.name == bundle.name and other.dist != bundle.dist for other in bundles)
    return bundle.qualified if shared and bundle.dist != CORE else bundle.name


def checked(bundle: InstalledBundle, *, load: bool = True) -> Bundle:
    """``bundle``'s folder, parsed and, with ``load``, checked as ``agent-env run`` checks it before it reads a
    store. Raises BundleError."""
    parsed = parse_bundle(bundle.root, id_root=bundle.id_root)
    if load:
        check_bundle(resolve_bundle(parsed), installed=installed_bundles)
    return parsed


def write_installed(entity_id: str) -> int:
    """Write ``entity_id``, an env, agent or artifact an installed bundle defines, with what it references, as a run of
    a bundle naming it does, and return its version: the one already written when nothing it's made from changed.
    Needs namespace routing. Raises BundleError."""
    bundles = installed_bundles()
    bundle = next((bundle for bundle in bundles if entity_id.startswith(f"{bundle.id_root}/")), None)
    if bundle is None:
        raise BundleError([f"{entity_id!r} isn't an entity of an installed bundle"])
    if bundle.problem:
        raise BundleError([f"{bundle.qualified}: {bundle.problem}"])
    plan = plan_entities(resolve_bundle(parse_bundle(bundle.root, id_root=bundle.id_root)), [entity_id],
                         installed=lambda: bundles)
    [write] = [write for write in plan.writes
               if write.id == entity_id and write.kind not in (BundleKind.TASK, BundleKind.EVAL)]
    return materialize(plan).version_of(write.kind.store, entity_id)


def _unknown(text: str, bundles: tuple[InstalledBundle, ...]) -> str:
    if not bundles:
        return f"{text!r} is neither a folder nor an installed bundle, and no bundles are installed"
    names = sorted({run_name(bundle, bundles) for bundle in bundles})
    close = difflib.get_close_matches(text, names, n=3)
    hint = f"; did you mean {' or '.join(repr(name) for name in close)}?" if close else ""
    return f"{text!r} is neither a folder nor an installed bundle{hint} (agent-env run lists the installed bundles)"
