"""Bundles installed with a package, found through the ``agent_env.bundles`` entry points (``agent_env.plugins._bundles``
finds them), and run by name: a name several distributions install is run as ``<dist>/<name>``, except that agent-env's
own bundles keep their bare names.
"""

from __future__ import annotations

import difflib
import os
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from packaging.utils import canonicalize_name

from agent_env.plugins._bundles import BUNDLES, CORE, InstalledBundle, installed_bundles

from .hub import HubBundle, fetch_bundle, is_hub_reference
from .materialize import materialize
from .parse import Bundle, BundleError, BundleKind, parse_bundle
from .plan import check_bundle, plan_entities
from .resolve import resolve_bundle

__all__ = ["BUNDLES", "CORE", "InstalledBundle", "Located", "checked", "find_bundle", "installed_bundles",
           "locate_bundle", "run_name", "write_installed"]


class Located(NamedTuple):
    """A bundle's folder, the id root of an installed or Hub bundle (a folder's ids come from its path), and what a Hub
    bundle was downloaded as."""

    root: Path
    id_root: str | None
    hub: HubBundle | None


def locate_bundle(
    text: str, installed: Callable[[], tuple[InstalledBundle, ...]] = installed_bundles, *,
    on_progress: Callable[[str], None] | None = None,
) -> Located:
    """The bundle ``text`` names, as ``agent-env run BUNDLE`` reads it: an ``hf://`` reference, fetched first; else a folder,
    when it starts like a path or exists spelled as written; else a bundle ``installed`` lists. A name that is both a
    folder here and an installed bundle is refused, so a folder in the current directory never stands in for the
    bundle meant. Raises BundleError."""
    if is_hub_reference(text):
        hub = fetch_bundle(text, on_progress=on_progress)
        return Located(hub.root, hub.id_root, hub)
    if path_like(text):
        return Located(Path(text), None, None)
    if _spelled_folder(text):
        if namesakes := _installed_as(text, installed()):
            raise BundleError([f"{text!r} is both a folder here and the installed bundle {namesakes[0].qualified}: run "
                               f"agent-env run ./{text} for the folder or agent-env run {namesakes[0].qualified} for "
                               "the bundle"])
        return Located(Path(text), None, None)
    found = find_bundle(text, installed())
    return Located(found.root, found.id_root, None)


def _installed_as(text: str, bundles: tuple[InstalledBundle, ...]) -> list[InstalledBundle]:
    """The installed bundles ``text`` could name, as ``<name>`` or ``<dist>/<name>``."""
    dist, _, name = text.rpartition("/")
    return [bundle for bundle in bundles if bundle.name == name and (not dist or bundle.dist == canonicalize_name(dist))]


def path_like(text: str) -> bool:
    """Text that names a folder even when there is none there."""
    return text.startswith((".", "/", "~"))


def _spelled_folder(text: str) -> bool:
    """Whether ``text`` names a folder each of whose names is spelled as written: on a case-insensitive filesystem
    ``desktop`` finds ``Desktop``, which isn't the folder meant."""
    if not os.path.isdir(text):
        return False
    above = Path()
    for part in Path(text).parts:
        if part not in (".", "..") and not _lists(above, part):
            return False
        above /= part
    return True


def _lists(folder: Path, name: str) -> bool:
    """Whether ``folder`` lists ``name`` as spelled; a folder that can be passed through but not read takes it as
    written."""
    try:
        names = os.listdir(folder)
    except OSError:
        return True
    return _nfc(name) in {_nfc(listed) for listed in names}


def _nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def find_bundle(text: str, bundles: tuple[InstalledBundle, ...] | None = None) -> InstalledBundle:
    """The installed bundle ``text`` names, as ``<name>`` or ``<dist>/<name>``. Raises BundleError."""
    bundles = installed_bundles() if bundles is None else bundles
    dist, _, name = text.rpartition("/")
    matches = _installed_as(text, bundles)
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
