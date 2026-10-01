"""Bundles installed with a package, found through the ``agent_env.bundles`` entry points.

A distribution registers ``<name> = "<package>"`` and ships the bundle folder at ``<package>/<name>/``. The folder is
found without importing any package code: only the top-level package is located, and the rest of the path is walked
as folders. An installed bundle's ids are rooted at its distribution, ``@local/<dist>/<name>``. A name several
distributions install is run as ``<dist>/<name>``, except that agent-env's own bundles keep their bare names.
"""

from __future__ import annotations

import difflib
import importlib.util
import os
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path

from packaging.utils import canonicalize_name

from agent_env.plugins._discovery import Plugin, claims, identity
from agent_env.plugins._report import INCOMPATIBLE_CORE, INVALID_PLUGIN, NAME_CONFLICT
from agent_env.plugins._requirements import incompatibility
from agent_env.store.ids import LOCAL_PREFIX

from ._fs import os_reason
from .parse import Bundle, BundleError, parse_bundle
from .plan import check_bundle
from .resolve import resolve_bundle

BUNDLES = "agent_env.bundles"
CORE = "agentenv-framework"


@dataclass(frozen=True)
class InstalledBundle:
    """A bundle an installed distribution registers: its folder, or why it can't run."""

    name: str
    package: str  # the distribution, as its metadata spells it
    version: str
    value: str  # the package the entry point names
    root: Path | None
    code: str | None  # the plugin report's code for ``problem``
    problem: str | None

    @property
    def dist(self) -> str:
        return canonicalize_name(self.package)

    @property
    def qualified(self) -> str:
        return f"{self.dist}/{self.name}"

    @property
    def id_root(self) -> str:
        return f"{LOCAL_PREFIX}{self.dist}/{self.name}"


class _Unresolved(Exception):
    pass


def installed_bundles() -> tuple[InstalledBundle, ...]:
    """Every bundle installed distributions register, in name order, each resolved to its folder or with the
    reason it can't run. Raises what reading the installed metadata raises."""
    found = []
    for name, registrations in claims(BUNDLES).items():
        per_package = Counter(plugin.dist for plugin, _ in registrations)
        for plugin, ep in registrations:
            found.append(_installed(name, plugin, ep, repeated=per_package[plugin.dist] > 1))
    return tuple(found)


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
        check_bundle(resolve_bundle(parsed))
    return parsed


def _installed(name: str, plugin: Plugin, ep: EntryPoint, *, repeated: bool) -> InstalledBundle:
    package, version = identity(ep)

    def installed(root: Path | None = None, code: str | None = None, problem: str | None = None) -> InstalledBundle:
        return InstalledBundle(name, package, version, plugin.value, root, code, problem)

    if not plugin.dist:
        return installed(code=INVALID_PLUGIN, problem="its distribution's metadata can't be read")
    if repeated:
        return installed(code=NAME_CONFLICT, problem=f"{package} registers the bundle {name!r} more than once")
    if unmet := incompatibility(getattr(ep, "dist", None)):
        return installed(code=INCOMPATIBLE_CORE, problem=f"{package} {unmet}")
    try:
        return installed(root=_folder(ep, name))
    except _Unresolved as e:
        return installed(code=INVALID_PLUGIN, problem=str(e))


def _folder(ep: EntryPoint, name: str) -> Path:
    """The folder ``<package>/<name>`` the entry point names, found without importing the package."""
    try:
        package, attr = ep.module, ep.attr
    except (AttributeError, AssertionError):  # a value that doesn't parse: AssertionError from 3.13 on
        package, attr = None, None
    if package is None or attr:
        raise _Unresolved(f"the entry point's value {ep.value!r} isn't a package name; name the package that holds "
                          f"the {name!r} folder")
    if "/" in name:
        raise _Unresolved(f"the bundle name {name!r} contains '/'")
    top, *rest = package.split(".")
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise _Unresolved(f"{top!r} isn't an installed package")
    # A namespace package can have several locations, and an editable install can add ones that aren't folders.
    locations = [Path(location) for location in spec.submodule_search_locations if os.path.isdir(location)]
    if not locations:
        raise _Unresolved(f"the package {top!r} isn't unpacked on disk; reinstall it unpacked")
    for location in locations:
        if (folder := _spelled(location, (*rest, name))) is not None:
            return folder
    raise _Unresolved(f"the package {package!r} has no folder {name!r}")


def _spelled(location: Path, parts: tuple[str, ...]) -> Path | None:
    """``location/parts`` when each part is a folder spelled exactly so, following links without resolving them,
    so a case-insensitive filesystem and a linked folder are judged as they are elsewhere."""
    folder = location
    for part in parts:
        try:
            if part not in os.listdir(folder) or not os.path.isdir(folder / part):
                return None
        except OSError as e:
            raise _Unresolved(f"the folder {folder} can't be read: {os_reason(e)}") from None
        folder /= part
    return folder


def _unknown(text: str, bundles: tuple[InstalledBundle, ...]) -> str:
    if not bundles:
        return f"{text!r} is neither a folder nor an installed bundle, and no bundles are installed"
    names = sorted({run_name(bundle, bundles) for bundle in bundles})
    close = difflib.get_close_matches(text, names, n=3)
    hint = f"; did you mean {' or '.join(repr(name) for name in close)}?" if close else ""
    return f"{text!r} is neither a folder nor an installed bundle{hint} (agent-env run lists the installed bundles)"
