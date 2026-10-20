"""The Python and global npm packages a stored image holds, read from its ``docker save`` tarball without Docker."""

from __future__ import annotations

import json
import posixpath
import re
import tarfile
from dataclasses import dataclass
from email.parser import HeaderParser
from typing import BinaryIO

# A distribution's metadata, or its record of a path, URL or VCS it was installed from, by the folder that holds it:
# site-packages or dist-packages, any interpreter or venv.
_DIST = re.compile(r"(?:.*/)?(?:site|dist)-packages/[^/]+\.(?:dist-info|egg-info)/(?P<file>METADATA|PKG-INFO|direct_url\.json)")
# A package npm installed globally: one folder under a global lib/node_modules, scoped or not, inside no other package.
_NPM = re.compile(r"(?P<prefix>.*/)?lib/node_modules/(?:@[^/]+/)?[^@/][^/]*/package\.json")
_WHITEOUT, _OPAQUE = ".wh.", ".wh..wh..opq"
_METADATA_BYTES = 65_536  # a distribution's name and version come first


@dataclass(frozen=True)
class Installed:
    """What an image holds: each Python distribution, by its normalized name, at each version found, and each package
    npm installed globally."""

    python: dict[str, tuple[str, ...]]
    npm: dict[str, str]
    direct: frozenset[str] = frozenset()  # Python distributions installed from a path, URL or VCS, not an index


@dataclass
class _Layer:
    found: dict[tuple[str, str], tuple[str, str]]  # (kind, metadata file's path) -> (name, version)
    direct: set[str]  # the direct_url.json paths of distributions installed from a path, URL or VCS
    whiteouts: list[str]  # paths a lower layer's files are removed at
    opaque: list[str]  # folders a lower layer's files are hidden in


def normalized(name: str) -> str:
    """A Python distribution's name as PEP 503 compares names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_packages(saved: BinaryIO) -> Installed | None:
    """What the ``docker save`` tarball ``saved`` (gzipped or not) holds, read in one pass; None when it names no
    order for its layers or a layer it names can't be read."""
    layers: dict[str, _Layer] = {}
    links: dict[str, str] = {}  # docker save links a layer it holds twice to its first copy
    order: list[str] | None = None
    with tarfile.open(fileobj=saved, mode="r|*") as outer:
        for member in outer:
            name = posixpath.normpath(member.name)
            if member.issym():
                links[name] = posixpath.normpath(posixpath.join(posixpath.dirname(name), member.linkname))
            elif not member.isreg():
                continue
            elif name == "manifest.json":
                order = _layer_order(outer.extractfile(member).read())
            elif (layer := _layer(outer.extractfile(member))) is not None:
                layers[name] = layer
    if order is None:
        return None
    order = [_followed(posixpath.normpath(name), links) for name in order]
    if not all(name in layers for name in order):
        return None
    python: dict[str, tuple[str, str]] = {}
    npm: dict[str, tuple[str, str]] = {}
    direct: set[str] = set()
    for name in order:
        layer = layers[name]
        for found in (python, npm):
            for path in [path for path in found if _hidden(path, layer)]:
                del found[path]
        direct = {path for path in direct if not _hidden(path, layer)} | layer.direct
        for (kind, path), package in layer.found.items():
            (python if kind == "python" else npm)[path] = package
    versions: dict[str, set[str]] = {}
    for package, version in python.values():
        versions.setdefault(normalized(package), set()).add(version)
    folders = {posixpath.dirname(path) for path in direct}
    return Installed(python={name: tuple(sorted(found)) for name, found in sorted(versions.items())},
                     npm={package: version for package, version in sorted(npm.values())},
                     direct=frozenset(normalized(package) for path, (package, _) in python.items()
                                      if posixpath.dirname(path) in folders))


def _followed(name: str, links: dict[str, str]) -> str:
    for _ in range(len(links)):
        if name not in links:
            break
        name = links[name]
    return name


def _layer_order(manifest: bytes) -> list[str] | None:
    """The layers ``docker save``'s manifest names for its image, bottom first; None when it names none."""
    try:
        layers = json.loads(manifest)[0]["Layers"]
    except (ValueError, LookupError, TypeError):
        return None
    return [layer for layer in layers if isinstance(layer, str)] if isinstance(layers, list) else None


def _layer(stream: BinaryIO) -> _Layer | None:
    """One layer's packages and whiteouts; None when ``stream`` is no tar, as an image's config isn't."""
    layer = _Layer({}, set(), [], [])
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                path = posixpath.normpath(member.name.lstrip("/"))  # normpath drops a leading ./
                folder, base = posixpath.split(path)
                if base == _OPAQUE:
                    layer.opaque.append(folder)
                elif base.startswith(_WHITEOUT):
                    layer.whiteouts.append(posixpath.join(folder, base[len(_WHITEOUT):]))
                elif member.isreg() and (match := _DIST.fullmatch(path)):
                    if match["file"] == "direct_url.json":
                        layer.direct.add(path)
                    elif (package := _distribution(tar.extractfile(member))) is not None:
                        layer.found["python", path] = package
                elif member.isreg() and (match := _NPM.fullmatch(path)) and "node_modules" not in (match["prefix"] or ""):
                    if (package := _npm_package(tar.extractfile(member))) is not None:
                        layer.found["npm", path] = package
    except tarfile.TarError:
        return None
    return layer


def _hidden(path: str, layer: _Layer) -> bool:
    """Whether ``layer``'s whiteouts or opaque folders hide the file a lower layer holds at ``path``."""
    return any(path == gone or path.startswith(f"{gone}/") for gone in layer.whiteouts) or any(
        path.startswith(f"{folder}/") for folder in layer.opaque)


def _distribution(stream: BinaryIO) -> tuple[str, str] | None:
    headers = HeaderParser().parsestr(stream.read(_METADATA_BYTES).decode("utf-8", "replace"), headersonly=True)
    name, version = headers.get("Name"), headers.get("Version")
    return (name.strip(), version.strip()) if name and version else None


def _npm_package(stream: BinaryIO) -> tuple[str, str] | None:
    try:
        package = json.loads(stream.read())
    except ValueError:
        return None
    name, version = package.get("name"), package.get("version")
    return (name, version) if isinstance(name, str) and isinstance(version, str) else None
