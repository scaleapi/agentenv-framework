"""What an image a registry serves holds, read without pulling it: the image manifest each Linux architecture runs, an
image's config, its os-release, from the lowest layer that holds it, and a disk form's disk, from the start of its
layer."""

from __future__ import annotations

import io
import json
import struct
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.store.image_store.registry_api import pin_digest, read_blob, read_blob_start, read_manifest

_INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
# os-release(5): /etc/os-release first, else /usr/lib/os-release; on Ubuntu the first links to the second.
_OS_RELEASE = ("etc/os-release", "usr/lib/os-release")
# tarfile reads zstd from Python 3.14.
_READS_ZSTD = "zst" in tarfile.TarFile.OPEN_METH
# How much of a disk layer is read: its tar headers and the disk's own header lie well within it, compressed.
_DISK_START = 1 << 16
_QCOW2 = b"QFI\xfb"


@dataclass(frozen=True)
class DiskLayer:
    """What the start of a disk form's layer shows: the first file in it, its owner, and the virtual size of the disk
    it holds, a qcow2's from its header, a raw one's its own."""

    path: str
    uid: int
    virtual_size: int


class UnreadableLayerError(ValueError):
    """A layer compressed in a way this Python can't read."""


def platform_manifests(ref: str, auth: RegistryAuth | None) -> tuple[str, dict[str, str]]:
    """``ref`` pinned to the digest its registry serves for it, and the digest of each Linux architecture's image
    manifest under it: an index's entries, or a single image's own, by its config. Raises as ``pin_digest`` does."""
    pinned = ref if "@" in ref else pin_digest(ref, auth)
    top = read_manifest(pinned, auth)
    if top.get("mediaType") in _INDEX_TYPES or "manifests" in top:
        return pinned, {entry["platform"]["architecture"]: entry["digest"] for entry in top.get("manifests") or ()
                        if (entry.get("platform") or {}).get("os") == "linux"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config"
        try:
            read_blob(pinned, top["config"]["digest"], auth, path)
            config = json.loads(path.read_text())
        except OSError as e:
            raise ValueError(f"its config {top['config']['digest']} couldn't be read: {e}") from e
    return pinned, {config["architecture"]: pinned.rpartition("@")[2]} if config.get("os") == "linux" else {}


def labels(ref: str, arch: str, auth: RegistryAuth | None) -> dict[str, str]:
    """The labels of ``ref``'s image for ``arch``, read from its config, an index's or a single image's. Raises
    ValueError when it has none for ``arch``, or as ``read_manifest`` does."""
    return image_config(ref, arch, auth).get("Labels") or {}


def image_config(ref: str, arch: str, auth: RegistryAuth | None) -> dict:
    """The ``config`` object of ``ref``'s image config for ``arch`` (its environment, entrypoint, command, user, working
    folder and labels), an index's or a single image's. Raises ValueError when it has none for ``arch``, or as
    ``read_manifest`` does."""
    pinned, manifests = platform_manifests(ref, auth)
    if arch not in manifests:
        raise ValueError(f"{ref} has no linux/{arch} image")
    manifest = read_manifest(f"{pinned.partition('@')[0]}@{manifests[arch]}", auth)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config"
        try:
            read_blob(ref, manifest["config"]["digest"], auth, path)
            config = json.loads(path.read_text())
        except OSError as e:
            raise ValueError(f"its config {manifest['config']['digest']} couldn't be read: {e}") from e
    return config.get("config") or {}


def disk_layer(ref: str, digest: str, auth: RegistryAuth | None) -> DiskLayer:
    """The disk the image manifest ``digest`` in ``ref``'s repository holds, a disk form's, read from the start of its
    one layer. Raises ValueError when it has another number of layers or its start isn't a disk's,
    UnreadableLayerError when this Python can't decompress it."""
    layers = read_manifest(f"{ref.partition('@')[0]}@{digest}", auth).get("layers") or []
    if len(layers) != 1:
        raise ValueError(f"it has {len(layers)} layers, not the one a disk's image has")
    if layers[0].get("mediaType", "").endswith("+zstd"):
        raise UnreadableLayerError(f"its layer {layers[0]['digest']} is zstd-compressed, which this reads no part of")
    start = read_blob_start(ref, layers[0]["digest"], auth, _DISK_START)
    data = zlib.decompressobj(zlib.MAX_WBITS | 32).decompress(start) if start[:2] == b"\x1f\x8b" else start
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as tar:
            member = next(member for member in tar if member.isfile())
            head = tar.extractfile(member).read(32)
    except (tarfile.TarError, StopIteration, OSError, EOFError) as e:
        raise ValueError(f"the start of its layer {layers[0]['digest']} holds no file: {type(e).__name__}: {e}") from e
    size = struct.unpack(">Q", head[24:32])[0] if head[:4] == _QCOW2 and len(head) == 32 else member.size
    return DiskLayer(member.name.removeprefix("./"), member.uid, size)


def os_release(ref: str, digest: str, auth: RegistryAuth | None) -> dict[str, str]:
    """The os-release fields of the image manifest ``digest`` in ``ref``'s repository, from the lowest layer that holds
    the file, normally the OS's own, so only that layer is fetched. Raises ValueError when no layer holds it or one
    can't be read, UnreadableLayerError when this Python can't decompress one it reaches."""
    manifest = read_manifest(f"{ref.partition('@')[0]}@{digest}", auth)
    found: dict[str, bytes | str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for layer in manifest.get("layers") or ():
            if layer.get("mediaType", "").endswith("+zstd") and not _READS_ZSTD:
                raise UnreadableLayerError(f"its layer {layer['digest']} is zstd-compressed, which Python reads from "
                                           "3.14 on")
            path = Path(tmp) / layer["digest"].replace(":", "-")
            try:
                read_blob(ref, layer["digest"], auth, path)
                with tarfile.open(path, "r:*") as tar:
                    for member in tar:
                        name = member.name.removeprefix("./")
                        if name in _OS_RELEASE:
                            if member.issym():
                                found[name] = member.linkname
                            elif (extracted := tar.extractfile(member)) is not None:
                                found[name] = extracted.read()
                path.unlink()
            except (OSError, tarfile.TarError) as e:
                raise ValueError(f"its layer {layer['digest']} couldn't be read: {e}") from e
            if (text := _resolved(found)) is not None:
                return _parsed(text)
    raise ValueError("no layer holds /etc/os-release or /usr/lib/os-release")


def _resolved(found: dict[str, bytes | str]) -> str | None:
    for name in _OS_RELEASE:
        value = found.get(name)
        if isinstance(value, str):  # a link, to a path in the image
            target = PurePosixPath(value) if value.startswith("/") else PurePosixPath("/", name).parent / value
            value = found.get(_normalized(target))
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return None


def _normalized(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts[1:]:
        if part == "..":
            parts = parts[:-1]
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _parsed(text: str) -> dict[str, str]:
    release = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            release[key] = value.strip().strip("\"'")
    return release
