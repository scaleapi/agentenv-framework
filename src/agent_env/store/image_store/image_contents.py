"""What an image a registry serves holds, read without pulling it: the image manifest each Linux architecture runs, and
an image's os-release, from the lowest layer that holds it."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.store.image_store.registry_api import pin_digest, read_blob, read_manifest

_INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
# os-release(5): /etc/os-release first, else /usr/lib/os-release; on Ubuntu the first links to the second.
_OS_RELEASE = ("etc/os-release", "usr/lib/os-release")


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
        read_blob(pinned, top["config"]["digest"], auth, path)
        config = json.loads(path.read_text())
    return pinned, {config["architecture"]: pinned.rpartition("@")[2]} if config.get("os") == "linux" else {}


def os_release(ref: str, digest: str, auth: RegistryAuth | None) -> dict[str, str]:
    """The os-release fields of the image manifest ``digest`` in ``ref``'s repository, from the lowest layer that holds
    the file, normally the OS's own, so only that layer is fetched. Raises ValueError when no layer holds it or one
    can't be read."""
    manifest = read_manifest(f"{ref.partition('@')[0]}@{digest}", auth)
    found: dict[str, bytes | str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for layer in manifest.get("layers") or ():
            path = Path(tmp) / layer["digest"].replace(":", "-")
            read_blob(ref, layer["digest"], auth, path)
            try:
                with tarfile.open(path, "r:*") as tar:
                    for member in tar:
                        name = member.name.removeprefix("./")
                        if name in _OS_RELEASE:
                            if member.issym():
                                found[name] = member.linkname
                            elif (extracted := tar.extractfile(member)) is not None:
                                found[name] = extracted.read()
            except (OSError, tarfile.TarError) as e:
                raise ValueError(f"its layer {layer['digest']} isn't a readable tar: {e}") from e
            path.unlink()
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
