"""A registry serving OCI images from memory, for code that reads an image's manifests and layers without pulling it:
index, image manifest and blob by digest, a tag pinned to the digest it names, each blob fetched recorded.
``Registry().serve(monkeypatch)`` puts it in place of every registry agent-env reads an image's contents from."""

import hashlib
import io
import json
import tarfile

from agent_env.store.image_store import image_contents
from agent_env.store.image_store.registry_api import NoSuchImageError

UBUNTU = b'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\n'
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"


class Registry:
    """A registry serving the manifests and blobs put in it, by digest, and recording the blobs fetched."""

    def __init__(self):
        self.manifests, self.blobs, self.tags, self.fetched = {}, {}, {}, []

    def serve(self, monkeypatch) -> "Registry":
        for name in ("pin_digest", "read_manifest", "read_blob"):
            monkeypatch.setattr(image_contents, name, getattr(self, name))
        return self

    def blob(self, data: bytes) -> str:
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        self.blobs[digest] = data
        return digest

    def image(self, *layers: bytes, arch: str = "amd64") -> str:
        config = self.blob(json.dumps({"os": "linux", "architecture": arch}).encode())
        manifest = {"mediaType": MANIFEST, "config": {"digest": config},
                    "layers": [{"digest": self.blob(layer)} for layer in layers]}
        return self._manifest(manifest)

    def index(self, **images: str) -> str:
        entries = [{"digest": digest, "platform": {"os": "linux", "architecture": arch}} for arch, digest in images.items()]
        entries.append({"digest": self.blob(b"attestation"), "platform": {"os": "unknown", "architecture": "unknown"}})
        return self._manifest({"mediaType": INDEX, "manifests": entries})

    def _manifest(self, manifest: dict) -> str:
        digest = self.blob(json.dumps(manifest).encode())
        self.manifests[digest] = manifest
        return digest

    def pin_digest(self, ref, auth):
        if ref not in self.tags:
            raise NoSuchImageError(f"{ref}: ghcr.io has no such image")
        return f"{ref}@{self.tags[ref]}"

    def read_manifest(self, ref, auth):
        digest = ref.rpartition("@")[2]
        if digest not in self.manifests:
            raise NoSuchImageError(f"{ref}: ghcr.io has no such image")
        return self.manifests[digest]

    def read_blob(self, ref, digest, auth, path):
        self.fetched.append(digest)
        path.write_bytes(self.blobs[digest])


def layer(files: dict[str, bytes | str]) -> bytes:
    """A tar.gz layer of ``files`` by path; a str is a link to that path."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            if isinstance(content, str):
                info.type, info.linkname = tarfile.SYMTYPE, content
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


UBUNTU_LAYER = layer({"etc/os-release": "../usr/lib/os-release", "usr/lib/os-release": UBUNTU})
