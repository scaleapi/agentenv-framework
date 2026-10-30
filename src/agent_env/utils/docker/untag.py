"""A ``docker save`` archive copied so that it names no image: loading it then tags nothing, where Docker would otherwise
apply every name it embeds, in place of whatever images the loading machine holds under them."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import posixpath
import tarfile
import zlib
from typing import BinaryIO

# The names an OCI index.json gives the image on its descriptor; manifest.json gives them as RepoTags.
_NAME_ANNOTATIONS = ("io.containerd.image.name", "org.opencontainers.image.ref.name")
_MANIFESTS = ("manifest.json", "index.json")


def write_untagged(saved: BinaryIO, out: BinaryIO, *, gzip_level: int | None = None) -> None:
    """Copy the ``docker save`` archive ``saved``, gzipped or not, to ``out``, gzipped at ``gzip_level`` when given, so
    that it names no image: its manifest.json tags none, its index.json annotates no name, and its legacy repositories
    file is left out; the layers are copied as they are. Raises ValueError when ``saved`` isn't an archive docker save
    writes."""
    found = False
    sink_file = (gzip.GzipFile(fileobj=out, mode="wb", compresslevel=gzip_level, mtime=0) if gzip_level is not None
                 else contextlib.nullcontext(out))
    try:
        with sink_file as sink_out, tarfile.open(fileobj=saved, mode="r|*") as source, \
                tarfile.open(fileobj=sink_out, mode="w|", format=tarfile.PAX_FORMAT) as sink:
            for member in source:
                name = posixpath.normpath(member.name)
                if member.isreg() and name == "repositories":
                    continue
                if member.isreg() and name in _MANIFESTS:
                    found = True
                    data = _unnamed(name, source.extractfile(member).read())
                    member.size = len(data)
                    member.pax_headers.pop("size", None)
                    sink.addfile(member, io.BytesIO(data))
                else:
                    sink.addfile(member, source.extractfile(member) if member.isreg() else None)
    except (tarfile.TarError, EOFError, zlib.error) as e:
        raise ValueError(f"it isn't a tar docker save writes: {e or type(e).__name__}") from None
    if not found:
        raise ValueError("it holds no manifest.json or index.json, so it isn't an image docker save wrote")


def _unnamed(name: str, data: bytes) -> bytes:
    """The manifest.json or index.json ``data``, naming no image."""
    try:
        document = json.loads(data)
    except ValueError as e:
        raise ValueError(f"its {name} isn't JSON: {e}") from None
    if name == "manifest.json":
        if not isinstance(document, list) or not all(isinstance(entry, dict) for entry in document):
            raise ValueError("its manifest.json isn't a list of images")
        for entry in document:
            entry["RepoTags"] = None
    elif isinstance(document, dict):
        for descriptor in document.get("manifests") or ():
            annotations = descriptor.get("annotations") if isinstance(descriptor, dict) else None
            if isinstance(annotations, dict):
                for key in _NAME_ANNOTATIONS:
                    annotations.pop(key, None)
                if not annotations:
                    del descriptor["annotations"]
    else:
        raise ValueError("its index.json isn't an image index")
    return json.dumps(document, separators=(",", ":")).encode()
