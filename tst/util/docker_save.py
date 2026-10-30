"""Synthetic ``docker save`` tarballs, for tests that read what a stored image holds."""

import gzip
import hashlib
import io
import json
import tarfile


def tar_bytes(files: dict[str, str | bytes], links: dict[str, str] | None = None, gz: bool = False) -> bytes:
    """A tar of ``files``, and of ``links`` as symlinks."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz" if gz else "w") as tar:
        for name, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
    return out.getvalue()


def saved_image(layers: list[dict[str, str | bytes]], oci: bool = False, gz_layers: bool = False, gz: bool = False,
                manifest_first: bool = False, tags: list[str] | None = None) -> bytes:
    """A ``docker save`` tarball of ``layers``, bottom first, in the legacy or the OCI layout; named by ``tags``, as
    docker save names an image in each, when given."""
    blobs = [tar_bytes(files, gz=gz_layers) for files in layers]
    names = ([f"blobs/sha256/{hashlib.sha256(blob).hexdigest()}" for blob in blobs] if oci
             else [f"{index:064x}/layer.tar" for index in range(len(blobs))])
    files: dict[str, str | bytes] = dict(zip(names, blobs))
    files["blobs/sha256/config" if oci else "config.json"] = json.dumps({"rootfs": {"type": "layers"}})
    image = {"Config": "config.json", "Layers": names, **({"RepoTags": tags} if tags is not None else {})}
    manifest = {"manifest.json": json.dumps([image])}
    extra = {"oci-layout": '{"imageLayoutVersion": "1.0.0"}', "index.json": "{}"} if oci else {}
    if oci and tags:
        extra["index.json"] = json.dumps({"schemaVersion": 2, "manifests": [{
            "mediaType": "application/vnd.oci.image.index.v1+json", "digest": "sha256:" + "d" * 64, "size": 1,
            "annotations": {"io.containerd.image.name": f"docker.io/{tags[0]}",
                            "org.opencontainers.image.ref.name": tags[0].rsplit(":", 1)[-1], "org.example.kept": "yes"},
        }]})
    elif tags:
        repository, tag = tags[0].rsplit(":", 1)
        extra["repositories"] = json.dumps({repository: {tag: "0" * 64}})
    data = tar_bytes({**manifest, **files, **extra} if manifest_first else {**files, **extra, **manifest})
    return gzip.compress(data) if gz else data
