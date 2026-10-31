"""What RUN instructions of a saved image wrote, read by the composer after a build (``builder.Builder.written``): each
RUN, given by its command in build order, is found in the image's history from the top, the last command first, each
in the newest layer below the one found before it that a shell ran exactly that command for, so a base's layers of the
same command never stand in for the build's own; and each entry of its layer listed by path, as ``<kind> <what>``:
``file`` or ``exec`` (a file any can run) with its sha256, ``link`` with its target, ``dir``, ``deleted``, or ``opaque``
for a folder whose earlier contents the layer hides. A hard link is listed as its target is. Standard library only, since a builder VM runs it on the image's own Python. Usage: ``python3 layers_in_image.py
<image.tar> <commands as JSON>``; it prints what each wrote, in order, as JSON."""

import hashlib
import json
import sys
import tarfile
from pathlib import PurePosixPath

_CHUNK = 1 << 20


def written(image: str, commands: list[str]) -> list[dict[str, str]]:
    with tarfile.open(image) as saved:
        manifest = json.load(saved.extractfile("manifest.json"))[0]
        history = json.load(saved.extractfile(manifest["Config"]))["history"]
        made = [entry.get("created_by", "") for entry in history if not entry.get("empty_layer")]
        # Matched from the top: a base built without history has fewer entries than layers.
        count = min(len(made), len(manifest["Layers"]))
        layers, found, at = list(zip(made[-count:], manifest["Layers"][-count:])), [], count
        for command in reversed(commands):
            at = next((i for i in range(at - 1, -1, -1) if _ran(layers[i][0], command)), None)
            if at is None:
                raise ValueError(f"no layer of the image below the last one found was made by {command!r}")
            found.append(_files(saved, layers[at][1]))
        return found[::-1]


def _ran(made_by: str, command: str) -> bool:
    """Whether the history entry ``made_by`` is a RUN of ``command`` by a shell, as BuildKit and the classic builder
    record one."""
    return made_by.removesuffix(" # buildkit").endswith(f"/bin/sh -c {command}")


def _files(saved: tarfile.TarFile, member: str) -> dict[str, str]:
    files: dict[str, str] = {}
    with tarfile.open(fileobj=saved.extractfile(member), mode="r|*") as layer:
        for entry in layer:
            path = PurePosixPath("/" + entry.name.removeprefix("./").rstrip("/"))
            kind = "exec" if entry.mode & 0o111 else "file"
            if path.name == ".wh..wh..opq":
                files[str(path.parent)] = "opaque"
            elif path.name.startswith(".wh."):
                files[str(path.with_name(path.name.removeprefix(".wh.")))] = "deleted"
            elif entry.isreg():
                files[str(path)] = f"{kind} {_sha256(layer.extractfile(entry))}"
            elif entry.issym():
                files[str(path)] = f"link {entry.linkname}"
            elif entry.islnk():
                target = "/" + entry.linkname.removeprefix("./")
                files[str(path)] = files.get(target, f"{kind} => {target}")
            elif entry.isdir():
                files[str(path)] = "dir"
    return files


def _sha256(file) -> str:
    hashed = hashlib.sha256()
    while chunk := file.read(_CHUNK):
        hashed.update(chunk)
    return hashed.hexdigest()


if __name__ == "__main__":
    json.dump(written(sys.argv[1], json.loads(sys.argv[2])), sys.stdout)
