"""What RUN instructions of a saved image wrote, read by the composer after a build (``builder.Builder.written``): each
RUN, given by its command, is found in the image's history, and each file and link of its layer listed by path, as the
file's sha256, ``-> <target>`` for a link, or what its target has for a hard link; folders and deletions are left out.
Standard library only, since a builder VM runs it on the image's own Python. Usage: ``python3 layers_in_image.py
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
        layers = list(zip(reversed(made), reversed(manifest["Layers"])))
        return [_files(saved, _layer(layers, command)) for command in commands]


def _layer(layers: list[tuple[str, str]], command: str) -> str:
    found = next((layer for made_by, layer in layers if command in made_by), None)
    if found is None:
        raise ValueError(f"no layer of the image was made by {command!r}")
    return found


def _files(saved: tarfile.TarFile, member: str) -> dict[str, str]:
    files: dict[str, str] = {}
    with tarfile.open(fileobj=saved.extractfile(member), mode="r|*") as layer:
        for entry in layer:
            path = "/" + entry.name.removeprefix("./").rstrip("/")
            if PurePosixPath(path).name.startswith(".wh."):
                continue
            if entry.isreg():
                files[path] = _sha256(layer.extractfile(entry))
            elif entry.issym():
                files[path] = f"-> {entry.linkname}"
            elif entry.islnk():
                target = "/" + entry.linkname.removeprefix("./")
                files[path] = files.get(target, f"=> {target}")
    return files


def _sha256(file) -> str:
    hashed = hashlib.sha256()
    while chunk := file.read(_CHUNK):
        hashed.update(chunk)
    return hashed.hexdigest()


if __name__ == "__main__":
    json.dump(written(sys.argv[1], json.loads(sys.argv[2])), sys.stdout)
