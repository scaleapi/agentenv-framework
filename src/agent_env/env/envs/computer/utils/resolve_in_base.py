"""Resolve what a computer env's layers ask for, run by the composer at put inside a container of the computer's base
for one arch (``packages`` module), so it resolves against the base's own installed packages: apt packages and their
dependencies, from the OS's repositories and the layers' sources, downloaded into ``<out>/apt``; each layer's Python
requirements as wheels into ``<out>/wheels/<name>``. Standard library only, since it runs on the base's Python.
Usage: ``python3 resolve_in_base.py <request.json> <out>``; it writes ``<out>/result.json``."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KEYRINGS = Path("/etc/apt/keyrings")
SOURCES = Path("/etc/apt/sources.list.d")
ARMORED = b"-----BEGIN PGP PUBLIC KEY BLOCK-----"


def main(request_path: str, out: str) -> None:
    request, folder = json.loads(Path(request_path).read_text()), Path(out)
    result = {"apt": _apt(request["apt"], request["sources"], folder / "apt") if request["apt"] else [],
              "wheels": {layer["name"]: _wheels(layer, folder / "wheels" / layer["name"]) for layer in request["python"]}}
    (folder / "result.json").write_text(json.dumps(result))
    uid, gid = request["owner"]
    for path in [folder, *folder.rglob("*")]:
        os.chown(path, uid, gid)


def _apt(packages: list[str], sources: list[dict], dest: Path) -> list[dict]:
    """Download ``packages`` with every dependency the base doesn't have, no recommends, from the OS's repositories and
    ``sources``, each signed by its key; each .deb with its package, version and arch."""
    KEYRINGS.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sources):
        key = Path(source["key"]).read_bytes()
        ring = KEYRINGS / f"agentenv-{i}{'.asc' if key.lstrip().startswith(ARMORED) else '.gpg'}"
        ring.write_bytes(key)
        components = f"Components: {' '.join(source['components'])}\n" if source["components"] else ""
        (SOURCES / f"agentenv-{i}.sources").write_text(
            f"Types: deb\nURIs: {source['apt']}\nSuites: {source['suite']}\n{components}Signed-By: {ring}\n")
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    (dest / "partial").mkdir(parents=True)
    subprocess.run(["apt-get", "update", "-q"], check=True, env=env)
    subprocess.run(["apt-get", "install", "-y", "-q", "--download-only", "--no-install-recommends",
                    "-o", f"Dir::Cache::Archives={dest}/", *packages], check=True, env=env)
    shutil.rmtree(dest / "partial")
    (dest / "lock").unlink(missing_ok=True)
    debs = []
    for deb in sorted(dest.glob("*.deb")):
        fields = subprocess.run(["dpkg-deb", "-f", str(deb), "Package", "Version", "Architecture"], check=True,
                                capture_output=True, text=True).stdout
        found = dict(line.split(": ", 1) for line in fields.splitlines())
        debs.append({"file": deb.name, "package": found["Package"], "version": found["Version"],
                     "arch": found["Architecture"]})
    return debs


def _wheels(layer: dict, dest: Path) -> list[str]:
    """The layer's requirements, and what its venv adds, as wheels for this Python and arch: a package published only
    as source is built into one here, once."""
    venv = Path("/tmp/agentenv-resolve") / layer["name"]
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    requirements = ["-r", layer["requirements"]] if layer["requirements"] else []
    subprocess.run([str(venv / "bin" / "pip"), "wheel", "--no-cache-dir", "--wheel-dir", str(dest), *requirements,
                    *layer["add"]], check=True, cwd=layer["folder"])
    return sorted(path.name for path in dest.glob("*.whl"))


if __name__ == "__main__":
    main(*sys.argv[1:])
