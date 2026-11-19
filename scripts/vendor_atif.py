"""Vendor Harbor's ATIF models into agentenv_protocol.atif.models from a named Harbor release on PyPI. Run from
anywhere, with network:

    python scripts/vendor_atif.py 0.26.0 [--check]

It downloads that release's wheel, takes the modules under harbor/models/trajectories, rewrites their imports of
harbor.models.trajectories to agentenv_protocol.atif.models, puts a first line on each saying where it's from and that it
was changed, as Apache-2.0 asks, and changes nothing else. It writes them with VENDORED.md, which records each upstream
file's sha256 so a unit test can check offline that the copies differ from Harbor's only in that line and the import
paths. ``--check`` compares instead of writing and exits 1 on any difference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM = "harbor.models.trajectories"
LOCAL = "agentenv_protocol.atif.models"
SOURCE = "harbor/models/trajectories/"
DEST = Path(__file__).resolve().parents[1] / "packages/agentenv-protocol/src/agentenv_protocol/atif/models"
_IMPORT = re.compile(r"^(\s*(?:from|import)\s+)" + re.escape(UPSTREAM) + r"\b", re.M)
_LOCAL_IMPORT = re.compile(r"^(\s*(?:from|import)\s+)" + re.escape(LOCAL) + r"\b", re.M)
_HEADER = re.compile(r"\A# From Harbor [^\n]*\n")


def localize(text: str, name: str, version: str) -> str:
    """Harbor's module text with its imports of the trajectory models pointed at the vendored copy, under a first line
    naming its source and the change."""
    header = f"# From Harbor {version}, {SOURCE}{name} (Apache-2.0), with its import paths changed: see VENDORED.md.\n"
    return header + _IMPORT.sub(r"\g<1>" + LOCAL, text)


def upstream(text: str) -> str:
    """A vendored module's text as Harbor wrote it: the inverse of ``localize``."""
    return _LOCAL_IMPORT.sub(r"\g<1>" + UPSTREAM, _HEADER.sub("", text))


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def fetch(version: str) -> tuple[str, str, dict[str, str], str]:
    """The release's wheel name, its sha256, its trajectory modules' text by file name, and its license."""
    release = json.loads(_get(f"https://pypi.org/pypi/harbor/{version}/json"))
    wheel = next(u for u in release["urls"] if u["packagetype"] == "bdist_wheel")
    data = _get(wheel["url"])
    digest = hashlib.sha256(data).hexdigest()
    if digest != wheel["digests"]["sha256"]:
        raise SystemExit(f"{wheel['filename']}: sha256 {digest} isn't PyPI's {wheel['digests']['sha256']}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {name[len(SOURCE):]: archive.read(name).decode("utf-8") for name in archive.namelist()
                 if name.startswith(SOURCE) and name.endswith(".py") and "/" not in name[len(SOURCE):]}
        licenses = [name for name in archive.namelist() if name.endswith(".dist-info/licenses/LICENSE")]
        if len(licenses) != 1:
            raise SystemExit(f"{wheel['filename']} has no single dist-info/licenses/LICENSE")
        license_text = archive.read(licenses[0]).decode("utf-8")
    if not files:
        raise SystemExit(f"{wheel['filename']} has no {SOURCE}*.py")
    return wheel["filename"], digest, files, license_text


def vendored_md(version: str, wheel: str, digest: str, files: dict[str, str]) -> str:
    rows = "\n".join(f"| `{name}` | `{hashlib.sha256(text.encode('utf-8')).hexdigest()}` |"
                     for name, text in sorted(files.items()))
    return f"""# Vendored ATIF models

These files are Harbor's ATIF models, from `{SOURCE}` in the Harbor {version} release on PyPI (`{wheel}`, sha256
`{digest}`). Harbor is Apache-2.0; `LICENSE` beside this file is the license that wheel ships.

Each file has a first line naming its source and saying it was changed, and the only change is import paths: each
`from {UPSTREAM}...` import reads `from {LOCAL}...`. Nothing else differs. A unit test takes off the first line, turns
the imports back and checks each file's sha256 against the table below.

Refresh with `python scripts/vendor_atif.py <version>` from the repository root, then review the diff and the fixtures.

| File | Upstream sha256 |
| --- | --- |
{rows}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="the Harbor release, such as 0.26.0")
    parser.add_argument("--check", action="store_true", help="compare with the checked-in copy instead of writing")
    args = parser.parse_args()
    wheel, digest, files, license_text = fetch(args.version)
    wanted = {name: localize(text, name, args.version) for name, text in files.items()}
    wanted["VENDORED.md"] = vendored_md(args.version, wheel, digest, files)
    wanted["LICENSE"] = license_text
    if args.check:
        present = {p.name for p in DEST.glob("*.py")} | {n for n in ("VENDORED.md", "LICENSE") if (DEST / n).exists()}
        differ = sorted(name for name, text in wanted.items()
                        if not (DEST / name).exists() or (DEST / name).read_text(encoding="utf-8") != text)
        extra = sorted(present - set(wanted))
        for name in differ:
            print(f"differs: {name}")
        for name in extra:
            print(f"not in {args.version}: {name}")
        print(f"{len(files)} modules from {wheel}: {'match' if not differ and not extra else 'mismatch'}")
        return 1 if differ or extra else 0
    DEST.mkdir(parents=True, exist_ok=True)
    for stale in DEST.glob("*.py"):
        if stale.name not in wanted:
            stale.unlink()
    for name, text in wanted.items():
        (DEST / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} modules from {wheel} to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
