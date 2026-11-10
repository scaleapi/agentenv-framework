"""Keep only the images of a local registry's repository that a run used, before the registry's storage is cached.

``prune`` drops each tag the registry's log has no manifest request for, then each manifest no remaining tag reaches,
following an image index to the manifests it lists. The registry's ``garbage-collect`` then frees the layers only those
manifests used; with ``--delete-untagged`` it would also delete a kept index's untagged platform manifests. ``links``,
run after it, drops the repository's links to layers it freed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_V2 = Path("docker") / "registry" / "v2"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pruning = commands.add_parser("prune", help="drop the tags and manifests the run didn't use")
    linking = commands.add_parser("links", help="drop links to layers garbage-collect freed")
    for command in (pruning, linking):
        command.add_argument("storage", type=Path, help="the registry's storage root")
        command.add_argument("repository")
    pruning.add_argument("log", type=Path, help="the registry's log of the run")
    args = parser.parse_args(argv)
    if args.command == "prune":
        kept, dropped = prune(args.storage, args.repository, used_tags(args.log.read_text(errors="replace"),
                                                                        args.repository))
        print(f"kept {kept} tags of {args.repository}, dropped {dropped}")
    else:
        print(f"dropped {drop_freed_links(args.storage, args.repository)} links to freed layers")
    return 0


def used_tags(log: str, repository: str) -> set[str]:
    """The tags the log names a request for a manifest of ``repository`` by."""
    return set(re.findall(rf"/v2/{re.escape(repository)}/manifests/([A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}})(?=[\s\"?])",
                          log))


def prune(storage: Path, repository: str, used: set[str]) -> tuple[int, int]:
    """Drop the tags of ``repository`` not in ``used``, then the manifests no remaining tag reaches; how many tags it
    kept and dropped."""
    manifests = storage / _V2 / "repositories" / repository / "_manifests"
    tags = _children(manifests / "tags")
    for tag in tags:
        if tag.name not in used:
            shutil.rmtree(tag)
    kept = [tag for tag in tags if tag.name in used]
    reached = _reached(storage, [(tag / "current" / "link").read_text().strip() for tag in kept])
    for revision in _children(manifests / "revisions" / "sha256"):
        if f"sha256:{revision.name}" not in reached:
            shutil.rmtree(revision)
    return len(kept), len(tags) - len(kept)


def drop_freed_links(storage: Path, repository: str) -> int:
    """Drop each link of ``repository`` to a layer that's no longer stored; how many."""
    dropped = 0
    for layer in _children(storage / _V2 / "repositories" / repository / "_layers" / "sha256"):
        if not _blob(storage, f"sha256:{layer.name}").is_file():
            shutil.rmtree(layer)
            dropped += 1
    return dropped


def _reached(storage: Path, digests: list[str]) -> set[str]:
    """``digests`` and every manifest an image index among them lists, recursively."""
    reached, pending = set(), list(digests)
    while pending:
        digest = pending.pop()
        if digest in reached:
            continue
        reached.add(digest)
        try:
            manifest = json.loads(_blob(storage, digest).read_bytes())
        except (FileNotFoundError, ValueError):
            continue
        pending += [listed["digest"] for listed in manifest.get("manifests", []) if "digest" in listed]
    return reached


def _children(folder: Path) -> list[Path]:
    return [child for child in folder.iterdir() if child.is_dir()] if folder.is_dir() else []


def _blob(storage: Path, digest: str) -> Path:
    hex_digest = digest.removeprefix("sha256:")
    return storage / _V2 / "blobs" / "sha256" / hex_digest[:2] / hex_digest / "data"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
