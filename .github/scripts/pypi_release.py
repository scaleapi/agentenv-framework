"""The files a release put on PyPI, for the CodeArtifact publish to upload as they are.

``download <dir> <name>==<version>...`` waits until PyPI lists each version's wheel and sdist, then fetches every file it
lists into ``<dir>/<name>/`` and checks each against the sha256 PyPI lists. The GitHub release uploads them once it has
published the images and recorded their digests in the wheel, so the CodeArtifact release ships the same wheel. It uses
only the standard library, so the CircleCI publish job can run it before installing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PYPI = "https://pypi.org/pypi"
POLL_SECONDS = 30
PACKAGE_TYPES = {"bdist_wheel", "sdist"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def release_files(name: str, version: str) -> list[dict] | None:
    """The files PyPI lists for ``name==version``, or None until it lists both a wheel and an sdist."""
    try:
        files = json.loads(_get(f"{PYPI}/{name}/{version}/json"))["urls"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            print(f"PyPI answered {error.code} for {name}=={version}", flush=True)
        return None
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"PyPI is unreachable for {name}=={version}: {error}", flush=True)
        return None
    return files if PACKAGE_TYPES <= {file["packagetype"] for file in files} else None


def download(directory: Path, requirements: list[str], wait_minutes: float) -> None:
    deadline = time.monotonic() + wait_minutes * 60
    for requirement in requirements:
        name, _, version = requirement.partition("==")
        while (files := release_files(name, version)) is None:
            if time.monotonic() >= deadline:
                sys.exit(f"{requirement} is not on PyPI after {wait_minutes:g} minutes")
            print(f"Waiting for {requirement} on PyPI", flush=True)
            time.sleep(POLL_SECONDS)
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        for file in files:
            data = _get(file["url"])
            digest = hashlib.sha256(data).hexdigest()
            if digest != file["digests"]["sha256"]:
                sys.exit(f"{file['filename']} has sha256 {digest}, but PyPI lists {file['digests']['sha256']}")
            (target / file["filename"]).write_bytes(data)
            print(f"{file['filename']} sha256:{digest}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download")
    command.add_argument("directory", type=Path)
    command.add_argument("requirements", nargs="+", metavar="NAME==VERSION")
    command.add_argument("--wait-minutes", type=float, default=240)
    args = parser.parse_args(argv)
    for requirement in args.requirements:
        if requirement.count("==") != 1 or requirement.startswith("=") or requirement.endswith("="):
            parser.error(f"{requirement!r} is not NAME==VERSION")
    download(args.directory, args.requirements, args.wait_minutes)


if __name__ == "__main__":
    main()
