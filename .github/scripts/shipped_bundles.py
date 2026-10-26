"""The bundles agentenv-framework ships, from its ``agent_env.bundles`` entry points: ``<name> = "<package>"`` ships the
folder ``src/<package as a path>/<name>/``.

``pin`` pins agentenv-framework-protocol, keeping its extras, in each shipped env's and artifact's requirements.txt to
the version ``packages/agentenv-protocol`` is at, which a release ships them with. An agent's keeps its floor: a pin is
one of its image's sources, so pinning it would build and publish a new image every release, and with the floor its
image is built only when its folder changes. Every release job that reads those folders runs it first, so an image's
tag, its build and the wheel all see the same sources. It uses only the standard library, so the jobs that push images
can run it.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLES = "agent_env.bundles"
PROTOCOL = "agentenv-framework-protocol"
_PINNED_KINDS = ("envs", "artifacts")
# A requirement on the protocol, with its extras, and not on a package whose name starts with the protocol's.
_PROTOCOL_REQUIREMENT = re.compile(rf"^{PROTOCOL}(\[[^\]]*\])?(?![\w.-])")


def bundle_roots(repo: Path = REPO) -> dict[str, Path]:
    """Each shipped bundle's folder, by its name."""
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    declared = project.get("entry-points", {}).get(BUNDLES, {})
    return {name: repo / "src" / Path(*package.split(".")) / name for name, package in sorted(declared.items())}


def requirements_files(repo: Path = REPO) -> list[Path]:
    """Every ``<bundle>/<envs|artifacts>/<name>/requirements.txt`` a shipped bundle holds: those a release pins."""
    return sorted(path for root in bundle_roots(repo).values() for kind in _PINNED_KINDS
                  for path in root.glob(f"{kind}/*/requirements.txt"))


def protocol_version(repo: Path = REPO) -> str:
    return tomllib.loads((repo / "packages" / "agentenv-protocol" / "pyproject.toml").read_text())["project"]["version"]


def pin(repo: Path = REPO) -> list[Path]:
    """Pin the protocol in each shipped requirements file that names it, and return those files."""
    files = requirements_files(repo)
    if not files:
        raise ValueError(f"the bundles {repo / 'pyproject.toml'} declares hold no env or artifact requirements.txt")
    version, pinned = protocol_version(repo), []
    for path in files:
        lines = path.read_text().splitlines(keepends=True)
        if any(_PROTOCOL_REQUIREMENT.match(line) for line in lines):
            path.write_text("".join(_pinned(line, version) for line in lines))
            pinned.append(path)
    return pinned


def _pinned(line: str, version: str) -> str:
    if not (match := _PROTOCOL_REQUIREMENT.match(line)):
        return line
    ending = line[len(line.rstrip("\r\n")):]
    return f"{PROTOCOL}{match.group(1) or ''}=={version}{ending}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_subparsers(dest="command", required=True).add_parser("pin")
    parser.parse_args()
    version = protocol_version()
    for path in pin():
        print(f"pinned {PROTOCOL}=={version} in {path.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
