"""The bundles agentenv-framework ships, from its ``agent_env.bundles`` entry points: ``<name> = "<package>"`` ships the
folder ``src/<package as a path>/<name>/``. Run, it prints each shipped env's, agent's and artifact's requirements.txt,
the files a release pins agentenv-framework-protocol in."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLES = "agent_env.bundles"


def bundle_roots(repo: Path = REPO) -> dict[str, Path]:
    """Each shipped bundle's folder, by its name."""
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    declared = project.get("entry-points", {}).get(BUNDLES, {})
    return {name: repo / "src" / Path(*package.split(".")) / name for name, package in sorted(declared.items())}


def requirements_files(repo: Path = REPO) -> list[Path]:
    """Every ``<bundle>/<envs|agents|artifacts>/<name>/requirements.txt`` a shipped bundle holds."""
    return sorted(path for root in bundle_roots(repo).values() for path in root.glob("*/*/requirements.txt"))


if __name__ == "__main__":
    print("\n".join(path.relative_to(REPO).as_posix() for path in requirements_files()))
