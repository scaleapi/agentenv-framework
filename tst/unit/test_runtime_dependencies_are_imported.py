"""Every runtime dependency must be imported by library code. A dependency nobody imports is dead
weight for every installer and, for the open-source wheel, an unexplained licence and supply-chain
surface. Imports under the container-only trees do not count: those images install their own
requirements and never install the library."""

import ast
import importlib.metadata
import re
import tomllib
from pathlib import Path

import agent_env

SRC = Path(agent_env.__file__).resolve().parent
PYPROJECT = SRC.parents[1] / "pyproject.toml"
CONTAINER_ONLY = (SRC / "env" / "gateway",)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_requirements() -> list[str]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    return [_normalize(re.split(r"[<>=!~\[ ;]", dep.strip(), 1)[0]) for dep in project["dependencies"]]


def _import_names(distribution: str) -> set[str]:
    names = {
        top for top, dists in importlib.metadata.packages_distributions().items()
        if any(_normalize(d) == distribution for d in dists)
    }
    # Editable installs (the in-repo agentenv-protocol) do not always publish a top-level map.
    return names or {distribution.replace("-", "_")}


def _imported_top_levels() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        if any(path.is_relative_to(tree) for tree in CONTAINER_ONLY):
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_every_runtime_dependency_is_imported_by_the_library():
    imported = _imported_top_levels()
    unused = [dep for dep in _runtime_requirements() if not (_import_names(dep) & imported)]
    assert unused == [], f"runtime dependencies nothing under src/agent_env imports: {unused}"
