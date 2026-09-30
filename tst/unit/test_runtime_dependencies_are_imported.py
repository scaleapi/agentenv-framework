"""Every runtime dependency must be imported by library code. A dependency nobody imports is dead
weight for every installer and, for the open-source wheel, an unexplained licence and supply-chain
surface. Imports under the container-only trees do not count: those images install their own
requirements and never install the library.

An extra's dependencies are matched by module path rather than top-level name, since distributions
such as ``google-auth`` and ``google-cloud-storage`` share the ``google`` namespace."""

import ast
import functools
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

import agent_env

SRC = Path(agent_env.__file__).resolve().parent
PYPROJECT = SRC.parents[1] / "pyproject.toml"
CONTAINER_ONLY = (SRC / "env" / "gateway",)
SELF = "agentenv-framework"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(dep: str) -> str:
    return _normalize(re.split(r"[<>=!~\[ ;]", dep.strip(), 1)[0])


def _runtime_requirements() -> list[str]:
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    return [_requirement_name(dep) for dep in project["dependencies"]]


def _optional_requirements() -> dict[str, list[str]]:
    """Each runtime extra's own distributions; ``dev`` is test tooling."""
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    return {
        extra: [name for dep in deps if (name := _requirement_name(dep)) != SELF]
        for extra, deps in extras.items()
        if extra != "dev"
    }


@functools.cache
def _import_names(distribution: str) -> frozenset[str]:
    names = {
        top for top, dists in importlib.metadata.packages_distributions().items()
        if any(_normalize(d) == distribution for d in dists)
    }
    # Editable installs (the in-repo protocol package) publish no top-level map, and that distribution's name
    # differs from its import package's, so read the packages under the directory its .pth file adds.
    return frozenset(names or _editable_import_names(distribution) or {distribution.replace("-", "_")})


def _editable_import_names(distribution: str) -> set[str]:
    dist = importlib.metadata.distribution(distribution)
    names = set()
    for file in dist.files or ():
        if file.suffix != ".pth":
            continue
        for line in Path(dist.locate_file(file)).read_text().splitlines():
            root = Path(line.strip())
            if root.is_dir():
                names |= {child.name for child in root.iterdir() if (child / "__init__.py").is_file()}
    return names


@functools.cache
def _module_paths(distribution: str) -> frozenset[str]:
    """Every dotted module path the installed distribution's files provide, namespaces included."""
    paths: set[str] = set()
    for file in importlib.metadata.distribution(distribution).files or []:
        if file.suffix != ".py" or ".." in file.parts:
            continue
        parts = file.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        paths.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return frozenset(paths)


def _provides(distribution: str, module: str) -> bool:
    paths = _module_paths(distribution)
    # An editable install lists no module files, only the path hook that finds them.
    return module in paths if paths else module.split(".")[0] in _import_names(distribution)


@functools.cache
def _installed_module_paths() -> frozenset[str]:
    return frozenset().union(
        *(_module_paths(_normalize(d.metadata["Name"])) for d in importlib.metadata.distributions())
    )


def _library_files() -> list[Path]:
    return [
        path for path in SRC.rglob("*.py")
        if not any(path.is_relative_to(tree) for tree in CONTAINER_ONLY)
    ]


def _imports(path: Path) -> list[tuple[str, ...]]:
    """Each absolute import in ``path`` as the module path it names. ``from X import Y`` names
    ``X.Y`` when that is a module of some installed distribution, else ``X``: a namespace such as
    ``google.cloud`` belongs to every distribution under it."""
    found: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [(alias.name,) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found += [
                (f"{node.module}.{alias.name}",)
                if f"{node.module}.{alias.name}" in _installed_module_paths()
                else (node.module,)
                for alias in node.names
            ]
    return found


def _third_party(candidates: tuple[str, ...]) -> bool:
    top = candidates[-1].split(".")[0]
    return top not in sys.stdlib_module_names and top not in ("__future__", "agent_env")


def _owners(candidates: tuple[str, ...], distributions: list[str]) -> set[str]:
    for candidate in candidates:
        owners = {d for d in distributions if _provides(d, candidate)}
        if owners:
            return owners
    return set()


def _imported_top_levels() -> set[str]:
    return {candidates[-1].split(".")[0] for path in _library_files() for candidates in _imports(path)}


def test_every_runtime_dependency_is_imported_by_the_library():
    imported = _imported_top_levels()
    unused = [dep for dep in _runtime_requirements() if not (_import_names(dep) & imported)]
    assert unused == [], f"runtime dependencies nothing under src/agent_env imports: {unused}"


def test_every_optional_dependency_is_imported_by_the_library():
    extras = _optional_requirements()
    declared = _runtime_requirements() + [d for deps in extras.values() for d in deps]
    imported = {
        owner
        for path in _library_files()
        for candidates in _imports(path)
        for owner in _owners(candidates, declared)
    }
    unused = {extra: [d for d in deps if d not in imported] for extra, deps in extras.items()}
    assert all(not deps for deps in unused.values()), f"extra dependencies nothing imports: {unused}"


def test_a_module_using_an_extra_imports_only_what_the_extra_and_core_declare():
    """Otherwise the extra works only while something else happens to install the package."""
    core = _runtime_requirements()
    for extra, deps in _optional_requirements().items():
        own = [d for d in deps if d not in core]
        for path in _library_files():
            imports = [c for c in _imports(path) if _third_party(c)]
            if not any(_owners(c, own) for c in imports):
                continue
            undeclared = [c[0] for c in imports if not _owners(c, core + deps)]
            assert undeclared == [], (
                f"{path.relative_to(SRC)} uses the {extra!r} extra but imports {undeclared}, "
                f"which neither it nor the core dependencies declare"
            )
