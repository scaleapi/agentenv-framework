"""The service-* deprecated aliases are GONE. These guards stop them coming back.

The aliases (ServiceArtifact, ServiceUniverseArtifact, the load_service_* methods, ...) were
PEP-562 shims over the environment-* names. Removing a shim is easy to half-do: drop the
__getattr__ but leave the name in __all__ and `from agent_env.env import *` raises
AttributeError while a direct import raises ImportError. That exact regression happened once
during the rename, so __all__/namespace agreement is asserted mechanically below.

Also pins the two boundaries the removal must NOT cross: the persisted `type` discriminator
still reads "service"/"service_universe", and ServiceDBEnv keeps its name permanently.
"""

import pathlib
import subprocess
import sys

import pytest

import agent_env.artifact as artifact_pkg
import agent_env.env as env_pkg
from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.utils.deprecation import deprecation_counts, reset_deprecation_state

# An explicit allowlist rather than importlib, so the set under test is checkable.
_SHIMMED_PACKAGES = {"agent_env.artifact": artifact_pkg, "agent_env.env": env_pkg}

# (module path, removed symbol). Both surfaces were shimmed — the package re-export and the
# deep module path people paste from notebooks — so both must now be gone.
_REMOVED_SYMBOLS = [
    ("agent_env.artifact", "ServiceArtifact"),
    ("agent_env.artifact", "ServiceUniverseArtifact"),
    ("agent_env.artifact.artifacts.environment", "ServiceArtifact"),
    ("agent_env.artifact.artifacts.environment_universe", "ServiceUniverseArtifact"),
    ("agent_env.env", "LoadServiceUniverseArtifactResult"),
    ("agent_env.env", "update_env_instance_service_universe"),
    ("agent_env.env.env", "LoadServiceUniverseArtifactResult"),
    ("agent_env.env.store", "update_env_instance_service_universe"),
]

# (module path, class, removed method, surviving canonical method)
_REMOVED_METHODS = [
    ("agent_env.env.envs.mcp_server", "MCPServerEnv", "load_service_artifact", "load_environment_artifact"),
    ("agent_env.env.envs.mcp_server", "MCPServerEnv", "load_service_universe_artifact", "load_environment_universe_artifact"),
    ("agent_env.env.envs.website", "WebsiteEnv", "load_service_artifact", "load_environment_artifact"),
    ("agent_env.env.envs.multi_env", "MultiEnv", "load_service_artifact", "load_environment_artifact"),
    ("agent_env.env.envs.multi_env", "MultiEnv", "load_service_universe_artifact", "load_environment_universe_artifact"),
    ("agent_env.artifact.artifacts.environment_universe", "EnvironmentUniverseArtifact", "get_service_artifacts", "get_environment_artifacts"),
    ("agent_env.env.store", "EnvInstanceStore", "set_service_universe", "set_environment_universe"),
    ("agent_env.env.store", "EnvInstanceStore", "get_service_universe", "get_environment_universe"),
]


@pytest.fixture(autouse=True)
def _clean_counter():
    reset_deprecation_state()
    yield
    reset_deprecation_state()


@pytest.mark.parametrize("module_path,removed", _REMOVED_SYMBOLS)
def test_removed_symbol_raises_attribute_error(module_path, removed):
    mod = __import__(module_path, fromlist=["__name__"])
    with pytest.raises(AttributeError):
        getattr(mod, removed)


@pytest.mark.parametrize("module_path,removed", _REMOVED_SYMBOLS)
def test_removed_symbol_is_not_exported(module_path, removed):
    """A leftover __all__ entry breaks `import *` even though the attribute is gone."""
    mod = __import__(module_path, fromlist=["__name__"])
    assert removed not in getattr(mod, "__all__", ())


@pytest.mark.parametrize("module_path,cls_name,removed,canonical", _REMOVED_METHODS)
def test_removed_method_is_gone_and_canonical_survives(module_path, cls_name, removed, canonical):
    cls = getattr(__import__(module_path, fromlist=[cls_name]), cls_name)
    assert not hasattr(cls, removed), f"{cls_name}.{removed} must be removed"
    assert callable(getattr(cls, canonical)), f"{cls_name}.{canonical} must survive"


@pytest.mark.parametrize("pkg_name", sorted(_SHIMMED_PACKAGES))
def test_every_all_entry_actually_resolves(pkg_name):
    """Caught a real regression: env/__init__ kept "update_env_instance_service_universe" in
    __all__ after the rename but neither imported it nor put it in the __getattr__ map, so
    `from agent_env.env import *` raised AttributeError and the direct import raised ImportError.

    Removing the shim re-opens exactly that failure mode, so assert they agree.
    """
    pkg = _SHIMMED_PACKAGES[pkg_name]
    assert len(pkg.__all__) > 5, f"{pkg_name}.__all__ is suspiciously small; test would be vacuous"
    unresolvable = []
    for name in pkg.__all__:
        try:
            getattr(pkg, name)
        except AttributeError:
            unresolvable.append(name)
    assert not unresolvable, f"{pkg_name}.__all__ lists unresolvable names: {unresolvable}"


def test_importing_the_packages_does_not_trip_the_deprecation_counter():
    """agent-env's own imports must use the canonical names. Run in a subprocess so this
    process's own imports don't pollute the count."""
    # Every argv element is a literal in place: the probe is fixed, nothing here is
    # caller-influenced, and keeping it inline is what makes that reviewable.
    # nosemgrep: dangerous-subprocess-use-audit -- argv is sys.executable plus a constant -c probe; no caller input reaches it
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent_env.artifact, agent_env.env;"
            "from agent_env.utils.deprecation import deprecation_counts;"
            "print(deprecation_counts())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "{}", f"package import tripped the counter: {out.stdout.strip()}"
    assert deprecation_counts() == {}


def test_registry_holds_the_canonical_type_values_only():
    """The legacy `type` discriminators are gone from the core.

    The Python aliases were removed while the persisted values were deliberately left alone;
    moving them is a separate migration step. A deployment still holding documents under the
    old spellings resolves them through `[artifacts] type_aliases`, not through a
    registry key the core ships."""
    from agent_env.artifact.registry import get_artifact_registry

    registry = get_artifact_registry()
    assert registry["environment"] is EnvironmentArtifact
    assert registry["environment_universe"] is EnvironmentUniverseArtifact
    assert "service" not in registry
    assert "service_universe" not in registry
    assert EnvironmentArtifact.model_fields["type"].default == "environment"
    assert EnvironmentUniverseArtifact.model_fields["type"].default == "environment_universe"


def test_servicedb_is_never_renamed():
    """Permanent exemption: ServiceDBEnv keeps its name and 'service_db' is a permanent type
    value. There is no StateDBEnv and no 'state_db' alias anywhere. Mechanical so a
    well-meaning future sweep cannot "finish" the rename."""
    from agent_env.env.envs.service_db import ServiceDBEnv
    from agent_env.env.registry import get_env_registry

    assert ServiceDBEnv.type == "service_db"
    assert get_env_registry()["service_db"] is ServiceDBEnv
    assert "state_db" not in get_env_registry()

    # Deliberately a filesystem walk, not `git grep`. git grep exits 1 on "no matches" but 128 on
    # any error (not a git repo, no .git in a zip-fetched CI source), and BOTH give empty stdout —
    # so reading stdout alone makes the check pass having tested nothing. Reproduced: run this
    # module from /tmp and the git-grep version passes green. An invariant that silently stops
    # enforcing is worse than no invariant, so this depends on nothing outside the filesystem.
    root = pathlib.Path(__file__).resolve().parents[3]  # tst/unit/artifact/<file> -> repo root
    roots = [root / "src", root / "packages"]
    assert all(r.is_dir() for r in roots), f"scan roots missing, test would be vacuous: {roots}"

    suffixes = {".py", ".toml", ".md", ".sh", ".yaml", ".yml", ".json", ".txt", ".cfg"}
    scanned, hits = 0, []
    for base in roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if any(part in {"__pycache__", ".venv", "venv", "dist", "build", ".egg-info"} for part in path.parts):
                continue
            scanned += 1
            if "state_db" in path.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(path.relative_to(root)))

    # Self-check: if the walk found almost nothing the assertion below is meaningless.
    assert scanned > 100, f"only scanned {scanned} files; the walk is broken, not the invariant"
    assert not hits, f"state_db must not exist anywhere; found in: {hits}"
