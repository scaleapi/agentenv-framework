"""Core imports without any optional extra, and an impl that needs one says which.

Each check runs in a subprocess that refuses every module only the extra installs: ``dev``
installs every extra, so installed state cannot show what an install without one would do."""

import importlib.metadata
import json
import subprocess
import sys

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tst.unit.test_runtime_dependencies_are_imported import SELF, _module_paths, _optional_requirements

_BLOCKING = """
import importlib, importlib.abc, json, sys
blocked = set(json.loads(sys.stdin.read()))
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
sys.meta_path.insert(0, Refuse())
"""

# Not core: the explorer and the sail_vm provider are their extras' own packages, the gateway runs
# only in its container image, and the code runner is a script that reads its arguments on import.
_NOT_CORE = (
    "agent_env.explorer", "agent_env.providers.sandbox_providers.sail_vm", "agent_env.env.gateway",
    "agent_env.task_step.task_steps.run_code_runner",
)

_IMPORT_CORE = _BLOCKING + """
import ast, pkgutil
import agent_env
not_core = tuple(json.loads(sys.argv[1]))
def imports_blocked(name):
    source = importlib.util.find_spec(name).origin
    tree = ast.parse(open(source).read())
    return any(
        alias.name in blocked or f"{node.module}.{alias.name}" in blocked or node.module in blocked
        if isinstance(node, ast.ImportFrom) else alias.name in blocked
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
bad = []
for module in pkgutil.walk_packages(agent_env.__path__, "agent_env.", onerror=bad.append):
    if module.name.startswith(not_core):
        continue
    try:
        importlib.import_module(module.name)
    except ImportError as e:
        if not imports_blocked(module.name):
            bad.append(f"{module.name}: {e}")
print(json.dumps(bad))
"""

_LOAD_IMPL = _BLOCKING + """
from agent_env.config.errors import ConfigError
from agent_env.config.loader import load_impl
module, _, name = sys.argv[2].partition(":")
try:
    load_impl(sys.argv[1], getattr(importlib.import_module(module), name))
except ConfigError as e:
    print(e)
"""


def _closure(extras: frozenset[str]) -> set[str]:
    """The distributions installing this package with ``extras`` pulls in."""
    seen: set[tuple[str, frozenset[str]]] = set()
    pending = [(SELF, extras)]
    while pending:
        name, wanted = pending.pop()
        if (name, wanted) in seen:
            continue
        seen.add((name, wanted))
        try:
            requires = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for line in requires:
            requirement = Requirement(line)
            marker = requirement.marker
            if marker is None or any(marker.evaluate({"extra": e}) for e in wanted | {""}):
                pending.append((canonicalize_name(requirement.name), frozenset(requirement.extras)))
    return {name for name, _ in seen} - {SELF}


def _only_in(extra: str) -> list[str]:
    """Module paths the extra installs and a core install does not."""
    core = _closure(frozenset())
    provided_by_core = set().union(*(_module_paths(d) for d in core))
    extra_only = _closure(frozenset({extra})) - core
    return sorted(set().union(*(_module_paths(d) for d in extra_only)) - provided_by_core)


def _run(script: str, blocked: list[str], *args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", script, *args], input=json.dumps(blocked),
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.mark.parametrize("extra", sorted(_optional_requirements()))
def test_core_and_every_store_module_import_without_the_extra(extra):
    blocked = _only_in(extra)
    assert blocked, f"the {extra!r} extra installs nothing core lacks, so this test proves nothing"
    assert json.loads(_run(_IMPORT_CORE, blocked, json.dumps(_NOT_CORE))) == []


@pytest.mark.parametrize(
    ("impl", "base"),
    [
        ("agent_env.store.object_store.gcs_object_store:GcsObjectStore", "agent_env.store.object_store:ObjectStore"),
        (
            "agent_env.store.secret_store.gcp_secret_manager_secret_store:GcpSecretManagerSecretStore",
            "agent_env.store.secret_store:SecretStore",
        ),
    ],
    ids=["gcs", "secret-manager"],
)
def test_a_gcp_impl_without_the_extra_names_the_extra_to_install(impl, base):
    message = _run(_LOAD_IMPL, _only_in("gcp"), impl, base)
    assert "pip install 'agentenv-framework[gcp]'" in message, message


def test_the_sail_vm_provider_without_the_extra_names_the_extra_to_install():
    impl = "agent_env.providers.sandbox_providers.sail_vm.provider:SailVmSandboxProvider"
    message = _run(_LOAD_IMPL, _only_in("sail"), impl, "agent_env.providers.sandbox_providers.sandbox_provider:SandboxProvider")
    assert "pip install 'agentenv-framework[sail]'" in message, message
