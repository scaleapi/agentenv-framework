"""Core imports without any optional extra, and an impl that needs one says which.

Each check runs in a subprocess that refuses every module only the extra installs: ``dev``
installs every extra, so installed state cannot show what an install without one would do."""

import ast
import importlib.metadata
import importlib.util
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

_STORE_PACKAGES = (
    "agent_env.store", "agent_env.store.document_store", "agent_env.store.image_store",
    "agent_env.store.object_store", "agent_env.store.secret_store",
)

_REEXPORT = _BLOCKING + """
for package in json.loads(sys.argv[2]):
    exec(f"from {package} import *", {})
module, _, name = sys.argv[1].partition(":")
try:
    getattr(importlib.import_module(module), name)
except ModuleNotFoundError as e:
    print(e)
"""

_ECR = _BLOCKING + """
import base64
from agent_env.config.errors import ConfigError
from agent_env.store.image_store import EcrCredentials, EcrImageStore
class Client:
    def get_authorization_token(self):
        return {"authorizationData": [{"authorizationToken": base64.b64encode(b"AWS:minted").decode()}]}
store = EcrImageStore("1.dkr.ecr.us-west-2.amazonaws.com", credentials=EcrCredentials(client=Client()))
print(store.auth("1.dkr.ecr.us-west-2.amazonaws.com/env:1").password)
try:
    EcrCredentials(region="us-west-2").client
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
    """Module paths the extra installs and neither a core install nor any other extra does: a
    distribution two extras share (grpcio, for gcp and tensorlake) stays for the other's modules."""
    others = frozenset(_optional_requirements()) - {extra}
    core = _closure(frozenset()) | set().union(*(_closure(frozenset({other})) for other in others))
    provided_by_core = set().union(*(_module_paths(d) for d in core))
    extra_only = _closure(frozenset({extra})) - core
    return sorted(set().union(*(_module_paths(d) for d in extra_only)) - provided_by_core)


def _not_in_core() -> list[str]:
    """Module paths some extra installs and a core install does not, including those two extras share."""
    core = _closure(frozenset())
    provided_by_core = set().union(*(_module_paths(d) for d in core))
    optional = set().union(*(_closure(frozenset({extra})) for extra in _optional_requirements())) - core
    return sorted(set().union(*(_module_paths(d) for d in optional)) - provided_by_core)


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


def test_core_and_every_store_module_import_without_any_extra():
    assert json.loads(_run(_IMPORT_CORE, _not_in_core(), json.dumps(_NOT_CORE))) == []


@pytest.mark.parametrize(
    ("extra", "impl", "base"),
    [
        (
            "gcp", "agent_env.store.object_store.gcs_object_store:GcsObjectStore",
            "agent_env.store.object_store:ObjectStore",
        ),
        (
            "gcp", "agent_env.store.secret_store.gcp_secret_manager_secret_store:GcpSecretManagerSecretStore",
            "agent_env.store.secret_store:SecretStore",
        ),
        (
            "aws", "agent_env.store.object_store.s3_object_store:S3ObjectStore",
            "agent_env.store.object_store:ObjectStore",
        ),
        ("aws", "agent_env.store.object_store:S3ObjectStore", "agent_env.store.object_store:ObjectStore"),
        ("aws", "agent_env.store.secret_store:AwsSecretsManagerSecretStore", "agent_env.store.secret_store:SecretStore"),
        (
            "aws", "agent_env.store.document_store.dynamodb_document_store:DynamoDbDocumentStore",
            "agent_env.store.document_store:DocumentStore",
        ),
        ("aws", "agent_env.store.document_store:DynamoDbDocumentStore", "agent_env.store.document_store:DocumentStore"),
        (
            "tensorlake",
            "agent_env.providers.sandbox_providers.tensorlake.provider:TensorlakeSandboxProvider",
            "agent_env.providers.sandbox_providers.sandbox_provider:SandboxProvider",
        ),
    ],
    ids=["gcs", "secret-manager", "s3", "s3-reexport", "secrets-manager-reexport", "dynamodb", "dynamodb-reexport", "tensorlake"],
)
def test_an_impl_without_its_extra_names_the_extra_to_install(extra, impl, base):
    message = _run(_LOAD_IMPL, _only_in(extra), impl, base)
    assert f"pip install 'agentenv-framework[{extra}]'" in message, message


@pytest.mark.parametrize(
    "name",
    [
        "agent_env.store:S3ObjectStore",
        "agent_env.store:AwsSecretsManagerSecretStore",
        "agent_env.store:DynamoDbDocumentStore",
        "agent_env.store.object_store:S3ObjectStore",
        "agent_env.store.secret_store:AwsSecretsManagerSecretStore",
        "agent_env.store.document_store:DynamoDbDocumentStore",
    ],
)
def test_the_store_packages_import_whole_without_boto3_and_an_aws_backend_names_the_extra(name):
    message = _run(_REEXPORT, _only_in("aws"), name, json.dumps(_STORE_PACKAGES))
    assert message.endswith("pip install 'agentenv-framework[aws]'"), message


def test_ecr_credentials_given_a_client_need_no_boto3_and_without_one_name_the_extra():
    password, message = _run(_ECR, _only_in("aws")).splitlines()
    assert password == "minted"
    assert message.endswith("pip install 'agentenv-framework[aws]'"), message


def test_the_sail_vm_provider_without_the_extra_names_the_extra_to_install():
    impl = "agent_env.providers.sandbox_providers.sail_vm.provider:SailVmSandboxProvider"
    message = _run(_LOAD_IMPL, _only_in("sail"), impl, "agent_env.providers.sandbox_providers.sandbox_provider:SandboxProvider")
    assert "pip install 'agentenv-framework[sail]'" in message, message


@pytest.mark.parametrize("package", _STORE_PACKAGES)
def test_a_backend_imported_on_first_use_is_visible_to_type_checkers(package):
    """A name only __getattr__ serves is Any to a type checker, which a strict consumer cannot subclass, so each one
    is also imported under TYPE_CHECKING, in the ``X as X`` form strict mode counts as a re-export."""
    tree = ast.parse(open(importlib.util.find_spec(package).origin, encoding="utf-8").read())
    lazy = {key.value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "lazy_backends"
            for key in node.args[1].keys}
    typed = {alias.name for node in tree.body
             if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"
             for imp in node.body if isinstance(imp, ast.ImportFrom)
             for alias in imp.names if alias.asname == alias.name}
    assert lazy == typed
