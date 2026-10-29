"""What the bundle tests share: laying out and planning a folder, the @local namespace's store, and a
configured store that fails any test reaching it."""

import shutil
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.bundle import parse_bundle
from agent_env.bundle import requires as requires_module
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import BuiltImage, resolve_bundle
from agent_env.config import get_config
from agent_env.env.published_images import AGENT_IMAGES
from agent_env.plugins import Inventory, _requirements
from agent_env.store.document_store import DocumentStore

# The digest the registry fixture pins every tag to.
DIGEST = "sha256:" + "a" * 64


def layout(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def context_image(root, id, files=None, dockerfile="Dockerfile", folder=None):
    """An image stored as its build context, as a put does, from ``root``/contexts/``folder``: by default a Dockerfile,
    the file it copies, and one it doesn't."""
    files = files or {dockerfile: "FROM python:3.12-slim\nCOPY app.py /app.py\n", "app.py": f"print({id!r})\n",
                      "unread.md": "never copied\n"}
    context = layout(root / "contexts" / (folder or id), files)
    return DockerImageArtifact.put_context(id, description=id, context_path=str(context), dockerfile_path=dockerfile)


def shipped_agent(root, name="solver"):
    """A copy of the Claude Code agent agent-env ships, as ``root``'s agent ``name``: the sources its image is published
    for."""
    shutil.copytree(AGENT_IMAGES[0].dockerfile.parent, root / "agents" / name,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return root / "agents" / name


def listing(folder):
    """The files under ``folder``, as POSIX paths relative to it, a link marked with a trailing ``@``."""
    return sorted(path.relative_to(folder).as_posix() + ("@" if path.is_symlink() else "")
                  for path in Path(folder).rglob("*") if not path.is_dir() or path.is_symlink())


def fake_plugins(monkeypatch, distributions=(), *, versions=None, packages=None, metadata=None, failures=None):
    """This machine's packages, as a bundle's requirements and detection read them: the plugin ``distributions`` the
    inventory lists, and the ``failures`` of those that didn't load; each installed package's version, by its
    canonical name; the packages each top-level module comes from; and each package's ``metadata`` (``requires``,
    ``files``)."""
    def read(name):
        if name not in (metadata or {}):
            raise PackageNotFoundError(name)
        return metadata[name]

    monkeypatch.setattr(requires_module, "inventory", lambda load: Inventory(
        distributions=tuple(distributions), group_errors={}, discovery_errors={}, config_path=None, config_error=None,
        loaded=load))
    monkeypatch.setattr(requires_module, "load_failures", lambda: failures or {})
    monkeypatch.setattr(requires_module, "packages_distributions", lambda: packages or {})
    monkeypatch.setattr(requires_module, "distribution", read)
    monkeypatch.setattr(_requirements, "installed", (versions or {}).get)
    monkeypatch.setattr(_requirements, "distribution", read)


def plan_of(root):
    return plan_bundle(resolve_bundle(parse_bundle(root)))


def local_builds(plan):
    """The images ``plan`` builds, which a run whose deploys all run on this machine builds here as it writes them."""
    return frozenset(write.id for write in plan.writes if isinstance(write.source, BuiltImage))


def local_store():
    return get_config().local_namespace_document_store()


class RefusingStore(DocumentStore):
    """A configured store that only index setup may touch: building an entity store sets up its indexes
    in every store it routes to, as every CLI command does."""

    def __getattribute__(self, name):
        if name.startswith("__") or name == "ensure_index":
            return object.__getattribute__(self, name)
        raise AssertionError(f"a bundle write reached the configured store: {name}")

    def ensure_index(self, collection, fields, unique=False, ttl_seconds=None):
        pass

    find_one = query = count = insert = update = update_one_and_get = replace = delete = None
