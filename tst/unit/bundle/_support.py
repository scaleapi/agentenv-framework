"""What the bundle tests share: laying out and planning a folder, the @local namespace's store, and a
configured store that fails any test reaching it."""

from pathlib import Path

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.bundle import parse_bundle
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import BuiltImage, resolve_bundle
from agent_env.config import get_config
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


def listing(folder):
    """The files under ``folder``, as POSIX paths relative to it, a link marked with a trailing ``@``."""
    return sorted(path.relative_to(folder).as_posix() + ("@" if path.is_symlink() else "")
                  for path in Path(folder).rglob("*") if not path.is_dir() or path.is_symlink())


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
