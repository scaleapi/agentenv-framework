"""LocalFilesystemObjectStore runs the full ObjectStore conformance suite.

Fast tier — no S3, no network (stdlib filesystem over a temp dir), so the same
assertions that prove S3 parity also give quick backend-neutral coverage.
"""

import pytest

from agent_env.store import LocalFilesystemObjectStore
from tst.store import object_conformance


@pytest.fixture
def store(tmp_path):
    return LocalFilesystemObjectStore(str(tmp_path))


@pytest.mark.parametrize("case", object_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store):
    case(store, "")


def test_resolve_rejects_key_escaping_root(store, tmp_path):
    """A key that traverses out of the store root is refused rather than
    silently written outside the directory."""
    with pytest.raises(ValueError):
        store.put("../escapes.bin", b"payload")
    assert not (tmp_path.parent / "escapes.bin").exists()


def test_root_gitignore_key_is_reserved(tmp_path):
    """The store writes its ignore rule at the root's .gitignore; an object there would
    clobber the rule and then vanish from listings."""
    store = LocalFilesystemObjectStore(str(tmp_path / "root"))
    for key in (".gitignore", "a/../.gitignore"):
        with pytest.raises(ValueError, match="reserved"):
            store.put(key, b"mine")
    assert not (tmp_path / "root").exists()
    store.put("a/.gitignore", b"nested is an ordinary key")
    assert (tmp_path / "root" / ".gitignore").read_text() == "*\n"
    assert store.list("") == ["a/.gitignore"]


def test_resolve_allows_nested_key(store):
    """A normal nested key still resolves and round-trips under the root."""
    locator = store.put("nested/dir/object.bin", b"payload")
    assert store.get(locator) == b"payload"


def test_at_ops_reject_foreign_scheme(store):
    """The url-addressed _at ops only serve file:// urls under the root; an s3:// url
    (cross-backend ingest) is refused rather than silently reaching out to S3."""
    with pytest.raises(ValueError):
        store.list_at("s3://bucket/prefix/")
    with pytest.raises(ValueError):
        store.get_object_metadata_at("s3://bucket/key")
    with pytest.raises(ValueError):
        store.put_file_at("s3://bucket/key", __file__)


def test_signed_get_url_is_none(store):
    """Local FS can't produce a fetchable URL — callers stream through agent-env instead."""
    assert store.signed_get_url("file:///whatever") is None


def test_signed_put_url_is_none(store):
    """Local FS can't presign uploads either — VM-upload flows require a signable backend."""
    assert store.signed_put_url("file:///whatever") is None
