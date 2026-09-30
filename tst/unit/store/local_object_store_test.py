"""LocalFilesystemObjectStore runs the full ObjectStore conformance suite.

Fast tier — no S3, no network (stdlib filesystem over a temp dir), so the same
assertions that prove S3 parity also give quick backend-neutral coverage.
"""

import errno
import os
from pathlib import Path

import pytest

from agent_env.store import LocalFilesystemObjectStore, ObjectAlreadyExistsError
from tst.store import object_conformance


@pytest.fixture
def store(tmp_path):
    return LocalFilesystemObjectStore(str(tmp_path))


@pytest.mark.parametrize("case", object_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store):
    case(store, "")


def test_owns_only_file_urls_under_its_root(store, tmp_path):
    """A bare path is readable but is not one of this store's urls, and nothing outside the root is."""
    url = store.put("k/x.bin", b"v")
    bare = url.removeprefix("file://")
    assert store.owns(url)
    assert not store.owns(bare)
    assert not store.owns(f"file://{tmp_path.parent}/elsewhere.bin")
    assert store.get(bare) == b"v"


def test_owns_answers_for_a_symlink_loop(store, tmp_path):
    """Path.resolve() raises on a loop on some Python versions and not others; owns() answers either way."""
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    url = f"file://{tmp_path}/loop/x"
    assert store.owns(url) in (True, False)
    if store.owns(url):
        assert store.get_object_metadata_at(url) is None


def test_download_into_a_directory_is_not_a_missing_object(store, tmp_path):
    """Only a missing source is ObjectNotFoundError; a bad destination is the filesystem's error."""
    url = store.put("k/x.bin", b"v")
    dest = tmp_path / "dest-dir"
    dest.mkdir()
    with pytest.raises(IsADirectoryError):
        store.download_to_file(url, str(dest))


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


def test_at_ops_reject_a_bare_path(store):
    """The _at ops take the store's own file:// urls, not bare paths under the root."""
    bare = store.put("k/x.bin", b"v").removeprefix("file://")
    with pytest.raises(ValueError):
        store.get_object_metadata_at(bare)
    with pytest.raises(ValueError):
        store.list_at(bare.rsplit("/", 1)[0] + "/")
    with pytest.raises(ValueError):
        store.put_file_at(bare + ".copy", __file__)


def test_signed_get_url_is_none(store):
    """Local FS can't produce a fetchable URL — callers stream through agent-env instead."""
    assert store.signed_get_url("file:///whatever") is None


def test_signed_put_url_is_none(store):
    """Local FS can't presign uploads either — VM-upload flows require a signable backend."""
    assert store.signed_put_url("file:///whatever") is None


def test_a_write_keeps_its_content_type(store):
    store.put("typed/a.json", b"{}", content_type="application/json")
    store.put_file("typed/b.csv", __file__, content_type="text/csv")
    assert store.get_object_metadata("typed/a.json").content_type == "application/json"
    assert store.get_object_metadata("typed/b.csv").content_type == "text/csv"


def test_a_write_that_names_no_type_reads_back_as_unknown(store):
    """Readers then guess from the name, as they did before types were kept."""
    store.put("untyped/a.zip", b"PK")
    assert store.get_object_metadata("untyped/a.zip").content_type is None


def test_an_overwrite_replaces_the_recorded_type(store):
    store.put("over/x", b"1", content_type="application/json")
    store.put("over/x", b"2", content_type="text/plain", allow_overwrite=True)
    assert store.get_object_metadata("over/x").content_type == "text/plain"
    store.put("over/x", b"3", allow_overwrite=True)
    assert store.get_object_metadata("over/x").content_type is None


def test_an_object_written_before_types_were_kept_has_none(store, tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "x.bin").write_bytes(b"legacy")
    md = store.get_object_metadata("old/x.bin")
    assert (md.content_type, md.size) == (None, 6)


def test_a_failed_write_leaves_no_object(store, monkeypatch):
    def fail(self, data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="disk full"):
        store.put("partial/x.bin", b"payload")
    monkeypatch.undo()
    assert not store.exists("partial/x.bin") and store.list("") == []


def test_a_no_overwrite_write_that_loses_a_race_keeps_the_winner(store, monkeypatch):
    """Another writer lands the key between the existence check and the commit."""
    real_link = os.link

    def racing_link(src, dst):
        Path(dst).write_bytes(b"winner")
        return real_link(src, dst)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(ObjectAlreadyExistsError):
        store.put("race/x", b"loser")
    assert store.read("race/x") == b"winner"


def test_a_filesystem_without_hard_links_still_refuses_an_existing_key(store, monkeypatch):
    def no_links(src, dst):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(os, "link", no_links)
    url = store.put("nolink/x", b"first", content_type="text/plain")
    assert store.get(url) == b"first"
    with pytest.raises(ObjectAlreadyExistsError):
        store.put("nolink/x", b"second")
    assert store.read("nolink/x") == b"first"


@pytest.mark.parametrize("key", [".agentenv-meta/x", ".agentenv-tmp/x", "a/../.agentenv-meta"])
def test_the_stores_own_directories_are_reserved(store, key):
    with pytest.raises(ValueError, match="reserved"):
        store.put(key, b"mine")


def test_listings_leave_out_the_stores_own_files(store, tmp_path):
    store.put("a/x.json", b"{}", content_type="application/json")
    (tmp_path / ".agentenv-tmp").mkdir(exist_ok=True)
    (tmp_path / ".agentenv-tmp" / "crash-residue").write_bytes(b"half")
    assert store.list("") == ["a/x.json"]
    assert store.list_at(store.object_url("")) == [store.object_url("a/x.json")]
    assert os.listdir(tmp_path / ".agentenv-tmp") == ["crash-residue"]  # committed writes leave nothing staged


def test_list_of_an_unwritten_store_is_empty(tmp_path):
    assert LocalFilesystemObjectStore(str(tmp_path / "never-written")).list("") == []
