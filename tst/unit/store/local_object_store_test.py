"""LocalFilesystemObjectStore runs the full ObjectStore conformance suite.

Fast tier — no S3, no network (stdlib filesystem over a temp dir), so the same
assertions that prove S3 parity also give quick backend-neutral coverage.
"""

import errno
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent_env.store import LocalFilesystemObjectStore, ObjectAlreadyExistsError
from agent_env.store.object_store.local import store as local_store
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


def test_without_hard_links_a_failed_write_once_put_leaves_no_object(store, monkeypatch):
    def no_links(src, dst):
        raise OSError(errno.EPERM, "hard links not supported")

    def failing_replace(src, dst):
        raise OSError(errno.EIO, "disk went away")

    monkeypatch.setattr(os, "link", no_links)
    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk went away"):
        store.put("nolink/y", b"never")
    assert not store.exists("nolink/y")


def test_without_hard_links_a_racing_write_once_put_keeps_the_winner(store, monkeypatch):
    def no_links(src, dst):
        raise OSError(errno.EPERM, "hard links not supported")

    real_replace = os.replace
    first_holds_its_claim = threading.Event()

    def slow_first_replace(src, dst):
        if threading.current_thread().name == "first" and str(dst).endswith("race/once"):
            first_holds_its_claim.set()
            time.sleep(0.3)
        real_replace(src, dst)

    monkeypatch.setattr(os, "link", no_links)
    monkeypatch.setattr(os, "replace", slow_first_replace)
    first = threading.Thread(target=store.put, args=("race/once", b"first"), name="first")
    first.start()
    first_holds_its_claim.wait(5)
    with pytest.raises(ObjectAlreadyExistsError):
        store.put("race/once", b"second")
    first.join()
    assert store.read("race/once") == b"first"


_needs_flock = pytest.mark.skipif(os.name == "nt", reason="holds a lock with flock, which Windows lacks")


def _hold_lock(lock_file):
    """A process holding ``lock_file`` exclusively until killed, as a writer mid-commit would."""
    holder = subprocess.Popen(
        [sys.executable, "-c", (
            "import fcntl, sys, time; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); "
            "print('locked', flush=True); time.sleep(60)"
        ), str(lock_file)],
        stdout=subprocess.PIPE, text=True,
    )
    assert holder.stdout.readline().strip() == "locked"
    return holder


@pytest.fixture
def no_hard_links(store, monkeypatch):
    def no_links(src, dst):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(os, "link", no_links)
    store.put("seed", b"s")  # creates the staging directory


@_needs_flock
def test_without_hard_links_a_claim_dies_with_its_holder(store, tmp_path, no_hard_links):
    """The claim is a lock the kernel drops when its holder exits, so a crashed writer never blocks the key."""
    holder = _hold_lock(store._lock_path((tmp_path / "left/x").resolve()))
    try:
        writer = threading.Thread(target=store.put, args=("left/x", b"mine"))
        writer.start()
        writer.join(0.3)
        assert writer.is_alive() and not (tmp_path / "left/x").exists()  # a live holder is waited for
        holder.kill()
        holder.wait()
        writer.join(5)
    finally:
        holder.kill()
    assert store.read("left/x") == b"mine"


def test_types_survive_a_copy_that_keeps_modification_times(store, tmp_path):
    store.put("copied/x", b"typed", content_type="text/plain")
    copy = tmp_path.parent / f"{tmp_path.name}-copy"
    shutil.copytree(tmp_path, copy)  # copies modification times, not inodes
    assert LocalFilesystemObjectStore(str(copy)).get_object_metadata("copied/x").content_type == "text/plain"


def test_a_type_recorded_for_another_write_reads_back_as_unknown(store, tmp_path):
    """Two overwrites of one key race: the one that renamed first records its type last. Its type must not
    be read as the other's."""
    store.put("race/x", b"first", content_type="text/plain")
    first_meta = (tmp_path / ".agentenv-meta" / "race" / "x").read_text()
    store.put("race/x", b"second!", content_type="application/json", allow_overwrite=True)
    (tmp_path / ".agentenv-meta" / "race" / "x").write_text(first_meta)
    metadata = store.get_object_metadata("race/x")
    assert (metadata.content_type, metadata.size) == (None, 7)


def test_an_overwrite_that_names_no_type_never_inherits_the_old_one(store, tmp_path, monkeypatch):
    store.put("keep/x", b"typed", content_type="text/plain")
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)  # the old record survives
    store.put("keep/x", b"untyped!", allow_overwrite=True)
    assert store.get_object_metadata("keep/x").content_type is None


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
    staging = [name for name in os.listdir(tmp_path / ".agentenv-tmp") if not name.startswith("lock-")]
    assert staging == ["crash-residue"]  # committed writes leave nothing staged


@_needs_flock
def test_a_type_is_never_read_while_a_write_to_the_key_is_in_progress(store, tmp_path):
    store.put("busy/x", b"typed", content_type="text/plain")
    holder = _hold_lock(store._lock_path((tmp_path / "busy/x").resolve()))
    try:
        read = []
        reader = threading.Thread(target=lambda: read.append(store.get_object_metadata("busy/x")))
        reader.start()
        reader.join(0.3)
        assert reader.is_alive()  # waits for the write holding the key
        holder.kill()
        holder.wait()
        reader.join(5)
    finally:
        holder.kill()
    assert read[0].content_type == "text/plain"


def _failing_replace(monkeypatch, suffix):
    real_replace = os.replace

    def replace(src, dst):
        if str(dst).endswith(suffix):
            raise OSError(errno.EIO, "disk went away")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)


def test_an_overwrite_whose_bytes_do_not_land_keeps_the_old_type(store, tmp_path, monkeypatch):
    store.put("cut/x", b"first", content_type="text/plain")
    _failing_replace(monkeypatch, str(tmp_path / "cut/x"))
    with pytest.raises(OSError, match="disk went away"):
        store.put("cut/x", b"second", content_type="application/json", allow_overwrite=True)
    assert (store.read("cut/x"), store.get_object_metadata("cut/x").content_type) == (b"first", "text/plain")


def test_an_overwrite_whose_cleanup_fails_still_succeeds(store, tmp_path, monkeypatch):
    store.put("clean/x", b"first", content_type="text/plain")
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if "aside-" in self.name:
            raise OSError(errno.EIO, "disk went away")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    store.put("clean/x", b"second!", content_type="application/json", allow_overwrite=True)
    assert (store.read("clean/x"), store.get_object_metadata("clean/x").content_type) == (b"second!", "application/json")


@_needs_flock
def test_a_write_that_cannot_lock_its_staged_file_leaves_nothing_staged(store, tmp_path, monkeypatch):
    def no_lock(fd, op):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(local_store.fcntl, "flock", no_lock)
    with pytest.raises(OSError, match="no locks"):
        store.put("nolock/x", b"v")
    assert [n for n in os.listdir(tmp_path / ".agentenv-tmp") if n.startswith("staged-")] == []


def test_an_overwrite_whose_type_cannot_be_recorded_reads_unknown_not_the_old_type(store, tmp_path, monkeypatch):
    store.put("cut/y", b"first", content_type="text/plain")
    _failing_replace(monkeypatch, str(tmp_path / ".agentenv-meta" / "cut/y"))
    store.put("cut/y", b"second!", content_type="application/json", allow_overwrite=True)
    assert (store.read("cut/y"), store.get_object_metadata("cut/y").content_type) == (b"second!", None)


@_needs_flock
def test_a_store_sweeps_only_files_no_live_write_holds(tmp_path):
    staging = tmp_path / ".agentenv-tmp"
    staging.mkdir()
    hour_ago = (time.time() - 3600, time.time() - 3600)
    dead, paused, fresh = staging / "staged-dead", staging / "staged-paused", staging / "staged-fresh"
    for f in (dead, paused, fresh, staging / "lock-00"):
        f.write_bytes(b"half")
    for f in (dead, paused, staging / "lock-00"):
        os.utime(f, hour_ago)
    holder = _hold_lock(paused)  # a live write paused for an hour
    try:
        LocalFilesystemObjectStore(str(tmp_path)).put("k", b"v")
    finally:
        holder.kill()
    assert not dead.exists()
    assert paused.exists() and fresh.exists() and (staging / "lock-00").exists()


def test_list_of_an_unwritten_store_is_empty(tmp_path):
    assert LocalFilesystemObjectStore(str(tmp_path / "never-written")).list("") == []
