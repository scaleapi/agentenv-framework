"""``tst.util.package_cache``: the computer tests' stores share their package locks and files through one folder."""

from __future__ import annotations

import errno
import hashlib

import pytest

from agent_env.config import configure, get_config, reset_config
from agent_env.env.envs.computer.utils.packages import PREFIX, Lock, LockedDeb, LockedDownload, LockedFile
from tst.util import package_cache

_INPUTS = "a" * 64


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """A local object store under this test's folder and the shared folder; the store's root."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv(package_cache.ENV_VAR, str(tmp_path / "shared"))
    configure()
    yield get_config().get_object_store()
    reset_config()


def test_a_lock_a_compose_stored_is_shared_with_its_files_and_seeds_the_next_store(stores, tmp_path, monkeypatch):
    files = _store_lock(stores, {"apt.deb": b"deb", "wheel.whl": b"wheel", "chrome.zip": b"zip"})

    package_cache.keep()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "next"))
    reset_config()
    configure()
    package_cache.seed()

    seeded = get_config().get_object_store()
    assert seeded.root != stores.root
    assert Lock.model_validate_json(seeded.read(_lock_key())).inputs == _INPUTS
    assert seeded.get_object_metadata(_lock_key()).content_type == "application/json"
    assert {sha256: seeded.read(f"{PREFIX}/files/{sha256}") for sha256 in files} == files


def test_where_folders_cant_share_a_hard_link_a_copied_lock_keeps_its_content_type(stores, tmp_path, monkeypatch):
    """The store names an object's content type by the object's size and modification time, so a copy keeps both."""
    files = _store_lock(stores, {"apt.deb": b"deb"})

    def cross_device(*_args):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    with monkeypatch.context() as across:
        across.setattr(package_cache.os, "link", cross_device)
        package_cache.keep()
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "next"))
        reset_config()
        configure()
        package_cache.seed()

    seeded = get_config().get_object_store()
    assert seeded.get_object_metadata(_lock_key()).content_type == "application/json"
    assert {sha256: seeded.read(f"{PREFIX}/files/{sha256}") for sha256 in files} == files
    assert not (tmp_path / "shared" / PREFIX / "files" / next(iter(files))).samefile(
        stores.root / PREFIX / "files" / next(iter(files)))


def test_a_lock_whose_files_arent_all_shared_isnt_seeded(stores, tmp_path, monkeypatch):
    files = _store_lock(stores, {"apt.deb": b"deb", "wheel.whl": b"wheel"})
    package_cache.keep()
    (tmp_path / "shared" / PREFIX / "files" / next(iter(files))).unlink()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "next"))
    reset_config()
    configure()

    package_cache.seed()

    assert not get_config().get_object_store().exists(_lock_key())


def test_keeping_never_replaces_what_is_shared(stores, tmp_path):
    files = _store_lock(stores, {"apt.deb": b"deb"})
    (sha256,) = files
    shared = tmp_path / "shared" / PREFIX / "files" / sha256
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"already shared")

    package_cache.keep()

    assert shared.read_bytes() == b"already shared"


def test_without_a_shared_folder_nothing_is_shared(stores, tmp_path, monkeypatch):
    monkeypatch.delenv(package_cache.ENV_VAR)
    _store_lock(stores, {"apt.deb": b"deb"})

    package_cache.keep()
    package_cache.seed()

    assert not (tmp_path / "shared").exists()


def _store_lock(store, named: dict[str, bytes]) -> dict[str, bytes]:
    """A lock of ``named``'s files, stored with them as a compose stores them; the files by their sha256."""
    files = {hashlib.sha256(data).hexdigest(): data for data in named.values()}
    sha = {name: hashlib.sha256(data).hexdigest() for name, data in named.items()}
    for sha256, data in files.items():
        store.put(f"{PREFIX}/files/{sha256}", data)
    debs = [LockedDeb(file=name, sha256=sha[name], size=len(named[name]), package="p", version="1", arch="amd64")
            for name in named if name.endswith(".deb")]
    wheels = [LockedFile(file=name, sha256=sha[name], size=len(named[name])) for name in named if name.endswith(".whl")]
    downloads = [LockedDownload(file=name, sha256=sha[name], size=len(named[name]), url=f"https://example.com/{name}",
                                to=name) for name in named if name.endswith(".zip")]
    lock = Lock(inputs=_INPUTS, arch="amd64", apt=debs, wheels={"driver": wheels} if wheels else {},
                downloads={"app": downloads} if downloads else {})
    store.put(_lock_key(), lock.model_dump_json().encode(), content_type="application/json")
    return files


def _lock_key() -> str:
    return f"{PREFIX}/locks/{_INPUTS}.json"
