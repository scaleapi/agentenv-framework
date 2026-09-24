"""FakeObjectStore's _at ops address by url, not by the configured home root.

This is the fast, network-free proxy for the cross-root (cross-bucket on S3)
behavior the real backends have: key-based ops target the home root, while the
url-addressed _at ops reach any root the url names.
"""

import pytest

from tst.unit.store.fakes import FakeObjectStore


def test_at_ops_address_by_url_root(tmp_path):
    store = FakeObjectStore(root="home")
    src = tmp_path / "f.bin"
    src.write_bytes(b"cross")
    foreign = "fake://other/dir/f.bin"

    url = store.put_file_at(foreign, str(src))
    assert url == foreign
    assert store.get(foreign) == b"cross"
    assert store.get_object_metadata_at(foreign) is not None
    assert store.list_at("fake://other/dir/") == [foreign]


def test_key_ops_stay_on_home_root(tmp_path):
    store = FakeObjectStore(root="home")
    src = tmp_path / "f.bin"
    src.write_bytes(b"cross")
    store.put_file_at("fake://other/dir/f.bin", str(src))

    # The foreign object is invisible to the home key-space and unaddressable by key.
    assert store.list("") == []
    with pytest.raises(ValueError):
        store.get_object_key("fake://other/dir/f.bin")
