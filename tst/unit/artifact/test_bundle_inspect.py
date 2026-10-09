"""A service bundle either ships file bytes or it does not: zip members under root/, or
inline files[] entries with content in data.json. The answer is read off the zip's central
directory, by ranged requests when the store can sign a URL, and never by downloading a
bundle whole past a bound."""

from __future__ import annotations

import gzip
import io
import json
import random
import zipfile
from unittest.mock import MagicMock

import pytest

from agent_env.artifact import bundle_inspect
from agent_env.artifact.bundle_inspect import (
    RangedReader,
    bundle_has_file_tree,
    environments_with_file_trees,
)
from agent_env.store.object_store.object_store import ObjectMetadata


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _store(payload: bytes, *, signs: bool, size: int | None = None):
    store = MagicMock()
    store.get.return_value = payload
    store.signed_get_url.return_value = "https://signed.example/bundle" if signs else None
    store.get_object_metadata_at.return_value = ObjectMetadata(size=len(payload) if size is None else size)
    return store


def _fetcher_counting(payload: bytes, calls: list[tuple[int, int]]):
    def fetch(start: int, end: int) -> bytes:
        calls.append((start, end))
        return payload[start:end + 1]
    return fetch


@pytest.fixture
def config_with(monkeypatch):
    def install(store, fetch=None):
        config = MagicMock()
        config.get_object_store_at.return_value = store
        monkeypatch.setattr(bundle_inspect, "get_config", lambda: config)
        if fetch is not None:
            monkeypatch.setattr(bundle_inspect, "_http_range_fetcher", lambda url: fetch)
        return config
    return install


INLINE = json.dumps({"directories": ["docs"], "files": [{"path": "docs/a.txt", "content": "hello"}]}).encode()
RECORDS_ONLY = json.dumps({"files": [{"id": "F1", "name": "a.pdf", "url_private": "https://x"}], "messages": []}).encode()


def test_root_members_mean_a_file_tree(config_with):
    config_with(_store(_zip({"data.json": b"{}", "root/legal/ledger.pdf": b"%PDF"}), signs=False))
    assert bundle_has_file_tree("s3://b/bundle.zip") is True


def test_data_only_bundle_has_no_file_tree(config_with):
    config_with(_store(_zip({"data.json": b"{}"}), signs=False))
    assert bundle_has_file_tree("s3://b/bundle.zip") is False


def test_an_empty_root_directory_entry_is_not_a_file_tree(config_with):
    config_with(_store(_zip({"data.json": b"{}", "root/": b""}), signs=False))
    assert bundle_has_file_tree("s3://b/bundle.zip") is False


def test_inline_files_in_a_zipped_data_json_are_a_file_tree(config_with):
    config_with(_store(_zip({"data.json": INLINE}), signs=False))
    assert bundle_has_file_tree("s3://b/bundle.zip") is True


def test_inline_files_in_a_bare_data_json_are_a_file_tree(config_with):
    config_with(_store(INLINE, signs=False))
    assert bundle_has_file_tree("s3://b/data.json") is True


def test_inline_files_in_a_gzipped_data_json_are_a_file_tree(config_with):
    config_with(_store(gzip.compress(INLINE), signs=False))
    assert bundle_has_file_tree("s3://b/data.json.gz") is True


def test_file_records_without_content_are_rows_not_files(config_with):
    config_with(_store(_zip({"data.json": RECORDS_ONLY}), signs=False))
    assert bundle_has_file_tree("s3://b/bundle.zip") is False
    config_with(_store(RECORDS_ONLY, signs=False))
    assert bundle_has_file_tree("s3://b/data.json") is False


def test_a_bundle_that_is_not_json_or_zip_has_no_file_tree(config_with):
    config_with(_store(b"not json at all", signs=False))
    assert bundle_has_file_tree("s3://b/data.json") is False


def test_a_huge_data_json_is_not_scanned_for_inline_files(config_with, monkeypatch):
    monkeypatch.setattr(bundle_inspect, "_INLINE_FILES_SCAN_MAX_BYTES", 10)
    config_with(_store(_zip({"data.json": INLINE}), signs=False))
    assert bundle_has_file_tree("s3://b/bundle.zip") is False
    config_with(_store(INLINE, signs=False))
    assert bundle_has_file_tree("s3://b/data.json") is False


def test_signed_url_reads_only_the_tail_and_central_directory(config_with):
    rng = random.Random(7)
    payload = _zip({"data.json": b"{}", **{f"root/f{i}.pdf": rng.randbytes(100_000) for i in range(50)}})
    calls: list[tuple[int, int]] = []
    store = _store(payload, signs=True)
    config_with(store, fetch=_fetcher_counting(payload, calls))

    assert bundle_has_file_tree("s3://b/bundle.zip") is True

    store.get.assert_not_called()
    fetched = sum(end - start + 1 for start, end in calls)
    # Incompressible members, so the bytes fetched are the end record, the central
    # directory and the read-ahead around them, none of the file contents.
    assert fetched < len(payload) / 3, (fetched, len(payload))
    assert all(start >= 0 and end < len(payload) for start, end in calls)


def test_signed_url_reads_a_data_json_member_in_few_requests(config_with):
    rng = random.Random(3)
    big_inline = json.dumps({"files": [{"path": f"f{i}", "content": rng.randbytes(2_000).hex()} for i in range(1_000)]}).encode()
    payload = _zip({"data.json": big_inline})
    calls: list[tuple[int, int]] = []
    config_with(_store(payload, signs=True), fetch=_fetcher_counting(payload, calls))

    assert bundle_has_file_tree("s3://b/bundle.zip") is True
    assert len(calls) <= 8, calls


def test_whole_object_fallback_when_the_store_cannot_sign(config_with):
    payload = _zip({"data.json": b"{}", "root/a.pdf": b"x"})
    store = _store(payload, signs=False)
    config_with(store)
    assert bundle_has_file_tree("s3://b/bundle.zip") is True
    store.get.assert_called_once_with("s3://b/bundle.zip")


def test_whole_object_fallback_refuses_a_bundle_past_the_bound(config_with, monkeypatch):
    monkeypatch.setattr(bundle_inspect, "_WHOLE_OBJECT_MAX_BYTES", 100)
    store = _store(b"x" * 101, signs=False)
    config_with(store)
    with pytest.raises(RuntimeError, match="cannot sign ranged reads"):
        bundle_has_file_tree("s3://b/bundle.zip")
    store.get.assert_not_called()


def test_environments_with_file_trees_keeps_universe_order(config_with):
    with_tree = _zip({"data.json": b"{}", "root/x.pdf": b"x"})
    without = _zip({"data.json": b"{}"})
    payloads = {"s3://b/gmail.zip": with_tree, "s3://b/linear.zip": without, "s3://b/gdrive.zip": with_tree, "s3://b/fs.json": INLINE}
    store = MagicMock()
    store.signed_get_url.return_value = None
    store.get.side_effect = lambda url: payloads[url]
    store.get_object_metadata_at.side_effect = lambda url: ObjectMetadata(size=len(payloads[url]))
    config_with(store)

    def env_artifact(name, url):
        ea = MagicMock()
        ea.environment_name = name
        ea.get_file_artifact.return_value.object_url = url
        return ea

    universe = MagicMock()
    universe.get_environment_artifacts.return_value = [
        env_artifact("gmail", "s3://b/gmail.zip"),
        env_artifact("linear", "s3://b/linear.zip"),
        env_artifact("gdrive", "s3://b/gdrive.zip"),
        env_artifact("filesystem", "s3://b/fs.json"),
    ]
    assert environments_with_file_trees(universe) == ["gmail", "gdrive", "filesystem"]


def test_ranged_reader_behaves_like_a_file_and_coalesces_reads():
    payload = bytes(range(256)) * 4
    calls: list[tuple[int, int]] = []
    reader = RangedReader(len(payload), _fetcher_counting(payload, calls), read_ahead=64)

    assert reader.seek(0, io.SEEK_END) == len(payload)
    assert reader.read() == b""
    assert reader.seek(-22, io.SEEK_END) == len(payload) - 22
    assert reader.read() == payload[-22:]
    assert reader.seek(10) == 10
    assert reader.read(5) == payload[10:15]
    assert reader.tell() == 15
    assert reader.read(5) == payload[15:20]  # served from the read-ahead window
    assert reader.seek(-5, io.SEEK_CUR) == 15
    buffer = bytearray(3)
    assert reader.readinto(buffer) == 3 and bytes(buffer) == payload[15:18]
    assert reader.read(10**6) == payload[18:]
    assert calls == [(len(payload) - 22, len(payload) - 1), (10, 73), (18, len(payload) - 1)]


def test_http_fetcher_refuses_a_response_that_ignored_the_range(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def read(self, *a):
            raise AssertionError("the body must not be read")

    monkeypatch.setattr(bundle_inspect.urllib.request, "urlopen", lambda req, timeout: Response(b"0123456789"))
    with pytest.raises(RuntimeError, match="ranged read refused: HTTP 200"):
        bundle_inspect._http_range_fetcher("https://x")(2, 4)


def test_http_fetcher_returns_a_partial_body(monkeypatch):
    class Response(io.BytesIO):
        status = 206
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bundle_inspect.urllib.request, "urlopen", lambda req, timeout: Response(b"234"))
    assert bundle_inspect._http_range_fetcher("https://x")(2, 4) == b"234"

def test_one_unreadable_bundle_fails_the_whole_answer(config_with):
    store = MagicMock()
    store.signed_get_url.return_value = None
    store.get_object_metadata_at.return_value = ObjectMetadata(size=10)
    store.get.side_effect = lambda url: (_ for _ in ()).throw(OSError("object store unreachable")) if "bad" in url else INLINE
    config_with(store)

    def env_artifact(name, url):
        ea = MagicMock()
        ea.environment_name = name
        ea.get_file_artifact.return_value.object_url = url
        return ea

    universe = MagicMock()
    universe.get_environment_artifacts.return_value = [env_artifact("ok", "s3://b/ok.json"), env_artifact("bad", "s3://b/bad.json")]
    with pytest.raises(OSError, match="object store unreachable"):
        environments_with_file_trees(universe)
