"""The data plane on a computer: data/add writes file parts atomically under the session user's home, from inline
bytes, an https URL or a read grant, with paths checked; data/get returns the data paths, inline or uploaded."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver, http_client, rpc  # noqa: E402

from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI  # noqa: E402
from agentenv_protocol.computer import ComputerSettings  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _part(path: str, data: bytes = b"hello", **marker) -> dict:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode()},
            "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path, **marker}}}


async def _add(driver: FakeDriver, *parts: dict) -> dict:
    async with http_client(driver) as http:
        return (await http.post("/agentenv", json=rpc("data/add", {"parts": list(parts)}))).json()


@pytest.mark.asyncio
async def test_file_parts_land_under_the_home_atomically(home):
    (home / "notes").mkdir()
    (home / "notes" / "a.txt").write_text("old")
    answer = await _add(FakeDriver(), _part("notes/a.txt", b"new"), _part("deep/er/b.bin", b"\x00\x01"))
    assert answer["result"] == {}
    assert (home / "notes" / "a.txt").read_text() == "new" and (home / "deep" / "er" / "b.bin").read_bytes() == b"\x00\x01"
    assert sorted(p.name for p in (home / "notes").iterdir()) == ["a.txt"], "no temporary file is left behind"


@pytest.mark.asyncio
@pytest.mark.parametrize("part, error", [
    (_part("../outside"), "is not a normalized path relative to the home"),
    (_part("/etc/passwd"), "is not a normalized path relative to the home"),
    ({"kind": "text", "text": "hi"}, "takes file parts naming their path"),
    ({"kind": "file", "file": {"bytes": "aGk="}}, "takes file parts naming their path"),
    ({"kind": "file", "file": {"bytes": "not base64!"}, "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": "a"}}},
     "the part's bytes are not base64"),
    ({"kind": "file", "file": {"uri": "http://plain/x"}, "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": "a"}}},
     "a file part's uri is an https URL"),
    (_part("a", read={"not": "a grant"}), "the part's read grant is not a valid ReadObject"),
])
async def test_a_bad_part_is_refused_and_nothing_is_written(home, part, error):
    answer = await _add(FakeDriver(), part)
    assert answer["error"]["data"]["code"] == "add_failed" and error in answer["error"]["message"]
    assert list(home.iterdir()) == []


@pytest.mark.asyncio
async def test_a_symlink_out_of_the_home_is_refused(home, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (home / "link").symlink_to(outside)
    answer = await _add(FakeDriver(), _part("link/escaped.txt"))
    assert "'link/escaped.txt' is outside the home" in answer["error"]["message"]
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_an_https_url_part_is_fetched_by_the_driver(home):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"from the web"))
    real = httpx.AsyncClient
    with patch("agentenv_protocol.computer.dataplane.httpx.AsyncClient",
               lambda **kwargs: real(**{"transport": transport, **kwargs})):
        answer = await _add(FakeDriver(), {"kind": "file", "file": {"uri": "https://example.test/f"},
                                           "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": "web.txt"}}})
    assert answer["result"] == {} and (home / "web.txt").read_bytes() == b"from the web"


@pytest.mark.asyncio
async def test_an_https_url_over_the_limit_is_refused(home):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 64))
    real = httpx.AsyncClient
    with patch("agentenv_protocol.computer.dataplane.httpx.AsyncClient",
               lambda **kwargs: real(**{"transport": transport, **kwargs})), \
            patch("agentenv_protocol.computer.dataplane.MAX_URL_BYTES", 10):
        answer = await _add(FakeDriver(), {"kind": "file", "file": {"uri": "https://example.test/f"},
                                           "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": "big"}}})
    assert "is over 10 bytes" in answer["error"]["message"] and not (home / "big").exists()


@pytest.mark.asyncio
async def test_a_read_grant_part_is_downloaded_through_the_transfers_helper(home):
    grant = {"media_type": "text/plain", "max_bytes": 100,
             "read": {"kind": "http-get", "url": "https://store.test/o",
                      "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}}

    async def fake_download(source, destination):
        assert source.read.url == "https://store.test/o"
        destination.write_bytes(b"granted")

    with patch("agentenv_protocol.computer.dataplane.download", fake_download):
        answer = await _add(FakeDriver(), _part("granted.txt", b"ignored", read=grant))
    assert answer["result"] == {} and (home / "granted.txt").read_bytes() == b"granted"


@pytest.mark.asyncio
async def test_data_get_returns_the_data_paths_inline(home, tmp_path_factory):
    (home / "app" / "db").mkdir(parents=True)
    (home / "app" / "db" / "state.json").write_text('{"n": 1}')
    (home / "app" / "db" / "blob.bin").write_bytes(b"\x01")
    (home / "notes.md").write_text("# notes")
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret").write_text("no")
    (home / "app" / "db" / "escape").symlink_to(outside / "secret")
    driver = FakeDriver(settings=ComputerSettings(data_paths=["app", "notes.md", "missing", "app/db"]))
    async with http_client(driver) as http:
        parts = (await http.post("/agentenv", json=rpc("data/get"))).json()["result"]["parts"]
    by_path = {p["metadata"][DATA_OBJECTS_EXTENSION_URI]["path"]: p["file"] for p in parts}
    assert sorted(by_path) == ["app/db/blob.bin", "app/db/state.json", "notes.md"]
    assert base64.b64decode(by_path["app/db/state.json"]["bytes"]) == b'{"n": 1}'
    assert by_path["app/db/state.json"]["mimeType"] == "application/json" and by_path["notes.md"]["name"] == "notes.md"
    assert by_path["app/db/blob.bin"]["mimeType"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_data_get_uploads_under_a_write_namespace_grant(home):
    (home / "notes.md").write_text("# notes")
    grant = {"root_path": "runs/1", "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
             "max_objects": 5, "max_object_bytes": 100, "max_total_bytes": 500,
             "write": {"kind": "http-post-policy", "url": "https://store.test/", "fields": {},
                       "path_field": "key", "file_field": "file"}}
    upload = AsyncMock()
    with patch("agentenv_protocol.computer.dataplane.NamespaceUploader") as uploader:
        uploader.return_value.upload = upload
        driver = FakeDriver(settings=ComputerSettings(data_paths=["notes.md"]))
        async with http_client(driver) as http:
            parts = (await http.post("/agentenv", json=rpc("data/get", {"write_namespace": grant}))).json()["result"]["parts"]
    assert upload.await_args.args[0] == "notes.md" and upload.await_args.args[1] == home / "notes.md"
    assert parts == [{"kind": "file", "file": {"uri": "notes.md", "mimeType": "text/markdown", "name": "notes.md"},
                      "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": "notes.md"}}}]


@pytest.mark.asyncio
async def test_a_failed_write_leaves_no_temporary_file(home):
    with patch("agentenv_protocol.computer.linux.os.replace", side_effect=OSError("disk full")):
        answer = await _add(FakeDriver(), _part("notes/a.txt"))
    assert "disk full" in answer["error"]["message"] and list((home / "notes").iterdir()) == []
