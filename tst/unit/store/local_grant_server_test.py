"""The local store's grants against its real grant server, over HTTPS on loopback: the conformance
grant cases, and the bounds and refusals S3 would apply to the same requests."""

import os
import subprocess
import sys
import threading
import time

import httpx
import pytest
from pytest_socket import enable_socket

from agent_env.a2a_agent import object_transfer
from agent_env.a2a_agent.object_transfer import TransferCall, invoke_transfer
from agent_env.config.paths import state_root
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.object_store.local import grant_server as server_module
from agent_env.store.object_store.local.grant_server import default_bind_host, grant_server, unreachable_hint
from agent_env.store.object_store.local.tls import local_ca
from tst.store import object_conformance


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A local store granting on loopback under this test's state root, trusted as a container would trust it."""
    monkeypatch.setattr(server_module, "_servers", {})
    monkeypatch.setenv("SSL_CERT_FILE", str(local_ca().bundle_path))
    yield LocalFilesystemObjectStore(str(tmp_path / "objects"), grant_bind_host="127.0.0.1", grant_advertise_host="localhost")
    for server in server_module._servers.values():
        server.close()


def _policy(store, prefix="ns/", max_object_bytes=16):
    return store.issue_upload_policy(store.object_url(prefix), max_object_bytes=max_object_bytes, expires_in=600).write


def _post(write, fields, data=b"data", *, file_first=False):
    files = {write.file_field: ("object", data, "application/octet-stream")}
    if file_first:  # httpx sends data fields before files, so build the body by hand
        request = httpx.Request("POST", write.url, files=files)
        body = request.read()
        boundary = request.headers["content-type"].split("boundary=")[1]
        extra = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()
        )
        body = body.replace(f"--{boundary}--".encode(), extra + f"--{boundary}--".encode())
        return httpx.post(write.url, content=body, headers={"Content-Type": request.headers["content-type"]})
    return httpx.post(write.url, data=fields, files=files)


@pytest.mark.parametrize("case", object_conformance.GRANT_CASES, ids=lambda c: c.__name__)
def test_grant_conformance(case, store):
    case(store, "")


def test_a_read_returns_the_exact_bytes_and_type(store):
    url = store.put("a/page.html", b"<p>hi</p>", content_type="application/json")
    response = httpx.get(store.issue_read_grant(url).url, headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.content == b"<p>hi</p>"
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-length"] == "9"
    assert "content-encoding" not in response.headers


@pytest.mark.skipif(os.name == "nt", reason="holds the key's lock with flock, which Windows lacks")
def test_a_download_waits_for_a_write_to_its_key(store):
    """Bytes and type are read from one file under the key's lock, so a download never mixes two writes."""
    grant = store.issue_read_grant(store.put("a/held", b"typed", content_type="text/plain")).url
    holder = subprocess.Popen(
        [sys.executable, "-c", (
            "import fcntl, sys, time; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); "
            "print('locked', flush=True); time.sleep(60)"
        ), str(store._lock_path((store.root / "a/held").resolve()))],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        got = []
        download = threading.Thread(target=lambda: got.append(httpx.get(grant, timeout=10)))
        download.start()
        download.join(0.3)
        assert download.is_alive()
        holder.kill()
        holder.wait()
        download.join(10)
    finally:
        holder.kill()
    assert (got[0].status_code, got[0].content, got[0].headers["content-type"]) == (200, b"typed", "text/plain; charset=utf-8")


def test_a_read_of_a_missing_object_is_404(store):
    assert httpx.get(store.issue_read_grant(store.object_url("a/none.bin")).url).status_code == 404


def test_a_head_reports_the_size(store):
    url = store.put("a/b.bin", b"12345")
    response = httpx.head(store.issue_read_grant(url).url)
    assert response.status_code == 200
    assert response.headers["content-length"] == "5"


def test_a_grant_allows_only_its_own_method(store):
    url = store.put("a/b.bin", b"v")
    read = store.issue_read_grant(url).url
    assert httpx.put(read, content=b"x").status_code == 403
    assert httpx.post(read, content=b"x").status_code == 403
    assert httpx.delete(read).status_code == 405
    assert store.get(url) == b"v"


def test_a_tampered_or_expired_grant_is_403(store):
    url = store.put("a/b.bin", b"v")
    read = store.issue_read_grant(url).url
    assert httpx.get(read[:-3] + "AAA").status_code == 403
    assert httpx.get(read.rsplit("/", 1)[0] + "/not-a-token").status_code == 403
    server = grant_server("127.0.0.1", "localhost")
    expired = server.issue(store, "get", int(time.time()) - 1, key="a/b.bin")
    assert httpx.get(expired).status_code == 403


def test_a_grant_from_a_closed_server_stops_working(store):
    url = store.put("a/b.bin", b"v")
    read = store.issue_read_grant(url).url
    grant_server("127.0.0.1", "localhost").close()
    with pytest.raises(httpx.ConnectError):
        httpx.get(read)
    assert httpx.get(store.issue_read_grant(url).url).content == b"v"  # the next grant starts it again


def test_a_write_must_declare_the_signed_type(store):
    url = store.object_url("a/w.json")
    grant = store.issue_write_grant(url, media_type="application/json", max_bytes=64)
    assert httpx.put(grant.url, content=b"{}", headers={"Content-Type": "text/plain"}).status_code == 403
    assert httpx.put(grant.url, content=b"{}").status_code == 403
    assert not store.exists("a/w.json")


def test_a_write_larger_than_the_grant_is_refused(store):
    url = store.object_url("a/w.bin")
    grant = store.issue_write_grant(url, media_type="application/octet-stream", max_bytes=4)
    assert httpx.put(grant.url, content=b"12345", headers=grant.headers).status_code == 400

    def chunks():  # no Content-Length: the bound is enforced as the body streams
        yield b"123"
        yield b"45"

    assert httpx.put(grant.url, content=chunks(), headers=grant.headers).status_code == 400
    assert not store.exists("a/w.bin")
    assert not any((store.root / ".agentenv-tmp").iterdir())


def test_a_write_grant_may_be_retried_and_overwrites(store):
    url = store.object_url("a/w.bin")
    grant = store.issue_write_grant(url, media_type="application/octet-stream", max_bytes=8)
    assert httpx.put(grant.url, content=b"first", headers=grant.headers).status_code == 200
    assert httpx.put(grant.url, content=b"second", headers=grant.headers).status_code == 200
    assert store.get(url) == b"second"


def test_an_upload_keeps_its_content_type(store):
    write = _policy(store)
    assert _post(write, {"key": "ns/x.txt", "Content-Type": "text/plain"}, b"hello").status_code == 204
    assert store.read("ns/x.txt") == b"hello"
    assert store.get_object_metadata("ns/x.txt").content_type == "text/plain"


def test_an_upload_may_nest_and_overwrite(store):
    write = _policy(store)
    assert _post(write, {"key": "ns/a/b/c"}, b"one").status_code == 204
    assert _post(write, {"key": "ns/a/b/c"}, b"two").status_code == 204
    assert store.read("ns/a/b/c") == b"two"


@pytest.mark.parametrize("key", ["ns/../x", "ns//x", "ns/./x", "/ns/x", "ns\\x", "ns-sibling/x", "other/x", "ns"])
def test_an_upload_key_outside_the_prefix_is_403(store, key):
    assert _post(_policy(store), {"key": key}).status_code == 403
    assert store.list("") == []


def test_an_upload_at_the_root_cannot_touch_the_stores_own_files(store):
    write = _policy(store, prefix="")
    assert _post(write, {"key": ".agentenv-meta/x"}).status_code == 403
    assert _post(write, {"key": "top.bin"}).status_code == 204
    assert store.read("top.bin") == b"data"


def test_an_upload_larger_than_the_grant_is_refused(store):
    write = _policy(store, max_object_bytes=4)
    assert _post(write, {"key": "ns/big"}, b"12345").status_code == 400
    assert not store.exists("ns/big")
    assert not any((store.root / ".agentenv-tmp").iterdir())


def test_an_upload_with_oversized_part_headers_is_refused(store):
    write = _policy(store)
    body = (
        b"--b\r\nContent-Disposition: form-data; name=\"key\"\r\nX-Padding: " + b"x" * 20_000
        + b"\r\n\r\nns/x\r\n--b--\r\n"
    )
    response = httpx.post(write.url, content=body, headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert response.status_code == 400
    assert not store.exists("ns/x")


def test_an_upload_must_name_its_key_before_its_file(store):
    write = _policy(store)
    assert _post(write, {"key": "ns/late"}, file_first=True).status_code == 400
    assert not store.exists("ns/late")


def test_fields_after_the_file_are_ignored(store):
    write = _policy(store)
    request = httpx.Request("POST", write.url, data={"key": "ns/second"}, files={"file": ("f", b"v", "text/plain")})
    boundary = request.headers["content-type"].split("boundary=")[1]
    body = request.read().replace(
        f"--{boundary}--".encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="key"\r\n\r\nns/../escape\r\n--{boundary}--'.encode(),
    )
    assert httpx.post(write.url, content=body, headers={"Content-Type": request.headers["content-type"]}).status_code == 204
    assert store.read("ns/second") == b"v"


@pytest.mark.parametrize("body, content_type", [
    (b"key=ns/x", "application/x-www-form-urlencoded"),
    (b"--b\r\nContent-Disposition: form-data; name=\"key\"\r\n\r\nns/x\r\n--b--\r\n", "multipart/form-data; boundary=b"),
])
def test_an_upload_that_is_not_a_form_with_a_file_is_400(store, body, content_type):
    write = _policy(store)
    assert httpx.post(write.url, content=body, headers={"Content-Type": content_type}).status_code == 400


def test_grants_refuse_the_stores_own_files(store):
    with pytest.raises(ValueError, match="reserved"):
        store.issue_read_grant(store.object_url(".agentenv-meta/a"))


def test_local_grants_last_the_stores_lifetime(store, tmp_path):
    def lasts(grant, seconds):
        return abs(grant.expires_at.timestamp() - time.time() - seconds) < 2

    assert lasts(store.issue_read_grant(store.put("k", b"v")), 12 * 60 * 60)
    short = LocalFilesystemObjectStore(
        str(tmp_path / "short"), grant_bind_host="127.0.0.1", grant_advertise_host="localhost", grant_lifetime_seconds=900
    )
    grant = short.issue_read_grant(short.put("k", b"v"))
    assert lasts(grant, 900) and httpx.get(grant.url).content == b"v"


def test_a_grant_expires_when_asked(store):
    url = store.put("a/b.bin", b"v")
    grant = store.issue_read_grant(url, expires_in=120)
    assert 118 <= (grant.expires_at.timestamp() - time.time()) <= 121
    with pytest.raises(ValueError):
        store.issue_read_grant(url, expires_in=0)


def test_two_stores_share_one_server_and_keep_their_own_objects(store, tmp_path):
    other = LocalFilesystemObjectStore(str(tmp_path / "other"), grant_bind_host="127.0.0.1", grant_advertise_host="localhost")
    a, b = store.put("k", b"mine"), other.put("k", b"theirs")
    ga, gb = store.issue_read_grant(a).url, other.issue_read_grant(b).url
    assert ga.split("/v1/")[0] == gb.split("/v1/")[0]
    assert (httpx.get(ga).content, httpx.get(gb).content) == (b"mine", b"theirs")


def test_grant_urls_name_the_advertised_host(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_servers", {})
    store = LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="127.0.0.1")
    try:
        assert store.issue_read_grant(store.put("k", b"v")).url.startswith("https://host.docker.internal:")
    finally:
        for server in server_module._servers.values():
            server.close()


def test_settings_are_checked_when_the_store_is_made(tmp_path):
    with pytest.raises(ValueError, match="IP address"):
        LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="localhost")
    with pytest.raises(ValueError, match="not a local name"):
        LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="example.com")
    with pytest.raises(ValueError, match="private address"):
        LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="8.8.8.8")


def test_the_root_defaults_to_the_state_root():
    assert LocalFilesystemObjectStore().root == state_root() / "object_store"


class TestDefaultBindHost:
    @pytest.fixture(autouse=True)
    def _fresh(self):
        default_bind_host.cache_clear()
        yield
        default_bind_host.cache_clear()

    def test_loopback_off_linux(self, monkeypatch):
        monkeypatch.setattr(server_module.platform, "system", lambda: "Darwin")
        assert default_bind_host() == "127.0.0.1"

    def test_the_bridge_gateway_on_linux(self, monkeypatch):
        monkeypatch.setattr(server_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            server_module.subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="172.17.0.1 \n", stderr=""),
        )
        assert default_bind_host() == "172.17.0.1"

    def test_loopback_with_a_warning_when_docker_cannot_say(self, monkeypatch, caplog):
        monkeypatch.setattr(server_module.platform, "system", lambda: "Linux")

        def fail(*a, **k):
            raise FileNotFoundError("docker")

        monkeypatch.setattr(server_module.subprocess, "run", fail)
        assert default_bind_host() == "127.0.0.1"
        assert "grant_bind_host" in caplog.text


def test_the_unreachable_hint_names_the_server_and_never_a_grant(store):
    grant = store.issue_read_grant(store.put("k", b"v"))
    hint = unreachable_hint({"objects": {"trajectory": {"read": grant.model_dump(mode="json")}}})
    assert grant.url.split("/v1/")[0] in hint
    assert "listens on 127.0.0.1" in hint and "grant_bind_host" in hint
    assert grant.url.rsplit("/", 1)[1] not in hint
    assert unreachable_hint({"objects": {"read": {"url": "https://bucket.example/obj"}}}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("code, hinted", [("transfer_unavailable", True), ("transfer_rejected", False)])
async def test_an_agent_that_could_not_reach_the_grant_server_is_told_where_to_look(store, monkeypatch, code, hinted):
    grant = store.issue_read_grant(store.put("k", b"v"))
    answer = httpx.Response(502, json={"error": {"code": code, "message": "m", "retryable": True}})
    client = httpx.AsyncClient
    monkeypatch.setattr(
        object_transfer.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(lambda _: answer))
    )
    call = TransferCall("objects", {"objects": {"read": grant.model_dump(mode="json")}})
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await invoke_transfer("http://agent.test/ext", call, verb="POST", operation="snapshot load", timeout=5)
    assert str(raised.value).startswith(f"snapshot load failed with HTTP 502: {code}")
    assert ("local grant server" in str(raised.value)) is hinted
