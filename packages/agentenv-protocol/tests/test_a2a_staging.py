from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import transfers
from agentenv_protocol.a2a_agent import (
    STAGING_V1_URI,
    AgentEnvAgent,
    AgentIdentity,
    StagingStore,
    TaskRequest,
    TaskResult,
    a2a_agent,
    custom_extension,
    staging_routes,
)
from agentenv_protocol.transfers import (
    HttpGetGrant,
    HttpPostPolicyGrant,
    HttpPutGrant,
    NamespaceUploader,
    ReadObject,
    WriteNamespaceGrant,
    WriteObject,
    download,
    upload,
)
from starlette.applications import Starlette
from starlette.testclient import TestClient

_UTC = timezone.utc  # noqa: UP017 -- package supports Python 3.10.
SESSION = "s" * 32
BASE = f"/ext/staging/{SESSION}"


@a2a_agent(identity=AgentIdentity(name="staging", description="test", version="1"))
class Agent(AgentEnvAgent):
    async def run(self, request: TaskRequest) -> TaskResult:
        return TaskResult.text("ok")


@pytest.fixture(autouse=True)
def _staging_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_STAGING_DIR", str(tmp_path / "staging"))


def _app(max_bytes: int = 1024) -> Starlette:
    return Starlette(routes=staging_routes(StagingStore(max_bytes=max_bytes)))


def _expiry() -> datetime:
    return datetime.now(_UTC) + timedelta(minutes=5)


def test_an_sdk_agent_advertises_staging_and_serves_it() -> None:
    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
        stored = client.put(f"{BASE}/a/b.txt", content=b"hello")
        fetched = client.get(f"{BASE}/a/b.txt")
        listed = client.get(f"{BASE}/")
        deleted = client.delete(f"{BASE}/a/b.txt")
        gone = client.get(f"{BASE}/a/b.txt")

    [staging] = [e for e in card["capabilities"]["extensions"] if e["uri"] == STAGING_V1_URI]
    assert staging["params"]["endpoint"] == "/ext/staging"
    assert stored.status_code == 201
    assert stored.json()["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert (fetched.content, fetched.headers["etag"]) == (b"hello", stored.json()["etag"])
    assert listed.json() == {
        "objects": [{"path": "a/b.txt", "size_bytes": 5, "etag": stored.json()["etag"]}]
    }
    assert (deleted.status_code, gone.status_code) == (204, 404)


def test_a_limit_of_zero_turns_staging_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_STAGING_MAX_BYTES", "0")
    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
        stored = client.put(f"{BASE}/a", content=b"x")
    assert STAGING_V1_URI not in {e["uri"] for e in card["capabilities"]["extensions"]}
    assert stored.status_code == 404


def test_staging_off_leaves_its_paths_to_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    @a2a_agent(identity=AgentIdentity(name="own-route", description="test", version="1"))
    class OwnRoute(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @custom_extension(uri="urn:example:own/v1", operation="read", method="GET", path="/ext/staging/own")
        async def read(self):
            return {"status": "ok"}

    with pytest.raises(ValueError, match="conflicts with the staging routes"):
        OwnRoute().create_app()
    monkeypatch.setenv("AGENTENV_STAGING_MAX_BYTES", "0")
    with TestClient(OwnRoute().create_app()) as client:
        assert client.get("/ext/staging/own").json() == {"status": "ok"}


def test_staged_objects_are_the_server_users_alone(tmp_path) -> None:
    with TestClient(_app()) as client:
        client.put(f"{BASE}/a", content=b"x")
    for directory in ("objects", "incoming"):
        assert (tmp_path / "staging" / directory).stat().st_mode & 0o777 == 0o700


def test_staging_tightens_its_own_directories_left_open(tmp_path) -> None:
    for directory in ("objects", "incoming"):
        (tmp_path / "staging" / directory).mkdir(parents=True)
        (tmp_path / "staging" / directory).chmod(0o755)
    with TestClient(_app()) as client:
        client.put(f"{BASE}/a", content=b"x")
    for directory in ("objects", "incoming"):
        assert (tmp_path / "staging" / directory).stat().st_mode & 0o777 == 0o700


def test_without_a_directory_each_server_stages_in_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTENV_STAGING_DIR")
    first, second = StagingStore(max_bytes=4), StagingStore(max_bytes=4)
    with TestClient(Starlette(routes=staging_routes(first))) as client:
        assert client.put(f"{BASE}/a", content=b"1234").status_code == 201
    with TestClient(Starlette(routes=staging_routes(second))) as client:  # what the first left is not counted
        assert client.put(f"{BASE}/a", content=b"1234").status_code == 201
        assert client.get(f"{BASE}/a").content == b"1234"
    assert first.root != second.root and first.root.stat().st_mode & 0o777 == 0o700
    shutil.rmtree(first.root)
    shutil.rmtree(second.root)


def test_a_servers_own_staging_directory_goes_when_it_exits(tmp_path) -> None:
    script = (
        "from agentenv_protocol.a2a_agent import StagingStore\n"
        "store = StagingStore(max_bytes=1)\n"
        f"open({str(tmp_path / 'root')!r}, 'w').write(str(store.root))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "AGENTENV_STAGING_DIR"}
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    assert not Path((tmp_path / "root").read_text()).exists()


def test_a_delete_spares_an_object_rewritten_since_it_was_read() -> None:
    with TestClient(_app()) as client:
        client.put(f"{BASE}/n", content=b"one")
        read_tag = client.get(f"{BASE}/n").headers["etag"]
        client.put(f"{BASE}/n", content=b"two")
        stale = client.delete(f"{BASE}/n", headers={"if-match": read_tag})
        kept = client.get(f"{BASE}/n")
        current = client.delete(f"{BASE}/n", headers={"if-match": kept.headers["etag"]})
    assert (stale.status_code, kept.content, current.status_code) == (412, b"two", 204)


@pytest.mark.parametrize(
    "path, status",
    [
        ("short/a", 404),  # an id too short to be unguessable
        (f"{SESSION}/%2E%2E/x", 400),  # sent encoded, as clients fold a literal ".."
        (f"{SESSION}//x", 400),
        (f"{SESSION}/a\\b", 400),
    ],
)
def test_staging_refuses_guessable_and_unnormalized_paths(path: str, status: int) -> None:
    with TestClient(_app()) as client:
        assert client.put(f"/ext/staging/{path}", content=b"x").status_code == status


def test_staging_lists_nothing_without_an_id() -> None:
    with TestClient(_app()) as client:
        client.put(f"{BASE}/a", content=b"x")
        assert client.get("/ext/staging/").status_code == 400


def test_staging_holds_to_its_limit_and_frees_what_it_drops() -> None:
    with TestClient(_app(max_bytes=10)) as client:
        assert client.put(f"{BASE}/a", content=b"123456").status_code == 201
        assert client.put(f"{BASE}/b", content=b"123456").status_code == 413
        # A replacement needs room beside the old copy while it lands, then frees the old one.
        assert client.put(f"{BASE}/a", content=b"1234").status_code == 201
        assert client.put(f"{BASE}/b", content=b"123456").status_code == 201
        client.delete(f"{BASE}/a")
        assert client.put(f"{BASE}/c", content=b"1234").status_code == 201
        client.delete(f"{BASE}/")
        assert client.put(f"{BASE}/d", content=b"1234567890").status_code == 201


def test_a_write_that_outgrows_the_limit_leaves_nothing_behind(tmp_path) -> None:
    with TestClient(_app(max_bytes=4)) as client:
        streamed = client.put(f"{BASE}/a", content=iter([b"12", b"345"]))  # no Content-Length
        assert (streamed.status_code, client.get(f"{BASE}/a").status_code) == (413, 404)
        assert client.put(f"{BASE}/a", content=b"1234").status_code == 201  # its reservation was released
    assert not any((tmp_path / "staging" / "incoming").iterdir())


def test_a_path_cannot_be_both_an_object_and_a_prefix() -> None:
    with TestClient(_app()) as client:
        client.put(f"{BASE}/a", content=b"x")
        assert client.put(f"{BASE}/a/b", content=b"x").status_code == 409


def test_an_upload_must_name_its_key_before_its_file() -> None:
    with TestClient(_app()) as client:
        missing = client.post(BASE, files={"file": ("f", b"x")})
        escaping = client.post(BASE, data={"key": "../x"}, files={"file": ("f", b"x")})
    assert (missing.status_code, escaping.status_code) == (400, 400)


@pytest.mark.asyncio
async def test_the_transfer_helpers_move_objects_through_staging(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(max_bytes=1 << 20)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        transfers.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=httpx.ASGITransport(app=app), **kwargs),
    )
    monkeypatch.setattr(
        transfers.httpx,
        "Client",
        lambda **kwargs: TestClient(app, follow_redirects=False),
    )
    url = f"https://agent.example.test{BASE}"
    body = b"x" * 300_000

    uploaded = await upload(
        WriteObject(
            media_type="application/octet-stream",
            max_bytes=len(body),
            write=HttpPutGrant(kind="http-put", url=f"{url}/0", expires_at=_expiry()),
        ),
        body,
    )
    await download(
        ReadObject(
            media_type="application/octet-stream",
            max_bytes=len(body),
            size_bytes=len(body),
            sha256=uploaded.sha256,
            read=HttpGetGrant(kind="http-get", url=f"{url}/0", expires_at=_expiry()),
        ),
        tmp_path / "out",
    )
    namespace = NamespaceUploader(
        WriteNamespaceGrant(
            root_path="changelog/run-1",
            expires_at=_expiry(),
            max_objects=10,
            max_object_bytes=1024,
            max_total_bytes=4096,
            write=HttpPostPolicyGrant(
                kind="http-post-policy",
                url=f"{url}/ns",
                fields={},
                path_field="key",
                file_field="file",
            ),
        )
    )
    await namespace.upload("000001.tar", b"increment")

    assert (tmp_path / "out").read_bytes() == body
    with TestClient(app) as client:
        listed = client.get(f"{BASE}/ns/").json()["objects"]
        increment = client.get(f"{BASE}/ns/changelog/run-1/000001.tar").content
    assert [entry["path"] for entry in listed] == ["changelog/run-1/000001.tar"]
    assert increment == b"increment"
