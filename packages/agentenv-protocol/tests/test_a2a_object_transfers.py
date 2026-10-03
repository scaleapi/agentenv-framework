from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import agentenv_protocol.a2a_agent as sdk
import httpx
import pytest
from agentenv_protocol import transfers
from agentenv_protocol.a2a_agent import (
    SNAPSHOT_V1,
    STANDARD_EXTENSIONS,
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    NamespaceChangelogEnableRequest,
    NativeTrajectory,
    ObjectChangelogApplyRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    ObjectSnapshotSaveResponse,
    RequestDefinition,
    SkillBundle,
    SkillBundleFile,
    SnapshotUploadedObjects,
    TaskObjectTrajectoryRequest,
    TaskRequest,
    TaskResult,
    TrajectoryWriteObjects,
    a2a_agent,
    card_request_accepts,
    enable,
    extension,
    request_fields,
)
from agentenv_protocol.a2a_agent.framework import _SdkServices
from agentenv_protocol.transfers import (
    HttpGetGrant,
    HttpPostPolicyGrant,
    HttpPutGrant,
    NamespaceUploader,
    ReadObject,
    TransferError,
    Uploaded,
    WriteNamespaceGrant,
    WriteObject,
    download,
    upload,
)
from pydantic import ValidationError
from starlette.testclient import TestClient

_UTC = timezone.utc  # noqa: UP017 -- package supports Python 3.10.
_SIGNED_QUERY = "signature=sig-secret&token=token-secret"


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfers, "_RETRY_BACKOFF_SECONDS", 0)


def _route(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Send every HTTP client the transfer helpers open through ``handler``."""
    real_async_client = httpx.AsyncClient
    real_client = httpx.Client
    monkeypatch.setattr(
        transfers.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    monkeypatch.setattr(
        transfers.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def _respond(requests: list[httpx.Request], status_code: int):
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code)

    return handle


def _time_out(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _served(content: bytes, **kwargs) -> httpx.Response:
    """A response whose body streams, as a real transport's does."""
    return httpx.Response(200, stream=httpx.ByteStream(content), **kwargs)


def _expiry() -> datetime:
    return datetime.now(_UTC) + timedelta(minutes=5)


def _read_object(content: bytes = b"# Skill") -> ReadObject:
    return ReadObject(
        media_type="text/markdown",
        max_bytes=1024,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        read=HttpGetGrant(
            kind="http-get",
            url=f"https://objects.example.test/read?{_SIGNED_QUERY}",
            expires_at=_expiry(),
        ),
    )


def _write_object(
    *, max_bytes: int = 1024, expires_at: datetime | None = None
) -> WriteObject:
    return WriteObject(
        media_type="application/json",
        max_bytes=max_bytes,
        write=HttpPutGrant(
            kind="http-put",
            url=f"https://objects.example.test/write?{_SIGNED_QUERY}",
            expires_at=expires_at or _expiry(),
        ),
    )


def _opaque_write_object() -> WriteObject:
    return _write_object().model_copy(update={"media_type": "application/octet-stream"})


def _opaque_read_object() -> ReadObject:
    return _read_object().model_copy(update={"media_type": "application/octet-stream"})


def _namespace_grant(
    *,
    max_objects: int = 10,
    max_object_bytes: int = 1024,
    max_total_bytes: int = 4096,
) -> WriteNamespaceGrant:
    return WriteNamespaceGrant(
        root_path="changelog/run-1",
        expires_at=_expiry(),
        max_objects=max_objects,
        max_object_bytes=max_object_bytes,
        max_total_bytes=max_total_bytes,
        write=HttpPostPolicyGrant(
            kind="http-post-policy",
            url=f"https://objects.example.test/upload?{_SIGNED_QUERY}",
            fields={"policy": "policy-secret"},
            path_field="key",
            file_field="file",
        ),
    )


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_transfer_timeout_allows_large_streams_without_being_unbounded() -> None:
    assert transfers._TRANSFER_TIMEOUT.connect == 30
    assert transfers._TRANSFER_TIMEOUT.read == 60
    assert transfers._TRANSFER_TIMEOUT.write == 60
    assert transfers._TRANSFER_TIMEOUT.pool == 30


def test_a2a_agent_reexports_the_transfer_module() -> None:
    for name in (
        "HttpGetGrant",
        "HttpPostPolicyGrant",
        "HttpPutGrant",
        "NamespaceUploader",
        "ReadObject",
        "TransferError",
        "Uploaded",
        "WriteNamespaceGrant",
        "WriteObject",
        "download",
        "upload",
    ):
        assert getattr(sdk, name) is getattr(transfers, name)


def test_transfer_models_are_closed_and_validate_contract_constraints() -> None:
    with pytest.raises(ValidationError, match="absolute HTTPS"):
        HttpGetGrant(
            kind="http-get",
            url="http://objects.example.test/read",
            expires_at=_expiry(),
        )
    with pytest.raises(ValidationError, match="must use UTC"):
        HttpPutGrant(
            kind="http-put",
            url="https://objects.example.test/write",
            expires_at=_expiry().astimezone(timezone(timedelta(hours=1))),
        )
    with pytest.raises(ValidationError, match="RFC 3339 UTC"):
        HttpPutGrant(
            kind="http-put",
            url="https://objects.example.test/write",
            expires_at=_expiry().replace(tzinfo=None),
        )
    with pytest.raises(ValidationError, match="lowercase hexadecimal"):
        ReadObject(
            media_type="application/json",
            max_bytes=10,
            sha256="A" * 64,
            read=HttpGetGrant(
                kind="http-get",
                url="https://objects.example.test/read",
                expires_at=_expiry(),
            ),
        )
    with pytest.raises(ValidationError, match="concrete MIME type"):
        WriteObject.model_validate(
            {**_write_object().model_dump(mode="json"), "media_type": "*/*"}
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        WriteObject.model_validate(
            {
                **_write_object().model_dump(mode="json"),
                "provider": "s3",
            }
        )


def test_transfer_errors_derive_status_and_retryability_from_their_code() -> None:
    unavailable = TransferError("transfer_unavailable", "Try again.")
    assert (unavailable.status_code, unavailable.retryable) == (502, True)
    assert unavailable.body() == {
        "error": {
            "code": "transfer_unavailable",
            "message": "Try again.",
            "retryable": True,
        }
    }
    too_large = TransferError("transfer_too_large", "Too large.")
    assert (too_large.status_code, too_large.retryable) == (413, False)
    expired_timeout = TransferError("transfer_timeout", "Slow.", retryable=False)
    assert (expired_timeout.status_code, expired_timeout.retryable) == (504, False)


def test_namespace_grant_writes_through_a_post_policy() -> None:
    grant = _namespace_grant()
    assert grant.write.kind == "http-post-policy"

    with pytest.raises(ValidationError, match="normalized relative POSIX"):
        WriteNamespaceGrant.model_validate(
            {**grant.model_dump(mode="json"), "root_path": "../escape"}
        )
    with pytest.raises(ValidationError):
        WriteNamespaceGrant.model_validate(
            {
                **grant.model_dump(mode="json"),
                "write": {
                    "kind": "http-put-namespace",
                    "base_url": "https://objects.example.test/container",
                },
            }
        )


@pytest.mark.asyncio
async def test_namespace_uploader_counts_overwrites_once_and_enforces_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(204)

    _route(monkeypatch, handle)
    uploader = NamespaceUploader(_namespace_grant(max_objects=2, max_total_bytes=6))

    await uploader.upload("000000.tar", _write(tmp_path / "first.tar", b"1234"))
    await uploader.upload("000000.tar", _write(tmp_path / "second.tar", b"12"))
    await uploader.upload("nested/000001.tar", _write(tmp_path / "third.tar", b"3456"))

    assert len(bodies) == 3
    assert b"changelog/run-1/000000.tar" in bodies[0]
    assert b"changelog/run-1/nested/000001.tar" in bodies[2]

    with pytest.raises(TransferError) as exc_info:
        await uploader.upload("000002.tar", _write(tmp_path / "fourth.tar", b"x"))
    assert exc_info.value.code == "transfer_too_large"
    assert "object limit" in str(exc_info.value)

    with pytest.raises(TransferError) as exc_info:
        await uploader.upload(
            "nested/000001.tar", _write(tmp_path / "fifth.tar", b"34567")
        )
    assert exc_info.value.code == "transfer_too_large"
    assert "total-byte limit" in str(exc_info.value)
    assert len(bodies) == 3

    with pytest.raises(TransferError) as exc_info:
        await uploader.upload("../escape.tar", tmp_path / "fourth.tar")
    assert exc_info.value.code == "invalid_transfer"


@pytest.mark.asyncio
async def test_namespace_uploader_runs_concurrent_uploads_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        started.set()
        release.wait(timeout=10)
        return httpx.Response(204)

    _route(monkeypatch, handle)
    uploader = NamespaceUploader(_namespace_grant(max_objects=2, max_total_bytes=10))
    first = asyncio.create_task(
        uploader.upload("000000.tar", _write(tmp_path / "first.tar", b"123456"))
    )
    await asyncio.to_thread(started.wait, 10)
    overwrite = asyncio.create_task(
        uploader.upload("000000.tar", _write(tmp_path / "second.tar", b"1234"))
    )
    await asyncio.sleep(0.05)
    assert len(requests) == 1

    release.set()
    await first
    await overwrite
    assert len(requests) == 2

    await uploader.upload("000001.tar", _write(tmp_path / "third.tar", b"abcdef"))
    with pytest.raises(TransferError) as exc_info:
        await uploader.upload("000002.tar", _write(tmp_path / "fourth.tar", b"x"))
    assert exc_info.value.code == "transfer_too_large"


@pytest.mark.asyncio
async def test_namespace_uploader_does_not_count_failed_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = iter([403, 204])
    _route(monkeypatch, lambda request: httpx.Response(next(statuses)))
    uploader = NamespaceUploader(_namespace_grant(max_objects=1))

    with pytest.raises(TransferError) as exc_info:
        await uploader.upload("000000.tar", _write(tmp_path / "first.tar", b"data"))
    assert exc_info.value.code == "transfer_rejected"

    await uploader.upload("000001.tar", _write(tmp_path / "second.tar", b"data"))


@pytest.mark.asyncio
async def test_namespace_uploader_refuses_a_source_that_grows_after_it_is_sized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(204)

    _route(monkeypatch, handle)
    sized = transfers._check_source
    growing = _write(tmp_path / "growing.tar", b"1234")

    def size_then_grow(source: Path, max_bytes: int) -> int:
        size_bytes = sized(source, max_bytes)
        if source == growing:
            source.write_bytes(b"1234" * 25)
        return size_bytes

    monkeypatch.setattr(transfers, "_check_source", size_then_grow)
    uploader = NamespaceUploader(
        _namespace_grant(max_objects=2, max_object_bytes=100, max_total_bytes=10)
    )

    with pytest.raises(TransferError) as exc_info:
        await uploader.upload("000000.tar", growing)
    assert exc_info.value.code == "invalid_transfer"
    assert bodies == []

    await uploader.upload("000001.tar", _write(tmp_path / "full.tar", b"0123456789"))
    assert len(bodies) == 1


@pytest.mark.asyncio
async def test_namespace_uploader_posts_policy_fields_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(204)

    _route(monkeypatch, handle)
    uploader = NamespaceUploader(
        _namespace_grant(max_objects=1, max_object_bytes=10, max_total_bytes=10)
    )

    await uploader.upload("000000.tar", _write(tmp_path / "increment.tar", b"data"))

    assert len(bodies) == 1
    assert b'name="policy"' in bodies[0]
    assert b"policy-secret" in bodies[0]
    assert b'name="key"' in bodies[0]
    assert b"changelog/run-1/000000.tar" in bodies[0]
    assert b'name="Content-Type"' in bodies[0]
    assert b"application/octet-stream" in bodies[0]
    assert b'name="file"' in bodies[0]
    assert b"data" in bodies[0]


def test_skill_bundle_requires_safe_unique_paths_and_root_skill_md() -> None:
    root = SkillBundleFile(path="SKILL.md", object=_read_object())
    bundle = SkillBundle(max_total_bytes=1024, files=[root])
    assert bundle.files == [root]

    with pytest.raises(ValidationError, match="normalized relative POSIX"):
        SkillBundleFile(path="../SKILL.md", object=_read_object())
    with pytest.raises(ValidationError, match="root SKILL.md"):
        SkillBundle(
            max_total_bytes=1024,
            files=[SkillBundleFile(path="reference.md", object=_read_object())],
        )
    with pytest.raises(ValidationError, match="file limits exceed max_total_bytes"):
        SkillBundle(
            max_total_bytes=1024,
            files=[
                root,
                SkillBundleFile(
                    path="reference.md",
                    object=_read_object(b"x").model_copy(update={"max_bytes": 1}),
                ),
            ],
        )


def test_sdk_snapshot_operations_take_objects_and_closed_responses() -> None:
    @a2a_agent(
        identity=AgentIdentity(
            name="portable-snapshot", description="test", version="1"
        )
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(SNAPSHOT_V1.save)
        async def save_objects(self, request: ObjectSnapshotSaveRequest):
            if request.context_id == "wrong-response":
                return {
                    "s3_prefix": "s3://legacy/snapshot",
                    "context_id": request.context_id,
                    "timestamp": "2026-09-11T00:00:00Z",
                }
            return ObjectSnapshotSaveResponse(
                context_id=request.context_id,
                objects=SnapshotUploadedObjects(trajectory=Uploaded(size_bytes=12)),
            )

        @extension(SNAPSHOT_V1.load)
        async def load_objects(self, request: ObjectSnapshotLoadRequest):
            return {"context_id": request.target_context_id or "loaded"}

        @extension(SNAPSHOT_V1.changelog.enable)
        async def enable_changelog_namespace(
            self, request: NamespaceChangelogEnableRequest
        ):
            return {"roots": request.roots or []}

        @extension(SNAPSHOT_V1.changelog.apply)
        async def apply_changelog_objects(self, request: ObjectChangelogApplyRequest):
            return {
                "count": len(request.increments),
                "context_id": request.target_context_id,
            }

    write = _opaque_write_object().model_dump(mode="json")
    read = _opaque_read_object().model_dump(mode="json")
    namespace = _namespace_grant().model_dump(mode="json")

    with TestClient(Agent().create_app()) as client:
        rejected_legacy = client.post(
            "/ext/snapshot",
            json={"context_id": "legacy", "s3_prefix": "s3://bucket/snapshot"},
        )
        portable = client.post(
            "/ext/snapshot",
            json={"context_id": "portable", "objects": {"trajectory": write}},
        )
        wrong_response = client.post(
            "/ext/snapshot",
            json={
                "context_id": "wrong-response",
                "objects": {"trajectory": write},
            },
        )
        loaded = client.put(
            "/ext/snapshot",
            json={
                "objects": {"trajectory": read},
                "target_context_id": "restored",
            },
        )
        enabled = client.post(
            "/ext/snapshot/changelog",
            json={"write_namespace": namespace, "roots": ["workspace"]},
        )
        applied = client.put(
            "/ext/snapshot/changelog",
            json={
                "increments": [{"sequence": 0, "object": read}],
                "target_context_id": "continued",
            },
        )

    assert rejected_legacy.status_code == 400
    assert portable.json() == {
        "context_id": "portable",
        "objects": {"trajectory": {"size_bytes": 12}},
    }
    assert wrong_response.status_code == 500
    assert "s3://legacy/snapshot" not in wrong_response.text
    assert loaded.json() == {"context_id": "restored"}
    assert enabled.json() == {"roots": ["workspace"]}
    assert applied.json() == {"count": 1, "context_id": "continued"}


@pytest.mark.asyncio
async def test_sdk_trajectory_handler_uploads_portable_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = _SdkServices((enable(TRAJECTORY_V1),), None)
    services.task_trajectories["task-1"] = NativeTrajectory(
        format="events/v1", payload=[{"type": "result"}]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.content == b'[{"type":"result"}]'
        return httpx.Response(200)

    _route(monkeypatch, handle)
    result = await services.trajectory_get(
        TaskObjectTrajectoryRequest(
            task_id="task-1",
            objects=TrajectoryWriteObjects(trajectory=_write_object()),
        )
    )

    assert result == {
        "objects": {
            "trajectory": {
                "size_bytes": 19,
                "sha256": hashlib.sha256(b'[{"type":"result"}]').hexdigest(),
            }
        }
    }
    assert services.task_trajectories.get("task-1") is not None


@pytest.mark.asyncio
async def test_upload_streams_with_bound_and_returns_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"trajectory":[]}'
    source = _write(tmp_path / "trajectory.json", content)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["content-type"] == "application/json"
        assert request.headers["content-length"] == str(len(content))
        assert request.content == content
        return httpx.Response(200)

    _route(monkeypatch, handle)
    result = await upload(_write_object(), source)

    assert result.size_bytes == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()

    with pytest.raises(TransferError) as exc_info:
        await upload(_write_object(max_bytes=1), source)
    assert exc_info.value.code == "transfer_too_large"
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [b"1234" * 25, b"12"])
async def test_upload_refuses_a_source_that_changes_after_it_is_sized(
    changed: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    _route(monkeypatch, _respond(requests, 200))
    sized = transfers._check_source

    def size_then_change(source: Path, max_bytes: int) -> int:
        size_bytes = sized(source, max_bytes)
        source.write_bytes(changed)
        return size_bytes

    monkeypatch.setattr(transfers, "_check_source", size_then_change)

    with pytest.raises(TransferError) as exc_info:
        await upload(_write_object(), _write(tmp_path / "source.json", b"1234"))
    assert exc_info.value.code == "invalid_transfer"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        lambda request: httpx.Response(503),
        lambda request: httpx.Response(429),
        lambda request: httpx.Response(408),
        lambda request: httpx.Response(
            400, content=b"<Error><Code>RequestTimeout</Code></Error>"
        ),
        _time_out,
        _refuse,
    ],
    ids=["503", "429", "408", "s3-request-timeout", "timeout", "connect-error"],
)
async def test_upload_retries_transient_failures_and_rereads_the_source(
    failure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"trajectory":[1]}'
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return failure(request) if len(bodies) == 1 else httpx.Response(200)

    _route(monkeypatch, handle)
    result = await upload(_write_object(), _write(tmp_path / "t.json", content))

    assert bodies == [content, content]
    assert result.sha256 == hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_transfers_stop_after_three_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    _route(monkeypatch, _respond(requests, 503))

    with pytest.raises(TransferError) as exc_info:
        await upload(_write_object(), _write(tmp_path / "t.json", b"{}"))
    assert exc_info.value.code == "transfer_unavailable"
    assert exc_info.value.retryable is True
    assert len(requests) == 3

    requests.clear()
    with pytest.raises(TransferError) as exc_info:
        await download(_read_object(), tmp_path / "skill.md")
    assert exc_info.value.code == "transfer_unavailable"
    assert len(requests) == 3

    requests.clear()
    uploader = NamespaceUploader(_namespace_grant())
    with pytest.raises(TransferError) as exc_info:
        await uploader.upload("000000.tar", _write(tmp_path / "i.tar", b"data"))
    assert exc_info.value.code == "transfer_unavailable"
    assert len(requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 403])
async def test_transfers_do_not_retry_rejections(
    status_code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    _route(monkeypatch, _respond(requests, status_code))

    with pytest.raises(TransferError) as exc_info:
        await upload(_write_object(), _write(tmp_path / "t.json", b"{}"))
    assert exc_info.value.code == "transfer_rejected"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_transfers_do_not_retry_once_the_grant_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _write_object()
    now = [datetime.now(_UTC)]

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr(transfers, "datetime", _Clock)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        now[0] = target.write.expires_at
        return httpx.Response(503)

    _route(monkeypatch, handle)

    with pytest.raises(TransferError) as exc_info:
        await upload(target, _write(tmp_path / "t.json", b"{}"))
    assert exc_info.value.code == "transfer_unavailable"
    assert len(requests) == 1


class _InterruptedBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"# Sk"
        raise httpx.ReadError("connection reset")


@pytest.mark.asyncio
async def test_download_retries_and_restarts_an_interrupted_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"# Skill"
    responses = iter(
        [httpx.Response(200, stream=_InterruptedBody()), _served(content)]
    )
    _route(monkeypatch, lambda request: next(responses))
    destination = tmp_path / "skill.md"

    await download(_read_object(content), destination)

    assert destination.read_bytes() == content
    assert [path.name for path in tmp_path.iterdir()] == ["skill.md"]


@pytest.mark.asyncio
async def test_download_stores_the_raw_bytes_the_store_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded = b"x" * 100_000
    encoded = gzip.compress(decoded)
    accept_encodings: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        accept_encodings.append(request.headers["accept-encoding"])
        return _served(encoded, headers={"Content-Encoding": "gzip"})

    _route(monkeypatch, handle)
    source = _read_object(encoded).model_copy(update={"max_bytes": len(encoded)})
    destination = tmp_path / "object.gz"

    await download(source, destination)

    assert accept_encodings == ["identity"]
    assert destination.read_bytes() == encoded


@pytest.mark.asyncio
async def test_download_is_atomic_and_checks_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"# Skill"
    destination = tmp_path / "skill.md"
    _route(monkeypatch, lambda request: _served(content))

    await download(_read_object(content), destination)
    assert destination.read_bytes() == content

    destination.write_text("existing")
    invalid = _read_object(content).model_copy(update={"sha256": "0" * 64})
    with pytest.raises(TransferError) as exc_info:
        await download(invalid, destination)
    assert exc_info.value.code == "integrity_mismatch"
    assert destination.read_text() == "existing"

    too_large = _read_object(content).model_copy(
        update={"max_bytes": 3, "size_bytes": None}
    )
    with pytest.raises(TransferError) as exc_info:
        await download(too_large, destination)
    assert exc_info.value.code == "transfer_too_large"
    assert destination.read_text() == "existing"


@pytest.mark.asyncio
async def test_transfer_helpers_keep_grant_urls_out_of_httpx_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="httpx")
    unrelated_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return _served(b"# Skill")
        return httpx.Response(204 if request.method == "POST" else 200)

    _route(monkeypatch, handle)
    await upload(_write_object(), _write(tmp_path / "t.json", b"{}"))
    await download(_read_object(), tmp_path / "skill.md")
    await NamespaceUploader(_namespace_grant()).upload(
        "000000.tar", _write(tmp_path / "i.tar", b"data")
    )
    async with unrelated_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        await client.get("https://unrelated.example.test/page?kept=1")

    messages = [
        record.getMessage() for record in caplog.records if record.name == "httpx"
    ]
    assert [message.split(" ")[2:4] for message in messages] == [
        ["PUT", "https://objects.example.test/<redacted>"],
        ["GET", "https://objects.example.test/<redacted>"],
        ["POST", "https://objects.example.test/<redacted>"],
        ["GET", "https://unrelated.example.test/page?kept=1"],
    ]
    assert "secret" not in caplog.text


_SIGNED_URL = "https://user:pw@objects.example.test/write?X-Amz-Signature=secret"


@pytest.mark.parametrize(
    "message, args",
    [
        ("HTTP Request: PUT %s", (httpx.URL(_SIGNED_URL),)),
        ("HTTP Request: PUT %s", (_SIGNED_URL,)),
        (f'HTTP Request: PUT {_SIGNED_URL} "HTTP/1.1 200 OK"', ()),
    ],
    ids=["url-argument", "string-argument", "preformatted"],
)
def test_grant_urls_are_redacted_however_httpx_formats_its_log_call(
    message: str, args: tuple, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="httpx")

    with transfers.redacting_request_urls():
        logging.getLogger("httpx").info(message, *args)

    (record,) = caplog.records
    assert record.getMessage().startswith(
        "HTTP Request: PUT https://objects.example.test/<redacted>"
    )
    assert "secret" not in caplog.text and "pw" not in caplog.text


def test_trajectory_object_request_fixes_json_media_type() -> None:
    request = TaskObjectTrajectoryRequest(
        task_id="task-1", objects=TrajectoryWriteObjects(trajectory=_write_object())
    )
    assert request.task_id == "task-1"

    with pytest.raises(ValidationError, match="application/json"):
        TrajectoryWriteObjects(
            trajectory=_write_object().model_copy(update={"media_type": "text/plain"})
        )


def test_snapshot_and_changelog_objects_fix_opaque_media_and_sequence() -> None:
    with pytest.raises(ValidationError, match="application/octet-stream"):
        ObjectSnapshotSaveRequest(
            context_id="context-1",
            objects={"trajectory": _write_object()},
        )

    with pytest.raises(ValidationError, match="strictly increasing"):
        ObjectChangelogApplyRequest(
            increments=[
                {"sequence": 2, "object": _opaque_read_object()},
                {"sequence": 1, "object": _opaque_read_object()},
            ]
        )

    request = ObjectChangelogApplyRequest(
        increments=[
            {"sequence": 2, "object": _opaque_read_object()},
            {"sequence": 4, "object": _opaque_read_object()},
        ]
    )
    assert [increment.sequence for increment in request.increments] == [2, 4]

    assert ObjectChangelogApplyRequest(increments=[]).increments == []


def test_card_request_accepts_every_variant_a_card_renders() -> None:
    """agent-env negotiates against rendered cards with ``card_request_accepts``: every variant
    ``RequestDefinition.to_card`` renders must be accepted, and a disabled one refused."""
    checked = 0
    for definition in STANDARD_EXTENSIONS.values():
        for operation in definition.operations.values():
            request = operation.request
            if not isinstance(request, RequestDefinition):
                continue
            card = request.to_card(optional=request.optional_variants)
            for variant in request.variants:
                fields = request_fields(variant.model).required
                assert card_request_accepts(card, fields), (definition.uri, operation.name, variant.name)
                checked += 1
    assert checked

    trajectory_get = TRAJECTORY_V1.operation("get").request
    assert not card_request_accepts(trajectory_get.to_card(), ("context_id",))
    assert card_request_accepts(trajectory_get.to_card(), ("task_id",))


def test_card_request_accepts_a_hand_written_supported_list() -> None:
    card = {"supported": ["model", "system_prompt"]}

    assert card_request_accepts(card, ("model",))
    assert not card_request_accepts(card, ("model", "role"))


@pytest.mark.asyncio
async def test_upload_sends_bytes_already_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    content = b'{"trajectory":[1]}'
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(200)

    _route(monkeypatch, handle)
    result = await upload(_write_object(), content)

    assert bodies == [content]
    assert result == Uploaded(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(TransferError) as exc_info:
        await upload(_write_object(max_bytes=1024), b"x" * 1025)
    assert exc_info.value.code == "transfer_too_large"


@pytest.mark.asyncio
async def test_namespace_upload_of_bytes_declares_its_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S3 refuses a chunked POST (411); an in-memory source must be sized like a file."""
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        return httpx.Response(204)

    _route(monkeypatch, handle)
    uploader = NamespaceUploader(_namespace_grant())

    from_bytes = await uploader.upload("000000.json", b'{"step":0}')
    from_file = await uploader.upload("000001.json", _write(tmp_path / "i.json", b'{"step":0}'))

    for request in requests:
        assert "transfer-encoding" not in request.headers
        assert int(request.headers["content-length"]) == len(request.content)
    assert from_bytes == from_file
