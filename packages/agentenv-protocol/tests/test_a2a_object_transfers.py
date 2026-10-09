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
    TaskTrajectoryRequest,
    TrajectoryState,
    TrajectoryWriteObjects,
    a2a_agent,
    card_request_accepts,
    enable,
    extension,
    request_fields,
)
from agentenv_protocol.a2a_agent.framework import _SdkServices
from agentenv_protocol.transfers import (
    MAX_PARTS,
    PARTS_IN_FLIGHT,
    HttpGetGrant,
    HttpPartsPutGrant,
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


def _parts_write_object(
    *, part_bytes: int = 4, parts: int = 4, max_bytes: int | None = None, headers: dict[str, str] | None = None
) -> WriteObject:
    return WriteObject(
        media_type="application/zip",
        max_bytes=part_bytes * parts if max_bytes is None else max_bytes,
        write=HttpPartsPutGrant(
            kind="http-put-parts",
            part_bytes=part_bytes,
            urls=[f"https://objects.example.test/part/{n}?{_SIGNED_QUERY}" for n in range(1, parts + 1)],
            expires_at=_expiry(),
            headers=headers,
        ),
    )


def _part_number(request: httpx.Request) -> int:
    return int(request.url.path.rsplit("/", 1)[-1])


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
        "HttpPartsPutGrant",
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
    services.task_trajectories.register("task-1", "context-1")
    services.task_trajectories.seal(
        "task-1", TrajectoryState.COMPLETED, native=b'[{"type":"result"}]'
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
    assert services.task_trajectories.final("task-1") is not None


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


def test_a_parts_grant_is_told_apart_from_a_put_grant_and_bounded_by_its_parts() -> None:
    parts = _parts_write_object(part_bytes=4, parts=3)
    assert WriteObject.model_validate(parts.model_dump(mode="json")) == parts
    assert WriteObject.model_validate(_write_object().model_dump(mode="json")).write.kind == "http-put"
    with pytest.raises(ValidationError, match="part_bytes times") as exc_info:
        _parts_write_object(part_bytes=4, parts=3, max_bytes=13)
    assert "secret" not in str(exc_info.value)
    with pytest.raises(ValidationError, match="union_tag_invalid") as exc_info:
        WriteObject.model_validate(
            {**parts.model_dump(mode="json"), "write": {**parts.write.model_dump(mode="json"), "kind": "http-put-chunks"}}
        )
    assert "secret" not in str(exc_info.value)
    for urls in ([], [parts.write.urls[0]] * (MAX_PARTS + 1)):
        with pytest.raises(ValidationError):
            HttpPartsPutGrant(kind="http-put-parts", part_bytes=4, urls=urls, expires_at=_expiry())


@pytest.mark.asyncio
@pytest.mark.parametrize("in_memory", [False, True], ids=["file", "bytes"])
async def test_a_parts_upload_puts_each_range_to_its_own_url(
    in_memory: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"0123456789"
    received: dict[int, httpx.Request] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        received[_part_number(request)] = request
        return httpx.Response(200)

    _route(monkeypatch, handle)
    source = content if in_memory else _write(tmp_path / "bundle.zip", content)
    result = await upload(_parts_write_object(part_bytes=4, parts=4, headers={"x-grant": "kept"}), source)

    assert sorted(received) == [1, 2, 3]
    assert [received[n].content for n in (1, 2, 3)] == [b"0123", b"4567", b"89"]
    assert [received[n].headers["content-length"] for n in (1, 2, 3)] == ["4", "4", "2"]
    assert all(r.headers["x-grant"] == "kept" and "content-type" not in r.headers for r in received.values())
    assert result == Uploaded(size_bytes=len(content), sha256=None)


@pytest.mark.asyncio
async def test_a_one_part_upload_reports_its_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []
    _route(monkeypatch, _respond(requests, 200))

    result = await upload(_parts_write_object(part_bytes=64, parts=2), _write(tmp_path / "small.zip", b"tiny"))

    assert [_part_number(r) for r in requests] == [1]
    assert result == Uploaded(size_bytes=4, sha256=hashlib.sha256(b"tiny").hexdigest())



@pytest.mark.asyncio
@pytest.mark.parametrize(("content", "hashed"), [(b"0123456789", 0), (b"tiny", 4)], ids=["parts", "one-part"])
async def test_only_a_one_part_upload_spends_cpu_on_a_checksum(
    content: bytes, hashed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fed = [0]
    real_sha256 = hashlib.sha256

    class _Counting:
        def __init__(self) -> None:
            self._digest = real_sha256()

        def update(self, data: bytes) -> None:
            fed[0] += len(data)
            self._digest.update(data)

        def hexdigest(self) -> str:
            return self._digest.hexdigest()

    monkeypatch.setattr(transfers.hashlib, "sha256", _Counting)
    _route(monkeypatch, _respond([], 200))
    await upload(_parts_write_object(part_bytes=4, parts=4), _write(tmp_path / "bundle.zip", content))

    assert fed[0] == hashed

@pytest.mark.asyncio
async def test_a_parts_upload_larger_than_its_grant_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    _route(monkeypatch, _respond(requests, 200))

    with pytest.raises(TransferError) as exc_info:
        await upload(_parts_write_object(part_bytes=4, parts=2), _write(tmp_path / "big.zip", b"123456789"))
    assert exc_info.value.code == "transfer_too_large"
    assert requests == []


@pytest.mark.asyncio
async def test_a_failed_part_is_retried_on_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(_part_number(request))
        return httpx.Response(503) if sent.count(2) == 1 and sent[-1] == 2 else httpx.Response(200)

    _route(monkeypatch, handle)
    await upload(_parts_write_object(part_bytes=4, parts=3), _write(tmp_path / "bundle.zip", b"0123456789"))

    assert sorted(sent) == [1, 2, 2, 3]


@pytest.mark.asyncio
async def test_a_rejected_part_fails_the_upload_without_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(_part_number(request))
        return httpx.Response(403 if sent[-1] == 2 else 200)

    _route(monkeypatch, handle)
    with pytest.raises(TransferError) as exc_info:
        await upload(_parts_write_object(part_bytes=4, parts=3), _write(tmp_path / "bundle.zip", b"0123456789"))
    assert exc_info.value.code == "transfer_rejected"
    assert "secret" not in str(exc_info.value)
    assert sent.count(2) == 1



@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"0123456789", b"012"], ids=["parts", "one-part"])
async def test_a_parts_upload_refuses_a_source_that_grows_while_it_uploads(
    content: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "bundle.zip", content)

    def handle(request: httpx.Request) -> httpx.Response:
        if _part_number(request) == 1:
            with source.open("ab") as appended:
                appended.write(b"more")
        return httpx.Response(200)

    _route(monkeypatch, handle)
    with pytest.raises(TransferError) as exc_info:
        await upload(_parts_write_object(part_bytes=4, parts=4), source)
    assert exc_info.value.code == "invalid_transfer"


@pytest.mark.asyncio
async def test_a_rejected_part_cancels_the_parts_still_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[int] = []
    finished: list[int] = []
    never = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        number = _part_number(request)
        started.append(number)
        if number == 2:
            await asyncio.sleep(0.01)
            return httpx.Response(403)
        await never.wait()
        finished.append(number)
        return httpx.Response(200)

    _route(monkeypatch, handle)
    with pytest.raises(TransferError) as exc_info:
        await asyncio.wait_for(
            upload(_parts_write_object(part_bytes=4, parts=3), _write(tmp_path / "bundle.zip", b"0123456789")), 5
        )
    assert exc_info.value.code == "transfer_rejected"
    assert sorted(started) == [1, 2, 3]
    assert finished == []

@pytest.mark.asyncio
async def test_a_parts_upload_keeps_a_bounded_number_of_parts_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_flight = peak = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200)

    _route(monkeypatch, handle)
    parts = 3 * PARTS_IN_FLIGHT
    await upload(_parts_write_object(part_bytes=1, parts=parts), _write(tmp_path / "bundle.zip", b"x" * parts))

    assert 1 < peak <= PARTS_IN_FLIGHT


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


_PUBLIC = "https://agent.example.test/sandbox/vm-8000"  # the provider routes this prefix to the agent
_STAGED = "/ext/staging/" + "c" * 32 + "/0"
_NAMESPACE = "/ext/staging/" + "n" * 32


def _on_own_staging(content: bytes) -> tuple[WriteObject, ReadObject, WriteNamespaceGrant]:
    """Grants naming paths on their holder's own staging, as agent-env stages them."""
    headers = {transfers.STAGING_PATH_HEADER: _STAGED}
    write = WriteObject(
        media_type="application/json",
        max_bytes=1024,
        write=HttpPutGrant(kind="http-put", url=_PUBLIC + _STAGED, expires_at=_expiry(), headers=headers),
    )
    read = ReadObject(
        media_type="application/json",
        max_bytes=1024,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        read=HttpGetGrant(kind="http-get", url=_PUBLIC + _STAGED, expires_at=_expiry(), headers=headers),
    )
    namespace = _namespace_grant().model_copy(
        update={
            "write": HttpPostPolicyGrant(
                kind="http-post-policy",
                url=_PUBLIC + _NAMESPACE,
                fields={},
                path_field="key",
                file_field="file",
                headers={transfers.STAGING_PATH_HEADER: _NAMESPACE},
            )
        }
    )
    return write, read, namespace


@pytest.mark.asyncio
async def test_a_grant_on_its_holders_own_staging_goes_to_its_server_over_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("A2A_PORT", "8123")
    content = b'{"trajectory":[]}'
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(f"{request.method} {request.url}")
        if request.url.host != "127.0.0.1":
            raise httpx.ConnectError("the sandbox can't call its own public URL", request=request)
        request.read()
        return _served(content) if request.method == "GET" else httpx.Response(201)

    _route(monkeypatch, handle)
    write, read, namespace = _on_own_staging(content)

    await upload(write, content)
    await download(read, tmp_path / "trajectory.json")
    await NamespaceUploader(namespace).upload("000000.tar", b"increment")

    assert sent == [
        f"PUT http://127.0.0.1:8123{_STAGED}",
        f"GET http://127.0.0.1:8123{_STAGED}",
        f"POST http://127.0.0.1:8123{_NAMESPACE}",
    ]
    assert (tmp_path / "trajectory.json").read_bytes() == content


@pytest.mark.asyncio
async def test_a_holder_whose_server_isnt_listening_on_loopback_uses_the_grants_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("A2A_PORT", "8123")
    content = b'{"trajectory":[]}'
    sent: list[str] = []
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.host)
        if request.url.host == "127.0.0.1":
            raise httpx.ConnectError("refused", request=request)
        bodies.append(request.read())
        return _served(content) if request.method == "GET" else httpx.Response(201)

    _route(monkeypatch, handle)
    write, read, namespace = _on_own_staging(content)

    uploaded = await upload(write, content)
    await download(read, tmp_path / "trajectory.json")
    increment = await NamespaceUploader(namespace).upload("000000.tar", b"increment")

    assert sent == ["127.0.0.1", "agent.example.test"] * 3
    assert bodies[0] == content and uploaded.size_bytes == len(content)
    assert b"increment" in bodies[2] and increment.size_bytes == len(b"increment")
    assert (tmp_path / "trajectory.json").read_bytes() == content


@pytest.mark.asyncio
async def test_an_answer_from_the_holders_own_server_is_final(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A2A_PORT", "8123")
    requests: list[httpx.Request] = []
    _route(monkeypatch, _respond(requests, 404))
    write, _, _ = _on_own_staging(b"")

    with pytest.raises(TransferError) as exc_info:
        await upload(write, b"{}")

    assert exc_info.value.code == "transfer_rejected"
    assert [request.url.host for request in requests] == ["127.0.0.1"]


@pytest.mark.parametrize(
    ("port", "path"),
    [
        (None, _STAGED),  # agent-env didn't deploy the holder
        ("http", _STAGED),
        ("0", _STAGED),
        ("65536", _STAGED),
        ("8123", None),  # not staged
        ("8123", "/ext/staging/" + "d" * 32 + "/0"),  # not this URL's path
        ("8123", _STAGED.lstrip("/")),
    ],
)
def test_a_grant_goes_to_its_url_unless_it_names_a_path_its_holders_known_port_serves(
    monkeypatch: pytest.MonkeyPatch, port: str | None, path: str | None
) -> None:
    if port is None:
        monkeypatch.delenv("A2A_PORT", raising=False)
    else:
        monkeypatch.setenv("A2A_PORT", port)
    headers = None if path is None else {transfers.STAGING_PATH_HEADER: path}

    assert transfers.loopback_url(_PUBLIC + _STAGED, headers) is None


def test_the_loopback_url_keeps_the_grants_query_and_reads_the_header_in_any_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A2A_PORT", "8123")
    headers = {transfers.STAGING_PATH_HEADER.lower(): _STAGED}

    assert (
        transfers.loopback_url(f"{_PUBLIC}{_STAGED}?part=1", headers)
        == f"http://127.0.0.1:8123{_STAGED}?part=1"
    )


@a2a_agent(identity=AgentIdentity(name="own-port", description="test", version="1"))
class _UploadsItsTrajectory(AgentEnvAgent):
    async def run(self, request: TaskRequest) -> TaskResult:
        return TaskResult.text("ok")

    @extension(TRAJECTORY_V1.get)
    async def trajectory(self, request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest):
        uploaded = await upload(request.objects.trajectory, b"[]")
        return {"objects": {"trajectory": uploaded.model_dump(exclude_none=True)}}


def test_the_sdk_server_reaches_its_staging_on_the_port_it_took_the_request_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A2A_PORT", "9999")  # not where this server listens
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        request.read()
        return httpx.Response(201)

    _route(monkeypatch, handle)
    write, _, _ = _on_own_staging(b"")
    payload = {"task_id": "t", "objects": {"trajectory": write.model_dump(mode="json")}}

    with TestClient(_UploadsItsTrajectory().create_app(), base_url="http://127.0.0.1:8456") as client:
        assert client.post("/ext/trajectory", json=payload).status_code == 200

    assert sent == [f"http://127.0.0.1:8456{_STAGED}"]
    assert transfers.loopback_url(write.write.url, write.write.headers) == f"http://127.0.0.1:9999{_STAGED}"
