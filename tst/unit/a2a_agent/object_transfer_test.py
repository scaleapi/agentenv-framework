"""The core side of object transfer: choosing a form from the card, descriptors, responses."""

import httpx
import pytest
from agentenv_protocol.a2a_agent import TrajectoryObjectsResponse
from agentenv_protocol.transfers import TRANSFER_STALL_BUDGET_SECONDS
from pydantic import ValidationError

from agent_env.a2a_agent import object_transfer
from agent_env.a2a_agent.object_transfer import (
    FetchedTrajectory,
    TrajectoryUpload,
    ObjectLimits,
    choose_transfer,
    fetch_trajectory,
    namespace_grant,
    parse_response,
    read_object,
    read_objects_under,
    write_object,
)
from agent_env.store.object_store import DEFAULT_GRANT_LIFETIME_SECONDS
from tst.util.granting_object_store import GrantingObjectStore


SDK_TASK_GET = {"request": {"required": ["task_id"], "oneOf": [{}, {"required": ["objects"]}]}}
FLEET_TASK_GET = {"request": {"required": ["task_id"], "optional": ["trajectory_s3_prefix"]}}
FLEET_CONTEXT_GET = {
    "request": {
        "oneOf": [{"required": ["task_id"]}, {"required": ["context_id"]}],
        "optional": ["trajectory_s3_prefix"],
    }
}
OBJECTS_ONLY_GET = {"request": {"required": ["task_id", "objects"]}}
TASK = {"objects": ("task_id", "objects"), "legacy": ("task_id",)}
CONTEXT = {"objects": ("context_id", "objects"), "legacy": ("context_id",)}
TASK_OBJECTS = {"objects": ("task_id", "objects")}
TASK_LEGACY = {"legacy": ("task_id",)}


@pytest.mark.parametrize(
    "method, fields, grants, expected",
    [
        (SDK_TASK_GET, TASK, True, "objects"),
        (SDK_TASK_GET, TASK, False, "legacy"),
        (FLEET_TASK_GET, TASK, True, "legacy"),
        (None, TASK, True, "legacy"),
        ({"method": "POST"}, TASK, True, "legacy"),
        (OBJECTS_ONLY_GET, TASK, True, "objects"),
        (OBJECTS_ONLY_GET, TASK, False, None),
        (FLEET_CONTEXT_GET, CONTEXT, True, "legacy"),
        (SDK_TASK_GET, CONTEXT, True, None),
        (FLEET_TASK_GET, TASK_OBJECTS, True, None),
        ({"method": "POST"}, TASK_OBJECTS, True, None),
        (SDK_TASK_GET, TASK_LEGACY, True, "legacy"),
        (OBJECTS_ONLY_GET, TASK_LEGACY, True, None),
    ],
    ids=[
        "sdk-granting-store",
        "sdk-store-without-grants",
        "legacy-card",
        "unadvertised-method",
        "undeclared-request",
        "objects-only",
        "objects-only-store-without-grants",
        "context-on-legacy-card",
        "context-on-task-only-card",
        "object-form-only-on-legacy-card",
        "object-form-only-on-undeclared-request",
        "legacy-form-only-on-sdk-card",
        "legacy-form-only-on-objects-only-card",
    ],
)
def test_choose_transfer(tmp_path, method, fields, grants, expected):
    store = GrantingObjectStore(str(tmp_path))
    store.supports_transfer_grants = grants
    assert choose_transfer(method, store=store, sandbox_type="local", **fields) == expected


def test_parse_response_ignores_fields_it_does_not_know():
    body = {"objects": {"trajectory": {"size_bytes": 3, "etag": "e"}, "extra": 1}, "ok": True}

    parsed = parse_response(TrajectoryObjectsResponse, body, operation="trajectory get")

    assert parsed.objects.trajectory.size_bytes == 3
    with pytest.raises(RuntimeError, match="trajectory get returned an invalid object response"):
        parse_response(TrajectoryObjectsResponse, {"ok": True}, operation="trajectory get")


def test_write_object_carries_the_grant_signed_for_it(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    url = store.object_url("traj/trajectory-1.json")

    descriptor = write_object(store, url, media_type="application/json", max_bytes=10)

    assert (descriptor.media_type, descriptor.max_bytes) == ("application/json", 10)
    assert descriptor.write.headers == {"Content-Type": "application/json"}


def test_read_object_is_bounded_by_the_stored_size(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    url = store.put("skill/SKILL.md", b"# skill", content_type="Text/Markdown; charset=utf-8")

    descriptor = read_object(store, url)

    assert (descriptor.media_type, descriptor.size_bytes, descriptor.max_bytes) == (
        "text/markdown", 7, 7,
    )
    with pytest.raises(ValueError, match="size_bytes must not exceed max_bytes"):
        read_object(store, url, max_bytes=6)
    with pytest.raises(ValueError, match="metadata is unavailable"):
        read_object(store, store.object_url("skill/missing.md"))


@pytest.mark.parametrize("stored", ["markdown", "*/*", "text/markdown charset=utf-8", ""])
def test_read_object_describes_an_unusable_stored_type_as_opaque_bytes(tmp_path, stored):
    store = GrantingObjectStore(str(tmp_path))
    url = store.put("skill/SKILL.md", b"# skill", content_type=stored)

    assert read_object(store, url).media_type == "application/octet-stream"


def test_read_objects_under_keys_each_object_by_its_relative_path(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    store.put("skill/SKILL.md", b"# skill")
    store.put("skill/ref/notes#1?.txt", b"notes")
    store.put("skill-other/SKILL.md", b"elsewhere")
    limits = ObjectLimits(max_objects=2, max_object_bytes=7, max_total_bytes=12)

    described = read_objects_under(store, store.object_url("skill"), limits=limits)

    assert [(path, descriptor.size_bytes) for path, descriptor in described] == [
        ("SKILL.md", 7),
        ("ref/notes#1?.txt", 5),
    ]


def test_read_objects_under_keeps_only_what_select_admits(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    for name in ("000002.tar", "000000.tar", "000001.tar"):
        store.put(f"ns/{name}", b"x", content_type="application/x-tar")
    limits = ObjectLimits(max_objects=2, max_object_bytes=1, max_total_bytes=2)

    described = read_objects_under(
        store,
        store.object_url("ns"),
        limits=limits,
        media_type="application/octet-stream",
        select=lambda path: path != "000002.tar",
    )

    assert [path for path, _ in described] == ["000000.tar", "000001.tar"]
    assert {descriptor.media_type for _, descriptor in described} == {"application/octet-stream"}
    assert store.granted == [store.object_url(f"ns/00000{n}.tar") for n in (0, 1)]


def test_read_objects_under_counts_stored_bytes_as_an_uploader_does(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    store.put("ns/000000.tar", b"")
    store.put("ns/000001.tar", b"ten bytes!")

    described = read_objects_under(
        store,
        store.object_url("ns"),
        limits=ObjectLimits(max_objects=2, max_object_bytes=10, max_total_bytes=10),
    )

    assert [descriptor.max_bytes for _, descriptor in described] == [1, 10]


@pytest.mark.parametrize(
    "limits, message",
    [
        (ObjectLimits(max_objects=1, max_object_bytes=7, max_total_bytes=12), "limit is 1"),
        (ObjectLimits(max_objects=2, max_object_bytes=6, max_total_bytes=12), "6-byte object"),
        (ObjectLimits(max_objects=2, max_object_bytes=7, max_total_bytes=11), "11-byte limit"),
    ],
    ids=["objects", "object-bytes", "total-bytes"],
)
def test_read_objects_under_refuses_a_prefix_over_its_limits(tmp_path, limits, message):
    store = GrantingObjectStore(str(tmp_path))
    store.put("skill/SKILL.md", b"# skill")
    store.put("skill/ref/notes.txt", b"notes")

    with pytest.raises(ValueError, match=message):
        read_objects_under(store, store.object_url("skill"), limits=limits)


@pytest.mark.asyncio
async def test_fetch_trajectory_waits_for_the_agent_upload_it_granted(tmp_path, monkeypatch):
    timeouts = []
    real_client = httpx.AsyncClient

    def answer(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"]["read"])
        if b"objects" in request.content:
            return httpx.Response(200, json={"objects": {"trajectory": {"size_bytes": 2}}})
        return httpx.Response(200, json={"trajectory": []})

    monkeypatch.setattr(
        object_transfer.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(answer), **kwargs),
    )
    store = GrantingObjectStore(str(tmp_path))
    url = store.object_url("t/trajectory.json")
    upload = TrajectoryUpload.to(store, url)
    endpoint = "https://agent.test/ext/trajectory"

    inline = await fetch_trajectory(endpoint, {"task_id": "t"})
    uploaded = await fetch_trajectory(endpoint, {"task_id": "t"}, upload=upload)
    await fetch_trajectory(endpoint, {"task_id": "t"}, upload=upload, timeout=5)

    assert inline == FetchedTrajectory(inline=[])
    assert uploaded == FetchedTrajectory(object_url=url)
    assert timeouts == [
        object_transfer.REPLY_TIMEOUT_SECONDS,
        object_transfer.TRANSFER_TIMEOUT_SECONDS,
        5,
    ]


@pytest.mark.asyncio
async def test_fetch_trajectory_reports_a_legacy_agents_own_upload(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        object_transfer.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"trajectory_s3_prefix": "s3://b/t/"})
            ),
            **kwargs,
        ),
    )

    fetched = await fetch_trajectory("https://agent.test/ext/trajectory", {"task_id": "t"})

    assert fetched == FetchedTrajectory(legacy_prefix="s3://b/t/")


def test_the_transfer_time_budget_nests():
    """A stalled connection is retried inside agent-env's wait, which ends before the grant."""
    assert (
        TRANSFER_STALL_BUDGET_SECONDS
        < object_transfer.TRANSFER_TIMEOUT_SECONDS
        < DEFAULT_GRANT_LIFETIME_SECONDS
    )


def test_a_write_grant_promises_no_more_than_one_upload_can_create(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    store.max_single_upload_bytes = 100
    url = store.object_url("snapshot/workspace")

    assert write_object(store, url, media_type="application/json", max_bytes=1_000).max_bytes == 100
    assert write_object(store, url, media_type="application/json", max_bytes=10).max_bytes == 10


def test_namespace_grant_wraps_the_stores_upload_policy_in_the_limits(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    limits = ObjectLimits(max_objects=20, max_object_bytes=1024, max_total_bytes=8192)

    grant = namespace_grant(store, store.object_url("changelog/run-1"), limits=limits, expires_in=60)

    assert grant.root_path == "changelog/run-1"
    assert (grant.max_objects, grant.max_object_bytes, grant.max_total_bytes) == (20, 1024, 8192)
    assert grant.write.fields["key"] == "changelog/run-1/${filename}"


@pytest.mark.parametrize(
    "limits",
    [
        ObjectLimits(max_objects=0, max_object_bytes=10, max_total_bytes=10),
        ObjectLimits(max_objects=1, max_object_bytes=0, max_total_bytes=10),
        ObjectLimits(max_objects=1, max_object_bytes=10, max_total_bytes=0),
    ],
)
def test_namespace_grant_rejects_invalid_limits(tmp_path, limits):
    store = GrantingObjectStore(str(tmp_path))
    with pytest.raises(ValidationError):
        namespace_grant(store, store.object_url("changelog/run-1"), limits=limits, expires_in=60)


@pytest.mark.asyncio
async def test_fetch_trajectory_reads_an_answer_that_is_not_an_object_as_none(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        object_transfer.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[1, 2])),
            **kwargs,
        ),
    )

    fetched = await fetch_trajectory("https://agent.test/ext/trajectory", {"task_id": "t"})

    assert fetched == FetchedTrajectory()
