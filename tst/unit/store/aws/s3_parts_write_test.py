"""A parts write on S3: a multipart upload whose parts the grant presigns, made into the object only when the
stored parts are the object the receiver reported."""

import math
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from agentenv_protocol.transfers import HttpPartsPutGrant, HttpPutGrant, Uploaded, WriteObject
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws

from agent_env.store import GrantUnavailableError, UploadFailedError
from agent_env.store.object_store import s3_object_store
from agent_env.store.object_store.s3_object_store import S3ObjectStore

BUCKET = "parts"
MIB = 1024 * 1024
KINDS = frozenset({"http-put", "http-put-parts"})


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def store(s3, monkeypatch):
    monkeypatch.setattr(s3_object_store, "PART_BYTES", 5 * MIB)
    return S3ObjectStore.from_config(bucket=BUCKET, region="us-east-1")


def _begin(store, key="snapshots/github.zip", max_bytes=12 * MIB):
    return store.begin_write(store.object_url(key), media_type="application/zip", max_bytes=max_bytes,
                             kinds=KINDS, expires_in=900)


def _receive(s3, grant: WriteObject, data: bytes, skip: int | None = None) -> None:
    """Store what a receiver's PUTs to the part URLs would: moto serves no HTTP, so each range goes up through
    the client, addressed by the upload and part number its URL was signed for."""
    write = grant.write
    for number, url in enumerate(write.urls[: max(1, math.ceil(len(data) / write.part_bytes))], start=1):
        if number == skip:
            continue
        query, key = parse_qs(urlsplit(url).query), urlsplit(url).path.lstrip("/")
        assert int(query["partNumber"][0]) == number
        s3.upload_part(Bucket=BUCKET, Key=key, UploadId=query["uploadId"][0], PartNumber=number,
                       Body=data[(number - 1) * write.part_bytes:number * write.part_bytes])


def _upload_is_gone(s3, write) -> bool:
    try:
        s3.list_parts(Bucket=BUCKET, Key="snapshots/github.zip", UploadId=write._upload_id)
    except ClientError as e:
        return e.response["Error"]["Code"] == "NoSuchUpload"
    return False


def test_a_parts_write_presigns_one_url_per_part_of_its_largest_object(store):
    write = _begin(store)

    grant = write.grant.write
    assert isinstance(grant, HttpPartsPutGrant)
    assert (grant.part_bytes, len(grant.urls)) == (5 * MIB, 3)
    queries = [parse_qs(urlsplit(url).query) for url in grant.urls]
    assert [q["partNumber"][0] for q in queries] == ["1", "2", "3"]
    assert len({q["uploadId"][0] for q in queries}) == 1
    write.abort()


def test_completing_makes_the_object_from_its_parts(s3, store):
    data = bytes(range(256)) * (11 * MIB // 256)
    write = _begin(store)
    _receive(s3, write.grant, data)

    write.complete(Uploaded(size_bytes=len(data)))

    head = s3.head_object(Bucket=BUCKET, Key="snapshots/github.zip")
    assert (head["ContentLength"], head["ContentType"]) == (len(data), "application/zip")
    assert s3.get_object(Bucket=BUCKET, Key="snapshots/github.zip")["Body"].read() == data


@pytest.mark.parametrize(("skip", "reported"), [(2, 11 * MIB), (None, 11 * MIB + 1), (None, 6 * MIB)],
                         ids=["missing-part", "longer-than-sent", "shorter-than-sent"])
def test_parts_that_are_not_the_reported_object_are_refused_and_discarded(s3, store, skip, reported):
    write = _begin(store)
    _receive(s3, write.grant, b"x" * 11 * MIB, skip=skip)

    with pytest.raises(UploadFailedError):
        write.complete(Uploaded(size_bytes=reported))

    assert _upload_is_gone(s3, write)
    assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET)


def test_an_upload_over_the_grants_limit_is_refused_and_discarded(s3, store):
    write = _begin(store)
    _receive(s3, write.grant, b"x" * 13 * MIB)

    with pytest.raises(UploadFailedError, match="limit"):
        write.complete(Uploaded(size_bytes=13 * MIB))

    assert _upload_is_gone(s3, write)
    assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET)


def test_leaving_the_write_unfinished_aborts_it(s3, store):
    with _begin(store) as write:
        _receive(s3, write.grant, b"x" * MIB)
    assert _upload_is_gone(s3, write)


def test_a_receiver_without_parts_gets_one_put_bounded_by_one_upload(store):
    write = store.begin_write(store.object_url("t.zip"), media_type="application/zip", max_bytes=16 * 1024 * MIB,
                              kinds=frozenset({"http-put"}))
    assert isinstance(write.grant.write, HttpPutGrant)
    assert write.grant.max_bytes == S3ObjectStore.max_single_upload_bytes


def test_a_bucket_that_refuses_to_start_the_upload_issues_no_grant(s3):
    store = S3ObjectStore.from_config(bucket="absent", region="us-east-1")
    with pytest.raises(GrantUnavailableError, match="NoSuchBucket"):
        store.begin_write(store.object_url("t.zip"), media_type="application/zip", max_bytes=MIB, kinds=KINDS)


def test_a_store_that_refuses_to_finish_fails_the_write_and_discards_it(s3, store, monkeypatch):
    write = _begin(store)
    _receive(s3, write.grant, b"x" * MIB)

    def denied(**_):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "CompleteMultipartUpload")

    monkeypatch.setattr(write._s3, "complete_multipart_upload", denied)
    with pytest.raises(UploadFailedError, match="AccessDenied"):
        write.complete(Uploaded(size_bytes=MIB))
    assert _upload_is_gone(s3, write)


def test_an_abort_that_cannot_reach_s3_does_not_hide_why_the_write_failed(s3, store, monkeypatch):
    write = _begin(store)
    _receive(s3, write.grant, b"x" * MIB)

    def denied(**_):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "CompleteMultipartUpload")

    def unreachable(**_):
        raise EndpointConnectionError(endpoint_url="https://s3.us-east-1.amazonaws.com")

    monkeypatch.setattr(write._s3, "complete_multipart_upload", denied)
    monkeypatch.setattr(write._s3, "abort_multipart_upload", unreachable)
    with pytest.raises(UploadFailedError, match="AccessDenied"):
        write.complete(Uploaded(size_bytes=MIB))


@pytest.mark.parametrize(("bucket", "host"), [
    ("parts", "parts.s3.us-west-2.amazonaws.com"),
    ("dotted.bucket", "s3.us-west-2.amazonaws.com"),
], ids=["virtual-hosted", "dotted-name"])
def test_grants_are_signed_against_the_buckets_region(s3, bucket, host):
    store = S3ObjectStore.from_config(bucket=bucket, region="us-west-2")
    assert urlsplit(store.issue_read_grant(f"s3://{bucket}/k.zip", expires_in=900).url).netloc == host
