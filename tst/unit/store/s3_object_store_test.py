"""S3ObjectStore runs the full ObjectStore conformance suite against moto."""

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from moto import mock_aws

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.store import S3ObjectStore, set_object_store
from tst.store import object_conformance

BUCKET = "conformance"


@pytest.fixture
def store():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3ObjectStore.from_config(bucket=BUCKET, region="us-east-1")


@pytest.mark.parametrize("case", object_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store):
    case(store, "")


def test_metadata_reports_how_the_stored_bytes_are_encoded(store):
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=BUCKET, Key="log.txt", Body=b"\x1f\x8b...", ContentEncoding="gzip"
    )
    assert store.get_object_metadata("log.txt").content_encoding == "gzip"
    assert store.get_object_metadata_at(f"s3://{BUCKET}/log.txt").content_encoding == "gzip"


def test_owns_only_its_own_bucket(store):
    """The explicit-url reads still reach other buckets; ownership is this bucket only."""
    assert store.owns(f"s3://{BUCKET}/k")
    assert not store.owns("s3://another-bucket/k")
    assert not store.owns(f"{BUCKET}/k")


@pytest.mark.parametrize(
    ("operation", "read"),
    [
        ("get_object", lambda s, url, tmp: s.get(url)),
        ("get_object", lambda s, url, tmp: s.open(url)),
        ("head_object", lambda s, url, tmp: s.download_to_file(url, str(tmp / "out.bin"))),
    ],
    ids=["get", "open", "download_to_file"],
)
def test_a_read_error_other_than_a_missing_object_is_not_mapped(operation, read, tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_client_error(operation, service_error_code="AccessDenied", http_status_code=403)
    with stubber, pytest.raises(ClientError):
        read(S3ObjectStore(client, BUCKET), f"s3://{BUCKET}/k", tmp_path)


def test_the_client_keeps_a_connection_for_every_default_executor_thread(store):
    assert store._s3.meta.config.max_pool_connections == 32


@pytest.mark.parametrize("url", [f"s3://{BUCKET}", f"s3://{BUCKET}/"])
def test_a_bucket_url_names_no_object(url):
    """Without a HEAD: S3 refuses an empty key as a malformed request, not as a missing object."""
    client = boto3.client("s3", region_name="us-east-1")
    with Stubber(client):  # nothing queued, so any request fails the test
        store = S3ObjectStore(client, BUCKET)
        assert store.get_object_metadata_at(url) is None
        set_object_store(store)
        with pytest.raises(ValueError, match="does not exist"):
            FileArtifact.put_existing(id="bucket-root", description="d", object_url=url.rstrip("/"))
