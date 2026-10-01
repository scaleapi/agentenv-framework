"""GcsObjectStore against an in-memory stand-in for the google-cloud-storage client."""

from __future__ import annotations

import base64
import io
import json
import logging
import ssl
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import google.auth
import pytest
import requests
from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed, RetryError
from google.auth import crypt, iam
from google.auth.credentials import Signing
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.oauth2 import service_account

from agent_env.a2a_agent.object_transfer import changelog_enable_call
from agent_env.config.errors import ConfigError
from agent_env.store import GrantUnavailableError, ObjectAlreadyExistsError, ObjectNotFoundError, _google
from agent_env.store.object_store import gcs_object_store
from agent_env.store.object_store.gcs_object_store import GcsObjectStore
from tst.store import object_conformance

BUCKET = "conformance"


@dataclass
class _Stored:
    data: bytes
    content_type: str | None
    metadata: dict | None  # None, as Cloud Storage reports it, for an object with no custom metadata
    generation: int
    updated: datetime
    content_encoding: str | None = None


@dataclass
class _FakeClient:
    """The slice of ``storage.Client`` the store uses, with the failure modes it must handle."""

    api_endpoint: str = "https://storage.test"
    objects: dict[tuple[str, str], _Stored] = field(default_factory=dict)
    generation: int = 0
    uploads: int = 0
    lost_create_acks: int = 0  # creates that land, then 412 on the client's retry
    read_error: Exception | None = None
    list_error: Exception | None = None
    listed_fields: list = field(default_factory=list)
    racer: tuple[str, bytes] | None = None  # another writer creating this key just before an upload
    fail_download_after: int | None = None  # bytes written before the connection drops

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self, name)

    def list_blobs(self, bucket: str, *, prefix: str = "", fields=None, max_results=None, timeout=None, retry=None):
        if self.list_error is not None:
            raise self.list_error
        self.listed_fields.append(fields)
        names = [name for b, name in sorted(self.objects) if b == bucket and name.startswith(prefix)]
        return [_FakeBlob(self, bucket, name) for name in names[:max_results]]


@dataclass
class _FakeBucket:
    client: _FakeClient
    name: str

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self.client, self.name, name)

    def get_blob(self, name: str) -> _FakeBlob | None:
        stored = self.client.objects.get((self.name, name))
        if stored is None:
            return None
        blob = _FakeBlob(self.client, self.name, name)
        blob.metadata, blob.content_type, blob.size = stored.metadata, stored.content_type, len(stored.data)
        blob.updated, blob.generation = stored.updated, stored.generation
        blob.content_encoding = stored.content_encoding
        return blob


class _FakeBlob:
    def __init__(self, client: _FakeClient, bucket: str, name: str) -> None:
        self.client, self.bucket_name, self.name = client, bucket, name
        self.metadata: dict | None = None
        self.content_type = self.size = self.updated = self.generation = self.content_encoding = None

    def upload_from_string(self, data, content_type=None, if_generation_match=None) -> None:
        self._write(data, content_type, if_generation_match)

    def upload_from_filename(self, filename, content_type=None, if_generation_match=None) -> None:
        self._write(Path(filename).read_bytes(), content_type, if_generation_match)

    def download_as_bytes(self, *, raw_download) -> bytes:
        assert raw_download is True, "reads must return the stored bytes"
        return self._stored().data

    def open(self, mode: str, *, raw_download):
        assert mode == "rb" and raw_download is True
        assert self.generation is not None, "open() must read a generation it looked up"
        return io.BytesIO(self._stored().data)

    def generate_signed_url(self, *, version, expiration, method, content_type, headers, credentials) -> str:
        # The real library adds Content-Type and Host to the headers dict it is given.
        if headers is not None:
            if content_type is not None:
                headers["Content-Type"] = content_type
            headers["Host"] = "storage.test"
        signature = credentials.sign_bytes(f"{method}\n{self.bucket_name}/{self.name}".encode()).hex()
        signed = ";".join(sorted(headers or {}))
        return (
            f"https://storage.test/{self.bucket_name}/{quote(self.name, safe='/~')}?version={version}"
            f"&method={method}&expires={int(expiration.total_seconds())}&content-type={content_type}"
            f"&headers={signed}&signature={signature}"
        )

    def _stored(self) -> _Stored:
        if self.client.read_error is not None:
            raise self.client.read_error
        stored = self.client.objects.get((self.bucket_name, self.name))
        if stored is None:
            raise NotFound(f"No such object: {self.bucket_name}/{self.name}")
        return stored

    def _write(self, data, content_type, if_generation_match) -> None:
        self.client.uploads += 1
        key = (self.bucket_name, self.name)
        if self.client.racer is not None and self.client.racer[0] == self.name:
            self.client.objects[key] = _Stored(self.client.racer[1], None, None, 0, datetime.now(UTC))
            self.client.racer = None
        if if_generation_match == 0 and key in self.client.objects:
            raise PreconditionFailed("At least one of the pre-conditions you specified did not hold.")
        self.client.generation += 1
        self.client.objects[key] = _Stored(
            data if isinstance(data, bytes) else data.encode(),
            content_type,
            dict(self.metadata) if self.metadata else None,
            self.client.generation,
            datetime.now(UTC),
        )
        if if_generation_match == 0 and self.client.lost_create_acks:
            self.client.lost_create_acks -= 1
            raise PreconditionFailed("At least one of the pre-conditions you specified did not hold.")


def _sliced_download(blob, filename, *, download_kwargs, worker_type, max_workers) -> None:
    """Stands in for transfer_manager.download_chunks_concurrently, which reloads the object
    (404 if absent), truncates ``filename`` and writes into it."""
    assert download_kwargs == {"raw_download": True} and worker_type == transfer_manager.THREAD
    data = blob._stored().data
    with open(filename, "wb") as fh:
        if blob.client.fail_download_after is not None:
            fh.write(data[: blob.client.fail_download_after])
            raise ConnectionError("connection reset mid-download")
        fh.write(data)


@pytest.fixture(autouse=True)
def _no_real_sliced_downloads(monkeypatch):
    monkeypatch.setattr(transfer_manager, "download_chunks_concurrently", _sliced_download)


class _FakeSigner(Signing):
    """Credentials that sign without a key of their own, as through IAM."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.messages: list[bytes] = []
        self.error = error

    def sign_bytes(self, message: bytes) -> bytes:
        self.calls += 1
        self.messages.append(message)
        if self.error is not None:
            raise self.error
        return b"signed"

    @property
    def signer_email(self) -> str:
        return "signer@example.test"

    @property
    def signer(self):
        return self


class _Key(crypt.Signer):
    calls = 0

    @property
    def key_id(self):
        return None

    def sign(self, message):
        _Key.calls += 1
        return b"key-signed"


def _key_credentials() -> service_account.Credentials:
    return service_account.Credentials(_Key(), "key@example.test", "https://oauth2.example.test/token")


@pytest.fixture
def client() -> _FakeClient:
    return _FakeClient()


@pytest.fixture
def signer() -> _FakeSigner:
    return _FakeSigner()


@pytest.fixture
def store(client, signer) -> GcsObjectStore:
    return GcsObjectStore(client, BUCKET, signer=signer)


@pytest.fixture
def unsigned(client) -> GcsObjectStore:
    return GcsObjectStore(client, BUCKET)


@pytest.mark.parametrize("case", object_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store):
    case(store, "")


def test_owns_only_gs_urls_in_its_bucket(store):
    assert store.owns(f"gs://{BUCKET}/k")
    assert store.owns(f"gs://{BUCKET}/")
    assert not store.owns("gs://another-bucket/k")
    assert not store.owns(f"s3://{BUCKET}/k")
    assert not store.owns(f"{BUCKET}/k")


@pytest.mark.parametrize("url", [f"s3://{BUCKET}/k", f"file:///{BUCKET}/k", f"{BUCKET}/k", "gs:///k"])
def test_a_url_that_is_not_a_gs_object_is_refused_before_any_request(store, url):
    with pytest.raises(ValueError):
        store.get(url)


@pytest.mark.parametrize("url", [f"gs://{BUCKET}/", f"gs://{BUCKET}"])
def test_a_url_naming_no_object_is_refused_even_by_a_store_that_cannot_sign(store, unsigned, url):
    """Signed, it would be a url that lists the bucket."""
    for s in (store, unsigned):
        with pytest.raises(ValueError, match="names no object"):
            s.signed_get_url(url)
    with pytest.raises(ValueError, match="names no object"):
        store.issue_read_grant(url)


def test_explicit_url_operations_reach_another_bucket(store, tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"cross")
    foreign = "gs://other/dir/f.bin"
    assert store.put_file_at(foreign, str(source)) == foreign
    assert store.get(foreign) == b"cross"
    assert store.get_object_metadata_at(foreign).size == 5
    assert store.list_at("gs://other/dir/") == [foreign]
    assert store.list("") == []


def test_keys_keep_hash_and_query_characters(store):
    url = store.put("dir/a#b?c d.txt", b"v")
    assert url == f"gs://{BUCKET}/dir/a#b?c d.txt"
    assert store.get_object_key(url) == "dir/a#b?c d.txt"
    assert store.get(url) == b"v"


def test_directory_placeholders_are_not_listed(store, client):
    store.put("p/a.txt", b"A")
    store.put("p/sub/", b"", allow_overwrite=True)
    assert store.list("p/") == ["p/a.txt"]


def test_a_create_whose_response_was_lost_is_our_own_write(store, client):
    client.lost_create_acks = 1
    url = store.put("once", b"payload")
    assert store.get(url) == b"payload"


def test_a_lost_file_create_is_our_own_write(store, client, tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"file")
    client.lost_create_acks = 1
    assert store.get(store.put_file("once.bin", str(source))) == b"file"


def test_the_same_bytes_from_another_writer_still_refuse_a_create(store):
    GcsObjectStore(store._client, BUCKET).put("taken", b"same")
    with pytest.raises(ObjectAlreadyExistsError):
        store.put("taken", b"same")


def test_metadata_reports_how_the_stored_bytes_are_encoded(store, client):
    url = store.put("log.txt", b"\x1f\x8b...", content_type="text/plain")
    client.objects[(BUCKET, "log.txt")].content_encoding = "gzip"
    assert store.get_object_metadata_at(url).content_encoding == "gzip"
    assert store.get_object_metadata("log.txt").content_encoding == "gzip"


def test_a_create_on_an_object_another_tool_wrote_is_refused(store, client):
    client.objects[(BUCKET, "k")] = _Stored(b"x", None, None, 1, datetime.now(UTC))
    with pytest.raises(ObjectAlreadyExistsError):
        store.put("k", b"x")


def test_a_large_file_create_on_a_taken_key_is_refused_before_uploading(store, client, tmp_path):
    source = tmp_path / "f.bin"
    source.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    store.put("taken.bin", b"first")
    uploads = client.uploads
    with pytest.raises(ObjectAlreadyExistsError):
        store.put_file("taken.bin", str(source))
    assert client.uploads == uploads


@pytest.mark.parametrize("size", [4, 8 * 1024 * 1024 + 1], ids=["multipart", "resumable"])
def test_a_file_create_racing_another_writer_keeps_the_winner(store, client, tmp_path, size):
    """The pre-check is only a shortcut: the precondition still refuses a key taken after it."""
    source = tmp_path / "f.bin"
    source.write_bytes(b"x" * size)
    client.racer = ("raced.bin", b"winner")
    with pytest.raises(ObjectAlreadyExistsError):
        store.put_file("raced.bin", str(source))
    assert store.get(store.object_url("raced.bin")) == b"winner"


def test_listing_keeps_the_page_token_in_its_field_mask(store, client):
    """Without it the client would stop after the first page."""
    store.list("p/")
    store.list_at(f"gs://{BUCKET}/p/")
    assert all("nextPageToken" in fields for fields in client.listed_fields)


def test_a_failed_download_leaves_the_destination_as_it_was(store, client, tmp_path):
    url = store.put("dl.bin", b"0123456789")
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"previous")
    client.fail_download_after = 4
    with pytest.raises(ConnectionError):
        store.download_to_file(url, str(dest))
    with pytest.raises(ObjectNotFoundError):
        store.download_to_file(store.object_url("absent.bin"), str(dest))
    assert dest.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.bin"]


def test_a_download_replaces_the_destination_and_keeps_its_permissions(store, tmp_path):
    url = store.put("dl.bin", b"fresh")
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"previous")
    dest.chmod(0o600)
    store.download_to_file(url, str(dest))
    assert dest.read_bytes() == b"fresh"
    assert dest.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.bin"]


def test_a_download_names_its_partial_file_briefly(store, tmp_path):
    """A partial named after a long destination would pass the file system's name limit."""
    url = store.put("dl.bin", b"fresh")
    dest = tmp_path / ("d" * 250)
    store.download_to_file(url, str(dest))
    assert dest.read_bytes() == b"fresh"


@pytest.mark.parametrize(
    "read",
    [
        lambda s, url, tmp: s.get(url),
        lambda s, url, tmp: s.download_to_file(url, str(tmp / "out.bin")),
    ],
    ids=["get", "download_to_file"],
)
def test_a_read_error_other_than_a_missing_object_is_not_mapped(store, client, read, tmp_path):
    url = store.put("k", b"v")
    client.read_error = Forbidden("caller lacks storage.objects.get")
    with pytest.raises(Forbidden):
        read(store, url, tmp_path)


def test_urls_signed_through_iam_last_at_most_twelve_hours(store, signer):
    url = store.put("k", b"v")
    signed = store.signed_get_url(url, expires_in=10**7)
    assert "method=GET" in signed and "version=v4" in signed and "expires=43200" in signed
    assert "method=PUT" in store.signed_put_url(url) and "expires=43200" in store.signed_put_url(url, expires_in=10**7)
    assert signer.calls == 3
    with pytest.raises(GrantUnavailableError, match="at most 43200s"):
        store.issue_read_grant(url, expires_in=43201)
    assert signer.calls == 3


def test_urls_signed_with_a_key_last_up_to_seven_days(client):
    store = GcsObjectStore(client, BUCKET, signer=_key_credentials())
    assert "expires=604800" in store.signed_get_url(f"gs://{BUCKET}/k", expires_in=10**7)
    store.issue_read_grant(f"gs://{BUCKET}/k", expires_in=604800)
    with pytest.raises(GrantUnavailableError, match="at most 604800s"):
        store.issue_read_grant(f"gs://{BUCKET}/k", expires_in=604801)


def test_without_a_signer_signed_urls_are_absent_and_warn_once(unsigned, caplog):
    with caplog.at_level(logging.WARNING, logger=gcs_object_store.__name__):
        assert unsigned.signed_get_url(f"gs://{BUCKET}/k") is None
        assert unsigned.signed_put_url(f"gs://{BUCKET}/k") is None
    assert len(caplog.records) == 1
    assert "signing_service_account" in caplog.records[0].getMessage()


def test_grants_need_a_signer(unsigned):
    assert unsigned.supports_transfer_grants is False
    with pytest.raises(GrantUnavailableError, match="no signer"):
        unsigned.issue_read_grant(f"gs://{BUCKET}/k")
    with pytest.raises(GrantUnavailableError, match="no signer"):
        unsigned.issue_write_grant(f"gs://{BUCKET}/k", media_type="application/json", max_bytes=8)


def test_a_plain_http_endpoint_offers_no_grants(signer):
    """Grants are HTTPS urls; a local emulator serves plain HTTP."""
    assert GcsObjectStore(_FakeClient(api_endpoint="http://localhost:4443"), BUCKET, signer=signer).supports_transfer_grants is False


def test_read_grant_is_a_signed_get(store, signer):
    grant = store.issue_read_grant(f"gs://{BUCKET}/k", expires_in=900)
    assert store.supports_transfer_grants is True
    assert grant.kind == "http-get" and "method=GET" in grant.url and "expires=900" in grant.url
    _assert_expiry(grant.expires_at, seconds=900)
    assert grant.expires_at.microsecond == 0
    assert signer.calls == 1


def test_write_grant_signs_its_media_type_and_size_bound(store):
    grant = store.issue_write_grant(f"gs://{BUCKET}/k.json", media_type="application/json", max_bytes=64)
    assert grant.kind == "http-put" and "method=PUT" in grant.url
    assert "content-type=application/json" in grant.url
    assert "x-goog-content-length-range" in grant.url
    assert grant.headers == {"Content-Type": "application/json", "x-goog-content-length-range": "0,64"}
    _assert_expiry(grant.expires_at, seconds=3600)


def _policy(fields: dict) -> dict:
    """The policy document, after checking that it binds every signed form field: Cloud Storage
    refuses a form carrying a field no condition names."""
    document = json.loads(base64.b64decode(fields["policy"]))
    exact = {k: v for c in document["conditions"] if isinstance(c, dict) for k, v in c.items()}
    prefixes = {c[1][1:]: c[2] for c in document["conditions"] if isinstance(c, list) and c[0] == "starts-with"}
    for name, value in fields.items():
        if name in ("policy", "x-goog-signature"):
            continue
        assert exact.get(name) == value or value.startswith(prefixes[name]), name
    assert fields["x-goog-algorithm"] == "GOOG4-RSA-SHA256"
    return document


def _expiration(document: dict) -> datetime:
    return datetime.strptime(document["expiration"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def test_an_upload_policy_confines_keys_to_its_prefix_and_bounds_each_object(store, signer):
    policy = store.issue_upload_policy(f"gs://{BUCKET}/ns/run", max_object_bytes=8, expires_in=600)
    write = policy.write
    assert (write.kind, write.url, write.path_field, write.file_field) == (
        "http-post-policy", f"https://storage.test/{BUCKET}/", "key", "file"
    )
    assert write.fields["key"] == "ns/run/${filename}"
    document = _policy(write.fields)
    assert ["starts-with", "$key", "ns/run/"] in document["conditions"]
    assert ["content-length-range", 0, 8] in document["conditions"]
    assert ["starts-with", "$Content-Type", ""] in document["conditions"]
    assert {"bucket": BUCKET} in document["conditions"]
    assert not any("key" in condition for condition in document["conditions"] if isinstance(condition, dict))
    date = write.fields["x-goog-date"]
    assert write.fields["x-goog-credential"] == f"signer@example.test/{date[:8]}/auto/storage/goog4_request"
    assert write.fields["x-goog-signature"] == b"signed".hex()
    assert signer.messages == [write.fields["policy"].encode()]
    _assert_expiry(policy.expires_at, seconds=600)
    assert _expiration(document) >= policy.expires_at


def test_an_upload_policy_lasts_at_most_what_its_signer_can_sign(store, client):
    with pytest.raises(GrantUnavailableError, match="at most 43200s"):
        store.issue_upload_policy(f"gs://{BUCKET}/ns/", max_object_bytes=8, expires_in=43201)
    keyed = GcsObjectStore(client, BUCKET, signer=_key_credentials())
    keyed.issue_upload_policy(f"gs://{BUCKET}/ns/", max_object_bytes=8, expires_in=604800)
    with pytest.raises(GrantUnavailableError, match="at most 604800s"):
        keyed.issue_upload_policy(f"gs://{BUCKET}/ns/", max_object_bytes=8, expires_in=604801)


def test_without_a_signer_there_is_no_upload_policy(unsigned):
    with pytest.raises(GrantUnavailableError, match="no signer"):
        unsigned.issue_upload_policy(f"gs://{BUCKET}/ns/", max_object_bytes=8, expires_in=600)
    assert unsigned.signed_post(f"gs://{BUCKET}/ns/") is None


def test_signed_post_is_shaped_as_the_s3_store_s_and_clamped_to_the_signer(store, client):
    post = store.signed_post(f"gs://{BUCKET}/snap/", expires_in=10**7)
    assert set(post) == {"url", "fields"} and post["fields"]["key"] == "snap/${filename}"
    document = _policy(post["fields"])
    assert _expiration(document) <= datetime.now(UTC) + timedelta(seconds=43200)
    assert not any(isinstance(c, list) and c[0] == "content-length-range" for c in document["conditions"])
    slashless = _policy(store.signed_post(f"gs://{BUCKET}/snap")["fields"])
    assert ["starts-with", "$key", "snap/"] in slashless["conditions"]
    root = store.signed_post(f"gs://{BUCKET}/")["fields"]
    assert root["key"] == "${filename}" and ["starts-with", "$key", ""] in _policy(root)["conditions"]
    bounded = _policy(store.signed_post(f"gs://{BUCKET}/snap/", max_bytes=5)["fields"])
    assert ["content-length-range", 0, 5] in bounded["conditions"]
    keyed = GcsObjectStore(client, BUCKET, signer=_key_credentials())
    expiration = _expiration(_policy(keyed.signed_post(f"gs://{BUCKET}/snap/", expires_in=10**7)["fields"]))
    assert datetime.now(UTC) + timedelta(seconds=43200) < expiration <= datetime.now(UTC) + timedelta(seconds=604800)


def test_changelog_capture_gets_the_object_form(store):
    only_objects = {"request": {"oneOf": [{"required": ["write_namespace"]}]}}
    call = changelog_enable_call(
        only_objects, store, agent_name="solver", namespace_url=f"gs://{BUCKET}/ns", expires_in=600,
        sandbox_type="modal",
    )
    assert call.mode == "objects"
    grant = call.payload["write_namespace"]
    assert grant["root_path"] == "ns" and grant["write"]["kind"] == "http-post-policy"


class _Credentials:
    """ADC that cannot sign, as a user login or a metadata-server identity."""

    quota_project_id = "billed-project"


class _IamSigner:
    """Stands in for google.auth.iam.Signer, which calls signBlob."""

    made: list[tuple] = []
    requests: list = []
    failures: list[Exception] = []

    def __init__(self, request, credentials, email) -> None:
        _IamSigner.made.append((credentials, email))
        _IamSigner.requests.append(request)
        self.calls = 0

    def sign(self, message) -> bytes:
        self.calls += 1
        if _IamSigner.failures:
            raise _IamSigner.failures.pop(0)
        return b"iam-signed"


@pytest.fixture
def adc(monkeypatch):
    built: list[dict] = []
    credentials = _Credentials()
    monkeypatch.setattr(google.auth, "default", lambda scopes: (credentials, "adc-project"))
    monkeypatch.setattr(storage, "Client", lambda **kwargs: built.append(kwargs) or _FakeClient())
    monkeypatch.setattr(iam, "Signer", _IamSigner)
    monkeypatch.setattr(_google, "RETRY", _google.RETRY.with_delay(initial=0.01, maximum=0.02).with_timeout(1))
    _IamSigner.made, _IamSigner.requests, _IamSigner.failures = [], [], []
    return credentials, built


def test_from_config_uses_adc_and_its_project(adc):
    credentials, built = adc
    store = GcsObjectStore.from_config(bucket=BUCKET)
    assert built == [{"project": "adc-project", "credentials": credentials}]
    assert store.supports_transfer_grants is False
    GcsObjectStore.from_config(bucket=BUCKET, project="explicit")
    assert built[1]["project"] == "explicit"


def test_the_iam_signer_uses_one_pooled_transport(adc, monkeypatch):
    pooled = []
    monkeypatch.setattr(_google, "pooled_request", lambda: pooled.append("pooled") or "pooled")
    GcsObjectStore.from_config(bucket=BUCKET, signing_service_account="signer@example.test")
    assert pooled == ["pooled"] and _IamSigner.requests == ["pooled"]


def test_from_config_signs_as_the_service_account_after_one_check(adc):
    credentials, _ = adc
    store = GcsObjectStore.from_config(bucket=BUCKET, signing_service_account="signer@example.test")
    assert _IamSigner.made == [(credentials, "signer@example.test")]
    assert store.supports_transfer_grants is True
    assert store._signer.signer.calls == 1
    store.signed_get_url(f"gs://{BUCKET}/k")
    assert store._signer.signer.calls == 2


def _transport_error(cause: Exception) -> TransportError:
    error = TransportError(cause)
    error.__cause__ = cause
    return error


@pytest.mark.parametrize("cause", [requests.ConnectionError("reset"), requests.ReadTimeout("slow")], ids=["reset", "timeout"])
def test_signing_retries_a_dropped_connection(adc, cause):
    store = GcsObjectStore.from_config(bucket=BUCKET, signing_service_account="signer@example.test")
    _IamSigner.failures = [_transport_error(cause)]
    assert store.signed_get_url(f"gs://{BUCKET}/k")


@pytest.mark.parametrize(
    "error",
    [
        TransportError("Error calling the IAM signBlob API: 403 denied"),
        _transport_error(requests.exceptions.SSLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))),
    ],
    ids=["denied", "certificate-rejected"],
)
def test_signing_does_not_retry_a_failure_that_would_repeat(adc, error):
    store = GcsObjectStore.from_config(bucket=BUCKET, signing_service_account="signer@example.test")
    _IamSigner.failures = [error, error]
    with pytest.raises(TransportError):
        store.signed_get_url(f"gs://{BUCKET}/k")
    assert _IamSigner.failures == [error]


def test_from_config_refuses_a_service_account_it_cannot_sign_as(adc):
    _IamSigner.failures = [TransportError("Permission 'iam.serviceAccounts.signBlob' denied")]
    with pytest.raises(ConfigError, match=r"Cannot sign as signer@example.test \(requests are billed to quota project 'billed-project'\).*signBlob"):
        GcsObjectStore.from_config(bucket=BUCKET, signing_service_account="signer@example.test")


def test_from_config_refuses_an_empty_service_account_before_looking_for_credentials(monkeypatch):
    monkeypatch.setattr(google.auth, "default", lambda scopes: pytest.fail("looked for credentials first"))
    with pytest.raises(ConfigError, match="signing_service_account is empty"):
        GcsObjectStore.from_config(bucket=BUCKET, signing_service_account=" ")


@pytest.mark.parametrize(
    "error",
    [DefaultCredentialsError("Your default credentials were not found."), IsADirectoryError(21, "Is a directory")],
    ids=["none", "a-directory"],
)
def test_from_config_without_credentials_is_a_config_error(monkeypatch, error):
    def no_adc(scopes):
        raise error

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(ConfigError, match="No Google credentials for gs://conformance"):
        GcsObjectStore.from_config(bucket=BUCKET)


@pytest.mark.parametrize(
    "error",
    [
        NotFound("The specified bucket does not exist."),
        ValueError("Bucket names must start and end with a number or letter."),
        RetryError("Timeout of 20.0s exceeded", cause=requests.ConnectionError("unreachable")),
        RefreshError("invalid_grant: Bad Request"),
    ],
    ids=["missing", "malformed", "unreachable", "expired-login"],
)
def test_from_config_refuses_a_bucket_it_cannot_list(adc, monkeypatch, error):
    missing = _FakeClient(list_error=error)
    monkeypatch.setattr(storage, "Client", lambda **kwargs: missing)
    with pytest.raises(ConfigError, match="Cannot list gs://typo"):
        GcsObjectStore.from_config(bucket="typo")


@pytest.mark.parametrize(
    "error", [TransportError("Gaia id not found"), requests.ConnectionError("reset")], ids=["refused", "unreachable"]
)
def test_from_config_checks_adc_that_sign_through_iam(monkeypatch, error):
    """Impersonated ADC sign over their own session, so a network error reaches the check unwrapped."""
    impersonated = _FakeSigner(error=error)
    monkeypatch.setattr(google.auth, "default", lambda scopes: (impersonated, None))
    monkeypatch.setattr(storage, "Client", lambda **kwargs: _FakeClient())
    with pytest.raises(ConfigError, match="Cannot sign as signer@example.test"):
        GcsObjectStore.from_config(bucket=BUCKET)


def test_from_config_signs_locally_with_a_key_and_checks_nothing(monkeypatch):
    key = _key_credentials()
    monkeypatch.setattr(google.auth, "default", lambda scopes: (key, None))
    monkeypatch.setattr(storage, "Client", lambda **kwargs: _FakeClient())
    _Key.calls = 0
    store = GcsObjectStore.from_config(bucket=BUCKET)
    assert _Key.calls == 0
    store.signed_get_url(f"gs://{BUCKET}/k")
    assert _Key.calls == 1


def _assert_expiry(expires_at: datetime, *, seconds: int) -> None:
    expected = datetime.now(UTC) + timedelta(seconds=seconds)
    assert expires_at.tzinfo is UTC
    assert abs((expires_at - expected).total_seconds()) < 1.5
