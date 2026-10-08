"""Google Cloud Storage implementation of the ObjectStore interface (the ``gcp`` extra)."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import stat
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import google.auth
import requests
from agentenv_protocol.transfers import HttpGetGrant, HttpPostPolicyGrant, HttpPutGrant
from google.api_core.exceptions import GoogleAPIError, NotFound, PreconditionFailed, RetryError
from google.auth import iam
from google.auth.credentials import Signing
from google.auth.exceptions import GoogleAuthError, TransportError
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.cloud.storage.retry import DEFAULT_RETRY
from google.oauth2 import service_account

from agent_env.config.errors import ConfigError
from agent_env.store import _google
from agent_env.store.base import GrantUnavailableError, ObjectAlreadyExistsError, ObjectNotFoundError
from agent_env.store.object_store.object_store import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_GRANT_LIFETIME_SECONDS,
    ObjectMetadata,
    ObjectStore,
    UploadPolicy,
    grant_lifetime,
)

logger = logging.getLogger(__name__)

_SCHEME = "gs://"
# A key's V4 signature lasts up to seven days; Google guarantees one made through IAM signBlob,
# which signs with a system-managed key, for twelve hours.
_MAX_KEY_SIGNED_SECONDS = 7 * 24 * 60 * 60
_MAX_IAM_SIGNED_SECONDS = 12 * 60 * 60
_PROBE_TIMEOUT_SECONDS = 20
# The client creates a larger object in a resumable session, which meets a taken key only at its
# last chunk.
_MULTIPART_LIMIT = 8 * 1024 * 1024
# Tags each write-once create, so a 412 for a create the client retried after losing its
# response can be told apart from an object another writer put there first.
_WRITE_ID = "agentenv-write-id"
_DOWNLOAD_WORKERS = 4


class _IamSigner(Signing):
    """Signs as a service account through IAM signBlob over one pooled session, retrying dropped
    connections. google-auth's impersonated credentials open a connection per signature."""

    def __init__(self, credentials, email: str) -> None:
        self._iam = iam.Signer(_google.pooled_request(), credentials, email)
        self._email = email

    def sign_bytes(self, message: bytes) -> bytes:
        try:
            return _google.RETRY(self._iam.sign)(message)
        except RetryError as e:
            raise TransportError(f"IAM signed nothing as {self._email} within the retry window: {e.cause}") from e

    @property
    def signer_email(self) -> str:
        return self._email

    @property
    def signer(self) -> iam.Signer:
        return self._iam


class GcsObjectStore(ObjectStore):
    """ObjectStore backed by a Cloud Storage bucket (injected client, config-free).

    ``signer`` signs URLs and grants: credentials that hold a key, or ones that sign through IAM.
    Without one the store issues neither. Its reads return an object's bytes as stored."""

    max_single_upload_bytes = 5 * 1024**4  # one Cloud Storage object

    def __init__(
        self,
        client,
        bucket: str,
        *,
        signer: Signing | None = None,
        grant_lifetime_seconds: int = DEFAULT_GRANT_LIFETIME_SECONDS,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._signer = signer
        # Grants are HTTPS URLs, so a plain-HTTP endpoint (a local emulator) issues none.
        self.supports_transfer_grants = signer is not None and client.api_endpoint.startswith("https://")
        self._max_signed_seconds = (
            _MAX_KEY_SIGNED_SECONDS if isinstance(signer, service_account.Credentials) else _MAX_IAM_SIGNED_SECONDS
        )
        self.grant_lifetime_seconds = grant_lifetime(grant_lifetime_seconds, most=self._max_signed_seconds)
        self._warned_unsigned = False

    @classmethod
    def from_config(
        cls,
        *,
        bucket: str,
        project: str | None = None,
        signing_service_account: str | None = None,
        grant_lifetime_seconds: int = DEFAULT_GRANT_LIFETIME_SECONDS,
    ) -> GcsObjectStore:
        """Authenticate through Application Default Credentials. With ``signing_service_account``
        the store always signs as that account through IAM, which needs
        ``iam.serviceAccounts.signBlob`` on it; without it, credentials that can sign do. Lists
        the bucket, and signs once when IAM signs, so a missing bucket or a signer that cannot sign
        fails here rather than at the first read or signed URL."""
        if signing_service_account is not None and not signing_service_account.strip():
            raise ConfigError("signing_service_account is empty; omit it to sign with the credentials themselves")
        try:
            credentials, default_project = google.auth.default(scopes=_google.SCOPES)
        except (GoogleAuthError, OSError) as e:
            raise ConfigError(f"No Google credentials for gs://{bucket}: {e}") from e
        client = storage.Client(project=project or default_project, credentials=credentials)
        if signing_service_account:
            signer = _IamSigner(credentials, signing_service_account)
        else:
            signer = credentials if isinstance(credentials, Signing) else None
        probe = DEFAULT_RETRY.with_timeout(_PROBE_TIMEOUT_SECONDS)
        try:
            next(iter(client.list_blobs(bucket, max_results=1, timeout=_PROBE_TIMEOUT_SECONDS, retry=probe)), None)
        except (GoogleAPIError, GoogleAuthError, requests.RequestException, ValueError) as e:
            raise ConfigError(f"Cannot list gs://{bucket}: {e}") from e
        if signer is not None and not isinstance(signer, service_account.Credentials):
            try:
                signer.sign_bytes(b"agent-env signing check")
            except (GoogleAPIError, GoogleAuthError, requests.RequestException) as e:
                quota = getattr(credentials, "quota_project_id", None)
                billed = f" (requests are billed to quota project {quota!r})" if quota else ""
                raise ConfigError(f"Cannot sign as {signer.signer_email}{billed}: {e}") from e
        return cls(client, bucket, signer=signer, grant_lifetime_seconds=grant_lifetime_seconds)

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        allow_overwrite: bool = False,
    ) -> str:
        url = self.object_url(key)
        logger.info("Uploading %d bytes to %s", len(data), url)
        blob = self._blob(url)
        if allow_overwrite:
            blob.upload_from_string(data, content_type=content_type)
        else:
            self._create(url, blob, lambda: blob.upload_from_string(
                data, content_type=content_type, if_generation_match=0
            ))
        return url

    def put_file(self, key: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self._put_file(self.object_url(key), file_path, content_type)

    def put_file_at(self, object_url: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self._put_file(object_url, file_path, content_type)

    def get(self, object_url: str) -> bytes:
        logger.info("Downloading from %s", object_url)
        with _missing_as_not_found(object_url):
            return self._blob(object_url).download_as_bytes(raw_download=True)

    def download_to_file(self, object_url: str, dest_path: str) -> None:
        """Fetch ranges concurrently, each retried on its own, check the whole object's crc32c,
        and rename into place: a failed download leaves neither a partial file nor a changed
        destination, and a replaced destination keeps its permissions."""
        logger.info("Streaming %s to %s", object_url, dest_path)
        blob = self._blob(object_url)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(f".{uuid.uuid4().hex[:16]}.part")
        try:
            with _missing_as_not_found(object_url):
                transfer_manager.download_chunks_concurrently(
                    blob,
                    str(partial),
                    download_kwargs={"raw_download": True},
                    worker_type=transfer_manager.THREAD,
                    max_workers=_DOWNLOAD_WORKERS,
                )
            with contextlib.suppress(FileNotFoundError):
                os.chmod(partial, stat.S_IMODE(dest.stat().st_mode))
            os.replace(partial, dest)
        finally:
            partial.unlink(missing_ok=True)

    def open(self, object_url: str) -> BinaryIO:
        blob = self._existing_blob(object_url)
        if blob is None:
            raise ObjectNotFoundError(f"No object at {object_url}.")
        return blob.open("rb", raw_download=True)

    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        return self.get_object_metadata_at(self.object_url(key))

    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        blob = self._existing_blob(object_url)
        if blob is None:
            return None
        return ObjectMetadata(
            content_type=blob.content_type,
            size=blob.size,
            last_modified=blob.updated,
            content_encoding=blob.content_encoding,
        )

    def list(self, prefix: str) -> list[str]:
        return self._list_keys(self._bucket, prefix)

    def list_at(self, url_prefix: str) -> list[str]:
        bucket, prefix = _split(url_prefix)
        return [_url(bucket, key) for key in self._list_keys(bucket, prefix)]

    def object_url(self, key: str) -> str:
        return _url(self._bucket, key)

    def get_object_key(self, object_url: str) -> str:
        bucket, key = _split(object_url)
        if bucket != self._bucket:
            raise ValueError(f"{object_url!r} is not an object in gs://{self._bucket}.")
        return key

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        blob = self._blob(object_url)
        if not self._can_sign():
            return None
        return self._sign(blob, "GET", min(expires_in, self._max_signed_seconds))

    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        blob = self._blob(object_url)
        if not self._can_sign():
            return None
        return self._sign(blob, "PUT", min(expires_in, self._max_signed_seconds))

    def issue_read_grant(self, object_url: str, *, expires_in: int | None = None) -> HttpGetGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        blob = self._blob(object_url)
        expires_at = self._grant_expiry(expires_in)
        return HttpGetGrant(kind="http-get", url=self._sign(blob, "GET", expires_in), expires_at=expires_at)

    def issue_write_grant(
        self,
        object_url: str,
        *,
        media_type: str,
        max_bytes: int,
        expires_in: int | None = None,
    ) -> HttpPutGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        blob = self._blob(object_url)
        expires_at = self._grant_expiry(expires_in)
        bound = {"x-goog-content-length-range": f"0,{max_bytes}"}
        return HttpPutGrant(
            kind="http-put",
            url=self._sign(blob, "PUT", expires_in, content_type=media_type, headers=dict(bound)),
            expires_at=expires_at,
            headers={"Content-Type": media_type, **bound},
        )

    def issue_upload_policy(
        self, prefix_url: str, *, max_object_bytes: int, expires_in: int
    ) -> UploadPolicy:
        """A V4 POST policy for any key below ``prefix_url``. Its signature lasts all of
        ``expires_in`` up to the signer's cap whatever the caller's own token has left, so it is
        refused only without a signer or past that cap."""
        expires_at = self._grant_expiry(expires_in)
        post = self._post_policy(prefix_url, expires_in, max_object_bytes)
        return UploadPolicy(
            write=HttpPostPolicyGrant(
                kind="http-post-policy",
                url=post["url"],
                fields=post["fields"],
                path_field="key",
                file_field="file",
            ),
            expires_at=expires_at,
        )

    def _post_policy(self, url_prefix: str, expires_in: int, max_bytes: int) -> dict:
        """Built by hand: the client library's helper adds an exact-key condition, which would
        confine the policy to one object. The prefix ends in ``/``, so that ``root`` does not
        admit ``root-evil/``."""
        bucket, prefix = _split(url_prefix.rstrip("/") + "/")
        now = datetime.now(UTC).replace(microsecond=0)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        credential = f"{self._signer.signer_email}/{now:%Y%m%d}/auto/storage/goog4_request"
        conditions: list = [
            ["starts-with", "$key", prefix],
            ["starts-with", "$Content-Type", ""],
            ["content-length-range", 0, max_bytes],
            {"bucket": bucket},
            {"x-goog-date": timestamp},
            {"x-goog-credential": credential},
            {"x-goog-algorithm": "GOOG4-RSA-SHA256"},
        ]
        expiration = (now + timedelta(seconds=expires_in)).strftime("%Y-%m-%dT%H:%M:%SZ")
        document = json.dumps({"conditions": conditions, "expiration": expiration}, separators=(",", ":"))
        policy = base64.b64encode(document.encode())
        return {
            "url": f"{self._client.api_endpoint}/{bucket}/",
            "fields": {
                "key": f"{prefix}${{filename}}",
                "policy": policy.decode(),
                "x-goog-algorithm": "GOOG4-RSA-SHA256",
                "x-goog-credential": credential,
                "x-goog-date": timestamp,
                "x-goog-signature": self._signer.sign_bytes(policy).hex(),
            },
        }

    def _put_file(self, object_url: str, file_path: str, content_type: str) -> str:
        logger.info("Uploading %d bytes from %s to %s", os.path.getsize(file_path), file_path, object_url)
        if os.path.getsize(file_path) > _MULTIPART_LIMIT and self._existing_blob(object_url) is not None:
            raise ObjectAlreadyExistsError(f"Object already exists at {object_url}.")
        blob = self._blob(object_url)
        self._create(object_url, blob, lambda: blob.upload_from_filename(
            file_path, content_type=content_type, if_generation_match=0
        ))
        return object_url

    def _create(self, object_url: str, blob, upload: Callable[[], None]) -> None:
        write_id = uuid.uuid4().hex
        blob.metadata = {_WRITE_ID: write_id}
        try:
            upload()
        except PreconditionFailed as e:
            existing = self._existing_blob(object_url)
            if existing is None or (existing.metadata or {}).get(_WRITE_ID) != write_id:
                raise ObjectAlreadyExistsError(f"Object already exists at {object_url}.") from e

    def _blob(self, object_url: str):
        bucket, key = _split_object(object_url)
        return self._client.bucket(bucket).blob(key)

    def _existing_blob(self, object_url: str):
        bucket, key = _split_object(object_url)
        return self._client.bucket(bucket).get_blob(key)

    def _list_keys(self, bucket: str, prefix: str) -> list[str]:
        blobs = self._client.list_blobs(bucket, prefix=prefix, fields="items(name),nextPageToken")
        return [blob.name for blob in blobs if not blob.name.endswith("/")]

    def _can_sign(self) -> bool:
        if self._signer is None and not self._warned_unsigned:
            logger.warning(
                "GcsObjectStore has no signer, so it issues no signed URLs; set "
                "signing_service_account, or run with credentials that can sign"
            )
            self._warned_unsigned = True
        return self._signer is not None

    def _grant_expiry(self, expires_in: int) -> datetime:
        if self._signer is None:
            raise GrantUnavailableError(
                "GcsObjectStore has no signer; set signing_service_account to issue grants"
            )
        if expires_in > self._max_signed_seconds:
            raise GrantUnavailableError(
                f"GcsObjectStore's signer signs grants for at most {self._max_signed_seconds}s; "
                f"{expires_in}s was requested"
            )
        # The signature's start is stamped to the second, after this.
        return datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=expires_in)

    def _sign(
        self,
        blob,
        method: str,
        expires_in: int,
        *,
        content_type: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expires_in),
            method=method,
            content_type=content_type,
            headers=headers,
            credentials=self._signer,
        )


def _url(bucket: str, key: str) -> str:
    return f"{_SCHEME}{bucket}/{key}"


def _split(object_url: str) -> tuple[str, str]:
    # Not urlparse: it would end the key at a "#" or "?", which object names may contain.
    if not object_url.startswith(_SCHEME):
        raise ValueError(f"{object_url!r} is not a gs:// object url.")
    bucket, _, key = object_url[len(_SCHEME):].partition("/")
    if not bucket:
        raise ValueError(f"{object_url!r} names no bucket.")
    return bucket, key


def _split_object(object_url: str) -> tuple[str, str]:
    """An object's url, which unlike a prefix's names a key: a signed url for an empty one
    would list the bucket."""
    bucket, key = _split(object_url)
    if not key:
        raise ValueError(f"{object_url!r} names no object.")
    return bucket, key


@contextlib.contextmanager
def _missing_as_not_found(object_url: str) -> Iterator[None]:
    try:
        yield
    except NotFound as e:
        raise ObjectNotFoundError(f"No object at {object_url}.") from e
