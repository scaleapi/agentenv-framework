"""S3 implementation of the ObjectStore interface."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import BinaryIO, NamedTuple

import boto3
from agentenv_protocol.transfers import (
    HttpGetGrant,
    HttpPostPolicyGrant,
    HttpPutGrant,
)
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

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

_MAX_SIGV4_EXPIRY_SECONDS = 7 * 24 * 60 * 60
# GET answers NoSuchKey; HEAD (and download_file, which starts with one) answers a bare 404.
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class _SigningCredentials(NamedTuple):
    expiry: datetime | None
    temporary: bool


# Steps call the store from the default executor's threads, of which there are up to 32
# (min(32, cpus + 4)); botocore's default pool of 10 would drop and reopen connections past that.
_MAX_POOL_CONNECTIONS = 32


class S3ObjectStore(ObjectStore):
    """ObjectStore backed by an S3 bucket (injected client, config-free)."""

    supports_transfer_grants = True
    max_single_upload_bytes = 5 * 1024 * 1024 * 1024  # one S3 PUT

    def __init__(
        self,
        client,
        bucket: str,
        *,
        region: str | None = None,
        share_credentials: bool = False,
        grant_lifetime_seconds: int = DEFAULT_GRANT_LIFETIME_SECONDS,
    ) -> None:
        if not isinstance(share_credentials, bool):
            raise ValueError(f"share_credentials must be true or false, got {share_credentials!r}")
        self.grant_lifetime_seconds = grant_lifetime(grant_lifetime_seconds, most=_MAX_SIGV4_EXPIRY_SECONDS)
        self._s3 = client
        self._bucket = bucket
        self._region = region
        self._share_credentials = share_credentials
        endpoint = getattr(getattr(client, "meta", None), "endpoint_url", None)
        # Grants are HTTPS URLs, so a plain-HTTP endpoint (a local S3 emulator) issues none.
        if isinstance(endpoint, str) and not endpoint.startswith("https://"):
            self.supports_transfer_grants = False

    @classmethod
    def from_config(
        cls,
        *,
        bucket: str,
        region: str | None = None,
        share_credentials: bool = False,
        grant_lifetime_seconds: int = DEFAULT_GRANT_LIFETIME_SECONDS,
    ) -> S3ObjectStore:
        """Build an adaptive-retry boto3 S3 client for ``bucket``."""
        kwargs = {
            "config": BotocoreConfig(
                signature_version="s3v4",
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=_MAX_POOL_CONNECTIONS,
            )
        }
        if region:
            kwargs["region_name"] = region
        return cls(
            boto3.client("s3", **kwargs), bucket, region=region, share_credentials=share_credentials,
            grant_lifetime_seconds=grant_lifetime_seconds,
        )

    @property
    def bucket(self) -> str:
        """The bucket this store reads and writes, for callers that mint raw s3:// URLs."""
        return self._bucket

    @property
    def region(self) -> str | None:
        """The configured region, else the client's effective one (AWS default chain)."""
        return self._region or self._s3.meta.region_name

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        allow_overwrite: bool = False,
    ) -> str:
        logger.info(f"Uploading {len(data)} bytes to s3://{self._bucket}/{key}")
        kwargs = dict(Body=data, Bucket=self._bucket, Key=key, ContentType=content_type)
        if not allow_overwrite:
            kwargs["IfNoneMatch"] = "*"
        try:
            self._s3.put_object(**kwargs)
        except ClientError as e:
            if e.response["Error"]["Code"] == "PreconditionFailed":
                raise ObjectAlreadyExistsError(f"Object already exists at s3://{self._bucket}/{key}.") from e
            raise
        return self.object_url(key)

    def put_file(self, key: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self._put_file(self._bucket, key, file_path, content_type)

    def put_file_at(self, object_url: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self._put_file(*self._split(object_url), file_path, content_type)

    def get(self, object_url: str) -> bytes:
        bucket, key = self._split(object_url)
        logger.info(f"Downloading from s3://{bucket}/{key}")
        with _missing_as_not_found(bucket, key):
            data = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        logger.info(f"Downloaded {len(data)} bytes")
        return data

    def download_to_file(self, object_url: str, dest_path: str) -> None:
        bucket, key = self._split(object_url)
        logger.info(f"Streaming s3://{bucket}/{key} to {dest_path}")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with _missing_as_not_found(bucket, key):
            self._s3.download_file(bucket, key, dest_path)

    def open(self, object_url: str) -> BinaryIO:
        bucket, key = self._split(object_url)
        with _missing_as_not_found(bucket, key):
            return self._s3.get_object(Bucket=bucket, Key=key)["Body"]

    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        return self._head(self._bucket, key)

    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        return self._head(*self._split(object_url))

    def list(self, prefix: str) -> list[str]:
        return self._list_keys(self._bucket, prefix)

    def list_at(self, url_prefix: str) -> list[str]:
        bucket, prefix = self._split(url_prefix)
        return [self._url(bucket, key) for key in self._list_keys(bucket, prefix)]

    def object_url(self, key: str) -> str:
        return self._url(self._bucket, key)

    def get_object_key(self, object_url: str) -> str:
        bucket, key = self._split(object_url)
        if not object_url.startswith("s3://") or bucket != self._bucket:
            raise ValueError(f"{object_url!r} is not an object in s3://{self._bucket}.")
        return key

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        return self._signed_url("get_object", object_url, expires_in)

    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        return self._signed_url("put_object", object_url, expires_in)

    def issue_read_grant(
        self, object_url: str, *, expires_in: int | None = None
    ) -> HttpGetGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        return HttpGetGrant(
            kind="http-get",
            url=self._signed_url("get_object", object_url, expires_in),
            expires_at=self._grant_expiry(expires_in),
        )

    def issue_write_grant(
        self,
        object_url: str,
        *,
        media_type: str,
        max_bytes: int,
        expires_in: int | None = None,
    ) -> HttpPutGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        # A presigned S3 PUT cannot bound the upload size, so the uploader enforces max_bytes.
        return HttpPutGrant(
            kind="http-put",
            url=self._signed_url(
                "put_object", object_url, expires_in, params={"ContentType": media_type}
            ),
            expires_at=self._grant_expiry(expires_in),
            headers={"Content-Type": media_type},
        )

    def issue_upload_policy(
        self, prefix_url: str, *, max_object_bytes: int, expires_in: int
    ) -> UploadPolicy:
        self._check_sigv4_expiry(expires_in)
        self._require_long_term_credentials(expires_in)
        post = self._post_policy(prefix_url, expires_in, max_object_bytes)
        return UploadPolicy(
            write=HttpPostPolicyGrant(
                kind="http-post-policy",
                url=post["url"],
                fields=post["fields"],
                path_field="key",
                file_field="file",
            ),
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        )

    def _post_policy(self, url_prefix: str, expires_in: int, max_bytes: int) -> dict:
        """A presigned POST for any key below ``url_prefix``: the key is a condition the uploader
        fills in, not part of what is signed. The prefix ends in ``/``, so that ``root`` does not
        admit ``root-evil/``."""
        bucket, prefix = self._split(url_prefix.rstrip("/") + "/")
        conditions: list = [["starts-with", "$key", prefix], ["content-length-range", 0, max_bytes]]
        # POST policies are default-deny on extra form fields: without this the
        # uploader's Content-Type 403s and objects store as binary/octet-stream.
        conditions.append(["starts-with", "$Content-Type", ""])
        return self._s3.generate_presigned_post(
            Bucket=bucket,
            Key=f"{prefix}${{filename}}",  # substituted by the uploader
            Conditions=conditions,
            ExpiresIn=expires_in,
        )

    def shared_credentials_env(self) -> dict[str, str]:
        """The AWS default-chain credentials, frozen now, when ``share_credentials`` is set."""
        creds = boto3.Session().get_credentials() if self._share_credentials else None
        if creds is None:
            return {}
        frozen = creds.get_frozen_credentials()
        env = {"AWS_ACCESS_KEY_ID": frozen.access_key, "AWS_SECRET_ACCESS_KEY": frozen.secret_key}
        if frozen.token:
            env["AWS_SESSION_TOKEN"] = frozen.token
        if self.region:
            env["AWS_DEFAULT_REGION"] = self.region
        return env

    def _put_file(self, bucket: str, key: str, file_path: str, content_type: str) -> str:
        logger.info(f"Uploading {os.path.getsize(file_path) / 1024**3:.2f} GB from {file_path} to s3://{bucket}/{key}")
        if self._head(bucket, key) is not None:
            raise ObjectAlreadyExistsError(f"Object already exists at s3://{bucket}/{key}.")
        config = TransferConfig(multipart_threshold=25 * 1024**2, multipart_chunksize=25 * 1024**2, max_concurrency=3)
        self._s3.upload_file(file_path, bucket, key, ExtraArgs={"ContentType": content_type}, Config=config)
        return self._url(bucket, key)

    def _head(self, bucket: str, key: str) -> ObjectMetadata | None:
        if not key:  # a bucket, which S3 would refuse as a malformed request rather than report missing
            return None
        try:
            head = self._s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in _MISSING_CODES:
                return None
            raise
        return ObjectMetadata(
            content_type=head.get("ContentType"),
            size=head.get("ContentLength"),
            last_modified=head.get("LastModified"),
            content_encoding=head.get("ContentEncoding"),
        )

    def _list_keys(self, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                if not obj["Key"].endswith("/"):
                    keys.append(obj["Key"])
        return keys

    def _signed_url(
        self, op: str, object_url: str, expires_in: int, *, params: dict | None = None
    ) -> str:
        bucket, key = self._split(object_url)
        return self._s3.generate_presigned_url(
            op, Params={"Bucket": bucket, "Key": key, **(params or {})}, ExpiresIn=expires_in
        )

    def _grant_expiry(self, expires_in: int) -> datetime:
        """When an exact-object grant just signed stops working: after ``expires_in``, or when
        the temporary credentials that signed it expire, if sooner. Read after signing, which
        may have refreshed them."""
        self._check_sigv4_expiry(expires_in)
        requested = datetime.now(UTC) + timedelta(seconds=expires_in)
        signing = self._signing_credentials()
        if signing is None or signing.expiry is None or signing.expiry >= requested:
            return requested
        return signing.expiry

    def _require_long_term_credentials(self, expires_in: int) -> None:
        """Refuse a grant that must last all of ``expires_in`` unless long-term credentials
        sign it: temporary ones would end it at a moment set by their age, so the same request
        would pass or fail by chance."""
        signing = self._signing_credentials()
        if signing is None:
            raise GrantUnavailableError(
                "the S3 client's signing credentials cannot be inspected; a namespace grant "
                "needs credentials known to be long-term"
            )
        if signing.temporary:
            lifetime = (
                f"expire at {signing.expiry.isoformat(timespec='seconds')}"
                if signing.expiry is not None
                else "carry a session token"
            )
            raise GrantUnavailableError(
                f"the S3 signing credentials are temporary (they {lifetime}); a {expires_in}s "
                "namespace grant needs long-term credentials, which do not expire during a capture"
            )

    def _signing_credentials(self) -> _SigningCredentials | None:
        """The credentials that sign this client's requests. botocore has no public accessor,
        so this reads its request signer's private attributes; None when the client does not
        expose them."""
        signer = getattr(self._s3, "_request_signer", None)
        credentials = getattr(signer, "_credentials", None)
        if credentials is None:
            return None
        expiry = getattr(credentials, "_expiry_time", None)
        if expiry is not None:
            expiry = (expiry if expiry.tzinfo else expiry.replace(tzinfo=UTC)).astimezone(UTC)
        return _SigningCredentials(
            expiry=expiry,
            temporary=expiry is not None or bool(getattr(credentials, "token", None)),
        )

    @staticmethod
    def _check_sigv4_expiry(expires_in: int) -> None:
        if expires_in > _MAX_SIGV4_EXPIRY_SECONDS:
            raise GrantUnavailableError(
                f"S3 SigV4 grants last at most {_MAX_SIGV4_EXPIRY_SECONDS}s; "
                f"{expires_in}s was requested"
            )

    @staticmethod
    def _url(bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"

    @staticmethod
    def _split(object_url: str) -> tuple[str, str]:
        # Not urlparse: it would end the key at a "#" or "?", which S3 keys may contain.
        bucket, _, key = object_url.removeprefix("s3://").partition("/")
        return bucket, key.lstrip("/")


@contextlib.contextmanager
def _missing_as_not_found(bucket: str, key: str) -> Iterator[None]:
    try:
        yield
    except ClientError as e:
        if e.response["Error"]["Code"] in _MISSING_CODES:
            raise ObjectNotFoundError(f"No object at s3://{bucket}/{key}.") from e
        raise
