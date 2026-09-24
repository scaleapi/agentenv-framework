"""S3 implementation of the ObjectStore interface."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.object_store.object_store import DEFAULT_CONTENT_TYPE, ObjectMetadata, ObjectStore

logger = logging.getLogger(__name__)


class S3ObjectStore(ObjectStore):
    """ObjectStore backed by an S3 bucket (injected client, config-free)."""

    def __init__(self, client, bucket: str, *, region: str | None = None) -> None:
        self._s3 = client
        self._bucket = bucket
        self._region = region

    @classmethod
    def from_config(cls, *, bucket: str, region: str | None = None) -> S3ObjectStore:
        """Build an adaptive-retry boto3 S3 client for ``bucket``."""
        kwargs = {"config": BotocoreConfig(retries={"max_attempts": 10, "mode": "adaptive"})}
        if region:
            kwargs["region_name"] = region
        return cls(boto3.client("s3", **kwargs), bucket, region=region)

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
        data = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        logger.info(f"Downloaded {len(data)} bytes")
        return data

    def download_to_file(self, object_url: str, dest_path: str) -> None:
        bucket, key = self._split(object_url)
        logger.info(f"Streaming s3://{bucket}/{key} to {dest_path}")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        self._s3.download_file(bucket, key, dest_path)

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
        parsed = urlparse(object_url)
        if parsed.scheme != "s3" or parsed.netloc != self._bucket:
            raise ValueError(f"{object_url!r} is not an object in s3://{self._bucket}.")
        return parsed.path.lstrip("/")

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        return self._signed_url("get_object", object_url, expires_in)

    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        return self._signed_url("put_object", object_url, expires_in)

    def signed_post(
        self, url_prefix: str, *, expires_in: int = 3600, max_bytes: int | None = None
    ) -> dict | None:
        bucket, prefix = self._split(url_prefix)
        conditions: list = [["starts-with", "$key", prefix]]
        if max_bytes is not None:
            conditions.append(["content-length-range", 0, max_bytes])
        # POST policies are default-deny on extra form fields: without this the
        # uploader's Content-Type 403s and objects store as binary/octet-stream.
        conditions.append(["starts-with", "$Content-Type", ""])
        return self._s3.generate_presigned_post(
            Bucket=bucket,
            Key=f"{prefix}${{filename}}",  # substituted by the uploader
            Conditions=conditions,
            ExpiresIn=expires_in,
        )

    def _put_file(self, bucket: str, key: str, file_path: str, content_type: str) -> str:
        logger.info(f"Uploading {os.path.getsize(file_path) / 1024**3:.2f} GB from {file_path} to s3://{bucket}/{key}")
        if self._head(bucket, key) is not None:
            raise ObjectAlreadyExistsError(f"Object already exists at s3://{bucket}/{key}.")
        config = TransferConfig(multipart_threshold=25 * 1024**2, multipart_chunksize=25 * 1024**2, max_concurrency=3)
        self._s3.upload_file(file_path, bucket, key, ExtraArgs={"ContentType": content_type}, Config=config)
        return self._url(bucket, key)

    def _head(self, bucket: str, key: str) -> ObjectMetadata | None:
        try:
            head = self._s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return ObjectMetadata(content_type=head.get("ContentType"), size=head.get("ContentLength"), last_modified=head.get("LastModified"))

    def _list_keys(self, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                if not obj["Key"].endswith("/"):
                    keys.append(obj["Key"])
        return keys

    def _signed_url(self, op: str, object_url: str, expires_in: int) -> str:
        bucket, key = self._split(object_url)
        return self._s3.generate_presigned_url(op, Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in)

    @staticmethod
    def _url(bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"

    @staticmethod
    def _split(object_url: str) -> tuple[str, str]:
        parsed = urlparse(object_url)
        return parsed.netloc, parsed.path.lstrip("/")
