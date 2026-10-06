"""A local object store that issues transfer grants, for tests of the object forms.

Each grant is an https URL naming the object's key, so a test reads from a request which
object a descriptor points at; ``granted`` lists the URLs granted, in order.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from agentenv_protocol.transfers import (
    HttpGetGrant,
    HttpPostPolicyGrant,
    HttpPutGrant,
)

from agent_env.store.object_store import (
    DEFAULT_CONTENT_TYPE,
    LocalFilesystemObjectStore,
    ObjectMetadata,
    UploadPolicy,
)

GRANT_ORIGIN = "https://objects.example.test"


class GrantingObjectStore(LocalFilesystemObjectStore):
    """Also keeps each object's content type, which the local store drops."""

    supports_transfer_grants = True

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.granted: list[str] = []
        self._content_types: dict[str, str] = {}

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        allow_overwrite: bool = False,
    ) -> str:
        url = super().put(key, data, content_type, allow_overwrite)
        self._content_types[self.get_object_key(url)] = content_type
        return url

    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        metadata = super().get_object_metadata(key)
        if metadata is None:
            return None
        return replace(metadata, content_type=self._content_types.get(key))

    def issue_read_grant(self, object_url: str, *, expires_in: int | None = None) -> HttpGetGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        return HttpGetGrant(
            kind="http-get",
            url=self._grant_url(object_url, "read"),
            expires_at=_expiry(expires_in),
        )

    def issue_write_grant(
        self, object_url: str, *, media_type: str, max_bytes: int, expires_in: int | None = None
    ) -> HttpPutGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        return HttpPutGrant(
            kind="http-put",
            url=self._grant_url(object_url, "write"),
            expires_at=_expiry(expires_in),
            headers={"Content-Type": media_type},
        )

    def issue_upload_policy(
        self, prefix_url: str, *, max_object_bytes: int, expires_in: int
    ) -> UploadPolicy:
        return UploadPolicy(
            write=HttpPostPolicyGrant(
                kind="http-post-policy",
                url=self._grant_url(prefix_url, "post"),
                fields={"key": f"{self.get_object_key(prefix_url).rstrip('/')}/${{filename}}"},
                path_field="key",
                file_field="file",
            ),
            expires_at=_expiry(expires_in),
        )

    def _grant_url(self, object_url: str, action: str) -> str:
        self.granted.append(object_url)
        return f"{GRANT_ORIGIN}/{quote(self.get_object_key(object_url))}?sig={action}"


def _expiry(expires_in: int) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=expires_in)
