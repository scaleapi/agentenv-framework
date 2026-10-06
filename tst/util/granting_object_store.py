"""A local object store that issues transfer grants, for tests of the object forms.

Each grant is an https URL naming the object's key, so a test reads from a request which
object a descriptor points at; ``granted`` lists the URLs granted, in order.

Its options model the stores a test needs: ``reaches`` sets whether its grants reach every sandbox or
none (default: only local ones, as the local store's), ``signs`` makes it sign URLs (recording each
lifetime asked for in ``signed``), ``grant_headers`` puts headers on its read grants, and
``max_grant_seconds`` refuses a read grant meant to last longer.
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

from agent_env.store.base import GrantUnavailableError
from agent_env.store.object_store import (
    DEFAULT_CONTENT_TYPE,
    LocalFilesystemObjectStore,
    ObjectMetadata,
    UploadPolicy,
)

GRANT_ORIGIN = "https://objects.example.test"
SIGNED_ORIGIN = "https://signed.example.test"


class GrantingObjectStore(LocalFilesystemObjectStore):
    """Also keeps each object's content type, which the local store drops."""

    supports_transfer_grants = True

    def __init__(
        self,
        root: str,
        *,
        reaches: bool | None = None,
        signs: bool = False,
        grant_headers: dict[str, str] | None = None,
        max_grant_seconds: int | None = None,
    ) -> None:
        super().__init__(root)
        self.granted: list[str] = []
        self.signed: list[int] = []
        self._content_types: dict[str, str] = {}
        self._reaches = reaches
        self._signs = signs
        self._grant_headers = grant_headers
        self._max_grant_seconds = max_grant_seconds

    def grants_reach(self, sandbox_type: str | None) -> bool:
        return super().grants_reach(sandbox_type) if self._reaches is None else self._reaches

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        if not self._signs:
            return super().signed_get_url(object_url, expires_in)
        self.signed.append(expires_in)
        return f"{SIGNED_ORIGIN}/{quote(self.get_object_key(object_url))}"

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
        if self._max_grant_seconds is not None and expires_in > self._max_grant_seconds:
            raise GrantUnavailableError(f"this store's grants last at most {self._max_grant_seconds}s")
        return HttpGetGrant(
            kind="http-get",
            url=self._grant_url(object_url, "read"),
            expires_at=_expiry(expires_in),
            headers=self._grant_headers,
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
