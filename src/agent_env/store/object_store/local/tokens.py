"""Grant tokens: signed claims naming one operation on one key or key prefix, until an expiry.

A token is ``v1.<claims>.<signature>``: base64url JSON claims and their HMAC-SHA256 under a key
that lives only in this process's memory, so a token stops working when the process exits and
the server trusts nothing but the claims it signed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from typing import Literal

_VERSION = "v1"
_OPS = ("get", "put", "post")

Op = Literal["get", "put", "post"]


class InvalidGrantError(Exception):
    """A token this process did not sign, or one that has expired."""


@dataclass(frozen=True)
class GrantClaims:
    """What a grant allows. ``key`` names the object of a get or put; ``prefix``, ending in ``/``,
    the keys a post may write. ``max_bytes`` bounds an upload and ``content_type`` is the type a
    put must declare."""

    op: Op
    store: str
    expires: int
    key: str | None = None
    prefix: str | None = None
    max_bytes: int | None = None
    content_type: str | None = None
    grant_id: str = field(default_factory=lambda: secrets.token_hex(8))

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"op must be one of {_OPS}, got {self.op!r}")
        if self.op == "post":
            if self.key is not None or self.prefix is None or not (self.prefix == "" or self.prefix.endswith("/")):
                raise ValueError("a post grant names a prefix ending in '/' (or the empty root prefix), not a key")
        elif self.key is None or self.prefix is not None:
            raise ValueError(f"a {self.op} grant names one key, not a prefix")
        if self.op != "get" and (not isinstance(self.max_bytes, int) or self.max_bytes < 0):
            raise ValueError(f"a {self.op} grant needs a non-negative max_bytes")
        if (self.op == "put") != (self.content_type is not None):
            raise ValueError("a put grant, and only a put grant, names a content type")


class GrantSigner:
    """Signs and verifies grant tokens with a key generated for this process."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def sign(self, claims: GrantClaims) -> str:
        body = {
            "op": claims.op,
            "s": claims.store,
            "exp": claims.expires,
            "id": claims.grant_id,
            "k": claims.key,
            "p": claims.prefix,
            "max": claims.max_bytes,
            "ct": claims.content_type,
        }
        payload = _b64encode(json.dumps({k: v for k, v in body.items() if v is not None}, separators=(",", ":")).encode())
        return f"{_VERSION}.{payload}.{_b64encode(self._mac(payload))}"

    def verify(self, token: str, *, now: float) -> GrantClaims:
        """The claims of ``token`` if this signer signed it and it has not expired at ``now``."""
        version, _, rest = token.partition(".")
        payload, _, signature = rest.partition(".")
        if version != _VERSION or not payload or not signature:
            raise InvalidGrantError("not a grant token")
        try:
            valid = hmac.compare_digest(_b64decode(signature), self._mac(payload))
        except (binascii.Error, ValueError):
            valid = False
        if not valid:
            raise InvalidGrantError("the grant's signature does not match")
        body = json.loads(_b64decode(payload))
        claims = GrantClaims(
            op=body["op"],
            store=body["s"],
            expires=body["exp"],
            grant_id=body["id"],
            key=body.get("k"),
            prefix=body.get("p"),
            max_bytes=body.get("max"),
            content_type=body.get("ct"),
        )
        if claims.expires <= now:
            raise InvalidGrantError("the grant has expired")
        return claims

    def _mac(self, payload: str) -> bytes:
        return hmac.new(self._key, f"{_VERSION}.{payload}".encode(), hashlib.sha256).digest()


def store_id(root: str) -> str:
    """A short, stable name for the store at ``root``, so a token names its store without its path."""
    return hashlib.sha256(root.encode()).hexdigest()[:16]


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
