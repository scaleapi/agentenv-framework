"""The shapes of a website browser's state and of agentenv's record of it: cookies and local storage as Playwright's
``storageState`` has them, so one exported from Playwright loads as is, and agentenv's own records in its names."""
from __future__ import annotations

from typing import Literal, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SameSite = Literal["Strict", "Lax", "None"]


class PartitionKey(BaseModel):
    """CDP's ``Network.CookiePartitionKey``: the top-level site a partitioned cookie was set under."""
    topLevelSite: str
    hasCrossSiteAncestor: bool


class Cookie(BaseModel):
    """A cookie as the browser has it, CDP's ``Network.Cookie`` fields that ``storageState`` keeps: ``expires`` in
    seconds since the epoch, -1 for one that ends with the session, and a ``domain`` with a leading dot for one its
    subdomains get too; and a partitioned cookie's ``partitionKey``, without which it would be another cookie."""
    name: str
    value: str
    domain: str
    path: str
    expires: float
    httpOnly: bool
    secure: bool
    sameSite: SameSite = "Lax"
    partitionKey: Optional[PartitionKey] = None


class CookieToSet(BaseModel):
    """A cookie to set, at ``url``, or at ``domain`` and ``path``, ``/`` by default. ``max_age``, in place of
    ``expires``, is seconds from when it's set on the browser's clock; with neither, or ``expires`` -1, it ends with
    the session."""
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str
    url: Optional[str] = None
    domain: Optional[str] = None
    path: Optional[str] = None
    expires: Optional[float] = None
    max_age: Optional[float] = None
    httpOnly: bool = False
    secure: bool = False
    sameSite: Optional[SameSite] = None
    partitionKey: Optional[PartitionKey] = None

    @model_validator(mode="after")
    def _placed_and_timed_once(self) -> CookieToSet:
        if not self.url and not self.domain:
            raise ValueError(f"cookie {self.name!r} names neither a url nor a domain")
        if self.expires is not None and self.max_age is not None:
            raise ValueError(f"cookie {self.name!r} has both expires and max_age")
        return self


class StorageItem(BaseModel):
    name: str
    value: str


class OriginStorage(BaseModel):
    """An origin's local storage as ``storageState`` has it."""
    origin: str
    localStorage: list[StorageItem]

    @field_validator("origin")
    @classmethod
    def _an_origin(cls, origin: str) -> str:
        return origin_of(origin, strict=True)


class OpenTab(BaseModel):
    """An open tab, by the browser's target id, with the tab that opened it, a popup's, and whether it's active."""
    id: str
    url: str
    title: str
    opened_at: str
    opener: Optional[str] = None
    active: bool


class ClosedTab(BaseModel):
    """A closed tab as of its page's last load, and who closed it: agentenv over its own connection, someone it can't
    tell, or the browser's exit or its server's stop."""
    id: str
    url: str
    title: str
    opened_at: str
    opener: Optional[str] = None
    closed_at: str
    closed_by: Literal["agentenv", "unknown", "exit", "stop"]


class Download(BaseModel):
    """A download into ``~/Downloads``; completed, with where it landed, its size and its sha256."""
    guid: str
    url: str
    filename: str
    state: Literal["inProgress", "completed", "canceled"]
    started_at: str
    ended_at: Optional[str] = None
    path: Optional[str] = None
    size: Optional[int] = None
    sha256: Optional[str] = None


class Dialog(BaseModel):
    """A page's JavaScript dialog, its answer and who gave it."""
    tab: Optional[str]
    type: str
    message: str
    default_text: str
    url: str
    opened_at: str
    answer: Optional[Literal["accepted", "dismissed"]] = None
    by: Optional[Literal["agentenv", "agent"]] = None
    answered_at: Optional[str] = None
    input: Optional[str] = None


class LaunchEntry(BaseModel):
    """A launch, stop or exit of the browser in its app server's log, with a launch's cause and an exit's ``code`` or
    the ``signal`` that ended it."""
    at: str
    event: Literal["launch", "stop", "exit"]
    pid: int
    cause: Optional[str] = None
    code: Optional[int] = None
    signal: Optional[str] = None


class BrowserRecord(BaseModel):
    """``browser/v1 record``: what agentenv's connection saw ``since`` its server started, and each launch since the
    computer booted."""
    since: str
    tabs: list[OpenTab]
    closed_tabs: list[ClosedTab]
    downloads: list[Download]
    dialogs: list[Dialog]
    launches: list[LaunchEntry]
    cdp_connections: list[dict]


def origin_of(url: str, *, strict: bool = False) -> Optional[str]:
    """``url``'s origin, ``scheme://host[:port]``, for an http or https URL, else None; ``strict``, ValueError for
    anything that isn't an origin, a path or query beyond ``/`` included."""
    parts = urlsplit(url)
    host = parts.netloc.rsplit("@", 1)[-1].lower()
    if parts.scheme not in ("http", "https") or not host:
        if strict:
            raise ValueError(f"{url!r} isn't an http or https origin")
        return None
    if strict and (parts.path not in ("", "/") or parts.query or parts.fragment or "@" in parts.netloc):
        raise ValueError(f"{url!r} is a URL, not an origin: give only its scheme, host and port")
    default = {"http": ":80", "https": ":443"}[parts.scheme]
    return f"{parts.scheme}://{host.removesuffix(default)}"
