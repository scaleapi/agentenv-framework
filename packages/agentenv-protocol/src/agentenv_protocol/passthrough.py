"""What a proxy passes on of the headers it relays, shared by the driver's ``/svc/<name>/`` route and a website that
forwards to an app beside it."""
from __future__ import annotations

from collections.abc import Iterable

#: The headers that belong to one connection, which a proxy doesn't pass on.
HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                         "te", "trailer", "transfer-encoding", "upgrade"})


def passed_on(headers: Iterable[tuple[str, str]], *dropped: str) -> list[tuple[str, str]]:
    """``headers`` as a proxy passes them on: without the hop-by-hop ones, those a ``Connection`` header names, and
    ``dropped``, each matched without regard to case."""
    headers = list(headers)
    named = {token.strip().lower() for name, value in headers if name.lower() == "connection"
             for token in value.split(",")}
    skipped = HOP_HEADERS | named | {name.lower() for name in dropped}
    return [(name, value) for name, value in headers if name.lower() not in skipped]
