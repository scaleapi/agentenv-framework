"""Entity ids in explorer URLs: an id is one percent-encoded path segment, so ``/`` in an id
travels as ``%2F``. ASGI servers decode ``%2F`` before routing, which would split a
namespaced id such as ``@local/pkg/name`` across segments; ``EncodedIdRouting`` routes the
collection paths on the raw path instead, and ``EntityId`` decodes the matched segment.
"""

from __future__ import annotations

import re
from typing import Annotated
from urllib.parse import unquote

from fastapi import Path
from pydantic import AfterValidator

_KEPT_ESCAPES = re.compile(r"(%2[fF]|%25)")

EntityId = Annotated[
    str,
    Path(description="The entity id, percent-encoded as one path segment: a '/' in the id is sent as %2F."),
    AfterValidator(unquote),
]


def route_path(raw_path: str) -> str:
    """Decode a raw URL path except ``%2F`` and ``%25``, so an encoded id stays one segment
    and ``unquote`` on the matched segment recovers it exactly."""
    return "".join(
        part if _KEPT_ESCAPES.fullmatch(part) else unquote(part)
        for part in _KEPT_ESCAPES.split(raw_path)
    )


class EncodedIdRouting:
    """ASGI middleware: for requests under ``prefixes``, route on ``route_path`` of the raw
    path. Other paths, including explorer-plugin routes, are left as the server decoded them."""

    def __init__(self, app, prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.prefixes = prefixes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope.get("raw_path"):
            root = scope.get("root_path", "")
            if scope["path"][len(root):].startswith(self.prefixes):
                scope = dict(scope, path=route_path(scope["raw_path"].decode("latin-1")))
        await self.app(scope, receive, send)
