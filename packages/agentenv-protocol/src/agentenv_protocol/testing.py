"""Helpers for tests of environments built on this package."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Mapping, Optional

from .tool_context import ToolContext, bound


@contextlib.contextmanager
def tool_context(*, role: Optional[str] = None, session: Optional[str] = None, tool: str = "test",
                 arguments: Optional[Mapping[str, Any]] = None) -> Iterator[ToolContext]:
    """Bind a caller for the block, as the dispatch does for one tool call, so a handler or
    database method called directly sees ``ToolContext.current()`` the way it would under the
    gateway. Nested blocks shadow and restore."""
    with bound(ToolContext.for_test(role=role, session=session, tool=tool, arguments=arguments)) as context:
        yield context
