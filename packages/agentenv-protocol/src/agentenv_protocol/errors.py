"""Errors a tool or extension raises to answer its caller with a code, not a crash."""
from __future__ import annotations

from typing import Literal

#: Whether the call's effect happened: ``unknown`` when it may have been delivered before something failed.
Applied = Literal["yes", "no", "unknown"]


class AgentEnvError(Exception):
    """Raised in an extension, answers with ``status`` and these fields as the error body; raised in a tool,
    answers ``isError`` with ``"<code>: <message>"`` as text and the fields as structured content. Callers retry
    on their own only when ``retryable`` and ``applied == "no"``."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, applied: Applied = "no",
                 hint: str | None = None, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.applied = applied
        self.hint = hint
        self.status = status

    def fields(self) -> dict:
        fields = {"code": self.code, "message": self.message, "retryable": self.retryable, "applied": self.applied}
        if self.hint is not None:
            fields["hint"] = self.hint
        return fields
