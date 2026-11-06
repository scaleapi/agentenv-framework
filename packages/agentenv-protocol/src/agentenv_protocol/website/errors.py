"""Errors an ``@api`` method raises to answer with an HTTP status: the body is ``{"error": message, "code": code}``."""
from __future__ import annotations


class ApiError(Exception):
    """An answer other than success, ``status`` with ``{"error": message, "code": code}``; a subclass sets both
    defaults, and either may be given per raise."""

    status: int = 500
    code: str = "error"

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).code
        self.status = status or type(self).status

    def body(self) -> dict:
        return {"error": self.message, "code": self.code}


class BadRequest(ApiError):
    status, code = 400, "bad_request"


class Unauthorized(ApiError):
    status, code = 401, "unauthorized"


class Forbidden(ApiError):
    status, code = 403, "forbidden"


class NotFound(ApiError):
    status, code = 404, "not_found"


class Conflict(ApiError):
    status, code = 409, "conflict"
