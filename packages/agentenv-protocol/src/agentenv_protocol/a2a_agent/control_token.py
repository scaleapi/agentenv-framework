"""The agent's control token: agent-env hands each deployed agent one (``CONTROL_TOKEN_ENV``), and the agent's
extension routes refuse a call that doesn't carry it in ``CONTROL_TOKEN_HEADER``. An agent with no token configured
serves them to anyone, as before."""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Mapping

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..types import CONTROL_TOKEN_ENV, CONTROL_TOKEN_HEADER
from .peer_agents import PEER_MCP_PATH
from .staging import STAGING_ENDPOINT

logger = logging.getLogger(__name__)

REFUSAL = f"this route takes the agent's control token in the {CONTROL_TOKEN_HEADER} header"


def take_control_token() -> str | None:
    """The token agent-env handed the agent, taken out of the environment so the tools and CLIs the agent runs never
    inherit it."""
    return os.environ.pop(CONTROL_TOKEN_ENV, None) or None


def carries_control_token(headers: Mapping[str, str], token: str | None) -> bool:
    """Whether a call's ``headers`` carry ``token``; with no token, any do."""
    if token is None:
        return True
    return hmac.compare_digest(headers.get(CONTROL_TOKEN_HEADER, "").encode(), token.encode())


def log_refusal(method: str, path: str, client: str | None) -> None:
    logger.warning("refused %s %s from %s: no valid %s", method, path, client or "unknown", CONTROL_TOKEN_HEADER)


class ControlTokenMiddleware:
    """The SDK's check for an agent that serves its ``/ext/*`` routes by hand: each one but staging and the peer-agents
    MCP endpoint refuses a call without the agent's token with a 401. The token is taken out of the environment when
    the application builds its middleware, as it starts serving, unless one is given."""

    def __init__(self, app: ASGIApp, *, token: str | None = None) -> None:
        self.app = app
        self.token = token or take_control_token()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _guarded(_route_path(scope)):
            if not carries_control_token(Headers(scope=scope), self.token):
                client = scope.get("client")
                log_refusal(scope.get("method", ""), scope["path"], client[0] if client else None)
                await JSONResponse({"detail": REFUSAL}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _route_path(scope: Scope) -> str:
    """The path the app's routes match, without the prefix it is mounted under, as Starlette's router reads it."""
    path, root = scope["path"], scope.get("root_path", "")
    return path[len(root):] if root and path.startswith(root) else path


def _guarded(path: str) -> bool:
    return path.startswith("/ext/") and not any(
        path == open_path or path.startswith(open_path + "/") for open_path in (STAGING_ENDPOINT, PEER_MCP_PATH)
    )
