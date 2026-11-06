"""The website SDK: a site's server, meeting ``urn:agentenv:website/v1``. Subclass WebsiteEnvironment, declare the
site with ``@website(domains=[...], static="dist/")``, mark its API with ``@api.get/post/put/patch/delete(path)``, and
``serve()`` it; it needs only the plain package, and uvicorn to serve. The class is optional: a server in any language
that meets the contract works, which ``python -m agentenv_protocol.website.conformance <url>`` checks. Whether a site
is ready is its ``WebsiteReadiness`` answer to ``ready``, which ``website_readiness`` reads of one site and
``env_readiness`` of every site of an env. What a task left a site with is its data plane's ``data/get``."""

from __future__ import annotations

from .clock import CLOCK_EXTENSION_URI, UNIVERSE_NOW_HEADER, ClockUnavailable, UniverseClock
from .environment import (
    SESSION_PATH,
    WebsiteDeclaration,
    WebsiteEnvironment,
    host_problem,
    ready_check,
    website,
)
from .errors import ApiError, BadRequest, Conflict, Forbidden, NotFound, Unauthorized
from .readiness import (
    DEFAULT_HEALTH_PATH,
    READY_TIMEOUT_S,
    SITE_READY_TIMEOUT_S,
    WEBSITE_BROWSE_URL_PARAM,
    WEBSITE_EXTENSION_URI,
    WEBSITE_READY_OP,
    WebsiteReadiness,
    env_readiness,
    website_readiness,
)
from .routes import ROLE_KEY_ENV, ROLE_KEY_HEADER, ApiContext, ApiRoute, api, reserved_path
from .session import SESSION_COOKIE, SESSION_SECRET_ENV, SessionSigner

__all__ = [
    "WebsiteEnvironment", "WebsiteDeclaration", "website", "api", "ApiContext", "ApiRoute", "ready_check", "host_problem",
    "ApiError", "BadRequest", "Unauthorized", "Forbidden", "NotFound", "Conflict",
    "UniverseClock", "ClockUnavailable", "SessionSigner", "reserved_path",
    "WEBSITE_EXTENSION_URI", "CLOCK_EXTENSION_URI", "UNIVERSE_NOW_HEADER", "DEFAULT_HEALTH_PATH", "SESSION_PATH",
    "SESSION_COOKIE", "SESSION_SECRET_ENV", "ROLE_KEY_ENV", "ROLE_KEY_HEADER",
    "WEBSITE_READY_OP", "WEBSITE_BROWSE_URL_PARAM", "READY_TIMEOUT_S", "SITE_READY_TIMEOUT_S", "WebsiteReadiness",
    "website_readiness", "env_readiness",
]
