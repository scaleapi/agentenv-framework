"""The local explorer: a FastAPI control plane over the configured stores and runner.

Serves eight surfaces: Home, Docs, Task Hub, Task Runner, Evals Hub, Environment Hub,
Agents Hub, Universes Hub. Universes are not a resource — they are ``artifacts`` filtered
by type — so they need no routes of their own.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_env.config import get_config, get_runner, runtime
from agent_env.config.errors import ConfigError
from agent_env.explorer.entity_ids import EncodedIdRouting
from agent_env.explorer.openapi_docs import docs_metadata, enrich_openapi_schema
from agent_env.explorer.plugin import load_plugins
from agent_env.explorer.routers import conversations as conversations_router
from agent_env.explorer.routers import objects as objects_router
from agent_env.explorer.routers import runs as runs_router
from agent_env.explorer.routers import triggers as triggers_router
from agent_env.explorer.routers.common import versioned_router
from agent_env.store import NotFoundError
from agent_env.store.routing import configured_store

logger = logging.getLogger(__name__)

API = "/api/v1"

# The routes that take an entity or instance id in the path (see explorer/entity_ids.py).
_ENTITY_ID_PREFIXES = tuple(f"{API}/{c}/" for c in ("artifacts", "envs", "tasks", "agents", "evals", "task-instances"))

# Host headers we always accept; everything else is refused — a DNS-rebinding defense.
# (The packaged explorer binds loopback only — see LOCAL_BIND_HOST — so this is
# defense-in-depth, not the sole barrier.)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})

# FastAPI's own API-docs pages, exempted from the app CSP (see the guard middleware).
# Keep in step with the docs_url / redoc_url / openapi_url passed to FastAPI below.
_DOCS_PATHS = frozenset({"/api/docs", "/api/redoc", "/openapi.json"})

# The packaged explorer is an unauthenticated local control plane, so it always binds
# loopback — a ``0.0.0.0`` footgun is simply unrepresentable. A hosted deployment that
# needs a public bind runs its own ``uvicorn agent_env.explorer.app:create_app --factory
# --host …`` and lists its public hostname(s) under ``[explorer] allowed_hosts``.
LOCAL_BIND_HOST = "127.0.0.1"


def _host_only(raw: str) -> str:
    """The host from a ``Host`` header with the port stripped (IPv6 literals stay bracketed)."""
    raw = raw.strip().lower()
    if raw.startswith("["):
        return raw[: raw.find("]") + 1]
    return raw.split(":", 1)[0]


def packaged_ui_dir() -> Optional[str]:
    """The UI shipped inside the wheel (``agent_env/explorer/static``), or None from a source
    checkout with no built ``static/`` — then the explorer is API-only unless ``--ui`` is given."""
    candidate = Path(__file__).parent / "static"
    return str(candidate) if (candidate / "index.html").is_file() else None


def explorer_settings() -> dict:
    """The ``[explorer]`` table: port, CORS origins, Host allow-list, and static dir.

    ``host`` is intentionally *not* read from config — the packaged explorer always binds
    loopback (see ``LOCAL_BIND_HOST``); a hosted deployment sets its bind via its own
    ``uvicorn --host`` and opts hostnames in through ``allowed_hosts``."""
    section = runtime.get_config().section("explorer")
    return {
        "host": LOCAL_BIND_HOST,  # loopback only; not configurable — see docstring
        "port": int(section.get("port", 8234)),
        "cors_origins": section.get("cors_origins"),
        "allowed_hosts": section.get("allowed_hosts") or [],
        "static_dir": section.get("static_dir") or None,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    logger.info("agent-env explorer starting")
    logger.info("document store: %s", type(configured_store(cfg.get_document_store())).__name__)

    runner = get_runner()
    logger.info("runner: %s (%s)", runner.type, type(runner).__name__)
    await runner.start()          # in-process for LocalRunner; a no-op for external ones
    try:
        yield
    finally:
        await runner.stop()


def create_app(static_dir: Optional[str] = None) -> FastAPI:
    settings = explorer_settings()
    app = FastAPI(
        title="agent-env explorer",
        description="Local control plane for agent-env: browse environments, tasks, "
                    "agents, universes and evals, and run tasks.",
        version="0.1.0",
        lifespan=lifespan,
        # Swagger under /api; the UI owns /docs. openapi_url stays at the root for the UI.
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/openapi.json",
    )
    # Loopback-only by default (a no-auth local explorer); an explicit [explorer] cors_origins overrides.
    cors = settings["cors_origins"]
    if cors and "*" in cors:
        raise ConfigError(
            "[explorer] cors_origins cannot contain '*': the explorer has no auth, so a "
            "wildcard would let any site you visit drive it. List the origins explicitly."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors or [],
        allow_origin_regex=None if cors else r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(EncodedIdRouting, prefixes=_ENTITY_ID_PREFIXES)

    # A no-auth local explorer needs two guards CORS alone doesn't give (CORS withholds
    # the *response*, not the *request*). The bind is loopback-only, so the Host allow-list
    # is just loopback plus any hostnames a hosted deployment explicitly opts into.
    allowed_hosts = set(_LOOPBACK_HOSTS) | {str(h).lower() for h in settings["allowed_hosts"]}
    # Configuring cors_origins used to switch the CSRF guard off for all of /api. On a
    # no-auth plane that traded a named allowlist for none at all: any site could drive
    # a cross-site write, not just the ones opted in. The guard now stays on and admits
    # a cross-site request only when its Origin is one of the configured values.
    allowed_origins = {str(o).rstrip("/").lower() for o in (cors or [])}

    @app.middleware("http")
    async def _guard(request, call_next):
        # Host allow-list: reject a Host that isn't loopback / configured — blocks
        # DNS-rebinding (an attacker domain pointed at 127.0.0.1 arrives with its own
        # Host). The bind is loopback-only, so this backs that up rather than standing alone.
        host = _host_only(request.headers.get("host") or "")
        if host and host not in allowed_hosts:
            return JSONResponse(status_code=421, content={"detail": f"host {host!r} not allowed"})
        # CSRF: a page you visit while `up` runs can't read our loopback-CORS reply,
        # but a cross-site *write* already fires. Browsers tag such requests
        # Sec-Fetch-Site: cross-site (a forbidden header page JS can't set); curl and
        # the agent-env CLI send none, so only a genuine browser cross-site hit to
        # /api is refused.
        if (request.url.path.startswith("/api")
                and request.headers.get("sec-fetch-site") == "cross-site"
                and (request.headers.get("origin") or "").rstrip("/").lower() not in allowed_origins):
            return JSONResponse(status_code=403, content={"detail": "cross-site request refused"})
        response = await call_next(request)
        # The object byte-proxy picks its own per-content-type policy (CSP: sandbox on
        # active types, deliberately none on passive ones), so leave it alone entirely
        # rather than layering a weaker app policy over it.
        if request.url.path.startswith(f"{API}/objects/"):
            return response
        # Swagger and ReDoc load their bundles from a CDN, so `default-src 'self'`
        # blanks both pages. They serve no agent-produced bytes; exempting them beats
        # widening the policy everywhere. Self-hosting the assets would be better.
        if request.url.path in _DOCS_PATHS:
            return response
        # These bound what a compromised page can reach (frame-ancestors, object-src,
        # base-uri, connect-src); they are NOT a backstop for an inline handler, since
        # script-src must keep 'unsafe-inline' for the static export's one bootstrap
        # script — containment for agent-produced bytes is the preview sandbox.
        # 'unsafe-eval' is withheld: the built bundle has no eval/new Function.
        # setdefault, so a route that chose its own policy keeps it.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    @app.exception_handler(NotFoundError)
    async def _not_found(request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    # Plugins mount BEFORE the core routers so a plugin may add a literal path under
    # a core prefix (e.g. /api/v1/agents/trajectory-url) that the core /{entity_id}
    # catch-all would otherwise shadow — Starlette matches in registration order.
    for plugin in load_plugins():
        logger.info("Mounting explorer plugin %r", plugin.type)
        app.include_router(plugin.router)

    # The five (id, version) collections behind Task/Env/Agent/Universe/Evals Hub.
    # Universes Hub reads /artifacts?type=environment_universe — no route of its own.
    app.include_router(versioned_router(prefix=f"{API}/artifacts", tag="artifacts", collection="artifacts", noun="artifact"))
    app.include_router(versioned_router(prefix=f"{API}/envs", tag="envs", collection="envs", noun="environment"))
    app.include_router(versioned_router(prefix=f"{API}/tasks", tag="tasks", collection="tasks", noun="task"))
    app.include_router(versioned_router(prefix=f"{API}/agents", tag="agents", collection="a2a_agents", noun="agent"))
    app.include_router(versioned_router(prefix=f"{API}/evals", tag="evals", collection="evals", noun="eval"))
    app.include_router(runs_router.router)
    app.include_router(triggers_router.router)
    app.include_router(objects_router.router)
    app.include_router(conversations_router.router)

    @app.get(f"{API}/docs/openapi", include_in_schema=False)
    def openapi_document() -> dict:
        """The explorer's OpenAPI schema plus its primitive catalogue, enriched per-request so a
        lazily-registered primitive still appears."""
        return enrich_openapi_schema(dict(app.openapi()))

    @app.get(f"{API}/docs/openapi/metadata", include_in_schema=False)
    def openapi_metadata() -> dict:
        """Provenance for the spec above."""
        return docs_metadata(app.openapi())

    def _health() -> dict:
        cfg = get_config()
        return {
            "status": "ok",
            "document_store": type(configured_store(cfg.get_document_store())).__name__,
            "runner": get_runner().type,
        }

    app.get("/health", summary="Health")(_health)

    static = static_dir or settings["static_dir"] or packaged_ui_dir()
    if not (static and Path(static).is_dir()):
        # Serve health at "/" only when no UI is mounted (else the SPA owns "/").
        app.get("/", include_in_schema=False)(_health)

    if static and Path(static).is_dir():
        _mount_spa(app, Path(static))
        logger.info("Serving UI from %s", static)

    return app


def _mount_spa(app: FastAPI, root: Path) -> None:
    """Serve a client-routed SPA from ``root``: real files served directly, all other paths
    fall back to ``index.html``. Registered after the API routers; refuses ``/api``/``/health``.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    index = root / "index.html"
    if not index.is_file():
        raise RuntimeError(
            f"{root} has no index.html — build the UI with `npm run build:static` "
            "(a plain `next build` leaves the catch-all as '[[...slug]].html')."
        )

    for sub in ("_next", "logos"):
        if (root / sub).is_dir():
            app.mount(f"/{sub}", StaticFiles(directory=root / sub), name=f"ui-{sub}")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/") or path in {"health", "api", "openapi.json"}:
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)


def __getattr__(name: str):
    # Lazy module-level ``app`` so importing this module (e.g. ``up`` importing
    # ``explorer_settings``, or the test suite) doesn't eagerly build the app — a bad
    # [explorer] static_dir or plugin entry would otherwise turn a config error into an
    # import error. Deployments can still point uvicorn at ``agent_env.explorer.app:app``.
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
