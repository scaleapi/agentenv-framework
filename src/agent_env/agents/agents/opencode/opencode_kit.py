"""The pieces of agent plumbing this agent shares in shape with other CLI agents, copied from the Claude Code agent's
``kit.py``: durable session references, skill installs and FastAPI-style error bodies."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from agentenv_protocol.a2a_agent import BundleSkillRequest, InlineSkillRequest, SkillAddResponse
from agentenv_protocol.transfers import download
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_REFS_ATTR = "_durable_session_refs"


class DurableSessionRefs:
    """Mixin, listed before ``AgentEnvAgent``, that keeps every context's session reference for the life of the
    process: the framework's own record is a bounded LRU, so an idle context would otherwise start a fresh CLI
    session."""

    def session_ref_for_context(self, context_id: str) -> str | None:
        return self.__dict__.get(_REFS_ATTR, {}).get(context_id) or super().session_ref_for_context(context_id)

    def set_session_ref_for_context(self, context_id: str, session_ref: str) -> None:
        super().set_session_ref_for_context(context_id, session_ref)
        self.__dict__.setdefault(_REFS_ATTR, {})[context_id] = session_ref


_NO_BODY_STATUSES = frozenset({204, 205, 304})


async def _detail_body(_request: Request, exc: HTTPException) -> Response:
    if exc.status_code < 200 or exc.status_code in _NO_BODY_STATUSES:
        return Response(status_code=exc.status_code, headers=exc.headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


def json_error_bodies(app: Starlette) -> Starlette:
    """Answer every ``HTTPException`` with a ``{"detail": ...}`` body, as agent-env reads errors."""
    app.add_exception_handler(HTTPException, _detail_body)
    return app


def _validate_skill_name(name: str) -> None:
    if name != Path(name).name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail=f"Invalid skill name '{name}': must not contain path separators or '..'")


async def install_skill(request: InlineSkillRequest | BundleSkillRequest, skills_dir: Path) -> SkillAddResponse:
    """Write ``skills_dir/<name>/``: the inline ``SKILL.md``, or every bundle file through its read grant. A failed
    install leaves no directory behind."""
    _validate_skill_name(request.name)
    skill_dir = skills_dir / request.name
    if skill_dir.exists():
        raise HTTPException(status_code=409, detail=f"Skill '{request.name}' is already registered")
    skill_dir.mkdir(parents=True)
    try:
        if isinstance(request, BundleSkillRequest):
            for item in request.skill_bundle.files:
                await download(item.object, skill_dir / item.path)
        else:
            if not request.skill_md or not request.skill_md.strip():
                raise HTTPException(status_code=400, detail="Skill content must not be empty")
            (skill_dir / "SKILL.md").write_text(request.skill_md)
    except BaseException:
        shutil.rmtree(skill_dir, ignore_errors=True)
        raise
    logger.info(f"Registered skill '{request.name}' at {skill_dir}")
    return SkillAddResponse(name=request.name)
