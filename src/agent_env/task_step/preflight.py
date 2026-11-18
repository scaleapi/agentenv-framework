"""What a step's preflight knows of the task it's in: the env version each deploy_env step before it pins, so a step
reads the env its run will deploy rather than the latest one stored."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Optional

from agent_env.env.env import Env
from agent_env.store.base import NotFoundError

_PINNED: ContextVar[Mapping[str, Optional[int]]] = ContextVar("pinned_env_versions", default={})


@contextlib.contextmanager
def pinned(versions: Mapping[str, Optional[int]]) -> Iterator[None]:
    """``versions``, by env id, the ones a preflight in the block reads its envs at."""
    token = _PINNED.set(dict(versions))
    try:
        yield
    finally:
        _PINNED.reset(token)


def stored_env(env_id: str) -> Optional[Env]:
    """The env ``env_id`` at the version a deploy_env step before this one pins, else its latest; None when the store
    has no such env, which the run reports."""
    try:
        return Env.get(env_id, _PINNED.get().get(env_id))
    except NotFoundError:
        return None
