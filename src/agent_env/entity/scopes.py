"""Whose a field's value is, beyond the entity it belongs to, and leaving out the fields of other scopes."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from agent_env.entity.paths import field_sites


class FieldScope(str, Enum):
    """Whose a field's value is. A task-scoped field says what its entity is, and travels with it wherever it's
    written. A field of another scope names something in someone's setup that another user may not have, so writing
    the entity out for elsewhere (``agent-env bundle init``) leaves it out, unless asked to include its scope, and the
    run there takes its own."""

    TASK = "task"  # what the entity is: every field not declared otherwise
    USER = "user"  # the user's setup: their agents, models, providers and endpoints, where outputs go, billing


def prune(data: dict, scopes: Mapping[str, FieldScope], include: frozenset[FieldScope]) -> list[tuple[str, str]]:
    """Leave out of ``data`` each set field ``scopes`` gives a scope ``include`` doesn't name, in place; each one's
    path and why."""
    left = []
    for path, scope in scopes.items():
        if scope in include:
            continue
        for where, holder, key in list(field_sites(data, path)):
            if holder[key] in ("", [], {}):
                continue  # set to nothing, so nothing is left out
            del holder[key]
            left.append((where, f"dropped: {scope.value} scope, so a run takes its own; --include-scope "
                                f"{scope.value} keeps it"))
    return left
