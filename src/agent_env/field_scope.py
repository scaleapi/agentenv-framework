"""Whose a field's value is, beyond the entity it belongs to."""

from __future__ import annotations

from enum import Enum


class FieldScope(str, Enum):
    """Whose a field's value is. A task-scoped field says what its entity is, and travels with it wherever it's
    written. A field of another scope names something in someone's setup that another user may not have, so writing
    the entity out for elsewhere (``agent-env bundle init``) leaves it out, unless asked to include its scope, and the
    run there takes its own."""

    TASK = "task"  # what the entity is: every field not declared otherwise
    USER = "user"  # the user's setup: their agents, models, providers and endpoints, where outputs go, billing
