"""Whose a field's value is, beyond the entity it belongs to."""

from __future__ import annotations

from enum import Enum


class FieldScope(str, Enum):
    """Whose a field's value is. A field with no scope is material: it says what its entity is, and travels with it
    wherever it's written. A scoped field says how someone's setup ran it, so writing the entity out for elsewhere
    (``agent-env bundle init``) leaves it out, unless asked to include its scope, and the run there takes its own."""

    USER = "user"  # how the user's setup ran it: the sandbox provider, where outputs go, service endpoints, billing
