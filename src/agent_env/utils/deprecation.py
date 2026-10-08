"""Deprecation shims for renamed names: the service_* -> environment_* rename and the S3-named ones.

Two things make this less trivial than ``warnings.warn``:

1. ``DeprecationWarning`` is invisible in production. CPython's default filters are
   ``default::DeprecationWarning:__main__`` then ``ignore``, and nothing in agent_env is ever
   ``__main__`` under the ``agent-env`` console script. So the countable channel is a structured
   log line, not the warning — the worker's JSON root handler promotes ``extra=`` keys to Datadog
   facets, which is what the removal gate reads. The warning is emitted too, for interactive
   users and pytest's ``-W error``.

2. The alias must preserve OBJECT IDENTITY. A subclass alias would break ``isinstance`` against the
   canonical class at ~20 sites in this repo plus the hub backend's universe_snapshot filters, where
   the failure is an empty list rather than an exception. Hence PEP-562 module ``__getattr__``:
   the only mechanism giving identity, a warning hook, and a counter simultaneously.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEPRECATION_EVENT = "agent_env_deprecated_symbol"

# Fired symbols, so a hot import path costs one log line per process rather than one per call.
# Also what the tests reset between cases; see reset_deprecation_state.
_fired: set[str] = set()
_counts: dict[str, int] = {}


def warn_deprecated(old: str, new: str, *, kind: str = "symbol", stacklevel: int = 3) -> None:
    """Record one use of a deprecated name. At most one warning per symbol per process.

    ``stacklevel`` 3 fits a module ``__getattr__`` shim; a helper called directly by the
    deprecated boundary is one frame deeper and passes 4 so the warning names the caller."""
    _counts[old] = _counts.get(old, 0) + 1
    if old in _fired:
        return
    _fired.add(old)
    message = f"{old} is deprecated and will be removed in a future release; use {new}"
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    logger.warning(
        message,
        extra={"event": _DEPRECATION_EVENT, "deprecated_symbol": old, "replacement": new, "kind": kind},
    )


# The default of a deprecated keyword, so passing it as None still counts as passing it.
OMITTED: Any = object()


def renamed_keyword(owner: str, new: str, new_value: Any, old: str, old_value: Any, *, stacklevel: int = 4) -> Any:
    """The value of keyword ``new``, taking the deprecated keyword ``old`` (default ``OMITTED``) in its place with a
    warning. Passing both is a TypeError, as a repeated keyword is. ``stacklevel`` 4 names the caller of a function
    that calls this directly."""
    if old_value is OMITTED:
        return new_value
    if new_value is not None:
        raise TypeError(f"{owner}() got both {new}= and its deprecated spelling {old}=")
    warn_deprecated(f"{owner}({old}=)", f"{new}=", kind="keyword", stacklevel=stacklevel)
    return old_value


def deprecated_names(mapping: dict[str, Any], *, kind: str = "symbol") -> Callable[[str], Any]:
    """Build a module ``__getattr__`` serving ``{old_name: canonical_object}`` with a warning.

    Returns the canonical object itself, so ``OldName is NewName`` holds and isinstance,
    pydantic discriminated deserialization and ``mock.patch(..., spec=)`` all keep working.
    """
    replacements = {old: getattr(obj, "__name__", str(obj)) for old, obj in mapping.items()}

    def __getattr__(name: str) -> Any:
        if name in mapping:
            warn_deprecated(name, replacements[name], kind=kind)
            return mapping[name]
        raise AttributeError(f"module has no attribute {name!r}")

    return __getattr__


def deprecated_alias(canonical: Callable, old_qualname: str) -> Callable:
    """A delegating alias for a renamed method or function.

    Must produce a REAL attribute rather than a ``__getattr__`` fallback, because
    ``cli/env/multi.py:107,149`` gate on ``hasattr(env, "load_environment_universe_artifact")``
    and print a "not supported" message on a miss — so a half-present alias degrades to a
    silent no-op instead of an AttributeError.
    """
    import functools
    import inspect

    new_name = canonical.__name__
    old_short = old_qualname.rsplit(".", 1)[-1]

    if inspect.iscoroutinefunction(canonical):
        @functools.wraps(canonical)
        async def _alias(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(old_qualname, new_name, kind="method")
            return await canonical(*args, **kwargs)
    else:
        @functools.wraps(canonical)
        def _alias(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(old_qualname, new_name, kind="method")
            return canonical(*args, **kwargs)

    _alias.__name__ = old_short
    _alias.__qualname__ = old_qualname
    _alias.__doc__ = f"Deprecated alias for :meth:`{new_name}`. Scheduled for removal."
    return _alias


def deprecation_counts() -> dict[str, int]:
    """Per-symbol hit counts for this process. The removal gate reads the Datadog facet, not this."""
    return dict(_counts)


def reset_deprecation_state() -> None:
    """Test hook. pytest's monkeypatch materializes a shimmed name into the module ``__dict__``
    on undo, permanently disabling ``__getattr__`` for that name in the worker — so suites that
    patch a deprecated alias must also pop it. See tst/unit/conftest.py."""
    _fired.clear()
    _counts.clear()
