"""Cost-attribution dimensions carried from the caller down to a sandbox provider.

Attribution is an open ``dict[str, str]``. Core threads it without reading it; each provider
picks out the keys its billing backend understands and ignores the rest, so a deployment can
attribute on whatever dimensions and terminology it uses.
"""

from __future__ import annotations

from typing import Any, Mapping

Attribution = dict[str, str]

# The one key in a step's open ``metadata`` map that is forwarded to compute as attribution.
ATTRIBUTION_KEY = "attribution"

# Flat keys older persisted step documents carry; read-only compatibility, never written.
_LEGACY_DOCUMENT_KEYS = ("product", "customer", "team", "project_id")


def attribution_of(step) -> Attribution:
    """A step's attribution: a copy of ``metadata["attribution"]`` (``{}`` when unset)."""
    return dict((getattr(step, "metadata", None) or {}).get(ATTRIBUTION_KEY) or {})


def metadata_from_legacy_document(data: Mapping[str, Any]) -> dict:
    """A persisted step's ``metadata`` with any flat legacy attribution keys folded into
    ``metadata["attribution"]``. Persisted documents are immutable, so this read path is
    permanent. The metadata sub-key wins over a flat key on conflict.
    """
    metadata = dict(data.get("metadata") or {})
    legacy = {name: data[name] for name in _LEGACY_DOCUMENT_KEYS if data.get(name) is not None}
    if legacy:
        metadata[ATTRIBUTION_KEY] = {**legacy, **(metadata.get(ATTRIBUTION_KEY) or {})}
    return metadata
