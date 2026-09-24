"""Pre-deploy fit-check for the data-plane intake declaration.

Pure functions over plain dicts — no I/O, no Mongo/S3 — so they run at authoring/
registration time. The agent-env-side ``preflight(universe, env)`` wraps these by
reading ``client.intake_declaration(card)`` and each service's seed ``data.json``.
"""
from __future__ import annotations

from typing import Optional


def _json_add_format(declaration: dict, content_format: str) -> Optional[dict]:
    """Return the declared ``data/add`` format for an inline (part=data) payload, or None."""
    for fmt in declaration.get("add") or []:
        if fmt.get("part") == "data" and fmt.get("format") == content_format:
            return fmt
    return None


def intake_fit_check(
    declaration: Optional[dict],
    data_json: dict,
    *,
    content_format: str = "json",
) -> list[str]:
    """Check a candidate seed ``data_json`` against a declared intake.

    ``declaration`` is the dict from ``client.intake_declaration(card)`` (an
    ``IntakeDeclaration``), or ``None``. ``data_json`` is the collection→rows seed the
    server will load via ``load_from_json``.

    Returns a list of human-readable issues; empty means it fits (or there is no claim
    to check against). Absence of a declaration reads as "no claim" and returns ``[]``,
    matching ``intake_declaration``'s absence semantics.

    The check models the *loadable subset* and tolerates extra fields (real
    ``data.json`` is a superset on validated-tier servers): it flags collections the
    server does not accept and rows missing a declared required field, but never
    complains about extra fields.
    """
    if not declaration:
        return []

    fmt = _json_add_format(declaration, content_format)
    if fmt is None:
        accepted = [
            f"{f.get('part')}/{f.get('format')}" for f in declaration.get("add") or []
        ]
        return [
            f"data/add does not accept part=data format={content_format} "
            f"(accepted: {accepted or 'nothing'})"
        ]

    tables = fmt.get("tables")
    if not tables:
        # Accepted, but no per-table schema was declared — nothing more to verify.
        return []

    issues: list[str] = []
    for collection, rows in data_json.items():
        if not isinstance(rows, list):
            # Scalar top-level extras (e.g. contacts.current_user_id, email.user_email)
            # ride alongside the collections on export; they are not tables to validate.
            continue
        if collection not in tables:
            issues.append(
                f"collection '{collection}' is not accepted "
                f"(declared: {sorted(tables)})"
            )
            continue
        schema = tables[collection] or {}
        required = schema.get("required") or []
        if not required:
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            missing = [r for r in required if r not in row]
            if missing:
                issues.append(f"{collection}[{i}] missing required field(s): {missing}")
    return issues
