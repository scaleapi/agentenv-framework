"""Pure comparison functions for universe load/export roundtrip validation."""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

_EMPTY = "\x00EMPTY\x00"  # unifies None / "" / {} / []


def _canon_ts(s: str):
    """Return ('DT', utc-iso) or ('D', date-iso) for a timestamp/date string, else None."""
    if len(s) == 10 and s[4] == "-" and s[7] == "-":  # date-only YYYY-MM-DD (fromisoformat would make it midnight)
        try:
            return ("D", _dt.date.fromisoformat(s).isoformat())
        except (ValueError, TypeError):
            return None
    try:
        ss = s.replace("Z", "+00:00")
        if " " in ss and "T" not in ss:
            ss = ss.replace(" ", "T", 1)
        d = _dt.datetime.fromisoformat(ss)
        if d.tzinfo is not None:
            d = d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return ("DT", d.isoformat())
    except (ValueError, TypeError):
        pass
    try:
        return ("D", _dt.date.fromisoformat(s).isoformat())
    except (ValueError, TypeError):
        return None


def _canon(v: Any) -> Any:
    """Canonicalize a value so equivalent representations compare equal:
    numeric strings -> float (rounded), timestamps -> UTC instant, empties -> sentinel."""
    if v is None or v == "" or v == {} or v == []:
        return _EMPTY
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        # Numbers stay numbers — we do NOT coerce them to timestamps. The old epoch heuristic
        # (1.4e9-2.0e9 s / 1.4e12-2.0e12 ms) collided with financial amounts in that range
        # (e.g. a ~$1.5B balance), and stripping sub-second precision made distinct amounts like
        # 1_500_000_001.50 and 1_500_000_001.75 canonicalize equal — a silent false-negative that
        # would mask real data loss. Timestamps are only canonicalized from strings (ISO/date),
        # which is unambiguous; an epoch carried as a bare number on one side and ISO on the other
        # is not a representation we produce, and if it ever occurred we'd rather flag than swallow.
        return round(float(v), 4)
    if isinstance(v, list):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    if isinstance(v, str):
        s = v.strip()
        # numeric string -> route through number canon (so "12.50" == 12.5), never a timestamp
        try:
            return _canon(float(s))
        except (ValueError, TypeError):
            pass
        ts = _canon_ts(s)
        return ts if ts is not None else s
    return v


def _is_ts(c: Any) -> bool:
    return isinstance(c, tuple) and len(c) == 2 and c[0] in ("DT", "D")


def _semeq(ca: Any, cb: Any) -> bool:
    """Recursive semantic equality over already-canonicalized values.
    ``ca`` is the universe (load) side, ``cb`` the export side."""
    if ca == cb:
        return True
    # universe-null vs export-value = server enrichment / stamping (created_time, derived period ids,
    # etc.) — round-trip-safe. Asymmetric: export-null vs universe-value (a real LOSS) stays flagged.
    if ca == _EMPTY:
        return True
    # date vs datetime on the same calendar day (server truncated a datetime to a date)
    if _is_ts(ca) and _is_ts(cb) and ca[1][:10] == cb[1][:10] and (ca[0] == "D" or cb[0] == "D"):
        return True
    # nested dicts: every non-empty universe key must survive in the export and match.
    # Export-only keys are server enrichment (ignored), but DROPPING a non-empty universe key is
    # real data loss and must fail; dropping an already-empty value (_EMPTY) is harmless.
    if isinstance(ca, dict) and isinstance(cb, dict):
        return all(ca[k] == _EMPTY or (k in cb and _semeq(ca[k], cb[k])) for k in ca)
    # lists of dicts: try order-aligned (fast), then order-independent greedy match (the two sides may
    # sort by different keys, e.g. export adds 'id'/'line_number' the universe lacks)
    if isinstance(ca, list) and isinstance(cb, list):
        if len(ca) != len(cb):
            return False
        if all(_semeq(x, y) for x, y in zip(ca, cb)):
            return True
        if ca and isinstance(ca[0], dict):
            pool = list(cb)
            for x in ca:
                for j, y in enumerate(pool):
                    if _semeq(x, y):
                        pool.pop(j)
                        break
                else:
                    return False
            return True
        return False
    return False


def _vals_equal(ov: Any, ev: Any) -> bool:
    """Semantic equality: ignores timezone/number/epoch/empty representation, tolerant of
    date-vs-datetime and of server-added keys inside nested structures."""
    return _semeq(_canon(ov), _canon(ev))


def normalize(obj: Any) -> Any:
    """Recursively sort lists of dicts by 'id' (or first key) for stable comparison."""
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        normalized = [normalize(item) for item in obj]
        if normalized and isinstance(normalized[0], dict):
            sort_key = "id" if "id" in normalized[0] else next(iter(normalized[0]), "")
            try:
                normalized.sort(key=lambda x: str(x.get(sort_key, "")))
            except Exception:
                pass
        return normalized
    return obj


def compare_dicts(a: dict[str, Any], b: dict[str, Any], label_a: str, label_b: str) -> list[dict[str, Any]]:
    """Compare two normalized data dicts, return deduplicated issues."""
    counts: dict[tuple[str, str, str], dict] = {}

    for entity in sorted(set(a.keys()) | set(b.keys())):
        a_list, b_list = a.get(entity, []), b.get(entity, [])

        if not isinstance(a_list, list) or not isinstance(b_list, list):
            if type(a_list) != type(b_list):
                counts[(entity, "", "type_coercion")] = {"from_type": type(a_list).__name__, "to_type": type(b_list).__name__, "count": 1}
            continue

        id_key = _find_id_key(a_list, b_list)

        def _key(item, i):
            base = str(item.get(id_key, i)) if id_key else str(i)
            ent = item.get("entity_id") if isinstance(item, dict) else None
            return f"{ent}|{base}" if ent is not None else base  # entity-scope so same-numbered accounts don't cross-compare

        a_by_id = {_key(item, i): item for i, item in enumerate(a_list)}
        b_by_id = {_key(item, i): item for i, item in enumerate(b_list)}

        if len(a_list) != len(b_list):
            counts[(entity, "", "count_mismatch")] = {"a_count": len(a_list), "b_count": len(b_list), "count": 1}

        for eid in set(a_by_id.keys()) | set(b_by_id.keys()):
            a_item, b_item = a_by_id.get(eid), b_by_id.get(eid)
            if a_item is None or b_item is None or not isinstance(a_item, dict) or not isinstance(b_item, dict):
                continue

            for field in set(a_item.keys()) | set(b_item.keys()):
                if field in a_item and field not in b_item:
                    key = (entity, field, "dropped_field")
                    counts.setdefault(key, {"count": 0, "non_empty_count": 0, "value_keys": set()})
                    counts[key]["count"] += 1
                    val = a_item[field]
                    if val not in (None, "", {}, []):
                        counts[key]["non_empty_count"] += 1
                    if isinstance(val, dict):
                        counts[key]["value_keys"].update(val.keys())  # for reshape detection
                elif field not in a_item and field in b_item:
                    key = (entity, field, "added_field")
                    counts.setdefault(key, {"count": 0})
                    counts[key]["count"] += 1
                elif field in a_item and field in b_item:
                    ov, ev = a_item[field], b_item[field]
                    if _vals_equal(ov, ev):
                        continue  # semantically equal (timezone/number/empty/date-vs-datetime representation)
                    if type(ov).__name__ != type(ev).__name__ and not (ov is None or ev is None):
                        key = (entity, field, "type_coercion")
                        if key not in counts:
                            counts[key] = {"from_type": type(ov).__name__, "to_type": type(ev).__name__, "count": 0}
                        counts[key]["count"] += 1
                    elif json.dumps(ov, sort_keys=True, default=str) != json.dumps(ev, sort_keys=True, default=str):
                        key = (entity, field, "value_mismatch")
                        if key not in counts:
                            counts[key] = {"sample_a": str(ov)[:100], "sample_b": str(ev)[:100], "count": 0}
                        counts[key]["count"] += 1

    # reshape detection: a dropped dict-field whose keys reappear as this table's added columns
    # is the server expanding a blob into columns (e.g. archived_reconciliations.snapshot) — benign.
    added_by_entity: dict[str, set] = {}
    for (entity, field, issue_type) in counts:
        if issue_type == "added_field":
            added_by_entity.setdefault(entity, set()).add(field)
    for (entity, field, issue_type), info in counts.items():
        if issue_type == "dropped_field":
            vk = info.get("value_keys") or set()
            added = added_by_entity.get(entity, set())
            if vk and len(vk & added) >= 0.6 * len(vk):
                info["benign_reshape"] = True

    issues = []
    for (entity, field, issue_type), info in sorted(counts.items()):
        if issue_type == "dropped_field":
            detail = f"Field present in {label_a} but missing in {label_b} ({info['count']} occurrences, {info['non_empty_count']} non-empty)"
        elif issue_type == "added_field":
            detail = f"Field present in {label_b} but missing in {label_a} ({info['count']} occurrences)"
        elif issue_type == "type_coercion":
            detail = f"{info['from_type']} \u2192 {info['to_type']} ({info['count']} occurrences)"
        elif issue_type == "count_mismatch":
            detail = f"{label_a} has {info['a_count']} entities, {label_b} has {info['b_count']}"
        elif issue_type == "value_mismatch":
            detail = f"Values differ ({info['count']} occurrences), sample: {info['sample_a']!r} vs {info['sample_b']!r}"
        else:
            detail = f"{info['count']} occurrences"

        issue: dict[str, Any] = {"entity": entity, "field": field, "type": issue_type, "detail": detail}
        if issue_type == "dropped_field":
            issue["non_empty_count"] = info["non_empty_count"]
            if info.get("benign_reshape"):
                issue["benign_reshape"] = True
        issues.append(issue)
    return issues


def classify_issues(load_issues: list[dict[str, Any]], idempotency_issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Classify issues with phase and critical flags. Returns (annotated_issues, is_compatible)."""
    _CRITICAL_TYPES = {"count_mismatch", "value_mismatch"}
    _PHASE = {"added_field": "export"}  # everything else is "load"; idempotency issues get "idempotency"

    def _is_critical(issue: dict) -> bool:
        if issue["type"] in _CRITICAL_TYPES:
            return True
        if issue["type"] == "dropped_field":
            return issue.get("non_empty_count", 0) > 0 and not issue.get("benign_reshape")
        return False

    annotated = [{**i, "phase": _PHASE.get(i["type"], "load"), "critical": _is_critical(i)} for i in load_issues]
    annotated += [{**i, "phase": "idempotency", "critical": True} for i in idempotency_issues]
    is_compatible = not any(i["critical"] for i in annotated)
    return annotated, is_compatible


def _find_id_key(a_list: list, b_list: list) -> str | None:
    """Detect the best ID key for matching entities between two lists."""
    sample = (a_list or b_list or [None])[0]
    if not isinstance(sample, dict):
        return None
    for candidate in ("id", "Id", "ID", "email_id", "ts", "name"):
        if candidate in sample:
            return candidate
    return None
