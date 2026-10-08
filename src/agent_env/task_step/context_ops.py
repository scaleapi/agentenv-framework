"""Build MongoDB update specs from TaskStepContext diffs.

Replaces the read-merge-write CAS pattern with path-level Mongo ops. A single
document's disjoint-path writes are atomic in Mongo, so concurrent workers no
longer need application-layer merging or a retry loop.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

from agent_env.task_step.context import PromptResponse, TaskStepContext, _REDACTED_KEYS, _strip_redacted_keys


@dataclass
class ContextUpdateOps:
    """A backend-agnostic update spec describing a single step's changes to TaskInstance.context.

    Dot-paths are rooted at the TaskInstance document (they include the
    ``context.`` prefix). Compile to a backend-agnostic ``UpdateSpec`` via `to_update_spec`.

    Lists are set-semantic: when a step's "post" snapshot reads the shared in-memory
    context it sees concurrent siblings' appends too, so the diff's tail may include
    their items and the write must dedupe (``$addToSet`` via ``to_update_spec``, or
    the Python ``_union`` the completion CAS in ``task/store.py`` applies). **Constraint:**
    items in any list must be uniquely identifiable by value, or duplicates are silently
    dropped on write; ``deployed_envs`` are told apart by ``instance_id`` (``step_journal._item_key``).
    """

    sets: dict[str, Any] = field(default_factory=dict)
    unsets: set[str] = field(default_factory=set)
    add_to_sets: dict[str, list[Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.sets or self.unsets or self.add_to_sets)

    def to_update_spec(self) -> "UpdateSpec":
        """Convert to a backend-agnostic UpdateSpec (1:1 field mapping)."""
        from agent_env.store.document_store import UpdateSpec

        return UpdateSpec(
            set=dict(self.sets),
            unset=set(self.unsets),
            add_to_set={p: list(v) for p, v in self.add_to_sets.items()},
        )

    def to_journal_dict(self) -> dict:
        """Mongo-safe encoding: dotted paths as values, never as keys."""
        return {
            "sets": [{"path": p, "value": v} for p, v in self.sets.items()],
            "unsets": sorted(self.unsets),
            "add_to_sets": [{"path": p, "items": list(v)} for p, v in self.add_to_sets.items()],
        }

    @classmethod
    def from_journal_dict(cls, d: dict) -> "ContextUpdateOps":
        return cls(
            sets={e["path"]: e["value"] for e in d.get("sets", [])},
            unsets=set(d.get("unsets", [])),
            add_to_sets={e["path"]: list(e["items"]) for e in d.get("add_to_sets", [])},
        )


@dataclass(frozen=True)
class _FieldClassification:
    scalars: tuple[str, ...]
    lists: tuple[str, ...]
    dicts: tuple[str, ...]


def _classify_context_fields() -> _FieldClassification:
    """Partition TaskStepContext fields into scalar/list/dict via type introspection.

    Lists are assumed to be set-semantic (emitted via `$addToSet`). Dicts are
    walked recursively for leaf-level path diffs. If a future field needs
    different semantics, classification must be made explicit here.
    """
    scalars: list[str] = []
    lists: list[str] = []
    dicts: list[str] = []
    hints = get_type_hints(TaskStepContext)
    for f in dataclasses.fields(TaskStepContext):
        kind = _origin_kind(hints[f.name])
        if kind is list:
            lists.append(f.name)
        elif kind is dict:
            dicts.append(f.name)
        else:
            scalars.append(f.name)
    return _FieldClassification(tuple(scalars), tuple(lists), tuple(dicts))


def _origin_kind(tp: Any) -> Any:
    """Resolve ``tp`` to list/dict/other, stripping Optional/Union where needed."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _origin_kind(non_none[0])
        return None
    return origin


_FIELDS = _classify_context_fields()


def build_context_update_ops(
    pre: TaskStepContext | None,
    post: TaskStepContext,
) -> ContextUpdateOps:
    """Diff two context snapshots into a path-level Mongo update spec.

    - Scalar fields: `$set` on change, `$unset` on value→None.
    - List fields (set-semantic): new items (by equality) go to `$addToSet`.
    - Dict fields (e.g. `metadata`): walked recursively so only leaf-level
      differences emit. Concurrent writers to sibling leaves of the same
      parent don't overwrite each other. Lists inside dicts use `$addToSet`
      for prefix-extending writes (so concurrent appenders don't double-add);
      non-prefix mutations (reorder, pop, replacement) emit wholesale `$set`.
    - Redaction (`_REDACTED_KEYS`) applied at every emission point.

    Dict keys containing ``.`` or starting with ``$`` fall back to a
    wholesale ``$set`` of the nearest enclosing subtree (see ``_diff_dict``).
    """
    ops = ContextUpdateOps()
    pre_ctx = pre if pre is not None else TaskStepContext()

    for name in _FIELDS.scalars:
        pre_v = getattr(pre_ctx, name)
        post_v = getattr(post, name)
        if pre_v == post_v:
            continue
        path = f"context.{name}"
        if post_v is None:
            ops.unsets.add(path)
        else:
            ops.sets[path] = post_v

    for name in _FIELDS.lists:
        pre_items = getattr(pre_ctx, name)
        post_items = getattr(post, name)
        candidates = post_items[len(pre_items):] if _is_prefix(pre_items, post_items) else post_items
        new_items = [x for x in candidates if x not in pre_items]
        if new_items:
            ops.add_to_sets[f"context.{name}"] = [_stored(x) for x in new_items]

    for name in _FIELDS.dicts:
        _diff_dict(f"context.{name}", getattr(pre_ctx, name), getattr(post, name), ops)

    return ops


def _stored(item: Any) -> Any:
    """A list item as it is stored: a prompt response with its legacy keys, any other dataclass as ``asdict``."""
    if isinstance(item, PromptResponse):
        return item.to_dict()
    return dataclasses.asdict(item) if dataclasses.is_dataclass(item) else item


def _diff_dict(
    path: str, pre: dict[str, Any], post: dict[str, Any], ops: ContextUpdateOps,
) -> None:
    """Recursively diff `pre` vs `post`, emitting path-level ops:

    - Keys removed from pre → `$unset`.
    - New dict-valued keys → recurse against `{}` so concurrent writers of
      disjoint sub-keys emit distinct leaf paths (not the whole subtree).
    - New list-valued keys → `$addToSet` (creates array if missing; dedupes
      against what concurrent first-creators already wrote).
    - New scalar keys → `$set`.
    - Both dicts → recurse.
    - Both lists, pre is prefix of post → `$addToSet` of the tail.
    - Lists otherwise (reorder, pop, replacement) → wholesale `$set` (last write wins).
    - Scalars differ → `$set`.

    Redacted keys (`_REDACTED_KEYS`) are skipped at every level; redacted
    descendants of emitted subtrees are stripped via `_strip_redacted_keys`.

    If immediate keys on either side are unmappable to Mongo paths
    (contain ``.`` or start with ``$``), emit a wholesale ``$set`` of the
    whole dict at ``path`` instead of recursing.
    """
    if _has_unmappable_keys(pre) or _has_unmappable_keys(post):
        ops.sets[path] = _strip_redacted_keys(post)
        return

    for k in pre.keys() - post.keys():
        if k in _REDACTED_KEYS:
            continue
        ops.unsets.add(f"{path}.{k}")

    for k, post_v in post.items():
        if k in _REDACTED_KEYS:
            continue
        sub_path = f"{path}.{k}"
        if k not in pre:
            if isinstance(post_v, dict):
                _diff_dict(sub_path, {}, post_v, ops)
            elif isinstance(post_v, list):
                if post_v:
                    ops.add_to_sets[sub_path] = [_strip_redacted_keys(x) for x in post_v]
            else:
                ops.sets[sub_path] = _strip_redacted_keys(post_v)
            continue
        pre_v = pre[k]
        if isinstance(pre_v, dict) and isinstance(post_v, dict):
            _diff_dict(sub_path, pre_v, post_v, ops)
        elif isinstance(pre_v, list) and isinstance(post_v, list):
            if _is_prefix(pre_v, post_v):
                # A re-append is a no-op on the write, so the journaled diff must be one too.
                tail = [x for x in post_v[len(pre_v):] if x not in pre_v]
                if tail:
                    ops.add_to_sets[sub_path] = [_strip_redacted_keys(x) for x in tail]
            elif pre_v != post_v:
                ops.sets[sub_path] = _strip_redacted_keys(post_v)
        elif pre_v != post_v:
            ops.sets[sub_path] = _strip_redacted_keys(post_v)


def _is_prefix(pre: list[Any], post: list[Any]) -> bool:
    return len(post) >= len(pre) and post[: len(pre)] == pre


def _has_unmappable_keys(d: dict[str, Any]) -> bool:
    """True if any immediate key in `d` can't be a Mongo dot-notation path component."""
    return any(not isinstance(k, str) or "." in k or k.startswith("$") for k in d)
