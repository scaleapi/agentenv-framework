"""What a folder ``bundle init`` wrote holds of the store it was imported from, for registering it back: each entity,
image, task and eval's store id and version, whether the folder still holds each as the import wrote it, and the
user-scoped fields the import left out of each task's steps."""

from __future__ import annotations

import copy
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from agent_env.entity.paths import field_sites
from agent_env.entity.refs import ref_sites
from agent_env.entity.scopes import FieldScope
from agent_env.env.registry import get_env_registry
from agent_env.store.base import NotFoundError
from agent_env.task.task import Task
from agent_env.task_step.registry import get_task_step_registry

from ._import_report import PROVENANCE, PROVENANCE_SCHEMA, store_fingerprint
from ._secrets import scrub
from .parse import CONFIG_FILES, Bundle, BundleEntry, BundleKind
from .plan import Plan, Write, env_folder_writer
from .resolve import BuiltImage, BundleImage


@dataclass(frozen=True)
class Imported:
    """What an import recorded: each entity folder, task file and eval file's store id and version, each image's by its
    folder and toml key, built or pulled, with the Dockerfile a built one is built from, and each file as it wrote
    it."""

    entities: Mapping[str, tuple[str, int]]
    images: Mapping[tuple[str, str], tuple[str, int]]
    files: Mapping[str, str]
    dockerfiles: Mapping[tuple[str, str], str]


def read_imported(root: Path) -> tuple[Imported | None, str | None]:
    """What the import of ``root`` recorded, when it was imported from the configured store; else None, and why when
    it has a record this can't use."""
    path = root / PROVENANCE
    if not path.is_file():
        return None, None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, f"{PROVENANCE} can't be read, so the folder registers as a new bundle"
    if not isinstance(record, dict) or record.get("schema") != PROVENANCE_SCHEMA:
        return None, f"{PROVENANCE} is from an older agent-env, so the folder registers as a new bundle"
    if record.get("store") != store_fingerprint():
        return None, "it was imported from another store, so its entries take new ids"
    entities = {item["folder"]: (item["id"], item["version"]) for item in record.get("written", [])}
    entities |= {item["file"]: (item["id"], item["version"]) for item in record.get("tasks", [])}
    if (stored_eval := record.get("eval")) is not None:
        entities[stored_eval["file"]] = (stored_eval["id"], stored_eval["version"])
    images = {(item["folder"], item["key"]): (item["id"], item["version"])
              for item in [*record.get("images", []), *record.get("pulled", [])]}
    dockerfiles = {(item["folder"], item["key"]): item.get("dockerfile") for item in record.get("images", [])}
    return Imported(entities, images, record.get("files", {}), dockerfiles), None


def where(root: Path, entry: BundleEntry) -> str:
    """``entry``'s folder or file in the bundle at ``root``, as the import's record names it."""
    return unicodedata.normalize("NFC", entry.path.relative_to(root).as_posix())


def source_of(imported: Imported, plan: Plan, write: Write) -> tuple[str, int] | None:
    """The store id and version ``write`` was imported from, or None when the import didn't write it, as for what an
    installed bundle holds."""
    entry = write.source.entry
    if entry in plan.origins:
        return None
    folder = where(plan.bundle.bundle.root, entry)
    if isinstance(write.source, BundleImage) and write.source.id != entry.id:
        return imported.images.get((folder, image_key(write.source)))
    return imported.entities.get(folder)


def image_key(image: BundleImage) -> str:
    """The toml key an image an env or agent folder builds is named by: ``image`` for its own, as the import names it."""
    role = image.id[len(image.entry.id) + 2:]
    return "image" if role == f"{image.entry.kind.store}_image" else role


def as_imported(imported: Imported, plan: Plan, write: Write, now: Mapping[str, str]) -> bool:
    """Whether the files ``write`` is made from are those the import wrote: a task's or eval's file; an image a toml
    key builds, its folder but the toml and the Dockerfile the key names; an env's or agent's toml, or every file of an
    env type written from its folder; every file of an artifact, skill or image folder."""
    entry = write.source.entry
    folder = where(plan.bundle.bundle.root, entry)
    if write.kind in (BundleKind.TASK, BundleKind.EVAL):
        return folder in imported.files and imported.files[folder] == now.get(folder)
    toml = f"{folder}/{CONFIG_FILES[entry.kind]}" if entry.kind in CONFIG_FILES else None
    then, current = _within(imported.files, folder), _within(now, folder)
    if isinstance(write.source, BuiltImage) and write.source.id != entry.id:
        if imported.dockerfiles.get((folder, image_key(write.source))) != write.source.dockerfile:
            return False
        then.pop(toml, None)
        current.pop(toml, None)
    elif (isinstance(write.source, BundleImage) and write.source.id != entry.id) or (
            entry.kind in (BundleKind.ENV, BundleKind.AGENT) and not (
                entry.kind is BundleKind.ENV and _written_from_folder(entry))):
        then, current = {toml: then.get(toml)}, {toml: current.get(toml)}
    return then == current


def grafted(bundle: Bundle, imported: Imported) -> tuple[Bundle, dict[Path, set[str]], dict[Path, list[str]]]:
    """``bundle`` with each imported task's steps given back, from the stored version each was imported from, matched
    by step id and type, what the import left out where the folder doesn't set it: each user-scoped field, but a
    reference's pin where the folder names something else; each list item's, from the stored item it was written from;
    and each env var and credential the import's scrub took, where the folder holds what the scrub left. And, for each
    task's file, the paths in its steps that took one, and a problem for each field still holding what the scrub left,
    which the folder changed too much to give back."""
    registry = get_task_step_registry()
    entries, taken, problems = [], {}, {}
    for entry in bundle.entries:
        source = imported.entities.get(where(bundle.root, entry)) if entry.kind is BundleKind.TASK else None
        try:
            stored = Task.get(*source).to_dict()["steps"] if source is not None else None
        except NotFoundError:
            stored = None
        if stored is None:
            entries.append(entry)
            continue
        by_id = {step.get("id"): step for step in stored}
        steps = copy.deepcopy(entry.config)
        for index, step in enumerate(steps):
            origin = by_id.get(step.get("id"))
            cls = registry.get(step.get("type"))
            if origin is None or cls is None or origin.get("type") != step.get("type"):
                continue
            given, left = _given_back(step, origin, cls)
            for path in given:
                taken.setdefault(entry.path, set()).add(f"[{index}].{path}")
            problems.setdefault(entry.path, []).extend(
                f"{where(bundle.root, entry)}: step {step.get('id')!r}: {path} still holds what the import masked, and "
                "the folder changed what it was masked from; set its real value" for path in left)
        entries.append(replace(entry, config=steps))
    return replace(bundle, entries=tuple(entries)), taken, problems


def _given_back(step: dict, origin: dict, cls: type) -> tuple[list[str], list[str]]:
    """Give ``step`` back what the import left out of ``origin``, in place; the path of each field it took, and of each
    still holding what the scrub left."""
    user = [path for path, scope in cls.field_scopes.items() if scope is FieldScope.USER]
    repinned = _repinned(step, origin, cls)
    given, lists = [], {}
    for path in user:
        if path in repinned:
            continue
        if "[]" in path:
            at, _, field = path.partition("[].")
            if "[]" not in field:
                lists.setdefault(at, []).append(field)
            continue
        for _, holder, key in field_sites(origin, path):
            if holder[key] not in ("", [], {}) and _graft(step, path.split("."), holder[key]):
                given.append(path)
    for at, fields in lists.items():
        given += _given_back_items(step, origin, at, fields)
    scrubbed = copy.deepcopy(origin)
    scrub(scrubbed)
    given += _unscrubbed(step, origin, scrubbed)
    masked = {text for _, text in _masks(origin, scrubbed)}
    return given, [path for path, text in _texts(step) if text in masked]


def _repinned(step: dict, origin: dict, cls: type) -> set[str]:
    """The path of each pin in ``origin`` beside a reference ``step`` names something else at, such as an entity the
    import wrote into the folder, which can't take the stored entity's pin."""
    refs = cls.entity_refs or ()
    held = {site.path: site.value for site in ref_sites(refs, step)}
    return {site.path[:len(site.path) - len(str(site.key))] + site.version_key for site in ref_sites(refs, origin)
            if site.version_key is not None and held.get(site.path, site.value) != site.value}


def _given_back_items(step: dict, origin: dict, at: str, fields: list[str]) -> list[str]:
    """Give each item of the list ``at`` the ``fields`` the import left out of it, from the stored item it was written
    from (``_source_item``): all of them, or none when the item sets one, since together they can be one choice, such as
    a source under either of its names."""
    items, stored = _at(step, at), _at(origin, at)
    if not isinstance(items, list) or not isinstance(stored, list):
        return []
    left_out = {field.split(".")[0] for field in fields}
    given = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or any(_at(item, field) is not None for field in fields):
            continue
        if (source := _source_item(item, stored, left_out)) is None:
            continue
        for field in fields:
            for _, holder, key in field_sites(source, field):
                if holder[key] not in ("", [], {}) and _graft(item, field.split("."), holder[key]):
                    given.append(f"{at}[{index}].{field}")
    return given


def _source_item(item: dict, stored: list, left_out: set[str]) -> dict | None:
    """The stored item ``item`` was written from: the one it equals but for the ``left_out`` fields, else the one alone
    in sharing its name, when ``item`` sets nothing that one doesn't."""
    candidates = [one for one in stored if isinstance(one, dict)]
    bare = _bare(item, left_out)
    same = [one for one in candidates if _bare(one, left_out) == bare]
    if not same and item.get("name") is not None:
        same = [one for one in candidates
                if one.get("name") == item["name"] and bare.keys() <= _bare(one, left_out).keys()]
    return same[0] if len(same) == 1 else None


def _bare(item: dict, left_out: set[str]) -> dict:
    return {key: value for key, value in item.items() if key not in left_out and value is not None}


def _at(step: dict, path: str) -> object:
    value = step
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _unscrubbed(value: object, stored: object, scrubbed: object, path: str = "") -> list[str]:
    """Put back in ``value``, in place, what scrubbing ``stored`` took where ``value`` holds what it left
    (``scrubbed``): each env var it dropped that ``value`` doesn't set, and each credential it masked; each one's
    path."""
    given = []
    if isinstance(value, dict) and isinstance(stored, dict) and isinstance(scrubbed, dict):
        for key, item in stored.items():
            at = f"{path}.{key}" if path else str(key)
            if key not in scrubbed:
                if key not in value:
                    value[key] = copy.deepcopy(item)
                    given.append(at)
            elif key in value:
                if isinstance(item, str):
                    if value[key] == scrubbed[key] != item:
                        value[key] = item
                        given.append(at)
                else:
                    given += _unscrubbed(value[key], item, scrubbed[key], at)
    elif isinstance(value, list) and isinstance(stored, list) and isinstance(scrubbed, list) and (
            len(stored) == len(scrubbed)):
        unchanged = value == scrubbed
        for index, item in enumerate(value):
            if (source := _counterpart(item, index, stored, scrubbed, unchanged)) is None:
                continue
            at = f"{path}[{index}]"
            if isinstance(stored[source], str):
                if item == scrubbed[source] != stored[source]:
                    value[index] = stored[source]
                    given.append(at)
            else:
                given += _unscrubbed(item, stored[source], scrubbed[source], at)
    return given


def _counterpart(item: object, index: int, stored: list, scrubbed: list, unchanged: bool) -> int | None:
    """The index of the stored item ``item``, at ``index``, was scrubbed from: its own while the list is as scrubbed;
    else the one scrubbed item it equals, or failing that the one whose masked values it holds, when only one does, so
    a reordered or edited list restores each item from its own."""
    if unchanged:
        return index
    same = [other for other, one in enumerate(scrubbed) if one == item]
    if not same:
        same = [other for other, one in enumerate(scrubbed) if (masks := _masks(stored[other], one))
                and all(_at_keys(item, keys) == text for keys, text in masks)]
    return same[0] if len(same) == 1 else None


def _masks(stored: object, scrubbed: object, keys: tuple = ()) -> list[tuple[tuple, str]]:
    """Each string scrubbing ``stored`` masked: the keys to it, and the text the scrub left."""
    if isinstance(stored, str):
        return [(keys, scrubbed)] if isinstance(scrubbed, str) and scrubbed != stored else []
    if isinstance(stored, dict) and isinstance(scrubbed, dict):
        return [mask for key in stored if key in scrubbed for mask in _masks(stored[key], scrubbed[key], (*keys, key))]
    if isinstance(stored, list) and isinstance(scrubbed, list) and len(stored) == len(scrubbed):
        return [mask for index, (one, other) in enumerate(zip(stored, scrubbed))
                for mask in _masks(one, other, (*keys, index))]
    return []


def _at_keys(value: object, keys: tuple) -> object:
    for key in keys:
        if isinstance(value, dict) and isinstance(key, str):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and key < len(value):
            value = value[key]
        else:
            return None
    return value


def _texts(value: object, path: str = "") -> list[tuple[str, str]]:
    """Each string in ``value``, with its path."""
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in _texts(item, f"{path}.{key}" if path else str(key))]
    if isinstance(value, list):
        return [text for index, item in enumerate(value) for text in _texts(item, f"{path}[{index}]")]
    return []


def _graft(step: dict, keys: list[str], value: object) -> bool:
    """Set ``keys`` in ``step`` to ``value`` when it's unset there; whether it was."""
    holder = step
    for key in keys[:-1]:
        child = holder.get(key)
        if child is None:
            child = holder[key] = {}
        if not isinstance(child, dict):
            return False
        holder = child
    if holder.get(keys[-1]) is not None:
        return False
    holder[keys[-1]] = copy.deepcopy(value)
    return True


def _within(files: Mapping[str, str], folder: str) -> dict[str, str]:
    return {path: digest for path, digest in files.items() if path.startswith(f"{folder}/")}


def _written_from_folder(entry: BundleEntry) -> bool:
    return env_folder_writer(get_env_registry().get(entry.type))
