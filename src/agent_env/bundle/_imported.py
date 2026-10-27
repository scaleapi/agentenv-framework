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
from agent_env.entity.scopes import FieldScope
from agent_env.env.registry import get_env_registry
from agent_env.store.base import NotFoundError
from agent_env.task.task import Task
from agent_env.task_step.registry import get_task_step_registry

from ._import_report import PROVENANCE, PROVENANCE_SCHEMA, store_fingerprint
from .parse import CONFIG_FILES, Bundle, BundleEntry, BundleKind
from .plan import Plan, Write, env_folder_writer
from .resolve import BuiltImage, BundleImage


@dataclass(frozen=True)
class Imported:
    """What an import recorded: each entity folder, task file and eval file's store id and version, each image's by its
    folder and toml key, and each file as it wrote it."""

    entities: Mapping[str, tuple[str, int]]
    images: Mapping[tuple[str, str], tuple[str, int]]
    files: Mapping[str, str]


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
    images = {(item["folder"], item["key"]): (item["id"], item["version"]) for item in record.get("images", [])}
    return Imported(entities, images, record.get("files", {})), None


def where(root: Path, entry: BundleEntry) -> str:
    """``entry``'s folder or file in the bundle at ``root``, as the import's record names it."""
    return unicodedata.normalize("NFC", entry.path.relative_to(root).as_posix())


def source_of(imported: Imported, plan: Plan, write: Write) -> tuple[str, int] | None:
    """The store id and version ``write`` was imported from, or None when the import didn't write it."""
    entry = write.source.entry
    folder = where(plan.bundle_of(entry).root, entry)
    if isinstance(write.source, BundleImage) and write.source.id != entry.id:
        return imported.images.get((folder, image_key(write.source)))
    return imported.entities.get(folder)


def image_key(image: BundleImage) -> str:
    """The toml key an image an env or agent folder builds is named by: ``image`` for its own, as the import names it."""
    role = image.id[len(image.entry.id) + 2:]
    return "image" if role == f"{image.entry.kind.store}_image" else role


def as_imported(imported: Imported, plan: Plan, write: Write, now: Mapping[str, str]) -> bool:
    """Whether the files ``write`` is made from are those the import wrote: a task's or eval's file; an image's build
    context, its folder but the toml; an env's or agent's toml, or every file of an env type written from its folder;
    every file of an artifact or skill folder."""
    entry = write.source.entry
    folder = where(plan.bundle_of(entry).root, entry)
    if write.kind in (BundleKind.TASK, BundleKind.EVAL):
        return folder in imported.files and imported.files[folder] == now.get(folder)
    toml = f"{folder}/{CONFIG_FILES[entry.kind]}" if entry.kind in CONFIG_FILES else None
    then, current = _within(imported.files, folder), _within(now, folder)
    if isinstance(write.source, BuiltImage):
        then.pop(toml, None)
        current.pop(toml, None)
    elif isinstance(write.source, BundleImage) or (entry.kind in (BundleKind.ENV, BundleKind.AGENT) and not (
            entry.kind is BundleKind.ENV and _written_from_folder(entry))):
        then, current = {toml: then.get(toml)}, {toml: current.get(toml)}
    return then == current


def grafted(bundle: Bundle, imported: Imported) -> tuple[Bundle, dict[Path, set[str]]]:
    """``bundle`` with each imported task's steps given back the user-scoped fields the import left out, from the
    stored version it was imported from, matched by step id and type, where the folder doesn't set them; and, for each
    task's file, the paths in its steps that took one."""
    registry = get_task_step_registry()
    entries, taken = [], {}
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
            for path, scope in cls.field_scopes.items():
                if scope is not FieldScope.USER or "[]" in path:
                    continue
                for _, holder, key in field_sites(origin, path):
                    if holder[key] not in ("", [], {}) and _graft(step, path.split("."), holder[key]):
                        taken.setdefault(entry.path, set()).add(f"[{index}].{path}")
        entries.append(replace(entry, config=steps))
    return replace(bundle, entries=tuple(entries)), taken


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
