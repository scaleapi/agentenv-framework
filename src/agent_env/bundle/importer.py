"""Write a stored task, or an eval and its tasks, as a bundle folder holding every env, agent and artifact they use, so it
runs with no store; nothing is written to a store."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import unicodedata
import uuid
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.registry import ARTIFACT_REGISTRY, canonical_type, get_artifact_registry
from agent_env.bundle import _toml
from agent_env.bundle._fs import fold, os_reason, show
from agent_env.bundle._images import ImageJob, context_groups, write_images
from agent_env.bundle._secrets import scrub
from agent_env.bundle.installed import InstalledBundle, installed_bundles
from agent_env.bundle.parse import (
    CONFIG_FILES,
    LEAVINGS,
    BundleError,
    BundleKind,
    parse_bundle,
    root_problems,
    segment_problem,
)
from agent_env.bundle.plan import STORE_GETTERS, check_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import get_config
from agent_env.entity.export import ExportContext, Unwritable, checked_path
from agent_env.entity.refs import EntityKind, EntityRef, RefRole, RefSite, default_agent_refs, ref_sites
from agent_env.entity.scopes import FieldScope, prune
from agent_env.env import registry as env_registry
from agent_env.env.env import Env
from agent_env.env.registry import get_env_registry
from agent_env.eval import Eval
from agent_env.plugins import ARTIFACTS, ENVS, TASK_STEPS, inventory, load_failures
from agent_env.store.base import NotFoundError
from agent_env.store.ids import LOCAL_PREFIX
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.task import Task
from agent_env.task_step import registry as task_step_registry
from agent_env.task_step.registry import get_task_step_registry
from agent_env.utils.docker.lock import Locked

# How long a name taken from an id may be, in bytes; ids are capped far above it, and a longer name helps nobody.
_MAX_NAME_BYTES = 200
# The file an import records what it wrote in, and from where, for publishing the folder back.
PROVENANCE = "provenance.json"
_DOWNLOADS = 8  # objects downloaded at once
_FOLDERS = {EntityKind.ENV: BundleKind.ENV, EntityKind.AGENT: BundleKind.AGENT, EntityKind.ARTIFACT: BundleKind.ARTIFACT}


@dataclass(frozen=True)
class Pinned:
    """A reference left to the store, pinned at the version the store held: the folder can't hold what it names."""

    where: str
    kind: EntityKind
    id: str
    version: int
    why: str


@dataclass(frozen=True)
class Floating:
    """A store reference written without a version, so a run reads whatever version is current then."""

    where: str
    kind: EntityKind
    id: str
    why: str


@dataclass(frozen=True)
class Dropped:
    """A value left out of the folder, or masked in it: a secret, a scoped field, or one its toml has no key for."""

    where: str
    why: str


@dataclass(frozen=True)
class Need:
    """Something a task uses that core doesn't ship, and what registers it here."""

    where: str
    what: str  # step type, env type or artifact type
    name: str
    by: str  # the installed distribution that registers it, or "configuration" for a config file's impls


@dataclass(frozen=True)
class Written:
    """An entity written into the folder."""

    folder: str  # in the bundle: envs/<name>, agents/<name>, artifacts/<name> or skills/<name>
    kind: EntityKind
    id: str
    version: int
    type: str


@dataclass(frozen=True)
class WrittenImage:
    """An image written into its entity's folder, from its stored build context."""

    folder: str
    key: str
    artifact: str
    version: int
    dockerfile: str
    found: bool  # the image records no Dockerfile, so the context's one was taken
    files: int
    size: int
    whole: str | None  # why all of the context was kept, when it was
    largest: tuple[tuple[str, int], ...]
    locked: Locked | None  # what was pinned to the stored image's versions
    unlocked: str | None  # why nothing was


@dataclass(frozen=True)
class Pulled:
    """An image a folder names by its registry ref, which the machine running it pulls."""

    folder: str
    key: str
    image: str


@dataclass(frozen=True)
class ImportedTask:
    path: str  # in the bundle: tasks/<name>.json
    id: str
    version: int
    steps: int


@dataclass(frozen=True)
class ImportedEval:
    path: str  # in the bundle: evals/<name>.toml
    id: str
    version: int


@dataclass(frozen=True)
class BundleImport:
    """What an import wrote, and how it treated each reference and secret."""

    root: Path
    files: tuple[str, ...]  # what it wrote beside the entities' folders
    tasks: tuple[ImportedTask, ...]
    eval: ImportedEval | None
    written: tuple[Written, ...]
    images: tuple[WrittenImage, ...]
    pulled: tuple[Pulled, ...]
    pinned: tuple[Pinned, ...]
    floating: tuple[Floating, ...]
    dropped: tuple[Dropped, ...]
    needs: tuple[Need, ...]
    undeclared: tuple[str, ...]  # the steps whose type declares no references, so any they make float
    duplicates: tuple[str, ...]  # the eval's entries that named a task it already named
    downloaded: int  # bytes, of what the store said its objects' sizes are

    def report(self) -> list[tuple[str, list[str]]]:
        """What still needs the store, what the task needs beyond core, what was written, and how each reference,
        secret and scoped field was treated, as titled lists of lines; an empty section is left out."""
        needed: dict[tuple[str, str, str], list[str]] = {}
        for need in self.needs:
            needed.setdefault((need.by, need.name, need.what), []).append(need.where)
        return _sections([
            ("Still needs the store", [f"{pin.kind.value} {pin.id} v{pin.version}: {pin.where}: {pin.why}"
                                       for pin in self.pinned]),
            ("Needs beyond core", [f"{by}: {name} ({what}), at {'; '.join(wheres)}"
                                   for (by, name, what), wheres in sorted(needed.items())]),
            ("Written", [f"{entity.folder}: {entity.type} {entity.id} v{entity.version}" for entity in self.written]),
            ("Images", [_image_line(image) for image in self.images]
             + [f"{pull.folder}: {pull.key} is pulled from its registry, as {pull.image}" for pull in self.pulled]),
            ("Floating", [f"{ref.kind.value} {ref.id}: {ref.where}: {ref.why}" for ref in self.floating]
             + [f"{step}: its type declares no references, so any it makes float" for step in self.undeclared]),
            ("Left out", [f"{drop.where}: {drop.why}" for drop in self.dropped]
             + [f"{entry}, which the eval names more than once" for entry in self.duplicates]),
        ])


def import_task(task_id: str, dest: Path | str, *, version: int | None = None,
                include_scopes: Collection[FieldScope] = (),
                progress: Callable[[str], None] | None = None) -> BundleImport:
    """Write the stored task ``task_id``, at ``version`` or its latest, and every entity it uses as a bundle at
    ``dest``. Its steps' scoped fields are left out unless ``include_scopes`` names their scope."""
    importer = _Importer(Path(dest), include_scopes, progress)
    task = Task.get(task_id, version)
    importer.add_task(task)
    return importer.write(f"Imported from the store's task {task.id} v{task.version}.", {"task": task.id,
                                                                                         "version": task.version})


def import_eval(eval_id: str, dest: Path | str, *, version: int | None = None,
                include_scopes: Collection[FieldScope] = (),
                progress: Callable[[str], None] | None = None) -> BundleImport:
    """Write the stored eval ``eval_id``, at ``version`` or its latest, the tasks it names and every entity they use as
    a bundle, leaving out their steps' scoped fields as ``import_task`` does."""
    importer = _Importer(Path(dest), include_scopes, progress)
    stored = Eval.get(eval_id, version)
    names, seen = [], set()
    for entry in stored.tasks:
        try:
            task = Task.get(entry.task_id, entry.task_version)
        except NotFoundError:
            at = f" v{entry.task_version}" if entry.task_version is not None else ""
            importer.problems.append(f"eval {stored.id} v{stored.version} names task {entry.task_id}{at}, which isn't "
                                     "in the store")
            continue
        if (task.id, task.version) in seen:
            importer.duplicates.append(f"task {task.id} v{task.version}")
            continue
        seen.add((task.id, task.version))
        names.append(importer.add_task(task))
    if names:
        importer.add_eval(stored, names)
    return importer.write(f"Imported from the store's eval {stored.id} v{stored.version} and the tasks it names.",
                          {"eval": stored.id, "version": stored.version})


class _Abandoned(Exception):
    """An entity can't be written and the import keeps nothing in the store, which a problem already says."""


class _EntityOut(ExportContext):
    """What one entity's ``to_toml`` writes through, kept apart until it returns so a refusal leaves nothing behind."""

    def __init__(self, importer: _Importer, folder: str, toml_name: str | None):
        self.importer = importer
        self.folder = folder
        self.toml_name = toml_name
        self.files: dict[str, str] = {}
        self.folded: set[str] = set()
        self.images: list[ImageJob] = []
        self.pulled: list[Pulled] = []
        self.left: list[Dropped] = []

    def ref(self, entity: Any) -> str | dict:
        kind = _kind_of(entity)
        local = self.importer.local(kind, entity, self.folder)
        return local if local is not None else {kind.value: entity.id, "version": entity.version}

    def image(self, artifact: DockerImageArtifact, key: str = "image") -> str | dict:
        return self.importer.image(self, artifact, key)

    def file(self, path: str, object_url: str) -> None:
        path = checked_path(path)
        if path == self.toml_name:
            raise Unwritable(f"it holds a file named {path}, which its folder's own {self.toml_name} would replace")
        if leaving := LEAVINGS.intersection(path.split("/")):
            raise Unwritable(f"it holds {path}, and a bundle leaves out {', '.join(sorted(leaving))} as the OS or "
                             "Python's own")
        if fold(path) in self.folded:
            raise Unwritable(f"it holds two files at {path}, by names that differ only in case")
        self.folded.add(fold(path))
        self.files[path] = object_url

    def left_out(self, field: str, why: str) -> None:
        self.left.append(Dropped(f"{self.folder}: {field}", why))


class _Importer:
    def __init__(self, dest: Path, include_scopes: Collection[FieldScope] = (),
                 progress: Callable[[str], None] | None = None):
        self.dest = Path(os.path.abspath(dest.expanduser()))  # so ``.`` has a name, and a sibling to stage in
        self.include_scopes = frozenset({FieldScope.TASK, *include_scopes})
        self.progress = progress or (lambda message: None)
        self.files: dict[str, str] = {}
        self.taken: dict[str, set[str]] = {kind.value: set() for kind in BundleKind}  # folded names, per folder
        self.tasks: list[ImportedTask] = []
        self.eval: ImportedEval | None = None
        self.written: list[Written] = []
        self.images: list[WrittenImage] = []
        self.pulled: list[Pulled] = []
        self.pinned: list[Pinned] = []
        self.floating: list[Floating] = []
        self.dropped: list[Dropped] = []
        self.needs: list[Need] = []
        self.undeclared: list[str] = []
        self.duplicates: list[str] = []
        self.problems: list[str] = []
        self.entities: dict[tuple[EntityKind, str, int | None], Any] = {}  # each read, or why it can't be
        self.exports: dict[tuple[EntityKind, str, int], Any] = {}  # each entity's name in the folder, or None or _Abandoned
        self.tomls: dict[str, tuple[str, dict]] = {}  # each entity folder's toml: its file's name, and its keys
        self.downloads: dict[str, str] = {}  # each file to write: its path in the bundle, and its object's url
        self.jobs: list[ImageJob] = []
        self.origins: dict[tuple[str, str], str] | None = None
        self.installed: tuple[InstalledBundle, ...] | None = None
        _check_destination(self.dest)

    def add_task(self, task: Task) -> str:
        """Write ``task`` under a name taken from its id, and what it references in folders of their own; return its
        name."""
        name = self._name("tasks", task.id)
        path = f"tasks/{name}.json"
        steps = task.to_dict()["steps"]
        registry = get_task_step_registry()
        found = []
        for step in steps:
            cls = registry[step["type"]]
            pruned = prune(step, cls.field_scopes, self.include_scopes)  # first, so no pruned field is written
            found.append((step, cls, pruned, [site for site in ref_sites(cls.entity_refs or (), step)
                                              if isinstance(site.value, str) and site.value]))
        sites = [site for *_, step_sites in found for site in step_sites]
        outputs = _output_names(sites)
        for site in sites:  # each output, and each step that reads it, takes the output's name
            if outputs.get(_key(site), site.value) != site.value:
                site.rewrite(outputs[_key(site)])
        for step, cls, pruned, step_sites in found:
            where = f"{path}: step {step.get('id')!r}"
            self.dropped += [Dropped(f"{where}: {at}", why) for at, why in pruned + scrub(step)]
            self._need(where, "step type", TASK_STEPS, cls.type, cls)
            if cls.entity_refs is None:
                self.undeclared.append(f"{where} ({cls.type})")
            for site in step_sites:
                if _key(site) not in outputs:  # the run writes an output again, so it's no entity to write here
                    self._ref(site, f"{where}: {site.path}")
            self._default_agents(step, cls.entity_refs or (), where)
        self.files[path] = json.dumps(steps, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.tasks.append(ImportedTask(path, task.id, task.version, len(steps)))
        return name

    def add_eval(self, stored: Eval, names: list[str]) -> None:
        path = f"evals/{self._name('evals', stored.id)}.toml"
        self.files[path] = _toml.dumps({"tasks": names})
        self.eval = ImportedEval(path, stored.id, stored.version)

    def local(self, kind: EntityKind, entity: Any, where: str) -> str | None:
        """The name ``entity`` takes in the folder, writing it there, or None when it stays in the store."""
        key = (kind, entity.id, entity.version)
        if key in self.exports:
            if self.exports[key] is _Abandoned:
                raise _Abandoned
            return self.exports[key]
        if getattr(type(entity), "to_toml", None) is None:
            return self._unwritable(key, where, f"a {entity.type} {kind.value} has no form a bundle folder holds")
        try:
            bundle_kind, name = self._folder(kind, entity)
        except Unwritable as e:
            return self._unwritable(key, where, str(e))
        folder = f"{bundle_kind.value}/{name}"
        self.exports[key] = name  # before to_toml, so a reference back to it finds its name
        out = _EntityOut(self, folder, CONFIG_FILES.get(bundle_kind))
        try:
            toml = entity.to_toml(out)
        except Unwritable as e:
            self.taken[_names_of(bundle_kind)].discard(fold(name))
            del self.exports[key]
            return self._unwritable(key, where, str(e))
        if out.toml_name is not None:
            self.dropped += [Dropped(f"{folder}/{out.toml_name}: {at}", why) for at, why in scrub(toml)]
            self.tomls[folder] = (out.toml_name, toml)
        self.downloads |= {f"{folder}/{path}": url for path, url in out.files.items()}
        self.jobs += out.images
        self.pulled += out.pulled
        self.dropped += out.left
        self.written.append(Written(folder, kind, entity.id, entity.version, entity.type))
        return name

    def image(self, out: _EntityOut, artifact: DockerImageArtifact, key: str) -> str | dict:
        """The toml value of an image a folder is built from: its stored context, pruned into the folder, else its
        registry ref, else a pinned store image."""
        if artifact.build_context_object_url:
            job = ImageJob(artifact, out.folder, key, {"dockerfile": artifact.dockerfile_path or ""}, out.toml_name)
            out.images.append(job)
            return job.toml
        if names_registry(artifact.image_name):
            out.pulled.append(Pulled(out.folder, key, artifact.image_name))
            return {"ref": artifact.image_name}
        why = (f"it's stored only as a docker save tarball, under {artifact.image_name}, which names no registry"
               if artifact.tar_gz_object_url else "it records neither a build context nor a registry to pull it from")
        self._unwritable((EntityKind.ARTIFACT, artifact.id, artifact.version), f"{out.folder}: {key}", why)
        return {EntityKind.ARTIFACT.value: artifact.id, "version": artifact.version}

    def write(self, description: str, source: dict) -> BundleImport:
        """Write the folder, with ``description`` as its README's first line, and say what was written."""
        if self.problems:
            raise BundleError(self.problems)
        staging = self.dest.parent / f".{self.dest.name}.import-{uuid.uuid4().hex[:8]}"
        try:
            try:
                staging.mkdir()
                scratch = staging / ".import"
                scratch.mkdir()
            except OSError as e:
                raise BundleError([f"{show(self.dest.parent)}: can't write the import there: {os_reason(e)}"]) from None
            downloaded = self._download(staging, scratch)
            if self.problems:
                raise BundleError(self.problems)
            shutil.rmtree(scratch)
            self.files[PROVENANCE] = _provenance(source, self)
            files = sorted([*self.files, "README.md"])
            imported = BundleImport(
                root=self.dest, files=tuple(files), tasks=tuple(self.tasks), eval=self.eval,
                written=tuple(self.written), images=tuple(self.images), pulled=tuple(self.pulled),
                pinned=tuple(self.pinned), floating=tuple(self.floating), dropped=tuple(self.dropped),
                needs=tuple(self.needs), undeclared=tuple(self.undeclared), duplicates=tuple(self.duplicates),
                downloaded=downloaded)
            self.files["README.md"] = _readme(description, imported)
            try:
                for folder, (name, toml) in self.tomls.items():
                    (staging / folder).mkdir(parents=True, exist_ok=True)
                    (staging / folder / name).write_text(_toml.dumps(toml), encoding="utf-8")
                for path, text in self.files.items():
                    (staging / path).parent.mkdir(parents=True, exist_ok=True)
                    (staging / path).write_text(text, encoding="utf-8")
            except OSError as e:
                raise BundleError([f"{show(self.dest.parent)}: can't write the import there: {os_reason(e)}"]) from None
            check_bundle(resolve_bundle(parse_bundle(staging)), installed=installed_bundles)
            self._move(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return imported

    def _download(self, staging: Path, scratch: Path) -> int:
        """Download every file and image the folder holds into ``staging``, saying how much first; return the bytes."""
        groups = context_groups(self.jobs)
        saved = [job.artifact.tar_gz_object_url for job in self.jobs if job.artifact.tar_gz_object_url]
        with ThreadPoolExecutor(_DOWNLOADS) as pool:
            sizes = [*pool.map(_size, self.downloads.values()), *pool.map(_saved_size, saved)]
        total = sum(size or 0 for size in sizes) + sum(group[0].size or 0 for group in groups)
        if self.downloads or groups:
            stored = f", and {len(saved)} stored image(s) to read their versions from" if saved else ""
            self.progress(f"Downloading {len(self.downloads)} file(s) and {len(groups)} build context(s){stored}, "
                          f"{_size_text(total)}...")
        try:
            with ThreadPoolExecutor(_DOWNLOADS) as pool:
                list(pool.map(lambda item: _fetch(item[1], staging / item[0]), self.downloads.items()))
            write_images(groups, staging, scratch,
                         on_group=lambda group: self.progress(f"  build context of {', '.join(job.folder for job in group)}"))
        except (OSError, ValueError) as e:
            raise BundleError([f"can't download what the folder holds: {e}"]) from None
        for job in self.jobs:
            if job.problem is not None:
                self._image_left_in_store(job)
            else:
                self.images.append(WrittenImage(
                    job.folder, job.key, job.artifact.id, job.artifact.version, job.pruned.dockerfile,
                    found=not job.artifact.dockerfile_path, files=job.pruned.files, size=job.pruned.size,
                    whole=job.pruned.whole, largest=job.pruned.largest, locked=job.locked, unlocked=job.unlocked))
        return total

    def _image_left_in_store(self, job: ImageJob) -> None:
        """An image whose context the folder can't hold: a problem, or a pinned store image when the import keeps one."""
        key = (EntityKind.ARTIFACT, job.artifact.id, job.artifact.version)
        try:
            self._unwritable(key, f"{job.folder}: {job.key}", job.problem)
        except _Abandoned:
            return
        job.toml.clear()
        job.toml.update({EntityKind.ARTIFACT.value: job.artifact.id, "version": job.artifact.version})

    def _unwritable(self, key: tuple[EntityKind, str, int], where: str, why: str) -> None:
        """An entity the folder can't hold: left in the store, pinned, with ``--include-scope user``; else a problem."""
        kind, entity_id, version = key
        if FieldScope.USER not in self.include_scopes:
            self.problems.append(f"{where}: {kind.value} {entity_id} v{version} can't be written into the folder: "
                                 f"{why}; with --include-scope user it stays in the store, for a folder run against the "
                                 "same store")
            self.exports[key] = _Abandoned
            raise _Abandoned
        self.exports[key] = None
        self.pinned.append(Pinned(where, kind, entity_id, version, why))
        return None

    def _ref(self, site: RefSite, where: str) -> None:
        """Write what ``site`` references into the folder and name it there, or leave it in the store, pinned."""
        kind, entity_id = site.ref.kind, site.value
        if (bundle := self._installed_for(kind, entity_id)) is not None:
            self._from_installed(site, where, bundle)
            return
        entity = self._entity(kind, entity_id, site.version)
        if isinstance(entity, str):
            self.problems.append(f"{where}: {entity}")
            return
        self._need_type(site, entity, where)
        try:
            local = self.local(kind, entity, where)
        except _Abandoned:
            return
        if local is not None:
            site.rewrite(local)
            if site.version_key is not None and site.version is not None:
                site.owner[site.version_key] = None  # what the bundle holds has no fixed version
        elif site.version_key is not None:
            site.rewrite(entity_id, entity.version)
        else:
            self.floating.append(Floating(where, kind, entity_id, "the field takes no version, so a run reads the "
                                                                  "latest"))

    def _folder(self, kind: EntityKind, entity: Any) -> tuple[BundleKind, str]:
        """The kind folder ``entity`` goes in, and the name it takes there; a skill's is its skill's name."""
        if isinstance(entity, SkillArtifact):
            name = entity.skill_name
            if fold(name) in self.taken[BundleKind.ARTIFACT.value]:
                raise Unwritable(f"a skill's folder takes its skill's name, {name!r}, which another artifact in the "
                                 "folder already has")
            self.taken[BundleKind.ARTIFACT.value].add(fold(name))
            return BundleKind.SKILL, name
        bundle_kind = _FOLDERS[kind]
        return bundle_kind, self._name(bundle_kind.value, entity.id)

    def _entity(self, kind: EntityKind, entity_id: str, version: int | None = None) -> Any:
        """``entity_id`` as the store holds it, at ``version`` or its latest, or why it can't be read."""
        if (kind, entity_id, version) not in self.entities:
            try:
                self.entities[kind, entity_id, version] = STORE_GETTERS[kind](entity_id, version)
            except NotFoundError:
                at = f" v{version}" if version is not None else ""
                self.entities[kind, entity_id, version] = f"there is no {kind.value} {entity_id!r}{at} in the store"
            except (ValueError, KeyError, TypeError) as e:
                self.entities[kind, entity_id, version] = (f"{kind.value} {entity_id!r} can't be read "
                                                           f"({type(e).__name__}: {e})")
        return self.entities[kind, entity_id, version]

    def _move(self, staging: Path) -> None:
        """Move ``staging`` to the destination, putting an emptied one back if the move fails or is interrupted."""
        emptied = False
        try:
            if self.dest.is_dir():
                self.dest.rmdir()  # empty, as _check_destination found it
                emptied = True
            os.replace(staging, self.dest)
        except BaseException as e:
            if emptied:
                with contextlib.suppress(OSError):
                    self.dest.mkdir(exist_ok=True)
            if not isinstance(e, OSError):
                raise
            _check_destination(self.dest)  # something else wrote it meanwhile
            raise BundleError([f"{show(self.dest)}: can't move the import into place: {os_reason(e)}"]) from None

    def _installed_for(self, kind: EntityKind, entity_id: str) -> InstalledBundle | None:
        """The installed bundle whose ids ``entity_id``, an env, agent or artifact, is under: a run writes it from
        there."""
        if kind is EntityKind.TASK or not entity_id.startswith(LOCAL_PREFIX):
            return None
        if self.installed is None:
            self.installed = installed_bundles()
        return next((bundle for bundle in self.installed if entity_id.startswith(f"{bundle.id_root}/")), None)

    def _from_installed(self, site: RefSite, where: str, bundle: InstalledBundle) -> None:
        """An installed bundle's entity has no fixed version, so it isn't pinned, and a pin the store held is dropped."""
        why = f"it comes with the installed bundle {bundle.qualified}, which a run writes it from"
        if site.version is not None and site.version_key is not None:
            site.owner[site.version_key] = None
            why += f"; the stored pin to v{site.version} is dropped, as such an entity has no fixed version"
        self.floating.append(Floating(where, site.ref.kind, site.value, why))

    def _need_type(self, site: RefSite, entity: Any, where: str) -> None:
        """An env or artifact the task references whose type core doesn't ship is a need."""
        if site.ref.kind is EntityKind.ENV:
            self._need(where, "env type", ENVS, entity.type, get_env_registry().get(entity.type))
        elif site.ref.kind is EntityKind.ARTIFACT:
            self._need(where, "artifact type", ARTIFACTS, (name := canonical_type(entity.type)),
                       get_artifact_registry().get(name))

    def _need(self, where: str, what: str, group: str, name: str, cls: type | None) -> None:
        """``name``, of the plugin ``group``, is a need when core doesn't ship it, named with what registers ``cls``,
        the class it resolves to here: the distribution whose entry point names that class, or configuration. A plugin
        the registry build skipped registers nothing, so it isn't loaded again to compare."""
        if name in _BUILTINS[group]():
            return
        if self.origins is None:
            self.origins = {(contribution.group, contribution.name): (distribution.name, contribution.value)
                            for distribution in inventory(load=False).distributions
                            for contribution in distribution.contributions}
        distribution, value = self.origins.get((group, name), (None, None))
        registered = value is not None and cls is not None and name not in load_failures().get(group, {})
        by = distribution if registered and _names(value, cls) else "configuration"
        self.needs.append(Need(where, what, name, by))

    def _default_agents(self, step: dict, refs: tuple[EntityRef, ...], where: str) -> None:
        """A step that leaves an agent field unset falls back to the configured default, which no field pins."""
        for ref in default_agent_refs(refs, step):
            self.floating.append(Floating(f"{where}: {ref.path}", EntityKind.AGENT,
                                          get_config().get_default_a2a_agent_id(),
                                          f"the step leaves {ref.path} unset, so it falls back to the configured "
                                          "default"))

    def _name(self, kind: str, entity_id: str) -> str:
        """A name in the ``kind`` folder: the id's last segment made valid, numbered when taken, caselessly."""
        names = _names_of(BundleKind(kind))
        base = _valid_name(entity_id.rsplit("/", 1)[-1]) or kind.removesuffix("s")
        name, n = base, 1
        while fold(name) in self.taken[names]:
            n += 1
            name = f"{base}-{n}"
        self.taken[names].add(fold(name))
        return name


# What core ships, per plugin group: anything else a task uses is a need.
_BUILTINS = {TASK_STEPS: task_step_registry._builtin_registry, ENVS: env_registry._builtin_registry,
             ARTIFACTS: lambda: ARTIFACT_REGISTRY}


def _names_of(kind: BundleKind) -> str:
    """The folder whose names ``kind``'s share: artifacts' and skills' are one."""
    return BundleKind.ARTIFACT.value if kind is BundleKind.SKILL else kind.value


def _kind_of(entity: Any) -> EntityKind:
    if isinstance(entity, Env):
        return EntityKind.ENV
    if isinstance(entity, A2AAgent):
        return EntityKind.AGENT
    if isinstance(entity, Artifact):
        return EntityKind.ARTIFACT
    raise TypeError(f"{entity!r} is no env, agent or artifact")


def _size(url: str) -> int | None:
    metadata = get_config().get_object_store_at(url).get_object_metadata_at(url)
    return metadata.size if metadata is not None else None


def _saved_size(url: str) -> int | None:
    """A stored image's size, None when it can't be read: its versions are then left unpinned, not the import failed."""
    try:
        return _size(url)
    except (ValueError, OSError):
        return None


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    get_config().get_object_store_at(url).download_to_file(url, str(dest))


def _size_text(size: int) -> str:
    return f"{size} B" if size < 1_000 else f"{size / 1_000:.1f} KB" if size < 1_000_000 else f"{size / 1_000_000:.1f} MB"


def _image_line(image: WrittenImage) -> str:
    """How an image was written: its Dockerfile, what its folder holds, and why it holds all of its context."""
    found = " (found: the image records no Dockerfile)" if image.found else ""
    largest = ", ".join(f"{name} ({_size_text(size)})" for name, size in image.largest)
    whole = f"; all of its build context, since {image.whole}" if image.whole else ""
    return (f"{image.folder}: {image.key} built from {image.dockerfile}{found}, {image.files} file(s), "
            f"{_size_text(image.size)}{whole}; largest: {largest}; {_locked_text(image)}")


def _locked_text(image: WrittenImage) -> str:
    """What of an image's installs is pinned to the stored image's versions, and what isn't."""
    if image.locked is None:
        return f"versions not pinned: {image.unlocked}"
    locked = image.locked
    pinned = ([f"{locked.python} Python package(s) by {locked.constraints}"] if locked.constraints else []) + (
        [f"Python packages by {locked.lock_file}, as the build installs them"] if locked.lock_file else []) + list(
        locked.npm)
    left = f"; left as the build resolves them: {'; '.join(locked.not_covered)}" if locked.not_covered else ""
    return f"pinned to the stored image: {', '.join(pinned) or 'nothing it installs'}{left}"


def _names(value: str, cls: type) -> bool:
    """Whether the entry point ``value``, ``module:attribute``, names ``cls``."""
    module, _, attribute = value.partition(":")
    try:
        named = importlib.import_module(module)
        for part in attribute.split("."):
            named = getattr(named, part)
    except (ImportError, AttributeError):
        return False
    return named is cls


def _output_names(sites: list[RefSite]) -> dict[tuple[EntityKind, str], str]:
    """The name each output takes: as written, or an ``@local`` id's last segment, numbered when it is taken."""
    written = {_key(site) for site in sites if site.ref.role is RefRole.OUTPUT}
    taken = {_key(site) for site in sites if _key(site) not in written or not site.value.startswith(LOCAL_PREFIX)}
    names: dict[tuple[EntityKind, str], str] = {}
    for site in sites:
        if site.ref.role is not RefRole.OUTPUT or _key(site) in names:
            continue
        name = site.value
        if name.startswith(LOCAL_PREFIX):
            base = name = name.rsplit("/", 1)[-1]
            n = 1
            while (site.ref.kind, _nfc(name)) in taken:
                n += 1
                name = f"{base}-{n}"
            taken.add((site.ref.kind, _nfc(name)))
        names[_key(site)] = name
    return names


def _provenance(source: dict, importer: _Importer) -> str:
    """What the folder was imported from and what it holds, by store id and version: no store's locations."""
    return json.dumps({
        "agent_env": package_version("agentenv-framework"),
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": source,
        "written": [{"folder": entity.folder, "kind": entity.kind.value, "type": entity.type, "id": entity.id,
                     "version": entity.version} for entity in importer.written],
        "images": [{"folder": job.folder, "key": job.key, "id": job.artifact.id, "version": job.artifact.version,
                    "dockerfile": job.toml.get("dockerfile"), "dockerfile_found": not job.artifact.dockerfile_path}
                   for job in importer.jobs if job.problem is None],
        "store": [{"where": pin.where, "kind": pin.kind.value, "id": pin.id, "version": pin.version, "why": pin.why}
                  for pin in importer.pinned],
    }, indent=2, ensure_ascii=False) + "\n"


def _readme(description: str, imported: BundleImport) -> str:
    """The bundle's README: its description, where it came from, how to run and share it, what it holds, then the
    import's report."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    held = ("Every env, agent and artifact it uses is written here, so `agent-env run` runs it with no store"
            + (", except what *Still needs the store* lists, which it reads from the store this was imported from."
               if imported.pinned else "."))
    lines = [description, "", f"Imported with agent-env {package_version('agentenv-framework')} on {stamp}. {held}", "",
             *_RUN_IT, *_needs_lines(imported), "", *_SHARE_IT, ""]
    contents = _sections([("Tasks", [f"{task.path}: task {task.id} v{task.version}" for task in imported.tasks]),
                          ("Eval", [f"{imported.eval.path}: eval {imported.eval.id} v{imported.eval.version}"]
                           if imported.eval else [])])
    for title, items in [*contents, *imported.report()]:
        lines += [f"## {title}", "", *(f"- {item}" for item in items), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


_RUN_IT = [
    "## Run it",
    "",
    "    agent-env run .",
    "",
    "A machine running it needs:",
    "",
    "- agent-env (`pip install agentenv-framework`).",
    "- Docker with its Compose and Buildx plugins. On Linux, Docker Engine and the `docker-compose-plugin` and "
    "`docker-buildx-plugin` packages (or a distribution's `docker.io`, `docker-compose-v2` and `docker-buildx`). On "
    "macOS, a Docker VM such as Docker Desktop or colima, and the `docker`, `docker-compose` and `docker-buildx` CLIs; "
    "Homebrew's plugins are found once `~/.docker/config.json` lists their folder in `cliPluginsExtraDirs`.",
    "- A model endpoint for its agents and judges: `LITELLM_BASE_URL` and `LITELLM_API_KEY`, or `[model] base_url` and "
    "`api_key` in `.agentenv/config.toml`.",
    "- An agent where a step names none: the configured default (`[agents] default_a2a_agent_id`).",
    "",
    "Its images are built from the build contexts in their folders on the machine that runs it, pinned to the "
    "Python and npm versions of the images the store held, as *Images* says. To build one unpinned, take out the "
    "COPY and two ARG lines naming agent-env-constraints.txt after each FROM in its Dockerfile.",
]
_SHARE_IT = [
    "## Share it",
    "",
    "The folder is source: commit it to git, or archive it without macOS's extended attributes, which tar on Linux warns "
    "about for every file (`COPYFILE_DISABLE=1 tar -czf` on macOS, or GNU tar's `--no-xattrs`).",
]


def _needs_lines(imported: BundleImport) -> list[str]:
    plugins = sorted({need.by for need in imported.needs if need.by != "configuration"})
    configured = any(need.by == "configuration" for need in imported.needs)
    lines = [f"- The plugins it uses: `{'`, `'.join(plugins)}`."] if plugins else []
    if configured:
        lines.append("- The configuration that registers what *Needs beyond core* says configuration registers.")
    return lines


def _sections(sections: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    return [(title, lines) for title, lines in sections if lines]


def _check_destination(dest: Path) -> None:
    """Refuse a destination that holds anything, isn't a folder, or would give the bundle no id."""
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise BundleError([f"{show(dest)}: not a folder; import into a new or empty folder"])
    if dest.is_dir() and any(dest.iterdir()):
        raise BundleError([f"{show(dest)}: not empty; import into a new or empty folder"])
    if not dest.parent.is_dir():
        raise BundleError([f"{show(dest.parent)}: no such folder"])
    problems = root_problems(dest)
    if problems:
        raise BundleError(problems)


def _valid_name(text: str) -> str:
    """``text`` as one bundle name, each character an id can't hold made ``-``; empty with no letter or digit left."""
    text = _nfc(text)
    name = "".join(char if char.isalnum() or unicodedata.category(char).startswith("M") or char in " ._-~@()+,&"
                   else "-" for char in text).strip(" ").lstrip(".")
    while len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        name = name[:-1].rstrip(" ")
    return name if any(char.isalnum() for char in name) and segment_problem(name) is None else ""


def _key(site: RefSite) -> tuple[EntityKind, str]:
    return site.ref.kind, _nfc(site.value)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)
