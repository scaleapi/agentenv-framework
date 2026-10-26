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
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.registry import ARTIFACT_REGISTRY, canonical_type, get_artifact_registry
from agent_env.bundle import _toml
from agent_env.bundle._fs import fold, show
from agent_env.bundle._images import ImageJob, context_groups, write_images
from agent_env.bundle._import_report import (
    PROVENANCE,
    BundleImport,
    Dropped,
    Floating,
    ImportedEval,
    ImportedTask,
    Need,
    Pinned,
    Pulled,
    Written,
    WrittenImage,
    provenance,
    readme,
    size_text,
)
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
from agent_env.task.task import Task
from agent_env.task_step import registry as task_step_registry
from agent_env.task_step.registry import get_task_step_registry
from agent_env.utils.paths import os_reason

# How long a name taken from an id may be, in bytes; ids are capped far above it, and a longer name helps nobody.
_MAX_NAME_BYTES = 200
_DOWNLOADS = 8  # objects downloaded at once
_FOLDERS = {EntityKind.ENV: BundleKind.ENV, EntityKind.AGENT: BundleKind.AGENT, EntityKind.ARTIFACT: BundleKind.ARTIFACT}


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
            self.left_out(path, f"dropped: a bundle leaves out {', '.join(sorted(leaving))} as the OS or Python's own")
            return
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
        pins: dict[str, set[int]] = {}  # the versions the task pins each env at
        for site in sites:
            if site.ref.kind is EntityKind.ENV and site.version is not None:
                pins.setdefault(_key(site)[1], set()).add(site.version)
        for step, cls, pruned, step_sites in found:
            where = f"{path}: step {step.get('id')!r}"
            self.dropped += [Dropped(f"{where}: {at}", why) for at, why in pruned + scrub(step)]
            self._need(where, "step type", TASK_STEPS, cls.type, cls)
            if cls.entity_refs is None:
                self.undeclared.append(f"{where} ({cls.type})")
            for site in step_sites:
                if _key(site) not in outputs:  # the run writes an output again, so it's no entity to write here
                    self._ref(site, f"{where}: {site.path}", pins)
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
            files = sorted([*self.files, PROVENANCE, "README.md"])
            imported = BundleImport(
                root=self.dest, files=tuple(files), tasks=tuple(self.tasks), eval=self.eval,
                written=tuple(self.written), images=tuple(self.images), pulled=tuple(self.pulled),
                pinned=tuple(self.pinned), floating=tuple(self.floating), dropped=tuple(self.dropped),
                needs=tuple(self.needs), undeclared=tuple(self.undeclared), duplicates=tuple(self.duplicates),
                downloaded=downloaded)
            self.files[PROVENANCE] = provenance(source, imported)
            self.files["README.md"] = readme(description, imported)
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
                          f"{size_text(total)}...")
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
                self.images.append(WrittenImage(job.folder, job.key, job.artifact.id, job.artifact.version,
                                                found=not job.artifact.dockerfile_path, pruned=job.pruned,
                                                locked=job.locked, unlocked=job.unlocked))
        return total

    def _image_left_in_store(self, job: ImageJob) -> None:
        """An image whose context the folder can't hold: its registry ref when it names one, else a problem, or a pinned
        store image when the import keeps one."""
        if names_registry(job.artifact.image_name):
            job.toml.clear()
            job.toml["ref"] = job.artifact.image_name
            self.pulled.append(Pulled(job.folder, job.key, job.artifact.image_name, job.problem))
            return
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

    def _ref(self, site: RefSite, where: str, pins: Mapping[str, set[int]]) -> None:
        """Write what ``site`` references into the folder and name it there, or leave it in the store, pinned; ``pins``
        are the versions its task pins each env at."""
        kind, entity_id = site.ref.kind, site.value
        if (bundle := self._installed_for(kind, entity_id)) is not None:
            self._from_installed(site, where, bundle)
            return
        entity = self._entity(kind, entity_id, self._version_for(site, where, pins))
        if isinstance(entity, str):
            self.problems.append(f"{where}: {entity}")
            return
        if isinstance(site.key, str) and (why := _not_taken(site, entity)) is not None:
            # The store holds the step as it is, though the step can't read what it names; a folder can't.
            site.rewrite(None)
            _unpin(site)
            self.dropped.append(Dropped(where, why))
            return
        self._need_type(site, entity, where)
        try:
            local = self.local(kind, entity, where)
        except _Abandoned:
            return
        if local is not None:
            site.rewrite(local)
            _unpin(site)  # what the bundle holds has no fixed version
        elif site.version_key is not None:
            site.rewrite(entity_id, entity.version)
        else:
            self.floating.append(Floating(where, kind, entity_id, "the field takes no version, so a run reads the "
                                                                  "latest"))

    def _version_for(self, site: RefSite, where: str, pins: Mapping[str, set[int]]) -> int | None:
        """The version ``site`` reads: its own; for an env it names with no version, the one its task pins elsewhere,
        since such a step means the env an earlier step deployed, which a folder holds once."""
        if site.version is not None or site.ref.kind is not EntityKind.ENV or len(pins.get(_key(site)[1], ())) != 1:
            return site.version
        [version] = pins[_key(site)[1]]
        self.dropped.append(Dropped(where, f"pinned: it names no version, so it takes v{version}, the one the task pins "
                                           "elsewhere"))
        return version

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
            why += f"; the stored pin to v{site.version} is dropped, as such an entity has no fixed version"
        _unpin(site)
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


def _not_taken(site: RefSite, entity: Any) -> str | None:
    """Why the field at ``site`` can't take ``entity``: it's of another type than the field takes."""
    expected = site.ref.artifact_type or site.ref.env_type
    actual = canonical_type(entity.type) if site.ref.kind is EntityKind.ARTIFACT else entity.type
    if not expected or actual == canonical_type(expected):
        return None
    return f"dropped: {site.value!r} is {actual}, and this field takes {expected}, so the step can't read it"


def _unpin(site: RefSite) -> None:
    """Clear the version ``site`` pins, when it pins one."""
    if site.version is not None and site.version_key is not None:
        site.owner[site.version_key] = None


def _key(site: RefSite) -> tuple[EntityKind, str]:
    return site.ref.kind, _nfc(site.value)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)
