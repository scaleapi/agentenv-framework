"""Write a stored task, or an eval and its tasks, as a bundle folder that runs what the store held, writing no store."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

from agent_env.bundle import _toml
from agent_env.bundle._fs import fold, os_reason, show
from agent_env.bundle._secrets import scrub
from agent_env.bundle.parse import BundleError, parse_bundle, root_problems, segment_problem
from agent_env.bundle.plan import STORE_GETTERS, check_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import get_config
from agent_env.entity_refs import EntityKind, RefRole, RefSite, ref_sites
from agent_env.eval import Eval
from agent_env.store.base import NotFoundError
from agent_env.store.ids import LOCAL_PREFIX
from agent_env.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

# How long a name taken from an id may be, in bytes; ids are capped far above it, and a longer name helps nobody.
_MAX_NAME_BYTES = 200


@dataclass(frozen=True)
class Pinned:
    """A store reference written with a version: the store's at import, or the task's own pin."""

    where: str
    kind: EntityKind
    id: str
    version: int
    stored: bool  # the task already pinned it


@dataclass(frozen=True)
class Floating:
    """A store reference written without a version, so a run reads whatever version is current then."""

    where: str
    kind: EntityKind
    id: str
    why: str


@dataclass(frozen=True)
class Dropped:
    """A value left out of the folder, or masked in it, because it looks like a secret."""

    where: str
    why: str


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
    files: tuple[str, ...]
    tasks: tuple[ImportedTask, ...]
    eval: ImportedEval | None
    pinned: tuple[Pinned, ...]
    floating: tuple[Floating, ...]
    dropped: tuple[Dropped, ...]
    undeclared: tuple[str, ...]  # the steps whose type declares no references, so any they make float
    duplicates: tuple[str, ...]  # the eval's entries that named a task it already named

    def report(self) -> list[tuple[str, list[str]]]:
        """How each reference and secret was written, as titled lists of lines; an empty section is left out."""
        return _sections([
            ("Pinned", [f"{pin.kind.value} {pin.id} v{pin.version}{' (as stored)' if pin.stored else ''}: "
                        f"{pin.where}" for pin in self.pinned]),
            ("Floating", [f"{ref.kind.value} {ref.id}: {ref.where}: {ref.why}" for ref in self.floating]
             + [f"{step}: its type declares no references, so any it makes float" for step in self.undeclared]),
            ("Left out", [f"{drop.where}: {drop.why}" for drop in self.dropped]
             + [f"{entry}, which the eval names more than once" for entry in self.duplicates]),
        ])


def import_task(task_id: str, dest: Path | str, *, version: int | None = None) -> BundleImport:
    """Write the stored task ``task_id``, at ``version`` or its latest, as a bundle at ``dest``."""
    importer = _Importer(Path(dest))
    task = Task.get(task_id, version)
    importer.add_task(task)
    return importer.write(f"Imported from the store's task {task.id} v{task.version}.")


def import_eval(eval_id: str, dest: Path | str, *, version: int | None = None) -> BundleImport:
    """Write the stored eval ``eval_id``, at ``version`` or its latest, and the tasks it names as a bundle."""
    importer = _Importer(Path(dest))
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
    return importer.write(f"Imported from the store's eval {stored.id} v{stored.version} and the tasks it names.")


class _Importer:
    def __init__(self, dest: Path):
        self.dest = Path(os.path.abspath(dest.expanduser()))  # so ``.`` has a name, and a sibling to stage in
        self.files: dict[str, str] = {}
        self.taken: dict[str, set[str]] = {"tasks": set(), "evals": set()}  # folded names, per folder
        self.tasks: list[ImportedTask] = []
        self.eval: ImportedEval | None = None
        self.pinned: list[Pinned] = []
        self.floating: list[Floating] = []
        self.dropped: list[Dropped] = []
        self.undeclared: list[str] = []
        self.duplicates: list[str] = []
        self.problems: list[str] = []
        self.read: dict[tuple[EntityKind, str, int | None], int | str] = {}
        _check_destination(self.dest)

    def add_task(self, task: Task) -> str:
        """Write ``task`` under a name taken from its id, and return the name."""
        name = self._name("tasks", task.id)
        path = f"tasks/{name}.json"
        steps = task.to_dict()["steps"]
        registry = get_task_step_registry()
        found = []
        for step in steps:
            cls = registry[step["type"]]
            found.append((step, cls, [site for site in ref_sites(cls.entity_refs or (), step)
                                      if isinstance(site.value, str) and site.value]))
        sites = [site for _, _, step_sites in found for site in step_sites]
        outputs = _output_names(sites)
        deployed = {_nfc(site.value) for _, cls, step_sites in found if issubclass(cls, DeployEnvTaskStep)
                    for site in step_sites if site.ref.kind is EntityKind.ENV}
        for site in sites:  # each output, and each step that reads it, takes the output's name
            if outputs.get(_key(site), site.value) != site.value:
                site.rewrite(outputs[_key(site)])
        for step, cls, step_sites in found:
            where = f"{path}: step {step.get('id')!r}"
            self.dropped += [Dropped(f"{where}: {at}", why) for at, why in scrub(step)]
            if cls.entity_refs is None:
                self.undeclared.append(f"{where} ({cls.type})")
            for site in step_sites:
                if _key(site) not in outputs:  # the run writes an output again, so it takes no pin
                    self._pin(site, f"{where}: {site.path}", deployed)
            self._default_agents(step, where)
        self.files[path] = json.dumps(steps, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.tasks.append(ImportedTask(path, task.id, task.version, len(steps)))
        return name

    def add_eval(self, stored: Eval, names: list[str]) -> None:
        path = f"evals/{self._name('evals', stored.id)}.toml"
        self.files[path] = _toml.dumps({"tasks": names})
        self.eval = ImportedEval(path, stored.id, stored.version)

    def write(self, description: str) -> BundleImport:
        """Write the folder, with ``description`` as its README's first line, and say what was written."""
        if self.problems:
            raise BundleError(self.problems)
        files = sorted([*self.files, "README.md"])
        imported = BundleImport(root=self.dest, files=tuple(files), tasks=tuple(self.tasks), eval=self.eval,
                                pinned=tuple(self.pinned), floating=tuple(self.floating),
                                dropped=tuple(self.dropped), undeclared=tuple(self.undeclared),
                                duplicates=tuple(self.duplicates))
        self.files["README.md"] = _readme(description, imported)
        staging = self.dest.parent / f".{self.dest.name}.import-{uuid.uuid4().hex[:8]}"
        try:
            try:
                for path, text in self.files.items():
                    (staging / path).parent.mkdir(parents=True, exist_ok=True)
                    (staging / path).write_text(text, encoding="utf-8")
            except OSError as e:
                raise BundleError([f"{show(self.dest.parent)}: can't write the import there: {os_reason(e)}"]) from None
            check_bundle(resolve_bundle(parse_bundle(staging)))
            self._move(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return imported

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

    def _pin(self, site: RefSite, where: str, deployed: set[str]) -> None:
        kind, entity_id = site.ref.kind, site.value
        if site.version is not None or site.version_key is not None:
            read = self._read(kind, entity_id, site.version)
            if isinstance(read, str):
                self.problems.append(f"{where}: {read}")
                return
            if site.version is None:
                site.rewrite(entity_id, read)
            self.pinned.append(Pinned(where, kind, entity_id, read, stored=site.version is not None))
        elif kind is not EntityKind.ENV:
            self.floating.append(Floating(where, kind, entity_id, "the field takes no version, so a run reads the "
                                                                  "latest"))
        elif _nfc(entity_id) not in deployed:
            self.floating.append(Floating(where, kind, entity_id, "the field takes no version and no step of the task "
                                                                  "deploys it, so a run reads the latest"))

    def _read(self, kind: EntityKind, entity_id: str, version: int | None = None) -> int | str:
        """The version of ``entity_id`` the store holds, at ``version`` or its latest, or why it can't be read."""
        if (kind, entity_id, version) not in self.read:
            try:
                self.read[kind, entity_id, version] = STORE_GETTERS[kind](entity_id, version).version
            except NotFoundError:
                at = f" v{version}" if version is not None else ""
                self.read[kind, entity_id, version] = f"there is no {kind.value} {entity_id!r}{at} in the store"
            except (ValueError, KeyError, TypeError) as e:
                self.read[kind, entity_id, version] = (f"{kind.value} {entity_id!r} can't be read "
                                                       f"({type(e).__name__}: {e})")
        return self.read[kind, entity_id, version]

    def _default_agents(self, step: dict, where: str) -> None:
        """A step that names no agent deploys the configured default, which no field pins."""
        if step["type"] == DeployAgentTaskStep.type and step.get("a2a_agent_id") is None:
            field, what = "a2a_agent_id", "names no agent"
        elif (step["type"] == RubricsVerifierTaskStep.type and step.get("use_agent_judge")
              and step.get("agent_name") is None and step.get("judge_a2a_agent_id") is None):
            field, what = "judge_a2a_agent_id", "names no judge agent"
        else:
            return
        self.floating.append(Floating(f"{where}: {field}", EntityKind.AGENT, get_config().get_default_a2a_agent_id(),
                                      f"the step {what}, so a run deploys the configured default"))

    def _name(self, kind: str, entity_id: str) -> str:
        """A name in the ``kind`` folder: the id's last segment made valid, numbered when taken, caselessly."""
        base = _valid_name(entity_id.rsplit("/", 1)[-1]) or kind.removesuffix("s")
        name, n = base, 1
        while fold(name) in self.taken[kind]:
            n += 1
            name = f"{base}-{n}"
        self.taken[kind].add(fold(name))
        return name


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


def _readme(description: str, imported: BundleImport) -> str:
    """The bundle's README: its description, where it came from, what it holds, then the import's report."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [description, "", f"Imported with agent-env {package_version('agentenv-framework')} on {stamp}. Each store "
             "reference is pinned at the version the store held then, unless it floats, so `agent-env run` runs this "
             "folder as the store ran it.", ""]
    held = _sections([("Tasks", [f"{task.path}: task {task.id} v{task.version}" for task in imported.tasks]),
                      ("Eval", [f"{imported.eval.path}: eval {imported.eval.id} v{imported.eval.version}"]
                       if imported.eval else [])])
    for title, items in [*held, *imported.report()]:
        lines += [f"## {title}", "", *(f"- {item}" for item in items), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


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
