"""Import a stored task, or an eval and its tasks, as a bundle folder that ``agent-env run`` runs.

Each task is written as ``task get`` prints it: the stored steps, read through the current step models. Every store
reference a step makes is pinned at the version the store holds now, so the folder runs what the store held when it
was imported; an id a step of the task writes stays unpinned, since the run writes it again. Values that look like
secrets are dropped. Nothing is written to a store, and the folder is written beside its destination, checked the way
``agent-env run`` checks a bundle before it reads a store, then moved into place, so a failed import leaves nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from agent_env.bundle import _toml
from agent_env.bundle._fs import fold, os_reason, show
from agent_env.bundle.parse import BundleError, parse_bundle, root_problems, segment_problem
from agent_env.bundle.plan import STORE_GETTERS, check_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import get_config
from agent_env.config.describe import is_secret_key
from agent_env.entity_refs import EntityKind, RefRole, ref_sites
from agent_env.eval import Eval
from agent_env.store.base import NotFoundError
from agent_env.store.ids import LOCAL_PREFIX
from agent_env.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

# The step fields that map environment variable names to values, where a credential is most often set.
_ENV_VAR_MAPS = ("env_vars", "default_env_vars")
# How long a name taken from an id may be, in bytes; ids are capped far above it, and a longer name helps nobody.
_MAX_NAME_BYTES = 200
# A URL's user info when it holds a password, or is long enough to be a token rather than a user name such as git.
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?:[^/@\s]*:[^/@\s]*|[^/@\s:]{16,})@")


@dataclass(frozen=True)
class Pinned:
    """A store reference written with a version: the one the store held at import, or the task's own pin."""

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
class BundleImport:
    """What an import wrote, and how it treated each reference and secret."""

    root: Path
    files: tuple[str, ...]
    tasks: tuple[ImportedTask, ...]
    eval: tuple[str, str, int] | None  # (path, id, version) of the eval, when one was imported
    pinned: tuple[Pinned, ...]
    floating: tuple[Floating, ...]
    dropped: tuple[Dropped, ...]
    undeclared: tuple[str, ...]  # the steps whose type declares no references, so any they make float
    duplicates: tuple[str, ...]  # the eval's entries that named a task it already named

    def sections(self) -> list[tuple[str, list[str]]]:
        """The import's report, as titled lists of lines; a section with nothing to say is left out."""
        sections = [
            ("Tasks", [f"{task.path}: task {task.id} v{task.version}" for task in self.tasks]),
            ("Eval", [f"{self.eval[0]}: eval {self.eval[1]} v{self.eval[2]}"] if self.eval else []),
            ("Pinned", [f"{pin.kind.value} {pin.id} v{pin.version}{' (as stored)' if pin.stored else ''}: "
                        f"{pin.where}" for pin in self.pinned]),
            ("Floating", [f"{ref.kind.value} {ref.id}: {ref.where}: {ref.why}" for ref in self.floating]
             + [f"{step}: its type declares no references, so any it makes float" for step in self.undeclared]),
            ("Left out", [f"{drop.where}: {drop.why}" for drop in self.dropped]
             + [f"{entry}, which the eval names more than once" for entry in self.duplicates]),
        ]
        return [(title, lines) for title, lines in sections if lines]


def import_task(task_id: str, dest: Path | str, *, version: int | None = None) -> BundleImport:
    """Write the stored task ``task_id``, at ``version`` or its latest, as a bundle at ``dest``, a folder that is
    empty or isn't there yet in one that is. Raises BundleError listing every problem, writing nothing."""
    importer = _Importer(Path(dest))
    task = Task.get(task_id, version)
    importer.add_task(task)
    return importer.write(f"Imported from the store's task {task.id} v{task.version}.")


def import_eval(eval_id: str, dest: Path | str, *, version: int | None = None) -> BundleImport:
    """Write the stored eval ``eval_id``, at ``version`` or its latest, and every task it names as a bundle at
    ``dest``, as ``import_task`` writes a task. Raises BundleError listing every problem, writing nothing."""
    importer = _Importer(Path(dest))
    stored = Eval.get(eval_id, version)
    names, seen = [], {}
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
        seen[task.id, task.version] = importer.add_task(task)
        names.append(seen[task.id, task.version])
    if names:
        importer.add_eval(stored, names)
    return importer.write(f"Imported from the store's eval {stored.id} v{stored.version} and the tasks it names.")


class _Importer:
    def __init__(self, dest: Path):
        self.dest = Path(os.path.abspath(dest.expanduser()))  # so ``.`` has a name, and a sibling to stage in
        self.files: dict[str, str] = {}
        self.taken: dict[str, dict[str, str]] = {"tasks": {}, "evals": {}}
        self.tasks: list[ImportedTask] = []
        self.eval: tuple[str, str, int] | None = None
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
        found = [(step, cls, list(ref_sites(cls.entity_refs or (), step)))
                 for step in steps for cls in (registry[step["type"]],)]
        sites = [site for _, _, step_sites in found for site in step_sites
                 if isinstance(site.value, str) and site.value]
        outputs = self._outputs(sites)
        deployed = {_nfc(site.value) for site in sites if site.ref.kind is EntityKind.ENV
                    and site.ref.role is RefRole.INPUT and site.version_key is not None}
        for step, cls, step_sites in found:
            where = f"{path}: step {step.get('id')!r}"
            self._drop_secrets(step, where)
            if cls.entity_refs is None:
                self.undeclared.append(f"{where} ({cls.type})")
            for site in step_sites:
                if site.ref.role is RefRole.OUTPUT or not (isinstance(site.value, str) and site.value):
                    continue
                output = outputs.get((site.ref.kind, _nfc(site.value)))
                if output is not None:  # written by a step of this task, which the bundle names under the task
                    if output != site.value:
                        site.rewrite(output)
                    continue
                self._pin(site, f"{where}: {site.path}", deployed)
            self._default_agents(step, where)
        self.files[path] = json.dumps(steps, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.tasks.append(ImportedTask(path, task.id, task.version, len(steps)))
        return name

    def add_eval(self, stored: Eval, names: list[str]) -> None:
        path = f"evals/{self._name('evals', stored.id)}.toml"
        self.files[path] = _toml.dumps({"tasks": names})
        self.eval = (path, stored.id, stored.version)

    def write(self, description: str) -> BundleImport:
        """Write the folder, with ``description`` as its README's first line, and say what was written."""
        if self.problems:
            raise BundleError(self.problems)
        files = sorted([*self.files, "README.md"])
        imported = BundleImport(self.dest, tuple(files), tuple(self.tasks), self.eval, tuple(self.pinned),
                                tuple(self.floating), tuple(self.dropped), tuple(self.undeclared),
                                tuple(self.duplicates))
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
        """Move ``staging`` to the destination, which is put back if it was an empty folder and the move fails or is
        interrupted."""
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

    @staticmethod
    def _outputs(sites: list[Any]) -> dict[tuple[EntityKind, str], str]:
        """The name each id a step of the task writes keeps. One a bundle named under its own task when it wrote the
        task takes its last segment, so this bundle names it under its task rather than writing more versions of the
        other's, numbered when another output or a store reference of its kind has it; the step that writes it is
        rewritten to the name."""
        written = {(site.ref.kind, _nfc(site.value)) for site in sites if site.ref.role is RefRole.OUTPUT}
        taken = {(site.ref.kind, _nfc(site.value)) for site in sites
                 if (site.ref.kind, _nfc(site.value)) not in written or not site.value.startswith(LOCAL_PREFIX)}
        outputs: dict[tuple[EntityKind, str], str] = {}
        for site in sites:
            key = (site.ref.kind, _nfc(site.value))
            if site.ref.role is not RefRole.OUTPUT or key in outputs:
                continue
            name = site.value
            if name.startswith(LOCAL_PREFIX):
                base = name = name.rsplit("/", 1)[-1]
                n = 1
                while (site.ref.kind, _nfc(name)) in taken:
                    n += 1
                    name = f"{base}-{n}"
                taken.add((site.ref.kind, _nfc(name)))
            outputs[key] = name
        for site in sites:
            if site.ref.role is RefRole.OUTPUT and outputs[site.ref.kind, _nfc(site.value)] != site.value:
                site.rewrite(outputs[site.ref.kind, _nfc(site.value)])
        return outputs

    def _pin(self, site: Any, where: str, deployed: set[str]) -> None:
        kind, entity_id = site.ref.kind, site.value
        if site.version is not None:
            read = self._read(kind, entity_id, site.version)
            if isinstance(read, int):
                self.pinned.append(Pinned(where, kind, entity_id, site.version, stored=True))
            else:
                self.problems.append(f"{where}: {read}")
        elif site.version_key is not None:
            read = self._read(kind, entity_id)
            if isinstance(read, int):
                site.rewrite(entity_id, read)
                self.pinned.append(Pinned(where, kind, entity_id, read, stored=False))
            else:
                self.problems.append(f"{where}: {read}")
        elif kind is not EntityKind.ENV:
            self.floating.append(Floating(where, kind, entity_id, "the field takes no version, so a run reads the "
                                                                  "latest"))
        elif _nfc(entity_id) not in deployed:  # an env field with no version names an env a step deploys, if one does
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

    def _drop_secrets(self, value: Any, where: str, path: str = "") -> None:
        """Drop each variable of an env-var map whose name looks like a secret or whose value holds a URL's
        credentials, and mask the credentials of a URL anywhere else, in place."""
        items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
        for key, item in list(items):
            at = f"{path}.{key}" if path and isinstance(key, str) else f"{path}[{key}]" if path else str(key)
            if isinstance(value, dict) and key in _ENV_VAR_MAPS and isinstance(item, dict):
                for name, setting in list(item.items()):
                    secret = is_secret_key(name)
                    if secret or (isinstance(setting, str) and _URL_CREDENTIALS.search(setting)):
                        del item[name]
                        why = "its name looks like a secret" if secret else "its value holds a URL's credentials"
                        self.dropped.append(Dropped(f"{where}: {at}.{name}", f"dropped: {why}"))
            elif isinstance(item, str) and _URL_CREDENTIALS.search(item):
                value[key] = _URL_CREDENTIALS.sub(r"\g<scheme>***@", item)
                self.dropped.append(Dropped(f"{where}: {at}", "masked: a URL's credentials"))
            else:
                self._drop_secrets(item, where, at)

    def _name(self, kind: str, entity_id: str) -> str:
        """A name for ``entity_id`` in the bundle's ``kind`` folder: its last segment, made valid, and numbered when
        another entry of the kind took it, as names compare caselessly."""
        base = _valid_name(entity_id.rsplit("/", 1)[-1]) or kind.removesuffix("s")
        name, n = base, 1
        while fold(name) in self.taken[kind]:
            n += 1
            name = f"{base}-{n}"
        self.taken[kind][fold(name)] = name
        return name


def _readme(description: str, imported: BundleImport) -> str:
    """The bundle's README: what it was imported from, its first line, then the import's report."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [description, "", f"Imported with agent-env {package_version('agentenv-framework')} on {stamp}. Each store "
             "reference is pinned at the version the store held then, unless it floats, so `agent-env run` runs this "
             "folder as the store ran it.", ""]
    for title, items in imported.sections():
        lines += [f"## {title}", "", *(f"- {item}" for item in items), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


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
    """``text`` as one bundle name: each character an id can't hold becomes ``-``, and dots that would hide it or
    spaces at its edges go. Empty when no letter or number is left."""
    text = unicodedata.normalize("NFC", text)
    name = "".join(char if char.isalnum() or unicodedata.category(char).startswith("M") or char in " ._-~@()+,&"
                   else "-" for char in text).strip(" ").lstrip(".")
    while len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        name = name[:-1].rstrip(" ")
    return name if any(char.isalnum() for char in name) and segment_problem(name) is None else ""


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)
