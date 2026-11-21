"""Reading ATIF documents, with or without ``extra.agentenv``: load and check one with the vendored models, slice it
into its A2A tasks, list the media it names, strip the profile, and tell whether a live page is a prefix of a later
read. Every helper accepts any valid ATIF document, such as a Harbor task's trajectory.json."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pydantic import ValidationError

from .models import Trajectory
from .profile import PROFILE_KEY, AtifError, profile_of


def _no_constant(name: str) -> Any:
    raise AtifError(f"{name} isn't JSON")


def load(source: str | bytes | Path | Mapping[str, Any]) -> dict[str, Any]:
    """A document from JSON text, a file or a parsed mapping, checked with the vendored ATIF models. JSON is read
    strictly: UTF-8, and no NaN or infinity."""
    if isinstance(source, Path):
        source = source.read_bytes()
    if isinstance(source, (bytes, bytearray)):
        try:
            source = bytes(source).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AtifError(f"a document is UTF-8: {exc}") from exc
    if isinstance(source, str):
        try:
            source = json.loads(source, parse_constant=_no_constant)
        except json.JSONDecodeError as exc:
            raise AtifError(f"a document isn't JSON: {exc}") from exc
    if not isinstance(source, Mapping):
        raise AtifError("a document is a JSON object")
    document = copy.deepcopy(dict(source))
    check(document)
    return document


def check(document: Mapping[str, Any]) -> Trajectory:
    """The document as ATIF's own models read it, or AtifError naming what they refused."""
    try:
        return Trajectory.model_validate(document)
    except ValidationError as exc:
        raise AtifError(f"not valid ATIF: {exc}") from exc


@dataclass(frozen=True)
class Task:
    """One A2A task's steps: from its user step up to the next task's. The ids and turn come from the user step's
    profile; without one, ``turn`` counts the tasks in order and the ids are None."""

    turn: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    task_id: str | None = None
    context_id: str | None = None
    message_id: str | None = None

    @property
    def state(self) -> str | None:
        """The task's terminal state, which the profile puts on its last step."""
        return profile_of(self.steps[-1]).get("state") if self.steps else None


def tasks(document: Mapping[str, Any]) -> list[Task]:
    """The root trajectory's steps sliced into A2A tasks. A task starts at a user step: one carrying a task id in its
    profile, or, in a document without the profile, any ``source: user`` step."""
    steps = list(document.get("steps") or [])
    profiled = any(profile_of(s).get("task_id") for s in steps if s.get("source") == "user")
    out: list[Task] = []
    for step in steps:
        profile = profile_of(step)
        starts = step.get("source") == "user" and (profile.get("task_id") is not None or not profiled)
        if starts or not out:
            turn = profile.get("turn") if isinstance(profile.get("turn"), int) else len(out)
            out.append(Task(turn=turn, task_id=profile.get("task_id"), context_id=profile.get("context_id"),
                            message_id=profile.get("message_id")))
        out[-1].steps.append(step)
    return out


def task_document(document: Mapping[str, Any], task_ids: Iterable[str]) -> dict[str, Any]:
    """The A2A tasks ``task_ids`` names, cut out of ``document`` as a document of their own that ATIF's models load: their
    steps in order, renumbered from 1 with each result's arrival step to match, the embedded sub-agents those steps reach
    (through one another too) kept whole, and the root's ``final_metrics`` and ``continued_trajectory_ref`` dropped, since
    they describe the whole document. A reader that shows one turn shows this. KeyError naming a task it doesn't hold."""
    wanted = list(task_ids)
    picked = [task for task in tasks(document) if task.task_id in wanted]
    missing = [task_id for task_id in wanted if task_id not in {task.task_id for task in picked}]
    if missing:
        raise KeyError(f"the document holds no task {', '.join(map(repr, missing))}")
    return tasks_document(document, picked)


def tasks_document(document: Mapping[str, Any], picked: Iterable[Task]) -> dict[str, Any]:
    """``task_document`` for tasks already found among ``tasks(document)``, such as the tasks of a document without the
    profile, which carry no ids."""
    steps = [copy.deepcopy(step) for task in picked for step in task.steps]
    numbers = {step.get("step_id"): n for n, step in enumerate(steps, 1)}
    for n, step in enumerate(steps, 1):
        step["step_id"] = n
        for result in (step.get("observation") or {}).get("results") or []:
            arrived = ((result.get("extra") or {}).get(PROFILE_KEY) or {}).get("arrived")
            if isinstance(arrived, dict) and "step_id" in arrived:
                if arrived["step_id"] in numbers:
                    arrived["step_id"] = numbers[arrived["step_id"]]
                else:
                    del arrived["step_id"]
    subs = {sub.get("trajectory_id"): sub for sub in document.get("subagent_trajectories") or []}
    reached: set[Any] = set()
    frontier = _references(steps)
    while frontier:
        trajectory_id = frontier.pop()
        if trajectory_id in subs and trajectory_id not in reached:
            reached.add(trajectory_id)
            frontier += _references(subs[trajectory_id].get("steps") or [])
    cut = {key: copy.deepcopy(value) for key, value in document.items()
           if key not in ("steps", "subagent_trajectories", "final_metrics", "continued_trajectory_ref")}
    cut["steps"] = steps
    kept = [copy.deepcopy(sub) for sub in document.get("subagent_trajectories") or []
            if sub.get("trajectory_id") in reached]
    if kept:
        cut["subagent_trajectories"] = kept
    return cut


def _references(steps: Iterable[Mapping[str, Any]]) -> list[Any]:
    return [ref.get("trajectory_id") for step in steps for result in (step.get("observation") or {}).get("results") or []
            for ref in result.get("subagent_trajectory_ref") or []]


def _trajectories(document: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Mapping[str, Any]]]:
    """The document and every sub-agent embedded in it, at any depth, with the path prefix of each."""
    yield prefix, document
    for index, sub in enumerate(document.get("subagent_trajectories") or []):
        yield from _trajectories(sub, f"{prefix}subagent_trajectories[{index}].")


#: A media file's name, ``<sha256>.<ext>``: what its bytes are, never a secret.
MEDIA_NAME = re.compile(r"(?P<sha256>[0-9a-f]{64})\.[A-Za-z0-9]+")
#: The only path a media file of a document can have: ``media/<sha256>.<ext>``, in the document's own folder.
MEDIA_PATH = re.compile(rf"^media/{MEDIA_NAME.pattern}$")


def media_file(path: str) -> str:
    """``path``'s sha256, once ``path`` is ``media/<sha256>.<ext>``: a part path resolves only to a file in the
    document's own media folder, never to another path or a URL. AtifError otherwise."""
    named = MEDIA_PATH.match(path) if isinstance(path, str) else None
    if named is None:
        raise AtifError(f"{path!r} isn't media/<sha256>.<ext> in the document's folder")
    return named["sha256"]


def check_media(path: str, data: bytes) -> None:
    """Refuse media whose bytes aren't what its name says: a reader checks each file it reads, since a store can hold
    another file under that name. AtifError naming the difference."""
    digest = hashlib.sha256(data).hexdigest()
    if media_file(path) != digest:
        raise AtifError(f"{path}'s bytes have sha256 {digest}")


@dataclass(frozen=True)
class MediaRef:
    """One place a document names a media file: its ``path``, ``where`` it's named (such as
    ``steps[2].message[1].source.path``), the profile's details for it ({} without) and the media type of the part
    that names it, else the details' own."""

    path: str
    where: str
    info: Mapping[str, Any]
    media_type: str | None


def media_refs(document: Mapping[str, Any]) -> list[MediaRef]:
    """Every place the document and the sub-agents embedded in it name a media file, in order: in each step's message
    and each result's content, the image and audio parts' paths, then the files the profile records beside them for
    media ATIF has no part type for."""
    refs: list[MediaRef] = []
    for prefix, trajectory in _trajectories(document):
        for index, step in enumerate(trajectory.get("steps") or []):
            where = f"{prefix}steps[{index}]"
            holders = [(where, step, step.get("message"), "message")] + [
                (f"{where}.observation.results[{n}]", result, result.get("content"), "content")
                for n, result in enumerate((step.get("observation") or {}).get("results") or [])]
            for owner, holder, content, field_name in holders:
                details = profile_of(holder).get("media") or {}
                details = details if isinstance(details, Mapping) else {}
                named: set[str] = set()
                for n, part in enumerate(content if isinstance(content, list) else []):
                    source = part.get("source") if isinstance(part, Mapping) else None
                    if isinstance(source, Mapping) and isinstance(source.get("path"), str):
                        path, info = source["path"], _details(details, source["path"])
                        named.add(path)
                        refs.append(MediaRef(path, f"{owner}.{field_name}[{n}].source.path", info,
                                             source.get("media_type") or info.get("media_type")))
                refs += [MediaRef(path, f"{owner}.extra.{PROFILE_KEY}.media[{path}]", _details(details, path),
                                  _details(details, path).get("media_type"))
                         for path in details if path not in named]
    return refs


def _details(details: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    info = details.get(path)
    return info if isinstance(info, Mapping) else {}


def media_paths(document: Mapping[str, Any]) -> list[str]:
    """Every media file the document names, once each, in the order they appear: image and audio parts' paths, and
    the files the profile records for media ATIF has no part type for."""
    return list(dict.fromkeys(ref.path for ref in media_refs(document)))


def strip_profile(document: Mapping[str, Any]) -> dict[str, Any]:
    """A copy without the profile: ``extra.agentenv`` taken off each ATIF object that has an ``extra``, and an
    ``extra`` left empty with it. Tool arguments and every other free-form value stay as they are."""
    stripped = copy.deepcopy(dict(document))
    _strip_trajectory(stripped)
    return stripped


def _strip_trajectory(trajectory: dict[str, Any]) -> None:
    for owner in (trajectory, trajectory.get("agent"), trajectory.get("final_metrics")):
        _strip(owner)
    for step in trajectory.get("steps") or []:
        _strip(step)
        _strip(step.get("metrics"))
        for call in step.get("tool_calls") or []:
            _strip(call)
        for result in (step.get("observation") or {}).get("results") or []:
            _strip(result)
            for ref in result.get("subagent_trajectory_ref") or []:
                _strip(ref)
    for sub in trajectory.get("subagent_trajectories") or []:
        _strip_trajectory(sub)


def _strip(owner: Any) -> None:
    extra = owner.get("extra") if isinstance(owner, dict) else None
    if isinstance(extra, dict) and PROFILE_KEY in extra:
        del extra[PROFILE_KEY]
        if not extra:
            del owner["extra"]


def is_prefix(page: Mapping[str, Any] | None, later: Mapping[str, Any] | None) -> bool:
    """Whether a live page's steps are the first steps of a later read, trajectory by trajectory. Nothing served is a
    prefix of anything."""
    if page is None:
        return True
    if later is None:
        return False
    if list(later.get("steps") or [])[:len(page.get("steps") or [])] != list(page.get("steps") or []):
        return False
    subs = {sub.get("trajectory_id"): sub for sub in later.get("subagent_trajectories") or []}
    for sub in page.get("subagent_trajectories") or []:
        if not is_prefix(sub, subs.get(sub.get("trajectory_id"))):
            return False
    return True
