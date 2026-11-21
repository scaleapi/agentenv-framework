"""Checking an ATIF document against ATIF and the agentenv profile: the vendored models, the profile's models, and the
rules neither checks, each a finding with a rule id PROFILE.md names. A producer is held to profile 1.0; a reader of
someone else's document is warned where a producer would fail, and fails only where agentenv couldn't read it."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pydantic import BaseModel, TypeAdapter, ValidationError

from .models import Trajectory
from .profile import (
    PROFILE_KEY,
    PROFILE_VERSION,
    PROFILES,
    RAW_ARGUMENTS,
    VOCABULARIES,
    AdapterRef,
    Arrival,
    MediaInfo,
    ReasoningBlock,
    StepError,
    profile_of,
)
from .reader import MEDIA_PATH, _trajectories, is_prefix, media_refs, tasks

ERROR = "error"
WARNING = "warning"

TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3}|\.\d{6})?Z$")
_COUNTS = (("prompt_tokens", "total_prompt_tokens"), ("completion_tokens", "total_completion_tokens"),
           ("cached_tokens", "total_cached_tokens"))
_INT = TypeAdapter(int)
#: A profile field that holds a nested profile object, by its owner and name.
_NESTED: dict[tuple[str, str], type[BaseModel]] = {
    ("Trajectory", "adapter"): AdapterRef, ("Step", "error"): StepError,
    ("ObservationResult", "arrived"): Arrival,
}
_LISTS: dict[tuple[str, str], type[BaseModel]] = {("Step", "reasoning"): ReasoningBlock}
_MAPS: dict[tuple[str, str], type[BaseModel]] = {
    ("Step", "media"): MediaInfo, ("ObservationResult", "media"): MediaInfo,
}
_VOCABULARY_AT = {
    ("Step", "state"): "StepProfile.state", ("ObservationResult", "status"): "ObservationResultProfile.status",
    ("Metrics", "cost_source"): "MetricsProfile.cost_source",
    ("FinalMetrics", "cost_source"): "FinalMetricsProfile.cost_source",
}


@dataclass(frozen=True)
class Rule:
    """A check: its id, its severity for a producer and for a reader (None: not checked there), and what it holds."""

    id: str
    producer: str
    reader: str | None
    holds: str


RULES: dict[str, Rule] = {rule.id: rule for rule in (
    Rule("json", ERROR, ERROR, "The document is a JSON object: UTF-8, with no NaN or infinity"),
    Rule("atif", ERROR, ERROR, "ATIF's own models, vendored from Harbor, accept it"),
    Rule("profile", ERROR, ERROR, "Each `extra.agentenv` parses with its profile model"),
    Rule("profile-key", ERROR, WARNING, "Each `extra.agentenv` holds only the fields profile 1.0 defines"),
    Rule("profile-version", ERROR, WARNING, "The root's `extra.agentenv.version`, when given, is `1.0`"),
    Rule("vocabulary", ERROR, WARNING, "A vocabulary field holds one of its values"),
    Rule("outside-key", ERROR, WARNING, "No `extra` holds a key outside `agentenv` but `ToolCall.extra.raw_arguments`"),
    Rule("call-id", ERROR, WARNING, "Each `tool_call_id` is unique across the document and its sub-agents"),
    Rule("reference", ERROR, WARNING, "Each `subagent_trajectory_ref` names an embedded sub-agent by `trajectory_id`"),
    Rule("totals", ERROR, WARNING, "A `final_metrics` total equals the sum over the model steps of the root and its "
                                   "embedded sub-agents, when every one has the count"),
    Rule("total-steps", ERROR, WARNING, "`final_metrics.total_steps` counts the root's steps, or the root's `notes` "
                                        "explain the difference"),
    Rule("usage", ERROR, WARNING, "A step's cache reads and writes fit within its `prompt_tokens`"),
    Rule("media-path", ERROR, WARNING, "A media path is `media/<sha256>.<ext>`"),
    Rule("media-bytes", ERROR, ERROR, "Given the media, each file the document names exists, with the sha256 its "
                                      "name and profile give and the profile's size"),
    Rule("timestamp", ERROR, None, "A timestamp is UTC in the builder's form: `YYYY-MM-DDTHH:MM:SSZ`, with no, 3 or "
                                   "6 fractional digits"),
    Rule("harbor-save", ERROR, WARNING, "Harbor's `to_json_dict` gives the document back unchanged"),
    Rule("text-message", ERROR, WARNING, "A step message whose parts are all text is one string"),
    Rule("live-prefix", ERROR, None, "Every live page is a prefix of the next page and of the final read"),
    Rule("live-unanswered", ERROR, None, "No live page serves a step with an unanswered call before its task ends"),
)}


@dataclass(frozen=True)
class Finding:
    """What a rule found, and where: a path into the document, such as ``steps[3].tool_calls[0]``."""

    rule: str
    severity: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity} {self.rule} {self.where or 'document'}: {self.message}"


class _Findings:
    def __init__(self, reader: bool, prefix: str = "") -> None:
        self.reader = reader
        self.prefix = prefix
        self.items: list[Finding] = []

    def add(self, rule: str, where: str, message: str) -> None:
        severity = RULES[rule].reader if self.reader else RULES[rule].producer
        if severity is not None:
            self.items.append(Finding(rule, severity, f"{self.prefix}{where}", message))


def errors_in(findings: Sequence[Finding]) -> list[Finding]:
    """The findings that fail a check."""
    return [finding for finding in findings if finding.severity == ERROR]


def parse(source: str | bytes | Path) -> tuple[Any, list[Finding]]:
    """JSON text or a file read as the ``json`` rule reads it: the value, or None with the finding."""
    def no_constant(name: str) -> Any:
        raise ValueError(f"{name} isn't JSON")
    try:
        if isinstance(source, Path):
            source = source.read_bytes()
        text = source.decode("utf-8") if isinstance(source, (bytes, bytearray)) else source
        return json.loads(text, parse_constant=no_constant), []
    except (UnicodeDecodeError, ValueError) as exc:
        return None, [Finding("json", ERROR, "", str(exc))]


@dataclass(frozen=True)
class StoredMedia:
    """Media files stored beside a document by one who checked each file's bytes against its name, as the worker does
    each file an agent pushes: their sizes by path, None where the store can't tell. Given these, ``validate`` holds
    the profile's sha256 to each name and its size to the stored file's without reading the bytes."""

    sizes: Mapping[str, int | None]


def validate(document: Any, *, reader: bool = False,
             media: Path | Mapping[str, bytes] | StoredMedia | None = None) -> list[Finding]:
    """Every finding for ``document``. ``reader`` checks someone else's document for reading rather than a producer's
    output. ``media`` is the folder the document's media paths are relative to, the files by path, or the files stored
    beside it (``StoredMedia``); without it, only the paths' form is checked."""
    found = _Findings(reader)
    _check(document, found, media)
    return found.items


def _check(document: Any, found: _Findings, media: Path | Mapping[str, bytes] | StoredMedia | None) -> None:
    if not isinstance(document, Mapping):
        found.add("json", "", "a document is a JSON object")
        return
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        found.add("json", "", str(exc))
        return
    try:
        parsed = Trajectory.model_validate(document)
    except ValidationError as exc:
        for error in exc.errors(include_url=False):
            found.add("atif", _loc(error["loc"]), error["msg"])
        return
    objects = list(_objects(document))
    _profiles(objects, found)
    _outside_keys(objects, found)
    _calls_and_references(document, found)
    _totals(document, found)
    _usage(document, found)
    _media(document, found, media)
    _timestamps(document, found)
    saved = parsed.to_json_dict()
    if saved != document:
        found.add("harbor-save", _difference(document, saved), "Harbor's to_json_dict gives this back changed")
    for where, step in _steps(document):
        message = step.get("message")
        if isinstance(message, list) and message and all(
                isinstance(p, Mapping) and p.get("type") == "text" for p in message):
            found.add("text-message", f"{where}.message", "every part is text, so the message is one string")


def validate_live(pages: Sequence[Mapping[str, Any] | None], final: Mapping[str, Any], *,
                  media: Path | Mapping[str, bytes] | None = None) -> list[Finding]:
    """The live rule, for a producer: each page a reader was served, in order, validates, is a prefix of the next and
    of the final read, and serves no step whose call is unanswered before its task ends. The final read is checked
    too. A page equal to the one before it is checked once, since most records close no step."""
    findings: list[Finding] = []
    served: tuple[int, Mapping[str, Any]] | None = None
    for index, page in enumerate(pages):
        if served is not None and page == served[1]:
            continue
        if page is None:
            if served is not None:
                findings.append(Finding("live-prefix", ERROR, f"page {index}: ",
                                        f"serves nothing after page {served[0]} served steps"))
            continue
        found = _Findings(False, f"page {index}: ")
        _check(page, found, media)
        _unanswered(page, found)
        findings.extend(found.items)
        if served is not None and not is_prefix(served[1], page):
            findings.append(Finding("live-prefix", ERROR, f"page {index}: ",
                                    f"page {served[0]} isn't a prefix of it; they first differ at "
                                    f"{_difference(served[1], page) or 'the document'}"))
        served = (index, page)
    found = _Findings(False, "final: ")
    _check(final, found, media)
    findings.extend(found.items)
    if served is not None and not is_prefix(served[1], final):
        findings.append(Finding("live-prefix", ERROR, "final: ",
                                f"page {served[0]} isn't a prefix of it; they first differ at "
                                f"{_difference(served[1], final) or 'the document'}"))
    return findings


# What the document holds

def _count(value: Any) -> int | None:
    """A token count as ATIF's models read it, a numeric string included, or None when there is none."""
    try:
        return None if value is None else _INT.validate_python(value)
    except ValidationError:
        return None


def _steps(document: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for prefix, trajectory in _trajectories(document):
        for index, step in enumerate(trajectory.get("steps") or []):
            yield f"{prefix}steps[{index}]", step


def _results(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list((step.get("observation") or {}).get("results") or [])


def _objects(document: Mapping[str, Any]) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
    """Every ATIF object that may have an ``extra``: its kind, as PROFILES names it, its path and the object."""
    for prefix, trajectory in _trajectories(document):
        yield "Trajectory", prefix.rstrip(".") or "", trajectory
        yield "Agent", f"{prefix}agent", trajectory.get("agent") or {}
        if trajectory.get("final_metrics"):
            yield "FinalMetrics", f"{prefix}final_metrics", trajectory["final_metrics"]
        for index, step in enumerate(trajectory.get("steps") or []):
            where = f"{prefix}steps[{index}]"
            yield "Step", where, step
            if step.get("metrics"):
                yield "Metrics", f"{where}.metrics", step["metrics"]
            for n, call in enumerate(step.get("tool_calls") or []):
                yield "ToolCall", f"{where}.tool_calls[{n}]", call
            for n, result in enumerate(_results(step)):
                yield "ObservationResult", f"{where}.observation.results[{n}]", result
                for m, ref in enumerate(result.get("subagent_trajectory_ref") or []):
                    yield "SubagentTrajectoryRef", f"{where}.observation.results[{n}].subagent_trajectory_ref[{m}]", ref


def _profiles(objects: list[tuple[str, str, Mapping[str, Any]]], found: _Findings) -> None:
    for kind, where, obj in objects:
        extra = obj.get("extra")
        if not isinstance(extra, Mapping) or PROFILE_KEY not in extra:
            continue
        at = f"{where}.extra.{PROFILE_KEY}".lstrip(".")
        profile = extra[PROFILE_KEY]
        if not isinstance(profile, Mapping):
            found.add("profile", at, "the profile is an object")
            continue
        try:
            PROFILES[kind].model_validate(profile)
        except ValidationError as exc:
            for error in exc.errors(include_url=False):
                found.add("profile", f"{at}.{_loc(error['loc'])}", error["msg"])
            continue
        _known_keys(kind, at, profile, found)
        for (owner, name), vocabulary in _VOCABULARY_AT.items():
            if owner == kind and profile.get(name) is not None:
                _vocabulary(vocabulary, f"{at}.{name}", profile[name], found)
        error = profile.get("error") if kind == "Step" else None
        if isinstance(error, Mapping) and error.get("kind") is not None:
            _vocabulary("StepError.kind", f"{at}.error.kind", error["kind"], found)
        for n, block in enumerate(profile.get("reasoning") or [] if kind == "Step" else []):
            if isinstance(block, Mapping) and block.get("visibility") is not None:
                _vocabulary("ReasoningBlock.visibility", f"{at}.reasoning[{n}].visibility", block["visibility"], found)
        if kind == "Trajectory" and where == "" and profile.get("version") not in (None, PROFILE_VERSION):
            found.add("profile-version", f"{at}.version",
                      f"this is profile {profile['version']!r}; the validator checks profile {PROFILE_VERSION}")


def _known_keys(kind: str, at: str, profile: Mapping[str, Any], found: _Findings) -> None:
    def unknown(model: type[BaseModel], value: Mapping[str, Any], path: str) -> None:
        for key in value:
            if key not in model.model_fields:
                found.add("profile-key", f"{path}.{key}", f"profile {PROFILE_VERSION} defines no {key!r} here")
    unknown(PROFILES[kind], profile, at)
    for (owner, name), model in _NESTED.items():
        if owner == kind and isinstance(profile.get(name), Mapping):
            unknown(model, profile[name], f"{at}.{name}")
    for (owner, name), model in _LISTS.items():
        if owner == kind:
            for n, item in enumerate(profile.get(name) or []):
                if isinstance(item, Mapping):
                    unknown(model, item, f"{at}.{name}[{n}]")
    for (owner, name), model in _MAPS.items():
        if owner == kind and isinstance(profile.get(name), Mapping):
            for key, item in profile[name].items():
                if isinstance(item, Mapping):
                    unknown(model, item, f"{at}.{name}[{key}]")


def _vocabulary(name: str, where: str, value: Any, found: _Findings) -> None:
    if value not in VOCABULARIES[name]:
        found.add("vocabulary", where, f"{value!r} isn't one of {', '.join(VOCABULARIES[name])}")


def _outside_keys(objects: list[tuple[str, str, Mapping[str, Any]]], found: _Findings) -> None:
    for kind, where, obj in objects:
        extra = obj.get("extra")
        for key in extra if isinstance(extra, Mapping) else ():
            if key != PROFILE_KEY and not (kind == "ToolCall" and key == RAW_ARGUMENTS):
                found.add("outside-key", f"{where}.extra.{key}".lstrip("."),
                          f"the profile names no key outside extra.{PROFILE_KEY} here")


def _resolver(document: Mapping[str, Any], prefix: str = "",
              outer: Mapping[Any, str] | None = None) -> dict[str, dict[Any, str]]:
    """For each trajectory, by its path prefix, the sub-agent each id a reference in it names: one embedded in it,
    else the nearest one embedded around it, by path prefix. ATIF makes an id unique only within its parent's array,
    and the builder embeds every sub-agent in the root."""
    subs = list(document.get("subagent_trajectories") or [])
    visible = {**(outer or {}), **{sub.get("trajectory_id"): f"{prefix}subagent_trajectories[{index}]."
                                   for index, sub in enumerate(subs)}}
    out = {prefix: visible}
    for index, sub in enumerate(subs):
        out.update(_resolver(sub, f"{prefix}subagent_trajectories[{index}].", visible))
    return out


def _calls_and_references(document: Mapping[str, Any], found: _Findings) -> None:
    resolve = _resolver(document)
    seen: dict[str, str] = {}
    for prefix, trajectory in _trajectories(document):
        for index, step in enumerate(trajectory.get("steps") or []):
            _step_calls(f"{prefix}steps[{index}]", step, resolve[prefix], seen, found)


def _step_calls(where: str, step: Mapping[str, Any], embedded: Mapping[Any, str], seen: dict[str, str],
                found: _Findings) -> None:
    for n, call in enumerate(step.get("tool_calls") or []):
        call_id = call.get("tool_call_id")
        if call_id in seen:
            found.add("call-id", f"{where}.tool_calls[{n}]", f"{call_id!r} is also the id at {seen[call_id]}")
        else:
            seen[call_id] = f"{where}.tool_calls[{n}]"
    for n, result in enumerate(_results(step)):
        for m, ref in enumerate(result.get("subagent_trajectory_ref") or []):
            at = f"{where}.observation.results[{n}].subagent_trajectory_ref[{m}]"
            trajectory_id = ref.get("trajectory_id")
            if trajectory_id is None:
                if not found.reader:
                    found.add("reference", at, "names no embedded sub-agent: the profile embeds every sub-agent")
            elif trajectory_id not in embedded:
                found.add("reference", at, f"no embedded sub-agent has trajectory_id {trajectory_id!r}")


def _totals(document: Mapping[str, Any], found: _Findings) -> None:
    totals = document.get("final_metrics")
    if not isinstance(totals, Mapping):
        return
    model_steps = [step.get("metrics") or {} for _, trajectory in _trajectories(document)
                   for step in trajectory.get("steps") or []
                   if step.get("source") == "agent" and step.get("llm_call_count") != 0]
    sums = [(f"final_metrics.{total}", _count(totals.get(total)), [_count(m.get(count)) for m in model_steps])
            for count, total in _COUNTS]
    sums.append((f"final_metrics.extra.{PROFILE_KEY}.total_cache_write_tokens",
                 _count(profile_of(totals).get("total_cache_write_tokens")),
                 [_count(profile_of(m).get("cache_write_tokens")) for m in model_steps]))
    for where, total, counts in sums:
        if total is not None and None not in counts and total != sum(counts):
            found.add("totals", where, f"is {total}, but its model steps sum to {sum(counts)}")
    steps = len(document.get("steps") or [])
    total_steps = _count(totals.get("total_steps"))
    if total_steps is not None and total_steps != steps and not document.get("notes"):
        found.add("total-steps", "final_metrics.total_steps",
                  f"is {total_steps}, but the root has {steps} steps and its notes don't say why")


def _usage(document: Mapping[str, Any], found: _Findings) -> None:
    for where, step in _steps(document):
        metrics = step.get("metrics") or {}
        prompt = _count(metrics.get("prompt_tokens"))
        if prompt is None:
            continue
        writes = _count(profile_of(metrics).get("cache_write_tokens"))
        cache = (_count(metrics.get("cached_tokens")) or 0) + (writes or 0)
        if cache > prompt:
            found.add("usage", f"{where}.metrics",
                      f"cache reads and writes ({cache}) exceed prompt_tokens ({prompt}), which include them")


def validate_media(document: Mapping[str, Any], media: Path | Mapping[str, bytes] | StoredMedia, *,
                   reader: bool = False) -> list[Finding]:
    """The ``media-bytes`` findings alone, for a document ``media_refs`` can walk, whether or not ATIF's models accept
    it: a document newer than these models, which ``validate`` stops at, still names files a reader needs."""
    found = _Findings(reader)
    _media_files(document, found, media)
    return found.items


def _media(document: Mapping[str, Any], found: _Findings,
           media: Path | Mapping[str, bytes] | StoredMedia | None) -> None:
    for ref in media_refs(document):
        if "://" in ref.path:
            found.add("media-path", ref.where,
                      f"{ref.path!r} is a URL; the profile's media are files beside the document")
        elif not MEDIA_PATH.match(ref.path):
            found.add("media-path", ref.where, f"{ref.path!r} isn't media/<sha256>.<ext>")
    if media is not None:
        _media_files(document, found, media)


def _media_files(document: Mapping[str, Any], found: _Findings, media: Path | Mapping[str, bytes] | StoredMedia) -> None:
    for ref in media_refs(document):
        where, path, info = ref.where, ref.path, ref.info
        if "://" in path:
            continue
        named = MEDIA_PATH.match(path)
        if isinstance(media, StoredMedia):
            _stored(found, where, path, named, info, media)
            continue
        data = _media_bytes(media, path)
        if data is None:
            found.add("media-bytes", where, f"{path} isn't in the media given")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if named and named["sha256"] != digest:
            found.add("media-bytes", where, f"{path}'s bytes have sha256 {digest}")
        if info.get("sha256") is not None and info["sha256"] != digest:
            found.add("media-bytes", where, f"the profile gives sha256 {info['sha256']}, but the bytes have {digest}")
        if info.get("bytes") is not None and info["bytes"] != len(data):
            found.add("media-bytes", where, f"the profile gives {info['bytes']} bytes, but the file has {len(data)}")


def _stored(found: _Findings, where: str, path: str, named: re.Match[str] | None, info: Mapping[str, Any],
            media: StoredMedia) -> None:
    if path not in media.sizes:
        found.add("media-bytes", where, f"{path} isn't in the media given")
        return
    if named and info.get("sha256") is not None and info["sha256"] != named["sha256"]:
        found.add("media-bytes", where, f"the profile gives sha256 {info['sha256']}, but the name has {named['sha256']}")
    size = media.sizes[path]
    if info.get("bytes") is not None and size is not None and info["bytes"] != size:
        found.add("media-bytes", where, f"the profile gives {info['bytes']} bytes, but the file has {size}")


def _media_bytes(media: Path | Mapping[str, bytes], path: str) -> bytes | None:
    if isinstance(media, Mapping):
        return media.get(path)
    root = media.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    return target.read_bytes()


def _timestamps(document: Mapping[str, Any], found: _Findings) -> None:
    for where, step in _steps(document):
        if isinstance(step.get("timestamp"), str) and not TIMESTAMP.match(step["timestamp"]):
            found.add("timestamp", f"{where}.timestamp", f"{step['timestamp']!r} isn't in the builder's form")
        for n, result in enumerate(_results(step)):
            arrived = profile_of(result).get("arrived")
            at = arrived.get("at") if isinstance(arrived, Mapping) else None
            if isinstance(at, str) and not TIMESTAMP.match(at):
                found.add("timestamp", f"{where}.observation.results[{n}].extra.{PROFILE_KEY}.arrived.at",
                          f"{at!r} isn't in the builder's form")


def _unanswered(page: Mapping[str, Any], found: _Findings) -> None:
    """A page's steps whose calls have no result yet, served before their task ended. A sub-agent's task is its
    spawning call's, found through the references; one no served step references can't be placed, so isn't checked."""
    ended_steps: set[int] = set()
    offset = 0
    for task in tasks(page):
        if task.state is not None:
            ended_steps.update(range(offset, offset + len(task.steps)))
        offset += len(task.steps)
    resolve = _resolver(page)
    spawner: dict[str, tuple[str, int]] = {}
    for prefix, trajectory in _trajectories(page):
        for index, step in enumerate(trajectory.get("steps") or []):
            for result in _results(step):
                for ref in result.get("subagent_trajectory_ref") or []:
                    target = resolve[prefix].get(ref.get("trajectory_id"))
                    if target is not None:
                        spawner.setdefault(target, (prefix, index))

    def root_step(prefix: str) -> int | None:
        seen = set()
        while prefix in spawner and prefix not in seen:
            seen.add(prefix)
            prefix, index = spawner[prefix]
            if not prefix:
                return index
        return None

    for prefix, trajectory in _trajectories(page):
        root_index = root_step(prefix) if prefix else None
        if prefix and root_index is None:
            continue
        for index, step in enumerate(trajectory.get("steps") or []):
            if (index if not prefix else root_index) in ended_steps:
                continue
            answered = {result.get("source_call_id") for result in _results(step)}
            for n, call in enumerate(step.get("tool_calls") or []):
                if call.get("tool_call_id") not in answered:
                    found.add("live-unanswered", f"{prefix}steps[{index}].tool_calls[{n}]",
                              f"call {call.get('tool_call_id')!r} has no result, and its task hasn't ended")


def _loc(loc: Sequence[Any]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _difference(a: Any, b: Any, where: str = "") -> str:
    """The first path at which two JSON values differ."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in list(a) + [k for k in b if k not in a]:
            if key not in a or key not in b:
                return f"{where}.{key}".lstrip(".")
            if a[key] != b[key]:
                return _difference(a[key], b[key], f"{where}.{key}".lstrip("."))
    if isinstance(a, list) and isinstance(b, list):
        for index, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _difference(x, y, f"{where}[{index}]")
        if len(a) != len(b):
            return f"{where}[{min(len(a), len(b))}]"
    return where
