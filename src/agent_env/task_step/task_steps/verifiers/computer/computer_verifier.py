"""computer_verifier: a computer-use task's rubric, graded by rubrics_verifier's judge on what the agent left in the
computer.

The rubric is a list of criteria (or ``{"criteria": [...]}``) with an ``id``, a ``title`` and a ``weight`` (0 when left
out), a ``negative`` one (a failure mode, its weight a penalty) when its ``annotations``' ``criteria_type`` /
``criteria_category`` say so or its weight is below 0, else a regular one; a ``visual`` category, and
``target_result_files`` / ``target_GT_files``, say how to grade it. A ``must-pass`` criterion is refused: must-pass
criteria are a step of their own, which an ``aggregate_verifiers`` names in its ``gates``.

What the judge reads, all through the env protocol's ``data/get``: each of ``result_files``, a path under the computer's
home, from the computer's data plane, or from the data plane of the app on it whose card claims a path above it,
rendered as text (``computer/files.py``), a file that isn't there named as one the agent did not produce; the export of
each of ``state_envs``, an environment by its name (the computer itself, an app on it, or a multi env's website or MCP
server), as the state the agent left; the golden files of ``expected_artifact_id``, for reference only; and
``instruction``, the task as the agent was given it. The judge also sees the agent's response to ``prompt_id``, and its
trajectory with ``use_trajectory``. Nothing is run in the computer. An env that takes the data-objects form exports
under a grant of the step's, within ``EXPORT_LIMITS``, so a file's size is known before it is read and one over
``MAX_READ_BYTES`` is only named; the uploads stay in the object store as the grade's evidence, each one's url in the
record. One that doesn't, or whose network can't reach the object store, answers inline, read only up to
``MAX_INLINE_BYTES``.

How the judge grades is ``guidance``, and how it treats the golden files ``golden_note``: the caller's own wording, or
short defaults of the framework's. The score is rubrics_verifier's, ``score_aggregator`` over the rows, a weighted
average by default; a negative criterion is graded as rubrics_verifier grades one, 1 when its failure mode is absent,
and keeps its penalty as a negative weight.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar, Optional

import httpx
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, METHOD_GET, DataPart, FilePart
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer.app import COMPUTER_APP
from agentenv_protocol.computer.dataplane import part_bytes
from agentenv_protocol.transfers import RelativePath
from agentenv_protocol.types import FileWithBytes, FileWithUri, GetDataResponse, uploaded_object_path
from pydantic import TypeAdapter, ValidationError

from agent_env.a2a_agent.object_transfer import ObjectLimits, namespace_grant
from agent_env.config import get_config
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv
from agent_env.store.base import GrantUnavailableError
from agent_env.store.object_store import MIN_GRANT_LIFETIME_SECONDS, ObjectStore
from agent_env.store.object_store.object_store import issues_grants_to
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.preflight import stored_env
from agent_env.task_step.task_steps.computer.lookup import computer_problem, find_computer
from agent_env.task_step.task_steps.verifiers.computer.files import Rendered, render_isolated
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator

#: Where the step records what the judge was shown, by step id.
RECORDS_KEY = "computer_verifier"
#: How the judge grades when the step names no ``guidance`` of its own.
DEFAULT_GUIDANCE = """Give each criterion 1.0 if what is shown above meets it and 0.0 if it doesn't, and say \
where you found the evidence.

- A criterion marked "negative": true names something that should not happen; score it as the instructions say.
- A criterion's "applies_to" lists the files it is about; one without it is about the outputs as a whole.
- A criterion marked "visual": true is about how a file looks, judged from its rendering.
- A golden file is a reference: what it holds counts only when the result holds it too.
- What the agent's response says it did is not evidence on its own: grade on the files and state shown."""
#: How the judge treats the golden files when the step names no ``golden_note`` of its own.
DEFAULT_GOLDEN_NOTE = ("Reference files: use one only for a criterion that refers to it or to an expected value, not to "
                       "compare the results with as a whole.")
#: How long one ``data/get`` may take.
READ_TIMEOUT_S = 120
#: The most of one file's rendering the judge reads.
MAX_FILE_CHARS = 40_000
#: The largest file the judge is shown; a larger one is named as too large.
MAX_READ_BYTES = 32 * 1024 * 1024
#: The most of an inline export the step reads, base64 and all; past it, the step fails rather than read the rest.
MAX_INLINE_BYTES = 128 * 1024 * 1024
#: What one export may upload under the step's grant: a file up to the limit is uploaded and then named as too large,
#: without being read; a larger one, or an export past the total, fails the step.
EXPORT_LIMITS = ObjectLimits(max_objects=10_000, max_object_bytes=1024**3, max_total_bytes=4 * 1024**3)
#: Where the step's exports are uploaded, under the artifact key prefix.
_EXPORT_KEY_PREFIX = "computer_verifier"
_PATHS = TypeAdapter(list[RelativePath])


@dataclass(frozen=True)
class _File:
    """A file an export gave: its size, its bytes unless it is over ``MAX_READ_BYTES``, and the url it was uploaded at
    under the step's grant, when it was."""

    size: int
    data: Optional[bytes]
    object_url: Optional[str] = None

    @classmethod
    def read(cls, data: bytes) -> _File:
        return cls(len(data), data)

    def record(self, path: str, shown: Rendered) -> dict:
        """What the step records of the file at ``path``, shown as ``shown``."""
        return {"path": path, "bytes": self.size, "rendered_as": shown.kind, "cut": shown.cut,
                **({"object_url": self.object_url} if self.object_url else {})}


def criteria_from_rubric(rubric: Any) -> list[dict]:
    """A rubric (``{"criteria": [...]}`` or the list) as rubrics_verifier's criteria: each with its ``id``,
    ``title`` and ``weight`` (a negative one's as a penalty, below 0), and ``negative``, ``visual`` and ``applies_to``
    when its annotations say; a must-pass one is refused, as a step of its own that an aggregate gates on."""
    items = rubric.get("criteria") if isinstance(rubric, dict) else rubric
    if not isinstance(items, list) or not items:
        raise ValueError("rubric must be a non-empty list of criteria, or an object whose criteria is one")
    criteria = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError(f"rubric criterion {index} needs an id")
        if not isinstance(item.get("title"), str) or not item["title"].strip():
            raise ValueError(f"rubric criterion {item['id']!r} needs a title, the text the judge grades")
        weight = item.get("weight", 0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"rubric criterion {item['id']!r}: weight must be a number")
        annotations = item.get("annotations") or {}
        tags = " ".join(str(annotations.get(key) or "").lower() for key in ("criteria_type", "criteria_category"))
        if "must-pass" in tags:
            raise ValueError(f"rubric criterion {item['id']!r} is must-pass: put must-pass criteria in a "
                             "computer_verifier of their own, with score_aggregator all_pass, and name that step in an "
                             "aggregate_verifiers' gates")
        negative = "negative" in tags or weight < 0
        criterion = {"id": item["id"], "title": item["title"],
                     "weight": -abs(float(weight)) if negative else float(weight)}
        if negative:
            criterion["negative"] = True
        if str(annotations.get("criteria_category") or annotations.get("category") or "").lower() == "visual":
            criterion["visual"] = True
        applies_to = {name: list(files) for name, key in (("result_files", "target_result_files"),
                                                          ("golden_files", "target_GT_files"))
                      if (files := annotations.get(key))}
        if applies_to:
            criterion["applies_to"] = applies_to
        criteria.append(criterion)
    return criteria


def _home_path(path: str) -> str:
    """``path`` as ``data/get`` names it, relative to the home: a leading ``~/`` taken off."""
    return path[2:] if path.startswith("~/") else path


class ComputerVerifierTaskStep(RubricsVerifierTaskStep):
    type: ClassVar[str] = "computer_verifier"
    DEFAULT_SCORE_AGGREGATOR: ClassVar[ScoreAggregator] = ScoreAggregator.WEIGHTED_AVERAGE
    entity_refs = (*RubricsVerifierTaskStep.entity_refs, EntityRef.env("env_id"),
                   EntityRef.artifact("expected_artifact_id", version_field="expected_artifact_version"))
    context_paths = RubricsVerifierTaskStep.context_paths

    def __init__(self, id: str, version: Optional[int], *, env_id: str, rubric: Any, result_files: list[str],
                 prompt_id: str, computer: Optional[str] = None,
                 expected_artifact_id: Optional[str] = None, expected_artifact_version: Optional[int] = None,
                 state_envs: Optional[list[str]] = None, instruction: Optional[str] = None,
                 guidance: Optional[str] = None, golden_note: Optional[str] = None, use_trajectory: bool = False,
                 criteria: Any = None, default_model: Optional[str] = None,
                 default_model_api_base: Optional[str] = None,
                 effort: Optional[str] = None, max_thinking_tokens: Optional[int] = None,
                 judge_a2a_agent_id: Optional[str] = None, judge_sandbox_type: Optional[str] = None,
                 judge_batch_size: Optional[int] = None, judge_batch_concurrency: Optional[int] = None,
                 **fields: Any):
        """``criteria`` is what ``rubric`` becomes (``criteria_from_rubric``), never read from a stored step; the rest is
        rubrics_verifier's."""
        where = f"computer_verifier '{id}'"
        for name, text in (("guidance", guidance), ("golden_note", golden_note)):
            if text is not None and (not isinstance(text, str) or not text.strip()):
                raise ValueError(f"{where}: {name} must be text, or left out for the default")
        if not isinstance(result_files, list) or not all(isinstance(p, str) and p for p in result_files):
            raise ValueError(f"{where}: result_files must be a list of paths in the computer")
        try:
            _PATHS.validate_python([_home_path(path) for path in result_files])
        except ValidationError:
            raise ValueError(f"{where}: result_files are normalized paths under the computer's home, as data/get takes "
                             f"them, relative or after ~/, not {result_files!r}") from None
        if state_envs is not None and (not isinstance(state_envs, list)
                                       or not all(isinstance(name, str) and name for name in state_envs)):
            raise ValueError(f"{where}: state_envs must be a list of environment names")
        self.env_id = env_id
        self.rubric = rubric
        self.result_files = result_files
        self.computer = computer
        self.expected_artifact_id = expected_artifact_id
        self.expected_artifact_version = expected_artifact_version
        self.state_envs = state_envs or []
        self.instruction = instruction
        self.guidance = guidance
        self.golden_note = golden_note
        super().__init__(id, version, criteria=criteria_from_rubric(rubric), prompt_id=prompt_id,
                         use_trajectory=use_trajectory,
                         default_model=default_model, default_model_api_base=default_model_api_base, effort=effort,
                         max_thinking_tokens=max_thinking_tokens, judge_a2a_agent_id=judge_a2a_agent_id,
                         judge_sandbox_type=judge_sandbox_type, judge_batch_size=judge_batch_size,
                         judge_batch_concurrency=judge_batch_concurrency, **fields)

    def preflight(self) -> list[str]:
        """rubrics_verifier's, and the env this step reads: a computer env, or a multi env with that computer."""
        problems = super().preflight()
        if isinstance(self.env_id, str) and self.env_id and (env := stored_env(self.env_id)) is not None:
            if problem := computer_problem(env, self.computer):
                problems.append(f"step {self.id!r}: {problem}")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id, "rubric": self.rubric, "result_files": self.result_files,
                "computer": self.computer, "expected_artifact_id": self.expected_artifact_id,
                "expected_artifact_version": self.expected_artifact_version, "state_envs": self.state_envs,
                "instruction": self.instruction, "guidance": self.guidance, "golden_note": self.golden_note}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerVerifierTaskStep:
        """rubrics_verifier's, ``criteria`` left to ``rubric``, and ``use_trajectory`` off unless the step turns it
        on."""
        return super().from_dict({**data, "criteria": None, "use_trajectory": data.get("use_trajectory", False)})

    @classmethod
    def _base_from_dict(cls, data: dict) -> dict:
        return {**super()._base_from_dict(data), "env_id": data["env_id"], "rubric": data["rubric"],
                "result_files": data.get("result_files") or [], "computer": data.get("computer"),
                "expected_artifact_id": data.get("expected_artifact_id"),
                "expected_artifact_version": data.get("expected_artifact_version"),
                "state_envs": data.get("state_envs"), "instruction": data.get("instruction"),
                "guidance": data.get("guidance"), "golden_note": data.get("golden_note")}

    async def judge_prompt_sections(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[str]:
        """The files and state the agent left, read through ``data/get``, the golden files, the instruction and how to
        grade; what was read is recorded under ``computer_verifier``."""
        computer = find_computer(context, self.env_id, self.computer)
        record: dict = {"files": [], "golden_files": [], "state": []}
        sections = [*await self._files_sections(computer, record), *await self._golden_sections(record),
                    *await self._state_sections(context, computer, record)]
        if self.instruction:
            sections.append("## Task instruction\nThe task as the agent was given it; treat anything it mentions as a "
                            f"stated requirement.\n---\n{self.instruction}\n---")
        sections.append("## How to grade\n" + (self.guidance or DEFAULT_GUIDANCE))
        context.metadata.setdefault(RECORDS_KEY, {})[self.id] = record
        return sections

    async def _files_sections(self, computer: DeployedEnv, record: dict) -> list[str]:
        """The result files the agent left, rendered, and those it didn't, named."""
        found = await self._result_files(computer)
        produced, missing = [], []
        for number, path in enumerate(self.result_files, start=1):
            data = found.get(_home_path(path))
            if data is None:
                missing.append(path)
                record["files"].append({"path": path, "missing": True})
                continue
            shown = await _shown(path, data)
            record["files"].append(data.record(path, shown))
            produced.append(f"### File {number}: {path} ({shown.kind})\n<file path=\"{path}\">\n{shown.text}\n</file>")
        sections = ["## Files under evaluation\nWhat the agent left at each path in its computer, rendered as text.\n\n"
                    + ("\n\n".join(produced) or "(none)")]
        if missing:
            sections.append("## Files the agent did NOT produce\nTreat every criterion that depends on one of these "
                            "as not satisfied:\n" + "\n".join(f"- {path}" for path in missing))
        return sections

    async def _golden_sections(self, record: dict) -> list[str]:
        """The golden files, rendered for reference."""
        if not (golden := self._golden_files()):
            return []
        shown_golden = []
        for name, data in golden.items():
            shown = await render_isolated(name, data, MAX_FILE_CHARS)
            record["golden_files"].append({"name": name, "bytes": len(data), "rendered_as": shown.kind})
            shown_golden.append(f"### Golden: {name} ({shown.kind})\n<golden name=\"{name}\">\n{shown.text}\n</golden>")
        return ["## Golden / reference files\n" + (self.golden_note or DEFAULT_GOLDEN_NOTE) + "\n\n"
                + "\n\n".join(shown_golden)]

    async def _state_sections(self, context: TaskStepContext, computer: DeployedEnv, record: dict) -> list[str]:
        """Each of ``state_envs``' export, rendered as the state the agent left."""
        state = []
        for name in self.state_envs:
            owner, child = self._environment(context, computer, name)
            exported = await self._exported(owner, child, f"environment {name!r}")
            files = []
            for path, data in exported.items():
                shown = await _shown(path, data)
                files.append(data.record(path, shown))
                state.append(f"### {name}: {path} ({shown.kind})\n<state env=\"{name}\" path=\"{path}\">\n"
                             f"{shown.text}\n</state>")
            if not exported:
                state.append(f"### {name}\n(it exported nothing)")
            record["state"].append({"env": name, "files": files})
        if not state:
            return []
        return ["## Environment state\nWhat each environment exported (data/get) after the agent finished, rendered as "
                "text.\n\n" + "\n\n".join(state)]

    async def _result_files(self, computer: DeployedEnv) -> dict[str, _File]:
        """Each of ``result_files`` there is, by its path under the home: from the export of the app on the computer
        whose card claims a path above it, as the computer's own ``data/get`` leaves those out, else from the
        computer's, asked for by path."""
        apps = [(child, owned) for child in protocol_v1.children(computer.environment_card or {})
                if (owned := protocol_v1.extension_params(child.card, COMPUTER_APP).get("paths"))]
        wanted: dict[Optional[str], list[str]] = {}
        for path in dict.fromkeys(_home_path(path) for path in self.result_files):
            app = next((child.name for child, owned in apps
                        if any(PurePosixPath(path).is_relative_to(prefix) for prefix in owned)), None)
            wanted.setdefault(app, []).append(path)
        found: dict[str, _File] = {}
        if own := wanted.pop(None, None):
            found |= await self._exported(computer, None, "its computer", paths=own)
        for child, _ in apps:
            if child.name in wanted:
                exported = await self._exported(computer, child, f"app {child.name!r}")
                found |= {path: data for path, data in exported.items() if path in wanted[child.name]}
        return found

    def _environment(self, context: TaskStepContext, computer: DeployedEnv,
                     name: str) -> tuple[DeployedEnv, protocol_v1.ChildEnv]:
        """The environment ``name`` as the cards give it, with the deploy record whose card holds it: the computer
        itself or an app on it, else a child of the env, a multi env's website or MCP server."""
        env = next(deployed for deployed in context.deployed_envs if deployed.env_id == self.env_id)
        for owner in [computer] if env is computer else [computer, env]:
            if (child := owner.child_env(name)) is not None:
                if child.serves(METHOD_GET) is False:
                    raise RuntimeError(f"computer_verifier '{self.id}': environment {name!r} of env {self.env_id!r} "
                                       "serves no data/get")
                return owner, child
        raise RuntimeError(f"computer_verifier '{self.id}': env {self.env_id!r} has no environment named {name!r}")

    async def _exported(self, owner: DeployedEnv, child: Optional[protocol_v1.ChildEnv], what: str,
                        paths: Optional[list[str]] = None) -> dict[str, _File]:
        """What the ``data/get`` of ``child`` of ``owner``'s card (``owner``'s own when None) answers, each file by its
        path: under the step's grant when it takes the data-objects form and the store's grants reach it, each file's
        size read before its bytes; asked again without, inline and read only up to ``MAX_INLINE_BYTES``, when it fails
        under one, as a computer whose network can't reach the store does."""
        base_url = child.base_url(owner.environment_url) if child is not None else owner.environment_url
        card = child.card if child is not None else owner.environment_card or {}
        upload = await asyncio.to_thread(_upload_grant, card, getattr(owner, "sandbox_type", None))
        params = {"timeout": READ_TIMEOUT_S, "paths": paths, "headers": owner.control_headers_for(base_url),
                  "max_bytes": MAX_INLINE_BYTES}
        answer = None
        try:
            if upload is not None:
                try:
                    answer = await protocol_v1.get_data(base_url, write_namespace=upload[1], **params)
                except RuntimeError:
                    upload = None
            if answer is None:
                answer = await protocol_v1.get_data(base_url, **params)
        except httpx.TransportError as e:
            raise RuntimeError(f"computer_verifier '{self.id}': data/get of {what} in env {self.env_id!r} failed: "
                               f"{type(e).__name__}: {e}") from e
        except protocol_v1.ResponseTooLarge as e:
            raise RuntimeError(f"computer_verifier '{self.id}': data/get of {what} in env {self.env_id!r} answered "
                               f"inline with over {MAX_INLINE_BYTES} bytes, more than the step reads; an env that takes "
                               "the data-objects form, on a network that reaches the object store, exports under "
                               "the step's grant instead") from e
        return await self._files_of(answer, upload, what)

    async def _files_of(self, answer: GetDataResponse, upload: Optional[tuple[ObjectStore, Any]],
                        what: str) -> dict[str, _File]:
        """Each file of ``answer`` by its path: one uploaded under ``upload``'s grant, read once its size is known; else
        one in the data-objects form by ``part_bytes`` (inline, through a read grant or from a URL); a plain inline one;
        or a data part, as JSON."""
        files: dict[str, _File] = {}
        for number, part in enumerate(answer.parts, start=1):
            if isinstance(part, DataPart):
                files[f"data-{number}.json"] = _File.read(json.dumps(part.data, indent=2).encode())
                continue
            if not isinstance(part, FilePart):
                raise RuntimeError(f"computer_verifier '{self.id}': data/get of {what} in env {self.env_id!r} gave a "
                                   f"{part.kind} part, not a file")
            marker = (part.metadata or {}).get(DATA_OBJECTS_EXTENSION_URI)
            name = (marker.get("path") if isinstance(marker, dict) else None) or part.file.name or f"file-{number}"
            if upload is not None and marker is not None and isinstance(part.file, FileWithUri) \
                    and "://" not in part.file.uri:
                files[name] = await asyncio.to_thread(_read_uploaded, *upload, uploaded_object_path(part))
            elif marker is not None:
                files[name] = _File.read(await part_bytes(part))
            elif isinstance(part.file, FileWithBytes):
                files[name] = _File.read(base64.b64decode(part.file.bytes))
            else:
                raise RuntimeError(f"computer_verifier '{self.id}': data/get of {what} in env {self.env_id!r} gave "
                                   f"{name} only by a link outside the data-objects form")
        return files

    def _golden_files(self) -> dict[str, bytes]:
        if self.expected_artifact_id is None:
            return {}
        from agent_env.artifact import Artifact

        artifact = Artifact.get(self.expected_artifact_id, self.expected_artifact_version)
        return {name: file.load() for name, file in artifact.get_file_artifacts().items()}


def _upload_grant(card: dict, sandbox_type: Optional[str]) -> Optional[tuple[ObjectStore, Any]]:
    """The store and a grant of the step's to upload an export under, within ``EXPORT_LIMITS``, for an env whose card
    takes the data-objects form; None when it doesn't, or the store issues no grant that reaches its sandbox."""
    if protocol_v1.find_extension(card, DATA_OBJECTS_EXTENSION_URI) is None:
        return None
    config = get_config()
    store = config.get_object_store()
    if not issues_grants_to(store, sandbox_type):
        return None
    cap = store.max_single_upload_bytes or EXPORT_LIMITS.max_object_bytes
    limits = ObjectLimits(max_objects=EXPORT_LIMITS.max_objects,
                          max_object_bytes=min(EXPORT_LIMITS.max_object_bytes, cap),
                          max_total_bytes=min(EXPORT_LIMITS.max_total_bytes, cap))
    namespace = store.object_url(f"{config.get_artifact_key_prefix()}{_EXPORT_KEY_PREFIX}/{uuid.uuid4().hex}")
    try:
        return store, namespace_grant(store, namespace, limits=limits,
                                      expires_in=max(READ_TIMEOUT_S, MIN_GRANT_LIFETIME_SECONDS))
    except GrantUnavailableError:
        return None


def _read_uploaded(store: ObjectStore, grant: Any, path: str) -> _File:
    """The file an export uploaded at ``path`` under ``grant``, its bytes read only when its size is under the limit."""
    url = store.object_url(f"{grant.root_path}/{path}")
    size = getattr(store.get_object_metadata_at(url), "size", None)
    if size is not None and size > MAX_READ_BYTES:
        return _File(size, None, url)
    data = store.get(url)
    return _File(len(data), data, url)


async def _shown(name: str, file: _File) -> Rendered:
    """``file`` as the judge reads it: rendered as text, or, over ``MAX_READ_BYTES``, named as too large."""
    if file.data is None or file.size > MAX_READ_BYTES:
        return Rendered("too_large", f"(a file of {file.size} bytes, too large to show: over {MAX_READ_BYTES} bytes)")
    return await render_isolated(name, file.data, MAX_FILE_CHARS)
