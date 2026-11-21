"""A prompt's atif/1 turns as judges read them. A run that prescribes atif/1 stores each turn's copy of the
conversation at one url that its later turns overwrite, so a turn labelled ``atif/1`` is read as its own A2A task cut
out of the document there (``task_document``): that turn alone, a document ATIF's models load. A turn with any other
label keeps the native readers. Each document is read once per judge, and the media a view names are read from the
document's folder, each checked against its name."""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from agentenv_protocol.atif import FORMAT, AtifError, check_media, media_file, media_paths, tasks, tasks_document

from agent_env.config import get_config
from agent_env.store.base import NotFoundError
from agent_env.task_step.context import PromptResponse
from agent_env.task_step.task_steps.verifiers.judge_utils.frame_selection import (
    AgentMessage,
    TrajectoryAction,
    Transcript,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    _TOOL_RESULT_FILES_DIR,
    _TOOL_RESULT_MAX_INLINE_SIZE,
    ImageFrame,
    TrajectoryFilter,
)

logger = logging.getLogger(__name__)

ATIF = FORMAT


#: How much of a media file is hashed at a time when it's checked for staging, so a check holds no whole file.
_HASH_CHUNK = 1 << 20
#: Each image format a frame's bytes can name, by its file's leading bytes.
_IMAGE_SIGNATURES = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"),
                     (b"GIF89a", "image/gif"))


@dataclass(frozen=True)
class TurnSource:
    """A turn's stored trajectory: its url and, for a turn labelled atif/1, the A2A task its view cuts out of the
    document there and the step the turn reached, which finds the task in a copy without the profile's ids;
    ``task_id`` None reads the url as the native readers do."""

    url: str
    task_id: Optional[str] = None
    through_step: Optional[int] = None

    @property
    def atif(self) -> bool:
        return self.task_id is not None


def turns(prompt_response: PromptResponse) -> list[Optional[TurnSource]]:
    """Each entry of the per-turn url list as its source, by its record's label, None for a turn that stored none."""
    records = prompt_response.target_agent_per_turn_trajectories or []
    sources: list[Optional[TurnSource]] = []
    for index, url in enumerate(prompt_response.target_agent_per_turn_trajectory_object_urls or []):
        record = records[index] if index < len(records) else None
        copy = (record.get("atif") or {}) if record is not None and record.get("format") == ATIF else {}
        if copy.get("url") and copy.get("task_id"):
            sources.append(TurnSource(copy["url"], copy["task_id"], copy.get("through_step")))
        else:
            sources.append(TurnSource(url) if url else None)
    return sources


def atif_turns(prompt_response: PromptResponse) -> Optional[list[TurnSource]]:
    """The turns with a stored trajectory, in order, when any is labelled atif/1; None leaves the whole response to the
    native readers, so a run that prescribes nothing is read as it always was."""
    sources = [source for source in turns(prompt_response) if source is not None]
    return sources if any(source.atif for source in sources) else None


def media_url(document_url: str, path: str) -> str:
    """Where a document's media part ``path`` is stored: its folder's ``media/<sha256>.<ext>``, never anywhere else."""
    media_file(path)
    return f"{document_url.rsplit('/', 1)[0]}/{path}"


class AtifDocuments:
    """The atif/1 documents one judge reads, each read once, and the turn views cut from them. A media file's bytes are
    read only when a judge is handed them, and staging keeps only whether each file is stored as its name says."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        self._views: dict[TurnSource, dict[str, Any]] = {}
        self._stored: dict[str, bool] = {}

    def document(self, url: str) -> dict[str, Any]:
        if url not in self._documents:
            self._documents[url] = json.loads(get_config().get_object_store_at(url).get(url))
        return self._documents[url]

    def view(self, source: TurnSource) -> dict[str, Any]:
        """The turn alone, as a document of its own: its task cut out of the conversation's document, found by its id,
        else, in a copy without the profile, as the task ending at the turn's step. A copy neither finds the turn in, or
        whose steps can't be sliced, is the turn's view as stored, since the requested trajectory is best effort."""
        if source not in self._views:
            document = self.document(source.url)
            try:
                found = tasks(document)
                picked = ([task for task in found if task.task_id == source.task_id]
                          or [task for task in found if task.steps and task.steps[-1].get("step_id") == source.through_step])
            except (AttributeError, TypeError):
                picked = []
            if picked:
                self._views[source] = tasks_document(document, picked[:1])
            else:
                logger.warning(f"the atif/1 copy at {source.url} holds neither task {source.task_id} nor a task ending at "
                               f"step {source.through_step}, so its judge reads the copy as stored")
                self._views[source] = document
        return self._views[source]

    def text(self, source: TurnSource) -> str:
        return json.dumps(self.view(source), ensure_ascii=False)

    def media(self, document_url: str, path: str) -> Optional[bytes]:
        """The media file ``path`` of the document at ``document_url``, checked against its name; None, logged, for one
        that isn't ``media/<sha256>.<ext>``, isn't stored or holds other bytes, which the judge goes without."""
        try:
            url = media_url(document_url, path)
            data = get_config().get_object_store_at(url).get(url)
            check_media(path, data)
        except (NotFoundError, AtifError) as exc:
            logger.warning(f"ATIF media {path!r} of {document_url} skipped: {exc}")
            return None
        return data

    def staged_media(self, source: TurnSource) -> list[tuple[str, str]]:
        """``(path, url)`` of each media file the turn's view names that is stored as its name says, each hashed as it
        streams in and its bytes let go, so staging a long run's screenshots holds none of them."""
        try:
            paths = media_paths(self.view(source))
        except (AttributeError, TypeError):  # a copy graded as stored whose shape names no media a reader can find
            paths = []
        return [(path, url) for path in paths if (url := self._checked(source.url, path)) is not None]

    def _checked(self, document_url: str, path: str) -> Optional[str]:
        try:
            url = media_url(document_url, path)
        except AtifError as exc:
            logger.warning(f"ATIF media {path!r} of {document_url} skipped: {exc}")
            return None
        if url not in self._stored:
            digest = hashlib.sha256()
            try:
                with get_config().get_object_store_at(url).open(url) as stream:
                    for chunk in iter(lambda: stream.read(_HASH_CHUNK), b""):
                        digest.update(chunk)
                self._stored[url] = digest.hexdigest() == media_file(path)
            except NotFoundError:
                self._stored[url] = False
            if not self._stored[url]:
                logger.warning(f"ATIF media {path} of {document_url} skipped: not stored as its name says")
        return url if self._stored[url] else None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content or []
                     if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str))


def _has_image(content: Any) -> bool:
    return isinstance(content, list) and any(isinstance(part, Mapping) and part.get("type") == "image"
                                             for part in content)


def _steps(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A trajectory's steps a reader can walk: a stored copy may be invalid, and is still graded as stored."""
    steps = trajectory.get("steps")
    return [step for step in steps if isinstance(step, Mapping)] if isinstance(steps, list) else []


def _results(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observation = step.get("observation")
    results = observation.get("results") if isinstance(observation, Mapping) else None
    return [result for result in results if isinstance(result, Mapping)] if isinstance(results, list) else []


def _by_call(step: Mapping[str, Any]) -> list[tuple[Optional[Mapping[str, Any]], list[Mapping[str, Any]]]]:
    """Each tool call of the step with its results by ``source_call_id``, then the results no call names."""
    calls = step.get("tool_calls")
    calls = [call for call in calls if isinstance(call, Mapping)] if isinstance(calls, list) else []
    results = _results(step)
    ids = {call.get("tool_call_id") for call in calls}
    paired = [(call, [r for r in results if r.get("source_call_id") == call.get("tool_call_id")]) for call in calls]
    unpaired = [r for r in results if r.get("source_call_id") not in ids]
    return paired + ([(None, unpaired)] if unpaired else [])


class _SubAgents:
    """The embedded sub-agents a trajectory's results start, each walked once: a reference names one among its
    trajectory's own, else the nearest around it, as ATIF makes an id unique only within its parent's array (the
    builder embeds them all in the root)."""

    def __init__(self) -> None:
        self._walked: set[int] = set()

    @staticmethod
    def visible(trajectory: Mapping[str, Any], outer: Mapping[Any, Mapping[str, Any]]) -> dict[Any, Mapping[str, Any]]:
        subs = trajectory.get("subagent_trajectories")
        return {**outer, **{sub.get("trajectory_id"): sub for sub in subs if isinstance(sub, Mapping)}} \
            if isinstance(subs, list) else dict(outer)

    def started(self, result: Mapping[str, Any],
                visible: Mapping[Any, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        refs = result.get("subagent_trajectory_ref")
        found = []
        for ref in refs if isinstance(refs, list) else []:
            sub = visible.get(ref.get("trajectory_id")) if isinstance(ref, Mapping) else None
            if sub is not None and id(sub) not in self._walked:
                self._walked.add(id(sub))
                found.append(sub)
        return found


def compact_atif_trajectory(
    document: Mapping[str, Any], trajectory_filter: TrajectoryFilter, result_file_prefix: str = "",
) -> tuple[list[dict], list[tuple[str, str]]]:
    """``compact_otel_trajectory``'s events from an ATIF document, in step order: an agent step's
    ``reasoning_content`` as thinking, each tool call and then its results (one over the inline size or holding an
    image written out to a file, as the OTel compaction does), a sub-agent's events after the result that started
    it, and the root's last agent message as the final response; each with its step's timestamp."""
    events: list[dict] = []
    files: list[tuple[str, str]] = []
    numbers = itertools.count(1)
    subagents = _SubAgents()
    calls_on = trajectory_filter.include_tool_calls
    results_on = trajectory_filter.include_tool_call_results

    def walk(trajectory: Mapping[str, Any], outer: Mapping[Any, Mapping[str, Any]]) -> None:
        visible = subagents.visible(trajectory, outer)
        for step in _steps(trajectory):
            if step.get("source") != "agent":
                continue
            timestamp = step.get("timestamp") or ""
            if trajectory_filter.include_thinking and step.get("reasoning_content"):
                events.append({"type": "thinking", "text": step["reasoning_content"], "timestamp": timestamp})
            for call, results in _by_call(step):
                tool = (call or {}).get("function_name") or ""
                if call is not None and calls_on:
                    events.append({"type": "tool_call", "tool": tool, "input": call.get("arguments"),
                                   "timestamp": timestamp})
                for result in results:
                    if results_on:
                        output = result.get("content")
                        output_json = json.dumps(output)
                        if len(output_json) > _TOOL_RESULT_MAX_INLINE_SIZE or _has_image(output):
                            path = f"{_TOOL_RESULT_FILES_DIR}/{result_file_prefix}tool_call_result_{next(numbers)}.json"
                            files.append((path, output_json))
                            events.append({"type": "tool_result", "tool": tool, "output_file": path,
                                           "timestamp": timestamp})
                        else:
                            events.append({"type": "tool_result", "tool": tool, "output": output,
                                           "timestamp": timestamp})
                    for sub in subagents.started(result, visible):
                        walk(sub, visible)

    walk(document, {})
    if trajectory_filter.include_response:
        final = next((step for step in reversed(_steps(document))
                      if step.get("source") == "agent" and _text(step.get("message")).strip()), None)
        if final is not None:
            events.append({"type": "final_response", "text": _text(final.get("message")),
                           "timestamp": final.get("timestamp") or ""})
    return events, files


def transcript_from_atif(
    document: Mapping[str, Any], media: Callable[[str], Optional[bytes]],
) -> tuple[Transcript, list[ImageFrame]]:
    """The agent's steps as ``transcript_from_raw`` reads a native run, a sub-agent's after the result that started it:
    each tool call an action, its screenshot the first image of its results, its result their text; each agent message
    placed after the calls before it. And every image of the steps' results in order, the frames a final-frames judge
    takes the last of, each typed by its bytes as the native reader types them. ``media`` reads a part's file, None for
    one the judge goes without."""
    actions: list[TrajectoryAction] = []
    messages: list[AgentMessage] = []
    frames: list[ImageFrame] = []
    subagents = _SubAgents()

    def walk(trajectory: Mapping[str, Any], outer: Mapping[Any, Mapping[str, Any]]) -> None:
        visible = subagents.visible(trajectory, outer)
        for step in _steps(trajectory):
            if step.get("source") != "agent":
                continue
            text = _text(step.get("message"))
            if text.strip():
                messages.append(AgentMessage(len(actions), text))
            for call, results in _by_call(step):
                shots: list[ImageFrame] = []
                for result in results:
                    content = result.get("content")
                    for part in content if isinstance(content, list) else []:
                        source = part.get("source") if isinstance(part, Mapping) and part.get("type") == "image" else None
                        data = media(source["path"]) if isinstance(source, Mapping) and source.get("path") else None
                        if data is not None:
                            shots.append(ImageFrame(media_type=_image_type(data, source.get("media_type")),
                                                    data=base64.b64encode(data).decode()))
                frames.extend(shots)
                if call is not None:
                    arguments = call.get("arguments")
                    result_text = "\n".join(t for t in (_text(r.get("content")) for r in results) if t)
                    actions.append(TrajectoryAction(
                        index=len(actions) + 1, tool=str(call.get("function_name") or ""),
                        args=arguments if isinstance(arguments, dict) else {},
                        screenshot=shots[0].data if shots else None, result=result_text or None,
                        screenshot_media_type=shots[0].media_type if shots else None))
                for result in results:
                    for sub in subagents.started(result, visible):
                        walk(sub, visible)

    walk(document, {})
    return Transcript(actions, messages), frames


def _image_type(data: bytes, declared: Optional[str]) -> str:
    """An image's media type by its leading bytes, which a judge's model checks the payload against; the type its part
    declares only when the bytes don't say."""
    for signature, media_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return declared or "image/png"
