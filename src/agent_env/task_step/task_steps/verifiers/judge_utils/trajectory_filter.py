"""Compact OTel trajectory filtering for judge evaluation.

Uses standard OpenTelemetry GenAI semantic conventions (gen_ai.*) to identify
span types — no vendor-specific attributes (e.g. LangSmith) are required.

Tool call results that exceed _TOOL_RESULT_MAX_INLINE_SIZE or contain base64
images are externalized to JSON files under _TOOL_RESULT_FILES_DIR.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_TOOL_RESULT_MAX_INLINE_SIZE = 1024
_TOOL_RESULT_FILES_DIR = "/tmp/tool_call_results"


# Upper bound on attached screenshots — a sanity cap so a misconfigured filter can't
# attach dozens of full-resolution frames to a single judge call.
_MAX_SCREENSHOT_LAST_N = 50
# PER_CRITERION frame budget (``frame_budget``); lives here because ``frame_selection`` imports this module.
# The cap keeps under the ~100-image limit of multimodal endpoints; the floor leaves the rest of the run
# at least as many frames as the final-frame reservation.
MAX_FRAME_BUDGET = 95
DEFAULT_FRAME_BUDGET = 90
FINAL_FRAMES_RESERVED = 3
MIN_FRAME_BUDGET = 2 * FINAL_FRAMES_RESERVED
# ACTION_LOG line cap (``action_log_max_lines``, see ``frame_selection.action_log``).
DEFAULT_ACTION_LOG_MAX_LINES = 120
MIN_ACTION_LOG_MAX_LINES = 10


class CompactionType(str, Enum):
    """How a trajectory is compacted for a judge.

    DEFAULT — reduce raw spans to a compact event list and externalize tool-result
    images to side-files (the agent judge opens them with tools). This is the only
    behavior that ever existed before screenshot grading.
    SCREENSHOT — for a direct multimodal LLM judge: strip ALL base64 out of the
    trajectory text and attach frames as images — the last N (``frame_strategy=final_frames``,
    see ``compact_screenshot_trajectory``) or frames chosen per criterion across the whole run
    (``per_criterion``, see ``frame_selection``).
    """

    DEFAULT = "default"
    SCREENSHOT = "screenshot"


class FrameStrategy(str, Enum):
    """Which frames the SCREENSHOT compaction attaches: the final ``screenshot_last_n`` frames, or up to
    ``frame_budget`` frames chosen across the whole run for the criteria being graded (``frame_selection``)."""

    FINAL_FRAMES = "final_frames"
    PER_CRITERION = "per_criterion"

    @classmethod
    def _missing_(cls, value: object) -> Optional[FrameStrategy]:
        return cls.FINAL_FRAMES if value == "last_n" else None   # earlier spelling of FINAL_FRAMES


class TrajectoryText(str, Enum):
    """What text accompanies the frames on the SCREENSHOT compaction: the whole trajectory with images
    stripped, or one line per tool call / agent message (``frame_selection.action_log``) — what lets a
    ~90-frame call fit, the full text being up to 2.4M chars."""

    FULL = "full"
    ACTION_LOG = "action_log"


def _compile_or_raise(pattern: object, what: str) -> None:
    """Reject a non-string or invalid regex at construction rather than at judge time."""
    if not isinstance(pattern, str):
        raise ValueError(f"{what} must be a regex string, got {pattern!r}")
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"{what} is not a valid regex: {e}") from e


@dataclass
class TrajectoryFilter:
    """Configures how an OTel trajectory is compacted for judge evaluation.

    ``compaction_type`` selects the variant: the ``include_*`` flags drive the DEFAULT event-reduction;
    the remaining fields drive the SCREENSHOT variant. The two regex knobs are matched case-insensitively
    against ``frame_selection.rendered_action`` (``"<tool> <args json>"``). The per_criterion knobs
    (``frame_budget``, ``evidence_exclude_pattern``, ``always_show_actions``) are validated whatever the
    strategy — a task document may set them while toggling it — and ignored unless it is per_criterion.
    """

    include_tool_calls: bool = True
    include_tool_call_results: bool = True
    include_response: bool = True
    include_thinking: bool = False
    compaction_type: CompactionType = CompactionType.DEFAULT
    screenshot_last_n: int = 3
    frame_strategy: FrameStrategy = FrameStrategy.FINAL_FRAMES
    frame_budget: int = DEFAULT_FRAME_BUDGET                    # MIN_FRAME_BUDGET..MAX_FRAME_BUDGET
    # What text accompanies the frames. None (the default) resolves in ``__post_init__`` to ACTION_LOG when the
    # strategy is per_criterion (the full text next to ~90 frames is never what that caller wants) and to FULL
    # otherwise; an explicit value always wins, and ``to_dict`` writes the resolved value.
    trajectory_text: Optional[TrajectoryText] = None
    # Actions whose rendering matches are not used as criterion evidence (e.g. navigation mechanics).
    evidence_exclude_pattern: Optional[str] = None
    # {label: regex}: the frame after every matching action is reserved and labelled "<label> action K"
    # (e.g. {"submitted": "^submit_form\\b"}); see ``frame_selection`` for the cap.
    always_show_actions: Optional[dict[str, str]] = None
    action_log_max_lines: int = DEFAULT_ACTION_LOG_MAX_LINES    # >= MIN_ACTION_LOG_MAX_LINES

    def __post_init__(self) -> None:
        # Accept plain strings (e.g. constructed from JSON) and coerce to the enums.
        if isinstance(self.compaction_type, str):
            self.compaction_type = CompactionType(self.compaction_type)
        if isinstance(self.frame_strategy, str):
            self.frame_strategy = FrameStrategy(self.frame_strategy)
        if self.trajectory_text is None:
            self.trajectory_text = (TrajectoryText.ACTION_LOG if self.frame_strategy == FrameStrategy.PER_CRITERION
                                    else TrajectoryText.FULL)
        elif isinstance(self.trajectory_text, str):
            self.trajectory_text = TrajectoryText(self.trajectory_text)
        if (self.compaction_type == CompactionType.SCREENSHOT
                and self.frame_strategy == FrameStrategy.FINAL_FRAMES
                and not 1 <= self.screenshot_last_n <= _MAX_SCREENSHOT_LAST_N):
            raise ValueError(
                f"screenshot_last_n must be between 1 and {_MAX_SCREENSHOT_LAST_N}, "
                f"got {self.screenshot_last_n}"
            )
        # Strict ints: JSON can hand over 90.0, which a range test alone accepts and a slice then rejects.
        frames = self.frame_budget
        if (isinstance(frames, bool) or not isinstance(frames, int)
                or not (MIN_FRAME_BUDGET <= frames <= MAX_FRAME_BUDGET)):
            raise ValueError(
                f"frame_budget must be an integer between {MIN_FRAME_BUDGET} and "
                f"{MAX_FRAME_BUDGET}, got {frames!r}"
            )
        if self.evidence_exclude_pattern is not None:
            _compile_or_raise(self.evidence_exclude_pattern, "evidence_exclude_pattern")
        if self.always_show_actions is not None:
            if not isinstance(self.always_show_actions, dict) or not all(
                isinstance(label, str) and label for label in self.always_show_actions
            ):
                raise ValueError(
                    "always_show_actions must map non-empty labels to regexes, "
                    'e.g. {"submitted": "^submit_form\\\\b"}; got ' + repr(self.always_show_actions)
                )
            for label, pattern in self.always_show_actions.items():
                _compile_or_raise(pattern, f"always_show_actions[{label!r}]")
        lines = self.action_log_max_lines
        if isinstance(lines, bool) or not isinstance(lines, int) or lines < MIN_ACTION_LOG_MAX_LINES:
            raise ValueError(
                f"action_log_max_lines must be an integer >= {MIN_ACTION_LOG_MAX_LINES}, got {lines!r}"
            )

    def to_dict(self) -> dict:
        return {
            "include_tool_calls": self.include_tool_calls,
            "include_tool_call_results": self.include_tool_call_results,
            "include_response": self.include_response,
            "include_thinking": self.include_thinking,
            "compaction_type": self.compaction_type.value,
            "screenshot_last_n": self.screenshot_last_n,
            "frame_strategy": self.frame_strategy.value,
            "frame_budget": self.frame_budget,
            "trajectory_text": TrajectoryText(self.trajectory_text).value,     # resolved in __post_init__
            "evidence_exclude_pattern": self.evidence_exclude_pattern,
            "always_show_actions": dict(self.always_show_actions) if self.always_show_actions is not None else None,
            "action_log_max_lines": self.action_log_max_lines,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrajectoryFilter:
        return cls(
            include_tool_calls=data.get("include_tool_calls", True),
            include_tool_call_results=data.get("include_tool_call_results", True),
            include_response=data.get("include_response", True),
            include_thinking=data.get("include_thinking", False),
            compaction_type=CompactionType(data.get("compaction_type", "default")),
            screenshot_last_n=data.get("screenshot_last_n", 3),
            frame_strategy=FrameStrategy(data.get("frame_strategy", "final_frames")),
            frame_budget=data.get("frame_budget", DEFAULT_FRAME_BUDGET),
            trajectory_text=data.get("trajectory_text"),        # absent -> the strategy's default (__post_init__)
            evidence_exclude_pattern=data.get("evidence_exclude_pattern"),
            always_show_actions=data.get("always_show_actions"),
            action_log_max_lines=data.get("action_log_max_lines", DEFAULT_ACTION_LOG_MAX_LINES),
        )

def compact_otel_trajectory(
    spans: list[dict], trajectory_filter: TrajectoryFilter, result_file_prefix: str = ""
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Filter raw OTel spans into a compact list of events.

    ``result_file_prefix`` is prepended to externalized tool-result filenames so a caller
    compacting several trajectories into one dir (e.g. per-turn files) avoids collisions.

    Returns:
        (events, externalized_files) where externalized_files is a list of
        (file_path, content_json) pairs for tool results externalized to files.
    """
    if not any("gen_ai.operation.name" in s.get("attributes", {}) for s in spans):
        logger.warning(f"No OTel GenAI attributes found, returning {len(spans)} spans unfiltered")
        return spans, []

    events: list[dict] = []
    externalized_files: list[tuple[str, str]] = []
    if trajectory_filter.include_thinking:
        events.extend(_extract_thinking_blocks(spans))
    if trajectory_filter.include_tool_calls or trajectory_filter.include_tool_call_results:
        tool_events, result_files = _extract_tool_events(
            spans,
            include_calls=trajectory_filter.include_tool_calls,
            include_results=trajectory_filter.include_tool_call_results,
            result_file_prefix=result_file_prefix,
        )
        events.extend(tool_events)
        externalized_files.extend(result_files)
    if trajectory_filter.include_response:
        events.extend(_extract_final_response(spans))
    events.sort(key=lambda e: e.get("timestamp", ""))

    logger.info(
        f"Compacted {len(spans)} OTel spans into {len(events)} events "
        f"(tool_calls={trajectory_filter.include_tool_calls}, "
        f"tool_results={trajectory_filter.include_tool_call_results}, "
        f"response={trajectory_filter.include_response}, "
        f"thinking={trajectory_filter.include_thinking}, "
        f"externalized_files={len(externalized_files)})"
    )
    return events, externalized_files


def _extract_thinking_blocks(spans: list[dict]) -> list[dict]:
    """Extract thinking blocks from assistant turn spans (gen_ai.operation.name == 'chat')."""
    events = []
    for span in spans:
        attrs = span.get("attributes", {})
        if attrs.get("gen_ai.operation.name") != "chat":
            continue
        completion_raw = attrs.get("gen_ai.completion", "")
        completion = parse_json_attr(completion_raw)
        if not isinstance(completion, dict):
            continue
        content = completion.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    events.append({
                        "type": "thinking",
                        "text": thinking_text,
                        "timestamp": span.get("start_time", ""),
                    })
    return events


def _output_contains_image(tool_output: dict | list | str) -> bool:
    """Check whether a tool output structure contains any base64 image blocks."""
    blocks: list = []
    if isinstance(tool_output, dict):
        blocks = tool_output.get("content", [])
        if not isinstance(blocks, list):
            blocks = [tool_output]
    elif isinstance(tool_output, list):
        blocks = tool_output
    elif isinstance(tool_output, str):
        try:
            parsed = json.loads(tool_output)
            return _output_contains_image(parsed)
        except (json.JSONDecodeError, TypeError):
            return False
    else:
        return False

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image" and block.get("source", {}).get("data"):
            return True
        if "content" in block and isinstance(block["content"], list):
            if _output_contains_image(block["content"]):
                return True
    return False


# Matches a base64 image payload: an optional data:image/...;base64, prefix, then a
# JPEG (/9j/) or PNG (iVBORw0KGgo) magic header, then a long base64 run. The {200,}
# floor keeps it off ordinary short tokens.
_BASE64_IMAGE_RE = re.compile(
    r"(?:data:image/[a-z]+;base64,)?(?:/9j/|iVBORw0KGgo)[A-Za-z0-9+/=\r\n]{200,}"
)
# Truncate backstop for the text fed to a direct judge (after base64 stripping the
# text is far smaller). Sized for a ~1M-token judge (gemini-pro-latest): ~2.4M chars
# ≈ 600-700k tokens, leaving the rest of the window for criteria, frames, and verdict.
MAX_JUDGE_TRAJECTORY_CHARS = 2_400_000
# base64 magic-byte prefixes for the image formats CUA harnesses emit — claude_cua
# encodes screenshots as PNG, the iOS bridge as JPEG. We set the data-URI media type
# from this because some multimodal APIs validate it against the payload.
_B64_MAGIC_MEDIA_TYPES = (("iVBORw0KGgo", "image/png"), ("/9j/", "image/jpeg"))


def media_type_for_b64(b64: str) -> str:
    """Infer an image media type from a base64 payload's magic-byte prefix.
    Defaults to image/png (CUA screenshots are PNG unless a JPEG header is seen)."""
    for prefix, media_type in _B64_MAGIC_MEDIA_TYPES:
        if b64.startswith(prefix):
            return media_type
    return "image/png"


@dataclass(frozen=True)
class ImageFrame:
    """A single image extracted from a trajectory — transport-agnostic. Callers adapt
    it to whatever the judge expects (e.g. a litellm image_url block); this module
    stays free of any judge/message-shape concerns."""

    media_type: str
    data: str  # base64-encoded image bytes, no `data:` prefix


def strip_base64_images(text: str, placeholder: str = "<image omitted>") -> str:
    """Replace every base64 image payload in a serialized trajectory with a placeholder.

    Text-level companion to the structured externalization above: `_output_contains_image`
    + `_extract_tool_events` externalize TOOL-RESULT images to side files during
    compaction, but base64 also rides along in chat/prompt-history spans (not
    externalized) and in the RAW trajectory when no compaction runs. For callers that
    feed trajectory TEXT to a model (e.g. a direct LLM judge, which can't open
    externalized files), this strips ALL of it — data: URIs and bare JPEG (/9j/) / PNG
    (iVBORw0KGgo) blobs alike, wherever they appear. The {200,} floor avoids touching
    ordinary short tokens.
    """
    return _BASE64_IMAGE_RE.sub(placeholder, text)


def final_frames_from_raw(raw_text: str, last_n: int) -> list[str]:
    """Return the last ``last_n`` base64 image frames from a RAW OTel trajectory.

    Screenshots are stored on each ``execute_tool`` span at
    ``attributes["gen_ai.completion"]`` (a JSON string) under ``"screenshot"``. Read
    these from the RAW trajectory on purpose: compaction externalizes every
    screenshot to a side file, so the compacted text no longer contains them —
    running this on the compacted trajectory would silently find zero frames.

    Returns raw base64 (no ``data:`` prefix). On a parse failure, a non-list payload,
    or a screenshot-less trajectory returns ``[]``. The caller is expected to release
    ``raw_text`` (it can be ~MBs) right after calling.
    """
    try:
        spans = json.loads(raw_text)
    except Exception:
        return []
    if not isinstance(spans, list):
        return []
    frames: list[str] = []
    for span in spans:
        attrs = span.get("attributes") if isinstance(span, dict) else None
        if not isinstance(attrs, dict):
            continue
        comp = attrs.get("gen_ai.completion")
        if isinstance(comp, str):
            try:
                comp_obj = json.loads(comp)
            except Exception:
                continue
        elif isinstance(comp, dict):
            comp_obj = comp
        else:
            continue
        if isinstance(comp_obj, dict) and comp_obj.get("screenshot"):
            frames.append(comp_obj["screenshot"])
    # last_n must be a positive count; non-positive returns nothing (never "all" — a
    # silent "attach every screenshot" is a footgun for long CUA trajectories).
    return frames[-last_n:] if last_n > 0 else []


def final_image_frames(raw_text: str, last_n: int) -> list[ImageFrame]:
    """The last ``last_n`` screenshots of a RAW trajectory as ``ImageFrame``s (media type + base64)."""
    return [ImageFrame(media_type=media_type_for_b64(b64), data=b64)
            for b64 in final_frames_from_raw(raw_text, last_n)]


def strip_trajectory_images(
    raw_text: str, *, attached: Iterable[str] = (), max_chars: Optional[int] = None,
) -> str:
    """The trajectory text with every base64 image stripped, truncated to its final ``max_chars``
    (default ``MAX_JUDGE_TRAJECTORY_CHARS``; the tail is kept because final-state outcomes live there).
    ``attached`` are the payloads the caller attaches as images: only then does the placeholder say frames
    are attached — a trajectory can carry prompt-history images without any screenshot to attach, and
    telling the judge to look for absent images would mislead it."""
    if max_chars is None:
        max_chars = MAX_JUDGE_TRAJECTORY_CHARS
    attached = list(attached)
    placeholder = "<image omitted; final frames attached separately>" if attached else "<image omitted>"
    text = raw_text
    for b64 in attached:
        text = text.replace(b64, placeholder)
    text = strip_base64_images(text, placeholder)
    if len(text) > max_chars:
        text = "<trajectory truncated; showing the final portion>\n" + text[-max_chars:]
    return text


def compact_screenshot_trajectory(
    raw_text: str,
    *,
    last_n: int,
    max_chars: Optional[int] = None,
) -> tuple[str, list[ImageFrame]]:
    """``CompactionType.SCREENSHOT`` compaction of a RAW trajectory for a direct judge: the base64-free
    text and the last ``last_n`` screenshots as attachable frames (the default compaction externalizes
    images to side-files only the agent judge can open)."""
    frames = final_image_frames(raw_text, last_n)
    text = strip_trajectory_images(raw_text, attached=[f.data for f in frames], max_chars=max_chars)
    return text, frames


def _extract_tool_events(
    spans: list[dict], include_calls: bool, include_results: bool, result_file_prefix: str = ""
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Extract tool call and/or result events from tool spans.

    Returns:
        (events, externalized_files) where externalized_files contains
        (path, content_json) for tool results externalized to files.
    """
    events = []
    externalized_files: list[tuple[str, str]] = []
    result_file_counter = 0

    for span in spans:
        attrs = span.get("attributes", {})
        if attrs.get("gen_ai.operation.name") != "execute_tool":
            continue

        tool_name = span.get("name", "")
        prompt_raw = attrs.get("gen_ai.prompt", "{}")
        prompt = parse_json_attr(prompt_raw)
        tool_input = prompt.get("input", {}) if isinstance(prompt, dict) else prompt

        completion_raw = attrs.get("gen_ai.completion", "{}")
        completion = parse_json_attr(completion_raw)
        tool_output = completion.get("output", completion) if isinstance(completion, dict) else completion

        if include_calls:
            events.append({
                "type": "tool_call",
                "tool": tool_name,
                "input": tool_input,
                "timestamp": span.get("start_time", ""),
            })

        if include_results:
            output_json = json.dumps(tool_output)
            if len(output_json) > _TOOL_RESULT_MAX_INLINE_SIZE or _output_contains_image(tool_output):
                result_file_counter += 1
                file_path = f"{_TOOL_RESULT_FILES_DIR}/{result_file_prefix}tool_call_result_{result_file_counter}.json"
                externalized_files.append((file_path, output_json))
                event = {
                    "type": "tool_result",
                    "tool": tool_name,
                    "output_file": file_path,
                    "timestamp": span.get("end_time", ""),
                }
            else:
                event = {
                    "type": "tool_result",
                    "tool": tool_name,
                    "output": tool_output,
                    "timestamp": span.get("end_time", ""),
                }
            events.append(event)

    return events, externalized_files


def _extract_final_response(spans: list[dict]) -> list[dict]:
    """Extract the final response from the conversation root span (gen_ai.operation.name == 'chain')."""
    for span in spans:
        attrs = span.get("attributes", {})
        if attrs.get("gen_ai.operation.name") != "chain":
            continue
        completion_raw = attrs.get("gen_ai.completion", "")
        completion = parse_json_attr(completion_raw)
        if not isinstance(completion, dict):
            continue
        content = completion.get("content", [])
        if not isinstance(content, list):
            continue
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_parts:
            return [{
                "type": "final_response",
                "text": "\n".join(text_parts),
                "timestamp": span.get("end_time", ""),
            }]
    return []


def parse_json_attr(value: str) -> dict | list | str:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
