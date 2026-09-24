"""Per-criterion screenshot selection for a direct multimodal judge. Instead of the last N frames
(``compact_screenshot_trajectory``), the frame budget goes to the actions that matter: the actions the task
pins (``TrajectoryFilter.always_show_actions``), the action best matching each criterion plus the one after
it (an action's effect shows on the following screen), the final frames, and the rest spread over the run.
A run that fits the budget is sent whole. Every frame carries a label (``"[c3] action 17"``,
``"submitted action 4"``, ``"final action 41"``, several tags composed) the judge cites. Everything here is
trajectory-level and judge-agnostic; the verifier adapts ``LabeledFrame`` to its judge's message shape.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    DEFAULT_ACTION_LOG_MAX_LINES,
    DEFAULT_FRAME_BUDGET,
    FINAL_FRAMES_RESERVED,
    ImageFrame,
    media_type_for_b64,
    parse_json_attr,
    strip_base64_images,
)

# Cuts (with an ellipsis) applied to the rendered JSON of tool arguments — the same text the action log,
# the regex knobs and criterion matching see — tool results and agent messages.
MAX_ARGS_CHARS = 110
MAX_RESULT_CHARS = 160
MAX_AGENT_MESSAGE_CHARS = 200
# Replaces a base64 image payload in rendered text before any cut, so the cut lands on text.
_IMAGE_PLACEHOLDER = "<image>"
# A second candidate scoring at least this fraction of a criterion's best match is kept as a near-tie.
_NEAR_TIE_RATIO = 0.6

# Words that carry no evidence about WHICH action shows a criterion.
_CRITERION_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "was", "were", "been", "being", "has", "have", "are",
    "its", "his", "her", "not", "any", "all", "one", "two", "from", "into", "than", "then", "there",
    "shows", "show", "shown", "screen", "state", "final", "trajectory", "agent", "task", "user", "app",
    "exactly", "visible", "displayed", "confirm", "confirmed", "set", "least", "valid", "correct",
    "matching", "either", "should", "must", "step", "steps", "run",
})


@dataclass(frozen=True)
class TrajectoryAction:
    """One ``execute_tool`` span of a raw OTel trajectory, in run order (``index`` is 1-based)."""

    index: int
    tool: str
    args: dict
    screenshot: str | None      # raw base64 of the post-action screenshot, if the span had one
    result: object = None       # the tool result minus its `screenshot`; None if nothing is left


@dataclass(frozen=True)
class AgentMessage:
    """A ``text`` block of a ``chat`` span, placed after ``after_action`` tool calls (0 = before the first)."""

    after_action: int
    text: str


@dataclass(frozen=True)
class Transcript:
    """What the RAW trajectory records the agent doing (``actions``) and saying (``messages``)."""

    actions: list[TrajectoryAction]
    messages: list[AgentMessage]


@dataclass(frozen=True)
class LabeledFrame:
    label: str
    frame: ImageFrame


def _drop_screenshots(value: object) -> object:
    """``value`` with every ``screenshot`` key removed, at any depth."""
    if isinstance(value, dict):
        return {k: _drop_screenshots(v) for k, v in value.items() if k != "screenshot"}
    if isinstance(value, list):
        return [_drop_screenshots(v) for v in value]
    return value


def _without_screenshot(completion: object) -> object:
    """The tool result minus its ``screenshot`` keys; None when nothing is left at the TOP level. Nested
    empty values are kept — an empty string inside a list is still what the tool returned."""
    stripped = _drop_screenshots(completion)
    return None if stripped in ("", None, {}) else stripped


def _text_blocks(completion: object) -> list[str]:
    """The non-empty ``text`` blocks of a chat completion — a bare block list, or ``{"content": [...]}``."""
    blocks = completion.get("content") if isinstance(completion, dict) else completion
    if not isinstance(blocks, list):
        return []
    return [b["text"] for b in blocks
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip()]


def transcript_from_raw(raw_text: str) -> Transcript:
    """Every ``execute_tool`` span as a ``TrajectoryAction`` and every text block of a ``chat`` span as an
    ``AgentMessage``, in span order. The ``chain`` root is skipped: its completion is the final response the
    judge already gets. Malformed input gives an empty transcript."""
    try:
        spans = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return Transcript([], [])
    if not isinstance(spans, list):
        return Transcript([], [])
    actions: list[TrajectoryAction] = []
    messages: list[AgentMessage] = []
    for span in spans:
        attrs = span.get("attributes") if isinstance(span, dict) else None
        if not isinstance(attrs, dict):
            continue
        operation = attrs.get("gen_ai.operation.name")
        completion = parse_json_attr(attrs.get("gen_ai.completion", "{}"))
        if operation == "chat":
            messages.extend(AgentMessage(len(actions), text) for text in _text_blocks(completion))
        elif operation == "execute_tool":
            prompt = parse_json_attr(attrs.get("gen_ai.prompt", "{}"))
            args = prompt.get("input", prompt) if isinstance(prompt, dict) else {}
            if not isinstance(args, dict):
                args = {}
            # The tool name: harnesses that name every action span generically (e.g. "execute_tool")
            # carry the real tool in the prompt attribute; prefer it, fall back to the span name.
            tool = prompt.get("tool") if isinstance(prompt, dict) else None
            shot = completion.get("screenshot") if isinstance(completion, dict) else None
            actions.append(TrajectoryAction(
                index=len(actions) + 1, tool=str(tool or span.get("name") or ""), args=args,
                screenshot=shot if isinstance(shot, str) and shot else None,
                result=_without_screenshot(completion),
            ))
    return Transcript(actions, messages)


def _cut(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _render_json(value: object, max_chars: int) -> str:
    """``value`` as compact JSON, images replaced by ``_IMAGE_PLACEHOLDER``, cut to ``max_chars``."""
    return _cut(strip_base64_images(json.dumps(value, ensure_ascii=False), _IMAGE_PLACEHOLDER), max_chars)


def rendered_action(a: TrajectoryAction) -> str:
    """``"<tool> <args json>"``, the arguments cut to ``MAX_ARGS_CHARS`` — the text the regex knobs
    (``evidence_exclude_pattern``, ``always_show_actions``) and criterion matching see, and what the action
    log shows, so a value past the cut cannot be matched."""
    return f"{a.tool} {_render_json(a.args, MAX_ARGS_CHARS)}"


def _action_line(a: TrajectoryAction) -> str:
    line = f"{a.index}. {rendered_action(a)}"
    if a.result is not None:
        line += f" -> {_render_json(a.result, MAX_RESULT_CHARS)}"
    return line


def _message_line(m: AgentMessage) -> str:
    # Images first, whitespace second: collapsing whitespace inside a base64 run would break it into
    # pieces too short for the image pattern to catch.
    text = " ".join(strip_base64_images(m.text, _IMAGE_PLACEHOLDER).split())
    return f'agent said: "{_cut(text, MAX_AGENT_MESSAGE_CHARS)}"'


def action_log(actions: Iterable[TrajectoryAction], *, messages: Iterable[AgentMessage] = (),
               max_lines: int = DEFAULT_ACTION_LOG_MAX_LINES) -> str:
    """The run as text: ``"<i>. <tool> <args> -> <result>"`` per tool call and, in run order between them,
    ``'agent said: "..."'`` per agent message — the tool lines are facts, the agent's lines its claims.
    Over ``max_lines`` the head and the tail are kept (half each, the head taking the odd line) around one
    ``"... (N lines not shown) ..."`` marker, so the final actions are always in the log."""
    pending = sorted(messages, key=lambda m: m.after_action)
    lines: list[str] = []
    for a in actions:
        while pending and pending[0].after_action < a.index:
            lines.append(_message_line(pending.pop(0)))
        lines.append(_action_line(a))
    lines.extend(_message_line(m) for m in pending)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    tail = max_lines // 2
    head = max_lines - tail
    hidden = len(lines) - max_lines
    return "\n".join([*lines[:head], f"... ({hidden} lines not shown) ...", *lines[-tail:]])


def _terms(text: object) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", str(text or "").lower()) if w not in _CRITERION_STOPWORDS}


def _criterion_id(criterion: dict) -> str:
    return str(criterion.get("id") or "?")


def criterion_matches(criteria: list[dict], actions: list[TrajectoryAction], *,
                      exclude_pattern: re.Pattern | None = None) -> list[tuple[str, TrajectoryAction]]:
    """``[(criterion_id, action), ...]`` in criteria order — each criterion's best-matching action, then a
    near-tie (>= ``_NEAR_TIE_RATIO`` of the best score) if there is one.

    Terms are weighted by inverse document frequency across this task's criteria. Candidates carrying a
    term unique to the criterion are preferred; when none does, the shared-only matches are used (weaker
    evidence is not a reason to send none). Actions whose rendering matches ``exclude_pattern`` are skipped."""
    docs = [_terms(c.get("description")) | _terms(c.get("id")) for c in criteria]
    df: dict[str, int] = {}
    for d in docs:
        for t in d:
            df[t] = df.get(t, 0) + 1
    out: list[tuple[str, TrajectoryAction]] = []
    for c, terms in zip(criteria, docs, strict=True):
        unique: list[tuple[float, int, TrajectoryAction]] = []
        shared: list[tuple[float, int, TrajectoryAction]] = []
        for a in actions:
            rendered = rendered_action(a)
            if exclude_pattern is not None and exclude_pattern.search(rendered):
                continue
            hit = terms & _terms(rendered)
            if not hit:
                continue
            cand = (sum(1.0 / df.get(t, 1) for t in hit), a.index, a)
            (unique if any(df.get(t, 9) == 1 for t in hit) else shared).append(cand)
        scored = unique or shared
        scored.sort(key=lambda x: (-x[0], -x[1]))
        keep = scored[:1]
        for cand in scored[1:2]:
            if cand[0] >= _NEAR_TIE_RATIO * scored[0][0]:
                keep.append(cand)
        out.extend((_criterion_id(c), a) for _, _, a in keep)
    return out


@dataclass(frozen=True)
class _Want:
    """One frame a criterion wants attached."""

    cid: str
    index: int
    rank: int       # 0 = the criterion's best match, 1 = its near-tie
    effect: bool    # the frame AFTER the matching action, where the action's effect shows

    @property
    def tier(self) -> int:
        """Drop order under budget pressure: 0 = effect frames, 1 = near-tie matches, 2 = best matches."""
        return 0 if self.effect else (1 if self.rank else 2)


def _wants(criteria: list[dict], actions: list[TrajectoryAction], has_shot: set[int],
           exclude_pattern: re.Pattern | None) -> list[_Want]:
    rank: dict[str, int] = {}
    out: list[_Want] = []
    for cid, a in criterion_matches(criteria, actions, exclude_pattern=exclude_pattern):
        r = rank.get(cid, 0)
        rank[cid] = r + 1
        for j, effect in ((a.index, False), (a.index + 1, True)):
            if j in has_shot:
                out.append(_Want(cid, j, r, effect))
    return out


def _pins(shots: list[TrajectoryAction], always_show: Mapping[str, re.Pattern]) -> dict[int, list[str]]:
    """``{action index: [labels]}`` for every action with a frame whose rendering matches a pin, labels
    in configuration order."""
    out: dict[int, list[str]] = {}
    for a in shots:
        rendered = rendered_action(a)
        labels = [label for label, pattern in always_show.items() if pattern.search(rendered)]
        if labels:
            out[a.index] = labels
    return out


def _spread(items: list[int], k: int) -> list[int]:
    """Up to ``k`` of ``items``, evenly spaced from the first to the last (both always kept); ``k == 1``
    keeps the last."""
    n = len(items)
    if k <= 0:
        return []
    if k >= n:
        return list(items)
    if k == 1:
        return items[-1:]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _trim(picked: set[int], reserved: set[int], wants: list[_Want], cids: list[str],
          max_frames: int) -> None:
    """Drop criterion frames from ``picked`` until it fits ``max_frames``: lower tiers first, one frame
    per criterion per round, so no criterion loses its best match while another keeps an effect frame.
    A frame wanted at several tiers is only droppable at its highest one."""
    over = len(picked) - max_frames
    value: dict[int, int] = {}
    for w in wants:
        value[w.index] = max(value.get(w.index, -1), w.tier)
    for tier in (0, 1, 2):
        queues = {
            cid: [w.index for w in sorted(wants, key=lambda w: -w.rank) if w.cid == cid and w.tier == tier]
            for cid in cids
        }
        while over > 0 and any(queues.values()):
            for cid in cids:
                if over <= 0:
                    break
                queue = queues[cid]
                while queue:
                    j = queue.pop(0)
                    if j in picked and j not in reserved and value[j] == tier:
                        picked.remove(j)
                        over -= 1
                        break


def _pick_within_budget(shots: list[TrajectoryAction], final: set[int], pins: Mapping[int, list[str]],
                        wants: list[_Want], cids: list[str], max_frames: int) -> set[int]:
    """The frame indices to keep when the run exceeds ``max_frames``: the final frames and the pinned
    frames (at most half the budget left after the final frames, spread over the run) are reserved, the
    criterion frames are trimmed to fit (see ``_trim``), and any room left is spread over the rest of the run."""
    pin_cap = (max_frames - len(final)) // 2
    reserved = final | set(_spread([p for p in sorted(pins) if p not in final], pin_cap))
    picked = reserved | {w.index for w in wants}
    if len(picked) > max_frames:
        _trim(picked, reserved, wants, cids, max_frames)
    room = max_frames - len(picked)
    cands = [a.index for a in shots if a.index not in picked]
    if room > 0 and cands:
        step = max(1.0, len(cands) / room)
        for k in range(room):
            picked.add(cands[min(int(k * step), len(cands) - 1)])
    return picked


def _labeled_frames(actions: list[TrajectoryAction], picked: set[int], wants: list[_Want], final: set[int],
                    pins: Mapping[int, list[str]], label_ids: Mapping[str, str]) -> list[LabeledFrame]:
    """``picked`` in run order, each labelled ``"[cA][cB] final <pin labels> action K"`` (parts present as
    they apply); ``label_ids`` maps a criterion id to the key written in its tag."""
    tags: dict[int, list[str]] = {}
    for w in wants:
        if w.cid not in tags.setdefault(w.index, []):
            tags[w.index].append(w.cid)
    by_index = {a.index: a for a in actions}
    frames = []
    for j in sorted(picked):
        a = by_index[j]
        assert a.screenshot is not None
        parts = ["".join(f"[{label_ids.get(cid, cid)}]" for cid in tags.get(j, ()))]
        if j in final:
            parts.append("final")
        parts.extend(pins.get(j, ()))
        parts.append(f"action {j}")
        frames.append(LabeledFrame(" ".join(part for part in parts if part),
                                   ImageFrame(media_type=media_type_for_b64(a.screenshot), data=a.screenshot)))
    return frames


def select_frames_per_criterion(actions: list[TrajectoryAction], criteria: list[dict], *,
                                max_frames: int = DEFAULT_FRAME_BUDGET,
                                exclude_pattern: re.Pattern | None = None,
                                always_show: Mapping[str, re.Pattern] | None = None,
                                label_ids: Mapping[str, str] | None = None) -> list[LabeledFrame]:
    """The labelled frames to attach for grading ``criteria`` from ``actions``, in run order, never more
    than ``max_frames``. The final-frame reservation is ``FINAL_FRAMES_RESERVED``, shrunk to half the
    budget for tiny budgets. ``always_show`` maps a pin label to a compiled regex over ``rendered_action``;
    ``label_ids`` maps a criterion id to the key written in its frame tags (the verifier passes c1..cN)."""
    shots = [a for a in actions if a.screenshot]
    has_shot = {a.index for a in shots}
    n_final = max(1, min(FINAL_FRAMES_RESERVED, max_frames // 2))
    final = {a.index for a in shots[-n_final:]}
    pins = _pins(shots, always_show or {})
    wants = _wants(criteria, actions, has_shot, exclude_pattern)
    if len(shots) <= max_frames:
        picked = has_shot
    else:
        picked = _pick_within_budget(shots, final, pins, wants, [_criterion_id(c) for c in criteria], max_frames)
    return _labeled_frames(actions, picked, wants, final, pins, label_ids or {})
