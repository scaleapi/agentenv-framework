"""A check on every simulated-user message before prompt_agent sends it to the agent.

An LLM playing the user can slip into the assistant's role: it answers its own question, reports work as
done, or offers to do the agent's job. Once one such message is in the conversation the rest of the run
is no longer a test of the agent. The gate checks each user-sim message, asks the user-sim to rewrite one
that fails, and, if the rewrites fail too, ends the conversation and marks the run.

Two checks, either of which can fail a message:

- ``patterns``: phrases and shapes that belong to the assistant's side (a report of work done, a
  formatted deliverable, offers to do the work). Free, and deliberately narrow.
- ``judge_model``: one model call that reads the last turns and the candidate message and says whether
  the message is written as the user.

Opt in with ``user_sim_gate`` on a ``prompt_agent`` step that has a user-sim (a step type that runs its own
conversation loop applies the gate with ``run_gate``, and sets ``USER_SIM_GATE_SUPPORTED``)::

    {"judge_model": "<a small, fast model>", "max_rewrites": 2, "on_fail": "fail"}
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, ClassVar, Optional

logger = logging.getLogger(__name__)

ON_FAIL = ("fail", "end", "send")

# ── Patterns ──────────────────────────────────────────────────────────────────────────────────────────

_REPORT_OPENING = re.compile(
    r"^\s*(?:\*\*)?(?:done\b|all done|checked\b|short (?:answer|version)\b"
    r"|here'?s (?:what (?:i|we) (?:found|did|checked|changed)|the (?:full |short |whole )?(?:picture|summary|breakdown"
    r"|rundown|update|status)|where (?:things|we|it|you) stands?|how (?:it|things) (?:stands?|looks?))"
    r"|summary\b|update:|findings\b|tl;?dr\b)",
    re.I,
)
_WORK_DONE = re.compile(
    r"\bI(?:'ve| have)? (?:checked|reviewed|pulled|updated|put together|drafted|flagged|verified|"
    r"cross-checked|went through|compared|audited|analy[sz]ed|reconciled|fixed|created|added|"
    r"rewrote|built|calculated|ran)\b",
    re.I,
)
_NOW_LIVE = re.compile(r"\b(?:now live|is now (?:up|live|updated)|has been (?:updated|created|sent|posted))\b", re.I)
_DELIVERED = re.compile(
    r"\b(?:(?:emails?|messages?|drafts?|replies|invites?|notes?) (?:sent|drafted|posted|scheduled)\b|incorporated\b|"
    r"(?:the|a|my) (?:cleaner|updated|revised|corrected|final) version\b)",
    re.I,
)
_AS_ASSISTANT = re.compile(
    r"\b(?:as an ai|i'?m an? (?:ai|assistant|language model)|my previous (?:answer|response|reply)|"
    r"i apologi[sz]e for (?:the|my)|let me know if you'?d like me to|"
    r"would you like me to|want me to (?:draft|send|update|create|go ahead)|"
    r"standing by|ready when you are|parked until|"
    r"anything else (?:you(?:'d)? (?:want|like) (?:me )?to|to (?:cover|look at|dig into))|"
    r"what do you want (?:me )?to do next|should i (?:send|draft|go ahead)|"
    r"i'll (?:draft|fold|pull|search|put together|pick (?:it |this )?up|rerun|re-run)|"
    r"i can (?:send|draft|search|pull) (?:it|that|them)|"
    r"i (?:don't|do not) have (?:the )?(?:tools|access)|corrections (?:noted|taken)|"
    r"(?:send|drop) me (?:whatever|that phrase|the results)|"
    r"how (?:can|may) i (?:help|assist)|i'?m (?:here|ready) to help|what would you like (?:me )?to (?:work on|do) today)\b",
    re.I,
)
_TABLE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*\|", re.M)
_HEADER = re.compile(r"^\s*#{1,4} \S", re.M)
_BOLD_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\*\*", re.M)


class PatternLimits:
    """How long or how formatted a message must be before a shape-based pattern counts. Phrases that only the
    assistant says (offers, "standing by", "now live") count at any length; shapes a user can also write (an
    opening like "Done.", a table, "I checked") count only on a message long enough to be a report."""

    REPORT_OPENING_MIN_CHARS: ClassVar[int] = 300
    DELIVERED_MIN_CHARS: ClassVar[int] = 300
    FORMATTED_MIN_CHARS: ClassVar[int] = 600
    FORMATTED_MIN_HEADERS: ClassVar[int] = 2
    FORMATTED_MIN_BOLD_BULLETS: ClassVar[int] = 3
    WORK_REPORT_MIN_CHARS: ClassVar[int] = 500
    WORK_REPORT_MIN_PHRASES: ClassVar[int] = 2


def pattern_reasons(text: str) -> list[str]:
    """Why ``text`` reads as the assistant's message, one reason per pattern that matched; empty if none."""
    lim = PatternLimits
    reasons = []
    n = len(text)
    if _AS_ASSISTANT.search(text):
        reasons.append("it talks like the assistant (offers to do the work, waits on the user, or speaks as the AI)")
    if _REPORT_OPENING.search(text) and n > lim.REPORT_OPENING_MIN_CHARS:
        reasons.append("it opens like a report of work done")
    if _NOW_LIVE.search(text):
        reasons.append("it says something is now live or updated")
    if _DELIVERED.search(text) and n > lim.DELIVERED_MIN_CHARS:
        reasons.append("it reports something delivered or revised")
    formatted = (_TABLE.search(text) or len(_HEADER.findall(text)) >= lim.FORMATTED_MIN_HEADERS
                 or len(_BOLD_BULLET.findall(text)) >= lim.FORMATTED_MIN_BOLD_BULLETS)
    if formatted and n > lim.FORMATTED_MIN_CHARS:
        reasons.append("it is a formatted deliverable (a table, headings or bold bullet points)")
    work = _WORK_DONE.findall(text)
    if work and (n > lim.WORK_REPORT_MIN_CHARS or len(work) >= lim.WORK_REPORT_MIN_PHRASES):
        reasons.append("it reports checking or building things itself")
    return reasons


# ── Judge ─────────────────────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """You check one message in a test conversation. A model is playing a USER who gave an AI ASSISTANT \
a task. The assistant does the work; the user asks, answers questions the assistant put to them, decides, approves, \
pushes back, or ends the conversation.

Decide whether the CANDIDATE message is written as the user. It is NOT written as the user when it does the \
assistant's job: it answers the question the user should be asking, reports work as done or findings it produced \
("Done. Here's what I found", "I've updated the page"), presents analysis or a deliverable, apologizes for "my \
previous answer", offers to do work ("I'll draft it", "Standing by", "Want me to send it?"), or speaks as the AI.

It IS written as the user when it asks, instructs, approves, pushes back, answers a question the assistant asked, \
or shares the user's own knowledge, even if it is long, formatted, quotes the assistant's numbers, or pastes an \
email or document. Length and markdown alone are not evidence.

Judge only whose voice the message is in. A user message that repeats an earlier one, ignores or contradicts what \
the assistant said, states a decision or a fact the user holds, or seems off topic is still written as the user.

CONVERSATION SO FAR (most recent last):
{history}

CANDIDATE (the user's next message):
<<<
{candidate}
>>>

Answer in JSON: {{"in_role": true or false, "reason": "one sentence citing the candidate"}}"""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"in_role": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["in_role", "reason"],
    "additionalProperties": False,
}


def render_history(history: list[tuple[str, str]], turns: int, max_chars: int) -> str:
    """The last ``turns`` messages as ``USER:`` / ``ASSISTANT:`` blocks, each cut to ``max_chars``."""
    out = []
    for role, text in history[-turns:]:
        body = text if len(text) <= max_chars else text[:max_chars] + f"\n[… {len(text) - max_chars} more characters]"
        out.append(f"{role.upper()}:\n{body}")
    return "\n\n".join(out) or "(nothing yet)"


def parse_judge(content: str) -> Optional[dict]:
    """The judge's JSON object, tolerating text around it; None if there's none with a boolean ``in_role``."""
    start = content.find("{")
    if start < 0:
        return None
    try:
        obj = json.JSONDecoder().raw_decode(content[start:])[0]
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("in_role"), bool) else None


# ── The gate ──────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class UserSimGate:
    """How ``prompt_agent`` checks user-sim messages; see the module docstring."""

    patterns: bool = True
    judge_model: Optional[str] = None
    max_rewrites: int = 2
    # What happens when a message still fails after its rewrites: "fail" ends the conversation and fails the
    # step, "end" ends the conversation and lets the step finish, "send" sends it anyway (record only).
    on_fail: str = "fail"
    context_turns: int = 4
    # The judge alone fails only a message at least this long; a shorter one also needs a pattern. Swaps are long or
    # match a pattern, and without the scenario the judge misreads short scripted answers that correct the agent.
    judge_min_chars: int = 300
    context_chars: int = 3000
    judge_timeout_seconds: float = 60

    # How much of a judge error and of the judge's reason a verdict keeps.
    RECORDED_ERROR_CHARS: ClassVar[int] = 300
    RECORDED_REASON_CHARS: ClassVar[int] = 500

    def __post_init__(self) -> None:
        if self.on_fail not in ON_FAIL:
            raise ValueError(f"user_sim_gate.on_fail must be one of {ON_FAIL}, not {self.on_fail!r}")
        if self.max_rewrites < 0:
            raise ValueError("user_sim_gate.max_rewrites must be >= 0")
        if not self.patterns and not self.judge_model:
            raise ValueError("user_sim_gate needs patterns, a judge_model, or both")

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional[UserSimGate]:
        """None (no gate) for None; the defaults for {}, as for ``UserSimGate()``."""
        if data is None:
            return None
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"user_sim_gate: unknown keys {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    async def check(self, candidate: str, history: list[tuple[str, str]], *, call_config=None,
                    judge_unavailable: Optional[str] = None) -> GateVerdict:
        """Check one message. ``history`` is the conversation so far as ``(role, text)``, role user/assistant.
        A judge call that errors or can't be parsed doesn't fail the message: the patterns' verdict stands. So does a
        judge that couldn't be set up (``judge_unavailable``, the reason), which isn't called."""
        reasons = pattern_reasons(candidate) if self.patterns else []
        verdict = GateVerdict(passed=not reasons, pattern_reasons=reasons)
        if self.judge_model and judge_unavailable is not None:
            verdict.judge_error = judge_unavailable[: self.RECORDED_ERROR_CHARS]
            return verdict
        if self.judge_model:
            try:
                judged = await self._judge(candidate, history, call_config)
            except Exception as e:  # the judge is a second opinion; never let it take the conversation down
                logger.warning(f"user-sim gate judge call failed ({type(e).__name__}: {e}); using patterns only")
                verdict.judge_error = f"{type(e).__name__}: {e}"[: self.RECORDED_ERROR_CHARS]
                return verdict
            if judged is None:
                verdict.judge_error = "unparseable judge output"
                return verdict
            verdict.judge_in_role = judged["in_role"]
            verdict.judge_reason = str(judged.get("reason", ""))[: self.RECORDED_REASON_CHARS]
            if not judged["in_role"] and (reasons or len(candidate) >= self.judge_min_chars):
                verdict.passed = False
        return verdict

    async def _judge(self, candidate: str, history: list[tuple[str, str]], call_config) -> Optional[dict]:
        import litellm

        from agent_env.config.model import ModelParam

        prompt = _JUDGE_PROMPT.format(
            history=render_history(history, self.context_turns, self.context_chars),
            candidate=candidate[: self.context_chars * 2],
        )
        response = await litellm.acompletion(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.judge_timeout_seconds,
            **{ModelParam.RESPONSE_FORMAT: {"type": "json_schema",
                                            "json_schema": {"name": "user_sim_gate", "schema": _JUDGE_SCHEMA,
                                                            "strict": True}}},
            **(call_config.client_kwargs() if call_config is not None else {}),
        )
        return parse_judge(response.choices[0].message.content or "")


@dataclass
class GateVerdict:
    passed: bool
    pattern_reasons: list[str] = field(default_factory=list)
    judge_in_role: Optional[bool] = None
    judge_reason: Optional[str] = None
    judge_error: Optional[str] = None

    def reasons(self) -> list[str]:
        out = list(self.pattern_reasons)
        if self.judge_in_role is False and self.judge_reason:
            out.append(self.judge_reason)
        return out

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [])}


def rewrite_request(verdict: GateVerdict) -> str:
    """What the user-sim is told when its message was held back."""
    why = "; ".join(verdict.reasons()) or "it reads as the assistant's message"
    return (
        "[Conversation system note, not from the assistant.] Your last message was not sent to the assistant, "
        f"because it reads as the assistant's reply rather than yours: {why}. You are the user. The assistant does "
        "the work and reports it; you ask, answer what you were asked, decide, push back, or end. Write your next "
        "message to the assistant again, as yourself, in the same reply format as before."
    )


@dataclass
class GateOutcome:
    """How one user-sim message fared: the text to send (the last one checked), every verdict in order, and
    ``invalid`` when it still failed after the rewrites and ``on_fail`` isn't "send". ``stopped`` is set when the
    rewrite callback ended the exchange instead of returning a new message."""

    text: Optional[str]
    verdicts: list[GateVerdict] = field(default_factory=list)
    invalid: bool = False
    stopped: bool = False

    def reasons(self) -> list[str]:
        return self.verdicts[-1].reasons() if self.verdicts else []


async def run_gate(
    gate: UserSimGate,
    text: str,
    history: list[tuple[str, str]],
    rewrite: Callable[[str], Awaitable[Optional[str]]],
    *,
    call_config=None,
    judge_unavailable: Optional[str] = None,
    on_verdict: Optional[Callable[[int, str, GateVerdict], None]] = None,
) -> GateOutcome:
    """Check ``text``; while it fails, send the user-sim ``rewrite_request`` through ``rewrite`` and check its new
    message, up to ``gate.max_rewrites`` times. ``rewrite`` returns the new message, or None when the user side
    ended instead. ``judge_unavailable`` is why the judge couldn't be set up, if it couldn't (see ``check``).
    ``on_verdict(attempt, text, verdict)`` sees every check, for recording. Any conversation loop
    can call this; ``prompt_agent`` does, between a user-sim's reply and the agent's next turn."""
    outcome = GateOutcome(text=text)
    for attempt in range(gate.max_rewrites + 1):
        extra = {} if judge_unavailable is None else {"judge_unavailable": judge_unavailable}
        verdict = await gate.check(text, history, call_config=call_config, **extra)
        outcome.verdicts.append(verdict)
        if on_verdict is not None:
            on_verdict(attempt, text, verdict)
        if verdict.passed:
            return outcome
        if attempt == gate.max_rewrites:
            break
        new_text = await rewrite(rewrite_request(verdict))
        if new_text is None:
            outcome.stopped = True
            return outcome
        text = outcome.text = new_text
    outcome.invalid = gate.on_fail != "send"
    return outcome


def judge_call_config(gate: UserSimGate, user_overrides: dict):
    """The model call settings for the gate's judge, honoring a run's judge or LiteLLM overrides the way the
    rubrics verifier does; None without a judge."""
    if not gate.judge_model:
        return None
    from agent_env.store import get_config

    config = get_config()
    return config.resolve_model_call(
        gate.judge_model,
        default_api_key=config.resolve_reference(
            user_overrides.get("judge_litellm_api_key") or user_overrides.get("litellm_api_key")),
        base_override=user_overrides.get("judge_litellm_base_url") or user_overrides.get("litellm_base_url"),
    )
