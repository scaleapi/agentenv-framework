"""A check on every simulated-user message before prompt_agent sends it to the agent.

An LLM playing the user can slip into the assistant's role: it answers its own question, reports work as
done, or offers to do the agent's job. Once one such message is in the conversation the rest of the run
is no longer a test of the agent. The gate checks each user-sim message, asks the user-sim to rewrite one
that fails, and, if the rewrites fail too, ends the conversation and marks the run.

The judge decides:

- ``judge_model``: one model call that reads the user sim's instructions, the last turns and the candidate message,
  and says whether the message is written as the user, by the rule user_sim_judge's rogue check fails a run for
  ("spoke as the assistant"). Only the judge holds a message back.
- ``patterns``: phrases and shapes that belong to the assistant's side (a report of work done, a formatted
  deliverable, offers to do the work). Free and narrow; a match is recorded on the verdict, never a reason to hold a
  message back on its own. Without a judge the gate only records them.

Opt in with ``user_sim_gate`` on a ``prompt_agent`` step whose user is a user-sim agent-env deployed: a peer
user-sim, or the built-in one deploy_user_sim_agent deploys, whose rewrite is parsed as its own answer. ``always_on``
applies it to each of the built-in user-sim's answers. A step type that runs its own conversation loop applies the gate
with ``run_gate``, and sets ``USER_SIM_GATE_SUPPORTED``::

    {"judge_model": "<a small, fast model>", "max_rewrites": 2, "on_fail": "fail"}

A message that still fails after its rewrites, with ``on_fail`` "fail", marks the run ``error_type: user_sim_error``,
as a failed user sim does, so the run is redone.

Each judge call is in the step's usage, role ``judge``. In ``always_on`` the gate costs time budget even when it never
fires: each read reserves room for ``max_rewrites`` more user sim turns and, with a judge, a judge timeout per check.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, ClassVar, Optional

if TYPE_CHECKING:
    from agent_env.config.model_call import ModelReply

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
_NOW_LIVE = re.compile(r"\b(?:now live|is now (?:up|live))\b", re.I)
# A user says these about someone else's work too ("the invoice has been sent to me already"), so they count only on a
# message long enough to be a report.
_DELIVERED = re.compile(
    r"\b(?:(?:emails?|messages?|drafts?|replies|invites?|notes?) (?:sent|drafted|posted|scheduled)\b|incorporated\b|"
    r"(?:the|a|my) (?:cleaner|updated|revised|corrected|final) version\b|has been (?:updated|created|sent|posted))",
    re.I,
)
# Phrases the assistant says and a user doesn't, at any length. A user does say "I don't have access to that folder",
# "Should I send you the spreadsheet?", "I'll pull the numbers tonight", "I can send it to you", "I apologize for the
# confusion" and "send me the results", so those aren't here.
_AS_ASSISTANT = re.compile(
    r"\b(?:as an ai|i'?m an? (?:ai|assistant|language model)|my previous (?:answer|response|reply)|"
    r"i apologi[sz]e for my (?:previous|earlier|last) (?:answer|response|reply|message)|"
    r"let me know if you'?d like me to|would you like me to|want me to (?:draft|send|update|create|go ahead)|"
    r"standing by|ready when you are|parked until|"
    r"anything else (?:you(?:'d)? (?:want|like) (?:me )?to|to (?:cover|look at|dig into))|"
    r"what do you want (?:me )?to do next|corrections (?:noted|taken)|"
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


TALKS_LIKE_THE_ASSISTANT = "it talks like the assistant (offers to do the work, waits on the user, or speaks as the AI)"
NOW_LIVE = "it says something is now live"


_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


def pattern_reasons(text: str) -> list[str]:
    """Why ``text`` reads as the assistant's message, one reason per pattern that matched; empty if none. Curly
    apostrophes read as straight ones, as many models write them."""
    lim = PatternLimits
    text = text.translate(_APOSTROPHES)
    reasons = []
    n = len(text)
    if _AS_ASSISTANT.search(text):
        reasons.append(TALKS_LIKE_THE_ASSISTANT)
    if _REPORT_OPENING.search(text) and n > lim.REPORT_OPENING_MIN_CHARS:
        reasons.append("it opens like a report of work done")
    if _NOW_LIVE.search(text):
        reasons.append(NOW_LIVE)
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

Decide whether the CANDIDATE message is written as the user. It is NOT written as the user when it speaks as the \
assistant: a report or summary in the assistant's voice (findings it produced, work it says it did for the task), \
an offer to do the assistant's tasks ("Want me to send it to you?" offering the assistant's own work, "Standing by"), \
the person's own questions asked back as if it were the assistant, an apology for "my previous answer", or \
speaking as the AI.

It IS written as the user when it asks, instructs, approves, pushes back, answers a question the assistant asked, \
or says what the user knows, has, did or checked themselves, as the user's instructions describe this person, even \
if it is long, formatted, quotes the assistant's numbers, or pastes an email or document. "I checked my bank app \
and the charge is there" is the user reporting their own check. Length and markdown alone are not evidence.

Judge only whose voice the message is in. A user message that repeats an earlier one, ignores or contradicts what \
the assistant said, states a decision or a fact the user holds, or seems off topic is still written as the user.
{instructions}
CONVERSATION SO FAR (most recent last):
{history}
{user_note}
CANDIDATE (the user's next message):
<<<
{candidate}
>>>

Answer in JSON: {{"in_role": true or false, "reason": "one sentence citing the candidate"}}"""

# What a post-turn hook told the user it found (prompt_agent's turn.ask_user(note=...)): a user who says it is
# reporting their own check, not the assistant's work.
_USER_NOTE = """
WHAT THE USER FOUND checking the work themselves just now (only the user saw this, not the assistant). Saying it in \
the user's own words ("I checked the folder and the sheet isn't there") is written as the user:
<<<
{note}
>>>
"""

# What the model playing the user was told: who this person is and what they know, do and decide.
_INSTRUCTIONS = """
THE USER'S INSTRUCTIONS (what the model playing the user was told; the assistant never saw them):
<<<
{instructions}
>>>
"""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"in_role": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["in_role", "reason"],
    "additionalProperties": False,
}


def render_history(history: list[tuple[str, str]], turns: int, max_chars: int) -> str:
    """The last ``turns`` messages as ``USER:`` / ``ASSISTANT:`` blocks, each cut to ``max_chars``; none for 0."""
    out = []
    for role, text in (history[-turns:] if turns else []):  # history[-0:] would be all of it
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
    # What happens when a message still fails after its rewrites: "fail" ends the conversation as a failed user sim
    # (error_type user_sim_error; prompt_agent also fails the step, so its retry_config redoes it), "end" ends it
    # plainly, "send" sends it anyway (record only).
    on_fail: str = "fail"
    context_turns: int = 4
    context_chars: int = 3000
    judge_timeout_seconds: float = 60

    # How much of a judge error and of the judge's reason a verdict keeps.
    RECORDED_ERROR_CHARS: ClassVar[int] = 300
    RECORDED_REASON_CHARS: ClassVar[int] = 500
    # How much of the user sim's instructions the judge is shown.
    INSTRUCTIONS_MAX_CHARS: ClassVar[int] = 40_000

    def __post_init__(self) -> None:
        """Refuse a setting of the wrong type or range here, when the task loads, rather than mid-conversation."""
        def need(ok: bool, name: str, what: str) -> None:
            if not ok:
                raise ValueError(f"user_sim_gate.{name} must be {what}, not {getattr(self, name)!r}")

        def whole(v: object) -> bool:  # an int, and not a bool (which is one in Python)
            return isinstance(v, int) and not isinstance(v, bool)

        need(isinstance(self.patterns, bool), "patterns", "true or false")
        need(self.judge_model is None or (isinstance(self.judge_model, str) and bool(self.judge_model.strip())),
             "judge_model", "a model name or null")
        need(whole(self.max_rewrites) and self.max_rewrites >= 0, "max_rewrites", "a whole number >= 0")
        need(self.on_fail in ON_FAIL, "on_fail", f"one of {ON_FAIL}")
        need(whole(self.context_turns) and self.context_turns >= 0, "context_turns", "a whole number >= 0")
        need(whole(self.context_chars) and self.context_chars > 0, "context_chars", "a whole number > 0")
        need(isinstance(self.judge_timeout_seconds, (int, float)) and not isinstance(self.judge_timeout_seconds, bool)
             and self.judge_timeout_seconds > 0, "judge_timeout_seconds", "a number > 0")
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
                    user_note: Optional[str] = None, instructions: Optional[str] = None,
                    on_call: Optional[Callable[[ModelReply], None]] = None) -> GateVerdict:
        """Check one message. ``history`` is the conversation so far as ``(role, text)``, role user/assistant;
        ``instructions`` the user sim's own, shown to the judge. Only the judge fails a message; pattern matches are
        recorded. A judge call that errors or can't be parsed passes the message with the error recorded. ``on_call``
        sees each judge call's reply, for the step's usage."""
        verdict = GateVerdict(passed=True, pattern_reasons=pattern_reasons(candidate) if self.patterns else [])
        if not self.judge_model:
            return verdict
        try:
            extra = {k: v for k, v in (("user_note", user_note), ("instructions", instructions),
                                       ("on_call", on_call)) if v}
            judged = await self._judge(candidate, history, call_config, **extra)
        except Exception as e:  # never let the judge take the conversation down
            logger.warning(f"user-sim gate judge call failed ({type(e).__name__}: {e}); the message is sent")
            verdict.judge_error = f"{type(e).__name__}: {e}"[: self.RECORDED_ERROR_CHARS]
            return verdict
        if judged is None:
            verdict.judge_error = "unparseable judge output"
            return verdict
        verdict.judge_in_role = judged["in_role"]
        verdict.judge_reason = str(judged.get("reason", ""))[: self.RECORDED_REASON_CHARS]
        verdict.passed = judged["in_role"]
        return verdict

    async def _judge(self, candidate: str, history: list[tuple[str, str]], route,
                     user_note: Optional[str] = None, instructions: Optional[str] = None,
                     on_call: Optional[Callable[[ModelReply], None]] = None) -> Optional[dict]:
        """One judge call through core's one model-call path (config/model_call.py), on ``route``
        (``judge_call_config``'s), or on the configured endpoint when none was given. A call cut off by its timeout or
        a cancel may still be billed, so ``on_call`` sees it as a reply of unknown usage, marking the usage a lower
        bound."""
        from agent_env.config import model_call
        from agent_env.config.model_pricing import Cost, CostStatus, Unpriced

        prompt = _JUDGE_PROMPT.format(
            history=render_history(history, self.context_turns, self.context_chars),
            candidate=candidate[: self.context_chars * 2],
            user_note=_USER_NOTE.format(note=user_note.strip()) if user_note else "",
            instructions=(_INSTRUCTIONS.format(instructions=instructions.strip()[: self.INSTRUCTIONS_MAX_CHARS])
                          if instructions else ""),
        )
        if route is None:
            route = judge_call_config(self, {})
        try:
            reply = await model_call.complete(
                route, [{"role": "user", "content": prompt}], timeout=self.judge_timeout_seconds,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "user_sim_gate", "schema": _JUDGE_SCHEMA, "strict": True}})
        except BaseException as exc:
            if on_call is not None and model_call.cut_off(exc):
                on_call(model_call.ModelReply("", route.model, route.endpoint_host, None,
                                              Cost(None, CostStatus.UNPRICED, None, None, unpriced=Unpriced.NO_TOKENS)))
            raise
        if on_call is not None:
            on_call(reply)
        return parse_judge(reply.content)


@dataclass
class GateVerdict:
    passed: bool
    pattern_reasons: list[str] = field(default_factory=list)
    judge_in_role: Optional[bool] = None
    judge_reason: Optional[str] = None
    judge_error: Optional[str] = None

    def reasons(self) -> list[str]:
        """Why the judge held the message back; empty for one it passed."""
        return [self.judge_reason or "it reads as the assistant's message"] if self.judge_in_role is False else []

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
    on_verdict: Optional[Callable[[int, str, GateVerdict], None]] = None,
    user_note: Optional[str] = None,
    instructions: Optional[str] = None,
    on_call: Optional[Callable[[ModelReply], None]] = None,
) -> GateOutcome:
    """Check ``text``; while it fails, send the user-sim ``rewrite_request`` through ``rewrite`` and check its new
    message, up to ``gate.max_rewrites`` times. ``rewrite`` returns the new message, or None when the user side
    ended instead.
    ``on_verdict(attempt, text, verdict)`` sees every check, for recording. ``user_note`` is what a post-turn hook
    told the user sim it found, and ``instructions`` the user sim's own, both shown to the judge. ``on_call`` sees each
    judge call's reply, for the step's usage.
    Any conversation loop
    can call this; ``prompt_agent`` does, between a user-sim's reply and the agent's next turn."""
    outcome = GateOutcome(text=text)
    for attempt in range(gate.max_rewrites + 1):
        extra = {}
        if user_note:  # what a post-turn hook told the user sim it found: the judge reads a report of it as the user's
            extra["user_note"] = user_note
        if instructions:
            extra["instructions"] = instructions
        if on_call is not None:
            extra["on_call"] = on_call
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


JUDGE_KEY_HOW = "judge_litellm_api_key, or litellm_api_key, in the run's overrides"


def judge_call_config(gate: UserSimGate, user_overrides: dict):
    """The route of the gate's judge calls (a ``model_call.ModelRoute``), honoring a run's judge or LiteLLM
    overrides as the rubrics verifier does; None without a judge. A judge that can't be set up (an endpoint the run
    gives no key for) raises, failing the step: without its judge the gate holds nothing back."""
    if not gate.judge_model:
        return None
    from agent_env.config import model_call

    try:
        return model_call.resolve_route(
            gate.judge_model,
            api_key=user_overrides.get("judge_litellm_api_key") or user_overrides.get("litellm_api_key"),
            api_base=user_overrides.get("judge_litellm_base_url") or user_overrides.get("litellm_base_url"),
            who="user_sim_gate", key_how=JUDGE_KEY_HOW)
    except Exception as e:
        raise RuntimeError(f"user_sim_gate: judge setup: {type(e).__name__}: {e}; only the judge holds a message "
                           "back, so the step fails rather than run unchecked") from e
