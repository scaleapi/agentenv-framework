"""What a verifier knows of the run it grades beyond the env: the conversation and the trigger firings."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, ClassVar, Literal, Mapping, Optional

from agent_env.a2a_agent import conversation_store
from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext


class RunRecordUnavailable(LookupError):
    """A query needs a record that could not be read: ``source`` is "conversation" or a ``Firing.source``."""

    def __init__(self, source: str, reason: str):
        super().__init__(f"{source}: {reason}")
        self.source, self.reason = source, reason


def _need(unavailable: Mapping[str, str], sources) -> None:
    for source in sources:
        if source in unavailable:
            raise RunRecordUnavailable(source, unavailable[source])


@dataclass(frozen=True)
class Message:
    turn: int
    role: Literal["user", "agent"]
    text: str
    sent_at: Optional[datetime]  # None only for an episode's turn, which has no stored conversation
    author: Optional[str] = None
    written_by_triggers: tuple[str, ...] = ()  # the user agent's triggers that wrote this user message
    user_sim: Optional[Mapping[str, Any]] = None  # the user sim's extra fields for this turn
    parts: tuple[Mapping[str, Any], ...] = ()
    env_clock: Optional[Mapping[str, Optional[datetime]]] = None  # each env's virtual time when it was recorded


@dataclass(frozen=True)
class Firing:
    trigger_id: str
    source: str  # "env:<env_id>" or "user_agent"
    turn: int  # the turn it fell in; 0 = before the first message
    at: datetime
    virtual_time: Optional[datetime]
    ok: bool  # fired, rather than failed
    provoked_by: Optional[Mapping[str, Any]]  # the env call that set it off, from its ``detected`` event


@dataclass(frozen=True)
class Turn:
    number: int
    message: Message  # the user-side message that opened it
    reply: Optional[Message]
    _firings: tuple[Firing, ...] = field(default=(), repr=False)
    _unavailable: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def firings(self) -> tuple[Firing, ...]:
        """What fired during this turn; raises ``RunRecordUnavailable`` if a firing log is missing."""
        _need(self._unavailable, self._unavailable)
        return self._firings


@dataclass(frozen=True)
class Run:
    """The graded prompt's turns and every trigger firing; a query that needs a missing record raises."""

    ERROR_MAX_CHARS: ClassVar[int] = 200  # of a read error's message, in an ``unavailable`` reason

    _turns: tuple[Turn, ...] = ()
    _firings: tuple[Firing, ...] = ()  # oldest first
    _logs: tuple[str, ...] = ()  # the firing logs that were read, as ``Firing.source`` names them
    unavailable: Mapping[str, str] = field(default_factory=dict)  # what could not be read, and why

    @property
    def turns(self) -> tuple[Turn, ...]:
        _need(self.unavailable, ["conversation"])
        return self._turns

    def turn(self, number: int) -> Turn:
        if not 1 <= number <= len(self.turns):
            raise LookupError(f"the run has no turn {number}")
        return self._turns[number - 1]

    def firings(self, trigger_id: Optional[str] = None, *, env: Optional[str] = None) -> tuple[Firing, ...]:
        """Oldest first, of ``trigger_id`` and ``env`` when given; raises ``RunRecordUnavailable`` for a missing log."""
        source = None if env is None else f"env:{env}"
        _need(self.unavailable, [source] if source else self.unavailable)
        if source and source not in self._logs:
            raise RunRecordUnavailable(source, "no firing log was captured")
        return tuple(f for f in self._firings
                     if (trigger_id is None or f.trigger_id == trigger_id) and (source is None or f.source == source))

    def firing(self, trigger_id: str, *, env: Optional[str] = None) -> Optional[Firing]:
        """The first firing of ``trigger_id``, or None if it never fired."""
        return next(iter(self.firings(trigger_id, env=env)), None)

    def as_of(self, turn: int) -> Run:
        """The run after ``turn``: its turns, and what fired before the next turn began."""
        if turn < 0:
            raise ValueError(f"turn must be >= 0, got {turn}")
        if "conversation" in self.unavailable or turn >= len(self._turns):
            return self
        return replace(self, _turns=self._turns[:turn], _firings=tuple(f for f in self._firings if f.turn <= turn))


def render_run_context(run: Run, parts: list[str], *, message_max_chars: int, firings_max: int) -> str:
    """The judge-prompt sections for ``parts`` of ``run``: the conversation by turn, then one firing a line."""
    sections: list[str] = []
    if "conversation" in parts:
        lines = ["## Conversation", "The run's messages, turn by turn: the user side (the task's prompt, then each "
                 "later message of a user simulator, a human or the triggers) and the agent's replies."]
        if "conversation" in run.unavailable:
            lines.append(f"(not available: {run.unavailable['conversation']})")
        for turn in run._turns:
            lines.append(f"### Turn {turn.number}")
            lines += [_render_message(m, message_max_chars) for m in (turn.message, turn.reply) if m]
        sections.append("\n".join(lines))
    if "trigger_log" in parts:
        lines = ["## Trigger firings", "Each time an env's trigger or the user agent's trigger fired or failed, "
                 "oldest first, with the turn it fell in."]
        if len(run._firings) > firings_max:
            lines.append(f"({len(run._firings) - firings_max} earlier firings not shown)")
        lines += [_render_firing(f) for f in run._firings[max(len(run._firings) - firings_max, 0):]]
        lines += [f"(not available: {what}: {why})" for what, why in run.unavailable.items() if what != "conversation"]
        if len(lines) == 2:
            lines.append("(no trigger firings were recorded)")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _render_message(m: Message, max_chars: int) -> str:
    fired = f"written by triggers {', '.join(m.written_by_triggers)}" if m.written_by_triggers else None
    who = ", ".join(filter(None, [m.author, fired]))
    files = sum(1 for p in m.parts if p.get("kind") != "text")
    extra = f" (+{files} non-text part{'s' if files != 1 else ''})" if files else ""
    text = m.text if len(m.text) <= max_chars else m.text[:max_chars] + "…"
    return f"- {m.role}{f' ({who})' if who else ''}: {text}{extra}"


def _render_firing(f: Firing) -> str:
    when = f"during turn {f.turn}" if f.turn else "before the first message"
    virtual = f", virtual time {f.virtual_time.isoformat()}" if f.virtual_time else ""
    provoked = f", provoked by {json.dumps(f.provoked_by, default=str)}" if f.provoked_by else ""
    return f"- {f.source}: {f.trigger_id} {'fired' if f.ok else 'failed'} {when}{virtual}{provoked}"


async def load_run_context(context: TaskStepContext, prompt_response: Optional[PromptResponse]) -> Run:
    """The ``Run`` of ``prompt_response``'s prompt (env firings alone without one); never raises for a record."""
    unavailable: dict[str, str] = {}
    conversation: list[Message] = []
    agent_events: list[dict] = []
    if prompt_response is not None:
        conversation = await asyncio.to_thread(_conversation, context, prompt_response, unavailable)
        agent_events = _agent_trigger_events(context, prompt_response.step_id, conversation, unavailable)
    env_events = await asyncio.to_thread(_env_trigger_events, context, unavailable)
    if prompt_response is not None:
        env_events = await asyncio.to_thread(_until_next_prompt, env_events, context, prompt_response, unavailable)
    return _run(conversation, {**{f"env:{e}": events for e, events in env_events.items()}, "user_agent": agent_events},
                unavailable)


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:Run.ERROR_MAX_CHARS]}"


def _time(ts: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(ts) if ts else None


def _run(conversation: list[Message], logs: dict[str, list[dict]], unavailable: dict[str, str]) -> Run:
    """Places every firing in its turn; without the conversation the firing logs are withheld, as they can't be."""
    if "conversation" in unavailable:
        why = "withheld: without the conversation, its firings can't be placed in turns"
        unavailable.update({source: why for source, events in logs.items() if events and source not in unavailable})
    starts = [m.sent_at for m in conversation if m.role == "user"]
    firings: list[Firing] = []
    for source, events in logs.items():
        if source in unavailable:
            continue
        try:
            firings += _fold(source, events, starts)
        except (KeyError, TypeError, ValueError) as e:
            unavailable[source] = f"unreadable firing log: {_error(e)}"
    firings.sort(key=lambda f: f.at)
    opened = [m for m in conversation if m.role == "user"]
    replies = {m.turn: m for m in reversed(conversation) if m.role == "agent"}
    turns = tuple(Turn(number=m.turn, message=m, reply=replies.get(m.turn), _unavailable=unavailable,
                       _firings=tuple(f for f in firings if f.turn == m.turn)) for m in opened)
    return Run(_turns=turns, _firings=tuple(firings), _logs=tuple(s for s in logs if s not in unavailable),
               unavailable=unavailable)


def _fold(source: str, events: list[dict], starts: list[Optional[datetime]]) -> list[Firing]:
    """One ``Firing`` per ``fired`` or ``failed`` event, provoked by its trigger's oldest unclaimed ``detected``."""
    detected: dict[str, deque] = defaultdict(deque)
    provoked: dict[str, Any] = {}  # a time trigger's one ``detected`` provokes each of its marks' fires
    firings = []
    for event in events:
        trigger_id, kind = event.get("trigger_id"), event.get("kind")
        if trigger_id and kind == "detected":
            detected[trigger_id].append(event.get("provoking"))
        elif trigger_id and kind in ("fired", "failed"):
            pending = detected[trigger_id]
            provoked[trigger_id] = pending.popleft() if pending else provoked.get(trigger_id)
            at = datetime.fromisoformat(event["ts"])
            firings.append(Firing(trigger_id=trigger_id, source=source, at=at, ok=kind == "fired",
                                  turn=sum(1 for s in starts if s is None or s <= at),
                                  virtual_time=_time(event.get("virtual_time")), provoked_by=provoked[trigger_id]))
    return firings


def _agent_trigger_events(context: TaskStepContext, step_id: Optional[str], conversation: list[Message],
                          unavailable: dict[str, str]) -> list[dict]:
    """The user agent's firing log; unavailable, not empty, when it had triggers but ``prompt_agent`` read none back."""
    logs = context.metadata.get("agent_trigger_state") or {}
    if step_id in logs:
        log = list(logs[step_id] or [])
        first = log[0].get("seq") if log and isinstance(log[0], dict) else None
        if isinstance(first, int) and first > 1:  # the user agent's capped log keeps its newest entries
            unavailable["user_agent"] = f"incomplete: the user agent's log lost its oldest {first - 1} entries"
            return []
        return log
    armed = {r.get("agent_name") for r in context.metadata.get("agent_trigger_registrations") or []}
    user_agent = next((m.author for m in conversation if m.role == "user"), None)
    # Without the conversation the user agent is unknown, so any agent's registered triggers may be its.
    had_triggers = bool(armed) if "conversation" in unavailable else user_agent in armed
    if had_triggers or step_id in (context.metadata.get("agent_trigger_firings") or {}):
        unavailable["user_agent"] = "agent triggers were registered, but the user agent's log wasn't read"
    return []


def _conversation(context: TaskStepContext, prompt_response: PromptResponse,
                  unavailable: dict[str, str]) -> list[Message]:
    """The stored conversation, numbered by user message; without one (an episode's turn), its prompt and reply."""
    conversation_id = (context.metadata.get("a2a_conversations") or {}).get(prompt_response.step_id)
    if conversation_id is None:
        return [_message(1, "user", [{"kind": "text", "text": prompt_response.prompt_text or ""}]),
                _message(1, "agent", [{"kind": "text", "text": prompt_response.response}])]
    try:
        doc = conversation_store.get_conversation(conversation_id)
    except Exception as e:
        unavailable["conversation"] = _error(e)
        return []
    if doc is None:
        unavailable["conversation"] = f"conversation {conversation_id!r} is not in the store"
        return []
    step_id = prompt_response.step_id
    fired = {f["turn"] + 1: f.get("fired") for f in (context.metadata.get("agent_trigger_firings") or {}).get(step_id) or []
             if f.get("fired")}
    fields = {f["turn"] + 1: f.get("fields") for f in (context.metadata.get("usersim_turn_outputs") or {}).get(step_id) or []}
    authors = {"user": doc.get("source_agent_name"), "agent": doc.get("target_agent_name")}
    messages: list[Message] = []
    turn = 0
    try:
        own = _own_messages(context, prompt_response, conversation_id, doc.get("messages") or [])
        if own is None:
            unavailable["conversation"] = ("withheld: it is shared with another prompt, and where this prompt's "
                                           "messages begin or end wasn't found")
            return []
        for stored in own:
            role = stored.get("role")
            if role == "user":
                turn += 1
            if role in ("user", "agent") and turn:
                user = role == "user"
                messages.append(_message(turn, role, stored.get("parts") or [], ts=stored.get("ts"),
                                         author=authors.get(role), fired=fired.get(turn) if user else None,
                                         user_sim=fields.get(turn) if user else None,
                                         env_clock=stored.get("env_clock")))
    except (TypeError, ValueError) as e:
        unavailable["conversation"] = f"unreadable conversation: {_error(e)}"
        return []
    return messages


def _own_messages(context: TaskStepContext, prompt_response: PromptResponse, conversation_id: str,
                  stored: list[dict]) -> Optional[list[dict]]:
    """``stored`` from the prompt's first message to the next prompt's sharing it; None if either isn't found."""
    prompts = context.prompt_responses
    index = next((i for i, pr in enumerate(prompts) if pr is prompt_response), None)
    if index is None:
        return stored
    conversations = context.metadata.get("a2a_conversations") or {}
    sharing = [i for i, pr in enumerate(prompts)
               if pr.step_id != prompt_response.step_id and conversations.get(pr.step_id) == conversation_id]
    following = next((prompts[i] for i in sharing if i > index), None)
    start = _first_message_index(prompt_response, stored) if any(i < index for i in sharing) else 0
    end = len(stored) if following is None else _first_message_index(following, stored)
    if start is None or end is None or end <= start:
        return None
    return stored[start:end]


def _first_message_index(prompt_response: PromptResponse, stored: list[dict]) -> Optional[int]:
    """Where in ``stored`` the prompt's first message is, when exactly one user message matches it."""
    parts = (prompt_response.source_agent_per_turn_prompt_parts or [None])[0]
    if parts is None and prompt_response.prompt_text is not None:
        parts = [{"kind": "text", "text": prompt_response.prompt_text}]
    found = [i for i, m in enumerate(stored) if parts and m.get("role") == "user" and m.get("parts") == parts]
    return found[0] if len(found) == 1 else None


def _message(turn: int, role: str, parts: list[dict], *, ts: Optional[str] = None, author: Optional[str] = None,
             fired: Optional[list[str]] = None, user_sim: Optional[Mapping[str, Any]] = None,
             env_clock: Optional[Mapping[str, Optional[str]]] = None) -> Message:
    text = "\n".join(p.get("text", "") for p in parts if p.get("kind") == "text")
    clock = None if env_clock is None else {env_id: _time(t) for env_id, t in env_clock.items()}
    return Message(turn=turn, role=role, text=text, sent_at=_time(ts), author=author or None,
                   written_by_triggers=tuple(fired or ()), user_sim=user_sim, parts=tuple(parts), env_clock=clock)


def _env_trigger_events(context: TaskStepContext, unavailable: dict[str, str]) -> dict[str, list[dict]]:
    """Each env's firing log, from the envelope ``prompt_agent`` stored at the prompt's end."""
    events: dict[str, list[dict]] = {}
    for env_id, entry in (context.metadata.get("env_trigger_state") or {}).items():
        if not isinstance(entry, dict) or not entry.get("object_url"):
            unavailable[f"env:{env_id}"] = (entry or {}).get("error") or "no firing log was captured"
            continue
        try:
            envelope = json.loads(get_config().get_object_store().get(entry["object_url"]))
        except Exception as e:
            unavailable[f"env:{env_id}"] = _error(e)
            continue
        state = envelope.get("state") or {}
        if state.get("events_dropped"):
            unavailable[f"env:{env_id}"] = f"incomplete: the gateway dropped {state['events_dropped']} of its events"
            continue
        events[env_id] = list(state.get("events") or [])
    return events


def _until_next_prompt(env_events: dict[str, list[dict]], context: TaskStepContext, prompt_response: PromptResponse,
                       unavailable: dict[str, str]) -> dict[str, list[dict]]:
    """An earlier prompt's env firing logs, cut where the next prompt began; withheld when that start is unknown."""
    index = next((i for i, pr in enumerate(context.prompt_responses) if pr is prompt_response), None)
    if index is None or index + 1 >= len(context.prompt_responses) or not env_events:
        return env_events
    following = context.prompt_responses[index + 1]
    conversations = context.metadata.get("a2a_conversations") or {}
    # Prompts sharing one conversation (one context_id) leave no boundary in it to cut at.
    shared = conversations.get(following.step_id) == conversations.get(prompt_response.step_id)
    start = None if shared else _first_message_ts(context, following)
    if start is None:
        why = "withheld: when the next prompt began is unknown, so its events can't be told apart"
        unavailable.update(dict.fromkeys((f"env:{env_id}" for env_id in env_events), why))
        return {}
    return {env_id: [e for e in events if _before(e.get("ts"), start)] for env_id, events in env_events.items()}


def _first_message_ts(context: TaskStepContext, prompt_response: PromptResponse) -> Optional[str]:
    conversation_id = (context.metadata.get("a2a_conversations") or {}).get(prompt_response.step_id)
    try:
        doc = conversation_store.get_conversation(conversation_id) if conversation_id else None
    except Exception:
        return None
    messages = (doc or {}).get("messages") or []
    return messages[0].get("ts") if messages else None


def _before(ts: Optional[str], cutoff: Optional[str]) -> bool:
    if ts is None or cutoff is None:
        return True
    return datetime.fromisoformat(ts) < datetime.fromisoformat(cutoff)
