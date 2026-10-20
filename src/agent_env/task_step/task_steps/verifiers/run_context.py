"""What a verifier knows of the run it grades beyond the env: the conversation and the trigger firing logs."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, ClassVar, Optional

from agent_env.a2a_agent import conversation_store
from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext


@dataclass(frozen=True)
class RunContext:
    """The prompt's messages and firing logs, oldest first; ``unavailable`` names what could not be read, and why."""

    ERROR_MAX_CHARS: ClassVar[int] = 200  # of a read error's message, in an ``unavailable`` reason

    conversation: list[dict] = field(default_factory=list)
    env_trigger_events: dict[str, list[dict]] = field(default_factory=dict)
    agent_trigger_events: list[dict] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return max((m["turn"] for m in self.conversation), default=0)

    def as_of_turn(self, turn: int) -> RunContext:
        """The run after ``turn``: its messages, and the events before the next turn's message (none if it's unknown)."""
        if self.conversation and turn >= self.turns:
            return self
        conversation = [m for m in self.conversation if m["turn"] <= turn]
        cutoff = next((m["ts"] for m in self.conversation if m["turn"] == turn + 1), None)
        if cutoff is None:
            why = f"withheld: when turn {turn + 1} began is unknown, so later events can't be told apart"
            withheld = [f"env_trigger_events.{env_id}" for env_id in self.env_trigger_events]
            withheld += ["agent_trigger_events"] if self.agent_trigger_events else []
            return replace(self, conversation=conversation, env_trigger_events={}, agent_trigger_events=[],
                           unavailable={**self.unavailable, **dict.fromkeys(withheld, why)})

        def before(events: list[dict]) -> list[dict]:
            return [e for e in events if _before(e.get("ts"), cutoff)]

        return replace(
            self,
            conversation=conversation,
            env_trigger_events={env_id: before(events) for env_id, events in self.env_trigger_events.items()},
            agent_trigger_events=before(self.agent_trigger_events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"conversation": self.conversation, "env_trigger_events": self.env_trigger_events,
                "agent_trigger_events": self.agent_trigger_events, "unavailable": self.unavailable}


async def load_run_context(context: TaskStepContext, prompt_response: Optional[PromptResponse]) -> RunContext:
    """The run context of ``prompt_response``'s prompt (env firing logs alone without one); never raises for a record."""
    unavailable: dict[str, str] = {}
    conversation: list[dict] = []
    agent_events: list[dict] = []
    if prompt_response is not None:
        conversation = await asyncio.to_thread(_conversation, context, prompt_response, unavailable)
        agent_events = _agent_trigger_events(context, prompt_response.step_id, conversation, unavailable)
    env_events = await asyncio.to_thread(_env_trigger_events, context, unavailable)
    run = RunContext(conversation=conversation, env_trigger_events=env_events, agent_trigger_events=agent_events,
                     unavailable=unavailable)
    return await asyncio.to_thread(_until_next_prompt, run, context, prompt_response) if prompt_response else run


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:RunContext.ERROR_MAX_CHARS]}"


def _agent_trigger_events(context: TaskStepContext, step_id: Optional[str], conversation: list[dict],
                          unavailable: dict[str, str]) -> list[dict]:
    """The user agent's firing log; unavailable, not empty, when it had triggers but ``prompt_agent`` read none back."""
    logs = context.metadata.get("agent_trigger_state") or {}
    if step_id in logs:
        return list(logs[step_id] or [])
    armed = {r.get("agent_name") for r in context.metadata.get("agent_trigger_registrations") or []}
    user_agent = next((m.get("author") for m in conversation if m["role"] == "user"), None)
    # Without the conversation the user agent is unknown, so any agent's registered triggers may be its.
    had_triggers = bool(armed) if "conversation" in unavailable else user_agent in armed
    if had_triggers or step_id in (context.metadata.get("agent_trigger_firings") or {}):
        unavailable["agent_trigger_events"] = "agent triggers were registered, but the user agent's log wasn't read"
    return []


def _conversation(context: TaskStepContext, prompt_response: PromptResponse, unavailable: dict[str, str]) -> list[dict]:
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
    messages: list[dict] = []
    turn = 0
    for stored in doc.get("messages") or []:
        role = stored.get("role")
        if role == "user":
            turn += 1
        message = _message(turn, role, stored.get("parts") or [], ts=stored.get("ts"), author=authors.get(role))
        if role == "user" and turn in fired:
            message["fired"] = fired[turn]
        if role == "user" and turn in fields:
            message["fields"] = fields[turn]
        messages.append(message)
    return messages


def _message(turn: int, role: str, parts: list[dict], *, ts: Optional[str] = None,
             author: Optional[str] = None) -> dict:
    text = "\n".join(p.get("text", "") for p in parts if p.get("kind") == "text")
    return {"turn": turn, "role": role, **({"author": author} if author else {}), "text": text, "parts": parts,
            "ts": ts}


def _env_trigger_events(context: TaskStepContext, unavailable: dict[str, str]) -> dict[str, list[dict]]:
    """Each env's firing log, from the envelope ``prompt_agent`` stored at the prompt's end."""
    events: dict[str, list[dict]] = {}
    for env_id, entry in (context.metadata.get("env_trigger_state") or {}).items():
        if not isinstance(entry, dict) or not entry.get("object_url"):
            unavailable[f"env_trigger_events.{env_id}"] = (entry or {}).get("error") or "no firing log was captured"
            continue
        try:
            envelope = json.loads(get_config().get_object_store().get(entry["object_url"]))
        except Exception as e:
            unavailable[f"env_trigger_events.{env_id}"] = _error(e)
            continue
        events[env_id] = list((envelope.get("state") or {}).get("events") or [])
    return events


def _until_next_prompt(run: RunContext, context: TaskStepContext, prompt_response: PromptResponse) -> RunContext:
    """The env firing logs are the whole run's (the last prompt's capture), so an earlier prompt's stop where the
    next prompt began; withheld when that start is unknown."""
    index = next((i for i, pr in enumerate(context.prompt_responses) if pr is prompt_response), None)
    if index is None or index + 1 >= len(context.prompt_responses) or not run.env_trigger_events:
        return run
    start = _first_message_ts(context, context.prompt_responses[index + 1])
    if start is None:
        why = "withheld: when the next prompt began is unknown, so its events can't be told apart"
        withheld = dict.fromkeys((f"env_trigger_events.{env_id}" for env_id in run.env_trigger_events), why)
        return replace(run, env_trigger_events={}, unavailable={**run.unavailable, **withheld})
    return replace(run, env_trigger_events={env_id: [e for e in events if _before(e.get("ts"), start)]
                                            for env_id, events in run.env_trigger_events.items()})


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
