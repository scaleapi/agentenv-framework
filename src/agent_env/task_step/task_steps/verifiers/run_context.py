"""The run a verifier grades, beyond the env's state: the conversation, every user-side turn included, and the
trigger firing logs.

``load_run_context`` reads it from what the run recorded: the conversation ``prompt_agent`` stored for the prompt
(``metadata["a2a_conversations"]``), the env firing logs it captured at the prompt's end
(``metadata["env_trigger_state"]``), and the user agent's own firing log (``metadata["agent_trigger_state"]``).
``RunContext.as_of_turn`` cuts it the way ``grade_up_to_turn`` cuts the trajectory.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Optional

from agent_env.a2a_agent import conversation_store
from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext


@dataclass(frozen=True)
class RunContext:
    """``conversation``: the prompt's messages, oldest first, each ``{"turn", "role", "text", "parts", "ts"}`` with
    ``author`` (who sent it, when recorded), and on a user message the agent triggers that wrote it (``fired``) and
    the extra fields the user-sim returned with it (``fields``). ``env_trigger_events``: each env's gateway firing
    log, oldest first. ``agent_trigger_events``: the user agent's firing log. ``unavailable`` names what the run
    meant to record but could not be read, and why, so a verifier can tell a missing log from an empty one."""

    conversation: list[dict] = field(default_factory=list)
    env_trigger_events: dict[str, list[dict]] = field(default_factory=dict)
    agent_trigger_events: list[dict] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return max((m["turn"] for m in self.conversation), default=0)

    def as_of_turn(self, turn: int) -> RunContext:
        """The run as it stood after ``turn``: its messages up to that turn's reply, and the firing-log events from
        before the next turn's user message (all of them when no turn follows)."""
        if turn >= self.turns:
            return self
        cutoff = next(m["ts"] for m in self.conversation if m["turn"] == turn + 1)

        def before(events: list[dict]) -> list[dict]:
            return [e for e in events if _before(e.get("ts"), cutoff)]

        return replace(
            self,
            conversation=[m for m in self.conversation if m["turn"] <= turn],
            env_trigger_events={env_id: before(events) for env_id, events in self.env_trigger_events.items()},
            agent_trigger_events=before(self.agent_trigger_events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"conversation": self.conversation, "env_trigger_events": self.env_trigger_events,
                "agent_trigger_events": self.agent_trigger_events, "unavailable": self.unavailable}


async def load_run_context(context: TaskStepContext, prompt_response: Optional[PromptResponse]) -> RunContext:
    """The run context of ``prompt_response``'s prompt (the env firing logs alone without one). Never raises for a
    record it cannot read; that lands in ``unavailable``."""
    unavailable: dict[str, str] = {}
    conversation: list[dict] = []
    agent_events: list[dict] = []
    if prompt_response is not None:
        conversation = await asyncio.to_thread(_conversation, context, prompt_response, unavailable)
        agent_events = list((context.metadata.get("agent_trigger_state") or {}).get(prompt_response.step_id) or [])
    env_events = await asyncio.to_thread(_env_trigger_events, context, unavailable)
    return RunContext(conversation=conversation, env_trigger_events=env_events, agent_trigger_events=agent_events,
                      unavailable=unavailable)


def _conversation(context: TaskStepContext, prompt_response: PromptResponse, unavailable: dict[str, str]) -> list[dict]:
    """The stored conversation, numbered by user message; a prompt with none recorded (an episode's turn) is one turn
    of its prompt and reply."""
    conversation_id = (context.metadata.get("a2a_conversations") or {}).get(prompt_response.step_id)
    if conversation_id is None:
        return [_message(1, "user", [{"kind": "text", "text": prompt_response.prompt_text or ""}]),
                _message(1, "agent", [{"kind": "text", "text": prompt_response.response}])]
    try:
        doc = conversation_store.get_conversation(conversation_id)
    except Exception as e:
        unavailable["conversation"] = f"{type(e).__name__}: {str(e)[:200]}"
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
            unavailable[f"env_trigger_events.{env_id}"] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        events[env_id] = list((envelope.get("state") or {}).get("events") or [])
    return events


def _before(ts: Optional[str], cutoff: Optional[str]) -> bool:
    if ts is None or cutoff is None:
        return True
    return datetime.fromisoformat(ts) < datetime.fromisoformat(cutoff)
