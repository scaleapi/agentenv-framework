"""How what came due on an env's notifications/v1 reaches an agent: the one place that says what a delivered line reads
like, for prompt_agent after each turn and for any post-turn hook that composes its own message.

Each item becomes one text part, labelled with when and through what, so nothing poses as the person:

    [Mon Feb 1, 3:00 PM · your scheduled check] See if Dana replied about the 2BR on Alder St
    [Mon Feb 1, 3:10 PM · Curbfinch alert] New listing: 2BR on Ivy St, $3,050/mo

and carries ``metadata.agentenv`` naming its source, so a verifier reading the stored conversation can tell a
delivered line from what the person wrote in the same message.

How prompt_agent delivers, after each agent turn:
- The user side answers first, as without notifications: a post-turn hook, else agent triggers, then the user sim or
  a human. Delivery changes neither who is asked nor what they see; the user sim only receives the agent's replies.
- An answer that ends the conversation (the user sim's done, an agent trigger's end, a hook's end), the last turn
  and a failed turn take nothing: what is due stays queued, and shows in notifications/v1 ``state``.
- Otherwise it takes what came due for its agent's role (``DeployedAgent.role``, else ``default``) on every env
  whose card offers notifications/v1 ``take``, and the next message is those lines, then the reply as it was.
- A reply that says nothing (a hook's ``{"message": []}``) sends only the lines; with none due either, the
  conversation ends (``nothing_to_send``). With nothing due, every message is what it was before.
- A hook that takes for itself leaves nothing here to repeat, since ``take`` hands each item out once.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from agent_env.env.gateway.constants import EXT_NOTIFICATIONS_URI

DEFAULT_TIME_ZONE = "UTC"
_TAKE_ATTEMPTS = 3
_TAKE_RETRY_BASE_DELAY_S = 0.5  # seconds; the nth retry waits n times this
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


def label_zone(name: Optional[str]) -> ZoneInfo:
    """The IANA zone delivered lines show their times in; UTC when none is named."""
    try:
        return ZoneInfo(name or DEFAULT_TIME_ZONE)
    except Exception as e:  # ZoneInfoNotFoundError, or a ValueError for a malformed key
        raise ValueError(f"time_zone must be an IANA name such as 'America/Los_Angeles', got {name!r}") from e


def label_time(at: Optional[str], zone: ZoneInfo) -> str:
    """'Mon Feb 1, 3:00 PM' in ``zone``; 'now' for an item with no time (queued while the env clock was unset)."""
    if not at:
        return "now"
    local = _utc(at).astimezone(zone)
    return f"{local.strftime('%a %b')} {local.day}, {local.strftime('%I:%M %p').lstrip('0')}"


def delivered_line(item: dict, zone: ZoneInfo) -> str:
    """One taken item as the agent reads it."""
    when = label_time(item.get("at"), zone)
    if item["kind"] == "scheduled_check":
        times = item.get("times", 1)
        again = f" (it came due {times} times while you were away)" if times > 1 else ""
        return f"[{when} · your scheduled check] {item['note']}{again}"
    return f"[{when} · {item['channel']}] {item['text']}"


def delivered_parts(items: Iterable[dict], zone: ZoneInfo) -> list[dict]:
    """One text part per taken item, oldest first, each with ``metadata.agentenv``: ``{"source": "scheduled_check",
    "check_id", "times", "at"}`` or ``{"source": "trigger", "id", "trigger_id", "channel", "at"}``, plus the
    ``env_id`` it came from when the item carries one."""
    parts = []
    for item in sorted(items, key=_oldest_first):
        if item["kind"] == "scheduled_check":
            source = {"source": "scheduled_check", "check_id": item["check_id"], "times": item.get("times", 1)}
        else:
            source = {"source": "trigger", "id": item["id"], "trigger_id": item.get("trigger_id"),
                      "channel": item["channel"]}
        meta = {**source, "at": item.get("at"), **({"env_id": item["env_id"]} if item.get("env_id") else {})}
        parts.append({"kind": "text", "text": delivered_line(item, zone), "metadata": {"agentenv": meta}})
    return parts


def next_message(delivered: list[dict], reply: list[dict]) -> list[dict]:
    """The agent's next message: what was delivered first, then the user side's reply as it is. A reply that says
    nothing (no parts, or only blank text) adds nothing to delivered lines."""
    if delivered and says_nothing(reply):
        return list(delivered)
    return [*delivered, *reply]


def says_nothing(parts: list[dict]) -> bool:
    return all(p.get("kind") == "text" and not str(p.get("text") or "").strip() for p in parts)


async def take_due(envs: Iterable[Any], to: str) -> list[dict]:
    """Take what has come due for the role ``to`` from each env offering notifications/v1 ``take``, oldest first,
    each item tagged with its ``env_id``. ``take`` hands each item out once, so a caller that takes first (a hook)
    leaves nothing here to repeat. Each env's take carries one take_id across its retries, so a retry after a lost
    response gets the same batch back. An env that keeps failing raises, after every env has been asked: the step
    fails rather than carry on with a due item silently dropped."""
    envs = [env for env in envs if env.supports(EXT_NOTIFICATIONS_URI, "take")]
    taken = await asyncio.gather(*(_take(env, to) for env in envs), return_exceptions=True)
    failed = [(env, error) for env, error in zip(envs, taken) if isinstance(error, BaseException)]
    if failed:
        raise RuntimeError("; ".join(str(error) for _, error in failed)) from failed[0][1]
    items = [{**item, "env_id": env.env_id} for env, answer in zip(envs, taken) for item in answer.get("items") or []]
    return sorted(items, key=_oldest_first)


async def _take(env: Any, to: str) -> dict:
    last: Optional[Exception] = None
    body = {"to": to, "take_id": uuid.uuid4().hex}
    for attempt in range(_TAKE_ATTEMPTS):
        try:
            return await env.invoke(EXT_NOTIFICATIONS_URI, "take", body)
        except Exception as e:
            last = e
            if attempt + 1 < _TAKE_ATTEMPTS:
                await asyncio.sleep(_TAKE_RETRY_BASE_DELAY_S * (attempt + 1))
    raise RuntimeError(f"could not take what came due for role '{to}' from env '{env.env_id}': {last}")


def _oldest_first(item: dict) -> tuple[bool, datetime]:
    at = item.get("at")
    return (at is not None, _utc(at) if at else _EARLIEST)


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
