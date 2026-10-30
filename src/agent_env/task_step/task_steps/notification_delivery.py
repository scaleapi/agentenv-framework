"""How what came due on an env's notifications/v1 reaches an agent: the one place that says what a delivered line reads
like, for prompt_agent after each turn and for any post-turn hook that composes its own message.

Every item ``take`` returns has one shape (``id``, ``to``, ``at``, ``times``, ``source``, ``source_id``, ``channel``,
``parts``), so each is delivered the same way, whatever its source: its text labelled with when and through what
(``channel``), so nothing poses as the person, and "came due N times" when ``times`` is more than one:

    [Mon Feb 1, 3:00 PM UTC · your scheduled check] See if Dana replied about the 2BR on Alder St
    [Mon Feb 1, 3:10 PM UTC · Curbfinch alert] New listing: 2BR on Ivy St, $3,050/mo

Times are in UTC, and say so, like everything else the agent reads about time (``get_time``, ``schedule_check``).

Each delivered part carries ``metadata.agentenv``, the item without its parts (``source`` and ``source_id`` among
them), so a verifier reading the stored conversation can tell a delivered line from what the person wrote in the same
message.

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

from agent_env.env.gateway.constants import EXT_NOTIFICATIONS_URI

_TAKE_ATTEMPTS = 3
_TAKE_RETRY_BASE_DELAY_S = 0.5  # seconds; the nth retry waits n times this
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


def label_time(at: Optional[str]) -> str:
    """'Mon Feb 1, 3:00 PM UTC'; 'now' for an item with no time (queued while the env clock was unset)."""
    if not at:
        return "now"
    at = _utc(at)
    return f"{at.strftime('%a %b')} {at.day}, {at.strftime('%I:%M %p').lstrip('0')} UTC"


def delivered_line(item: dict) -> str:
    """A taken item's text as the agent reads it: ``[when · channel]``, then its text parts."""
    when = label_time(item.get("at"))
    times = item.get("times", 1)
    again = f" (it came due {times} times while you were away)" if times > 1 else ""
    text = "\n\n".join(p["text"] for p in item["parts"] if p.get("kind") == "text")
    return f"[{when} · {item['channel']}] {text}{again}"


def delivered_parts(items: Iterable[dict]) -> list[dict]:
    """Per taken item, oldest first: one text part, its ``delivered_line``, then any parts that aren't text, as they
    are. Each carries ``metadata.agentenv``: the item without its parts, with the ``env_id`` it came from when the
    item has one."""
    parts = []
    for item in sorted(items, key=_oldest_first):
        meta = {"agentenv": {k: v for k, v in item.items() if k != "parts"}}
        parts.append({"kind": "text", "text": delivered_line(item), "metadata": meta})
        parts.extend({**p, "metadata": meta} for p in item["parts"] if p.get("kind") != "text")
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
