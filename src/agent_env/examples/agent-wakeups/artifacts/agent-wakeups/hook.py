"""A prompt_agent post-turn hook that runs a conversation over simulated days, waking the agent only for what would
really reach it: a check it scheduled for itself, a notification a trigger sent it (notify_agent), or a message the
person writes at a set time. Everything else the world does happens silently while the agent is away."""
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CLOCK = "urn:agentenv:clock/v1"
TRIGGERS = "urn:agentenv:triggers/v1"
WAKEUPS = "urn:agentenv:wakeups/v1"
IN_FLIGHT = ("firing", "queued")
PARAMS = ("end_time", "time_zone", "person_messages", "settle_timeout_seconds", "max_jumps_per_turn")
POLL_SECONDS = 0.5

logger = logging.getLogger(__name__)


async def post_turn(turn, params):
    """A firing agent trigger answers first; else jump, silently, to the next thing that would wake the agent."""
    if turn.last:
        return None
    end_time, zone, person, settle_timeout, max_jumps = _params(params)
    envs = [env for env in turn.envs if env.supports(CLOCK, "advance") and env.supports(WAKEUPS, "take")]
    if not envs:
        return _end(turn, "no_env_with_clock_and_wakeups")
    await _settle(turn, envs, settle_timeout)
    answer = await turn.user_triggers()
    if answer is not None:
        return answer
    t0 = min([_utc((await env.invoke(CLOCK, "state"))["t0"]) for env in envs])
    end = _end_instant(end_time, t0)
    sent = _delivered_person(turn)
    for _ in range(max_jumps):
        now = await _now(envs)
        items = await _take(envs, now)
        due_person = [m for m in person if m["at"] <= now and _iso(m["at"]) not in sent]
        if items or due_person:
            turn.record({"woke_at": _iso(now), "for": [i["kind"] for i in items] + ["person"] * len(due_person),
                         **({"person_at": [_iso(m["at"]) for m in due_person]} if due_person else {})})
            return {"message": _message(items, due_person, zone)}
        nxt = await _next(envs, person, now, sent)
        if nxt is None:
            return _end(turn, "nothing_can_wake_the_agent", at=_iso(now))
        if end is not None and nxt > end:
            return _end(turn, "end_time", next_at=_iso(nxt))
        for env in envs:
            await _advance(env, nxt)
        await _settle(turn, envs, settle_timeout)
    return _end(turn, "max_jumps_per_turn", at=_iso(await _now(envs)))


def _params(params):
    unknown = sorted(set(params) - set(PARAMS))
    if unknown:
        raise ValueError(f"agent-wakeups takes {list(PARAMS)}, not {unknown}")
    end_time = params.get("end_time")
    if end_time is not None:
        _end_spec(end_time)
    zone_name = params.get("time_zone", "UTC")
    try:
        zone = ZoneInfo(zone_name)
    except Exception as e:
        raise ValueError(f"time_zone must be an IANA name such as 'America/Los_Angeles': {e!r}") from e
    person = []
    for m in params.get("person_messages", []):
        if not isinstance(m, dict) or not isinstance(m.get("at"), str) or not isinstance(m.get("text"), str):
            raise ValueError("each person_messages entry is {at: RFC 3339 instant, text: what the person writes}")
        person.append({"at": _utc(m["at"]), "text": m["text"]})
    timeout = params.get("settle_timeout_seconds", 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"settle_timeout_seconds must be a positive number, got {timeout!r}")
    max_jumps = params.get("max_jumps_per_turn", 500)
    if isinstance(max_jumps, bool) or not isinstance(max_jumps, int) or max_jumps < 1:
        raise ValueError(f"max_jumps_per_turn must be a positive integer, got {max_jumps!r}")
    return end_time, zone, sorted(person, key=lambda m: m["at"]), timeout, max_jumps


async def _now(envs):
    return min([_utc((await env.invoke(CLOCK, "state"))["virtual_time"]) for env in envs])


async def _take(envs, now):
    items = []
    for env in envs:
        items += (await env.invoke(WAKEUPS, "take", {"until": _iso(now)})).get("items", [])
    return sorted(items, key=lambda i: i.get("at") or "")


async def _next(envs, person, now, sent):
    """The earliest instant after now at which anything happens: a world event (silent), a scheduled check, a
    notification, or a person's message."""
    candidates = [m["at"] for m in person if m["at"] > now and _iso(m["at"]) not in sent]
    for env in envs:
        wake = await env.invoke(WAKEUPS, "state")
        if wake.get("next_check_at"):
            candidates.append(_utc(wake["next_check_at"]))
        if env.supports(TRIGGERS, "state"):
            world = await env.invoke(TRIGGERS, "state")
            if world.get("next_event_at"):
                candidates.append(_utc(world["next_event_at"]))
    later = [c for c in candidates if c > now]
    return min(later) if later else None


def _delivered_person(turn):
    """The person's messages this hook already delivered, from its own records (a module global would outlive the run)."""
    sent = set()
    for entries in (turn.metadata.get("post_turn_hooks") or {}).values():
        for entry in entries:
            sent.update(entry.get("person_at", []))
    return sent


def _message(items, person, zone):
    """One line per thing, oldest first, each saying when and through what, so nothing poses as the person."""
    lines = []
    for item in items:
        when = _local(item.get("at"), zone)
        if item["kind"] == "scheduled_check":
            again = f" (it came due {item['times']} times while you were away)" if item.get("times", 1) > 1 else ""
            lines.append((item.get("at") or "", f"[{when} · your scheduled check] {item['note']}{again}"))
        else:
            lines.append((item.get("at") or "", f"[{when} · {item['channel']}] {item['text']}"))
    for m in person:
        lines.append((_iso(m["at"]), f"[{_local(_iso(m['at']), zone)}] {m['text']}"))
    return "\n".join(text for _, text in sorted(lines))


def _local(at, zone):
    if not at:
        return "now"
    local = _utc(at).astimezone(zone)
    return f"{local.strftime('%a %b')} {local.day}, {local.strftime('%I:%M %p').lstrip('0')}"


async def _advance(env, target):
    try:
        await env.invoke(CLOCK, "advance", {"virtual_time": _iso(target)})
    except Exception:
        clock = await env.invoke(CLOCK, "state")
        if not (clock.get("armed") and _utc(clock["virtual_time"]) >= target):
            raise


async def _settle(turn, envs, timeout):
    """Wait until no trigger still fires (bar one waiting for a later time), so what an event set off has landed."""
    deadline = time.monotonic() + timeout
    waiting = {i for i, env in enumerate(envs) if env.supports(TRIGGERS, "state")}
    while waiting and time.monotonic() < deadline:
        for i in sorted(waiting):
            env = envs[i]
            clock, state = await env.invoke(CLOCK, "state"), await env.invoke(TRIGGERS, "state")
            now = _utc(clock["virtual_time"])
            if state.get("next_event_at") is not None and _utc(state["next_event_at"]) <= now:
                await _advance(env, now)
            elif not any(_busy(row, now) for row in state["triggers"]):
                waiting.discard(i)
        if waiting:
            turn.heartbeat()
            await asyncio.sleep(POLL_SECONDS)
    if waiting:
        logger.warning(f"env(s) {sorted(envs[i].env_id for i in waiting)} still had triggers firing {timeout}s on")


def _busy(row, now):
    if row["status"] not in IN_FLIGHT and not row.get("pending"):
        return False
    return not (row.get("waiting") and row.get("resumes_at") and _utc(row["resumes_at"]) > now)


def _end_spec(end_time):
    if isinstance(end_time, str) and end_time.startswith("P"):
        return "after", _duration(end_time)
    if isinstance(end_time, str):
        at = _utc(end_time)
        return "at", at
    raise ValueError(f"end_time {end_time!r} is neither an RFC 3339 instant with an offset nor a duration like 'P4D'")


def _end_instant(end_time, t0):
    if end_time is None:
        return None
    kind, value = _end_spec(end_time)
    return value if kind == "at" else t0 + value


def _duration(text):
    m = re.fullmatch(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", text)
    if m is None or not any(m.groups()):
        raise ValueError(f"invalid ISO 8601 duration {text!r}")
    weeks, days, hours, minutes, seconds = (float(g or 0) for g in m.groups())
    return timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)


def _end(turn, reason, **detail):
    turn.record({"end": reason, **detail})
    return {"end": reason}


def _utc(text):
    at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError(f"{text!r} has no UTC offset")
    return at.astimezone(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
