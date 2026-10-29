"""A prompt_agent post-turn hook that runs a conversation over simulated days by moving a frozen clock to the next
thing due for the agent: a check it scheduled for itself or a notification a trigger sent it (notifications/v1, which
prompt_agent delivers), or a message the person writes at a set time (which this hook sends). Everything else the
world does happens silently on the way, while the agent is away."""
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone

from agent_env.task_step.task_steps.notification_delivery import label_time, label_zone

CLOCK = "urn:agentenv:clock/v1"
TRIGGERS = "urn:agentenv:triggers/v1"
NOTIFICATIONS = "urn:agentenv:notifications/v1"
IN_FLIGHT = ("firing", "queued")
PARAMS = ("end_time", "role", "time_zone", "person_messages", "settle_timeout_seconds", "max_jumps_per_turn")
POLL_SECONDS = 0.5

logger = logging.getLogger(__name__)


async def post_turn(turn, params):
    """A firing agent trigger answers first; else move the clock, silently, to the next thing due for the agent."""
    if turn.last:
        return None
    end_time, role, zone, person, settle_timeout, max_jumps = _params(params)
    envs = await _armed(turn.envs)
    if not envs:
        return _end(turn, "no_armed_env_with_clock_and_notifications")
    if (stuck := await _settle(turn, envs, settle_timeout)):
        return _end(turn, "triggers_still_firing", envs=stuck)
    answer = await turn.user_triggers()
    if answer is not None:
        return answer
    t0 = min([_utc((await env.invoke(CLOCK, "state"))["t0"]) for env in envs])
    end = _end_instant(end_time, t0)
    sent = _delivered_person(turn)
    for jump in range(max_jumps + 1):
        now = await _now(envs)
        due = await _due(envs, role, now)
        due_person = [m for m in person if m["at"] <= now and _iso(m["at"]) not in sent]
        if due or due_person:
            turn.record({"woke_at": _iso(now), "for": due + ["person"] * len(due_person),
                         **({"person_at": [_iso(m["at"]) for m in due_person]} if due_person else {})})
            # prompt_agent puts what came due on notifications/v1 ahead of this; [] sends only that
            return {"message": _person_message(due_person, zone) if due_person else []}
        if jump == max_jumps:  # checked above: what the last jump made due still wakes the agent
            return _end(turn, "max_jumps_per_turn", at=_iso(now))
        nxt = await _next(envs, role, person, now, sent)
        if nxt is None:
            return _end(turn, "nothing_can_wake_the_agent", at=_iso(now))
        if end is not None and nxt > end:
            return _end(turn, "end_time", next_at=_iso(nxt))
        for env in envs:
            await _advance(env, nxt)
        if (stuck := await _settle(turn, envs, settle_timeout)):
            # ending here would read unfinished work as nothing left to happen, and lose what it sends
            return _end(turn, "triggers_still_firing", envs=stuck, at=_iso(nxt))

def _params(params):
    unknown = sorted(set(params) - set(PARAMS))
    if unknown:
        raise ValueError(f"advance-to-next-event takes {list(PARAMS)}, not {unknown}")
    end_time = params.get("end_time")
    if end_time is not None:
        _end_spec(end_time)
    role = params.get("role", "default")
    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"role must name the agent's AgentEnv-Role, e.g. 'default', got {role!r}")
    zone = label_zone(params.get("time_zone"))
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
    return end_time, role.strip(), zone, sorted(person, key=lambda m: m["at"]), timeout, max_jumps


async def _armed(envs):
    """The envs with clock advance and notifications state whose clock is armed; an unarmed one has no time to move."""
    armed = []
    for env in envs:
        if env.supports(CLOCK, "advance") and env.supports(NOTIFICATIONS, "state"):
            if (await env.invoke(CLOCK, "state")).get("armed"):
                armed.append(env)
    return armed


async def _now(envs):
    return min([_utc((await env.invoke(CLOCK, "state"))["virtual_time"]) for env in envs])


async def _due(envs, role, now):
    """The source of each check and notification due for ``role`` by now, oldest first. Read, not taken: prompt_agent
    takes and delivers them."""
    due = []
    for env in envs:
        state = await env.invoke(NOTIFICATIONS, "state", {"to": role})
        due += [(_utc(c["next"]), "scheduled_check") for c in state.get("checks", [])
                if c["status"] == "active" and _utc(c["next"]) <= now]
        due += [(_utc(n["at"]) if n.get("at") else now, "trigger") for n in state.get("pending_notifications", [])
                if not n.get("at") or _utc(n["at"]) <= now]
    return [source for _, source in sorted(due, key=lambda d: d[0])]


async def _next(envs, role, person, now, sent):
    """The earliest instant after now at which anything happens: a world event (silent), a check or notification due
    for ``role``, or a person's message."""
    candidates = [m["at"] for m in person if m["at"] > now and _iso(m["at"]) not in sent]
    for env in envs:
        notes = await env.invoke(NOTIFICATIONS, "state", {"to": role})
        if notes.get("next_due_at"):
            candidates.append(_utc(notes["next_due_at"]))
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


def _person_message(person, zone):
    """One line per message, oldest first, each saying when it was written."""
    return "\n".join(f"[{label_time(_iso(m['at']), zone)}] {m['text']}" for m in person)


async def _advance(env, target):
    try:
        await env.invoke(CLOCK, "advance", {"virtual_time": _iso(target)})
    except Exception:
        clock = await env.invoke(CLOCK, "state")
        if not (clock.get("armed") and _utc(clock["virtual_time"]) >= target):
            raise


async def _settle(turn, envs, timeout):
    """Wait until no trigger still fires (bar one waiting for a later time), so what an event set off has landed.
    Returns the ids of the envs still firing when ``timeout`` ran out, empty when all settled."""
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
    stuck = sorted(envs[i].env_id for i in waiting)
    if stuck:
        logger.warning(f"env(s) {stuck} still had triggers firing {timeout}s on")
    return stuck


def _busy(row, now):
    if row.get("evaluating"):  # a state check the agent's turn set off is still running
        return True
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
