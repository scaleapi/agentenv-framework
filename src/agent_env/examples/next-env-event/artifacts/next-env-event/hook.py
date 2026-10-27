"""A prompt_agent post-turn hook that jumps env clocks to the next event their triggers act on, then wakes the agent."""
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone

CLOCK = "urn:agentenv:clock/v1"
TRIGGERS = "urn:agentenv:triggers/v1"
IN_FLIGHT = ("firing", "queued")
NUDGE = "[{virtual_time}] Time has passed. Check for anything new and handle it."
PARAMS = ("end_time", "nudge", "settle_timeout_seconds")
POLL_SECONDS = 0.5
DURATION = re.compile(r"P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?")

logger = logging.getLogger(__name__)


async def post_turn(turn, params):
    """A firing agent trigger answers first; else jump to the next env event, let it settle, and wake the agent."""
    if turn.last:
        return None
    end_time, nudge, settle_timeout = _params(params)
    envs = await _armed(turn)
    old = [env.env_id for env, _, state in envs if "next_event_at" not in state or not env.supports(CLOCK, "advance")]
    if envs and not old:
        await _settle(turn, envs, settle_timeout)
    answer = await turn.user_triggers()
    if answer is not None:
        return answer
    if not envs:
        return _end(turn, "no_armed_env_clock")
    if old:
        logger.warning(f"env(s) {old} have no clock advance or next_event_at; rebuild their gateway image to jump "
                       "to the next event")
        return _end(turn, "gateway_without_advance", envs=old)
    envs = await _armed(turn)
    pending = [_utc(state["next_event_at"]) for _, _, state in envs if state["next_event_at"]]
    if not pending:
        return _end(turn, "no_pending_events")
    event_at = min(pending)
    end = _end_instant(end_time, envs)
    if end is not None and event_at > end:
        return _end(turn, "end_time", next_event_at=_iso(event_at))
    due = [await _advance(env, event_at) if _utc(clock["virtual_time"]) < event_at else [] for env, clock, _ in envs]
    settled = await _settle(turn, envs, settle_timeout)
    turn.record({"event_at": _iso(event_at), "envs": [
        {"env_id": env.env_id, "env_step_id": env.env_step_id, "due": d, "settled": ok}
        for (env, _, _), d, ok in zip(envs, due, settled)]})
    answer = await turn.user_triggers()
    return answer if answer is not None else {"message": nudge.format(virtual_time=_iso(event_at))}


def _params(params):
    unknown = sorted(set(params) - set(PARAMS))
    if unknown:
        raise ValueError(f"next-env-event takes {list(PARAMS)}, not {unknown}")
    end_time = params.get("end_time")
    if end_time is not None:
        _end_spec(end_time)
    nudge = params.get("nudge", NUDGE)
    try:
        nudge.format(virtual_time="")
    except (AttributeError, KeyError, IndexError, ValueError) as e:
        raise ValueError(f"nudge is text whose one placeholder is {{virtual_time}}: {e!r}") from e
    timeout = params.get("settle_timeout_seconds", 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"settle_timeout_seconds must be a positive number, got {timeout!r}")
    return end_time, nudge, timeout


async def _armed(turn):
    """Each env offering clock and triggers state whose clock is armed, with its clock and triggers state."""
    armed = []
    for env in turn.envs:
        if not (env.supports(CLOCK, "state") and env.supports(TRIGGERS, "state")):
            continue
        clock = await env.invoke(CLOCK, "state")
        if clock.get("armed"):
            armed.append((env, clock, await env.invoke(TRIGGERS, "state")))
    return armed


async def _advance(env, target):
    """Advance to ``target``, running the env's time check: what it made due, none if a running clock got there."""
    try:
        return (await env.invoke(CLOCK, "advance", {"virtual_time": _iso(target)})).get("due", [])
    except Exception:
        clock = await env.invoke(CLOCK, "state")
        if clock.get("armed") and _utc(clock["virtual_time"]) >= target:
            return []
        raise


async def _settle(turn, envs, timeout):
    """Wait until no trigger on each env still fires, bar one waiting for later; whether each settled in time."""
    deadline = time.monotonic() + timeout
    waiting = set(range(len(envs)))
    while True:
        for i in sorted(waiting):
            env = envs[i][0]
            clock, state = await env.invoke(CLOCK, "state"), await env.invoke(TRIGGERS, "state")
            now = _utc(clock["virtual_time"])
            if state["next_event_at"] is not None and _utc(state["next_event_at"]) <= now:
                await _advance(env, now)  # an event already reached waits for the time check, so run it now
            elif not any(_busy(row, now) for row in state["triggers"]):
                waiting.discard(i)
        if not waiting or time.monotonic() >= deadline:
            break
        turn.heartbeat()
        await asyncio.sleep(POLL_SECONDS)
    if waiting:
        logger.warning(f"env(s) {sorted(envs[i][0].env_id for i in waiting)} still had triggers firing {timeout}s "
                       "on; waking the agent anyway")
    return [i not in waiting for i in range(len(envs))]


def _busy(row, now):
    """A trigger is still firing or checking its state, and not just waiting for a later time."""
    if row.get("evaluating"):
        return True
    if row["status"] not in IN_FLIGHT and not row.get("pending"):
        return False
    return not (row.get("waiting") and row.get("resumes_at") and _utc(row["resumes_at"]) > now)


def _end_spec(end_time):
    """``end_time`` as ("at", instant) for RFC 3339 with an offset, or ("after", timedelta) for an ISO 8601 duration."""
    if isinstance(end_time, str) and end_time.startswith("P"):
        match = DURATION.fullmatch(end_time)
        if match is not None and any(match.groups()):
            weeks, days, hours, minutes, seconds = (float(g or 0) for g in match.groups())
            return "after", timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
    elif isinstance(end_time, str):
        try:
            at = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            at = None
        if at is not None and at.tzinfo is not None:
            return "at", at.astimezone(timezone.utc)
    raise ValueError(f"end_time {end_time!r} is neither an RFC 3339 instant with an offset nor a duration like 'P4D'")


def _end_instant(end_time, envs):
    """``end_time`` as an instant, a duration counting from the earliest env t0."""
    if end_time is None:
        return None
    kind, value = _end_spec(end_time)
    return value if kind == "at" else min(_utc(clock["t0"]) for _, clock, _ in envs) + value


def _end(turn, reason, **detail):
    turn.record({"end": reason, **detail})
    return {"end": reason}


def _utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
