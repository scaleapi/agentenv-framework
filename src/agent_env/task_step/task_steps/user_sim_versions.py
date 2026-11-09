"""The user sim plug-in the always_on step calls when a user_sim step names none: each version decides the user sim's
system prompt and answer format (``setup``), renders its own input for one read and parses the answer (``answer``).

The interface, which any plug-in a ``user_sim`` step names (a file artifact) must offer:

    setup(version, person) -> {"system_prompt": str | None, "output_format": dict | None}
    async answer(version, person, history, read, ask) -> {"outcome": "reply" | "no_reply" | "done", "message": str,
                                                          "used": str | None, "input": str, "raw": ..., ...}

``person`` is the user_sim step's settings (name, first, time_zone, description, versions, ...). A plug-in may also
offer ``configure(params)``, called once with the step's ``plugin.params``. ``history`` is every
earlier read, oldest first, each ``{"read": <read>, "answer": <answer>}``. ``read`` is this read:
``{"time": RFC 3339, "assistant_messages": [{"at", "text"}], "phone": [{"at", "channel", "text"}],
"you_sent": [{"at", "text"}]}``. ``ask(text)`` sends the rendered text to the deployed user sim (one session per run)
and returns ``{"text": raw text, "structured": parsed object or None}``.

Versions here: ``ours`` (the default) and its ablations ours_no_feed, ours_no_labels, ours_no_time, ours_no_used and
ours_always_replies, and ``today`` (the description as the whole prompt, the agent's raw text as the input, answers
{message, done}). ``ours`` shows each read labelled (the time, what's new on the phone, what the user sent, what the
assistant sent) and answers {outcome, message, used}; its prompt is ``versions.ours.system_prompt`` when the person
has one, else the rules below followed by the description, which then says only who the person is.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

# prompt_agent's default user-sim output format, used when a task sets none.
ML_DEFAULT_FORMAT = {"type": "json_schema", "schema": {"type": "object", "properties": {
    "message": {"type": "string", "description": "Your next reply"},
    "done": {"type": "boolean", "description": "Set true when you're satisfied and the conversation should end."}},
    "required": ["message", "done"]}}

_OURS_PROPS = {
    "outcome": {"type": "string", "enum": ["reply", "no_reply", "done"]},
    "message": {"type": "string"},
    "used": {"type": "string"},
    "score": {"type": "integer", "description": "Only with outcome done: your 1-5 rating"},
}
FOOTER_Q = "Your AI assistant sent you this. As {first}, decide: reply, don't reply, or say you're done."
FOOTER_Q_ALWAYS = "Your AI assistant sent you this. As {first}, decide: reply, or say you're done."

VERSIONS = ("today", "ours", "ours_no_feed", "ours_no_labels", "ours_no_time", "ours_no_used", "ours_always_replies")


def _ours_schema(version: str) -> dict:
    props = dict(_OURS_PROPS)
    required = ["outcome", "message", "used"]
    if version == "ours_no_used":
        props.pop("used")
        required.remove("used")
    if version == "ours_always_replies":
        props["outcome"] = {"type": "string", "enum": ["reply", "done"]}
    return {"type": "json_schema", "schema": {"type": "object", "additionalProperties": False, "properties": props,
                                              "required": required}}


def _json_line(version: str) -> str:
    outcomes = '"reply" | "done"' if version == "ours_always_replies" else '"reply" | "no_reply" | "done"'
    used = '' if version == "ours_no_used" else ', "used": "..."'
    return '{"outcome": ' + outcomes + ', "message": "..."' + used + '}'


def setup(version: str, person: dict) -> dict:
    if version not in VERSIONS:
        raise ValueError(f"unknown user sim version {version!r}; this plug-in has {list(VERSIONS)}")
    if version == "today":
        return {"system_prompt": person.get("description"),
                "output_format": person.get("output_format") or ML_DEFAULT_FORMAT}
    ours = (person.get("versions") or {}).get("ours") or {}
    prompt = ours.get("system_prompt")
    if not prompt:
        if not person.get("description"):
            raise ValueError(f"user sim {person.get('name')!r} needs a description or versions.ours.system_prompt")
        return {"system_prompt": ours_prompt(version, person), "output_format": _ours_schema(version)}
    if version == "ours_no_used":
        prompt = _drop_used(prompt)
    if version == "ours_always_replies":
        prompt = _drop_no_reply(prompt)
    return {"system_prompt": prompt, "output_format": _ours_schema(version)}


def ours_prompt(version: str, person: dict) -> str:
    """How a person reads and answers in ``ours``, then who they are (the description)."""
    full = person.get("name") or "the user"
    first = person.get("first") or full.split()[0]
    feed, clock = version != "ours_no_feed", version != "ours_no_time"
    shows = [*(["the time"] if clock else []), *(["what's new on your phone"] if feed else []), "what you sent",
             "what your AI assistant sent"]
    lines = [f"You are {full}, a real person messaging your personal AI assistant. You are not the AI assistant and "
             f"not a tester. Each message you get shows {', '.join(shows[:-1])}, and {shows[-1]}. Reply only as "
             f"{first}. Don't do your assistant's work: never look things up for it or hand over facts it should find "
             "itself. You can't open your apps" + ("; what's new on your phone is listed in each message." if feed
                                                    else ".")]
    if feed:
        lines.append("What's new on your phone is for you: your assistant reads your inbox itself. Don't pass it on or "
                     "make calls about it unless the assistant asks, or your description says to.")
    if clock:
        lines.append("Go by the time each message gives. Any other date you see is the computer's clock, not today's.")
    lines.append("If the assistant asks something your description doesn't cover: for a choice, choose the way you "
                 "would; for a fact about you or anyone else that isn't written there, say you don't know. Never "
                 "guess facts.")
    lines.append('Only answer "done" with a short sign-off. If your message answers a question or tells the assistant '
                 'to do something, answer "reply": a "done" message never reaches the assistant.')
    used = ('In "used", give the tag or quote the words of your description you relied on, or "none". '
            if version != "ours_no_used" else "")
    lines.append(used + "Answer in JSON:\n" + _json_line(version))
    return "\n".join(lines) + "\n\nWHO YOU ARE\n" + person["description"].strip() + "\n"


def _drop_used(prompt: str) -> str:
    """Remove the lines that explain the `used` field (the tagged persona lines stay)."""
    return "\n".join(ln for ln in prompt.splitlines() if '"used"' not in ln and "`used`" not in ln
                     and not ln.lower().lstrip().startswith("used:"))


def _drop_no_reply(prompt: str) -> str:
    return prompt.replace('"reply" | "no_reply" | "done"', '"reply" | "done"').replace(
        'reply, don\'t reply, or say you\'re done', "reply, or say you're done")


# ---- rendering ----------------------------------------------------------------------------------------------------
def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _local(text: Optional[str], tz: ZoneInfo) -> Optional[datetime]:
    return _utc(text).astimezone(tz) if text else None


def _hm(d: datetime) -> str:
    return d.strftime("%I:%M %p").lstrip("0")


def _stamp(t: datetime, now: datetime) -> str:
    days = (now.date() - t.date()).days
    if days == 0:
        return _hm(t)
    if days < 6:
        return f"{t.strftime('%a')} {_hm(t)}"
    return f"{t.strftime('%a %b')} {t.day}, {_hm(t)}"


def render_today(read: dict) -> str:
    """ML's setup: the assistant's raw message text, nothing else (several messages joined by a blank line)."""
    return "\n\n".join(m["text"].strip() for m in read["assistant_messages"])


def render_ours(version: str, person: dict, history: list[dict], read: dict) -> str:
    tz = ZoneInfo(person.get("time_zone") or "UTC")
    now = _local(read["time"], tz)
    first = person.get("first") or person.get("name") or "you"
    labels = version != "ours_no_labels"
    out: list[str] = []
    if version != "ours_no_time":
        out.append(f"It's {now.strftime('%a, %b')} {now.day}, {_hm(now)}.")
        last = _last_messaged(history, read)
        if last:
            lt = _local(last, tz)
            days = (now.date() - lt.date()).days
            when = (f"at {_hm(lt)}" if days == 0 else f"on {lt.strftime('%a')} at {_hm(lt)}" if days < 6
                    else f"on {lt.strftime('%a %b')} {lt.day} at {_hm(lt)}")
            out.append(f"You last messaged your AI assistant {when}.")
    if version != "ours_no_feed":
        if read.get("phone"):
            out.append("\nNEW ON YOUR PHONE (since you last checked):" if labels else "")
            for e in read["phone"]:
                head, *rest = e["text"].split("\n")
                t = _local(e.get("at"), tz) or now
                out.append(f"  [{_stamp(t, now)} · {e['channel']}] {head}" + "".join(f"\n    {x}" for x in rest))
        earlier = [e for h in history for e in h["read"].get("phone") or []]
        if earlier:
            out.append("\nEARLIER ON YOUR PHONE (you've already seen these):" if labels else "")
            for e in earlier:
                t = _local(e.get("at"), tz) or now
                out.append(f"  [{_stamp(t, now)} · {e['channel']}] {e['text'].splitlines()[0] if e['text'] else ''}")
    if read.get("you_sent"):
        out.append("\nYOU SENT YOUR AI ASSISTANT:" if labels else "")
        for s in read["you_sent"]:
            t = _local(s.get("at"), tz)
            out.append((f"  [{_stamp(t, now)}] " if t else "  ") + s["text"])
    out.append("\nNEW FROM YOUR AI ASSISTANT (this is what you're answering):" if labels else "")
    for m in read["assistant_messages"]:
        t = _local(m.get("at"), tz)
        body = "\n".join("    " + ln if ln.strip() else "" for ln in m["text"].strip().splitlines())
        out.append((f"  [{_stamp(t, now)}]\n" if t else "") + body)
    q = FOOTER_Q_ALWAYS if version == "ours_always_replies" else FOOTER_Q
    out.append("\n" + q.format(first=first) + "\nAnswer in JSON: " + _json_line(version))
    return "\n".join(x for x in out if x != "" or labels)


def _last_messaged(history: list[dict], read: dict) -> Optional[str]:
    """When the person last sent their assistant anything before this read: their scripted lines (this read's
    included, as the h2h runner did) and their earlier replies."""
    times = [s["at"] for h in history for s in h["read"].get("you_sent") or [] if s.get("at")]
    times += [s["at"] for s in read.get("you_sent") or [] if s.get("at")]
    times += [h["read"]["time"] for h in history if (h.get("answer") or {}).get("outcome") in ("reply", "done")]
    times = [t for t in times if _utc(t) <= _utc(read["time"])]
    return max(times, key=_utc) if times else None


# ---- answers ------------------------------------------------------------------------------------------------------
def _schema_errors(obj: Any, schema: dict) -> list[str]:
    if not isinstance(obj, dict):
        return ["not an object"]
    errs = [f"missing {k}" for k in schema.get("required", []) if k not in obj]
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        errs += [f"extra key {k}" for k in obj if k not in props]
    for k, v in obj.items():
        p = props.get(k) or {}
        if "enum" in p and v not in p["enum"]:
            errs.append(f"{k}: {v!r} not in enum")
    return errs


def _parse_text(text: str) -> Optional[dict]:
    s = (text or "").replace("```json", "").replace("```", "")
    dec, i, found = json.JSONDecoder(), s.find("{"), None
    while i != -1:
        try:
            obj, end = dec.raw_decode(s, i)
            if isinstance(obj, dict) and ("message" in obj or "outcome" in obj):
                found = obj
            i = s.find("{", end)
        except ValueError:
            i = s.find("{", i + 1)
    return found


async def answer(version: str, person: dict, history: list[dict], read: dict,
                 ask: Callable[[str], Awaitable[dict]]) -> dict:
    text = render_today(read) if version == "today" else render_ours(version, person, history, read)
    got = await ask(text)
    obj = got.get("structured")
    if not isinstance(obj, dict):
        obj = _parse_text(got.get("text") or "")
    fmt = setup(version, person)["output_format"]
    errors = _schema_errors(obj, fmt["schema"]) if obj is not None else ["no JSON object in the answer"]
    out = {"version": version, "input": text, "raw": got.get("text"), "structured": obj, "schema_errors": errors}
    if obj is None:
        # A malformed answer reaches the assistant as raw text, as prompt_agent does today.
        return {**out, "outcome": "reply", "message": got.get("text") or "", "used": None}
    if version == "today":
        done = obj.get("done") is True
        return {**out, "outcome": "done" if done else "reply", "message": str(obj.get("message") or ""),
                "used": None, "done": obj.get("done")}
    outcome = obj.get("outcome")
    if outcome not in ("reply", "no_reply", "done"):
        outcome = "reply"
    return {**out, "outcome": outcome, "message": str(obj.get("message") or ""), "used": obj.get("used"),
            "score": obj.get("score")}
