"""The built-in user sim: deploy_user_sim_agent writes its system prompt with ``setup``, and the always_on step reads
with ``answer``. The interface is in deploy_user_sim_agent.py; how to write a user for it is in user_sim.md. It has
one version, ``ours``.

``setup`` writes the system prompt once per run: fixed lines on how a real person reads and answers, the description
as tagged sections (with the opening request), the user's rules from ``permissions`` and ``keep_private``, and the
answer format. Its done line follows ``person["done_ends_run"]``, the multi-day wording when that's absent.
``answer`` renders one read as one labelled message, asks the user sim, and parses its answer; a malformed answer
gets one retry in the same session. What earlier reads showed, and when the user last wrote, come from ``history``;
keys of ``read`` it doesn't know are ignored. A line the user sent with files (the opening's ``files``,
``{"name", "description"}``) shows each after its text, "(you attached itinerary.pdf: Mom and Dad's flight
itinerary)", or by name alone with no description: the user sim can't open them, and the opening's line says it
knows only that much of them.
``tagged_lines`` is each tag the system prompt shows with its line, which deploy_user_sim_agent records so an answer's
``used`` resolves against the lines the user sim saw.
``rate`` asks the user sim, once the run is over, to rate the outcome 1-5 by the ``satisfaction`` lines it is given
(the rate_user_satisfaction step's, never the persona's, so the system prompt never has them), with a one-line
reason.
``problems`` is the load check for the settings."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

VERSION = "ours"

# (key in description, section title, tag letter), in the order the prompt shows them. No letter: untagged.
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("who_you_are", "WHO YOU ARE", "W"),
    ("how_you_write", "HOW YOU WRITE", ""),
    ("what_you_know", "WHAT YOU KNOW (to use only when the assistant asks about it)", "K"),
    ("already_told", "THINGS YOU ALREADY TOLD THE ASSISTANT (if it asks for one again, remind it, a little annoyed, "
                     "that you already told it, and tell it again)", "T"),
    ("should_already_know", "THINGS THE ASSISTANT SHOULD ALREADY KNOW (never bring these up; if it gets one wrong, "
                            "correct it briefly, a little irritated, even when you'd otherwise let something slide)",
     "C"),
    ("what_to_look_up", "WHAT TO LOOK UP (say where; don't give the value)", "L"),
    ("decisions", "YOUR DECISIONS (when the assistant asks)", "D"),
    ("how_you_react", "HOW YOU REACT", "H"),
    ("dont_mention", "DON'T MENTION THESE UNLESS THE ASSISTANT ASKS", "M"),
    ("phone", "WHAT YOU DO WITH YOUR EMAILS, TEXTS AND NOTIFICATIONS", "F"),
    ("when_done", "WHEN YOU'RE DONE", ""),
)
SECTION_KEYS = tuple(key for key, _, _ in SECTIONS)
RULES_TITLE = "YOUR RULES FOR THE ASSISTANT (it has them in its notes)"
PRIVATE_TITLE = "PRIVATE"
RULES_AFTER = "decisions"
# A description section that isn't one, with where its lines go: a task that has it is refused at load with that.
MOVED_SECTIONS: dict[str, str] = {
    "never_supply": "put facts the assistant should find in what_to_look_up, and secrets in keep_private",
}

PERMISSION_KINDS: dict[str, str] = {
    "just_do": "Just do",
    "ask_first": "Ask you first",
    "never": "Never",
    "tell_after": "Tell you after",
}
BROKEN_RULE_LINE = ('If it does an "ask first" or a "never" without asking and tells you, say once, briefly, that '
                    "you're not happy about it.")
OPENING_LINE = 'Earlier you texted the assistant:\n"{text}"'
# A file the user sent with a line, after its text: the user sim can't open it, so it's told what it attached.
ATTACHED = "(you attached {name}: {description})"
ATTACHED_NAME_ONLY = "(you attached {name})"
# Under an opening with files, so the user sim never makes up what one holds.
ATTACHED_KNOWN = ("You remember only what's said here about each file you attached, not what's in it; if the "
                  "assistant asks what's in one, say you don't remember or point it to the file.")

OUTCOMES = ("reply", "no_reply", "done")
ANSWER_LINE = '{"outcome": "reply" | "no_reply" | "done", "message": "...", "used": "..."}'
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "outcome": {"type": "string", "enum": list(OUTCOMES)},
        "message": {"type": "string"},
        "used": {"type": "string"},
        "score": {"type": "integer"},
    },
    "required": ["outcome", "message"],
}
OUTPUT_FORMAT: dict[str, Any] = {"type": "json_schema", "schema": ANSWER_SCHEMA}

DONE_ENDS_RUN = ('Only answer "done" with a short sign-off. If your message answers a question or tells the assistant '
                 'to do something, answer "reply": after "done", the conversation is over.')
DONE_MULTI_DAY = ("Answer \"done\" only as WHEN YOU'RE DONE says; your message reaches the assistant either way, and "
                  "the week goes on.")
DONE_MULTI_DAY_NO_SECTION = ('Only answer "done" with a short sign-off; your message reaches the assistant either way, '
                             "and the week goes on.")
CLOSING = "Your AI assistant sent you this. As {first}, decide: reply, don't reply, or say you're done."
RETRY = "Your last answer wasn't in the required format. Answer again, only as JSON: " + ANSWER_LINE
# The earlier list keeps a long run's reads short: the last week's items, at most this many.
EARLIER_DAYS = 7
EARLIER_MAX = 20
EARLIER_CUT = "  (older messages not shown)"

# The rating after the run (``rate``), asked by the rate_user_satisfaction step with that step's ``satisfaction`` lines,
# shown in the ask under RATING_TITLE, tagged S1... Its answer fits OUTPUT_FORMAT, which the agent was given at deploy.
SCORES = (1, 2, 3, 4, 5)
RATING_TITLE = "HOW YOU'D RATE THE JOB"
RATING_TAG = "S"
RATING_LINE = '{"outcome": "done", "score": 1 | 2 | 3 | 4 | 5, "message": "...", "used": "..."}'
RATING_OVER = "The job is over: your AI assistant won't see this, or anything you write from now on."
RATING_UNREAD = "\nNEW FROM YOUR AI ASSISTANT (you haven't read this yet):"
RATING_ASK = ("As {first}, rate how the job went, from everything you've seen: 5 if it was done perfectly, 3 if it "
              "wasn't perfect but was fine, 1 if you'd have done it yourself; 2 and 4 are in between. HOW YOU'D RATE "
              "THE JOB says what those mean for this job. Something you were never told was done doesn't count as "
              "done.")
RATING_ANSWER = ('In "message", give your reason in one line. In "used", write the tag of the line it rests on, or '
                 '"none". Answer in JSON: ' + RATING_LINE)
RATING_RETRY = "Your last answer wasn't in the required format. Answer again, only as JSON: " + RATING_LINE

# The kinds of line the user sent; a you_sent item of any other kind (an alert, a nudge) is never shown as theirs.
OWN_LINE_KINDS = frozenset({"opening", "user_says", "final"})


# ---- the system prompt --------------------------------------------------------------------------------------------
def _first(person: dict) -> str:
    return person.get("first") or person["name"].split()[0]


def _preamble(person: dict) -> list[str]:
    full, first = person["name"], _first(person)
    if person.get("done_ends_run"):
        done = DONE_ENDS_RUN
    elif person["description"].get("when_done"):
        done = DONE_MULTI_DAY
    else:
        done = DONE_MULTI_DAY_NO_SECTION
    return [
        f"You are {full}, a real person messaging your personal AI assistant. You are not the AI assistant and not a "
        "tester. Each message you get shows the time, your new emails, texts and notifications, what you sent, and what "
        "your AI "
        f"assistant sent. Reply only as {first}. Don't do your assistant's work: never look things up for it or hand "
        "over facts it should find itself. You can't open your apps; your new emails, texts and notifications are "
        "listed in each message.",
        "What other people and apps send you is for you: your assistant reads your inbox itself. Don't pass it on or "
        "make calls about it unless the assistant asks, or your description says to.",
        "If the assistant asks something no line below covers: for a choice, choose the way you would; for a fact "
        "about you or anyone else that isn't written below, say you don't know. Never guess facts.",
        done,
        'In "used", write the tag of the line you relied on, or "none". Answer in JSON:\n' + ANSWER_LINE,
    ]


def tag_line(tag: str, text: str) -> str:
    """One description line under its tag; the line's later lines hang under its text."""
    head, *rest = text.split("\n")
    if tag:
        return "\n".join([f"  {tag:<4}{head}"] + ["  " + r if r.startswith("    ") else "      " + r.lstrip()
                                                    for r in rest])
    return "\n".join([f"  {head}"] + [r if r.startswith("    ") else "    " + r.lstrip() for r in rest])


def _items(value: Any) -> tuple[list[str], str]:
    if isinstance(value, dict):
        return list(value.get("items") or []), value.get("then") or ""
    return list(value or []), ""


def rule_lines(permissions: dict) -> list[str]:
    """The permissions in plain English, one line per kind, then how the user takes a broken rule."""
    lines = []
    for kind, label in PERMISSION_KINDS.items():
        items, then = _items(permissions.get(kind))
        if items:
            lines.append(f"{label}: {'; '.join(items)}." + (f" {then}" if then else ""))
    if _items(permissions.get("ask_first"))[0] or _items(permissions.get("never"))[0]:
        lines.append(BROKEN_RULE_LINE)
    return lines


def attached_note(files: Optional[list[dict]]) -> str:
    """What a line's ``files`` (``{"name", "description"}``) add after its text: nothing, when it has none."""
    return "".join(" " + (ATTACHED.format(**f) if f.get("description") else ATTACHED_NAME_ONLY.format(name=f["name"]))
                   for f in files or [])


def _who(person: dict) -> list[str]:
    """Who they are, with the opening request right after the first line."""
    lines = list(person["description"].get("who_you_are") or [])
    opening = person.get("opening") or {}
    if opening.get("text"):
        line = OPENING_LINE.format(text=opening["text"]) + attached_note(opening.get("files"))
        lines.insert(min(1, len(lines)), line + ("\n" + ATTACHED_KNOWN if opening.get("files") else ""))
    return lines


def _section(out: list[str], title: str, tag: str, lines: list[str]) -> None:
    if lines:
        out += ["", title] + [tag_line(f"{tag}{i}" if tag else "", text) for i, text in enumerate(lines, 1)]


def _sections(person: dict) -> list[tuple[str, str, list[str]]]:
    """Each section the system prompt shows, in order: its title, its tag letter ("" for none) and its lines."""
    out = []
    for key, title, tag in SECTIONS:
        out.append((title, tag, _who(person) if key == "who_you_are" else list(person["description"].get(key) or [])))
        if key == RULES_AFTER:
            out.append((RULES_TITLE, "A", rule_lines(person.get("permissions") or {})))
            out.append((PRIVATE_TITLE, "P", list(person.get("keep_private") or [])))
    return out


def system_prompt(person: dict) -> str:
    out = _preamble(person)
    for title, tag, lines in _sections(person):
        _section(out, title, tag, lines)
    return "\n".join(out) + "\n"


def tagged_lines(person: dict) -> dict[str, str]:
    """Each tag the system prompt shows, with its line as written there ("W2": the opening request, ...): what an
    answer's ``used`` names, recorded with the run so it resolves however the persona's lines change later."""
    return {f"{tag}{i}": text for _, tag, lines in _sections(person) if tag for i, text in enumerate(lines, 1)}


def rating_tagged_lines(satisfaction: list[str]) -> dict[str, str]:
    """The rating ask's tags, ``S1``..., with their ``satisfaction`` lines, as ``tagged_lines`` for the rating."""
    return {f"{RATING_TAG}{i}": text for i, text in enumerate(satisfaction, 1)}


# ---- the load check -----------------------------------------------------------------------------------------------
def _lines(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def problems(person: Any) -> list[str]:
    """What stops these settings from making a prompt, one message each; empty when they're fine. ``person`` has the
    shape ``setup`` gets."""
    if not isinstance(person, dict):
        return ["the user sim's settings must be an object"]
    out = []
    if not isinstance(person.get("name"), str) or not person["name"].strip():
        out.append("name is missing")
    if person.get("first") is not None and (not isinstance(person["first"], str) or not person["first"].strip()):
        out.append("first_name must be the name the user goes by")
    description = person.get("description")
    if not isinstance(description, dict):
        out.append(f"description must be an object of sections, from {list(SECTION_KEYS)}")
    else:
        unknown = sorted(set(description) - set(SECTION_KEYS))
        if unknown:
            out.append(f"description has unknown sections {unknown}; the sections are {list(SECTION_KEYS)}"
                       + ("; satisfaction goes on the rate_user_satisfaction step, so the user sim's prompt never has "
                          "it" if "satisfaction" in unknown else "")
                       + "".join(f"; {MOVED_SECTIONS[key]}" for key in unknown if key in MOVED_SECTIONS))
        out += [f"description.{key} must be a list of non-empty lines" for key in SECTION_KEYS
                if key in description and not _lines(description[key])]
        if not any(_lines(description.get(key)) and description[key] for key in SECTION_KEYS):
            out.append("description is empty: write at least who they are")
    permissions = person.get("permissions")
    if permissions is not None and not isinstance(permissions, dict):
        out.append(f"permissions must be an object of {list(PERMISSION_KINDS)}")
    elif permissions:
        unknown = sorted(set(permissions) - set(PERMISSION_KINDS))
        if unknown:
            out.append(f"permissions has unknown kinds {unknown}; the kinds are {list(PERMISSION_KINDS)}")
        for kind in PERMISSION_KINDS:
            value = permissions.get(kind)
            whole = isinstance(value, dict)
            if value is not None and not (_lines(value.get("items") if whole else value)
                                          and isinstance(value.get("then", "") if whole else "", str)
                                          and not (whole and set(value) - {"items", "then"})):
                out.append(f"permissions.{kind} must be a list of rules, or {{items: [rules], then: text}}")
    if person.get("keep_private") is not None and not _lines(person["keep_private"]):
        out.append("keep_private must be a list of non-empty lines")
    if person.get("time_zone") is not None:
        out += zone_problems(person["time_zone"])
    return out


def zone_problems(zone: Any) -> list[str]:
    """What's wrong with an IANA time zone setting: one message, or none."""
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError, TypeError, OSError):  # a region folder ("Pacific") is an OSError
        return [f"time_zone {zone!r} is not an IANA time zone"]
    return []


# ---- one read -----------------------------------------------------------------------------------------------------
def own_line(item: dict) -> bool:
    kind = item.get("kind")
    return kind is None or kind.split(":", 1)[0] in OWN_LINE_KINDS


def _own(read: dict) -> list[dict]:
    return [s for s in read.get("you_sent") or [] if own_line(s)]


def _messaged(answer: Optional[dict]) -> bool:
    answer = answer or {}
    return answer.get("outcome") in ("reply", "done") and bool((answer.get("message") or "").strip())


def _local(when: str, zone: ZoneInfo) -> datetime:
    at = datetime.fromisoformat(when.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError(f"{when!r} has no UTC offset")
    return at.astimezone(zone)


def _hm(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def _stamp(t: datetime, now: datetime) -> str:
    days = (now.date() - t.date()).days
    if days == 0:
        return _hm(t)
    if days < 6:
        return f"{t.strftime('%a')} {_hm(t)}"
    return f"{t.strftime('%a %b')} {t.day}, {_hm(t)}"


def _message(m: dict, stamp: str) -> str:
    """One of the assistant's messages: its time, then its text indented under it."""
    return f"  [{stamp}]\n" + "\n".join("    " + line if line.strip() else "" for line in m["text"].strip().splitlines())


def render(person: dict, history: list[dict], read: dict) -> Optional[str]:
    """The message the user sim gets for ``read``, or None when the assistant sent nothing. An item without ``at``
    shows at the time of the read it came in."""
    messages = [m for m in read.get("assistant_messages") or [] if (m.get("text") or "").strip()]
    if not messages:
        return None
    zone = ZoneInfo(person.get("time_zone") or "UTC")
    now = _local(read["time"], zone)

    def at(item: dict) -> datetime:
        return _local(item.get("at") or read["time"], zone)

    def phone_line(item: dict, full: bool) -> str:
        head, *rest = (item.get("text") or "").split("\n")
        tag = " · ".join(x for x in (_stamp(at(item), now), item.get("channel")) if x)
        return f"  [{tag}] {head}" + (
            "".join(f"\n    {line}" for line in rest) if full else "")

    earlier = [h["read"] for h in history]
    own = _own(read)

    def dated(r: dict, items: list) -> list[dict]:  # an earlier read's items, each without ``at`` at that read's time
        return [item if item.get("at") else {**item, "at": r["time"]} for item in items]

    wrote = [at(s) for r in [*earlier, read] for s in dated(r, _own(r))]
    wrote += [_local(h["read"]["time"], zone) for h in history if _messaged(h.get("answer"))]
    out = [f"It's {now.strftime('%a, %b')} {now.day}, {_hm(now)}."]
    if wrote:
        last = max(wrote)
        days = (now.date() - last.date()).days
        when = (f"at {_hm(last)}" if days == 0 else f"on {last.strftime('%a')} at {_hm(last)}" if days < 6
                else f"on {last.strftime('%a %b')} {last.day} at {_hm(last)}")
        out.append(f"You last messaged your AI assistant {when}.")
    if read.get("phone"):
        out.append("\nYOUR NEW EMAILS, TEXTS AND NOTIFICATIONS (since you last checked):")
        out += [phone_line(item, True) for item in read["phone"]]
    seen = [item for r in earlier for item in dated(r, r.get("phone") or [])]
    recent = [item for item in seen if (now - at(item)).days < EARLIER_DAYS][-EARLIER_MAX:]
    if seen:
        out.append("\nYOUR EARLIER EMAILS, TEXTS AND NOTIFICATIONS (you've already seen these):")
        out += [phone_line(item, False) for item in recent]
        if len(recent) < len(seen):
            out.append(EARLIER_CUT)
    if own:
        out.append("\nYOU SENT YOUR AI ASSISTANT:")
        out += [f"  [{_stamp(at(s), now)}] {s['text']}{attached_note(s.get('files'))}" for s in own]
    out.append("\nNEW FROM YOUR AI ASSISTANT (this is what you're answering):")
    out += [_message(m, _stamp(at(m), now)) for m in messages]
    out.append("\n" + CLOSING.format(first=_first(person)))
    out.append("Answer in JSON: " + ANSWER_LINE)
    return "\n".join(out)


# ---- the answer ---------------------------------------------------------------------------------------------------
def _object(raw: Any) -> Any:
    """The answer object: as given, or the JSON object in the text (in a ```json fence, or after other words)."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder, found, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "outcome" in obj:
            found = obj
        i = text.find("{", end)
    return found


def answer_problems(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["the answer is not a JSON object"]
    out = [f"missing {key}" for key in ANSWER_SCHEMA["required"] if key not in obj]
    out += [f"unknown field {key}" for key in obj if key not in ANSWER_SCHEMA["properties"]]
    if "outcome" in obj and obj["outcome"] not in OUTCOMES:
        out.append(f"outcome {obj['outcome']!r} is not one of {list(OUTCOMES)}")
    out += [f"{key} must be text" for key in ("message", "used") if key in obj and not isinstance(obj[key], str)]
    if "score" in obj and (not isinstance(obj["score"], int) or isinstance(obj["score"], bool)):
        out.append("score must be a whole number")
    return out


def parse(raw: Any) -> dict:
    """The answer as the step records it. A reply with no text is no_reply; a malformed answer is no_reply with its
    ``schema_errors``."""
    obj = _object(raw)
    errors = answer_problems(obj)
    if errors:
        return {"outcome": "no_reply", "message": "", "used": None, "schema_errors": errors}
    outcome = obj["outcome"]
    if outcome == "reply" and not obj["message"].strip():
        outcome = "no_reply"
    out = {"outcome": outcome, "message": obj["message"], "used": obj.get("used"), "schema_errors": []}
    if "score" in obj:
        out["score"] = obj["score"]
    return out


# ---- the plug-in --------------------------------------------------------------------------------------------------
def _check(version: str) -> None:
    if version != VERSION:
        raise ValueError(f"the default user sim has one version, {VERSION!r}, not {version!r}")


def setup(version: str, person: dict) -> dict:
    _check(version)
    found = problems(person)
    if found:
        raise ValueError(f"user sim {person.get('name')!r}: " + "; ".join(found))
    return {"system_prompt": system_prompt(person), "output_format": OUTPUT_FORMAT}


async def answer(version: str, person: dict, history: list[dict], read: dict,
                 ask: Callable[[str], Awaitable[dict]]) -> dict:
    _check(version)
    text = render(person, history, read)
    if text is None:
        return {"outcome": "no_reply", "message": "", "used": None, "input": None, "raw": None, "turns": 0}
    got = await ask(text)
    first = _parsed(got)
    if not first["schema_errors"] or got.get("error"):
        return {**first, "input": text, "raw": got.get("text"), "retried": False}
    again = await ask(RETRY)
    second = _parsed(again)
    if second["schema_errors"]:
        second["schema_errors"] = ([f"first answer: {e}" for e in first["schema_errors"]]
                                   + [f"answer after the retry: {e}" for e in second["schema_errors"]])
    return {**second, "input": text, "raw": again.get("text"), "retried": True, "turns": 2,
            "first_answer": {"raw": got.get("text"), "schema_errors": first["schema_errors"]}}


def _parsed(got: dict) -> dict:
    structured = got.get("structured")
    return parse(structured if isinstance(structured, dict) else got.get("text"))


# ---- the rating after the run -------------------------------------------------------------------------------------
def satisfaction_problems(lines: Any) -> list[str]:
    """What's wrong with the lines a rating goes by: one message, or none."""
    if not (_lines(lines) and lines):
        return ["satisfaction must be a list of non-empty lines: what each end of the scale means for this job"]
    return []


def render_rating(person: dict, satisfaction: list[str], read: dict) -> str:
    """The message that asks for the rating at ``read["time"]``: the assistant's messages the user hasn't read
    (``read["assistant_messages"]``, maybe none), the ``satisfaction`` lines tagged S1..., then the ask."""
    zone = ZoneInfo(person.get("time_zone") or "UTC")
    now = _local(read["time"], zone)
    out = [f"It's {now.strftime('%a, %b')} {now.day}, {_hm(now)}. {RATING_OVER}"]
    messages = [m for m in read.get("assistant_messages") or [] if (m.get("text") or "").strip()]
    if messages:
        out.append(RATING_UNREAD)
        out += [_message(m, _stamp(_local(m.get("at") or read["time"], zone), now)) for m in messages]
    _section(out, RATING_TITLE, RATING_TAG, list(satisfaction))
    out.append("\n" + RATING_ASK.format(first=_first(person)))
    out.append(RATING_ANSWER)
    return "\n".join(out)


def rating_problems(obj: Any) -> list[str]:
    out = answer_problems(obj)
    if out:
        return out
    if "score" not in obj:
        return ["missing score"]
    if obj["score"] not in SCORES:
        out.append(f"score {obj['score']!r} is not one of {list(SCORES)}")
    if not obj["message"].strip():
        out.append("message must give the reason")
    return out


def parse_rating(raw: Any) -> dict:
    """The rating as the step records it: ``score`` (1-5, None when malformed), ``reason``, ``used`` and
    ``schema_errors``."""
    obj = _object(raw)
    errors = rating_problems(obj)
    if errors:
        return {"score": None, "reason": None, "used": None, "schema_errors": errors}
    return {"score": obj["score"], "reason": obj["message"].strip(), "used": obj.get("used"), "schema_errors": []}


async def rate(version: str, person: dict, satisfaction: list[str], read: dict,
               ask: Callable[[str], Awaitable[dict]]) -> dict:
    """The user's rating of the run once it's over, by the ``satisfaction`` lines: one ask, in the run's session, with
    one retry when the answer is malformed. ``read`` is ``{"time": RFC 3339 UTC, "assistant_messages": [{"at",
    "text"}]}``, the messages the user hasn't read. Returns ``score``, ``reason``, ``used``, ``schema_errors``,
    ``input``, ``raw``, ``retried``, ``turns`` and, after a retry, ``first_answer``. The rating is never a reply: the
    caller records it and sends it nowhere."""
    _check(version)
    found = satisfaction_problems(satisfaction)
    if found:
        raise ValueError("; ".join(found))
    text = render_rating(person, satisfaction, read)
    got = await ask(text)
    first = _parsed_rating(got)
    if not first["schema_errors"] or got.get("error"):
        return {**first, "input": text, "raw": got.get("text"), "retried": False, "turns": 1}
    again = await ask(RATING_RETRY)
    second = _parsed_rating(again)
    if second["schema_errors"]:
        second["schema_errors"] = ([f"first answer: {e}" for e in first["schema_errors"]]
                                   + [f"answer after the retry: {e}" for e in second["schema_errors"]])
    return {**second, "input": text, "raw": again.get("text"), "retried": True, "turns": 2,
            "first_answer": {"raw": got.get("text"), "schema_errors": first["schema_errors"]}}


def _parsed_rating(got: dict) -> dict:
    structured = got.get("structured")
    return parse_rating(structured if isinstance(structured, dict) else got.get("text"))
