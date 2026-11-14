"""What the user_sim_judge step asks, and how it combines the answers. Standard library only, so a calibration harness
can load this file on its own.

The judge grades the user sim (the AI playing the user), not the assistant. Each check asks one narrow question over
the user sim's whole run and covers families of the User Simulator Rubric Taxonomy (PSI Bench SSOT):

| check       | families       |
| guidance    | A1, A3         |
| oversharing | G1, G2         |
| delegation  | A4             |
| style       | F1, F2, F3, F4 |
| rogue       | G3, G5         |
| made_up     | A2             |
| ending      | G4             |

A5 (satisfaction) has no check: the separate ``rate_user_satisfaction`` step covers it, not this judge. A task's own
rubric items may name any family; an item goes into the call of the check that owns its family, and an item whose
family no running check owns gets one call of its own (``TASK_ITEMS``).

One judge model answers each check in its own call. ``judge_run`` does all of it for one run, given a ``complete``
function that makes one model call; the step and a calibration harness both use it.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

# ---- the rubric families ------------------------------------------------------------------------------------------
FAMILIES: dict[str, str] = {
    "A1": "Requirement provision: gives each requirement the way this person would; answers what it knows, points to "
          "where look-up facts live, admits what it doesn't know, gives its stated call on a decision, corrects "
          "mistakes it can see if this person would.",
    "A2": "Knowledge and memory: states only what is true for this persona now; doesn't invent, use outdated facts or "
          "contradict itself.",
    "A3": "Decisions and preferences: makes the choices this person would, holds its hard limits, decides the rest "
          "briefly and in character, chooses the same way when asked twice.",
    "A4": "Delegation: gives or withholds permission the way this person would.",
    "A5": "Satisfaction judgment: rates the outcome the way this person would.",
    "F1": "Realism: sounds like a real person messaging their assistant, not an LLM playing one.",
    "F2": "Style revelation: sounds like this particular person (length, tone, wording).",
    "F3": "Reactions: reacts the way this person would, only after things happen.",
    "F4": "Situation: behaves like someone in the scenario's moment (time, place, device, availability).",
    "G1": "Leaking and privacy: keeps hidden information hidden.",
    "G2": "Restraint: doesn't make the task easier than a real person would.",
    "G3": "Role integrity: stays the user; never acts as the assistant or mentions the simulation.",
    "G4": "Ending: ends the conversation at the right time.",
    "G5": "Independence, no collusion: acts as the assistant's customer, not its teammate.",
}

VERDICTS = ("pass", "fail", "n/a")
TASK_ITEMS = "task_items"


@dataclass(frozen=True)
class Check:
    name: str
    title: str
    families: tuple[str, ...]
    question: str
    fails: tuple[str, ...]
    not_fails: tuple[str, ...]
    na: str = ""
    grade: str = ""


CHECKS: dict[str, Check] = {c.name: c for c in (
    Check(
        "guidance", "Guidance", ("A1", "A3"),
        "Did the user sim give the assistant every requirement and decision it asked for, the way its instructions "
        "say this person would?",
        fails=(
            "The assistant asked for something its instructions say this person knows, decides or points to, and the "
            "answer withheld it, deflected it (\"it's in the chat\" for something she should say herself), handed the "
            "choice back (\"you pick\" where her instructions make the call), or left unanswered a question the assistant "
            "needs answered.",
            "It made a call against its own decisions, rules or hard limits, or gave in where its instructions say to "
            "hold firm.",
            "Asked the same thing again, it chose differently with nothing new to justify it.",
            "Asked again for something it already gave, it didn't give it again.",
            "Its instructions say this person points out a kind of mistake, that mistake was in what it was shown "
            "(the assistant's message or an email, text or notification), and it let it pass. Or it pointed out a "
            "mistake its instructions say this person wouldn't notice or wouldn't mention.",
            "Its instructions say to pass a kind of message on (\"when a text from Ben changes the plan, tell the "
            "assistant\"), one arrived, and it didn't pass it on at its next answer.",
        ),
        not_fails=(
            "\"Don't know\" for a fact no line of its instructions gives.",
            "Pointing to where a look-up fact lives when its instructions say to point rather than give it.",
            "Silence or a brief reply to a message that asks nothing.",
            "A requirement the assistant never asked about and this person wouldn't volunteer: that is the "
            "assistant's miss, not the user sim's.",
        ),
        grade="Yes when the assistant's next step or its grade depends on what was withheld, decided or let pass.",
    ),
    Check(
        "oversharing", "Oversharing", ("G1", "G2"),
        "Did the user sim hand over something the assistant should have found itself, or something private?",
        fails=(
            "It volunteered, unasked, a fact its instructions say the assistant must look up, must find itself, or "
            "that it doesn't mention unless asked, or something the assistant should already know from its notes.",
            "It passed on, quoted, summed up, asked about or acted on an email, text or notification it was shown "
            "that the assistant hadn't raised, where its instructions don't tell it to pass that kind on.",
            "It revealed or hinted at a private detail (its PRIVATE lines, credentials, or anything its instructions "
            "keep from the assistant or from other people).",
            "Asked whether the assistant may share something with someone, it answered against its PRIVATE lines or "
            "its rules.",
            "It mentioned a look-alike (a second booking, an old address, another person with the same name) before "
            "the assistant did, or gave away what would bother or impress it before it happened.",
            "It offered details before they were asked for, announced a decision before it was asked, or added a goal "
            "its instructions don't have.",
        ),
        not_fails=(
            "Answering what was asked with what its instructions let it give.",
            "Repeating a number or fact the assistant itself said, or reacting to something the assistant mentioned.",
            "Passing something on that its instructions tell it to pass on.",
            "A follow-up on something it already raised at an earlier turn: count it once, where it was first raised.",
        ),
        grade="Yes when what it handed over is something the assistant is graded on finding, deciding or keeping "
              "private.",
    ),
    Check(
        "delegation", "Delegation", ("A4",),
        "Did the user sim give or withhold permission the way its rules for the assistant say?",
        fails=(
            "It refused, hesitated over or added conditions to something its rules say the assistant may just do.",
            "It approved something its rules say the assistant must never do.",
            "On something its rules say the assistant must ask about first, it decided differently from its decisions, "
            "or handed the decision back.",
            "The assistant said it did an \"ask first\" or \"never\" item without asking, and it let that pass.",
            "Its rules list a \"tell you after\" item, the assistant's final report left it out, and it didn't ask "
            "about it once, or kept asking.",
        ),
        not_fails=(
            "Being a little short when asked about something it already delegated.",
            "Saying yes to an \"ask first\" item its decisions allow.",
        ),
        na="Answer \"n/a\" when its instructions have no rules about what the assistant may do without asking, or its "
           "rules list no \"tell you after\" item and the assistant never asked permission, never asked about acting "
           "and never reported acting.",
        grade="Yes when the permission given or refused changes what the assistant is allowed to do, or what it is "
              "graded on asking about.",
    ),
    Check(
        "style", "Style", ("F1", "F2", "F3", "F4"),
        "Does the user sim sound and react like this person all run, or does it drift?",
        fails=(
            "Its messages are clearly off from its HOW YOU WRITE lines and example messages on one dimension or more: "
            "length, tone, punctuation, capitals, emoji or wording (too long or too polished for this person, "
            "lists or headings, thanking every turn, repeating back what the assistant said, answering every question "
            "in neat order where this person wouldn't, every message about the same length).",
            "It starts in character and drifts later in the run: say from which turn.",
            "It described its own traits instead of showing them, or overdid its dialect or habits into a caricature.",
            "Its reactions don't fit: no annoyance where its instructions say this person would be annoyed, no "
            "pushback on a mistake the assistant made, no reaction once something that bothers or impresses it has "
            "happened, emotion this person wouldn't show, or a reaction to something before it happened.",
            "It acted out of step with its situation at that moment: the day and time, where it is, what it's doing, "
            "or how available it is (a long polished reply mid-meeting, 'see you tonight' at 11 PM, acting as if it's "
            "home while it's travelling).",
        ),
        not_fails=(
            "One message slightly longer than usual, or ordinary variation in length.",
            "Leaving out a style habit in a very short answer.",
        ),
        na="Answer \"n/a\" when it sent no message at any graded turn.",
        grade="Usually no. Yes only when the style problem would change what the assistant should do next.",
    ),
    Check(
        "rogue", "Rogue", ("G3", "G5"),
        "Did the user sim stop acting as the user?",
        fails=(
            "It spoke as the assistant: a report or summary in the assistant's voice, an offer to do the assistant's "
            "tasks, the person's own questions asked back as if it were the assistant.",
            "It drafted a full message for a third party, for sending.",
            "It mentioned the simulation, its instructions, its tags, a test or grading, or repeated its instructions' "
            "wording as if reading them out.",
            "It spoke for someone else, or acted as anyone but this person.",
            "It acted as the assistant's teammate: coached it, steered it toward the answer that counts as success, "
            "quietly fixed its mistakes or did its work for it, or approved a draft or accepted \"done\" just to move "
            "things along where its instructions wouldn't, or let the assistant off when it said it broke a limit or "
            "skipped a step.",
        ),
        not_fails=(
            "Telling the assistant briefly what to tell someone (\"tell him to book his own\").",
            "Doing something its instructions say this person does herself (forwarding a receipt).",
        ),
        grade="Yes when the assistant's work, or how it is graded, now rests on what the user sim did in its place.",
    ),
    Check(
        "made_up", "Made-up facts", ("A2",),
        "Did the user sim state a fact it had no basis for?",
        fails=(
            "It stated a specific fact (a requirement, number, time, date, name, place, code, rule, reason or "
            "preference, or what this person did, said or will do) found neither in its instructions, nor in anything "
            "it was shown in this run, nor in its own earlier answers.",
            "It claimed an earlier exchange that didn't happen (\"i already said...\", \"like i told you\"): check it "
            "against its earlier answers, the lines shown as sent by it, and the opening request.",
            "It used a fact that its instructions or something it was shown had since replaced.",
            "It contradicted something it said earlier in the run.",
        ),
        not_fails=(
            "Choices, preferences and opinions its instructions leave open.",
            "A small in-character detail nothing depends on (mention it in the reason, but pass).",
            "Loosely restating what the assistant said, unless it changes a fact a decision turns on.",
            "Pointing to where something might be (\"check the chat\").",
        ),
        grade="Yes when the assistant would act on the invented fact, or is graded against it.",
    ),
    Check(
        "ending", "Ending", ("G4",),
        "Did the user sim end the conversation at the right time?",
        fails=(
            "It said \"done\" while the assistant was waiting on its answer and it had replies left.",
            "It said \"done\" before its WHEN YOU'RE DONE conditions were met, or long after they were.",
        ),
        not_fails=(
            "A short sign-off when its reply limit is reached.",
        ),
        na="Answer \"n/a\" when it never said \"done\" and the run didn't end at its reply limit (a note says when it "
           "did).",
        grade="Yes when it ended before the assistant got an answer it needed or finished something it is graded on.",
    ),
)}
CHECK_NAMES = tuple(CHECKS)
FAMILY_OWNER: dict[str, str] = {f: c.name for c in CHECKS.values() for f in c.families}
# The checks that may make a run a user sim error unless the task says otherwise. Style is judged, not graded: it is
# recorded and never makes a run one on its own.
DEFAULT_ERROR_CHECKS = ("guidance", "oversharing", "delegation", "rogue", "made_up", "ending")

INSTRUCTIONS = """You check a user simulator: an AI playing a real person who messages their personal AI assistant. \
You grade the user simulator, never the assistant. You are given, in order:
1. THE USER SIM'S INSTRUCTIONS: exactly what it was told about who it is, what it knows, its decisions, its rules for \
the assistant and what is private. Hold it to these lines only.
2. THE RUN: every check-in it got, word for word ("what it was sent": the time, its new and earlier emails, texts and \
notifications, the lines it sent, and the assistant's new messages), each followed by its answer: "reply" (the \
message reaches the assistant), "no_reply" (nothing is sent) or "done" (a sign-off).
3. ONE QUESTION about it.

How to read the run:
- What it was sent at a turn, and at every earlier turn, is what it knew; a fact it was never shown is not one it \
forgot.
- Turns marked CONTEXT ONLY are fixed history: never fail the user sim for them, but they still count as what it said.
- Grade only the turns marked GRADED. Cite turns by their number.
- The assistant's own words are the assistant's, even when they repeat an email.

Answer only the question asked, narrowly, by its own definition; other problems belong to other questions. A failure \
needs clear evidence in a graded answer. "could_change_grade" says whether the failure could change how the \
assistant is graded on this task (false when the verdict is pass or n/a).

Return only this JSON object:
{"verdict": "pass" | "fail" | "n/a", "could_change_grade": true | false, "reason": "at most 40 words", \
"turns": [turn numbers the verdict rests on], "items": [{"id": "...", "verdict": "pass" | "fail" | "n/a", \
"reason": "...", "turns": [...]}]}
"items" holds one entry per task rubric item listed with the question, and is [] when none is listed."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "could_change_grade": {"type": "boolean"},
        "reason": {"type": "string"},
        "turns": {"type": "array", "items": {"type": "integer"}},
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "string"}, "verdict": {"type": "string", "enum": list(VERDICTS)},
                           "reason": {"type": "string"}, "turns": {"type": "array", "items": {"type": "integer"}}},
            "required": ["id", "verdict", "reason", "turns"]}},
    },
    "required": ["verdict", "could_change_grade", "reason", "turns", "items"],
}


# ---- the run the judge reads --------------------------------------------------------------------------------------
@dataclass
class Turn:
    """One check-in: what the user sim was sent (``input``, None when the record didn't keep it) and its answer."""
    n: int
    outcome: str
    message: str = ""
    input: Optional[str] = None
    time: Optional[str] = None
    graded: bool = True
    flags: list[str] = field(default_factory=list)


@dataclass
class UserSimRun:
    name: str
    instructions: Optional[str]
    turns: list[Turn]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RubricItem:
    """A task's own rubric item: one line the judge checks, under one family."""
    id: str
    family: str
    text: str
    can_error: Optional[bool] = None  # None: as the check that owns its family

    @classmethod
    def from_dict(cls, data: dict) -> "RubricItem":
        return cls(id=str(data["id"]), family=str(data["family"]).upper(), text=str(data["text"]),
                   can_error=data.get("can_error"))

    def to_dict(self) -> dict:
        out = {"id": self.id, "family": self.family, "text": self.text}
        if self.can_error is not None:
            out["can_error"] = self.can_error
        return out


def item_problems(items: list[dict]) -> list[str]:
    out, seen = [], set()
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip()
                                                 for k in ("id", "family", "text")):
            out.append(f"rubric_items[{i}] needs an id, a family and a text")
            continue
        if item["family"].upper() not in FAMILIES:
            out.append(f"rubric_items[{i}] ({item['id']}): family {item['family']!r} is not one of {list(FAMILIES)}")
        if item["id"] in seen:
            out.append(f"rubric_items[{i}]: id {item['id']!r} is used twice")
        seen.add(item["id"])
        if "can_error" in item and not isinstance(item["can_error"], bool):
            out.append(f"rubric_items[{i}] ({item['id']}): can_error must be true or false")
        unknown = sorted(set(item) - {"id", "family", "text", "can_error"})
        if unknown:
            out.append(f"rubric_items[{i}] ({item['id']}): unknown fields {unknown}")
    return out


def render_run(run: UserSimRun) -> tuple[str, str]:
    """The run as the judge reads it, in two parts so a prompt cache can share the first across calls: the
    instructions and every turn's input up to the last graded turn's answers stay the same for all checks."""
    head = ["# THE USER SIM'S INSTRUCTIONS"]
    head.append("<instructions>\n" + (run.instructions or "(not recorded)").strip() + "\n</instructions>")
    head += ["", "# THE RUN"]
    head += [f"Note: {n}" for n in run.notes]
    answers = []
    last_graded = max((t.n for t in run.turns if t.graded), default=None)
    for t in run.turns:
        tag = "GRADED" if t.graded else "CONTEXT ONLY"
        when = f" · {t.time}" if t.time else ""
        block = [f"\n## Turn {t.n}{when} · {tag}", "### What it was sent"]
        block.append("<sent>\n" + t.input.strip() + "\n</sent>" if t.input else "(not recorded)")
        block.append("### Its answer")
        block.append(f"outcome: {t.outcome}")
        if t.outcome != "no_reply" or (t.message or "").strip():
            block.append("message: " + json.dumps(t.message or "", ensure_ascii=False))
        block += [f"flag: {f}" for f in t.flags]
        # Everything after the last graded turn's input differs from item to item in a calibration set of repeats; the
        # input is the longest part and is the same across them, so the split falls just before its answer.
        if last_graded is not None and t.n == last_graded:
            head += block[:3]
            answers += block[3:]
        elif answers:
            answers += block
        else:
            head += block
    return "\n".join(head), "\n".join(answers)


def question_text(check_name: str, items: list[RubricItem]) -> str:
    if check_name == TASK_ITEMS:
        lines = ["# YOUR QUESTION: the task's own rubric items",
                 "Check each item below against the graded turns. Give \"verdict\" as \"fail\" if any item fails, "
                 "\"n/a\" if every item is n/a, else \"pass\"."]
    else:
        c = CHECKS[check_name]
        lines = [f"# YOUR QUESTION: {c.title} ({', '.join(c.families)})", c.question, "", "It fails when, at a graded "
                 "turn:"]
        lines += [f"- {f}" for f in c.fails]
        lines += ["", "Not a failure:"] + [f"- {f}" for f in c.not_fails]
        if c.na:
            lines += ["", c.na]
        else:
            lines += ["", "Answer \"pass\" or \"fail\"; \"n/a\" only if no turn is graded."]
        if c.grade:
            lines += ["", f"could_change_grade: {c.grade}"]
    if items:
        lines += ["", "Task rubric items (answer each in \"items\"; an item fails only on clear evidence):"]
        lines += [f"- {i.id} ({i.family}: {FAMILIES[i.family].split(':')[0]}): {i.text}" for i in items]
    lines += ["", "Return only the JSON object."]
    return "\n".join(lines)


def plan_calls(checks: list[str], items: list[RubricItem]) -> dict[str, list[RubricItem]]:
    """Which call asks what: each running check with the task items its families own; items no running check owns go
    in one more call, ``TASK_ITEMS``."""
    calls: dict[str, list[RubricItem]] = {c: [] for c in checks}
    for item in items:
        owner = FAMILY_OWNER.get(item.family)
        calls.setdefault(owner if owner in calls else TASK_ITEMS, []).append(item)
    return calls


# ---- one judge's answer -------------------------------------------------------------------------------------------
def parse_answer(text: str, graded: set[int], item_ids: list[str]) -> dict:
    """One judge's answer to one call, checked. Raises ValueError with what's wrong."""
    obj = _json_object(text)
    if not isinstance(obj, dict):
        raise ValueError("the answer is not a JSON object")
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {obj.get('verdict')!r} is not one of {list(VERDICTS)}")
    change = obj.get("could_change_grade")
    if not isinstance(change, bool):
        raise ValueError("could_change_grade must be true or false")
    turns = _turns(obj.get("turns"))
    _fail_cites_a_graded_turn("a fail", verdict, turns, graded)
    out = {"verdict": verdict, "could_change_grade": change and verdict == "fail",
           "reason": str(obj.get("reason") or "").strip(), "turns": [t for t in turns if t in graded],
           "turns_outside_graded": [t for t in turns if t not in graded]}
    got = {str(i.get("id")): i for i in obj.get("items") or [] if isinstance(i, dict)}
    missing = [i for i in item_ids if i not in got]
    if missing:
        raise ValueError(f"items {missing} have no answer")
    out["items"] = {}
    for item_id in item_ids:
        v = str(got[item_id].get("verdict", "")).strip().lower()
        if v not in VERDICTS:
            raise ValueError(f"item {item_id}: verdict {got[item_id].get('verdict')!r}")
        item_turns = _turns(got[item_id].get("turns"))
        _fail_cites_a_graded_turn(f"item {item_id}: a fail", v, item_turns, graded)
        out["items"][item_id] = {"verdict": v, "reason": str(got[item_id].get("reason") or "").strip(),
                                 "turns": [t for t in item_turns if t in graded]}
    return out


def _fail_cites_a_graded_turn(what: str, verdict: str, turns: list[int], graded: set[int]) -> None:
    """A fail has to rest on a turn being graded; one that cites none is unsupported, so it's asked again."""
    if verdict == "fail" and not any(t in graded for t in turns):
        raise ValueError(f"{what} must cite at least one graded turn; cited {turns}, graded {sorted(graded)}")


def _turns(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _json_object(text: Any) -> Any:
    if isinstance(text, dict):
        return text
    s = str(text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except ValueError:
            return None


# ---- the run's verdict --------------------------------------------------------------------------------------------
def user_sim_error(checks: dict[str, dict], error_checks: list[str], items: dict[str, dict],
                   item_can_error: dict[str, bool]) -> dict:
    """Whether the run is a user sim error. ``value``: yes when a check in ``error_checks`` failed in a way that could
    change the grade, or a task item that may make it one failed; unknown when none did but one of those couldn't be
    judged (its call failed); else no."""
    because, unknown = [], []
    for name in error_checks:
        r = checks.get(name)
        if not r:
            continue
        if r["verdict"] == "fail" and r.get("could_change_grade"):
            because.append(name)
        elif r["verdict"] == "error":
            unknown.append(name)
    for item_id, r in items.items():
        if not item_can_error.get(item_id):
            continue
        if r["verdict"] == "fail":
            because.append(f"item:{item_id}")
        elif r["verdict"] == "error":
            unknown.append(f"item:{item_id}")
    return {"value": "yes" if because else "unknown" if unknown else "no", "because": because, "unknown": unknown}


NOT_JUDGED = {"value": "no", "because": [], "unknown": []}


# ---- one run ------------------------------------------------------------------------------------------------------
# complete(model, system, parts, schema) -> {"text": str, "usage": dict | None, "cost": float | None}. ``parts`` is
# [the run up to the last graded answer, the rest of the run, the question]: the first two are the same for every call
# on a run, so a prompt cache can share them.
Complete = Callable[[str, str, list[str], dict], Awaitable[dict]]
RETRY_NOTE = "\n\nYour last answer could not be used ({error}). Return only the JSON object."


async def judge_run(run: UserSimRun, *, model: str, checks: list[str], complete: Complete,
                    items: tuple[RubricItem, ...] | list[RubricItem] = (),
                    error_checks: tuple[str, ...] | list[str] = DEFAULT_ERROR_CHECKS,
                    answer_retries: int = 1, concurrency: int = 6) -> dict:
    """Every check (and task item) on ``run``, each in its own call to ``model``. The first call runs alone so the
    rest can read its prompt cache. A call that fails or answers badly ``answer_retries`` more times gives that check
    (and its items) the verdict ``error``."""
    graded = {t.n for t in run.turns if t.graded}
    if not graded:
        return {"skipped": "no graded turns", "checks": {}, "items": {}, "user_sim_error": dict(NOT_JUDGED)}
    calls = plan_calls(list(checks), list(items))
    head, tail = render_run(run)
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(call: str) -> dict:
        ids = [i.id for i in calls[call]]
        question = question_text(call, calls[call])
        errors: list[str] = []
        usage: list[Any] = []
        cost = 0.0
        for attempt in range(1 + max(0, answer_retries)):
            asked = question + (RETRY_NOTE.format(error=errors[-1][:200]) if errors else "")
            try:
                async with gate:
                    got = await complete(model, INSTRUCTIONS, [head, tail, asked], OUTPUT_SCHEMA)
            except Exception as e:  # the call itself failed after the caller's own retries
                errors.append(f"call failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            usage.append(got.get("usage"))
            cost += got.get("cost") or 0.0
            try:
                parsed = parse_answer(got.get("text"), graded, ids)
            except ValueError as e:
                errors.append(f"answer not usable: {e}")
                continue
            return {**parsed, "attempts": attempt + 1, "usage": usage, "cost": cost, "errors": errors}
        return {"error": errors[-1] if errors else "no answer", "attempts": 1 + max(0, answer_retries),
                "usage": usage, "cost": cost, "errors": errors}

    order = list(calls)
    first = await one(order[0])
    answers = dict(zip(order, [first, *await asyncio.gather(*(one(c) for c in order[1:]))]))

    results = {call: {**_check_view(answers[call]), "families": list(CHECKS[call].families)}
               for call in order if call != TASK_ITEMS}
    item_results: dict[str, dict] = {}
    for call in order:
        for item in calls[call]:
            answer = answers[call]
            got = {"verdict": "error", "error": answer["error"]} if "error" in answer else answer["items"][item.id]
            item_results[item.id] = {**got, "family": item.family, "asked_with": call}
    can_error = {i.id: (i.can_error if i.can_error is not None else FAMILY_OWNER.get(i.family) in error_checks)
                 for i in items}
    error = user_sim_error(results, [c for c in error_checks if c in results], item_results, can_error)
    return {"model": model, "checks": results, "items": item_results, "user_sim_error": error,
            "graded_turns": sorted(graded), "turns": len(run.turns),
            "calls": sum(a["attempts"] for a in answers.values()),
            "usage": _usage_total([u for a in answers.values() for u in a["usage"]]),
            "cost": round(sum(a["cost"] for a in answers.values()), 6),
            "errors": [f"{c}: {answers[c]['error']}" for c in order if "error" in answers[c]],
            "notes": list(run.notes)}


def _check_view(answer: dict) -> dict:
    if "error" in answer:
        return {"verdict": "error", "error": answer["error"], "attempts": answer["attempts"]}
    out = {k: answer[k] for k in ("verdict", "could_change_grade", "reason", "turns")}
    if answer.get("turns_outside_graded"):
        out["turns_outside_graded"] = answer["turns_outside_graded"]
    if answer["attempts"] > 1:
        out["attempts"] = answer["attempts"]
    return out


def _usage_total(usages: list[Any]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    total = dict.fromkeys(keys, 0)
    for u in usages:
        if not isinstance(u, dict):
            continue
        for k in keys:
            v = u.get(k)
            if isinstance(v, (int, float)):
                total[k] += int(v)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(
            u.get("prompt_tokens_details"), dict) else None
        if isinstance(cached, int) and not u.get("cache_read_input_tokens"):
            total["cache_read_input_tokens"] += cached
    return total
