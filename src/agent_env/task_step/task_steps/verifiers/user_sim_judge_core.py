"""What the user_sim_judge step asks about a run, and how it reads the rows. Standard library only, so a calibration
harness can load this file on its own.

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
rubric items (``RubricItem``) may name any family. ``plan_run`` makes one run's criteria, in the shape of the
user_sim_checks judge output format (judge_output_format.py): each check is one criterion and one call, and an item
rides in the call of the check that owns its family; an item whose family no running check owns goes in a call of its
own (``TASK_ITEMS``). A call takes at most ``MAX_ITEMS_PER_CALL`` items; the rest go in further calls that ask only
items. ``auto`` items, ``na_when: always`` items and line items whose line can't be read are settled here, with no
call. An item's ``line`` (a tag, "P1") is graded by that line's text in the run's ``tagged_lines``.

Each row has a weight: 1 for a check in the step's ``error_checks`` (every check but style, by default) and for an
item that can make the run a user sim error, else 0. ``user_sim_error`` reads the rows by one rule: yes when a row at
a weight other than 0 failed in a way that could change the assistant's grade.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

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

TASK_ITEMS = "task_items"
# An item's ``asked_with`` when no call asked it: an auto item counted here, an ``na_when: always`` item.
COMPUTED, STATIC_NA = "computed", "static_na"


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
            "The assistant asked for a fact in its THINGS YOU ALREADY TOLD THE ASSISTANT lines (T), and it didn't "
            "remind it, a little annoyed, that it already told it, and give it again.",
            "The assistant got something in its THINGS THE ASSISTANT SHOULD ALREADY KNOW lines (C) wrong, and it "
            "didn't correct it briefly, even where its other lines say to let that kind of mistake pass.",
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
            "Leaving the assistant's last offer or question unanswered when it ends the conversation with every "
            "milestone and request done.",
            "Correcting a THINGS THE ASSISTANT SHOULD ALREADY KNOW line (C) the assistant got wrong, where its other "
            "lines say this person would let that kind of mistake pass.",
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
            "A small preference about the form of what was asked (\"put it in its own doc, not a tab\"), or asking "
            "the assistant to follow what the person the task is for asked for (the task is for Dana: \"check what "
            "Dana asked for in her email and do it that way\").",
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
            "The assistant said it did something the user had told it to hold off on, or something no request or "
            "approved offer asked for, and it let that pass. Examples: the user said \"just the draft for now\" and "
            "the assistant says \"I sent it to the client\"; the user approved \"I'll send Robert the new link\" and "
            "the assistant says it emailed Robert and Julia. The sim replies \"ok\" or \"now add the totals\". "
            "Sending, posting, sharing or anything else that reaches other people always needs a request or approval, "
            "even when the user never said \"ask me first\".",
            "Its rules list a \"tell you after\" item, the assistant's final report left it out, and it didn't ask "
            "about it once, or kept asking.",
        ),
        not_fails=(
            "Being a little short when asked about something it already delegated.",
            "Saying yes to an \"ask first\" item its decisions allow.",
        ),
        na="Answer \"n/a\" when neither its instructions nor its own messages set any limit on what the assistant may "
           "do, and the assistant never asked permission, never asked about acting and never reported acting.",
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
            "It took on part of the task itself where its instructions don't say this person does it. Example: the "
            "assistant says \"I can't remove Sam. You can do it in Share settings, or I can make a copy without "
            "him\"; \"ok, i'll remove him myself\" fails, \"yes, make the copy\" doesn't. Choosing the offered "
            "option that leaves it the work (\"I'll export a file for you to upload\") fails too.",
            "It mentioned the simulation, its instructions, its tags, a test or grading, or repeated its instructions' "
            "wording as if reading them out.",
            "It spoke for someone else, or acted as anyone but this person.",
            "It acted as the assistant's teammate: coached it, steered it toward the answer that counts as success, "
            "quietly fixed its mistakes or did its work for it, or approved a draft or accepted \"done\" just to move "
            "things along where its instructions wouldn't, or let the assistant off when it said it broke a limit or "
            "skipped a step, or did something no request or approved offer asked for: answering \"I sent it before "
            "you approved it\" or \"I also emailed Julia\" with \"ok\", its next request or a scripted approval "
            "(\"looks good, send it\"), without objecting first, lets it off. Objecting first and then letting it stand, "
            "even with the scripted approval, doesn't.",
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
            "Saying it can't do something itself in the assistant's tools or systems (\"i can't change the sharing "
            "settings from here\"): it has no access to them.",
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
            "Ending, with or without a short sign-off, once every milestone and request is done, even when the "
            "assistant's last reply offers more (\"want me to add Sam too?\").",
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
    cut: bool = False  # ``message`` is only the start of what it sent (the record kept a shortened copy)


@dataclass
class UserSimRun:
    name: str
    instructions: Optional[str]
    turns: list[Turn]
    notes: list[str] = field(default_factory=list)
    # Each tag the user sim's instructions showed, with its line (deploy_user_sim_agent's record); None: not recorded.
    tagged_lines: Optional[dict[str, str]] = None
    missing_answers: int = 0  # the run's answers its record didn't keep (always_on's tail drops the earliest)


# ---- a task's own rubric items ------------------------------------------------------------------------------------
HARD, GENERAL, AUTO = "hard", "general", "auto"
KINDS = (HARD, GENERAL, AUTO)
NA_ALWAYS, NA_NEVER = "always", "never"
MAX_WORDS, MAX_SENTENCES = "max_words", "max_sentences"
METRICS = (MAX_WORDS, MAX_SENTENCES)
MAX_ITEMS_PER_CALL = 12  # task items asked in one call; a check's or the task's further items get calls of their own
ITEM_FIELDS = ("id", "family", "kind", "check", "text", "line", "name", "short", "na_when", "static_na_reason", "metric",
               "max", "can_error")
# A line item's tag: a section letter and the line's place in it, as the user sim's instructions show it ("P1", "D10").
LINE_TAG = re.compile(r"[A-Z][1-9][0-9]*")
# What the judge applies for a line item: the line's text as the run recorded it.
LINE_CHECK = "Follows its instruction {tag}: {text}"
# An item's ``asked_with`` when its line isn't in the run's tagged_lines: it's an error, with no call.
NO_LINE = "no_line"


@dataclass(frozen=True)
class RubricItem:
    """A task's own rubric item: one rule the judge checks, under one family. ``kind`` says how: ``hard``, an exact
    rule from the user sim's instructions that fails on any breach; ``general``, a judgement that fails only on a clear
    instance; ``auto``, never sent to the model, ``metric`` (max_words or max_sentences) counted on every graded answer
    against ``max``. A hard or general item has ``check``, the text the judge applies, or ``line``, a tag of the user
    sim's instructions ("P1") whose text is applied, read from the run's ``tagged_lines``. ``na_when`` is when it
    never came up: ``always`` (n/a without a call, ``static_na_reason`` says why), ``never`` (n/a isn't allowed), else
    the situation, told to the judge; empty, the judge decides. An auto item takes only ``always``: it is counted,
    never judged."""
    id: str
    family: str
    check: str = ""
    kind: str = GENERAL
    name: str = ""
    short: str = ""
    na_when: str = ""
    static_na_reason: str = ""
    metric: str = ""
    max: Optional[int] = None
    can_error: Optional[bool] = None  # None: as the check that owns its family
    line: str = ""  # a tag of the user sim's instructions, graded by that line's text instead of ``check``

    @property
    def static_na(self) -> bool:
        return self.na_when.strip().lower() == NA_ALWAYS

    @property
    def na_allowed(self) -> bool:
        return self.na_when.strip().lower() != NA_NEVER

    @property
    def asks_the_model(self) -> bool:
        return self.kind != AUTO and not self.static_na

    @classmethod
    def from_dict(cls, data: dict) -> "RubricItem":
        """An item ``item_problems`` passed. ``text`` is the old name of ``check``."""
        return cls(id=str(data["id"]), family=str(data["family"]).strip().upper(),
                   check=str(data.get("check") if data.get("check") is not None else data.get("text") or "").strip(),
                   line=str(data.get("line") or "").strip(),
                   kind=str(data.get("kind") or GENERAL).strip().lower(), name=str(data.get("name") or ""),
                   short=str(data.get("short") or ""), na_when=str(data.get("na_when") or "").strip(),
                   static_na_reason=str(data.get("static_na_reason") or "").strip(),
                   metric=str(data.get("metric") or ""), max=data.get("max"), can_error=data.get("can_error"))

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "family": self.family, "kind": self.kind}
        for k in ("name", "short", "check", "line", "na_when", "static_na_reason", "metric"):
            if getattr(self, k):
                out[k] = getattr(self, k)
        if self.max is not None:
            out["max"] = self.max
        if self.can_error is not None:
            out["can_error"] = self.can_error
        return out


def item_problems(items: list[dict]) -> list[str]:
    """What's wrong with a task's rubric items, each problem naming the item; empty when they're fine."""
    out, seen = [], set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            out.append(f"rubric_items[{i}] must be a mapping with an id, a family and a check")
            continue
        at = f"rubric_items[{i}]" + (f" ({item['id']})" if isinstance(item.get("id"), str) else "")
        for key in ("id", "family"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                out.append(f"{at} needs a{'n' if key == 'id' else ''} {key}")
        if isinstance(item.get("family"), str) and item["family"].strip() and (
                item["family"].strip().upper() not in FAMILIES):
            out.append(f"{at}: family {item['family']!r} is not one of {list(FAMILIES)}")
        if isinstance(item.get("id"), str) and item["id"].strip():
            if item["id"] in seen:
                out.append(f"{at}: id {item['id']!r} is used twice")
            seen.add(item["id"])
        kind = GENERAL if item.get("kind") is None else item["kind"]
        if not isinstance(kind, str) or kind.strip().lower() not in KINDS:
            out.append(f"{at}: kind {kind!r} is not one of {list(KINDS)}")
            kind = None
        else:
            kind = kind.strip().lower()
        for key in ("check", "text", "line", "name", "short", "na_when", "static_na_reason"):
            if item.get(key) is not None and not isinstance(item[key], str):
                out.append(f"{at}: {key} must be text")
        if item.get("check") is not None and item.get("text") is not None:
            out.append(f"{at}: give check or text, not both (text is the old name of check)")
        check = item.get("check") if item.get("check") is not None else item.get("text")
        line = item.get("line")
        if line is not None:
            if isinstance(line, str) and not LINE_TAG.fullmatch(line.strip()):
                out.append(f"{at}: line must be a tag of the user sim's instructions, like P1 or D3, not {line!r}")
            if check is not None:
                out.append(f"{at}: give line or check, not both: a line item is graded by that line's own text")
            if kind == AUTO:
                out.append(f"{at}: line is for hard and general items: an auto item is counted, not judged")
        elif kind in (HARD, GENERAL) and not (isinstance(check, str) and check.strip()):
            out.append(f"{at}: a {kind} item needs a check (the text the judge applies) or a line (a tag of the user "
                       "sim's instructions, like P1)")
        if kind == AUTO:
            if item.get("metric") not in METRICS:
                out.append(f"{at}: an auto item's metric must be one of {list(METRICS)}, not {item.get('metric')!r}")
            limit = item.get("max")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                out.append(f"{at}: an auto item needs max, a whole number of at least 1, not {limit!r}")
        elif kind is not None:
            extra = [k for k in ("metric", "max") if item.get(k) is not None]
            if extra:
                out.append(f"{at}: {' and '.join(extra)} are for kind auto only")
        na_when = item.get("na_when")
        always = isinstance(na_when, str) and na_when.strip().lower() == NA_ALWAYS
        reason = item.get("static_na_reason")
        if always and not (isinstance(reason, str) and reason.strip()):
            out.append(f"{at}: na_when always needs a static_na_reason saying why it can't apply")
        if reason is not None and not always:
            out.append(f"{at}: static_na_reason is only for na_when always")
        if kind == AUTO and isinstance(na_when, str) and na_when.strip() and not always:
            out.append(f"{at}: auto items are counted, so na_when can only be 'always' (with a static_na_reason), "
                       f"not {na_when!r}")
        if "can_error" in item and not isinstance(item["can_error"], bool):
            out.append(f"{at}: can_error must be true or false")
        unknown = sorted(set(item) - set(ITEM_FIELDS))
        if unknown:
            out.append(f"{at}: unknown fields {unknown}")
    return out


# ---- the auto items -----------------------------------------------------------------------------------------------
# A sentence ends at a run of . ! or ? followed by a space or the end, or at a line break; a piece with no letter or
# digit (an emoji, "...") isn't one. So "$15.50" stays one sentence and "Mr. Lee" counts two: simple on purpose.
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)|\n+")


def count_words(message: str) -> int:
    """Words in a message: its pieces between whitespace, an emoji on its own counting as one."""
    return len(message.split())


def count_sentences(message: str) -> int:
    return sum(1 for piece in _SENTENCE_END.split(message) if re.search(r"\w", piece))


COUNTERS: dict[str, tuple[Callable[[str], int], str]] = {MAX_WORDS: (count_words, "words"),
                                                          MAX_SENTENCES: (count_sentences, "sentences")}


def sent_answers(run: "UserSimRun") -> list["Turn"]:
    """The graded answers an auto item counts: each graded turn's message that isn't a no_reply."""
    return [t for t in run.turns if t.graded and t.outcome != "no_reply" and (t.message or "").strip()]


def auto_result(item: RubricItem, run: "UserSimRun") -> dict:
    """An auto item, computed: fail when any graded answer is over ``max``, naming each turn that is."""
    count, unit = COUNTERS[item.metric]
    counts = [(t.n, count(t.message)) for t in sent_answers(run)]
    if not counts:
        return {"verdict": "n/a", "reason": "it sent no message at a graded turn", "turns": []}
    over = [(n, c) for n, c in counts if c > item.max]
    if over:  # a shortened answer over the max is over it whole too
        listed = ", ".join(f"turn {n}: {c}" for n, c in over)
        return {"verdict": "fail", "reason": f"over the {item.max} {unit} max at {listed}",
                "turns": [n for n, _ in over]}
    if run.missing_answers:
        return {"verdict": "error", "turns": [],
                "error": f"{run.missing_answers} of the run's answers weren't kept, so they can't be counted"}
    cut = [t.n for t in sent_answers(run) if t.cut]
    if cut:
        return {"verdict": "error", "turns": cut,
                "error": f"only the start of the answer at turn{'s' if len(cut) > 1 else ''} "
                         f"{', '.join(map(str, cut))} was kept, so its full length can't be counted"}
    n, c = max(counts, key=lambda x: x[1])
    return {"verdict": "pass", "reason": f"every answer at most {item.max} {unit}; the longest, turn {n}, has {c}",
            "turns": []}


def resolve_lines(items: list[RubricItem], tagged_lines: Optional[dict[str, str]],
                  unusable: str = "") -> tuple[list[RubricItem], dict[str, dict]]:
    """Each item with its ``line``'s text as its check (``LINE_CHECK``), and, by id, the line items that can't be
    graded: their ``error``, never a guess. ``unusable`` says why no line can be trusted to be the one an item was
    written for. An ``na_when: always`` item isn't graded, so its line isn't needed."""
    out, missing = [], {}
    for item in items:
        if not item.line or item.static_na:
            out.append(item)
        elif unusable:
            missing[item.id] = {"verdict": "error", "turns": [], "line": item.line,
                                "error": f"line {item.line} can't be graded: {unusable}"}
        elif tagged_lines is None:
            missing[item.id] = {"verdict": "error", "turns": [], "line": item.line,
                                "error": f"the run recorded no tagged_lines for the user sim, so its line {item.line} "
                                         "can't be read"}
        elif item.line not in tagged_lines:
            missing[item.id] = {"verdict": "error", "turns": [], "line": item.line,
                                "error": f"the user sim's instructions have no line {item.line}"}
        else:
            text = tagged_lines[item.line]
            out.append(replace(item, check=LINE_CHECK.format(tag=item.line, text=text.replace("\n", "\n  "))))
    return out, missing


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




def run_text(run: UserSimRun) -> str:
    """The run as one text, the part of every call's prompt that its calls share (``render_run``'s two parts)."""
    return "\n".join(part for part in render_run(run) if part)


# ---- what the judge is asked --------------------------------------------------------------------------------------
def call_check(call: str) -> Optional[str]:
    """The check a call asks; None for a call that asks only task items (``TASK_ITEMS``, or a check's or the task's
    further items, ``<check>#2``)."""
    return call if call in CHECKS else None


def plan_calls(checks: list[str], items: list[RubricItem],
               per_call: int = MAX_ITEMS_PER_CALL) -> dict[str, list[RubricItem]]:
    """Which call asks what: each running check with the task items its families own, the first ``per_call`` of
    them; items no running check owns go in one more call, ``TASK_ITEMS``. Items past ``per_call`` go, ``per_call``
    at a time, in calls that ask only items (``<check>#2``, ``task_items#2``...). Only items that ask the model."""
    grouped: dict[str, list[RubricItem]] = {c: [] for c in checks}
    for item in items:
        if not item.asks_the_model:
            continue
        owner = FAMILY_OWNER.get(item.family)
        grouped.setdefault(owner if owner in grouped else TASK_ITEMS, []).append(item)
    per_call = max(1, per_call)
    calls: dict[str, list[RubricItem]] = {}
    for call, its in grouped.items():
        if call == TASK_ITEMS and not its:
            continue
        calls[call] = its[:per_call]
        for part, start in enumerate(range(per_call, len(its), per_call), 2):
            calls[f"{call}#{part}"] = its[start:start + per_call]
    return calls


def check_title(name: str) -> str:
    c = CHECKS[name]
    return f"{c.title} ({', '.join(c.families)})"


def check_question(name: str) -> str:
    """A check's question as the judge reads it: what it asks, what fails, what doesn't, when it's n/a and when a fail
    could change the assistant's grade."""
    c = CHECKS[name]
    lines = [c.question, "", "It fails when, at a graded turn:", *[f"- {f}" for f in c.fails],
             "", "Not a failure:", *[f"- {f}" for f in c.not_fails], "",
             c.na or "Answer \"pass\" or \"fail\"; \"n/a\" only if no turn is graded."]
    if c.grade:
        lines += ["", f"could_change_grade: {c.grade}"]
    return "\n".join(lines)


def family_label(family: str) -> str:
    return f"{family} ({FAMILIES[family].split(':')[0]})"


# ---- one run's criteria and rows ----------------------------------------------------------------------------------
# The fields a row's verdict is in, as a judge's row (judge_output_format's user_sim_checks) or one computed here has
# them; a stored row is these, after what a reviewer reads about it.
VERDICT_FIELDS = ("verdict", "reason", "turns", "turns_outside_graded", "could_change_grade", "assistant_asked",
                  "error", "score", "result", "justification")


@dataclass
class Plan:
    """What one run's judge is asked, and the rows no model grades.

    ``criteria``: user_sim_checks criteria in call order, each check followed by the task items riding with it; what
    the judge reads of each (``id``, ``check`` or ``item``, ``family``, ``kind``, ``question``, ``na_when``,
    ``graded_turns``, ``weight``). ``batches``: how many of them each call asks, in order. ``unjudged``: the rows no
    model grades (auto items, ``na_when: always`` items, line items whose line can't be read). ``about``: by row id,
    what a reviewer reads on the row and the judge never sees (an item's ``name``, ``short``, ``asked_with`` ...)."""
    criteria: list[dict] = field(default_factory=list)
    batches: list[int] = field(default_factory=list)
    unjudged: list[dict] = field(default_factory=list)
    about: dict[str, dict] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # the row ids: the checks in order, then the task's items


def item_weight(item: RubricItem, error_checks: list[str]) -> float:
    """1 when a fail of the item can make the run a user sim error, else 0: its ``can_error``, unset as the check
    that owns its family (an error check, or not)."""
    can = item.can_error if item.can_error is not None else FAMILY_OWNER.get(item.family) in error_checks
    return 1.0 if can else 0.0


def computed_row(got: dict, weight: float) -> dict:
    """A row no model graded (``got``: verdict, reason or error, turns), scored as a judge's: it lowers the score only
    on a fail at a weight other than 0. A computed fail could change the grade; ``assistant_asked`` is None, since no
    judge looked."""
    fail = got["verdict"] == "fail"
    lowers = fail and weight != 0
    return {**got, "could_change_grade": fail, "assistant_asked": None,
            "justification": got.get("reason") or got.get("error") or "",
            "score": 0.0 if lowers else 1.0, "result": not lowers}


def plan_run(run: UserSimRun, *, checks: list[str], items: list[RubricItem], error_checks: list[str],
             lines_unusable: str = "") -> Plan:
    """The criteria the judge is asked about ``run`` and the rows settled here: each check in ``checks`` (weight 1
    when it is in ``error_checks``, else 0) with the task items its families own, at most ``MAX_ITEMS_PER_CALL`` to a
    call; an auto item counted, an ``na_when: always`` item n/a, a ``line`` item graded by its line's text in
    ``run.tagged_lines``, or an error when the run has no such tag or ``lines_unusable`` says why none can be
    trusted."""
    graded = sorted(t.n for t in run.turns if t.graded)
    resolved, no_line = resolve_lines(list(items), run.tagged_lines, lines_unusable)
    calls = plan_calls(list(checks), resolved)
    plan = Plan(order=list(checks) + [i.id for i in items])
    for name in checks:
        plan.about[name] = {"title": CHECKS[name].title, "families": list(CHECKS[name].families),
                            "weight": 1.0 if name in error_checks else 0.0}
    lines = run.tagged_lines or {}
    for item in items:
        about = {k: v for k, v in item.to_dict().items() if k != "id"}
        if item.line and item.asks_the_model and item.id not in no_line:
            about["line_text"] = lines[item.line]
        plan.about[item.id] = {**about, "weight": item_weight(item, error_checks)}
    for call, its in calls.items():
        size = 0
        if call_check(call):
            plan.criteria.append({"id": call, "check": check_title(call), "question": check_question(call),
                                  "graded_turns": graded, "weight": plan.about[call]["weight"]})
            size += 1
        for item in its:
            criterion = {"id": item.id, "item": item.id, "family": family_label(item.family), "kind": item.kind,
                         "question": item.check, "graded_turns": graded, "weight": plan.about[item.id]["weight"]}
            if item.na_when:
                criterion["na_when"] = item.na_when
            plan.criteria.append(criterion)
            plan.about[item.id]["asked_with"] = call
            size += 1
        if size:
            plan.batches.append(size)
    for item in items:
        if item.id in no_line:
            got, asked_with = no_line[item.id], NO_LINE
        elif item.kind == AUTO and not item.static_na:
            got, asked_with = auto_result(item, run), COMPUTED
        elif item.static_na:
            got, asked_with = {"verdict": "n/a", "reason": item.static_na_reason, "turns": []}, STATIC_NA
        else:
            continue
        got = {k: v for k, v in got.items() if k != "line"}
        plan.about[item.id]["asked_with"] = asked_with
        plan.unjudged.append({"id": item.id, **computed_row(got, plan.about[item.id]["weight"])})
    return plan


def undecided_row(criterion: dict, error: str) -> dict:
    """A criterion whose judge never gave a usable answer: ``error``, which never lowers the score."""
    return {"id": criterion["id"], "verdict": "error", "error": error, "turns": [], "could_change_grade": False,
            "assistant_asked": None, "justification": error, "score": 1.0, "result": True}


def finish_rows(rows: list[dict], plan: Plan) -> list[dict]:
    """The stored rows, in ``plan.order``: each row's id, what a reviewer reads about it, then its verdict."""
    by_id = {r.get("id"): r for r in rows}
    return [{"id": rid, **plan.about[rid], **{k: by_id[rid][k] for k in VERDICT_FIELDS if k in by_id[rid]}}
            for rid in plan.order if rid in by_id]


# ---- the run's verdict --------------------------------------------------------------------------------------------
def user_sim_error(rows: list[dict]) -> dict:
    """Whether the run is a user sim error, by one rule for checks and task items: ``value`` yes when a row at a weight
    other than 0 failed in a way that could change the grade (``because`` names them); unknown when none did but such
    a row is undecided (``error``: its judge never gave a usable answer, or it couldn't be counted or read); else no.
    A weight-0 row (style, by default) never makes it either."""
    counts = [r for r in rows if float(r.get("weight", 1.0)) != 0]
    because = [r["id"] for r in counts if r.get("verdict") == "fail" and r.get("could_change_grade")]
    unknown = [r["id"] for r in counts if r.get("verdict") == "error"]
    return {"value": "yes" if because else "unknown" if unknown else "no", "because": because, "unknown": unknown}


NOT_JUDGED = {"value": "no", "because": [], "unknown": []}


def usage_total(usages: list[Any]) -> dict[str, int]:
    """The token counts of several calls, summed; a cache read reported as ``prompt_tokens_details.cached_tokens``
    counts as one."""
    keys = ("prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    total = dict.fromkeys(keys, 0)
    for u in usages:
        if not isinstance(u, dict):
            continue
        for k in keys:
            v = u.get(k)
            if isinstance(v, (int, float)):
                total[k] += int(v)
        details = u.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(cached, int) and not u.get("cache_read_input_tokens"):
            total["cache_read_input_tokens"] += cached
    return total
