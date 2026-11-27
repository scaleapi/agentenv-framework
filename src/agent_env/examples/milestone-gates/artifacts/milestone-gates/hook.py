"""milestone-gates: a prompt_agent post-turn hook that checks each milestone's gates while the conversation runs.

After every agent turn it runs the open milestone's gates against the env (tool state, the artifact's text, and one
LLM judge call for a content gate), then tells the user sim what it found in a note only the sim sees
(``turn.ask_user(note=...)``):
- all gates pass: the milestone is done; the note opens the next one, or asks the sim to wrap up after the last;
- a gate fails: the note carries that gate's nudge, written by the task author and vague on purpose;
- a gate still fails after ``max_nudges`` failing turns: a blocking milestone ends the conversation
  (``gate_never_passed:<id>``); any other is let go and the next one opens.
Every turn that ends with a failing gate counts toward ``max_nudges``, whether or not the sim pressed on it.

The hook follows the sim: when the sim's answer opens a later milestone (its message carries that milestone's
``line``, or the sim reports it in a ``milestone`` field), the hook moves there before the agent acts, taking that
milestone's no_new baselines then. And when the sim sends the same message ``idle_end`` turns in a row, it has nothing
left to ask: the agent still gets that message (a hook that asked returns its answer), and the conversation
ends at the start of the next turn (``sim_idle``).

A judge gate that can't get an answer (after one retry) is unknown (``passed`` None): it neither fails nor nudges.

The sim words every message; the hook only decides where the conversation is. See the README beside this bundle.

Params:
  role            the env role the gates read as (needs read access to everything they check)
  env             the deployed env's env_id to read (default: the first deployed env)
  judge_model     model for an artifact's judge gates (needed when one is used)
  excerpt_chars   how much of a milestone's ``show`` artifact goes in the note (default 600)
  final_note      the note after the last milestone passes, when every milestone passed
  final_note_gave_up  the note after the last milestone when one was let go (never says everything is done)
  idle_end        end after the sim sends the same message this many turns in a row (default 2; 0 = never)
  trajectory_judge_model   model for trajectory_judge gates (needed when one is used; a small model misreads long
                           logs)
  trajectory_max_chars     the most rendered trajectory one judge call gets (default 600000); the oldest turns
                           before the open milestone are cut first
  milestones      [{id, line?, type?, skip_if_done?, blocking?, max_nudges?, show?, group?, carry_gates?,
                    artifacts: {name: artifact}, gates: [gate]}]
                  ``line`` is the text the sim sends to open the milestone, used to tell when it has moved on.
                  When a milestone is done, each next one the agent has already done is skipped (its gates all pass
                  then, at least one of them on something the agent made): the note tells the sim to skip its line.
                  ``skip_if_done: false`` turns that off for a milestone; type ``correct`` is never skipped.
                  ``group`` makes an entry a revision of an earlier milestone (its id): a change asked of work already
                  done, with gates of its own. A milestone's revisions follow it one after another, each opened,
                  nudged, given up on and ended like any milestone, and by default never skipped. ``carry_gates:
                  true`` checks the previous revision's gates (and keeps its artifacts) too, but for any gate the
                  revision writes with the same id, which replaces it. The summary nests revisions under their
                  milestone.

After the wrap-up is answered, the last milestone is checked once more, and so is the latest revision of every earlier
milestone, on its state gates only (not no_new or trajectory_judge, which are about the turns it was open): one found
broken is ``undone``.

A gate is {id, nudge, check}, with check one of:
  {"trajectory_judge": "statement"}            one LLM call: is the statement true of the agent's work, read from its
                                               trajectory (every turn so far, rendered compactly: writes in full,
                                               reads shortened; the open milestone's turns marked THIS STEP). Needs
                                               the framework to hand hooks ``turn.trajectory_urls``; without them the
                                               gate is unknown. A task's rubric criteria can be used as they are.
  {"call": call, "expect": expect}             a read tool's result meets ``expect``
  {"artifact": name, "expect": "exists"}       the artifact was found
  {"artifact": name, "format": {...}}          code over the artifact's text: max_paragraphs, max_sentences,
                                               contains_all [text, ...], contains_none [text, ...]
  {"artifact": name, "judge": "statement"}     one yes/no LLM call: is the statement true of the artifact's text
  {"no_new": [call + {"baseline_count"?}]}     no item added since the milestone opened (the first milestone,
                                               which opens before the hook first runs, needs ``baseline_count``)
  {"no_removed": [call + {"baseline_count"?}]} no item removed since the milestone opened: the count is not below
                                               its count then (``baseline_count`` as for no_new)
  {"trajectory": {contains_none?, no_call?, scope?}}
                                               code over the agent's own work in its trajectories, no model call:
                                               ``contains_none`` [text, ...] appears in none of its tool calls'
                                               arguments or what it said (its reads' results aren't its work), and
                                               ``no_call`` [{tool?, args_contains?}, ...] matches none of its tool
                                               calls (``tool`` the name without its app prefix). ``scope`` is
                                               "step" (the open milestone's turns, the default), "run" (every turn
                                               so far) or {"before": milestone id} (the turns before that milestone
                                               opened or was skipped; every turn so far if it hasn't been)
A call is {tool, args?, path?, where?, number?}: ``path`` is a dotted path into the tool's JSON result, ``where``
keeps the list items whose fields contain the given text, and ``number`` reads a count the tool reports (that field of
the first item) instead of counting the list. ``expect`` is "exists", or {"count": {"eq"|"min"|"max": n}},
{"contains": text} or {"equals": value}.
An artifact is {find: call + {pick?: "first"|"last"}, read?: {tool, args, extract?}, field?}: ``read`` args may use
"{field}" from the item ``find`` picked, and ``extract: "gmail"`` turns a Gmail API message into its headers and
plain-text body; without ``read``, ``field`` (a dotted path) takes that field of the picked item as its text, else the
artifact's text is the picked item itself, as JSON.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# When none of the open milestone's gates could be checked: the sim stays on it, and it counts toward max_nudges, so a
# milestone whose evidence never arrives is let go (or ends the run, if blocking) instead of holding forever.
HOLD_NOTE = ("Nothing could be checked this time. Don't move on: reply to what the assistant said, and stay on the "
             "milestone you're on.")
MILESTONE_KEYS = {"id", "line", "type", "skip_if_done", "blocking", "max_nudges", "show", "artifacts", "gates", "group",
                  "carry_gates"}
DONE = ("passed", "skipped")  # the statuses that count a milestone done
DEFAULT_FINAL = "Everything you asked for is done. Wrap up the way you planned (send your last line), then stop."
# The wrap-up after a milestone was let go: nothing claims it was all done, and the user takes on nothing new.
DEFAULT_FINAL_GAVE_UP = ("Wrap up now: send your last line, then stop. Don't say everything is done, and don't ask "
                         "for or agree to anything new.")
JUDGE_SYSTEM = ("You check one statement against a document. Use only the document. Answer passed=true only if the "
                "document clearly shows the statement is true. Keep why to one sentence of at most 30 words.")
TRAJECTORY_JUDGE_SYSTEM = (
    "You check one statement about what an AI assistant did, against a log of its work: every tool call it made with "
    "the result, its writes in full and its reads shortened, and what it said. Judge only from the log. Turns marked "
    "THIS STEP are what the statement means by 'this step' or 'during this step'; earlier turns are context, such as "
    "what the assistant made before. Judge what the statement is checking, the way a careful reviewer would read it: "
    "mentioning something only to exclude it or say it doesn't apply doesn't fail a statement about what is "
    "included. Answer passed=true only if the log clearly shows the statement is true. Keep why to one sentence of at "
    "most 30 words.")
JUDGE_FORMAT = {"type": "json_schema", "json_schema": {"name": "gate", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["passed", "why"],
    "properties": {"passed": {"type": "boolean"}, "why": {"type": "string"}}}}}


def check_params(params):
    """Called by the framework when the hook loads, before the conversation starts: refuse params the hook can't run
    with, so the step fails before the first message instead of after the agent's first turn."""
    _spec(params)


async def post_turn(turn, params):
    spec = _spec(params)
    answer = await _post_turn(turn, spec)
    if hasattr(turn, "summarize"):  # a framework older than Turn.summarize keeps only the records
        turn.summarize(_summary(spec, turn.records()))
    return answer


async def _post_turn(turn, spec):
    if not spec["enabled"]:  # configured but switched off: the conversation runs as if no hook were listed
        if not turn.records():
            turn.record({"decision": "disabled"})
        return None
    state = _state(turn.records())
    milestones = spec["milestones"]
    if state["finished"]:  # the wrap-up was sent and the agent has answered it
        # Check the work once more: the agent may have changed it while answering the wrap-up, or broken an earlier
        # milestone's work since, and the summary should match what it left, as the task's final rubric will.
        env = _env(turn, spec)
        m = milestones[state["index"]]
        gates = await _run_gates(turn, spec, env, m, state)
        record = {"decision": "done", "milestone": m["id"], "gates": gates}
        earlier = []
        for o in _latest_revisions(spec):
            if o["group"] == m["group"]:
                continue
            kept = [g for g in o["gates"] if _state_check(g["check"])]
            if kept:
                earlier.append({"milestone": o["id"], "gates": await _run_gates(
                    turn, spec, env, {**o, "gates": kept}, {**state, "baselines": {}})})
        if earlier:
            record["final_checks"] = earlier
        turn.record(record)
        return {"end": "milestones_done"}
    if state["idle"]:  # the sim repeated itself last turn; the agent has answered that, so end before asking again
        turn.record({"decision": "end", "end": "sim_idle"})
        return {"end": "sim_idle"}
    m = milestones[state["index"]]
    env = _env(turn, spec)
    gates = await _run_gates(turn, spec, env, m, state)
    failed = [g for g in gates if g["passed"] is False]
    entry = {"milestone": m["id"], "gates": gates}
    checks = {g["id"]: g["check"] for g in m["gates"]}
    # Unknown: no gate checked at all, or a tool/state gate (not a judge) that couldn't be checked, such as a no_new
    # gate whose baseline couldn't be read. Either way the milestone isn't shown done. A judge that can't answer
    # doesn't hold it up (see _judge).
    unchecked = bool(gates) and (all(g["passed"] is None for g in gates) or any(
        g["passed"] is None and not _costly(checks[g["id"]]) for g in gates if g["id"] in checks))
    if turn.last:  # nothing more can be asked: record where it stood, passed if its gates passed on this turn
        turn.record({**entry, "decision": "last_turn", "nudges": state["nudges"],
                     "passed": not failed and not unchecked})
        return None
    if not failed and not unchecked:
        return await _advance(turn, spec, env, state, entry, "passed")
    if state["nudges"] >= m["max_nudges"]:
        if m["blocking"]:
            end = f"gate_never_passed:{m['id']}"
            turn.record({**entry, "decision": "end", "end": end, "nudges": state["nudges"]})
            return {"end": end}
        return await _advance(turn, spec, env, state, entry, "gave_up")
    if failed:
        note = failed[0]["nudge"] + await _excerpt(turn, spec, env, m, state)
    else:  # no gate could be checked (say, the turn's trajectory wasn't fetched): hold, without a nudge on nothing
        note = HOLD_NOTE
    answer = await turn.ask_user(note=note)
    record = {**entry, "decision": "nudge" if failed else "hold", "note": note, "nudges": state["nudges"] + 1,
              "sim_nudged": _sim_field(turn, "nudged")}
    return await _after_answer(turn, spec, env, state, record, answer, state["index"])


async def _after_answer(turn, spec, env, state, record, answer, expected):
    """Record the sim's answer, follow it to a later milestone if it opened one, and end if it has gone idle."""
    message = _answer_text(answer)
    record["sim_message"] = message[:300]
    moved = _sim_milestone(turn, spec, message)
    if moved is not None and moved > expected:
        record.update(followed_sim=spec["milestones"][moved]["id"], next_index=moved, wrap_up=False,
                      baselines=await _baselines(env, spec, spec["milestones"][moved]))
    idle = spec["idle_end"]
    if idle >= 2 and message.strip():
        before = [r.get("sim_message") for r in turn.records() if "sim_message" in r][-(idle - 1):]
        if len(before) == idle - 1 and all(_same(b, message) for b in before):
            # A hook that asked returns the answer, so the agent gets this message and the next turn ends.
            record["idle"] = True
    turn.record(record)
    return answer


# ---- state ---------------------------------------------------------------------------------------------------------

def _state(records: list[dict]) -> dict:
    """Where the conversation is, from this hook's own records: the open milestone, nudges spent on it, the no_new
    baselines taken when it opened, the turn its line first reached the agent (``since``), and whether the wrap-up
    has gone out."""
    index, nudges, baselines, finished, idle, since = 0, 0, {}, False, False, 1
    for r in records:
        idle = bool(r.get("idle"))
        if r.get("next_index") is not None:  # passed, let go, or followed the sim to a later milestone
            index, nudges, baselines = r["next_index"], 0, r.get("baselines") or {}
            finished = r.get("wrap_up", False)
            if not finished:  # the wrap-up keeps the last milestone's turns as the step a final check reads
                since = int(r.get("turn", 0)) + 1  # its line goes to the agent in the next turn
        elif r.get("decision") in ("nudge", "hold"):
            nudges = r["nudges"]
    return {"index": index, "nudges": nudges, "baselines": baselines, "finished": finished, "idle": idle,
            "since": since}


def _gate_type(check: dict) -> str:
    """A gate's type when the spec doesn't give one, from how it's checked: format (code over text), action (tool
    state), content (a judge). A spec's ``type`` wins: a trajectory judge can check an action."""
    if "judge" in check or "trajectory_judge" in check:
        return "content"
    return "format" if "format" in check else "action"


def _summary(spec: dict, records: list[dict]) -> dict:
    """Where the conversation got to, from this hook's records, for whoever grades it (Turn.summarize): each
    milestone's status (not_reached, open, passed, skipped (done before it opened), gave_up (let go after max_nudges),
    undone (done, then found broken by the check after the wrap-up),
    passed_over (the user sim moved past it), derailed (a blocking one that never passed, which ended the run)), the
    turn it opened and closed, nudges spent, and each gate's last verdict and type; then the furthest milestone
    reached, how many were done, and the end reason once the hook ended the conversation. With revisions, each
    milestone's row holds its revisions' rows and rolls them up (``_rollup``)."""
    if not spec["enabled"]:
        return {"enabled": False}
    milestones = spec["milestones"]
    # By milestone entry and gate: a revision may override a gate with another check, and each row shows its own.
    types = {(m["id"], g["id"]): g.get("type") or _gate_type(g["check"]) for m in milestones for g in m["gates"]}
    rows = [{"id": m["id"], "status": "not_reached", "opened_turn": None, "closed_turn": None, "nudges": 0,
             "gates": {}} for m in milestones]
    rows[0].update(status="open", opened_turn=1)
    by_id = {r["id"]: r for r in rows}
    end = None

    def verdicts(row, gates):
        row["gates"].update({g["id"]: {"passed": g["passed"], "type": types.get((row["id"], g["id"]))}
                             for g in gates})

    for r in records:
        turn, row = r.get("turn"), by_id.get(r.get("milestone"))
        if row is not None:
            verdicts(row, r.get("gates") or [])
            decision = r.get("decision")
            if decision in ("nudge", "hold"):
                row["nudges"] = r.get("nudges", row["nudges"])
            elif decision in ("passed", "gave_up"):
                row.update(status=decision, closed_turn=turn)
            elif decision == "last_turn" and r.get("passed"):  # done on the final turn, with nothing left to ask
                row.update(status="passed", closed_turn=turn)
            elif decision == "end" and r.get("end", "").startswith("gate_never_passed"):
                row.update(status="derailed", closed_turn=turn, nudges=r.get("nudges", row["nudges"]))
        for s in r.get("skipped") or []:
            verdicts(by_id[s["milestone"]], s["gates"])
            by_id[s["milestone"]].update(status="skipped", opened_turn=turn, closed_turn=turn)
        nxt = r.get("next_index")
        if nxt is not None and not r.get("wrap_up"):
            if r.get("followed_sim"):  # the sim opened a later milestone itself: those before it are passed over
                for skipped_over in rows[:nxt]:
                    if skipped_over["status"] in ("open", "not_reached"):
                        skipped_over.update(status="passed_over", closed_turn=turn)
            if rows[nxt]["status"] == "not_reached":
                rows[nxt].update(status="open", opened_turn=turn + 1)
        if r.get("decision") == "done":
            end = "milestones_done"
            final = r.get("gates") or []
            if row is not None and final:
                if all(g["passed"] for g in final) and row["status"] not in ("passed", "skipped"):
                    row.update(status="passed", closed_turn=turn)  # done in the end, after the wrap-up
                elif any(g["passed"] is False for g in final) and row["status"] in ("passed", "skipped"):
                    row.update(status="undone", closed_turn=turn)  # done before, broken while answering the wrap-up
            # An earlier milestone's latest revision, checked on its state gates only: those passing don't prove a
            # revision let go is done, so the check can only find one broken.
            for f in r.get("final_checks") or []:
                earlier = by_id.get(f["milestone"])
                if earlier is None:
                    continue
                verdicts(earlier, f["gates"])
                if any(g["passed"] is False for g in f["gates"]) and earlier["status"] in ("passed", "skipped"):
                    earlier.update(status="undone", closed_turn=turn)
        elif r.get("decision") == "end":
            end = r.get("end")
    for m, row in zip(milestones, rows):
        for gid in [g["id"] for g in m["gates"] if g.get("carried")]:
            if gid in row["gates"]:
                row["gates"][gid]["carried"] = True
    groups: dict[str, list[dict]] = {}
    for m, row in zip(milestones, rows):
        groups.setdefault(m["group"], []).append(row)
    if all(len(revs) == 1 for revs in groups.values()):  # no revisions: one row per milestone, as always
        reached = [row["id"] for row in rows if row["status"] != "not_reached"]
        return {"enabled": True, "reached": reached[-1] if reached else None,
                "done": sum(row["status"] in DONE for row in rows), "total": len(rows),
                "end": end, "milestones": rows}
    out = [revs[0] if len(revs) == 1 else _rollup(head, revs) for head, revs in groups.items()]
    reached = [row for row in out if row["status"] != "not_reached"]
    return {"enabled": True, "reached": reached[-1]["id"] if reached else None,
            "reached_revision": (reached[-1].get("reached_revision", reached[-1]["id"]) if reached else None),
            "done": sum(row["status"] in DONE for row in out), "total": len(out),
            "revisions_done": sum(row["status"] in DONE for row in rows), "revisions_total": len(rows),
            "end": end, "milestones": out}


def _rollup(head: str, revs: list[dict]) -> dict:
    """A milestone with revisions: its revisions' rows (numbered from 1, the milestone as first written), under the
    status of the furthest one reached, as that's the work the user last asked for."""
    for k, rev in enumerate(revs, 1):
        rev["revision"] = k
    reached = [rev for rev in revs if rev["status"] != "not_reached"]
    top = reached[-1] if reached else None
    return {"id": head, "status": top["status"] if top else "not_reached",
            "opened_turn": reached[0]["opened_turn"] if reached else None,
            "closed_turn": top["closed_turn"] if top else None, "nudges": sum(rev["nudges"] for rev in revs),
            "reached_revision": top["id"] if top else None, "revisions_done": sum(rev["status"] in DONE for rev in revs),
            "revisions": revs}


async def _skips(turn, spec, env, state, nxt) -> tuple[int, list[dict]]:
    """From milestone ``nxt`` on, the ones the agent has already done: every gate passes now, at least one of them
    checks something the agent made (a no_new gate alone, its baselines taken now, proves nothing), and the milestone
    may be skipped (``skip_if_done``, never type ``correct``, whose line is always sent). Returns the first milestone
    not skipped and what each skipped one's gates found."""
    milestones, skipped = spec["milestones"], []
    while nxt < len(milestones):
        m = milestones[nxt]
        if not m["skip_if_done"] or m["type"] == "correct":
            break
        gates = await _run_gates(turn, spec, env, m, {**state, "baselines": await _baselines(env, spec, m)})
        made = [g for g in m["gates"] if not set(g["check"]) & {"no_new", "no_removed", "trajectory"}]
        if not made or not all(r["passed"] is True for r in gates):
            break
        skipped.append({"milestone": m["id"], "gates": gates})
        nxt += 1
    return nxt, skipped


async def _advance(turn, spec, env, state, entry, how):
    """The open milestone is passed or let go: open the next one the agent hasn't already done (taking its no_new
    baselines now, before the agent works on it), or send the wrap-up after the last."""
    milestones = spec["milestones"]
    head = (f"{milestones[state['index']]['id']} isn't done, but let it go." if how == "gave_up"
            else f"{milestones[state['index']]['id']} is done.")
    nxt, skipped = await _skips(turn, spec, env, state, state["index"] + 1)
    if skipped:
        ids = ", ".join(s["milestone"] for s in skipped)
        head += (f" {ids} {'is' if len(skipped) == 1 else 'are'} already done too, so skip "
                 f"{'its line' if len(skipped) == 1 else 'their lines'}.")
    if nxt < len(milestones):
        baselines = await _baselines(env, spec, milestones[nxt])
        line = milestones[nxt].get("line")
        note = (f"{head} Open {milestones[nxt]['id']} now: send this line: {line}" if line
                else f"{head} Open {milestones[nxt]['id']} now: send its line.")
        record = {**entry, "decision": how, "next_index": nxt, "baselines": baselines, "note": note}
    else:
        let_go = how == "gave_up" or any(r.get("decision") == "gave_up" for r in turn.records())
        final = spec["final_note_gave_up"] if let_go else spec["final_note"]
        note = (f"{head} " if how == "gave_up" or skipped else "") + final
        # The final check after the wrap-up reads the last milestone: the open one, or the last one skipped as done
        # (its baselines taken now, as the skip took them).
        last = len(milestones) - 1 if skipped else state["index"]
        baselines = state["baselines"] if last == state["index"] else await _baselines(env, spec, milestones[last])
        record = {**entry, "decision": how, "next_index": last, "wrap_up": True, "note": note,
                  "baselines": baselines}
    if skipped:
        record["skipped"] = skipped
    answer = await turn.ask_user(note=note)
    return await _after_answer(turn, spec, env, state, record, answer, record["next_index"])


def _sim_field(turn, name) -> Any:
    """The peer sim's own field for this turn (prompt_agent records extra JSON fields), or None."""
    outputs = (turn.metadata.get("usersim_turn_outputs") or {})
    step_id = getattr(turn, "step_id", None)  # turn numbers restart in each prompt_agent step
    for entries in ([outputs.get(step_id) or []] if step_id is not None else outputs.values()):
        for e in entries:
            if e.get("turn") == turn.number and name in (e.get("fields") or {}):
                value = e["fields"][name]
                return bool(value) if name == "nudged" else value
    return None


def _answer_text(answer: Any) -> str:
    if not isinstance(answer, dict):
        return ""
    return "\n".join(p.get("text", "") for p in answer.get("message") or [] if isinstance(p, dict))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _same(a: Optional[str], b: str) -> bool:
    return a is not None and _norm(a) == _norm(b[:300])


def _sim_milestone(turn, spec, message) -> Optional[int]:
    """The latest milestone the sim's message opens: one whose ``line`` it carries, else the one its ``milestone``
    field names. None when neither says."""
    said = _norm(message)
    hits = [i for i, m in enumerate(spec["milestones"]) if m.get("line") and _norm(m["line"])[:120] in said]
    if hits:
        return max(hits)
    field = _sim_field(turn, "milestone")
    ids = [m["id"] for m in spec["milestones"]]
    return ids.index(field) if isinstance(field, str) and field in ids else None


# ---- gates ---------------------------------------------------------------------------------------------------------

async def _run_gates(turn, spec, env, m, state) -> list[dict]:
    """Every tool, format and no_new gate; the judge gates only when all of those pass (they cost a model call)."""
    artifacts: dict[str, Optional[str]] = {}
    out = []
    for g in m["gates"]:
        if _costly(g["check"]):
            continue
        out.append(await _gate(turn, spec, env, m, state, g, artifacts))
    cheap_ok = all(r["passed"] for r in out)
    for g in m["gates"]:
        if not _costly(g["check"]):
            continue
        if cheap_ok:
            out.append(await _gate(turn, spec, env, m, state, g, artifacts))
        else:
            out.append({"id": g["id"], "passed": None, "why": "not run: an earlier gate failed", "nudge": g["nudge"],
                        **({"type": g["type"]} if g.get("type") else {})})
    order = {g["id"]: i for i, g in enumerate(m["gates"])}
    return sorted(out, key=lambda r: order[r["id"]])


def _state_check(check: dict) -> bool:
    """A gate on the state the agent left, which still means something after its milestone closed: not a no_new or
    no_removed (nothing added or removed while the milestone was open) or a trajectory check (what the agent did)."""
    return not set(check) & {"no_new", "no_removed", "trajectory_judge", "trajectory"}


def _latest_revisions(spec: dict) -> list[dict]:
    """Each milestone's last revision (a milestone without revisions is its own), in order."""
    last: dict[str, dict] = {}
    for m in spec["milestones"]:
        last[m["group"]] = m
    return list(last.values())


def _costly(check: dict) -> bool:
    """A gate that costs a model call: run only when every other gate passes."""
    return "judge" in check or "trajectory_judge" in check


async def _gate(turn, spec, env, m, state, g, artifacts) -> dict:
    check = g["check"]
    try:
        if "trajectory_judge" in check:
            passed, why = await _trajectory_gate(turn, spec, state, check["trajectory_judge"])
        elif "call" in check:
            passed, why = _expect(await _call(env, spec["role"], check["call"]), check["expect"])
        elif "no_new" in check:
            passed, why = await _no_new(env, spec["role"], m, state, check["no_new"], g["id"])
        elif "no_removed" in check:
            passed, why = await _no_new(env, spec["role"], m, state, check["no_removed"], g["id"], removed=True)
        elif "trajectory" in check:
            passed, why = await _trajectory_check(turn, spec, state, check["trajectory"])
        else:
            text = await _artifact(env, spec["role"], m, check["artifact"], artifacts)
            if text is None:
                passed, why = False, f"artifact {check['artifact']!r} not found"
            elif check.get("expect") == "exists":
                passed, why = True, "found"
            elif "format" in check:
                passed, why = _format(text, check["format"])
            else:
                passed, why = await _judge(spec, check["judge"], text)
    except Exception as e:  # noqa: BLE001 - a gate that can't be checked counts as failed, and says why
        passed, why = False, f"error: {type(e).__name__}: {str(e)[:200]}"
    return {"id": g["id"], "passed": passed, "why": why, "nudge": g["nudge"],
            **({"type": g["type"]} if g.get("type") else {})}  # format, action or content, when the spec says


async def _call(env, role, call) -> Any:
    resp = await env.call_tool(call["tool"], call.get("args") or {}, role)
    value = _result(resp)
    for key in [k for k in (call.get("path") or "").split(".") if k]:
        value = value[int(key)] if isinstance(value, list) else value[key]
    where = call.get("where")
    if where:
        value = [item for item in value if all(str(v).lower() in str(_dig(item, k)).lower() for k, v in where.items())]
    if call.get("number"):  # a count the tool reports (the first item's field), not the length of a list
        item = value[0] if isinstance(value, list) and value else value
        value = int(_dig(item, call["number"]) or 0) if isinstance(item, dict) else 0
    return value


def _size(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return len(value) if isinstance(value, list) else (1 if value else 0)


def _result(resp: dict) -> Any:
    """A /step call_tool response's content as JSON when it is JSON, else its text."""
    if isinstance(resp, dict) and resp.get("isError"):
        raise RuntimeError(f"tool error: {_text(resp)[:200]}")
    text = _text(resp)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _text(resp: Any) -> str:
    if isinstance(resp, dict) and isinstance(resp.get("content"), list):
        return "\n".join(c.get("text", "") for c in resp["content"] if isinstance(c, dict))
    return resp if isinstance(resp, str) else json.dumps(resp)


def _dig(item: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        item = item.get(key) if isinstance(item, dict) else None
    return item


def _expect(value: Any, expect: Any) -> tuple[bool, str]:
    if expect == "exists":
        ok = bool(value)
        return ok, "found" if ok else "nothing found"
    if "count" in expect:
        n = _size(value)
        c = expect["count"]
        ok = (c.get("eq") is None or n == c["eq"]) and (c.get("min") is None or n >= c["min"]) and \
            (c.get("max") is None or n <= c["max"])
        return ok, f"count {n}"
    if "contains" in expect:
        hay = value if isinstance(value, str) else json.dumps(value)
        ok = expect["contains"].lower() in hay.lower()
        return ok, ("contains " if ok else "missing ") + repr(expect["contains"])
    if "equals" in expect:
        ok = value == expect["equals"]
        return ok, f"value {value!r}"
    raise ValueError(f"unknown expect {expect!r}")


def _baseline_key(m: dict, gate_id: str, i: int) -> str:
    return f"{m['id']}:{gate_id}:{i}"


async def _no_new(env, role, m, state, calls, gate_id: str, removed: bool = False) -> tuple[Optional[bool], str]:
    """Nothing added (or, with ``removed``, nothing removed) since the milestone opened. Unknown (None) when a
    baseline couldn't be read then: comparing against nothing would be a guess."""
    changed, missing = [], []
    for i, call in enumerate(calls):
        base = state["baselines"].get(_baseline_key(m, gate_id, i), call.get("baseline_count"))
        if base is None:
            missing.append(call["tool"])
            continue
        n = _size(await _call(env, role, call))
        if removed and n < base:
            changed.append(f"{call['tool']} -{base - n}")
        elif not removed and n > base:
            changed.append(f"{call['tool']} +{n - base}")
    if changed:  # a failure another call found stands, whatever couldn't be compared
        return False, ("removed: " if removed else "new: ") + ", ".join(changed)
    if missing:
        return None, f"unknown: no baseline for {', '.join(missing)} (it couldn't be read when {m['id']} opened)"
    return True, "nothing removed" if removed else "nothing new"


async def _baselines(env, spec, m) -> dict:
    """Each no_new and no_removed call's count when ``m`` opens, keyed by gate and call. A read that fails leaves its
    baseline missing (its gate is then unknown), rather than failing the step over a temporary tool error."""
    out = {}
    for g in m["gates"]:
        for i, call in enumerate(g["check"].get("no_new") or g["check"].get("no_removed") or []):
            try:
                out[_baseline_key(m, g["id"], i)] = _size(await _call(env, spec["role"], call))
            except Exception:  # noqa: BLE001 - a missing baseline makes only this gate unknown
                continue
    return out


async def _artifact(env, role, m, name, cache) -> Optional[str]:
    if name in cache:
        return cache[name]
    a = m["artifacts"][name]
    found = await _call(env, role, a["find"])
    items = found if isinstance(found, list) else ([found] if found else [])
    if not items:
        cache[name] = None
        return None
    item = items[-1] if a["find"].get("pick", "last") == "last" else items[0]
    if a.get("read"):
        args = {k: (v.format(**item) if isinstance(v, str) and isinstance(item, dict) else v)
                for k, v in (a["read"].get("args") or {}).items()}
        resp = await env.call_tool(a["read"]["tool"], args, role)
        if isinstance(resp, dict) and resp.get("isError"):  # a failed read is not the artifact's text
            raise RuntimeError(f"tool error reading {name!r}: {_text(resp)[:200]}")
        text = _text(resp)
        if a["read"].get("extract") == "gmail":
            text = _gmail_text(text)
    elif a.get("field") and isinstance(item, dict):  # one field of the item, as its text (a draft's body)
        value = _dig(item, a["field"])
        text = None if value is None else (value if isinstance(value, str) else json.dumps(value, indent=1))
    else:
        text = item if isinstance(item, str) else json.dumps(item, indent=1)
    cache[name] = text
    return text


def _gmail_text(raw: str) -> str:
    """A Gmail API message as a person reads it: subject, recipients, thread, then the plain-text body (base64url in
    ``payload``, in the first text/plain part, else text/html with the tags dropped)."""
    import base64
    msg = json.loads(raw)
    payload = msg.get("payload") or {}
    headers = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers") or []}

    def parts(p):
        yield p
        for c in p.get("parts") or []:
            yield from parts(c)

    def decode(p):
        data = (p.get("body") or {}).get("data") or ""
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace") if data else ""

    every = list(parts(payload))
    body = next((decode(p) for p in every if p.get("mimeType") == "text/plain" and decode(p)), "")
    if not body:
        html = next((decode(p) for p in every if p.get("mimeType") == "text/html" and decode(p)), "")
        body = re.sub(r"<[^>]+>", " ", html)
    head = [f"Subject: {headers.get('subject', '')}", f"To: {headers.get('to', '')}",
            f"Cc: {headers.get('cc', '')}", f"Thread: {msg.get('threadId', '')}"]
    return "\n".join(head) + "\n\n" + (body.strip() or msg.get("snippet", ""))


_HEADER = re.compile(r"^(Subject|To|Cc|Thread): ")
_LIST_ITEM = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s")


def _body(text: str) -> str:
    """The text a reader counts: without the header block ``extract: "gmail"`` puts first."""
    blocks = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    if len(blocks) == 2 and all(_HEADER.match(line) for line in blocks[0].splitlines() if line.strip()):
        return blocks[1]
    return text


def _paragraphs(text: str) -> list[str]:
    """Paragraphs as a reader counts them: a greeting or sign-off line ("Dana,", "Thanks, Fred") isn't one, and a
    list continues the paragraph before it."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) <= 2 and sum(len(line.split()) for line in lines) <= 4 and lines[-1].rstrip()[-1:] not in ".!?:":
            continue  # a greeting or sign-off; a short sentence, or a line ending in ":" leading into a list, counts
        if out and all(_LIST_ITEM.match(line) for line in lines):
            out[-1] += "\n" + block
            continue
        out.append(block)
    return out


def _format(text: str, fmt: dict) -> tuple[bool, str]:
    problems = []
    body = _body(text)  # counts are of the body; contains_* also see the header block (the thread id is there)
    paragraphs = _paragraphs(body)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s.strip()]
    if fmt.get("max_paragraphs") is not None and len(paragraphs) > fmt["max_paragraphs"]:
        problems.append(f"{len(paragraphs)} paragraphs > {fmt['max_paragraphs']}")
    if fmt.get("max_sentences") is not None and len(sentences) > fmt["max_sentences"]:
        problems.append(f"{len(sentences)} sentences > {fmt['max_sentences']}")
    low = text.lower()
    problems += [f"missing {t!r}" for t in fmt.get("contains_all") or [] if t.lower() not in low]
    problems += [f"has {t!r}" for t in fmt.get("contains_none") or [] if t.lower() in low]
    return (not problems), ("format ok" if not problems else "; ".join(problems))


async def _judge(spec, statement: str, text: str, *, model: Optional[str] = None, system: str = JUDGE_SYSTEM,
                 max_chars: Optional[int] = 40000) -> tuple[bool, str]:
    """One judge call on ``statement`` against ``text``, cut to ``max_chars`` (an artifact's text; a trajectory gate
    sizes its own log, so passes None)."""
    import httpx
    from agent_env.config import get_config
    config = get_config()
    base = config.get_litellm_base_url().rstrip("/")
    headers = {"Authorization": f"Bearer {config.get_litellm_api_key()}", **(config.get_model_headers(base) or {})}
    body = {"model": model or spec["judge_model"], "max_tokens": 2000, "response_format": JUDGE_FORMAT,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"STATEMENT: {statement}\n\nDOCUMENT:\n{text if max_chars is None else text[:max_chars]}"}]}
    problem = ""
    for _ in range(2):  # one retry; then the gate is unknown, never a fail
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{base}/chat/completions", json=body, headers=headers, timeout=120)
            r.raise_for_status()
            return _judge_answer(r.json()["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001
            problem = f"{type(e).__name__}: {str(e)[:150]}"
    return None, f"judge unknown: {problem}"


# ---- trajectory_judge: the agent's work, read from its trajectory -------------------------------------------------

async def _trajectory_gate(turn, spec, state, statement: str) -> tuple[Optional[bool], str]:
    """The statement judged against every turn's rendered trajectory so far, the open milestone's turns (from
    ``state["since"]``) marked THIS STEP. Unknown (None) when the framework hands the hook no trajectories."""
    urls = tuple(getattr(turn, "trajectory_urls", ()) or ())
    if not urls:
        return None, "unknown: no trajectories (the framework or the agent offers none)"
    step = range(state.get("since", 1), turn.number + 1)
    missing = [n for n in step if n > len(urls) or urls[n - 1] is None]
    if missing:  # the step's own work can't be seen, so a fail would be a guess; an earlier turn's gap is only context
        return None, f"unknown: no trajectory for turn(s) {missing} of this step (the fetch failed or returned none)"
    texts = []
    for n, url in enumerate(urls, 1):
        if url is None:
            texts.append((n, "(no trajectory for this turn)"))
            continue
        if url not in _RENDERED:
            messages = await _messages(turn, n, url)
            _RENDERED[url] = render_trajectory(messages) if messages is not None else "(empty trajectory)"
        if n in step and not _RENDERED[url].strip():
            # Work in a format this renderer doesn't read (not claude-code's stream-json) would show as nothing.
            return None, f"unknown: turn {n}'s trajectory isn't in a format this hook reads"
        texts.append((n, _RENDERED[url]))
    log = _evidence(texts, step, spec["trajectory_max_chars"])
    return await _judge(spec, statement, log, model=spec["trajectory_judge_model"], system=TRAJECTORY_JUDGE_SYSTEM,
                        max_chars=None)


_RENDERED: dict[str, str] = {}  # rendered turns by trajectory URL, kept for the step: a turn's trajectory never changes
_MESSAGES: dict[str, Optional[list]] = {}  # each turn's messages by URL, read once for every gate that reads it


async def _messages(turn, n: int, url: str) -> Optional[list]:
    if url not in _MESSAGES:
        raw = await turn.read_trajectory(n)
        _MESSAGES[url] = json.loads(raw) if raw else None
    return _MESSAGES[url]


def _opened_turn(spec: dict, records: list[dict], mid: str) -> Optional[int]:
    """The turn milestone ``mid``'s line reached the agent (the turn after the one that opened or skipped it), or
    None if it hasn't been reached."""
    index = [m["id"] for m in spec["milestones"]].index(mid)
    if index == 0:
        return 1
    for r in records:
        if (r.get("next_index") == index and not r.get("wrap_up")) or any(
                s["milestone"] == mid for s in r.get("skipped") or []):
            return int(r.get("turn", 0)) + 1
    return None


async def _trajectory_check(turn, spec, state, check: dict) -> tuple[Optional[bool], str]:
    """Code over the agent's tool calls and what it said in the turns ``scope`` names: none carries a
    ``contains_none`` text, and no call matches a ``no_call``. A breach in the turns that could be read fails it; a
    turn in scope that couldn't be read otherwise makes it unknown."""
    scope = check.get("scope", "step")
    if scope == "step":
        turns = range(state.get("since", 1), turn.number + 1)
    elif scope == "run":
        turns = range(1, turn.number + 1)
    else:
        opened = _opened_turn(spec, turn.records(), scope["before"])
        turns = range(1, (opened if opened is not None else turn.number + 1))
    urls = tuple(getattr(turn, "trajectory_urls", ()) or ())
    texts = [t.lower() for t in check.get("contains_none") or []]
    breaches, unread = [], []
    for n in turns:
        url = urls[n - 1] if n <= len(urls) else None
        try:
            messages = await _messages(turn, n, url) if url is not None else None
        except Exception:  # noqa: BLE001 - a turn that can't be fetched or parsed is unread, not a breach
            messages = None
        calls, said = _work(messages if isinstance(messages, list) else [])
        if not messages or not (calls or said):  # not fetched, empty, or a format this hook doesn't read
            unread.append(n)
            continue
        for name, args in calls:
            short, values = name.split("__")[-1], [s.lower() for s in _strings(args)]
            breaches += [f"turn {n}: {short} has {t!r}" for t in texts if any(t in v for v in values)]
            for rule in check.get("no_call") or []:
                if (not rule.get("tool") or rule["tool"] in (short, name)) and \
                        (not rule.get("args_contains") or any(rule["args_contains"].lower() in v for v in values)):
                    breaches.append(f"turn {n}: called {short}({_cut(args, 120)})")
        for text in said:
            breaches += [f"turn {n}: said {t!r}" for t in texts if t in text.lower()]
    if breaches:
        return False, "; ".join(breaches[:5]) + (f" (+{len(breaches) - 5} more)" if len(breaches) > 5 else "")
    if unread:
        return None, f"unknown: no readable trajectory for turn(s) {unread}"
    return True, f"clean over turn(s) {turns.start}-{turns.stop - 1}" if len(turns) else "no turns in scope"


def _strings(value: Any) -> list[str]:
    """Every text a tool call's arguments hold, nested values included, as written (not JSON-escaped)."""
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [] if value is None else [value if isinstance(value, str) else json.dumps(value)]


def _work(messages: list) -> tuple[list[tuple[str, Any]], list[str]]:
    """A claude-code turn's (stream-json) tool calls, as (name, input), and the text the agent wrote."""
    calls, said = [], []
    for m in messages:
        if not isinstance(m, dict) or m.get("type") != "assistant":
            continue
        for block in (m.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append((block.get("name", "?"), block.get("input")))
            elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                said.append(block["text"])
    return calls, said


def _evidence(texts: list[tuple[int, str]], step: range, max_chars: int) -> str:
    """The turns' logs, step turns marked; past ``max_chars``, the oldest turns before the step are cut first."""
    blocks = [(n, f"===== Turn {n}{' (THIS STEP)' if n in step else ''} =====\n{t}") for n, t in texts]
    while sum(len(b) for _, b in blocks) > max_chars and blocks and blocks[0][0] not in step:
        blocks.pop(0)
    out = "\n\n".join(b for _, b in blocks)
    return out if len(out) <= max_chars else out[-max_chars:]


# Writes in full (they are what a gate judges); reads and the agent's text shortened. trajectory_max_chars bounds it all.
TRAJ_WRITE_CHARS, TRAJ_READ_CHARS, TRAJ_TEXT_CHARS = None, 160, 1500
_WRITE_VERBS = re.compile(r"(create|update|add|send|reply|post|comment|delete|remove|trash|write|batch|append|set_|"
                          r"share|move|rename|edit|insert|archive|label|assign|draft|upload|invite|cancel|close)", re.I)
_READ_VERBS = re.compile(r"^(\w+_)?(get|list|search|read|find|query|fetch|view|describe|lookup|download|export|count)"
                         r"(_|$)", re.I)


def is_write(tool: str) -> bool:
    """A call that changes something: a write verb in its name, which doesn't start (after the app prefix) with a
    read verb, so trello_get_comments and gmail_get_messages_batch are reads."""
    name = tool.split("__")[-1]
    return not _READ_VERBS.match(name) and bool(_WRITE_VERBS.search(name))


def _cut(value, limit: Optional[int]) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text if limit is None or len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"


def _result_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def render_trajectory(messages: list) -> str:
    """One claude-code turn trajectory (stream-json messages) as text: each tool call in order with its result,
    writes in full, reads up to TRAJ_READ_CHARS, and the agent's text. Empty for a trajectory in another format,
    which the trajectory gate then treats as unreadable."""
    results: dict = {}
    for m in messages:
        if isinstance(m, dict) and m.get("type") == "user":
            for block in (m.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results[block.get("tool_use_id")] = _result_text(block.get("content"))
    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("type") != "assistant":
            continue
        for block in (m.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                lines.append(f"AGENT SAYS: {_cut(block['text'], TRAJ_TEXT_CHARS)}")
            elif block.get("type") == "tool_use":
                tool = block.get("name", "?")
                write = is_write(tool)
                limit = TRAJ_WRITE_CHARS if write else TRAJ_READ_CHARS
                lines.append(f"{'WRITE' if write else 'read'} {tool.split('__')[-1]}({_cut(block.get('input'), limit)})")
                if block.get("id") in results:
                    lines.append(f"  -> {_cut(results[block['id']], limit)}")
    return "\n".join(lines)


def _judge_answer(content: Optional[str]) -> tuple[bool, str]:
    content = content or ""
    got = json.loads(content[content.find("{"): content.rfind("}") + 1])
    return bool(got["passed"]), str(got.get("why", ""))[:300]


async def _excerpt(turn, spec, env, m, state) -> str:
    name = m.get("show")
    if not name:
        return ""
    try:
        text = await _artifact(env, spec["role"], m, name, {})
    except Exception:  # noqa: BLE001 - the excerpt is a courtesy; the nudge stands without it
        return ""
    if not text:
        return ""
    cut = text[: spec["excerpt_chars"]]
    return f"\nYou opened it and saw:\n{cut}" + ("\n[...]" if len(text) > len(cut) else "")


# ---- params --------------------------------------------------------------------------------------------------------

def _env(turn, spec):
    envs = turn.envs
    if not envs:
        raise RuntimeError("milestone-gates needs a deployed env to read")
    if spec.get("env"):
        match = [e for e in envs if e.env_id == spec["env"]]
        if not match:
            raise RuntimeError(f"no deployed env {spec['env']!r}")
        return match[0]
    return envs[0]


GATE_KEYS = {"id", "nudge", "check", "type"}
GATE_TYPES = ("format", "action", "content")
FORMAT_KEYS = {"max_paragraphs", "max_sentences", "contains_all", "contains_none"}
TRAJECTORY_KEYS = {"contains_none", "no_call", "scope"}
CHECK_KINDS = ("trajectory_judge", "trajectory", "call", "no_new", "no_removed", "artifact")


def _check_problem(check: dict) -> Optional[str]:
    """What's wrong with a gate's check, or None: exactly one kind, with what that kind needs."""
    kinds = [k for k in CHECK_KINDS if k in check]
    if len(kinds) != 1:
        return f"check needs exactly one of {', '.join(CHECK_KINDS)}; got {sorted(check)}"
    kind = kinds[0]
    if kind == "trajectory_judge":
        return None if isinstance(check[kind], str) and check[kind].strip() else "trajectory_judge needs a statement"
    if kind == "trajectory":
        return _trajectory_problem(check[kind])
    if kind == "call":
        if not isinstance(check["call"], dict) or not check["call"].get("tool"):
            return "call needs a tool"
        return _expect_problem(check.get("expect"))
    if kind in ("no_new", "no_removed"):
        calls = check[kind]
        if not isinstance(calls, list) or not calls or not all(isinstance(c, dict) and c.get("tool") for c in calls):
            return f"{kind} needs a list of calls, each with a tool"
        bad = [c["baseline_count"] for c in calls if c.get("baseline_count") is not None and not _whole(c["baseline_count"])]
        return f"{kind} baseline_count must be a whole number, got {bad[0]!r}" if bad else None
    how = [k for k in ("expect", "format", "judge") if k in check]
    if len(how) != 1:
        return f"an artifact check needs exactly one of expect, format, judge; got {how or 'none'}"
    if how == ["expect"] and check["expect"] != "exists":
        return "an artifact's expect can only be \"exists\""
    if how == ["format"]:
        fmt = check["format"]
        if not isinstance(fmt, dict) or not fmt or set(fmt) - FORMAT_KEYS:
            return f"format needs one or more of {sorted(FORMAT_KEYS)}"
        for key in ("max_paragraphs", "max_sentences"):
            if key in fmt and not _whole(fmt[key]):
                return f"format.{key} must be a whole number, got {fmt[key]!r}"
        for key in ("contains_all", "contains_none"):
            if key in fmt and not (isinstance(fmt[key], list) and fmt[key]
                                   and all(isinstance(t, str) and t for t in fmt[key])):
                return f"format.{key} must be a list of text, got {fmt[key]!r}"
    if how == ["judge"] and not (isinstance(check["judge"], str) and check["judge"].strip()):
        return "judge needs a statement"
    return None


def _trajectory_problem(spec: Any) -> Optional[str]:
    if not isinstance(spec, dict) or set(spec) - TRAJECTORY_KEYS:
        return f"trajectory takes {sorted(TRAJECTORY_KEYS)}"
    none, calls = spec.get("contains_none"), spec.get("no_call")
    if none is None and calls is None:
        return "trajectory needs contains_none or no_call"
    if none is not None and not (isinstance(none, list) and none and all(isinstance(t, str) and t for t in none)):
        return f"trajectory.contains_none must be a list of text, got {none!r}"
    if calls is not None and not (isinstance(calls, list) and calls and all(
            isinstance(c, dict) and c and not set(c) - {"tool", "args_contains"}
            and all(isinstance(v, str) and v for v in c.values()) for c in calls)):
        return "trajectory.no_call must be a list of {tool?, args_contains?}, each with at least one, as text"
    scope = spec.get("scope", "step")
    if scope not in ("step", "run") and not (isinstance(scope, dict) and set(scope) == {"before"}
                                             and isinstance(scope["before"], str) and scope["before"]):
        return f'trajectory.scope must be "step", "run" or {{"before": milestone id}}, got {scope!r}'
    return None


def _expect_problem(expect: Any) -> Optional[str]:
    if expect == "exists":
        return None
    if not isinstance(expect, dict) or len(expect) != 1 or not set(expect) <= {"count", "contains", "equals"}:
        return 'expect must be "exists" or one of {count}, {contains}, {equals}'
    if "count" in expect and (not isinstance(expect["count"], dict) or not expect["count"]
                              or set(expect["count"]) - {"eq", "min", "max"}
                              or not all(_whole(v) for v in expect["count"].values())):
        return "expect.count needs eq, min or max, each a whole number"
    if "contains" in expect and not (isinstance(expect["contains"], str) and expect["contains"]):
        return "expect.contains must be text"
    return None


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _spec(params: dict) -> dict:
    if not isinstance(params.get("enabled", True), bool):
        raise ValueError("milestone-gates enabled must be true or false")
    if not isinstance(params.get("role"), str) or not params["role"]:
        raise ValueError("milestone-gates needs role: the env role its gates read as")
    milestones = params.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("milestone-gates needs milestones")
    out, authored = [], []  # authored: (gate id, overrides a carried gate) for every gate the spec writes
    for i, m in enumerate(milestones):
        unknown = set(m) - MILESTONE_KEYS
        if unknown:
            raise ValueError(f"milestone {i} has unknown keys {sorted(unknown)}")
        mid = m.get("id") or f"M{i + 1}"
        if any(mid == o["id"] for o in out):
            raise ValueError(f"milestone ids must be unique: {mid!r} is repeated")
        prev = out[-1] if out else None
        group, carry = m.get("group"), m.get("carry_gates", False)
        if not isinstance(carry, bool):
            raise ValueError(f"milestone {mid}: carry_gates must be true or false")
        if group is not None:
            # A revision names the milestone it revises: an earlier one of the group just before it, so a
            # milestone's revisions run one after another.
            named = next((o for o in out if o["id"] == group), None)
            if named is None:
                raise ValueError(f"milestone {mid}: group {group!r} names no earlier milestone")
            if named["group"] != prev["group"]:
                raise ValueError(f"milestone {mid}: a revision of {named['group']} must follow it or its last "
                                 f"revision, not {prev['id']}")
        elif carry:
            raise ValueError(f"milestone {mid}: carry_gates needs group (the milestone it revises)")
        own = m.get("gates") or []
        if not own:
            raise ValueError(f"milestone {mid} has no gates")
        own_ids = [g.get("id") for g in own if isinstance(g, dict) and g.get("id")]
        twice = sorted({gid for gid in own_ids if own_ids.count(gid) > 1})
        if twice:  # two gates of one id: one override would replace the other, and its check would never run
            raise ValueError(f"milestone {mid}: gate ids must be unique within a milestone; repeated: {twice}")
        carried = {g["id"]: g for g in prev["gates"]} if carry else {}
        artifacts = {**(prev["artifacts"] if carry else {}), **(m.get("artifacts") or {})}
        for g in own:
            if not g.get("id") or not g.get("nudge") or not isinstance(g.get("check"), dict):
                raise ValueError(f"milestone {mid}: each gate needs id, nudge and check")
            unknown_gate = set(g) - GATE_KEYS
            if unknown_gate:
                raise ValueError(f"gate {g['id']} has unknown keys {sorted(unknown_gate)}")
            if g.get("type") is not None and g["type"] not in GATE_TYPES:
                raise ValueError(f"gate {g['id']}: type {g['type']!r} is not one of {list(GATE_TYPES)}")
            art = g["check"].get("artifact")
            if art is not None and art not in artifacts:
                raise ValueError(f"gate {g['id']} names artifact {art!r}, not in its milestone's artifacts")
            problem = _check_problem(g["check"])
            if problem:
                raise ValueError(f"gate {g['id']}: {problem}")
            if i == 0:
                for kind in ("no_new", "no_removed"):
                    if any(c.get("baseline_count") is None for c in g["check"].get(kind) or []):
                        raise ValueError(f"gate {g['id']}: a {kind} gate in the first milestone needs baseline_count")
            authored.append((g["id"], g["id"] in carried))
        # Carried gates keep their place; one the revision overrides (same id) takes its place, new ones follow.
        mine = {g["id"]: g for g in own}
        gates = [mine.pop(gid) if gid in mine else {**g, "carried": True} for gid, g in carried.items()]
        gates += list(mine.values())
        revision = prev["revision"] + 1 if group is not None else 1
        out.append({"id": mid, "line": m.get("line"), "blocking": m.get("blocking", True),
                    "max_nudges": int(m.get("max_nudges", 3)), "show": m.get("show"), "type": m.get("type"),
                    # a revision asks for a change to work already done, so by default it's never skipped as done
                    "skip_if_done": bool(m.get("skip_if_done", group is None)),
                    "group": prev["group"] if group is not None else mid, "revision": revision,
                    "artifacts": artifacts, "gates": gates})
    ids = [gid for gid, override in authored if not override]
    repeated = sorted({gid for gid in ids if ids.count(gid) > 1})
    if repeated:
        raise ValueError(f"gate ids must be unique (a revision with carry_gates may reuse one it carries, to "
                         f"override it); repeated: {repeated}")
    checks = [g["check"] for m in out for g in m["gates"]]
    known = {m["id"] for m in out}
    for c in checks:
        before = (c.get("trajectory") or {}).get("scope")
        if isinstance(before, dict) and before["before"] not in known:
            raise ValueError(f"trajectory scope names milestone {before['before']!r}, which the spec doesn't have")
    for key, kind in (("judge_model", "judge"), ("trajectory_judge_model", "trajectory_judge")):
        if any(kind in c for c in checks) and not (isinstance(params.get(key), str) and params[key].strip()):
            raise ValueError(f"milestone-gates needs {key}: a gate uses {kind}")
    return {"enabled": params.get("enabled", True), "role": params["role"], "env": params.get("env"), "milestones": out,
            "judge_model": params.get("judge_model"),
            "trajectory_judge_model": params.get("trajectory_judge_model"),
            "trajectory_max_chars": int(params.get("trajectory_max_chars", 600_000)),
            "excerpt_chars": int(params.get("excerpt_chars", 600)),
            "final_note": params.get("final_note", DEFAULT_FINAL),
            "final_note_gave_up": params.get("final_note_gave_up", DEFAULT_FINAL_GAVE_UP),
            "idle_end": int(params.get("idle_end", 2))}
