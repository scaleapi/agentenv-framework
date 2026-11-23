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
  final_note      the note after the last milestone passes
  idle_end        end after the sim sends the same message this many turns in a row (default 2; 0 = never)
  trajectory_judge_model   model for trajectory_judge gates (needed when one is used; a small model misreads long
                           logs)
  trajectory_max_chars     the most rendered trajectory one judge call gets (default 600000); the oldest turns
                           before the open milestone are cut first
  milestones      [{id, line?, type?, skip_if_done?, blocking?, max_nudges?, show?, artifacts: {name: artifact},
                    gates: [gate]}]
                  ``line`` is the text the sim sends to open the milestone, used to tell when it has moved on.
                  When a milestone is done, each next one the agent has already done is skipped (its gates all pass
                  then, at least one of them on something the agent made): the note tells the sim to skip its line.
                  ``skip_if_done: false`` turns that off for a milestone; type ``correct`` is never skipped.

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
A call is {tool, args?, path?, where?, number?}: ``path`` is a dotted path into the tool's JSON result, ``where``
keeps the list items whose fields contain the given text, and ``number`` reads a count the tool reports (that field of
the first item) instead of counting the list. ``expect`` is "exists", or {"count": {"eq"|"min"|"max": n}},
{"contains": text} or {"equals": value}.
An artifact is {find: call + {pick?: "first"|"last"}, read?: {tool, args, extract?}}: ``read`` args may use
"{field}" from the item ``find`` picked, and ``extract: "gmail"`` turns a Gmail API message into its headers and
plain-text body; without ``read`` the artifact's text is the picked item itself.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# When none of the open milestone's gates could be checked: the sim stays on it, and it counts toward max_nudges, so a
# milestone whose evidence never arrives is let go (or ends the run, if blocking) instead of holding forever.
HOLD_NOTE = ("Nothing could be checked this time. Don't move on: reply to what the assistant said, and stay on the "
             "milestone you're on.")
MILESTONE_KEYS = {"id", "line", "type", "skip_if_done", "blocking", "max_nudges", "show", "artifacts", "gates"}
DEFAULT_FINAL = "Everything you asked for is done. Wrap up the way you planned (send your last line), then stop."
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


async def post_turn(turn, params):
    spec = _spec(params)
    state = _state(turn.records())
    milestones = spec["milestones"]
    if state["finished"]:  # the wrap-up was sent and the agent has answered it
        turn.record({"decision": "done"})
        return {"end": "milestones_done"}
    if state["idle"]:  # the sim repeated itself last turn; the agent has answered that, so end before asking again
        turn.record({"decision": "end", "end": "sim_idle"})
        return {"end": "sim_idle"}
    m = milestones[state["index"]]
    env = _env(turn, spec)
    gates = await _run_gates(turn, spec, env, m, state)
    failed = [g for g in gates if g["passed"] is False]
    entry = {"milestone": m["id"], "gates": gates}
    if turn.last:  # nothing more can be asked: record where it stood
        turn.record({**entry, "decision": "last_turn", "nudges": state["nudges"]})
        return None
    unchecked = bool(gates) and all(g["passed"] is None for g in gates)
    if not failed and not unchecked:
        return await _advance(turn, spec, env, state, entry, "passed")
    if state["nudges"] >= m["max_nudges"]:
        if m["blocking"]:
            turn.record({**entry, "decision": "end", "nudges": state["nudges"]})
            return {"end": f"gate_never_passed:{m['id']}"}
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
            since = int(r.get("turn", 0)) + 1  # its line goes to the agent in the next turn
        elif r.get("decision") in ("nudge", "hold"):
            nudges = r["nudges"]
    return {"index": index, "nudges": nudges, "baselines": baselines, "finished": finished, "idle": idle,
            "since": since}


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
        made = [g for g in m["gates"] if "no_new" not in g["check"]]
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
        note = (f"{head} " if how == "gave_up" or skipped else "") + spec["final_note"]
        record = {**entry, "decision": how, "next_index": state["index"], "wrap_up": True, "note": note}
    if skipped:
        record["skipped"] = skipped
    answer = await turn.ask_user(note=note)
    return await _after_answer(turn, spec, env, state, record, answer, record["next_index"])


def _sim_field(turn, name) -> Any:
    """The peer sim's own field for this turn (prompt_agent records extra JSON fields), or None."""
    outputs = (turn.metadata.get("usersim_turn_outputs") or {})
    for entries in outputs.values():
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
            passed, why = await _no_new(env, spec["role"], m, state, check["no_new"])
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


async def _no_new(env, role, m, state, calls) -> tuple[bool, str]:
    added = []
    for i, call in enumerate(calls):
        value = await _call(env, role, call)
        n = _size(value)
        base = state["baselines"].get(f"{m['id']}:{i}", call.get("baseline_count"))
        if base is None:
            raise ValueError(f"no baseline for {call['tool']} in {m['id']}")
        if n > base:
            added.append(f"{call['tool']} +{n - base}")
    return (not added), ("nothing new" if not added else "new: " + ", ".join(added))


async def _baselines(env, spec, m) -> dict:
    out = {}
    for g in m["gates"]:
        for i, call in enumerate(g["check"].get("no_new") or []):
            value = await _call(env, spec["role"], call)
            out[f"{m['id']}:{i}"] = _size(value)
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
        text = _text(await env.call_tool(a["read"]["tool"], args, role))
        if a["read"].get("extract") == "gmail":
            text = _gmail_text(text)
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
            raw = await turn.read_trajectory(n)
            _RENDERED[url] = render_trajectory(json.loads(raw)) if raw else "(empty trajectory)"
        texts.append((n, _RENDERED[url]))
    log = _evidence(texts, step, spec["trajectory_max_chars"])
    return await _judge(spec, statement, log, model=spec["trajectory_judge_model"], system=TRAJECTORY_JUDGE_SYSTEM,
                        max_chars=None)


_RENDERED: dict[str, str] = {}  # rendered turns by trajectory URL, kept for the step: a turn's trajectory never changes


def _evidence(texts: list[tuple[int, str]], step: range, max_chars: int) -> str:
    """The turns' logs, step turns marked; past ``max_chars``, the oldest turns before the step are cut first."""
    blocks = [(n, f"===== Turn {n}{' (THIS STEP)' if n in step else ''} =====\n{t}") for n, t in texts]
    while sum(len(b) for _, b in blocks) > max_chars and blocks and blocks[0][0] not in step:
        blocks.pop(0)
    out = "\n\n".join(b for _, b in blocks)
    return out if len(out) <= max_chars else out[-max_chars:]


TRAJ_WRITE_CHARS, TRAJ_READ_CHARS, TRAJ_TEXT_CHARS = 3000, 160, 1500
_WRITE_VERBS = re.compile(r"(create|update|add|send|reply|post|comment|delete|remove|trash|write|batch|append|set_|"
                          r"share|move|rename|edit|insert|archive|label|assign|draft|upload|invite|cancel|close)", re.I)
_READ_VERBS = re.compile(r"^(\w+_)?(get|list|search|read|find|query|fetch|view|describe|lookup|download|export|count)"
                         r"(_|$)", re.I)


def is_write(tool: str) -> bool:
    """A call that changes something: a write verb in its name, which doesn't start (after the app prefix) with a
    read verb, so trello_get_comments and gmail_get_messages_batch are reads."""
    name = tool.split("__")[-1]
    return not _READ_VERBS.match(name) and bool(_WRITE_VERBS.search(name))


def _cut(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"


def _result_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def render_trajectory(messages: list) -> str:
    """One claude-code turn trajectory (stream-json messages) as text: each tool call in order with its result,
    writes up to TRAJ_WRITE_CHARS, reads up to TRAJ_READ_CHARS, and the agent's text."""
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


def _spec(params: dict) -> dict:
    if not isinstance(params.get("role"), str) or not params["role"]:
        raise ValueError("milestone-gates needs role: the env role its gates read as")
    milestones = params.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("milestone-gates needs milestones")
    out = []
    for i, m in enumerate(milestones):
        unknown = set(m) - MILESTONE_KEYS
        if unknown:
            raise ValueError(f"milestone {i} has unknown keys {sorted(unknown)}")
        gates = m.get("gates") or []
        if not gates:
            raise ValueError(f"milestone {m.get('id', i)} has no gates")
        for g in gates:
            if not g.get("id") or not g.get("nudge") or not isinstance(g.get("check"), dict):
                raise ValueError(f"milestone {m.get('id', i)}: each gate needs id, nudge and check")
            art = g["check"].get("artifact")
            if art is not None and art not in (m.get("artifacts") or {}):
                raise ValueError(f"gate {g['id']} names artifact {art!r}, not in its milestone's artifacts")
            if i == 0:
                for call in g["check"].get("no_new") or []:
                    if call.get("baseline_count") is None:
                        raise ValueError(f"gate {g['id']}: a no_new gate in the first milestone needs baseline_count")
        out.append({"id": m.get("id") or f"M{i + 1}", "line": m.get("line"), "blocking": m.get("blocking", True),
                    "max_nudges": int(m.get("max_nudges", 3)), "show": m.get("show"),
                    "type": m.get("type"), "skip_if_done": bool(m.get("skip_if_done", True)),
                    "artifacts": m.get("artifacts") or {}, "gates": gates})
    checks = [g["check"] for m in out for g in m["gates"]]
    for key, kind in (("judge_model", "judge"), ("trajectory_judge_model", "trajectory_judge")):
        if any(kind in c for c in checks) and not (isinstance(params.get(key), str) and params[key].strip()):
            raise ValueError(f"milestone-gates needs {key}: a gate uses {kind}")
    return {"role": params["role"], "env": params.get("env"), "milestones": out,
            "judge_model": params.get("judge_model"),
            "trajectory_judge_model": params.get("trajectory_judge_model"),
            "trajectory_max_chars": int(params.get("trajectory_max_chars", 600_000)),
            "excerpt_chars": int(params.get("excerpt_chars", 600)),
            "final_note": params.get("final_note", DEFAULT_FINAL),
            "idle_end": int(params.get("idle_end", 2))}
