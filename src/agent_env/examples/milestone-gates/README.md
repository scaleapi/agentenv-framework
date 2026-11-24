A prompt_agent post-turn hook that runs a multiturn task as a chain of milestones, each blocked by a few gates checked
while the conversation runs, and tells the user sim what it found.

`artifacts/milestone-gates/hook.py` is the hook. After each agent turn it runs the open milestone's gates against the
env and the agent's own work, then asks the user side with a note only the user sim sees (`turn.ask_user(note=...)`;
the agent never sees it):

- every gate passes: the milestone is done, and the note opens the next one with its `line`, or asks the user sim to
  wrap up after the last;
- a gate fails: the note carries that gate's `nudge`, which the task author writes and keeps vague (where to look, never
  the value), and the turn counts toward the milestone's `max_nudges`;
- still failing after `max_nudges` turns: a `blocking` milestone ends the conversation (`gate_never_passed:<id>`), any
  other is let go and the next one opens;
- no gate could be checked at all (say, a turn's trajectory wasn't fetched): the user sim is asked to stay on the
  milestone, and the turn counts toward `max_nudges`, so evidence that never arrives ends in the usual give-up.

The user sim words every message; the hook decides only where the conversation is. A user sim that moves on by itself
(its message carries a later milestone's `line`) is followed. When a milestone passes, later ones the agent has already
done are skipped. The conversation ends with `milestones_done` once the wrap-up is answered, or with `sim_idle` when
the user sim sends the same message `idle_end` turns in a row.

Gates are cheap first: code and tool-state checks run before any judge, and a judge runs only when they pass. A judge
that can't answer after one retry leaves its gate unknown, which neither fails nor nudges.

```json
{"id": "talk", "type": "prompt_agent", "prompt": "Make me a sheet of every active subscription.",
 "user_agent_name": "user", "max_conversation_turns": 20,
 "post_turn_hooks": [{"file_artifact_id": "@local/agentenv-framework/milestone-gates/milestone-gates", "params": {
   "role": "admin",
   "milestones": [
     {"id": "M1", "max_nudges": 3, "blocking": true,
      "artifacts": {"sheet": {"find": {"tool": "drive_list", "path": "files", "where": {"name": "subscriptions"}}}},
      "gates": [
        {"id": "m1-sheet", "nudge": "I don't see the sheet yet.", "check": {"artifact": "sheet", "expect": "exists"}},
        {"id": "m1-quiet", "nudge": "Hold off on emailing anyone.",
         "check": {"no_new": [{"tool": "gmail_sent", "path": "messages", "baseline_count": 0}]}}]},
     {"id": "M2", "line": "Now put what you found on the cut-list card. Don't email anyone yet.",
      "gates": [
        {"id": "m2-comment", "nudge": "Did the comment go on the card?",
         "check": {"trajectory_judge": "During this step the assistant adds one comment to the cut-list card."}}]}]}}]}
```

The gates are opt-in: a task runs them only when a `prompt_agent` step lists the hook in `post_turn_hooks`, and no
other step or task is affected. `"enabled": false` switches them off while keeping the milestones in the task, for a
run with and without gates: the hook still checks its params, but runs no gate and sends the user sim no note, so
the conversation goes as if no hook were listed. It records `disabled` once.

Params:

- `enabled`: `true` by default; `false` switches the gates off, as above.
- `milestones`: `[{id, line?, type?, blocking?, max_nudges?, skip_if_done?, show?, artifacts, gates}]`, in order. `line`
  is what the user sim sends to open the milestone (the first one's is the step's prompt). `show` names an artifact
  whose start goes in a nudge's note, so the user sim can react to what it "opened". `type: "correct"` is never
  skipped.
- `role`: the env role the gates read as; it needs read access to everything they check.
- `env`: the deployed env's `env_id` to read; the first deployed env by default.
- `judge_model`: the model for `judge` gates on an artifact. `trajectory_judge_model`: the model for `trajectory_judge`
  gates; a small model misreads long logs, so pick a strong one.
- `trajectory_max_chars`: the most rendered trajectory one judge call gets (600000 by default); the oldest turns before
  the open milestone are cut first.
- `excerpt_chars` (600), `final_note`, `idle_end` (2; 0 never ends on it).

A gate is `{id, nudge, check}`. The check kinds, by the gate types of a multiturn task:

| Gate type | Check | What it reads |
|---|---|---|
| format | `{"artifact": name, "format": {max_paragraphs, max_sentences, contains_all, contains_none}}` | the artifact's text, by code |
| action | `{"call": call, "expect": expect}` | a read tool's result: `"exists"`, `{"count": {"eq"/"min"/"max": n}}`, `{"contains": text}`, `{"equals": value}` |
| action | `{"artifact": name, "expect": "exists"}` | the artifact was found |
| action | `{"no_new": [call + {baseline_count?}]}` | nothing added since the milestone opened |
| content | `{"artifact": name, "judge": "statement"}` | one model call over the artifact's text |
| content | `{"trajectory_judge": "statement"}` | one model call over the agent's work: every turn's trajectory so far, writes in full, reads shortened, the open milestone's turns marked `THIS STEP` |

A call is `{tool, args?, path?, where?, number?}`: `path` is a dotted path into the tool's JSON result, `where` keeps
list items whose fields contain the given text, `number` reads a count the tool reports. An artifact is
`{find: call + {pick?: "first" | "last"}, read?: {tool, args, extract?}, field?}`: `read` args may use `"{field}"`
from the item `find` picked, and `extract: "gmail"` turns a Gmail API message into its headers and plain-text body.
Without `read`, `field` (a dotted path) takes that field of the picked item as the artifact's text, such as a draft's
`body`; otherwise the text is the whole item as JSON, where a format check would count escaped line breaks.

The params are checked when the step loads the hook, before the first message (the hook's `check_params`): unknown
keys on a milestone or gate, a check that isn't exactly one kind or lacks what its kind needs, an artifact a milestone
doesn't define, a gate `type` outside format, action and content, repeated gate or milestone ids, and a judge gate
with no judge model each fail the step there. Without this, a malformed check would fail as a gate on every turn and
nudge the agent about nothing.

After the wrap-up has been answered, the hook checks the last milestone once more and records it, so the summary matches what the agent left: work it changed while answering the wrap-up counts. If any milestone was let go, the wrap-up note (`final_note_gave_up`) asks the user sim to wrap up without saying everything is done and without taking on anything new.

After every turn the hook also sets its summary of where the conversation got to, for a verifier or the hub to read
without going through its records (`turn.summarize`), at
`metadata["post_turn_hook_summaries"][step_id]["<this hook's artifact id>"]`:

```json
{"enabled": true, "reached": "M2", "done": 2, "total": 2, "end": "milestones_done",
 "milestones": [{"id": "M1", "status": "passed", "opened_turn": 1, "closed_turn": 1, "nudges": 0,
                 "gates": {"m1-draft": {"passed": true, "type": "action"}, "m1-short": {"passed": true, "type": "format"}}}]}
```

A milestone's `status` is `not_reached`, `open`, `passed`, `skipped` (done before it opened), `gave_up` (let go after
`max_nudges`), `passed_over` (the user sim moved past it) or `derailed` (a blocking one that never passed, which ended
the conversation: `end` is then `gate_never_passed:<id>`). A gate's `type` follows its check, as in the table above.
`end` stays null when the conversation ended some other way, such as running out of turns. Switched off, the summary
is `{"enabled": false}`.

The hook keeps its state in its own records (`metadata["post_turn_hooks"]`): each turn's milestone, every gate's
verdict and reason, the decision (passed, nudge, hold, gave_up, end) and the user sim's answer. `trajectory_judge`
gates need the agent to offer per-turn trajectories (`turn.trajectory_urls`); without them those gates are unknown.

The user sim should treat the note as what it found when it checked, and never judge "done" itself: the hook ends the
conversation. A hook runs in the control-plane process with the trust of an `env_outcome_verifier` script, and its
model calls are not in the step's usage.
