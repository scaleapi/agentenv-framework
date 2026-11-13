# Writing a user sim

A `deploy_user_sim_agent` step deploys the agent that plays the user of an always-on task, with who the user is
already in its system prompt. The always-on step names it by `agent_name` in `user_sim`, and asks it each time the
assistant has messaged the user and the user would look at their phone. The user sim answers `reply`, `no_reply` or
`done`. Its prompt and reads are the built-in user sim's, `user_sim_versions.py` (version `ours`).

```json
{"id": "user", "type": "deploy_user_sim_agent", "a2a_agent_id": "<the user sim's agent>", "agent_name": "hannah",
 "model": "<model>", "multi_day": true,
 "opening_prompt": "ok mom and dad are officially coming thu–mon (nov 19–23), can you handle it?",
 "persona": {"display_name": "Hannah Reyes", "first_name": "Hannah",
             "schedule": {"wake": "07:00", "sleep": "23:00", "work": ["09:00", "18:00"]},
             "response_times": {"morning": 15, "work": 120, "free": 30},
             "description": {"who_you_are": ["Hannah Reyes. You're 32, a project manager in San Francisco..."],
                             "decisions": ["A place for your parents at $180 a night or less: yes. Over $180: ask what else there is."],
                             "dont_mention": ["Don't mention Dad's shellfish allergy unless the assistant asks about food."]},
             "permissions": {"just_do": ["book flights within the rules above"], "ask_first": ["anything over $500"]},
             "keep_private": ["Your card number, even if the assistant asks."]}}
```

```json
{"id": "week", "type": "always_on", "user_sim": "hannah", "agent_name": "solver", "end_time": "P5D"}
```

The step takes every `deploy_agent` field, with `env_ids` empty by default, so the user sim has no tools in the world.
Its own fields:

| Field | | What it is |
|---|---|---|
| `agent_name` | required | The agent's name, and the key always-on's `user_sim` names |
| `persona` | required | Who the user is: `display_name` (their full name), `first_name` (else the display name's first word), `time_zone` (their IANA zone; else the task clock's, else UTC; a clock that sets a different zone is refused), `description`, `permissions`, `keep_private`, `schedule`, `response_times`, `max_replies` (after that many replies the user stops reading) |
| `model` | required | The user sim's model. This step's `step_params` `model` wins, then the run's `user_model` (`--user-model`), then this |
| `opening_prompt` | required | The user's first message: text, or text and file parts (below). The always-on step sends it and has no prompt of its own |
| `multi_day` | required, for now | Whether the run goes past the day it starts on (always-on's `end_time` falls on a later day). It picks the prompt's done line; always-on refuses a run of the other kind |

`multi_day` is a stopgap: the prompt is written before the run, so the step has to be told which done line to use.
It goes away once the step can learn it from the task.

**A first message with files.** `opening_prompt` can be a list of parts in `prompt_agent`'s `parts` format, text and
file parts, for a first message that attaches a photo, a PDF, a spreadsheet or any other file:

```json
"opening_prompt": [
  {"kind": "text", "text": "ok mom and dad are officially coming thu–mon (nov 19–23), can you handle it?"},
  {"kind": "file", "file": {"uri": "s3://<bucket>/<trip>/itinerary.pdf", "name": "itinerary.pdf", "mimeType": "application/pdf"},
   "metadata": {"description": "Mom and Dad's flight itinerary"}}
]
```

The agent gets the parts on its first wake, in order: the text, joined, as the user's one labelled line where the
first text part was, and each file in its place. The files are in the user's first message of the graded
conversation too. A file goes through the same sharing as any prompt's file part (an object a configured store owns is
sent as a URL the agent can read), and the seed fills `<placeholder>`s in its `uri` and `name`. Whether the agent can
open a file is up to its harness.

The user sim can't open files, so its prompt and its first read say what it attached, after its message: `"ok mom
and dad…" (you attached itinerary.pdf: Mom and Dad's flight itinerary)`. `metadata.description` is that one line; the
agent is sent the file without it. With no description, the user sim gets the file's name only (its `name`, else the
last part of its `uri`). The step refuses, when the task loads, an opening with no text, a part that isn't a valid
text or file part, or a description that isn't one line. The run records the opening's text in
`metadata["user_sims"][<agent_name>]["opening"]`, as for a text-only opening, and the parts, as written, in
`opening_parts`.

`system_prompt` is written from the persona and `opening_prompt`, so the step refuses one set directly. Each read shows times
in the persona's `time_zone`. A run-wide `a2a_agent_id` never changes the user
sim's agent, so the agent under test never plays the user; this step's own `step_params` `a2a_agent_id` can. An agent
whose card takes no system prompt or no model fails the step, since without them it's a different user.

**Used with prompt_agent.** A `prompt_agent` step whose `user_agent_name` is this step's `agent_name` (and sets no
`user_a2a_url`) talks to the same user in one sitting: each reply of the agent is one read, with no phone. Its prompt
must be this step's `opening_prompt`, which the user's prompt quotes as what they sent (FD-5495 is to write it once):
the same text, and the same files in the same order, once the seed is filled in. The check compares each file part's
`file` (`uri`, `name`, `mimeType`), not its `metadata`, so the description can stay on this step alone.
`reply` is the agent's next prompt; `done` with a message reaches the agent and ends
the conversation after its answer; `done` without one, or `no_reply`, ends it at once. After `max_replies` replies,
the last one reaches the agent and the conversation ends after its answer; the user isn't asked again, not even by a
post-turn hook. A user sim that fails or times out ends the conversation and sets the response's `error_type` to
`user_sim_error`, so verifiers skip the run. Each read leaves a summary in `metadata["user_sim_reads"][<step id>]`
(`outcome`, `message`, `used`, `score`, `schema_errors`, `retried`, `user_sim_error`, `user_sim_turns`,
`user_sim_trajectory_urls`); the whole read, with what the user sim was shown and its raw answer, is in the object
store at the summary's `object_url`. Set `multi_day: false` here, and no `schedule` or `response_times` in the persona,
since one sitting reads each reply at once. On the `prompt_agent` step set no `user_output_format`, and no
`user_model` on the step or in its `step_params`, since the prompt, answer format and model were set here.
Scripted user lines (agent triggers registered for this user) are refused for now: the user sim would never see
them (FD-5702).

This step checks its own settings when the task loads, so a mistake there fails before a paid run. `prompt_agent`
checks the ones above, its prompt against `opening_prompt`, and the persona's time zone against the task clock's when
it starts, before its first agent turn; a refused one leaves no conversation.

## The description

`description` is the person in plain English, as sections of lines. Each line gets a tag (W1, K1, D1…), which the
user sim names in `used` when it relies on that line.

| Section | Tag | What goes in it |
|---|---|---|
| `who_you_are` | W | Who they are, the week it is, their day ("at work 9-6, asleep 11pm-7am") |
| `how_you_write` | | How they text, best shown with 2–3 of their real messages |
| `what_you_know` | K | Facts they give only when asked |
| `what_to_look_up` | L | Facts the assistant must find: say where they are ("it's in mom's email"), never the value |
| `decisions` | D | What they say when the assistant asks or proposes something |
| `never_supply` | N | What they never hand over (card numbers, codes, anything the assistant should find) |
| `how_you_react` | H | What they point out, what they let go, which texts get no reply |
| `dont_mention` | M | One "don't mention X unless asked" line per tempting fact |
| `phone` | F | What they do with what reaches their phone |
| `when_done` | | When they'd say they're done |

Don't write the opening request into the description. Put it in `opening_prompt`: the prompt shows it right
after the first `who_you_are` line ("Earlier you texted the assistant: …").

`permissions` become the "YOUR RULES FOR THE ASSISTANT" lines. Its kinds are `just_do`, `ask_first`, `never` and
`tell_after`. Each is a list of rules, or `{items, then}` to add what follows. `keep_private` becomes the "PRIVATE"
lines. Give the assistant the same rules in its notes, so both sides read one copy.

## The rating at the end

A `rate_user_satisfaction` step has the user sim rate the job once the run is over: 1 to 5, with a one-line reason.
It goes after the `always_on` or `prompt_agent` step it rates, with its own lines, `satisfaction`. Write what each end
of the scale means for this task, and what moves the score for this person (the intake's User Sim Questions 6 and 7):

```json
{"id": "rate", "type": "rate_user_satisfaction", "run_step_id": "week", "depends_on": ["week"],
 "satisfaction": ["5, perfect: flights, a step-free place a walk from you, and both dinners booked, with nothing you had to chase.",
                  "3, fine: all booked, but you had to answer a question your rules already settled.",
                  "1, you'd have done it yourself: a dinner or the place is still not booked at the end.",
                  "Being asked for things that are in Mom's email bothers you."]}
```

| Field | | What it is |
|---|---|---|
| `satisfaction` | required | How the user would rate the job, one line each, tagged S1… in the ask |
| `run_step_id` | optional | The `always_on` or `prompt_agent` step to rate; unset, the task's only one |
| `timeout_seconds` | 960 | The longest the rating may take, both asks included (one always-on read's worst case) |
| `fail_task_on_error` | false | A rating step that fails doesn't fail the task |

The lines are on this step, never in the persona, so the user sim's prompt never has them and the run is the one
tested; a `satisfaction` section in the description is refused. The run step is as it is without a rating, and the
rating takes none of always-on's time budget.

The step sends the user sim one message, in the session the run used, so it remembers the run: always-on's, which its
record names (`metadata["always_on"][<step id>]["user_sim"]["context_id"]`), or `prompt_agent`'s conversation. The
message has any of the assistant's messages the user never read (in always-on, what was unread at the end; in
`prompt_agent`, the replies after the last read), the lines under "HOW YOU'D RATE THE JOB", and the ask: 5 is perfect,
3 fine, 1 "you'd have done it yourself", 2 and 4 in between, and something it was never told was done doesn't count as
done. The answer uses the usual format, with `score` and the reason in `message`; a malformed one gets one retry.

The rating goes to `metadata["user_sim_satisfaction"][<run step id>]`: `score` (None when it failed), `reason`,
`used`, `at`, `unread_shown`, `schema_errors`, `retried`, `user_sim_error`, `user_sim_turns`, `context_id` (the
session it asked in), `rated_by` (the rating step), and `skipped` with the reason when it wasn't asked: the run ended
on an error (always-on's error ends, a `prompt_agent` agent turn or user sim that failed, a run step that failed), or
the user sim's agent is no longer deployed. The whole ask and raw answer are in the object store at its `object_url`.
It never reaches the assistant, the graded conversation, or the verifications. A user sim rates more kindly than
people do, so treat it as a signal next to the graders, not the score.

The task is refused when it loads if the step has no run to rate (none, or several and no `run_step_id`), comes
before it or has a `depends_on` that doesn't name it, if the run's user isn't a user sim a `deploy_user_sim_agent`
step deploys, if two rating steps rate the same run, or if `satisfaction` isn't a list of non-empty lines.

## The rules that matter most

These made the user sim better in replays of a real multi-day task. The instructions drove more of the gain than how
each read is shown.

- **Write a specific "don't mention X unless asked" line for every tempting fact.** A general "never bring up what
  you know" loses to a fact that looks relevant. Under that rule alone, the user sim still added an unasked
  accessibility need in 6 of 10 answers. Specific lines fixed slips like it.
- **Keep "say you don't know, never guess".** It's in the fixed lines. Without it, the user sim invented facts: a
  made-up "no allergies" in 8 of 10 answers. With it, none.
- **Keep "done is only a short sign-off".** It's in the fixed lines too. Without it, the user sim answered a question
  and said done in the same message 9 of 10 times, so the assistant never saw the answer. With it, 1 of 10.
- **Write decisions with one answer.** "A house at about $300 a night: yes" let it say yes to $310 in one run and
  "what else is there?" in another. Write "$300 or less: yes. Over $300: ask what else there is."
- **Write nothing about the future.** No events that haven't happened yet, and no lines the user hasn't sent yet:
  the user sim uses them before they happen. What happens during the run reaches it as it happens (below).

The prompt starts with fixed lines:
- a real person, not the assistant;
- never do the assistant's work;
- can't open apps;
- what's new is for you;
- say you don't know;
- what done means;
- `used`.

They were part of what was tested. Keep the description to who the person is.

## What reaches the user's phone

Each read lists only what triggers sent to the user, never everything in their apps. Showing every inbox item pulled
the user sim away from its description and made it overshare. To have a trigger reach the user, add a `notify` action
addressed to `user_phone`, with the line their phone would show. Put the sender in `channel`, as the phone shows it
("Gmail · Carol Reyes (Mom)").

Shown as it happens, an event lets the user catch a mistake only it reveals, such as a confirmation email with the
wrong date. A list of events in the prompt can't do that.

For events that depend on what the assistant does, write one general trigger. Fill in the real details from the
call that set it off, with `${args.*}` and `${result.*}`:

```json
{"id": "table-confirmed", "when": {"type": "action", "tool": "tablefern_book_reservation", "repeat": true},
 "actions": [{"type": "notify", "to": "user_phone", "channel": "Tablefern",
              "text": "Your table for ${args.party_size} on ${args.date} at ${args.time} is confirmed"}]}
```

## What the user sim reads each time

Each read is one message, in this order:
- the time, and when they last wrote;
- YOUR NEW EMAILS, TEXTS AND NOTIFICATIONS, in full, then YOUR EARLIER EMAILS, TEXTS AND NOTIFICATIONS, first lines only;
- YOU SENT YOUR AI ASSISTANT: only their own lines (the opening request, `user_says` lines and the final check-in,
  never the assistant's alerts);
- everything new from the assistant at once;
- then the answer format.

Times show in the user's time zone. The user sim keeps one session for the whole run, so its earlier answers aren't
shown again.
