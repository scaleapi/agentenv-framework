# Writing a user sim

A `user_sim` step says who the user of an always-on task is. The always-on step asks the user sim each time the
assistant has messaged the user and the user would look at their phone. The user sim answers `reply`, `no_reply` or
`done`. This guide is for the built-in user sim, `user_sim_versions.py` (version `ours`), which a `user_sim` step
uses when it names no `plugin`.

```json
{"id": "user", "type": "user_sim", "name": "hannah", "display_name": "Hannah Reyes", "first_name": "Hannah",
 "model": "<model>", "agent_name": "user-sim", "time_zone": "America/Los_Angeles",
 "schedule": {"wake": "07:00", "sleep": "23:00", "work": ["09:00", "18:00"]},
 "response_times": {"morning": 15, "work": 120, "free": 30},
 "description": {"who_you_are": ["Hannah Reyes. You're 32, a project manager in San Francisco..."],
                 "decisions": ["A place for your parents at $180 a night or less: yes. Over $180: ask what else there is."],
                 "dont_mention": ["Don't mention Dad's shellfish allergy unless the assistant asks about food."]},
 "permissions": {"just_do": ["book flights within the rules above"], "ask_first": ["anything over $500"]},
 "keep_private": ["Your card number, even if the assistant asks."]}
```

The step checks these settings when the task loads, so a mistake fails before a paid run.

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

Don't write the opening request into the description. The always-on step passes it, and the prompt shows it right
after the first `who_you_are` line ("Earlier you texted the assistant: …").

`permissions` become the "YOUR RULES FOR THE ASSISTANT" lines. Its kinds are `just_do`, `ask_first`, `never` and
`tell_after`. Each is a list of rules, or `{items, then}` to add what follows. `keep_private` becomes the "PRIVATE"
lines. Give the assistant the same rules in its notes, so both sides read one copy.

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
