# FD-4601 trial: a week in one chat with Nora

One real run of the `advance-to-next-event` hook with a live user sim. It uses the week inbox from `next-env-event`,
the solver `psi-sh-neutral-openai-agents-sdk`, and Nora played by `psi-sh-usersim-claude-code`.

This folder is for the trial only. It names PSI agents, so drop it from the branch before the PR merges.

## What happens in the week (simulated time, clock starts Mon Mar 2, 08:00 UTC)

- **Mon 19:00**: Nora's scripted line, *"Heads up: I can't do school pick-up before 3:30 on Thursday."*
- **Tue 09:00**: the school emails to say pick-up moves to 15:00 on Thursday, please confirm, and a Mail alert wakes
  the agent. This conflicts with her line, and only she can decide what to do.
- **Wed 18:00**: Nora's scripted line, *"Anything from the dentist yet?"*
- **Thu 14:00**: the dentist emails to say the check-up moves to Friday 10:00. This email sends no alert.
- **Run ends:** when she says she's all sorted, or 5 days in (`end_time: P5D`), whichever comes first.

Her description says when she replies: questions and decisions only she can make get an answer; plain updates don't.
The agent's prompt tells it to write only when there's something she needs to know or decide.

## Command

Run from this branch's checkout, with Docker running and your PSI self-host config in effect (the store that has both
agents, and a model endpoint). Run the dry run first; it's free.

```bash
uv sync --extra dev && . .venv/bin/activate
agent-env a2a-agent get psi-sh-usersim-claude-code     # its card should list output_format and system_prompt
agent-env run trial/fd-4601-nora-week --dry-run
agent-env run trial/fd-4601-nora-week --model "$SOLVER_MODEL" --user-model "$NORA_MODEL"
```

`$SOLVER_MODEL` is the GPT model for the solver and `$NORA_MODEL` is Nora's. Drop either flag to use the agent's own
default. Add `--keep` to leave the sandboxes up for inspection.

## Checklist

Read the run in the explorer (`agent-env up`), or with `agent-env task get-instance --id <instance id>`. The hook's
records are under `metadata.post_turn_hooks.week`. Each record has a `turn` and either `person` (her outcome),
`woke_at` / `for` (a clock jump) or `told_person_at` (a scripted line put into her history).

1. **Setup.** The log says `Configured user-sim 'human_agent' with structured output_format`, and nothing says
   `skipping system_prompt`. If either is missing, the sim can't answer with outcomes, and every turn ends as
   `user_malformed`.
2. **No errors from her.** No record has `person: user_malformed`. If one does, the warning in the log shows her raw
   output.
3. **She stays quiet on updates.** Messages that only report ("checked the inbox, nothing new") get `no_reply`.
4. **She answers what needs her.** When the agent raises the 15:00 pick-up, she replies (`person: reply`), and her
   answer fits her Monday line: she can't make 15:00 and wants it changed, or asks the agent to sort it out.
5. **The clock waits for her.** A `woke_at` record only ever follows a `no_reply` or `not_asked` in the same turn,
   never a `reply`.
6. **Scripted lines arrive word for word.** The agent receives `[Mon Mar 2, 7:00 PM] Heads up: I can't do school
   pick-up before 3:30 on Thursday.` and `[Wed Mar 4, 6:00 PM] Anything from the dentist yet?`. Each one has a
   `told_person_at` record in the same turn.
7. **The agent never sees what only she is told.** No message to the agent contains "You just sent your assistant".
8. **Empty agent turns.** If the agent writes nothing on a wake-up, that turn shows `person: not_asked`, and the clock
   moves on.
9. **Ending.** Best case, she ends it with `done` after the pick-up and the dentist are settled. Otherwise, note which
   end it hit: `end_time`, `nothing_can_wake_the_agent` or `max_conversation_turns`.
10. **Cost.** Note the spend for both models. The plan was about $3.

Report back for each item: pass or fail, the turn numbers, and any message that looks off.
