A prompt_agent post-turn hook that runs one conversation from env event to env event on simulated time.

`artifacts/next-env-event/hook.py` is the hook. After each agent turn it first waits for what the agent's turn
set off in the env to finish, then lets a firing agent trigger answer. Otherwise it reads `next_event_at` from
each env whose clock is armed: the earliest simulated time that env's triggers act on their own, whether a time
trigger's mark, a delayed action resuming, or an executor's hours opening. It advances every clock that is behind
it, waits for the triggers that came due to settle, and asks the agent triggers again, or else wakes the agent
with a nudge naming the time. It ends the conversation when nothing is pending or the next event is past
`end_time`.

`tasks/week.json` runs it over a week of mail in the `inbox` env, with the default agent. It needs Docker and
a model endpoint: `agent-env run next-env-event`.

Reference the hook from a `prompt_agent` step:

```json
{"id": "week", "type": "prompt_agent", "prompt": "Look after my inbox this week.", "max_conversation_turns": 12,
 "post_turn_hooks": [{"file_artifact_id": "@local/agentenv-framework/next-env-event/next-env-event",
                      "params": {"end_time": "P5D"}}]}
```

Its params, all optional:

- `end_time`: an RFC 3339 instant with an offset, or an ISO 8601 duration from the clock's t0 (`P5D`).
- `nudge`: the message that wakes the agent; `{virtual_time}` is its one placeholder.
- `settle_timeout_seconds`: how long to wait for triggers to settle before going on anyway, 300 by default.

It is made for a frozen clock (`virtual_seconds_per_real_second: 0`): on a running clock a trigger that recurs
faster than a turn keeps firing, and each settle waits out its timeout. The envs need a gateway image that offers
clock/v1 `advance` and triggers/v1 `next_event_at`; on an older one the hook logs a warning naming the env and
ends the conversation without jumping. A user agent with agent triggers needs a triggers/v1 `decide` that takes
`record_unfired`, since the hook asks its triggers twice a turn. A hook runs in the control-plane process with the
trust of an `env_outcome_verifier` script.
