A prompt_agent post-turn hook that runs one conversation over simulated days and wakes the agent only for what
would really reach it.

`artifacts/agent-wakeups/hook.py` is the hook. After each agent turn it waits for what the turn set off to finish,
then lets a firing agent trigger answer. Otherwise it jumps the clock, silently, to the next thing that happens: a
world event (a trigger's mark, a delay ending, an executor's hours opening), a check the agent scheduled for itself,
or a message the person writes at a set time. World events change the env but wake no one. The agent is woken only
when one of these has come due:

- a check it scheduled with `schedule_check` (turned on by a `configure_wakeups` step),
- a notification a trigger sent it with a `notify_agent` action, through a channel the task declares,
- a message from `person_messages`.

Everything due goes into one message, each line saying when and through what, so nothing poses as the person:

```
[Tue Feb 2, 9:00 AM · Curbfinch alert] New for "rug 8x10": 1970s wool rug · $360
[Tue Feb 2, 6:00 PM · your scheduled check] Look for new 8x10 rugs under $300.
```

The conversation ends when nothing can wake the agent any more, or the next thing is past `end_time`.

Reference the hook from a `prompt_agent` step, after `sync_env_clock` (frozen: `virtual_seconds_per_real_second: 0`)
and `configure_wakeups`:

```json
{"id": "week", "type": "prompt_agent", "prompt": "Keep an eye on curbfinch this week for an 8x10 rug.",
 "max_conversation_turns": 20,
 "post_turn_hooks": [{"file_artifact_id": "@local/agentenv-framework/agent-wakeups/agent-wakeups",
                      "params": {"end_time": "P5D", "time_zone": "America/Los_Angeles"}}]}
```

Its params, all optional:

- `end_time`: an RFC 3339 instant with an offset, or an ISO 8601 duration from the clock's t0 (`P5D`).
- `time_zone`: the IANA zone the wake lines show times in; UTC by default.
- `person_messages`: `[{"at": RFC 3339 instant, "text": "..."}]`, what the person writes and when.
- `settle_timeout_seconds`: how long to wait for triggers to settle, 300 by default.
- `max_jumps_per_turn`: silent world events it may pass between two wakes, 500 by default.

`max_conversation_turns` caps the wakes. The env needs a gateway with clock/v1 `advance` and wakeups/v1. A hook runs
in the control-plane process with the trust of an `env_outcome_verifier` script.
