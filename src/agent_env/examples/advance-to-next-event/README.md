A prompt_agent post-turn hook that runs one conversation over simulated days and wakes the agent only for what
would really reach it. Unlike `next-env-event`, which wakes the agent with a nudge at every env event, it moves time
past world events silently.

`artifacts/advance-to-next-event/hook.py` is the hook. After each agent turn it waits for what the turn set off to
finish, then lets a firing agent trigger answer. Otherwise it moves the frozen clock, silently, to the next thing
that happens: a world event (a trigger's mark, a delay ending, an executor's hours opening), something due for the
agent on notifications/v1, or a message the person writes at a set time. World events change the env but wake no
one. The agent is woken only when one of these has come due:

- a check it scheduled with `schedule_check` (which a `configure_scheduled_checks` step gives its role),
- a notification a trigger sent it with a `notify` action, through a channel the task declares,
- a message from `person_messages`.

The hook takes nothing from notifications/v1: it returns `{"message": []}` and `prompt_agent` delivers what came due,
each line saying when and through what, so nothing poses as the person. A person's message is the hook's own reply,
after those lines:

```
[Tue Feb 2, 9:00 AM · Curbfinch alert] New for "rug 8x10": 1970s wool rug · $360
[Tue Feb 2, 6:00 PM · your scheduled check] Look for new 8x10 rugs under $300.
```

The conversation ends when nothing can wake the agent any more, or the next thing is past `end_time`. It also ends,
recorded as `triggers_still_firing`, if triggers are still running when `settle_timeout_seconds` runs out, rather
than treat unfinished work as nothing left to happen. Envs whose clock was never armed are left out.

Reference the hook from a `prompt_agent` step, after `sync_env_clock` (frozen: `virtual_seconds_per_real_second: 0`)
and `configure_scheduled_checks` (`"roles": ["default"]`). `time_zone` on `prompt_agent` sets the zone the delivered
lines show:

```json
{"id": "week", "type": "prompt_agent", "prompt": "Keep an eye on curbfinch this week for an 8x10 rug.",
 "max_conversation_turns": 20, "time_zone": "America/Los_Angeles",
 "post_turn_hooks": [{"file_artifact_id": "@local/agentenv-framework/advance-to-next-event/advance-to-next-event",
                      "params": {"end_time": "P5D", "time_zone": "America/Los_Angeles"}}]}
```

`tasks/week.json` runs it on `next-env-event`'s inbox: a school email that alerts the agent, a dentist email
that doesn't, and a message from the person midweek.

Its params, all optional:

- `end_time`: an RFC 3339 instant with an offset, or an ISO 8601 duration from the clock's t0 (`P5D`).
- `role`: the agent's AgentEnv-Role, whose checks and notifications it moves the clock to; `default` by default.
- `time_zone`: the IANA zone the person's messages show their times in; UTC by default.
- `person_messages`: `[{"at": RFC 3339 instant, "text": "..."}]`, what the person writes and when.
- `settle_timeout_seconds`: how long to wait for triggers to settle before ending the run, 300 by default.
- `max_jumps_per_turn`: silent world events it may pass between two wakes, 500 by default.

`max_conversation_turns` caps the wakes. The env needs a gateway with clock/v1 `advance` and notifications/v1. A hook
runs in the control-plane process with the trust of an `env_outcome_verifier` script.
