# Harbor's conversion beside the adapter's

`subagent.harbor.json` is Harbor 0.26.0's own Claude Code converter (`ClaudeCode.convert_trajectory`) run on the
`subagent` recording's session files after the run, with its stream-json as `claude-code.txt`. `subagent.expected.json`
one folder up is what `ClaudeCodeAtif` built from the same run, live. Each difference, and why:

- **Sub-agents.** Harbor puts the sub-agent's steps in the main list, ordered by time and marked `is_sidechain` and
  `agent_id`, so `total_steps` is 6. The adapter embeds them as an ATIF sub-agent trajectory, named by the Agent call
  that spawned it and referenced from that call's result, so the main list has 3 steps. Its last answer
  is in both, though stdout never carried it: both read the sub-agent's transcript.
- **Step usage.** Equal on every step: both take each message's final usage from its transcript record. Harbor copies
  the raw usage object into `metrics.extra`; the adapter writes ATIF's fields and puts cache writes in the profile's
  `cache_write_tokens`.
- **Cost.** Harbor writes a LiteLLM estimate on every step (`cost_source: litellm_estimate`) and takes the total from
  the stream-json `result`. The adapter writes no step cost, and the task's cost is the one the agent reports for the
  turn, the CLI's own tally (`cost_source: harness`). The totals agree here: one turn, so the CLI's tally and its
  `result` are the same figure.
- **Agent block.** Harbor names the CLI (`claude-code`, its version) and the model as the transcript has it. The
  adapter's comes from the agent's card and config (`Claude Code`, the agent's version, the model as configured), and
  carries `tool_definitions`, the 21 tools of the request the CLI sent, which Harbor leaves out. Steps name the model
  as the transcript has it in both.
- **The user step.** Harbor takes the prompt and its time from the transcript; the adapter's comes from the A2A
  message and the time the task started, with the task's ids and turn index in the profile.
- **Tool results.** Harbor writes a list-shaped result as its JSON text; the adapter writes its text parts as text and
  its images as media files.
- **Ids.** Harbor's `session_id` is the CLI's session; the adapter's is the A2A context, as for every agent.
- **Thinking** (not in this recording; see the `thinking` one). Harbor drops a thinking block's signature, and writes
  nothing for thinking whose text the API withheld. The adapter keeps the signature and records the block as
  `omitted`.
