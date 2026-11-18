# The agentenv profile of ATIF

Profile 1.0. agentenv records what an agent did as an ATIF v1.8 document: Harbor's
[Agent Trajectory Interchange Format](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md),
whose models are vendored in `models/`. An agent serves it as format `atif/1`, beside its native log. What ATIF has no
typed field for, agentenv adds under `extra.agentenv` on the objects that have an `extra`. This file is the profile's spec.

## The rule

- Typed fields mean exactly what ATIF says they mean. A document with `extra.agentenv` removed, or with any part of it
  missing, is still correct ATIF and means the same.
- Every profile field is optional, for readers and for producers. A producer that can't fill one leaves it out.
- No typed field holds a placeholder or a value whose meaning needs the profile.
- Readers keep profile keys they don't know and write them back unchanged. Producers write only fields this version
  defines; the validator holds them to it.
- An adapter writes nothing in an `extra` outside `extra.agentenv`, except the one key this file names:
  `ToolCall.extra.raw_arguments`.

The native log is stored beside every ATIF copy, so a copy may leave things out. Nothing needs the copy to convert back
to native exactly.

## Documents

- One document per agent per A2A context. Each A2A task starts at a `source: user` step that holds the inbound
  message, with the task's ids in that step's profile. Steps number from 1 across the whole document.
- A context's record may span several documents: after an agent instance restarts, after a snapshot restore, and when
  long runs are sealed into segments. They share `session_id`, the first user step of each carries the continuing turn
  index, and each earlier document names the next in `continued_trajectory_ref`. The worker that stores the earlier
  document sets that link, since the agent that wrote it may be gone.
- A sub-agent is an embedded trajectory in `subagent_trajectories`, with its own steps, referenced from the spawning
  call's result by `subagent_trajectory_ref`. It belongs to the task of the call that spawned it. A sub-agent that
  wrote no step is left out, and so is the reference to it.
- `schema_version` is the oldest ATIF version that holds the document's fields: `ATIF-v1.8` for profile 1.0.
- Format id: `atif/1`. The profile's own version is the root's `extra.agentenv.version`.

## Steps

- A step is one model response, keyed by its native id (a message id), or a step with no model call
  (`llm_call_count` 0, no metrics or reasoning). A step closes when the next step of its trajectory starts, or when its
  task ends.
- A result always sits in the step that made its call, as ATIF requires. One that arrives after that step closed records
  when, in `arrived`.
- A live reader is served closed steps only, each trajectory up to its first step still waiting on a result or
  referencing a sub-agent the reader isn't served a step of yet. Steps are ordered, so such a step holds back the steps
  after it until the result or the sub-agent's step comes, or the task ends. Everything a reader is served is a prefix
  of every later read, and every reference it is served resolves.
- The final response is written once. When the task's last agent step already holds the same text, nothing is added;
  otherwise one step is appended, `source: agent` and `llm_call_count` 0. The task's terminal state goes on the task's
  last step, which closes only when the task ends, so it is never served before the state is known.
- An agent message whose parts are all text is one string, the parts joined by a blank line; Harbor's tools read only
  strings there. Text output always goes in whole: there is no reference part and no cap.
- `reasoning_content` joins a step's reasoning blocks' texts with a blank line. The profile's `reasoning` lists every
  block, in native order, with its share of that text, so a reader that knows the profile can rebuild exact blocks.
- Timestamps are UTC, written `YYYY-MM-DDTHH:MM:SS(.fff|.ffffff)Z`. Other precisions are normalized: ATIF's Step
  validator on Python 3.10 reads no other.
- Usage follows ATIF: `prompt_tokens` includes cache reads and writes, and `cached_tokens` is the cache reads. Each
  adapter's `input_includes_cache` constant says whether its native input count already includes the cache.
  Cache writes go in `Metrics.extra.agentenv.cache_write_tokens`. An adapter may give a step's counts over several
  records: each count is the latest given, and `prompt_tokens` is written once the input count is known.
- `final_metrics` totals a count only when every model step, every agent step but those with `llm_call_count` 0, has
  it; otherwise the total is left out, which ATIF readers read as unknown.
- A tool call whose input isn't a JSON object is wrapped as Harbor wraps it: a string as `{"input": ...}`, a missing
  input as `{}`, any other value as `{"value": ...}`. The native input is kept in `ToolCall.extra.raw_arguments`, only
  when wrapping changed it. Harbor's ATIF-to-Codex resume reads that key back.
- A native record no adapter method takes is counted by its type in the root's `unmatched`; it stays in the native copy.

## Cost

ATIF defines a step's `cost_usd` as the cost "based on current provider pricing for this step", and says the cost fields
"store the calculated cost at the time of execution".

- A cost the harness calculated from its own price table, or the endpoint's billed figure (the provider's, or a
  proxy's such as LiteLLM's `x-litellm-response-cost`), goes in `Metrics.cost_usd` or `FinalMetrics.total_cost_usd`.
  Its `cost_source` in the profile says which: `harness` or `provider`. These are the framework's reported costs.
- A cost agentenv computes itself, from its own price table after the run, never goes in a typed field. The framework
  records that one with the run's usage instead, as a `priced` cost whose `cost_source` names the price table's
  version; `harness` and `provider` mean the same there as here.
- `total_cost_usd` is the sum of the tasks' reported costs when every task has one, else of the steps' costs when
  every model step has one. When nothing reports a cost, the typed fields stay empty, which ATIF readers read as
  unknown.

## Errors

One vocabulary, in three places:

- A tool result's `status` and `exit_code` go on its ObservationResult's profile. No status means `ok`.
- A step-level error, such as a refused or failed model call or a harness error, goes on its Step's profile as `error`.
- A task's terminal state goes on its last step's profile as `state`.

## Media

- Media are files beside the trajectory, at `media/<sha256>.<ext>`, relative to the trajectory file.
- An image or audio part of a type ATIF has names that path.
- Media ATIF has no part type for is stored the same way, with a plain note in the typed content, such as
  `[video: media/<sha256>.mp4, 4.4 MB, 31 s]`. That covers video, documents, images other than JPEG, PNG, GIF and WebP
  (such as HEIC or SVG), audio outside ATIF's eight types, and other files.
- Each file's details go in the `media` map of the profile of the object that holds the part (Step for a message,
  ObservationResult for a result), keyed by its path.

## Fields

| Object | Field | Meaning | Read by |
| --- | --- | --- | --- |
| Trajectory | `version` | The profile version: `1.0` | every reader |
| Trajectory | `native_format` | The native format id the copy was built from | the worker that stores copies; renormalizing |
| Trajectory | `adapter` | The adapter's `name` and `version` | the worker that stores copies; renormalizing |
| Trajectory | `unmatched` | Native records no method took, counted by type | the adapter kit |
| Step | `context_id`, `task_id`, `message_id`, `turn` | On a task's user step: its A2A ids and turn index | the SDK, the worker, the hub's turns view |
| Step | `state` | On a task's last step: its terminal state | the explorer, the hub's turns view |
| Step | `attempt` | The harness attempt the step belongs to; absent means 1 | the explorer |
| Step | `error` | A step-level error: `kind`, `message`, `code` | the explorer |
| Step | `reasoning` | Each reasoning block: `chars`, `visibility`, `signature`, `data` | resuming a session; RL replay |
| Step | `media` | Details of the media the message names, by path | media transfer; the explorer |
| ObservationResult | `status` | The result's status; absent means `ok` | the explorer, judges' turn events |
| ObservationResult | `exit_code` | The command's exit code | the explorer, judges' turn events |
| ObservationResult | `arrived` | A late result's arrival: `at`, and the `step_id` open when it came | live readers, the explorer |
| ObservationResult | `media` | Details of the media the result names, by path | media transfer; the explorer |
| Metrics | `cost_source` | Where `cost_usd` came from | the run's usage and cost result |
| Metrics | `cache_write_tokens` | Cache writes, beside ATIF's `cached_tokens` | the run's usage and cost result |
| FinalMetrics | `cost_source` | Where `total_cost_usd` came from | the run's usage and cost result |
| FinalMetrics | `total_cache_write_tokens` | Cache writes summed over the steps | the run's usage and cost result |

A media entry (MediaInfo) holds `sha256`, `bytes`, and, where they apply, `width`, `height`, `duration_ms`, `pages`
and `name`, plus `media_type` for media ATIF has no part type for. A reasoning block (ReasoningBlock) holds `chars`,
its share of `reasoning_content` (0 when the provider withheld the text), its `visibility`, the provider's `signature`,
and a redacted block's ciphertext as `data`, byte for byte. A step error (StepError) holds `kind`, `message` and `code`.
An arrival (Arrival) holds `at` and `step_id`. The Agent, ToolCall and SubagentTrajectoryRef profiles define no fields
in 1.0.

## Vocabularies

A reader keeps a value it doesn't know.

| Field | Values |
| --- | --- |
| `StepProfile.state` | `completed`, `failed`, `canceled`, `rejected` (A2A's terminal task states) |
| `ObservationResultProfile.status` | `ok`, `error`, `canceled`, `timeout` |
| `StepError.kind` | `agent_error`, `infra_error` |
| `ReasoningBlock.visibility` | `raw`, `summary`, `omitted`, `redacted` |
| `MetricsProfile.cost_source`, `FinalMetricsProfile.cost_source` | `harness`, `provider` |

## Versions

A minor version only adds optional fields and vocabulary values, each with the reader or producer that needs it. A
field added in a later minor names the version it arrived in. Proposed for a later minor, with their first producer:
`input_includes_cache` on the root and `cost_status` (`reported`, `priced` or `unpriced`), with the usage
cross-check that reads them; per-part origins, when a reader needs them; and an always-on harness's fields.

## Fixtures

`fixtures/` holds the profile's canonical examples, built through the adapter framework from a toy format's records:
`coding`, `computer_use`, `late_result`, `multi_agent`, `odd_parts`, `subagent` and `two_turns`, with their media in
`fixtures/media/`, and `coding.stripped.json`, the coding case without its profile.
