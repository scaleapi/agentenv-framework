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
- A served document leaves out the secrets its agent's SDK knows: each is written `[redacted:<length>]` wherever it
  occurs, in any string, the aux log and turn context served with it included. The native log keeps them.
- A context's record may span several documents: after an agent instance restarts or drops the context from memory,
  after a snapshot restore, and when long runs are sealed into segments. They share `session_id`, the first user step of each carries the continuing turn
  index, and each earlier document names the next in `continued_trajectory_ref`. The worker that stores the earlier
  document sets that link, since the agent that wrote it may be gone.
- `session_id` is the A2A context id, and a served document's `trajectory_id` names that one document of the
  context's record. A snapshot restored into a new context id starts a new session, since one snapshot can be loaded
  many times; its turn index goes on from the snapshot's.
- A sub-agent is an embedded trajectory in `subagent_trajectories`, with its own steps, referenced from the spawning
  call's result by `subagent_trajectory_ref`. It belongs to the task of the call that spawned it. A sub-agent that
  wrote no step is left out, and so is the reference to it. A spawning call with no result when its task ends gets
  one, with no content and `status: canceled`, so its sub-agent stays referenced.
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
  last step, which closes only when the task ends, so it is never served before the state is known, and so does the
  task's error, unless the adapter wrote one there.
- A2A DataParts: what the conversation says is text, and what describes the run is profile. An inbound DataPart is a
  text part holding its compact JSON, and so is a final response's `structured_output`, the task's answer. The final
  response's other DataParts, a harness's diagnostics, go in its last step's `data`. The SDK's own usage and error
  part is left out: `final_metrics` and the last step's `state` and `error` hold it.
- An agent message whose parts are all text is one string, the parts joined by a blank line; Harbor's tools read only
  strings there. Text output always goes in whole: there is no reference part and no cap.
- `reasoning_content` joins a step's reasoning blocks' texts with a blank line. The profile's `reasoning` lists every
  block, in native order, with its share of that text, so a reader that knows the profile can rebuild exact blocks.
- Timestamps are UTC, written `YYYY-MM-DDTHH:MM:SS(.fff|.ffffff)Z`. Other precisions are normalized: ATIF's Step
  validator on Python 3.10 reads no other. A document keeps one clock: the steps the SDK writes, each task's user step
  and final response, are on the clock the agent's programs read, the universe's when the agent follows the env's
  clock.
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
  ObservationResult for a result), keyed by its path. A PNG, JPEG, GIF or WebP image's `width` and `height` are read
  from its header when the producer gives none.
- A reader takes a part path only as `media/<sha256>.<ext>` in the document's own folder, never another path or a URL,
  and checks each file's bytes against its name (`media_file`, `check_media`).

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
| Step | `data` | On a task's last step: its final response's diagnostic DataParts, in order, each `data` and its `metadata` | the explorer |
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

## Validator

`validate(document)` returns findings, each with a rule id below, a severity and where in the document it is. It holds a
producer's output to profile 1.0. `reader=True` checks someone else's document for reading instead: what a producer
must not write only warns, and the document fails only where agentenv couldn't read it. `media` is the folder the
document's media paths are relative to, the one holding `media/`, or the files by path; without it only the paths' form
is checked. `validate_live(pages, final)` checks what a live reader was served, page by page, with the final read. A sub-agent's task is its spawning call's, found through the references, so an embedded sub-agent that no served step references isn't held to `live-unanswered`. On
the command line: `python -m agentenv_protocol.atif validate FILE [--media DIR] [--reader]`, which prints each finding
and exits 1 when one is an error.

| Rule | Holds | Producer | Reader |
| --- | --- | --- | --- |
| `json` | The document is a JSON object: UTF-8, with no NaN or infinity | error | error |
| `atif` | ATIF's own models, vendored from Harbor, accept it | error | error |
| `profile` | Each `extra.agentenv` parses with its profile model | error | error |
| `profile-key` | Each `extra.agentenv` holds only the fields profile 1.0 defines | error | warning |
| `profile-version` | The root's `extra.agentenv.version`, when given, is `1.0` | error | warning |
| `vocabulary` | A vocabulary field holds one of its values | error | warning |
| `outside-key` | No `extra` holds a key outside `agentenv` but `ToolCall.extra.raw_arguments` | error | warning |
| `call-id` | Each `tool_call_id` is unique across the document and its sub-agents | error | warning |
| `reference` | Each `subagent_trajectory_ref` names an embedded sub-agent by `trajectory_id` | error | warning |
| `totals` | A `final_metrics` total equals the sum over the model steps of the root and its embedded sub-agents, when every one has the count | error | warning |
| `total-steps` | `final_metrics.total_steps` counts the root's steps, or the root's `notes` explain the difference | error | warning |
| `usage` | A step's cache reads and writes fit within its `prompt_tokens` | error | warning |
| `media-path` | A media path is `media/<sha256>.<ext>` | error | warning |
| `media-bytes` | Given the media, each file the document names exists, with the sha256 its name and profile give and the profile's size | error | error |
| `timestamp` | A timestamp is UTC in the builder's form: `YYYY-MM-DDTHH:MM:SSZ`, with no, 3 or 6 fractional digits | error | not checked |
| `harbor-save` | Harbor's `to_json_dict` gives the document back unchanged | error | warning |
| `text-message` | A step message whose parts are all text is one string | error | warning |
| `live-prefix` | Every live page is a prefix of the next page and of the final read | error | not checked |
| `live-unanswered` | No live page serves a step with an unanswered call before its task ends | error | not checked |

The adapter kit, `check_adapter(adapter, recorded, expected=, from_atif=)`, is for an agent's CI. It runs an adapter
over a context's recorded tasks as the SDK runs it live, applies the live rule after every record and the validator to
the final document, and adds its own rules. `kit-from-atif` only warns when the adapter has no `from_atif`.

| Rule | Holds | Severity |
| --- | --- | --- |
| `kit-run` | The adapter takes every recorded record without raising | error |
| `kit-expected` | The document equals the checked-in expected document | error |
| `kit-unmatched` | Every native record has a handler; one that none takes is only counted | warning |
| `kit-from-atif` | `from_atif` turns a document without the profile into a native log | error |

## Fixtures

`fixtures/` holds the profile's canonical examples, built through the adapter framework from a toy format's records:
`coding`, `computer_use`, `late_result`, `multi_agent`, `odd_parts`, `subagent` and `two_turns`, with their media in
`fixtures/media/`, and `coding.stripped.json`, the coding case without its profile.
