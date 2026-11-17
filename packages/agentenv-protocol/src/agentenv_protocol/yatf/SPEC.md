# yatf/1: the agent trajectory format

Spec 1.0. A yatf/1 stream is an append-only sequence of small JSON events that records what one agent did in one A2A
context. Each agent builds it from its own native log with an adapter, and serves it beside that log through
`urn:agentenv:trajectory/v1`. The native log stays the agent's default, so a reader that names no format sees no
change.

Three terms are used throughout:

- The **native log** is the record the agent's native format id names (`claude-code-stream-json/1`,
  `pydantic-ai-messages/v1`). The agent serves it today, and judges read it.
- An **aux file** is a second record the harness writes, such as Claude Code's session JSONL or Codex's rollout. The
  adapter reads it to fill gaps in the native log. It is stored beside the native copy, is never served as a format,
  and is never read by judges.
- The **YATF copy** is the output: the stream this spec describes.

The machine-readable form of this spec is the strict JSON schema `yatf-1.schema.json`, which validates one event. The
worked examples in `examples/` are its canonical fixtures.

## Streams

- A stream belongs to one agent and one A2A context. It holds one **turn** per A2A task, numbered from 0 within the
  context, and each turn holds its events.
- Events are final once appended. A live reader sees the same events as the final read, in the same order.
- The order is (`turn`, `seq`), never timestamps. `seq` is dense from 0 within each turn.
- On the wire a stream is a JSON array. In storage it is JSONL, one event per line. JSON is read strictly: UTF-8,
  with no NaN or Infinity. A field name is valid Unicode, with no lone surrogate, as I-JSON (RFC 7493) requires. A
  string value may hold one, as a native log that cut an emoji in half can.
- Values are taken as written, never coerced, so whatever a reader parses it writes back exactly. An integer is a
  JSON integer, with no fraction or exponent: JSON Schema's `integer` also matches `5.0`, but a reader refuses it.
- The agent SDK writes each turn's **frame**: `turn.start`, the inbound `message`, `final_response` and `turn.end`.
  It also stamps every event's `seq`, `id`, `turn` and `observer`. The adapter writes everything between, from the
  native records, one event per native block in native order.

## Event envelope

Every event is one JSON object: these fields, plus the body of its type.

| Field | Required | Meaning |
| --- | --- | --- |
| `seq` | yes | Position from 0 in its turn |
| `id` | yes | `<turn>.<seq>`, stable |
| `type` | yes | A core type, or an extension type `<namespace>/<name>` |
| `turn` | yes | Index of the A2A task within its context |
| `actor` | yes | The declared actor the event belongs to |
| `observer` | yes | Who recorded it: `agent`, `gateway`, `browser` or `framework`; `agent` in an agent's stream |
| `time` | yes | A Time object; an unknown time is an explicit null, never omitted |
| `step` | content events | Step key: groups the events of one inference, dispatch or inbound message |
| `attempt` | no | Harness retry within the task; absent means 1 |
| `is_copied_context` | no | ATIF's flag, kept by name: replayed context that SFT may exclude |
| `derived_from` | recommended | The record the event came from: `{"index": n}` in the turn's native log, or `{"aux": name, "index": n}` in an aux log |
| `x` | no | Extension data keyed by namespace, kept untouched |

## Time

- `at`: when it happened per the native record, RFC 3339 with a zone, or null. It is never an empty string, never 0
  and never an estimate. The time a record arrived is not a native time.
- `start` and `end`: only when the native gives both ends. `duration_ms`: only when the native reports one.
- `raw`: the native time string, verbatim, whenever it differs from `at`.
- `virtual`: the env clock's time, one `{env_id, env_step_id, at}` per armed env, never derived from wall time.

## Event types

Spec 1.0 defines these types. The names `delegation`, `context`, `producer`, `activation` and `usage.checkpoint` are
reserved: a later 1.x defines each one with its first producer, and no 1.0 producer writes them.

| Type | Written by | Body |
| --- | --- | --- |
| `turn.start` | SDK | `spec_version`, `context_id`, `task_id`, `message_id`, `native_format`, `live`, `adapter` (`name`, `version`, `lossy`), `usage_input_includes_cache`, `actors` |
| `message` | SDK (inbound), adapter | `role` (`user`, `assistant`, `system`, `developer`), `content` (parts) |
| `final_response` | SDK | `content`: the A2A result's parts |
| `turn.end` | SDK | `state` (`completed`, `failed`, `canceled`), `tool_call_count`, `final_metrics` |
| `step` | adapter | `source` (`agent`, `user`, `system`), `llm_call_count` (0 dispatch, 1 call, more aggregated), `model_name`, `reasoning_effort`, `metrics`, `stop_reason`, `provider_ids`, `first_token_at`, `sampling` |
| `reasoning` | adapter | `visibility` (`raw`, `summary`, `omitted`, `redacted`), `text` (null when omitted), `opaque`: provider payloads kept byte for byte (`kind` `signature`, `encrypted` or `redacted_data`, `provider`, `value`) |
| `tool_call` | adapter | `tool_call_id`, `function_name`, `arguments` (any JSON value), `arguments_raw` (the native string when it was one), `kind` (`builtin`, `mcp`, `function`, `custom`), `server`, `executor` (`agent`, `env`, `provider`, `client`), `env_id` |
| `tool_result` | adapter | `source_call_id`, `status` (`ok`, `error`, `canceled`, `timeout`), `content` (parts), `exit_code` |
| `tools` | adapter | `mode` (`replace`, `add`, `remove`), `tools`: `name`, `kind`, `server`, `description`, `input_schema` |
| `actor` | adapter | `declared`: an actor that appears mid-turn, such as a sub-agent |
| `error` | adapter | `error_type` (`agent_error`, `infra_error`), `message`, `code`, `fatal` |
| `attempt` | adapter | `number`, `reason`: a harness retry; this event and every later one carry `attempt` = `number` |
| `note` | adapter | `text` |

- Content events (messages, reasoning, tool calls and an inference's errors) carry the key of the step they belong
  to, and the inbound message's key is `in-<turn>`. `tools`, `tool_result` and the frame's other events carry none.
- A `step` event closes the step whose key it carries. Its content events (messages, reasoning, tool calls) share the
  key and may come before or after it.
- A result is a separate event that answers any earlier call in the context, by `source_call_id`, and stays where
  it arrived. Folding into ATIF steps regroups results into their call's step.
- Strings named as vocabularies above may gain values in a minor version. A reader keeps a value it doesn't know.

## Parts

Content is a list of parts.

| Part `type` | Fields |
| --- | --- |
| `text` | `text` (inline up to 16 KiB), `origin`, `elision` |
| `ref` | `preview` (at most 2 KiB) and `source`: text over 16 KiB, by reference |
| `image`, `audio`, `video`, `document`, `file` | `source`, `purpose` (such as `screenshot`), `origin`: media, by reference only |
| `data` | `data` (any JSON value), `origin`: a structured A2A part |
| `elided` | `elision`, `media_type`: content that isn't there, and why |

- A **source** is `url`, `media_type`, `bytes` and `sha256`, plus `width` and `height`, `duration_ms`, `pages` or
  `name` where they apply. A `sha256:<hex>` URL resolves under the run's blob prefix. Media is never inline base64.
- An **elision** is `reason` (`truncated`, `redacted`, `digest_only`, `evicted`, `not_captured`, `not_requested`),
  the original `bytes` and its `sha256`. A secret the agent was given is scrubbed by the SDK before the event is
  appended, as a `redacted` elision with its length and no hash, since a hash of a short secret can be brute-forced.
- A user-side part carries an **origin**: `kind` (`framework`, `user_sim`, `trigger`, `notification`, `peer`,
  `human`) beside the part's own metadata, such as a notification's channel and id.

## Actors

- Actors are declared before any event names them, in `turn.start` or in an `actor` event.
- An actor is `id`, `kind` (`agent`, `subagent`, `user`, `user_sim`, `human`, `peer`, `trigger_executor`, `env`,
  `judge`, `framework`), and optionally `name`, `version`, `model_name`, `model_source` (`configured`, `reported`,
  `unknown`), `role` (the gateway role, which joins the actor to env events), `participant` (its run-level id),
  `a2a_agent_id`, `parent`, `spawned_by` and `stream`.
- A sub-agent lives in its parent's stream, under its own actor, with `parent` and `spawned_by` (the spawning
  `tool_call_id`). It gets a `stream` of its own only when the native keeps it in its own file.
- A step event may belong to any actor, so a summarizer's inference is attributed where it happened.

## Steps and usage

- Usage is inclusive, as in ATIF and OTel: `prompt_tokens` includes cache reads and writes, and `completion_tokens`
  includes reasoning. `cached_tokens`, `cache_write_tokens` and `reasoning_tokens` are subsets of those. `raw` keeps
  the native counters.
- `usage_input_includes_cache` in `turn.start` says whether the native input count already included the cache, so a
  reader can line YATF's numbers up with the native ones. Each adapter declares it as a constant, backed by a
  recorded real run.
- `cost_status` is `reported` (the harness's own figure), `priced` (with `cost_source` naming the price table and
  version) or `unpriced`. `cost_usd` is null exactly when unpriced. Agents serve what they report; pricing happens
  elsewhere, never in the YATF copy.
- An unknown count is null or absent, never 0.
- `turn.end.final_metrics` uses ATIF's FinalMetrics names: `total_prompt_tokens`, `total_completion_tokens`,
  `total_cached_tokens`, `total_cache_write_tokens`, `total_reasoning_tokens`, `total_cost_usd`, `cost_status` and
  `cost_source`, plus:
  - `basis`: `steps` when the totals sum the turn's step events, `reported` when they are what the harness reported;
  - `total_steps`: the number of `step` events;
  - `partial`: some usage is missing, such as an earlier attempt's after a retry;
  - `includes_subagents`;
  - `by_actor` and `by_model`: totals broken down.

## Extensions and unknown fields

- An **extension type** is `<namespace>/<name>`, lowercase, outside the `yatf` namespace. Readers keep it and may skip
  it. An adapter keeps a native record that no handler takes as `<namespace>/raw`, so converting back to native loses
  nothing.
- `x` carries extension data on any event, keyed by namespace, such as `x.claude_code` for native fields YATF has no
  field for.
- **Readers are tolerant.** A reader accepts any 1.x stream. It keeps fields, event types and part types it doesn't
  know, and writes them back out unchanged.
- **Producers are strict.** A producer writes only what the minor version it declares defines. The validator reports
  an unknown field as an error for a producer and as a warning for a reader. The JSON schema forbids unknown fields
  everywhere except in an `origin`, an `x` and an extension event's body.

## Versioning

- The major version is in the format id. A breaking change is `yatf/2`, served beside `yatf/1`.
- A minor version only adds optional fields, types and vocabulary values. A field added in a minor version names the
  version it arrived in, so the validator catches a producer that declares 1.0 but uses it.
- `turn.start` names the spec version, the native format and the adapter's name and version. The solver's actor names
  the agent's version.

## Validator rules

The validator reports findings tagged with these rules, each fatal or a warning. V1 to V12 and V14 apply to agent
streams. `a2a-agent validate` makes them fatal for an agent that advertises `yatf/1`.

| Rule | Checks |
| --- | --- |
| V1 frame | Each turn starts with `turn.start` and ends with `turn.end`. The inbound message follows `turn.start`, and at most one `final_response` comes just before `turn.end` |
| V2 positions | `seq` is dense from 0 in each turn and equals the cursor position; ids are `<turn>.<seq>` and unique |
| V3 actors | Every actor is declared before use, and every sub-agent's parent resolves |
| V4 calls | `tool_call_id` is unique in the context, and every `source_call_id` names an earlier call |
| V5 time | `at` is null or RFC 3339 with a zone; `start` is before `end`; `raw` is present when it differs |
| V6 usage | Subsets fit inside totals; cost is null exactly when unpriced; with basis `steps`, totals equal the step sums; `total_steps` counts the step events |
| V7 references | Every source has a URL, media type, size and sha256; `sha256:` URLs match their hash; a media part's kind agrees with its media type |
| V8 no inline payloads | No `data:` URIs, and no base64 run over 1 KiB outside opaque reasoning; an event is at most 256 KiB |
| V9 live equals final | The concatenated live pages equal the final read, event for event |
| V10 format labels | The card lists `yatf/1`, and every read names the format it served |
| V11 unknown fields | An error for a producer, a warning for a reader |
| V12 elision | Elided content has a reason and its original size, and a hash unless it was never captured or is a redacted secret |
| V14 round trip | Native to YATF to native equals the original up to the adapter's normal form, and YATF to native to YATF is the identity, on recorded fixtures |

## Adapters

- An adapter reads only what the run stores: the turn's context as the SDK records it in `turn.start`, the turn's
  native records, and its aux records. The agent's own code feeds aux records in, and the SDK stores them in the
  order fed. Replaying a stored turn therefore gives the events the agent served live.
- One adapter instance serves an A2A context across its turns. Renormalizing replays a context from turn 0.
- An adapter converts back with `from_yatf`. It gives each turn's native log, for the round trip, and the files the
  CLI resumes from, or none.
- `lossy` names only what the native never records. Everything the native does record has a YATF field, sits under
  `x.<namespace>`, or is kept as a `<namespace>/raw` event.

## Examples

`examples/` holds the design's three worked examples. Each one is a single turn's events.

- `coding.json`: Claude Code fixes a failing test. Steps come from stream-json, with times null where stream-json has
  none. A large read result is a `ref` part, and the turn totals are what the harness reported.
- `multi_agent.json`: a pydantic-ai solver's second turn. A simulated user's reply and a delivered notification arrive
  as one inbound message, each part with its origin.
- `computer_use.json`: Claude Code drives a desktop. Screenshots are image parts by reference, and one reasoning
  block's text was withheld (`visibility` `omitted`).
