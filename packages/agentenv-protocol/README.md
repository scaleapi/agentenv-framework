# agentenv-framework-protocol

Open data-plane protocol and server SDK for agent environments. Install it with
`uv add agentenv-framework-protocol` or `pip install agentenv-framework-protocol`; the import
package is `agentenv_protocol`.

An environment author writes a class with decorated methods and serves it:

```python
import base64
import json
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from urllib.request import urlopen

from pydantic import Field
from agentenv_protocol import (
    AgentEnvEnvironment, DataPart, FilePart,
    environment_card, reset_data, add_data, get_data, tool,
)


@environment_card(name="slack")
class SlackEnv(AgentEnvEnvironment):

    def __init__(self):
        self.channels, self.messages = {}, []

    @reset_data
    async def _reset(self):
        self.channels.clear(); self.messages.clear()

    @add_data
    async def _add(self, parts):
        # Seeds arrive as an inline DataPart (live deploy) OR a FilePart whose
        # file carries inline bytes, a file:// URI (a staged file or an exported
        # bundle) or an https:// URL (a signed URL, when agent-env can't stage the
        # file). Handle all four — dropping the FilePart branch makes bundles load empty.
        for p in parts:
            if isinstance(p, DataPart):
                payload = p.data
            elif isinstance(p, FilePart):
                f = p.file
                if getattr(f, "bytes", None) is not None:
                    payload = json.loads(base64.b64decode(f.bytes))
                elif urlparse(f.uri).scheme in ("http", "https"):
                    with urlopen(f.uri) as response:
                        payload = json.loads(response.read())
                else:
                    payload = json.loads(Path(urlparse(f.uri).path).read_bytes())
            else:
                continue  # TextPart / unknown — nothing to load
            self.messages.extend(payload.get("messages", []))

    @get_data
    async def _state(self):
        return [DataPart(data={"channels": list(self.channels.values()), "messages": self.messages})]

    @tool(name="{environment_name}_send_message")
    def send_message(
        self,
        channel: Annotated[str, Field(description="Channel to post to.")],
        text: Annotated[str, Field(description="Message text.")],
    ) -> str:
        """Send a message to a channel."""
        self.messages.append({"channel": channel, "text": text})
        return "ok"


if __name__ == "__main__":
    SlackEnv().serve()
```

`@tool` methods are registered as real MCP tools on the FastMCP app at mount and advertised
under the card's `capabilities.tools` (name, description, signature-derived `inputSchema` —
`Annotated[..., Field(description=...)]` param descriptions included). `{environment_name}` in a
tool name is resolved to the card's name at mount, so a shared mixin or base class can declare
environment-prefixed tools without knowing the name at class-definition time; any other unresolved
`{...}` token raises. Duplicate tool names raise at construction.

`AgentEnvStarletteApplication` mounts the same handler onto a Starlette/FastAPI app instead of
FastMCP; since those apps have no MCP tool registry, constructing one with `@tool` methods
raises.

## Serving

`serve()` builds the FastMCP app via `create_fastmcp_app()`, which encodes the agent-env deploy
contract once — the name resolution order (`ENVIRONMENT_NAME`, which agent-env sets to the env's
registered name, then `@environment_card`'s name, then the class name; `SERVICE_NAME` is no
longer consulted), `MCP_HOST`/`MCP_PORT`
binding (default 18765), DNS-rebinding protection off (gateways reach servers by compose
hostname, which mcp's localhost-only default allowlist rejects), and the AgentEnv mount — then
runs `streamable-http`. Pre-declared card content is more `@environment_card(...)` kwargs — any
`EnvironmentCard` field (keys are validated at decoration time); an undecorated class defaults
its card name to the class name. To mutate
the app before serving (extra imperative tools, custom routes), call `create_app()` first — it
returns the app un-served.

An environment that already owns its FastMCP app keeps full control: construct and configure
`self.mcp` yourself, then `mount(self.mcp)` — `serve()` runs it as-is (mounting first if you
haven't) and never alters a caller-built app's settings.

```python
@environment_card(name="legacy")
class LegacyEnv(AgentEnvEnvironment):

    def __init__(self):
        self.mcp = FastMCP("legacy")   # yours: settings, guards, extra routes
        self.mount(self.mcp)


LegacyEnv().serve()  # or run your app your own way; mount() alone is enough
```

The composition style — no base class, just
`AgentEnvFastMCPApplication(environment_card=card, handler=handler).add_routes_to_app(app)` —
remains fully supported; the base class is sugar over it.

A FastMCP-backed card declares its MCP endpoint in `additionalInterfaces` when it is mounted:
`{"url": <path>, "transport": "mcp"}`, where the path is the app's `streamable_http_path` (`/mcp`,
`MCP_PATH`, unless configured); a FastMCP-shaped app that does not expose that setting declares no
entry. That is the streamable-HTTP endpoint, which `serve()` runs by default and agent-env deploys
against; an app served over another transport, such as SSE, must declare its own entry. A card
that already declares an interface with the `mcp` transport (`MCP_TRANSPORT`) keeps it; interfaces
with any other transport are kept beside the SDK's entry. Card URLs are paths, relative to the
address the card was fetched from. On the client side, `client.mcp_path(card)` returns the
declared path, or `/mcp` for a card without one.

Extensions are invoked from what the card advertises. `client.find_extension_method(card, uri,
method)` returns one advertised method, whose `endpoint` is the method's own or else the
extension's, and `client.invoke_extension(base_url, card, uri, params, method=...)` calls it with
its HTTP verb. Without `method`, `invoke_extension` calls the first method listed; an extension
with several methods, such as a gateway's `urn:agentenv:clock/v1`, should always be called by name.

### Uploading an export through a grant

An export too large to return from `data/get` can go straight to the caller's object store. A
`@get_data` handler that takes a `write_namespace` parameter is advertised under
`urn:agentenv:data-objects/v1` (`DATA_OBJECTS_EXTENSION_URI`), and a caller may then send a
`transfers.WriteNamespaceGrant` with the call: `client.get_data(base_url,
write_namespace=grant)`. The handler is called with the grant, or with None when the caller
sent none. It uploads under the grant with `transfers.NamespaceUploader` and answers with
`uploaded_file_part(path, name=..., mime_type=...)`, where `path` is relative to the grant's
`root_path`; the caller reads it back with `uploaded_object_path(part)`, which refuses a path
outside the root. A handler that cannot use the grant (its export outgrows the grant's limits,
say) answers as it would without one.

```python
from agentenv_protocol import AgentEnvEnvironment, DataPart, environment_card, get_data, uploaded_file_part
from agentenv_protocol.transfers import NamespaceUploader, WriteNamespaceGrant


@environment_card(name="slack")
class SlackEnv(AgentEnvEnvironment):
    @get_data
    async def _state(self, write_namespace: WriteNamespaceGrant | None = None):
        bundle = self.write_bundle()  # a Path
        if write_namespace is not None and bundle.stat().st_size <= write_namespace.max_object_bytes:
            await NamespaceUploader(write_namespace).upload("slack.zip", bundle)
            return [uploaded_file_part("slack.zip", name="slack.zip", mime_type="application/zip")]
        return [DataPart(data=self.state())]
```

Dependencies are intentionally light (`pydantic`, `starlette`) so the package can be added to environment server images without pulling a heavier framework — `mcp` is imported lazily inside `create_fastmcp_app()` and is deliberately not a dependency.

## A2A agent framework

The distribution exposes two unrelated decorators named `extension`:
`agentenv_protocol.extension` declares environment/MCP extensions, while
`agentenv_protocol.a2a_agent.extension` binds an operation handler on an A2A
agent. Import the decorator from the namespace matching the application you are
building.

Install the optional agent dependencies with
`agentenv-framework-protocol[agent]`. The framework generates the Agent Card,
extension routes, A2A task lifecycle, and detached task boundary from one
agent definition:

```python
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    Usage,
    a2a_agent,
    create_app,
    enable,
    serve,
)


class MyAgentConfig(AgentConfig):
    model: str | None = "my-default-model"
    system_prompt: str | None = None
    timeout_seconds: int = 1800


@a2a_agent(
    identity=AgentIdentity(
        name="my-cli-agent",
        description="Runs My CLI",
        version="1.0.0",
        input_modes=("text", "image/png"),
    ),
    config=MyAgentConfig,
    config_description="Configure the My CLI runtime.",
    extensions=(
        MCP_CONFIG_V1,
        enable(
            TRAJECTORY_V1,
            description="Retrieve the My CLI native event trajectory.",
        ),
        TRIGGERS_V1,
    ),
)
class MyAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[MyAgentConfig]) -> TaskResult:
        execution = await run_my_cli(request)
        return (
            TaskResult.builder()
            .succeeded()
            .add_text(execution.output)
            .session_ref(execution.session_id)
            .usage(Usage(tool_call_count=execution.tool_calls))
            .native_trajectory(format="my-cli-events/v1", payload=execution.events)
            .build()
        )


agent = MyAgent()
app = agent.create_app()  # equivalently: create_app(agent)

if __name__ == "__main__":
    agent.serve()  # equivalently: serve(agent)
```

`AgentEnvAgent` mirrors `AgentEnvEnvironment`: it makes the `run()`,
`create_app()`, and `serve()` authoring surface visible to static type checking.
`@a2a_agent(...)` attaches declarative metadata to that base class; it does not
inject methods dynamically. The concrete `TaskRequest[ConfigT]` annotation on
`run()` provides typed configuration access without repeating the config type
in the base class.
When `AgentIdentity.skills` is omitted, the generated Agent Card advertises an
empty skill list. Declare explicit skills when clients need capability discovery.

The framework derives core A2A capabilities from implemented behavior.
Async-generator `run()` methods advertise streaming; coroutine `run()` methods
do not. Synchronous `run()` methods are rejected at startup. Push notifications
and state-transition history remain `False` until the framework supplies their
required runtime services.
Extensions come from explicit definitions, configured activations, and
decorated handlers in one validated registry, so routes and card advertisement
cannot drift.

`run(request)` is the required execution contract. It is an ordinary method
and needs no decorator.

For request-scoped streaming, implement `run()` as an async generator. Yield
`TaskProgress` for informational updates or validated non-terminal A2A status
updates. End every stream with one authoritative `TaskResult`; the framework
closes the generator after that result and owns terminal task state and
persistence. A runtime that already has a final file may return it as a
`FilePart` in the result message. Files created in an agent workspace remain a
runtime/control-plane collection concern rather than an A2A task-result API.

Each `run()` invocation maps to one A2A task execution. Related tasks share a
`context_id`. When a native runtime assigns a different conversation, session,
or thread identifier, return it as `TaskResult.session_ref`; the framework
supplies it as `TaskRequest.session_ref` on the next task in that context.
`session_ref` is SDK-local runtime state and is not added to the A2A wire
protocol. `TaskRequest` and `TaskResult` are framework boundary types; the
executor maps them to and from the wire-level `a2a.types.Task` lifecycle.

`TaskRequest` is a frozen record whose nested JSON values are detached copies.
Its `config`, `mcp_servers`, `skills`, `metadata`, and inbound `DataPart.data`
retain their declared `dict`/`list` types, so normal Pydantic serialization,
copying, and `json.dumps(...)` work. Within `tasks.v1`, new request fields are
additive and have framework defaults. Agent code returns a `TaskResult` through
its factories or builder so additions to the result contract do not break
existing handlers.

Expected execution failures are returned as `TaskResult.failure(code, message)`
and become failed A2A tasks with `error_type`, `error_code`, and `error_message`
in the terminal message. `error_type` is the platform classification
(`agent_error` by default, or explicitly `infra_error` for a retryable
infrastructure failure); `error_code` preserves the author's machine-readable
code. An exception escaping `run()`, an invalid return value, or a
result-mapping failure is logged with a correlation ID and reported as an
`infra_error` with code `framework.unhandled_exception`; raw exception text is
never sent to callers.
Invalid input that prevents task creation returns JSON-RPC `InvalidParams`.
Once a task exists, setup failures—including config construction—also produce
a terminal failed task rather than leaving it submitted or working.
A successful `TaskResult` must contain at least one text, file, or data part;
the framework rejects empty successes rather than emitting an ungradeable task.
The SDK does not retry tasks.

A client that gives up on a task sends `tasks/cancel`. The framework marks the
task canceled and cancels `run()`, which gets `asyncio.CancelledError` at its
next `await`. A process `run()` started keeps running unless `run()` stops it,
so kill it before re-raising.

`enable(..., description="...")` is reserved for declarations carrying
configuration or metadata. It preserves the agent-specific extension prose
published in the Agent Card. The versioned SDK definition provides a generic
fallback, while the activation can describe runtime-specific behavior without
putting mutable card metadata on `@extension(...)` operation references.
A bare definition in `extensions=` declares support implemented opaquely inside
`run()` or entirely by the framework. Binding a standard or custom operation
with `@extension(...)` automatically activates its extension, so no duplicate
entry in `extensions=` is required.

Configuration keywords are definition-owned rather than hardcoded in
`enable()`. A configurable `ExtensionDefinition` supplies a named keyword-only
`configuration_validator` returning `ExtensionConfiguration` with card
`wire_params`, internal `options`, and optional `features`. `enable()` only
dispatches to that callable. Definitions without a validator reject
configuration keywords, and third-party definitions use the same public API as
the built-ins.

Extension request parsing and schema validation failures return HTTP 400.
Unhandled exceptions raised by an implementation handler return HTTP 500;
handlers use `HTTPException` when they intentionally need another status.

Passing `config=MyAgentConfig` automatically enables `AGENT_CONFIG_V1`; direct
`enable(AGENT_CONFIG_V1, ...)` declarations are rejected. The model's fields become the Agent Card's
supported config fields, its defaults seed every task, and Pydantic validates
each deployment-time update. `request.config` is a detached, frozen
`MyAgentConfig`; its JSON-native nested fields remain mutable and serializable.
Runtime code uses typed attributes such as `request.config.model` rather than
string-keyed lookups. `AgentConfig` supplies the platform-owned `name`,
`description`, `role`, and `timeout_seconds` fields. Agents apply
`request.config.timeout_seconds` to their runtime, model, or subprocess call.
Simple Pydantic field aliases are the corresponding wire names in the Agent
Card and `/ext/agent-config` payloads.
The card publishes the validation schema but omits literal default values;
runtime-derived defaults therefore remain private to the agent process.
For compatibility with the current AgentEnv control plane, `role` is also
projected into the generic `TaskRequest.metadata` mapping. Incoming A2A message
metadata is preserved, and a non-null configured value takes precedence.
Runtime-specific fields must also have defaults, allowing partial updates to be
validated against a complete model.
Readback returns only explicitly set values.
Declare sensitive fields as `WriteOnly[T]`; the generated schema advertises
them as `writeOnly`, readback returns `"***"`, and agent code still receives the
validated value as type `T`. Pass `config_readback=False` to `@a2a_agent(...)`
to omit the GET operation entirely.

```python
from agentenv_protocol.a2a_agent import AgentConfig, WriteOnly


class MyAgentConfig(AgentConfig):
    provider_token: WriteOnly[str | None] = None
```

`output_format` is author-owned in v1 rather than a field on the base
`AgentConfig`. An agent that supports structured output must declare the field
on its config subclass, apply it to its model or runtime, and return the value
with `TaskResult.builder().add_structured_output(...)`. When an agent does not
advertise the field, AgentEnv's config negotiation omits it; the task may still
succeed with text-only output, and callers must not assume `structured_output`
will be present.

`request.metadata` exposes generic A2A message metadata. The SDK does not assign
provider-specific attribution semantics to it; an agent may pass the mapping to
downstream clients that accept metadata. The examples forward it unchanged to
their model client rather than declaring provider-specific config fields.

Runtime-owned extension behavior is attached with the single generic
`@extension(...)` decorator. Its argument is a versioned SDK operation
reference, so agent code does not repeat URIs, paths, or wire schemas. The
following decorators activate `SNAPSHOT_V1` and `TRAJECTORY_V1` automatically:

Extension handlers must be async functions and use the request type owned by
their operation. Operations with a body require exactly one argument annotated
with that public Pydantic model; bodyless operations require a zero-argument
handler. The framework validates the handler and request before invocation and
derives the Agent Card's required and optional fields from the same model.
Optional fields are part of the operation contract: every implementation
accepts them, while callers may omit them. Agent-specific capabilities use
explicit features or request variants.

```python
from agentenv_protocol.a2a_agent import (
    ContextObjectTrajectoryRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    SNAPSHOT_V1,
    TRAJECTORY_V1,
    extension,
)


@extension(SNAPSHOT_V1.save)
async def save_snapshot(self, request: ObjectSnapshotSaveRequest):
    ...


@extension(SNAPSHOT_V1.load)
async def load_snapshot(self, request: ObjectSnapshotLoadRequest):
    ...


@extension(TRAJECTORY_V1.get.context_objects)
async def get_live_trajectory(self, request: ContextObjectTrajectoryRequest):
    ...
```

The last handler opts that agent into the optional live-context variant of
`TRAJECTORY_V1.get` that uploads through an object grant;
`TRAJECTORY_V1.get.context` (`ContextTrajectoryRequest`) is its inline
counterpart. Without them the generated card advertises only the
framework-owned completed-task variants: `{task_id}`, answered inline, and
`{task_id, objects}`, answered by upload.
Snapshot `save` and `load` are an atomic core contract, while its optional
changelog handlers are enabled as an atomic feature group.
Each operation has at most one response model. The framework validates a
handler's return value against it before serializing, so a response with a
missing or unknown field is an HTTP 500.

After a successful `SKILL_CONFIG_V1.add` handler call, the framework records
the skill's `name` and `description`, plus `skill_md` for an inline skill, and
includes that record in the detached `TaskRequest.skills` snapshot for later
task executions; a bundle's read grants are not kept. It also owns
`SKILL_CONFIG_V1.list` and projects the installed skill onto the live Agent
Card. Agent implementations only install the skill into their runtime; they do
not implement listing or mutate framework/card state. The SDK contract accepts
inline `skill_md` for simple single-file skills and `BundleSkillRequest` for
multi-file or stored skills. A skill name is one path segment matching
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`, so a runtime can use it as a directory name.
Identity skills remain discoverable but are not injected into
`TaskRequest.skills`. Duplicate names are rejected before installation.

An agent can narrowly replace an SDK implementation while retaining the SDK's
wire contract:

```python
from agentenv_protocol.a2a_agent import (
    TRIGGERS_V1,
    TriggerDecideRequest,
    TriggerRegisterRequest,
    extension,
)


@extension(TRIGGERS_V1.register)
async def register_triggers(self, request: TriggerRegisterRequest):
    return await self.default_handlers.call(TRIGGERS_V1.register, request)


@extension(TRIGGERS_V1.decide)
async def decide_trigger(self, request: TriggerDecideRequest):
    decision = await self.default_handlers.call(TRIGGERS_V1.decide, request)
    # Augment the SDK decision while preserving register/decide/state storage.
    ...


@extension(TRIGGERS_V1.state)
async def trigger_state(self):
    return await self.default_handlers.call(TRIGGERS_V1.state)
```

Because `TRIGGERS_V1.decide` is SDK-owned, the registry automatically
classifies this handler as an override. Such overrides are logged at startup
and reported by `app.state.agentenv_a2a.registry.conformance()`. The override API
deliberately accepts no operational metadata. Non-standard extensions use
`@custom_extension(...)`; that escape hatch rejects the `urn:agentenv:*`
namespace.

All active SDK-owned operations for an extension form one override group. An
agent must override every operation in that group or none of them, preventing
custom and default handlers from observing different state. Partial overrides
fail during application creation. A complete override can reuse SDK behavior
through `await self.default_handlers.call(OPERATION, request)` and augment the
returned value while retaining the default shared state.

`@custom_extension(...)` is single-operation sugar. A custom URI with multiple
operations must use one shared public `ExtensionDefinition`, with each method
bound through `@extension(DEFINITION.operation)`. Repeating
`@custom_extension(...)` for the same URI creates conflicting definitions.
Shared custom definitions activate from their discovered handlers and produce
one Agent Card extension containing all operations.

Reserved `urn:agentenv:*` URIs must use the canonical SDK definition even when
constructing `ExtensionDefinition` or `OperationReference` directly. Extension
routes are rejected when they collide with `GET /health`, the Agent Card route,
or `POST` on the configured A2A JSON-RPC URL.

Existing v1 extensions keep their frozen unversioned routes. New extension
versions must use distinct resource-local versioned paths (for example
`/ext/mcp-config/v2`), and startup rejects duplicate `(path, HTTP method)`
registrations. Consumers use the endpoint advertised by the selected Agent Card
extension rather than constructing paths.

`MCP_CONFIG_V1.list` returns a name-keyed object, never a bare list:
`{"mcp_servers": {name: {"url": url, "has_headers": bool}}}`. Header values
are not exposed. `MCP_CONFIG_V1.add` requires `url` and advertises `headers` and
`name` as optional request fields: `headers` so authenticated deployments match
discovery, `name` so the caller can choose the server alias the harness prefixes
tools with (agent-env relays the env card's name: the MultiEnv's declared name, else
`env` + 4 random digits, giving e.g. `mcp__env4821__<tool>`); when absent the agent mints
`mcp_<8 hex>`.

`ATTRIBUTION_PROBE_V1.probe` lets a caller check that an agent forwards attribution
upstream. Attribution is an open map of string dimensions, like AgentEnv's cost
attribution: the SDK assigns no keys, and each deployment uses whatever dimensions
its model gateway understands. An agent records the dimensions it last sent with a
model request and returns them from a bodyless
`POST /ext/attribution-probe` as an `AttributionProbeResponse`:
`{"last_seen_attribution": {<dimension>: <value>, ...}, "last_seen_at_utc": ...}`.
The SDK does not record attribution itself, because only the agent knows what it sent.
The answer is agent-wide rather than per caller, and like every extension route the
probe carries no access check of its own, so expose an agent only to the control plane
that drives it.

Runnable, self-contained reference agents live in [`examples/`](https://github.com/scaleapi/agentenv-framework/tree/main/packages/agentenv-protocol/examples):
the normal, streaming, and multimodal `run(request)` paths, single- and
multi-operation custom extensions, and advanced ASGI-lifespan plus
common SDK-operation override hooks.

The agent examples make real OpenAI-compatible model calls. Set
`LITELLM_API_KEY` and, when needed, `LITELLM_BASE_URL`; their typed agent config
selects the model and system prompt for each deployment. Tests inject a fake
model client, so the example suite remains offline and deterministic.

### Object transfer

Skill bundles, trajectories, snapshots and changelog increments move as bytes
through short-lived HTTPS grants that AgentEnv issues from its object store, so
an agent never holds storage credentials or a provider location. The types and
helpers live in `agentenv_protocol.transfers`, which depends only on `pydantic`
and `httpx`; `agentenv_protocol.a2a_agent` re-exports them.

| Type | Wire shape |
| --- | --- |
| `HttpGetGrant`, `HttpPutGrant` | `{kind: "http-get" \| "http-put", url, expires_at, headers?}`, one exact object |
| `HttpPostPolicyGrant` | `{kind: "http-post-policy", url, fields, path_field, file_field, headers?}`, multipart POST |
| `WriteNamespaceGrant` | `{root_path, expires_at, max_objects, max_object_bytes, max_total_bytes, write: HttpPostPolicyGrant}` |
| `ReadObject` | `{media_type, max_bytes, size_bytes?, sha256?, read: HttpGetGrant}` |
| `WriteObject` | `{media_type, max_bytes, write: HttpPutGrant}` |
| `Uploaded` | `{size_bytes, sha256?}` |

URLs are absolute HTTPS and timestamps are UTC. `size_bytes` and `sha256`
describe the stored bytes. The extensions use them as follows (`?` marks an
optional field, `|` an alternative request):

| Operation | Request | Response |
| --- | --- | --- |
| skill `add` | `{name, description, skill_md}` \| `{name, description, skill_bundle: {max_total_bytes, files: [{path, object: ReadObject}]}}` | `{name}` |
| trajectory `get` | `{task_id}` \| `{context_id}` | `{trajectory}` |
| | `{task_id \| context_id, objects: {trajectory: WriteObject}}` | `{objects: {trajectory: Uploaded}}` |
| snapshot `save` | `{context_id, objects: {trajectory: WriteObject, workspace?: WriteObject}}` | `{context_id, objects: {trajectory: Uploaded, workspace?: Uploaded}}` |
| snapshot `load` | `{objects: {trajectory: ReadObject, workspace?: ReadObject}, target_context_id?}` | `{context_id}` |
| `enable-changelog` | `{write_namespace: WriteNamespaceGrant, roots?}` | `{roots}` |
| `apply-changelog` | `{increments: [{sequence, object: ReadObject}], resume_conversation?, target_context_id?}` | `{count, context_id?}` |

A bundle contains a root `SKILL.md`; its paths are unique, normalized and
relative, and their `max_bytes` sum to at most `max_total_bytes`. An uploaded
trajectory is the JSON encoding of the trajectory value. Snapshot objects are
opaque `application/octet-stream` in the runtime's own format; when AgentEnv
sends a workspace grant, the agent uploads the workspace. A changelog agent
names each increment under the namespace root by its absolute zero-based
tool-call position, six digits plus an optional extension (`000042.tar`);
positions are unique but may be sparse, and apply receives them in increasing
`sequence` order, or none when a rewind stops before the first tool call.
AgentEnv ignores response fields it does not know, so a response may carry more
than these shapes; the SDK still refuses unknown request fields.

`upload(target, source)` and `download(source, destination)` move one object;
`NamespaceUploader(grant).upload(relative_path, source)` writes under a
namespace. An upload's source is a file path or bytes already in memory. Use one uploader per grant: it runs uploads one at a time, counts an
overwrite once and enforces the grant's limits. The helpers stream within the
limits, refuse expired grants and redirects, download with identity encoding and
check the raw bytes' size and hash, and retry `transfer_unavailable` and
`transfer_timeout` up to three attempts while the grant is unexpired. A
`TransferError` raised by a handler is returned with the status below and the
body `{"error": {"code", "message", "retryable"}}`; the message never carries
grant material, provider response bodies or local paths.

| Code | HTTP | Retryable with the same grant |
| --- | ---: | --- |
| `invalid_transfer` | 400 | No |
| `grant_expired` | 410 | No |
| `transfer_too_large` | 413 | No |
| `integrity_mismatch` | 422 | No |
| `transfer_rejected` | 502 | No |
| `transfer_unavailable` | 502 | Yes |
| `transfer_timeout` | 504 | While unexpired |

Grants are secrets: agents must not log, store or echo them, and the helpers
keep grant URLs out of `httpx` logs. A provider signature authorizes storage
access but does not prove that AgentEnv chose the URL, so endpoint and egress
controls remain the trust boundary. The limits are enforced by the helpers,
that is by the uploading agent. AgentEnv accepts a trajectory upload response
without reading the object back, and registers a snapshot only once both its
objects are in the store.

SDK agents advertise only these shapes, and AgentEnv sends no others. It moves
a skill bundle, snapshot or changelog only through grants, so the call needs an
agent that advertises the object variant and an object store that issues grants
(the S3 store does, and namespace grants for changelog capture only when it
signs with long-term credentials); otherwise it fails before anything is sent.
Skills given as SKILL.md text and trajectories returned inline need no grants.
A snapshot is restored only from the snapshot objects in the table above; one
that holds the runtime's own files instead cannot be. An
agent-env release without the object variants sends other shapes, which SDK
agents refuse apart from inline skills and trajectories, and cannot load
portable snapshots or changelogs, so do not run one alongside these agents.

#### Staging

When the object store's grants cannot reach the agent, as with a local store
and an agent on a remote sandbox, AgentEnv moves the objects through the agent
itself. Every SDK agent serves the staging extension, `urn:agentenv:staging/v1`,
a small object store at `/ext/staging` on its own server. Before a call
AgentEnv pushes what the agent will read into it; after the call it pulls what
the agent wrote. The grants it sends are the ordinary ones above, with URLs
naming staged paths on the agent's own URL, so handlers are unchanged.

A sandbox can't always call its own public URL, so each of those grants also
carries an `AgentEnv-Staging-Path` header: the path the agent's own server
serves the URL at, such as `/ext/staging/{path}`. The helpers send such a
request to that server over loopback, `http://127.0.0.1:$A2A_PORT/ext/staging/{path}`
(AgentEnv deploys every agent with `A2A_PORT` set), and to the grant's URL only
when nothing listens there. An agent with transfer code of its own gets the same
URL from `transfers.loopback_url(url, headers)`.

| Route | Does |
| --- | --- |
| `PUT /ext/staging/{path}` | stores the body |
| `GET /ext/staging/{path}` | returns it, with an `ETag` |
| `POST /ext/staging/{prefix}` | stores a multipart upload's `file` at `{prefix}/{key}`, `key` being a form field sent before it |
| `GET /ext/staging/{prefix}/` | lists `{objects: [{path, size_bytes, etag}]}` below the prefix |
| `DELETE /ext/staging/{path}` | removes the object, only while it still matches an `If-Match` tag when one is given |
| `DELETE /ext/staging/{prefix}/` | removes everything below the prefix |

A path's first segment is an id AgentEnv generates, at least 22 characters, and
never shares, which is what keeps staged objects private on a public agent URL.
Everything staged counts against `AGENTENV_STAGING_MAX_BYTES` (16 GiB unless
set; `0` turns staging off) and lives until AgentEnv removes it or the server
process ends, in a directory of the process's own (or under
`AGENTENV_STAGING_DIR` when set) that only the server's user can read. One
server process owns a staging directory. An agent that doesn't use the SDK can serve the same routes
and add `{uri: "urn:agentenv:staging/v1", params: {endpoint: "/ext/staging"}}`
to its card.
