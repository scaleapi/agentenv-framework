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
`{...}` token raises. Duplicate tool names raise at construction. Names must also differ across the
servers an env deploys behind one gateway: if two serve a name, the gateway serves none of their
tools and the deploy fails naming it, so prefix with `{environment_name}` when servers may share one.

`@tool(..., hidden_by_default=True)` makes being hidden a fixed property of the tool, for plumbing a
task drives (a trigger's tool action, a simulated party) that the agent under test must not see. The
tool is listed with `_meta` `{"agentenv.io/hiddenByDefault": true}` (`HIDDEN_BY_DEFAULT_META_KEY`),
and the gateway hides it from every role, in its MCP `tools/list` and `/step`, until a tool-access
step names it. Steps that name it show and hide it as they do any tool. Steps for every tool
(`tools: "*"`) never show it, but as with any tool they replace earlier steps: one for a role clears
that role's earlier named steps, and a full reset (`role: "*", tools: "*"`) clears all of them, so
grant a hidden tool after such steps. Trigger tool actions call it as usual.

`@tool(..., annotations={"readOnlyHint": True}, structured_output=True, meta={...})` lists the tool
with MCP `ToolAnnotations`, an output schema and `_meta`, on mcp's FastMCP and on fastmcp alike.
`structured_output` follows mcp's FastMCP (None derives an output schema from the return annotation
where it can, False lists none); a tool returning `Annotated[CallToolResult, Model]` sends its
result as is, image blocks included, and is listed with `Model`'s schema. An environment chooses
at mount which of its `@tool` methods it serves, and with which descriptions, by overriding
`served_tools(tools)`. After the mount, `env.mounted_card()` is the card the application serves, and
`await env.update_tool(name, description=...)` changes a served tool's description on it and in
`tools/list` alike, on both FastMCPs.

`AgentEnvStarletteApplication` mounts the same handler onto a Starlette/FastAPI app instead of
FastMCP; since those apps have no MCP tool registry, constructing one with `@tool` methods
raises.

### Errors

Raise `AgentEnvError(code, message, *, retryable=False, applied="no", hint=None, status=400)` to
answer a caller with a code rather than a crash. From an `@extension` method it answers with
`status` and the body `{"ok": false, "error": {code, message, retryable, applied, hint?}}`; from a
`@tool` it answers `isError` with `"<code>: <message>"` as the text and the same fields as the
structured content. `applied` says whether the call's effect happened (`yes`, `no`, or `unknown`
when it may have been delivered before something failed), and callers retry on their own only when
`retryable` is true and `applied` is `no`. Any other exception in an extension still answers
`extension_failed` with status 500, now with `applied: "unknown"`. On the client side,
`invoke_extension` raises `ExtensionCallError`, an `httpx.HTTPStatusError` carrying `code`,
`detail`, `retryable`, `applied` and `hint`; a body without those fields, such as a proxy's error
page, leaves `code` None and `applied` `unknown`.

### Who is calling

A tool that needs the caller declares a parameter annotated `ToolContext` (any name) and receives
it on every call. The SDK fills it, never the client: it is absent from the advertised
`inputSchema` and from `tools/list`. Tools that declare no such parameter are registered unchanged.

```python
from agentenv_protocol import ToolContext

    @tool(name="{environment_name}_list_inbox")
    def list_inbox(self, ctx: ToolContext) -> list[dict]:
        """Messages addressed to the caller."""
        return [m for m in self.messages if m["to"] == ctx.caller.role]
```

`ctx.caller.role` is the role the agent was deployed with, as the gateway forwarded it (`None` when
nothing was forwarded; the gateway's `default` role counts as nothing). It is the caller's side's
assertion, trusted exactly as far as that side is: behind a gateway it is the gateway's view of the
agent it deployed; a caller that reaches the server directly chooses it in `AgentEnv-Role`. It says
whose data a call acts on, not that the caller was authenticated, so a server exposed to untrusted
direct callers keeps access control on its own boundary. `ctx.caller.session` tells
callers apart when the gateway shares one connection to the server. `ctx.tool`, `ctx.arguments`
and `ctx.call_id` describe the call. Code that runs on behalf
of the call without a parameter of its own, such as a database method, reads the same object through
`ToolContext.current()`, which returns an empty context outside any call. The binding is per call
and follows the request's task, so concurrent callers never see each other's context. It is made
at FastMCP's own tool dispatch, for which `mcp` 1.x offers no public hook, and the mount refuses an
app without one, so a server that mounted is a server whose tool calls are bound.

To answer a call before the tool runs, override `on_tool_call(self, context)`: return `None` to let
it proceed, or an `mcp.types.CallToolResult` (`isError=True` for a refusal) to send that as the
tool's reply. An `@extension` method may declare the same parameter; it is filled from the request's
`AgentEnv-Role` header. Code that acts as a caller outside any dispatch, such as a test calling a
handler directly or a seeding script, binds a context it builds:
`with bound(ToolContext.as_caller("alice")): ...`.

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

An extension with several methods is served by giving each handler an `op`. Handlers sharing a URI
merge into one card entry, each method under `params.methods.<op>` with its own verb and endpoint,
by default `/agentenv/ext/<extension>/<op>`, where `<extension>` is the URN's name without
`urn:agentenv:` and the version. Extension-level params, such as features and limits, come from the
environment's `extension_params(uri)`, merged beside `methods`. A declaration-only extension, a
promise with params and no handler like `intake/v1`, is returned from `declared_extensions()`.
Two handlers for one method, or a handler without `op` on a URI another handler also serves, are
refused at mount.

An environment chooses at mount which of its `@extension` handlers it serves by overriding
`served_extensions(extensions)`: an extension it leaves out is not served, and of one served with
`op`, only the methods its returned descriptor still lists under `params.methods`.
`served_operations(operations)` chooses the data-plane methods (`data/add` and the others it
marks) the same way. Either hook returning something with no handler is refused at mount.

```python
@extension("urn:agentenv:computer-display/v1", op="get", method="GET")
async def display_get(self) -> dict: ...

@extension("urn:agentenv:computer-display/v1", op="set")
async def display_set(self, width: int, height: int) -> dict: ...
```

### Tool access without a gateway

An environment agents reach directly, with no gateway in front, can enforce per-role tool access
itself: set `tool_access = True` on the class. It then filters `tools/list` and refuses `tools/call`
(`permission_denied`) by the caller's role, read from `AgentEnv-Role` or a gateway-stamped `_meta`
role, and serves `urn:agentenv:disable-tool/v1` and `urn:agentenv:enable-tool/v1` at the gateway's
paths (`POST /tools/disable`, `POST /tools/enable`) with its bodies (`{role, tools: [...] | "*"}` to
`{role, disabled, allowed}`) and its rules, hidden-by-default tools included. So agent-env's
`modify_env_tool_access` step works on such an env unchanged. A call with no role is `default`, as
at the gateway.

Where the binding lives depends on the app. On mcp's own FastMCP it wraps the tool manager's dispatch; on the standalone `fastmcp` package it joins the app's middleware chain as the outermost layer, so a handler's own middleware runs under it too. An app with neither seam mounts unbound, with a warning, unless a tool declares a `ToolContext` parameter or the handler defines `on_tool_call`: then the mount is refused rather than serving an empty caller.

### Tools served on another server's behalf

An environment can serve tools another server defines, each with that server's own schema:
`forwarded_tools()` returns `ForwardedTool(tool, call, annotations, output_schema)` entries, where
`tool` is an `EnvironmentTool` carrying the other server's `inputSchema`, and `call` takes the
call's arguments as keywords, untouched, and returns that server's `CallToolResult`, which reaches
the caller as it is. An `output_schema` is listed as the tool's `outputSchema` and isn't checked
here, so `call` answers every result, its errors too, in a shape that schema admits. They're listed
on the card beside the `@tool` methods, chosen by `served_tools` and under tool access like them;
one named like another tool is refused. They need mcp's FastMCP. The cua-driver driver serves
cua-driver's own tools this way.

### Resources

Mark a method that serves MCP resources with `@resource`; an env may mark several, and each read
calls the method, which returns text or bytes. The URI decides what's served:

```python
@environment_card(name="orders")
class Orders(AgentEnvEnvironment):
    @resource("docs://orders/refund-policy", mime_type="text/markdown")
    def refund_policy(self) -> str:
        """When a refund is allowed and who approves it."""
        return POLICY.read_text()

    @resource("orders://{order_id}")
    async def order(self, order_id: str) -> str:
        return json.dumps(await self.db.order(order_id))

    @resource("skill://upstream/{file}", listed=lambda env: UPSTREAM["resources"])
    async def upstream_file(self, file: str) -> str:
        return await read_upstream(f"skill://upstream/{file}")
```

- A fixed URI is one resource. Its name defaults to the method's name and its description to the
  docstring.
- A template such as `orders://{order_id}` is an MCP resource template; the method takes its parts.
- A template with `listed` serves the resources `listed` returns, each as MCP lists one (`uri`,
  `name`, `description`, `mimeType`), and only those. Each is read through the method with its
  URI's parts. That's how an env serves another server's resources from a list: the cua-driver
  driver serves cua-driver's skill files this way.

A URI served twice, a listed URI that doesn't match its template, and a method that doesn't take
exactly the template's parts are refused. Resources need mcp's FastMCP.

### Instructions at connect

An MCP server may answer `initialize` with `instructions`: how to use it, which a client may put in
the agent's context, as Claude Code does. Mark the plain method that returns them with
`@server_instructions`:

```python
@environment_card(name="orders")
class Orders(AgentEnvEnvironment):
    @server_instructions
    def how_to(self) -> str:
        return "Look an order up with get_order before refunding it; refunds can't be undone."
```

They're read once, when the SDK builds the app (`create_app`, `serve`, `create_fastmcp_app`), and
every session gets the same text. An env has at most one such method. An app you build yourself
and pass to `mount` keeps its own instructions: set them when you construct it.

### Getting part of the data

A `@get_data` handler that takes a `paths` parameter returns only the data under the paths a caller
names (`client.get_data(base_url, paths=[...])`), and is called with None when the caller names
none. A handler without one answers `invalid_params` to a call that names paths, rather than
returning everything.

### A skill for agents

An env can hand agents a skill: a SKILL.md an agent reads before it uses the env. Mark the method
that writes it with `@agent_skill`; it returns the skill's description and its markdown body,
sync or async:

```python
@agent_skill
def guide(self):
    return "How to place and track orders.", "# The shop\n\nPlace orders with `place_order`.\n"
```

The card then declares `urn:agentenv:skill/v1` (`SKILL_EXTENSION_URI`), whose `get {name}`
answers `{name, description, skill_md}`, the shape `skill-config/v1` takes, with the frontmatter
written under the name the caller gives; a name that isn't lowercase letters, digits and single
hyphens, up to 64, is refused `invalid_argument`. A skill with other files beside its SKILL.md
returns them as a third value, `{relative path: text}`, and `get` answers them as `files`.
agent-env's `add_skills` step installs it on an agent through `env_ids`, named after the env's
environment name, a skill with files as a bundle of them; for an env behind a gateway, whose card
nests its servers' cards, it installs each server's skill under that server's name.

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

Dependencies are intentionally light (`pydantic`, `starlette`, `httpx`) so the package can be added to environment server images without pulling a heavier framework — `mcp` is imported lazily inside `create_fastmcp_app()` and is deliberately not a dependency.

## Calling an env's tools

A loop of your own, such as a trainer's rollout or an eval harness, reaches an env's tools through `ToolSession`. It
is an MCP client for the env's streamable-HTTP endpoint, written on `httpx`, so it works wherever this package
installs, whichever `mcp` version the loop's own environment pins.

```python
from agentenv_protocol import ToolSession

async with ToolSession(mcp_url=mcp_url, role="alice@example.com") as session:
    tools = await session.list_tools()                 # EnvironmentTool: name, description, inputSchema
    result = await session.call_tool("slack_send_message", {"channel": "general", "text": "hi"})
    print(result.text, result.isError)
```

- **The URL:** `mcp_url` is `DeployedEnv.mcp_url` in agent-env, or the address a card was fetched from joined with
  `client.mcp_path(card)`.
- **Sessions:** a `ToolSession` is one MCP session, and calls on it may run concurrently. `async with` opens and ends
  it; `open()` and `close()` do the same for a session held across calls. A tool tells sessions apart by
  `ctx.caller.session`: the session's id when the server is reached directly, or, behind a gateway, a key the gateway
  gives each session. A session belongs to the event loop that opened it, and using or closing it on another raises
  `RuntimeError`, so a caller with a synchronous API runs all of a session's calls on one loop of its own.
- **Roles:** `role` is sent as `AgentEnv-Role` (`ROLE_HEADER`) and says who the caller is. A gateway lists only the
  tools that role may use, answers a call to a hidden one with an error result, and forwards the role to the servers
  behind it; a server reached directly reads the header itself. Either way a tool reads it as `ctx.caller.role`
  ([Who is calling](#who-is-calling)), where `default`, like no role, is `None`. `headers` adds any others.
- **Errors:**
  - A tool that fails, or a call the server refuses (a tool hidden from the role, or an `on_tool_call` refusal),
    returns a result with `isError` set and the server's text.
  - `ToolSessionError` means the request failed at the MCP level: a JSON-RPC error, an HTTP error status, a reply
    that can't be read, or a session the server no longer knows. After a restart, the env's state is usually gone too.
    Only the server the session talks to reports this: a gateway reconnects to a child that restarted and retries the
    call, and a stateless server has no session to lose, so neither restart raises anything.
  - `httpx.TransportError` means the connection failed, and `TimeoutError` that a request outlived `timeout`
    (600 s by default, the gateway's own limit on a tool call).
  - `ToolSession` retries nothing. A call that raised may still have run on the server, or may yet run: a request
    stalled on a bad link can arrive after the client has given up. So after any failed call the env's state is
    unknown; read it back (`data/get`) or end the episode.
- **Tools:** `list_tools()` returns the tools the role may use as `EnvironmentTool`, the type a card's
  `capabilities.tools` holds. Each `inputSchema` is the JSON Schema a model API takes as the tool's parameters.

## Computer drivers

A computer driver is what runs inside a computer an agent uses through screenshots and input. Write
one as a `ComputerDriverEnvironment` subclass from `agentenv_protocol.computer`, which needs the
`computer` extra (`pip install 'agentenv-framework-protocol[computer]'`: mcp, uvicorn, Pillow and PyAV).
The driver marks the methods that capture the screen and send input; the base class serves the
computer contract from them, so the tools, results and errors can't drift from it. A complete
driver for an X11 desktop is agent-env's reference driver, in its
[`drivers` bundle](https://github.com/scaleapi/agentenv-framework/tree/main/src/agent_env/drivers).

```python
from agentenv_protocol.computer import (
    ComputerDriverEnvironment, Display, computer_driver,
    capture, click, display, drag, press, scroll, type_text,
)


@computer_driver(name="my-driver", supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]}])
class MyDriver(ComputerDriverEnvironment):
    buttons = ("left", "right")       # features, on the card
    max_click_count = 2

    @display
    async def size(self) -> Display: ...                    # width, height, scale
    @capture(display="x11")
    async def grab(self): ...                               # the whole display, native resolution
    @click
    async def click_at(self, x, y, button, count, modifiers): ...
    @drag
    async def drag_along(self, path, button, duration_ms, modifiers): ...
    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers): ...
    @type_text
    async def type_in(self, text): ...
    @press
    async def press_keys(self, keys): ...


if __name__ == "__main__":
    MyDriver().serve()
```

**Operations.** Operations get points in screen pixels, after the base class has mapped, checked
and queued the call, and may be sync or async. `@capture` returns a Pillow image or an
`Image(width, height, data, mode)` (PNG or JPEG bytes, or raw `RGB`, `RGBA`, `BGRA` or `BGRX`
pixels), at the display's size times its scale. Input operations may return an `Effect`.

| Decorator | Required | Gives |
|---|---|---|
| `@display`, `@capture`, `@click`, `@drag`, `@scroll`, `@type_text`, `@press` | yes | the seven `computer_*` tools and `computer/v1` |
| `@set_display` | no, with `display_modes` | `computer-display/v1` `set` (`get` is always served) |
| `@pointer_move`, `@pointer_down`, `@pointer_up`, `@pointer_position` | all or none | `computer-pointer/v1` and its five tools; click and scroll at the pointer |
| `@key_down`, `@key_up` | both or none | `computer-keyboard/v1` and its two tools |
| `@run_in_session` | no; a Linux default | `computer-exec/v1` |
| `@write_files` | no; a Linux default | `data/add` |
| `@record` | no | the stream `computer-recording/v1` records from; without it the base class polls `@capture` |

The class is checked when it is defined: a missing required operation, half a group, two methods
for one operation that some platform could both match, or features that don't match the
decorators raise `TypeError`. Predicates (`os`, `distribution`, `version`, `arch`, `display`)
choose a method per platform, read from `/etc/os-release`, the machine and `XDG_SESSION_TYPE` or
`DISPLAY`: `@capture(display="x11")` and `@capture(display="wayland")` can sit side by side. A
driver's own `@run_in_session` or `@write_files` replaces the base class's Linux default, which runs
as the driver's user inside the desktop session. The base class keeps its state in `self._computer`
and names its own members `_computer_*` and `computer_*`; a driver that redefines one of them, or
another private member of the base class, is refused when it is defined, so a helper of the
driver's can't silently replace one. What the optional operations give is served only when the
driver has them, through the `served_tools`, `served_extensions` and `served_operations` hooks.

**What the base class serves:**

- **Tools.** `computer_screenshot` (a frame: `frame_id`, image size, display resolution and scale;
  `region` crops a frame at native resolution), `computer_click`, `computer_drag`,
  `computer_scroll`, `computer_type`, `computer_key` (W3C key names) and `computer_wait`, plus the
  group tools. Every result is structured, with a one-line summary as its text block; input results
  carry `applied`, `delivery` and `others_since_frame`, and `screenshot: true` returns a fresh frame
  after acting. Click and scroll leave out `x` and `y` to act at the pointer on a computer with
  `computer-pointer/v1`; elsewhere a call without them is refused `invalid_argument`.
- **Frames and spaces.** Points are pixels of a named frame (your latest by default), or a
  0..`norm_scale` grid over it with `space: "norm"`. A point outside the frame or the range is
  refused (`out_of_bounds`), never clamped. The last 64 frames are kept, and a frame goes stale
  (`stale_frame`) when the display geometry it was captured at changes.
- **The input queue.** Input calls run one at a time in arrival order; one that waits past
  `queue_wait_s` is refused `input_busy`. Screenshots and waits never queue.
- **The computer skill.** `computer/v1` declares the seven tools, its params the driver's
  features. The driver serves its skill through `skill/v1` (an `@agent_skill` handler on the base
  class): a SKILL.md written from the computer as it booted (its platform, session user and home,
  the live screen size, what its input takes, its OS's shortcuts) and ending with the driver
  class's `notes`, markdown for what its card can't say. A driver whose software ships a skill of
  its own returns it from `bundled_skill()` (its description, body and other files), and the
  computer skill leads with it, the computer's part following.
- **Framework extensions.** `computer-ready/v1` (`status`: real checks of the display and a
  capture, the driver's own `ready_checks()`, and one per app on the computer, asked of the app's
  own server (see Computer apps); tools refuse `device_unavailable` until they pass once), `computer-display/v1`, `computer-spaces/v1` (the default space, written into the tool
  descriptions), `computer-keyboard/v1`, `computer-exec/v1` (`run {argv, cwd?, env?, timeout_s?,
  request_id}`: output capped per stream, a repeated `request_id` answered with the first result) and,
  with `recording`, `computer-recording/v1` (below).
- **The data plane.** `data/add` takes file parts that name their path, relative to the session
  user's home, under the `data-objects/v1` marker (`{"path": ...}`), as inline bytes, an https URL,
  or a `transfers.ReadObject` read grant under the marker's `read`, and writes them atomically.
  `data/get` returns the files under the `paths` it's given, relative to the home, or under the
  card's `default_data_paths` (`Desktop`, `Documents`, `Downloads`) when it's given none, inline or
  uploaded under a `write_namespace` grant, leaving out the paths its apps keep their data in. An
  environment whose `@get_data` hook takes no `paths` refuses them.
- **The card** (the name from `ENVIRONMENT_NAME`, then `@computer_driver`'s), and per-role tool
  access (`tool_access` is on).

**Errors.** A driver raises the errors table's exceptions, all `AgentEnvError`s:
`InvalidArgument`, `OutOfBounds`, `StaleFrame`, `InputBusy`, `PermissionDenied`, `Unsupported`,
`BackgroundUnavailable`, `TargetUnavailable`, `DeviceUnavailable` and `LimitExceeded` mean nothing
was sent (`applied: "no"`); `Timeout`, `DeliveryFailed` and `Internal` mean it may have been. Any
other exception from an input operation is reported `delivery_failed` with `applied: "unknown"`.

**Settings** reach the driver as one JSON value in `AGENTENV_COMPUTER_SETTINGS`
(`ComputerSettings`): `default_space`, `default_norm_scale`, `queue_wait_s`,
`tap_hold_ms`, `type_delay_ms`, `max_hold_ms`, `layout`, `drag_duration_ms`, `exec_output_bytes`
and `exec_timeout_s`. A driver reads them as `self.settings`, for example its tap hold in `@press`.

Every caller is `default` until agents carry an identity of their own; `caller_identity(context)`
names the caller whose latest frame and others' input a call uses.

`MyDriver.declaration(platform)` returns the card the class serves on a `Platform`, built without
running the driver, its operations or its dependencies.

### Recording the display

Every driver serves `computer-recording/v1`: video of the display, as MP4 (H.264) segments, for the
framework to start and end around the work it wants on record. `recording = Recording(...)` sets its
limits, and `recording = None` turns it off.

```python
from agentenv_protocol.computer import RecordedFrame, Recording, record


class MyDriver(ComputerDriverEnvironment):
    recording = Recording(max_fps=30)      # max_concurrent=2, max_duration_s=86400, max_segment_s=15

    @record
    async def stream(self, hints):         # hints.scope, fps, max_edge, cursor, max_segment_s, directory
        async for image, captured_at in my_screen_stream(hints.fps):
            yield RecordedFrame(image, captured_at)   # a Pillow image or a PyAV VideoFrame
```

- **The stream.** `@record` yields `RecordedFrame`s from a stream the OS pushes, which the base class
  scales, encodes and splits into segments, each frame timed by its `captured_at`; or
  `RecordedSegment`s, files a native recorder finished under `hints.directory`, which it only hashes,
  uploads and lists. The base class closes the stream at `end`. A source that blocks, as most OS
  capture APIs do, goes through `frames_in_thread(frames)`: the generator `frames()` runs in a thread
  of its own, the latest 8 frames wait for the encoder, and it is closed when the recording stops. A
  driver without `@record` records by the base class polling `@capture`, with a pointer drawn at
  `@pointer_position`: the slow path, at most 15 fps.
- **The methods.** `start {request_id, scope?, fps?, max_edge?, cursor?, upload?}` returns
  `{recording_id, started_at, format}`; `end {recording_id, upload?, reason?}` closes the last segment
  and returns the manifest; `end_all {reason?}` ends every recording not yet ended, each under the
  grant it started with, and returns `{manifests}`, as the framework does before it resets or tears a
  computer down. A repeated `request_id`, or a repeated `end`, answers the same. A start past
  `max_concurrent` is refused `limit_exceeded`; `max_duration_s`, free disk under 512 MB or a stream
  failing for 60 s end a recording by itself.
- **Segments.** A segment closes at `max_segment_s`, when the frame size changes or after a gap in the
  stream. Each uploads as it closes under `start`'s `write_namespace` grant, at
  `<recording_id>/<index>.mp4`, and is then deleted; `end`'s own grant uploads any that missed. With no
  grant they stay under `~/.agentenv-recordings/`. An ended recording's manifest uploads beside its
  segments, at `<recording_id>/manifest.json`, so a reader that didn't end it finds it there.
- **The manifest** (`RecordingManifest`). Its `segments` (`ManifestSegment`) carry each one's path,
  `started_at` and `ended_at` (its first and last frames' capture times), size, frame count, byte size,
  SHA-256 and whether it uploaded, and `ended_by` says why the recording ended: `end`, `reset`,
  `teardown`, `limit` or `error`. A field it doesn't know is left out.

### Checking a driver: the conformance probe

The probe checks a running driver against the contract through its card, tools and extensions. It
reports each check, and gives each extension the card declares a verdict: verified, unverified or
failed.

```bash
python -m agentenv_protocol.computer.conformance http://localhost:18765 [--level basic|full] [--json report.json]
```

```python
from agentenv_protocol.computer.conformance import run_probe

report = await run_probe("http://localhost:18765", "full")   # a ProbeReport
failed = [c for c in report.checks if c.status == "failed"]
```

The CLI exits 0 when no check failed, 1 when one did and 2 when it can't reach the driver.

| Level | Checks | Needs on the computer |
|---|---|---|
| `basic` | P0 ready and a screenshot; P1 the card and tool schemas; P2 native frames, regions and JPEG; P5 refusals and their codes; X the declared extensions (`computer-display`, `computer-spaces`, `computer-keyboard`, `computer-exec`, `data/get`, per-role tool access, the pointer) | nothing |
| `full` | `basic`, then P3 colours land where the mapping says; P4 clicks show their effect; P5 refusals change nothing; P6 typed and pressed keys arrive; P7 scrolls and drags arrive; P9 commands and `data/add` run as one unprivileged user; X pointer and keyboard input, `data/add`, and a recording's segments matching their files with no gap, a repeated `request_id` joining it and `max_concurrent` held | Linux, `python3` with Tk, a 1280x800 screen or larger, `computer-exec` and `data/add` |

At `full` the probe writes the Fixture Desk, a Tk window (`conformance/fixture.py`), into
`~/.agentenv-probe` with `data/add` and starts it with `computer-exec`. It aims at the boxes the
window's widgets report, reads back what reached them, and then removes the fixture. When the
fixture can't be staged, the `full` checks are unverified rather than failed. The probe leaves out
the checks for an extension the card doesn't declare. P8 is not applicable to a driver without
leasing, and unverified for one with it until the probe has lease checks. The probe calls as the
default role; the tool-access check hides a tool from a role of its own, `agentenv-probe-<random>`,
and gives it back.

## Computer apps

A computer app is software an agent uses on a computer, beside the driver: a desktop app, a service
or both. Write one as a `ComputerAppEnvironment` subclass from `agentenv_protocol.computer`, an
`AgentEnvEnvironment` whose class also says what it is to a computer. Writing one and reading what
it declares need only the plain package: `agentenv_protocol.computer` imports what a driver needs
(mcp, uvicorn, Pillow, PyAV) only when a driver's names are used. On a computer it runs as its own
server, which needs the `app` extra (below).

```python
from pydantic import BaseModel, SecretStr
from agentenv_protocol.computer import (ComputerAppEnvironment, Launch, Packages, Service, computer_app,
                                        install_step, linux)

class LinuxSettings(BaseModel):
    font: str = "DejaVu Sans"

@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["thunderbird"]), python="requirements.txt",
       services=[Service(name="imap", command="venv/bin/python imap.py")],
       launch=Launch(command=["thunderbird"], window={"class": "thunderbird"}), settings=LinuxSettings)
class MailApp(ComputerAppEnvironment):
    skill = "skill.md"

    class Settings(BaseModel):
        signature: str = ""
        password: SecretStr = SecretStr("")

    class BuildSettings(BaseModel):
        locale: str = "en_US"

    @install_step(scope="session")
    def profile(self) -> None: ...
```

- **`@computer_app`** names the platforms the app supports, as a driver names them, and the
  `permissions` it needs of the computer. Its env.toml names the rest: the app itself
  (`environment_name`), the `computer` and `driver` it deploys alone on, and `build_settings`, its
  build settings' values (`build_settings = { locale = "fr_FR" }`).
- **`@linux(...)`** is how a Linux computer installs it, `Linux`'s fields: `files`, a folder of the app's
  copied under its prefix, which holds the class's file; `packages`, the OS's, as apt takes them
  (`Packages(apt=["name", "name=version", "name/suite"])`); `python`, a requirements file there,
  installed into the venv every app gets beside them; `sources`, extra apt repositories
  (`AptSource`, with a key file in the folder); `downloads`, files fetched by URL and checked by
  sha256 (`Download`); `run`, scripts of the folder run as root or the session user (`RunScript`);
  `services`, programs the computer's init runs from the prefix as the session user; and `ports`,
  those it listens on. Packages, requirements and downloads are resolved and fetched once, when the
  computer env is put, and every build installs only those stored files, with the network closed.
  Each value is checked as the class is defined: a URL, a sha256, a path that stays in the folder.
  Installs are exact: apt installs each package without its recommends, and the venv gets exactly
  the wheels resolved for it (`pip install --no-deps`). A requirements file lists everything itself,
  since its lock covers that file alone: a line including another file (`-r`, `-c`), an editable
  install (`-e`), a local path or a local folder of wheels is refused at put.
- **Each OS has its own decorator,** and what differs by OS goes in it: `launch=` always; and
  optionally `skill=`, a file used instead of the class's, and `settings=` and `build_settings=`,
  models whose fields add to the class's `Settings` and `BuildSettings` on that OS. What's on the
  class applies to every OS. `@install(os_type, spec, launch=..., ...)` is the generic form, for an
  OS type a plugin adds, with that OS type's own install model as `spec`; `@linux(...)` is
  `@install("linux", Linux(...), ...)`. On the computer the class reads what's declared for the OS
  it runs on (`platform.detect()`).
- **`launch=Launch(...)`** says how it starts in the desktop session: with it (`at="boot"`) or when
  the agent opens it (`"demand"`). `ready=Ready(...)` says when it's ready, within `timeout_s` of its
  launch: a window matching `launch.window` by its class, instance or title, a port answering on
  loopback, or a command exiting 0. `@ready_check` marks a method for readiness that needs logic (a
  sync one runs in a thread), and `@install_step(scope=...)` one a computer runs as it installs the
  app, after its run scripts, offline and without the class's `__init__`. Each runs on every OS the
  app installs on, or on those its `os=` names (`@install_step(os="linux")`, `@ready_check(os=[...])`);
  naming one the app declares no install for is refused when the class is read. A step reads
  `self.build_settings`; a deploy's `self.settings` aren't known at build, and reading one raises
  `AttributeError` (`python -m agentenv_protocol.computer.run_install_step <file>:<Class> <method>`).
- **Its skill** is a file of its folder (`skill = "skill.md"`) or an `@agent_skill` method, not both.
- **Its data** is under the `paths` its OS's decorator names, relative to the session user's home,
  none inside another (`@linux(..., paths=["Mail", ".config/mail/accounts.json"])`), or what its own
  `@add_data`, `@get_data` and `@reset_data` hooks serve instead (below).
- **`Settings`** is typed; a computer hands each app the values a deploy gives it, as one JSON value
  in `APP_SETTINGS_ENV` (`AGENTENV_APP_SETTINGS`, the apps' values by name), and the base class reads
  its own, by `ENVIRONMENT_NAME`, as `self.settings`. A `SecretStr` field is a secret.
- **`BuildSettings`** is typed too: values the app's env fixes in its computer's image, which only
  another image changes, so a deploy can't give one. The composer writes them to
  `.agentenv/build_settings.json` in the app's folder, and the base class reads them, over the
  defaults, as `self.build_settings`, in its install steps and on the computer alike. None is a
  secret, since the image keeps them; `declaration()` refuses a `SecretStr` one.

On a computer each app is its own server: `python -m agentenv_protocol.computer.serve_app
<file>:<Class>`, a program the computer's init runs from the app's folder in the app's venv, which
has the `app` extra (`agentenv-framework-protocol[app]`: mcp and uvicorn). It serves the class's
card and tools on the Unix socket `.agentenv/server.sock` in that folder, with `computer-app/v1`:
the app's `launch` as its params, and the methods `launch` (POST), which starts the app in the
desktop session, and `ready` (GET: an `AppReadiness` of `ready`, `problems`, `launched`, `waited_s`
and `timeout_s`), which runs the app's readiness checks then, at once, each within 10 s. A server
that can't start, or whose start in the session fails (adding the launcher, say), writes why to
`.agentenv/server.failure` in the folder and exits, and the driver reports that in its place. Once
the session is up the server adds the app's launcher to the session user's menu, which runs `python
-m agentenv_protocol.computer.launch <folder>` to ask the server, so the server starts every copy of
the app, and launches a `boot` app, once per boot. The driver finds each app by the socket in its
folder: the computer is ready once every app's server answers and each `boot` app, and each app with
no launch, says it's ready; a `demand` app counts once something launches it. What the server does
in the session (`agentenv_protocol.computer.session`) is Linux's.

The server serves the app's data plane and its skill. With `paths` alone, `data/add` writes the files
it's given under them and refuses any other path, or one a link leads out of them; `data/get`
returns the files there, a file tree; and `data/reset` puts them back as the server found them when
it first started on this boot, which is the app as installed. Hooks serve what they mark instead.
A hook's `data/add` parts come as they were sent, inline or as a read grant, and
`part_bytes(part)` returns a part's bytes either way. Either way the base class takes data calls and
launches one at a time, runs a load and a reset with the app stopped and relaunches it after,
follows a failed load with a reset, and fails a call that runs past `data_timeout_s` (300 s); a sync
hook runs in a thread, which the reset and relaunch after it wait for. The driver fronts its apps:
its card lists each one's card as a child environment, rooted at `/svc/<app>/`, where it forwards
requests to the app's server.

`MailApp.declaration()` returns what the class declares, an `AppDeclaration`, which a `computer_app`
env stores, and `python -m agentenv_protocol.computer._declare <folder> <file> <class>` prints it as
JSON, or a driver's `DriverDeclaration`, without the class's own dependencies installed.

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
copying, and `json.dumps(...)` work. `TaskRequest.trajectory` is the exception:
it is the task's live trajectory log, described below. Within `tasks.v1`, new
request fields are additive and have framework defaults. Agent code returns a `TaskResult` through
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
so kill it before re-raising. Trajectory events it appended, including any it
appends while cleaning up, are kept, and its log ends `canceled`.

#### Live trajectories

A runtime that sees its native events as they happen appends each complete
event to `request.trajectory` and enables `TRAJECTORY_V1` with `live=True`:

```python
@a2a_agent(identity=..., extensions=(enable(TRAJECTORY_V1, live=True),))
class MyAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[MyAgentConfig]) -> TaskResult:
        request.trajectory.set_format("my-cli-events/1")
        execution = await run_my_cli(request, on_event=request.trajectory.append)
        return TaskResult.text(execution.output)
```

`append(event)` takes a JSON object and serializes it at once, so an event
never changes after it is appended and a read never returns part of one. It does
no I/O; call it from one thread at a time. After the task ends the log refuses
further events. `set_format` names the events' versioned format, which readers
use to parse them; call it before the first `append`, which otherwise raises.
Once an event is appended the format cannot change.

The framework keeps each task's log and serves it while the task runs. When the
result carries no `native_trajectory`, the log is the task's final trajectory:
the completed-task reads `{task_id}` and `{task_id, objects}` return it as a JSON
array. A `native_trajectory` remains the final record of a runtime that can only
produce one at the end, and takes precedence over appended events.

`live=True` adds two cursor reads to `TRAJECTORY_V1.get`, both framework-owned:

| Request | Reads |
| --- | --- |
| `{task_id, after, limit?}` | one task's events |
| `{context_id, after, limit?}` | the events of every task in a context, in execution order |

Both answer `{context_id, task_id, state, format, events, next, has_more}`
(`TrajectoryEventsResponse`):

- `events` are the events from position `after`, zero-based. `next` is
  `after` plus the number returned, the position to read from next; `has_more`
  says more events are already readable past this page. A page holds at most
  `limit` events (1 to 1,000, default 1,000) and about 4 MiB, but always at
  least one event when one exists. A `limit` outside 1 to 1,000, or `after` past
  the last event, is HTTP 400.
- `state` is `pending` before the first event, `running` while the task runs,
  then `completed`, `failed` or `canceled`. A terminal state is reported only
  once the log can no longer change, so a reader that has reached `has_more:
  false` in a terminal state has every event. The terminal state is recorded
  before the task's terminal A2A status, so the completed-task reads succeed as
  soon as a client sees the task end.
- A context read reports the context's current task, the running one or else the
  latest, and the most recent `format` any of its tasks set; its `state` is
  `running` while any task runs.
- A task or context the agent has not seen, or no longer keeps, is HTTP 404. A
  task is known from the moment `message/send` returns its id, so a 404 for an
  id that was readable means it was evicted, and any cursor held for it, or for
  its context, is no longer valid.

`live=True` also enables the context reads `{context_id}` and
`{context_id, objects}`; the framework answers them with the context's events
unless the agent binds its own handlers, which take precedence. An agent that
overrides the whole `get` operation must accept the cursor requests too.
Clients detect live support from the card, with
`card_request_accepts(get_request, {"task_id", "after"})`.

The logs are kept in memory for up to 1,024 contexts. Past that, whole contexts
are dropped, least recently used first; a context with a task still pending or
running is never dropped, nor is the one whose task has just ended. A running
task's log is logged as a warning once it passes 100 MiB.

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
`wire_params`, internal `options`, and optional `features` and request
`variants`. `enable()` only
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
counterpart. Without them, or `enable(TRAJECTORY_V1, live=True)`, the
generated card advertises only the framework-owned completed-task variants:
`{task_id}`, answered inline, and `{task_id, objects}`, answered by upload.
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

`PEER_AGENTS_V1` lets an agent delegate to the other agents in its task. AgentEnv
sets the routing table with `POST /ext/peer-agents` (`{"peers": [{"name", "url",
"card", "description"}]}`; an empty list clears it, and names must be unique), and
`GET /ext/peer-agents` returns it. An agent that enables the extension without
handlers of its own gets the SDK's: the SDK serves six MCP tools, `peer_list`,
`peer_get_card`, `peer_send_message` (send and wait, for at most 50 s, so a call
answers within the 60 s MCP clients commonly give a request), `peer_start_task`,
`peer_get_task` and `peer_cancel_task`, at `/ext/peer-agents/mcp` on the agent's
own port, and adds them to every task's `request.mcp_servers` as `peer_agents`, so
a harness that connects the task's MCP servers gives its model the tools with no
code of its own. The tools wrap the peers' A2A `message/send`, `tasks/get` and
`tasks/cancel`; they answer loopback callers only, while the application's lifespan
runs, and the name `peer_agents` is reserved from `MCP_CONFIG_V1.add`. An agent that
overrides both operations keeps the table itself and gets no tools.

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

`PROXY_V1` hands an agent's turns to a driver outside its sandbox, and `ProxyAgent`
(`agentenv_protocol.a2a_agent.proxy`) serves it. The proxy has no model: each message it
gets stays an open turn, its task working, until the driver replies, and every action the
driver's model takes runs in the proxy, where the agent would work. The driver's first
`POST /ext/proxy/claim` (`{"token"}`, at least 22 characters) sets the token every later
call carries in its body. A claim with another token gets 409, and any other call with a
wrong one 401. The token is held in memory only, never in the environment a command
inherits. Content travels as parts, `{"kind": "text", "text": ...}`; a message with any
other part fails as `proxy.unsupported_part` before the driver sees it. The other
operations:

- `next` (`{"token", "after"}`) waits up to 25 s for the turns' version to pass `after`,
  then returns `{"version", "turns": [{"task_id", "context_id", "parts", "system_prompt",
  "skills"}]}`. A turn its sender cancels drops out of the list.
- `tools` (`{"token", "task_id"}`) lists what the turn can call, with input schemas: the
  tools of each env server registered through mcp-config, and `bash`. Two servers offering
  one name are refused with 409, and servers that haven't all listed their tools within
  600 s get 502. `record` waits for the same listing.
- `act` (`{"token", "task_id", "action_id", "action"}`) runs one action: `{"kind": "tool",
  "server", "name", "arguments"}` calls an env server through `ToolSession`, with the URL
  and headers registered for it, and `{"kind": "bash", "command"}` runs `bash -c` in the
  proxy's folder, as its user, in a process group of its own. Where there is no `bash`, as
  in a minimal image, it runs `sh -c`, and the `bash` tool's description says so. A command
  is done when its shell exits: what it started in the background keeps running, and output
  that arrives more than a second later is read and dropped. Either takes
  `"timeout_seconds"` (3600 for `bash`, 600 for a tool by default). The driver picks
  `action_id`, unique for the proxy, so a retried `act` returns the action it already
  started. It answers `{"done", "result"}` at once when the action ends within the long-poll;
  `result` (`{"token", "action_id"}`) long-polls one that doesn't, and gets 404 once the
  action's turn has closed. A `bash` result is `{"exit_code", "stdout", "stderr",
  "timed_out"}`, with no `exit_code` when the command was stopped and 126 when it couldn't
  start; a tool's is `{"content", "is_error"}`.
- `record` (`{"token", "task_id", "message", "model", "usage"}`) records one model call: an
  OpenAI chat message, an Anthropic message or a string; anything else gets 400. Once a turn
  records a call, each action must be one the last recorded call made, by name and
  arguments (a `bash` call by its command alone, a whole float equal to its int), and all of
  them must run before the next `record` or the `reply`, else 409.
  Calls to tools the turn doesn't offer aren't checked.
- `reply` (`{"token", "task_id", "parts"}`, or `"error"` in place of `"parts"`) ends the
  turn: parts complete the task, and an error fails it as `proxy.driver_error`.

A turn's actions belong to it: when it closes, by reply or withdrawal, those still running
are stopped, their whole process groups killed, and its results forgotten; the app's
lifespan stops what runs when it shuts down. The proxy writes the turn's trajectory as it
goes, readable live, as OTel GenAI spans in format `agentenv-proxy/v1`: `chat` for each
recorded call, `execute_tool` for each action, and `chain` for the turn, each with an
OTel status: `ERROR` for an action that failed, a command that exited nonzero or timed out,
and a turn that ended in an error. Skills it is given
are written to `~/skills/<name>/`, once each, and a turn names each one's folder.
`ProxyAgent(shell=False)` is a proxy for an agent that works in no sandbox: it offers no
`bash` and keeps no skills. It holds the env servers' headers as any agent does, so run it
with no credentials in its environment and expose it only to the control plane that drives
it.

Runnable, self-contained reference agents live in [`examples/`](https://github.com/scaleapi/agentenv-framework/tree/main/packages/agentenv-protocol/examples):
the normal, streaming, and multimodal `run(request)` paths, single- and
multi-operation custom extensions, and advanced ASGI-lifespan plus
common SDK-operation override hooks.

The agent examples make real OpenAI-compatible model calls. Set
`LITELLM_API_KEY` and, when needed, `LITELLM_BASE_URL`; they also send the headers in
`LITELLM_EXTRA_HEADERS`, a JSON object AgentEnv sets when the endpoint needs any. Their typed
agent config selects the model and system prompt for each deployment. Tests inject a fake
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
| | `{task_id \| context_id, after, limit?}` | `{context_id, task_id, state, format, events, next, has_more}` |
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
(the S3, Cloud Storage and local stores do, though S3 issues the namespace grants
changelog capture needs only when it signs with long-term credentials); otherwise
it fails before anything is sent.
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
request to that server over loopback, `http://127.0.0.1:{port}/ext/staging/{path}`,
and to the grant's URL only when nothing listens there. The port is the one the
SDK's server took the request being served on, or else `A2A_PORT` (set by AgentEnv
on agents it deploys from an image), as in a separate process such as a hook.
An agent with transfer code of its own gets the same URL from
`transfers.loopback_url(url, headers)`, and an agent with a server of its own
marks each request with `transfers.serving_on(port)`.

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
