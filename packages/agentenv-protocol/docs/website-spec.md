# The website spec

A website env used to be described in three places: its `env.toml` (`type = "website"`, `health`, `backend_image`),
the decorators in its code (`@website(domains, static, health, data)` and `@api` routes with names and roles), and the
card it serves (the `urn:agentenv:website/v1` extension: domains, health, routes, session cookie). The website spec
is one declaration of a site. Each way of serving a site either produces a spec or is checked against one, and specs
compose, so a site can be assembled from parts.

The code is `agentenv_protocol.website.spec`. The JSON Schema of a spec file is
`python -m agentenv_protocol.website.spec schema`.

## Shape

A spec is a TOML file. Its top-level keys are fixed and every section is optional:

| Section | What it says |
| --- | --- |
| `extends` | The path of a spec this one overlays (see [Overlays](#overlays)). |
| `[site]` | `name`, the card's name; `domains`, the host names the site answers for, with `*.` as the leftmost label matching any subdomain; `health`, the path that answers 200 once the site is usable (`/api/health` by default), read only for a server whose card has no `website/v1` `ready`, which is deprecated: a deploy and the conformance check read readiness from the card. |
| `[source]` | What serves the site, exactly one of `image`, `sdk`, `static`, `live` or `replay` (see [Sources](#sources)). |
| `[data]` | `collections` (each with `key`, `owner` and an optional JSON `schema`), or one `schema` of the whole state, or `models = "mcp:<server>"` to bind the site's data to an MCP server's schema; and `seed`, the files it starts with, in order. |
| `[users]` | `personas` (ids, or `{id, name, role}` tables); `acting_user`, who a browser with no session is; `sign_in`, which is `cookie` (the site's signed session cookie, by default), `none` or `live`; and, for a cookie sign-in by a server not built on the SDK, its `session_cookie` and `session_path`. |
| `[roles]` | A role name mapped to the routes it may call. These roles are added to each route's own `roles`. |
| `[routes]` | `ops`, the named operations, each `name = "METHOD /path"` or a table `{method, path, roles, served_by}`; and `openapi`, an OpenAPI document whose operations, named by `operationId` (each name once) with roles from `x-agentenv-roles`, are routes too. Triggers and records call these names `op`. |
| `[state]` | How a grader reads the site: `read = "data_plane"` (`data/get`), `"go"` (`GET <path>`, `/go` by default, the session's `{initial_state, current_state, state_diff}`), or `"diff"` (the writes a run made, as the env's proxy recorded them). |
| `[clock]` | How the site follows the universe's time: `follow = "header"` (`X-Universe-Now` on every answer), `"date_patch"` (its pages' `Date` is shifted too), or `"none"` (wall time). |
| `[[uses]]` | Components: another spec mounted under a path prefix (see [Composition](#composition)). |

Paths in a spec (Dockerfiles, folders, seeds, schemas, captures, the OpenAPI document and components) are relative to
the spec file.

The Notes example's spec, `examples/website/notes/website.toml`:

```toml
[site]
name = "notes"
domains = ["notes.test"]
health = "/healthz"

[source]
sdk = "site:Notes"
assets = "web/dist"

[data]
seed = ["data/seed.json"]

[data.collections.users]

[data.collections.notes]
owner = "owner"

[users]
personas = [{ id = "ann", name = "Ann Lee" }, { id = "ben", name = "Ben Ortiz" }]
acting_user = "ann"
sign_in = "cookie"

[roles]
admin = ["all_notes"]

[routes.ops]
list_users = "GET /api/users"
sign_in = "POST /api/sign-in"
list_notes = "GET /api/notes"
create_note = "POST /api/notes"
get_note = "GET /api/notes/{note_id}"
edit_note = "PATCH /api/notes/{note_id}"
toggle_note = "POST /api/notes/{note_id}/toggle"
delete_note = "DELETE /api/notes/{note_id}"
all_notes = "GET /api/admin/notes"

[state]
read = "go"

[clock]
follow = "header"
```

## Validation

`load(path)` reads a spec, resolves `extends` and `uses`, and validates the result. If anything is wrong, it raises
`SpecError`, whose `problems` lists every problem as `<file>: <where>: <why>`. For example,
`.../shop.toml: routes.ops: b and a are both GET /x`. The checks within a section are: unknown keys, exactly one
source, lower-case host names, health and route paths outside the data plane's `/agentenv` and the card's
`/.well-known/agent-env.json`, credentials that are `secret:`/`env:` references rather than values, and an
`acting_user` that is one of the `personas`.

The checks across sections are:

- No two routes share a verb and path, and no GET route is served at the health path.
- `[roles]` names only routes that exist, including the namespaced routes of components.
- `sign_in = "live"` needs a `live` or `replay` source.
- A `live` or `replay` site is read as `diff`, and only such a site is.
- An `sdk` site follows the clock, so `follow = "none"` is refused for it.
- A `static` site serves no routes.
- `served_by` is allowed only on a route of a live or replayed site.
- A collection with an `owner` needs a site that signs users in.

`python -m agentenv_protocol.website.spec validate PATH` runs these checks. `show PATH` prints the resolved spec as
JSON, and `show --card PATH` prints the card params it advertises.

## Sources

Every source maps onto the same spec. Each row below says how a source is described, what it produces, and what it is
checked against.

| Source | `[source]` | Where the spec comes from | Deployable today |
| --- | --- | --- | --- |
| SDK class | `sdk = "module:Class"`, `assets = "<static folder>"` | `@website(spec="website.toml")` declares the class from the spec, and the class is refused if it doesn't serve what the spec declares. `from_environment(cls)` derives a spec from the decorators of a class declared the old way. | Yes, as a website env's server image |
| Any-stack image | `image = "<ref>"` or `image = { dockerfile, context }` | Written by hand. `conformance --spec` checks the running server against it. | Yes, through `backend_image` in env.toml |
| Static | `static = "<folder>"` | Written by hand. | Modeled only: no generic static server image ships yet |
| Live real site | `live = { allow_hosts, credentials }` | Written by hand; `from_card` reads one back from a proxy that serves a card. | Modeled only; a website env refuses it |
| Replayed capture | `replay = { capture, allow_hosts }` | Usually an overlay of the live spec. | Modeled only; a website env refuses it |

For every source, `to_card_extension()` returns the `urn:agentenv:website/v1` params a server of the spec advertises:
`domains`, `health` and `routes` (each `{method, path, name, roles}`); `session_cookie` and `session_path` for a cookie
sign-in; and `state` and `clock` when they are declared. A WebsiteEnvironment builds its card from these params, so
the card and the spec cannot drift apart. `from_card(card)` reads a running site's card back into a spec. It recovers
the site, routes, sign-in, state and clock. A card does not carry a site's source, data, personas or the
`[roles]` table; roles are folded into each route.

### The SDK

```python
@website(spec="website.toml")
class Notes(WebsiteEnvironment):
    ...
```

`@website(spec=...)` takes a path relative to the class's file, or a loaded `WebsiteSpec`. The spec supplies the
domains, health path, static folder (`assets`), seed files and clock patch. If any of those are also passed to
`@website`, it is refused. When the class is declared, it is checked against the spec:

- Its `@api` routes must match the spec's routes by name, verb, path and roles, whenever the spec declares routes.
- `source.sdk` must name the class.
- The sign-in must be the SDK's cookie.
- The state read must be how the class is read.
- `acting_user` must be the seed's `current_user`.

If the class has no `@environment_card` name, the card is named by `site.name`. A card name that differs from
`site.name` is refused. Classes declared with `@website(domains=..., ...)` work as before; their card goes through
`from_environment(cls).to_card_extension()`.

### env.toml

A website env's `env.toml` can point to a spec with `spec = "website.toml"`, or carry one inline as a `[spec]`
table:

```toml
type = "website"
spec = "website.toml"
backend_image = { dockerfile = "Dockerfile" }
```

The env takes its `environment_name` from `site.name` and its health path from `site.health` when the env.toml leaves
them out. If the env.toml sets one to a different value, the env is refused. The env stores the spec's domains
(`WebsiteEnv.domains`). A spec with a `live` or `replay` source is refused, because a server image can't serve it.
`backend_image` still says how the image is built.

### Conformance

`python -m agentenv_protocol.website.conformance <url> --spec website.toml` runs the usual checks plus two more:

- `spec` checks that the card's name, domains, health path, routes, session cookie and session path are the spec's,
  and its `state` and `clock` too when the spec declares them.
- `spec_routes` checks that every declared route is served. An `OPTIONS` to the route's path, with each path parameter
  filled in by a value the converter of the card's route of that name takes (a UUID for `uuid`, else `0`), must find
  something there; a GET route without path parameters must answer neither 404 nor 405; and each
  route must be an operation of the server's OpenAPI document when its card offers one. When the card offers no
  document, a route only the `OPTIONS` can check (a non-GET route, or a GET route with path parameters) counts as
  served only when that `OPTIONS` answers below 500 with an `Allow` listing its method. Any other answer, such as a
  catch-all `OPTIONS` handler's 200 without `Allow`, a 500, or an `Allow` that doesn't list the method, leaves the
  route unverified: the check is then a `warning` (a failure with `--strict`) rather than a pass, and fails only
  when a route is shown missing. Only GET routes are called directly, so the check changes no state.

The `health` check reads readiness from the card's `website/v1` `ready`. For a card without it, it reads the health path, which is deprecated: with `--spec`, the spec's rather than the card's.

## Overlays

`extends = "<spec>"` merges this spec over the one it names. The merge works like this:

- Tables merge key by key, recursively.
- Arrays and scalars replace. For example, `domains = [...]` or `personas = [...]` in the overlay replace the base's.
- `[source]` replaces as a whole, so an overlay can turn a live site into a replay without leftover keys.
- `name = false` removes an inherited entry of `[routes.ops]`, `[roles]` or `[data.collections]`. A route removed
  this way is removed from the operations of the `openapi` document too.
- A route written as `"METHOD /path"` is expanded to a table before merging, so an overlay can change one key of it,
  such as `post_message = { served_by = "..." }`. An operation of the `openapi` document is one too: a table naming it,
  such as `add_item = { roles = ["staff"] }`, is written over the document's operation.
- `[[uses]]` entries are added. An entry with the same `as` (or, without one, the same `spec`) replaces the inherited
  entry.
- The base's relative paths are rewritten relative to the overlay's folder, so a base in another folder still finds
  its files.

`extends` chains are followed, and a cycle is refused. In code, `spec.overlay({...})` applies the same merge to a
loaded spec. The Notes test uses it to swap in a stand-in static folder.

Overlays are how one site gets variants:

- A frozen capture with one synthetic route: `examples/website/specs/slack.toml` extends `slack-live.toml`, replaces
  the live source with a replay of a capture, and swaps `post_message` to an SDK class.
- The same site with another persona set: `examples/website/specs/intranet-ben.toml` signs in as `ben`, with personas
  `ben` and `cy`, and drops the `editor` role.

```toml
# slack-live.toml
[site]
name = "slack"
domains = ["slack.com", "*.slack.com"]
health = "/api/api.test"

[source.live]
allow_hosts = ["*.slack-edge.com", "*.slack-imgs.com"]
credentials = { token = "secret:SLACK_USER_TOKEN" }

[users]
sign_in = "live"

[routes.ops]
list_conversations = "GET /api/conversations.list"
history = "GET /api/conversations.history"
post_message = "POST /api/chat.postMessage"
add_reaction = "POST /api/reactions.add"

[state]
read = "diff"

[clock]
follow = "none"
```

```toml
# slack.toml: the live site frozen as a replay, one route answered by a synthetic implementation
extends = "slack-live.toml"

[source.replay]
capture = "captures/slack-workspace.har"
allow_hosts = ["*.slack-edge.com"]

[routes.ops]
post_message = { served_by = "slack_synthetic:SlackChat" }

[clock]
follow = "date_patch"
```

## Composition

A `[[uses]]` entry mounts another spec as a component:

```toml
[[uses]]
spec = "components/sso.toml"   # relative to this spec
mount = "/sso"                 # the path prefix its routes are served under
as = "sso"                     # its namespace; the component's site.name by default
data = { accounts = "users" }  # a collection of the component's, mapped onto one of this site's
```

When a spec is resolved, each component is resolved first, including its own `extends` and `uses`. It is then merged
into the site:

- **Routes.** Each of the component's routes is served at `<mount><path>` and named `<as>.<name>`. The roles its own
  `[roles]` grants come with it. The site's `[roles]` can name the composed routes, for example
  `editor = ["sso.login"]`.
- **Data.** A collection named in `data` merges into the site's collection of that name. They must agree on `key`,
  `owner` and `schema` where both set them. Any other collection is added as `<as>.<name>`. A schema file a
  component names stays the component's file, its path made relative to the site's spec. A component's seed files
  are not merged: the site seeds the shared collections.
- **Users.** The component's personas are added to the site's, merging the fields of a persona they share; fields
  that contradict are a conflict. A component must sign in the way the site does, or not at all (`none`).
- **The rest.** The component's `site` (domains, health), `source`, `state` and `clock` are not merged. A mounted
  component lives under the site's own domains and is served by the site's source.

These are conflicts, each reported as `uses[<i>]: ...`:

- A composed route name that the site already declares.
- A composed verb and path that another route already serves.
- A route of the site's own under the component's mount prefix (unless the mount is `/`).
- A `data` mapping of a collection the component doesn't declare.
- A collection or persona that disagrees with the site's.
- A sign-in mismatch.
- A component with no name.
- A cycle of `uses`.

A resolved spec has no `extends`, `uses` or `routes.openapi` left: the document's operations are in `routes.ops`.
`spec.components` lists what was mounted, each with its name, mount and own resolved spec. `to_dict()` of a resolved
spec reads back as the same spec.

`examples/website/specs/intranet.toml` uses a fake SSO component, `components/sso.toml`, mounted at `/sso`. The
component's `accounts` collection is mapped onto the intranet's `users`, and its `grants` collection becomes
`sso.grants`:

```toml
[site]
name = "intranet"
domains = ["intranet.test", "*.intranet.test"]

[source.image]
dockerfile = "Dockerfile.intranet"

[data.collections.users]

[data.collections.pages]
owner = "author"

[users]
personas = ["ann", "ben"]
acting_user = "ann"

[roles]
editor = ["publish_page"]

[routes.ops]
list_pages = "GET /api/pages"
publish_page = "POST /api/pages"

[[uses]]
spec = "components/sso.toml"
mount = "/sso"
data = { accounts = "users" }

[state]
read = "go"
```

Resolved, its routes are `list_pages`, `publish_page`, `sso.authorize` (`GET /sso/authorize`), `sso.login`,
`sso.token` and `sso.userinfo`, and its collections are `users`, `pages` and `sso.grants`.

## The name table

A multi env routes every host name its sites answer for through one table. `name_table(specs)` builds it:

- **Kinds.** A site's `domains` are `synthetic` when the env serves the site (an image, sdk, static or replay source)
  and `live` when the site is the real one. A live or replay source's `allow_hosts` are `allowlist`: hosts let through
  to the internet, not served.
- **Precedence.** When two kinds claim one pattern, the higher kind wins: synthetic, then live, then allowlist. The
  loser is kept in `shadowed` beside the entry it lost to. This is how a synthetic overlay takes over a name a live
  site or an allowlist also claims.
- **Resolution.** `table.resolve(host)` looks first at the highest kind of the entries matching the host, then at the
  most specific pattern of that kind: an exact name, then the longest wildcard. So a synthetic `*.slack.com` beats a
  live `status.slack.com`. A synthetic `inbox.example.test` beats a synthetic `*.example.test` of another site.
- **Wildcards.** `*.example.test` matches every subdomain at any depth, but not `example.test` itself.

These are conflicts, each a `SpecError`:

- Two sites claim the same pattern at the same kind, synthetic or live, whichever kind wins the pattern. A host that
  two different patterns match (`*.example.test` and `inbox.example.test`) is not a conflict, because specificity
  decides it.
- A site has no `site.name`.
- Two sites share a name.

Two sites allowlisting the same host is not a conflict, because an allowlist only lets traffic through.

## What is wired and what is only modeled

These parts are wired:

- The spec model, loader, validation, `extends`, `uses`, the card params, `from_card`, `from_environment`, the name
  table, the JSON Schema and the command line.
- `@website(spec=...)` and the SDK card generated through the spec.
- `spec` in a website env's env.toml (name, health and domains).
- `conformance --spec`.

These parts are only modeled: they are declared and validated, but nothing serves them yet.

- The `static`, `live` and `replay` sources, and route `served_by`.
- `[data] models` and collection schemas.
- `state.read = "diff"`.
- Mounting a component's implementation. An SDK class that uses a component must serve its routes itself.
- The name table's use by the gateway.
- Component references other than file paths.
- The env's stored domains: deploy doesn't yet check them against the card.
