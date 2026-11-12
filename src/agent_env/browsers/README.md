The website browsers that come with agent-env, today `chromium`, a `computer_app` env a computer env lists in its
`apps`. Deployed alone, `chromium` is a browser computer, `chromium-driver`: the `ubuntu-24.04-browser` base, with no
desktop, and the `page` driver, whose screen is the browser's active tab; the browser runs headful there, on an X
server only it uses, and keeps its name, `chromium`.

`envs/chromium/` is Chromium for an X11 desktop or a computer with no display on Linux (`amd64` and `arm64`): Chrome for Testing 156.0.8078.4, the
build Playwright 1.64.1 pins, fetched by sha256 when a computer env that lists it is put and unpacked into
`/opt/agentenv/apps/chromium/` by its install step. It's a website browser
(`agentenv_protocol.website_browser.WebsiteBrowserEnvironment`): agentenv launches it on a debugging pipe with its own
flags, and writes managed policies to `/etc/opt/chrome_for_testing/policies/managed/` for unattended use. Flags and policies of its own
(`QUIET_FLAGS`, `DISABLED_FEATURES` and `QUIET_POLICIES` in `chromium.py`, found against this build, to check again
when its version moves) stop the requests Chrome makes of its own: component and Safe Browsing updates, the network
time, a spellcheck dictionary, the AI Mode check, autofill predictions and its search engine's preconnect. Its proxy
sees only its pages' requests, but for one Google account check at launch. It runs
without Chromium's sandbox, which a container can't give it. Its env.toml spells out its settings and their defaults
under `[settings]`: `start_urls`, opened at boot in order, the first tab active, each waited on to load; `window`,
alone only; `locale`; `time_zone`; and
`geolocation` and `notifications`, `deny` or `ask`; `dialogs`, `agent`, `accept` or `dismiss`; and `network`, whether
`data/get` exports its network log, on by default. A computer env sets its own beside it
(`apps = [{ env = "chromium", settings = { start_urls = ["https://example.test/"] } }]`), and a deploy overrides any of
them (`env_params.app_settings = {chromium = {start_urls = [...]}}`). Its data is its profile, `~/.config/chromium`, which
the image has none of, so each deploy and each reset starts with a clean one. Name it from another bundle as `chromium`,
the alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/browsers/chromium`. A multi env names it as
its website browser with `browser_envs = ["chromium"]`, in place of the run's `playwright-mcp`: it then deploys on a
sandbox of its own beside the multi env's computers, sized by the deploy step's `child_envs.chromium`, its tools served
under `chromium_`, routing the multi env's synthetic sites through the gateway, its network its own env's, as any
computer child's.
Its server keeps its log in `/var/log/agentenv/browser-chromium.jsonl`: an entry for each CDP command, MCP tool call
and pixel action an agent takes, and an event for each tab opened, navigated or closed with what caused it, secrets
taken out (`agentenv_protocol.website_browser.log`); `browser/v1` `log` reads it and `data/get` exports it as
`browser-log.jsonl`. It lasts the computer's boot: `reset_env` boots the computer afresh with a new one, so a task that
needs the log before a reset takes it first with `snapshot_env`.
Its proxy keeps its network log (`agentenv_protocol.website_browser.network`), which `browser/v1` `network` reads and
`data/get` exports as HAR 1.2, `network.har`, the file a network grader such as WebArena-Verified's reads: routing
sites, the shared site proxy's, `/etc/agentenv/sites/network.jsonl`, each exchange whole, a site's sign-in headers
taken out; else the base's stand-in's, `/var/log/agentenv/network.jsonl`, each connection, since it decrypts none. On
a computer with a desktop it has none, and its export no `network.har`.
Like its log, it lasts the computer's boot. It keeps cookies and form bodies as the site saw them, sign-in passwords
among them, so a run that needs no network grader turns it off with `network = false`: its export then has no
`network.har` and `network` returns no lines.

A task changes a browser's state with the steps in `agent_env.task_step.task_steps.browser`, each a call to one
`browser/v1` operation on the browser in the env `env_id`, the one `browser` names when the env has several:
`browser_set_cookies {cookies}`, in the shape `cookies/set` takes; `browser_delete_cookies {name?, domain?, path?}`,
every cookie matching all of those given, a domain as the cookie stores it (`.shop.test` for a domain cookie), every
cookie with none, the count deleted recorded; `browser_set_local_storage {origin, items}`, `items` a map of keys to
values, the origin's other keys left as they are; `browser_clear_local_storage {origin?}`, every origin seen with none;
`browser_open_tab {url, background?}`, in front, or with `background` behind the active tab, which stays active; and
`browser_close_tab {url}` and `browser_activate_tab {url}`, on the one open tab at that URL as the browser shows it
(scheme and host lowercase, no default port, `/` for an empty path), none or several failing the step with the tabs
open listed. `browser_lease_context {label, grants?}` leases a context of the browser's beside its default one, its own
cookies, local storage, tabs and downloads (in `~/Contexts/<label>/Downloads`), recorded as granted to the agents
`grants` names, which nothing enforces yet; leasing a label again finds the same lease. Each step above, and
`browser_sign_in`, takes a `context`, a lease's label, to act in it instead of the default context.
`browser_release_context {label}` ends a lease, closing its tabs and deleting its folder; a reset or relaunch of the
browser ends every lease, so a task leases again after `reset_env`. A context an agent makes over its CDP address is
leased to it as `agent-<agent name>-<n>`, and only that agent may dispose it. `browser_sign_in {site, user}` signs the browser in to `site`, a website env deployed in the run on its
own or in a multi env, by its env id: the site's `website/v1` `session {user}` gives the cookies and local storage,
each cookie is set for every name the site is reached by (an exact name by its URL, a `*.` name as a domain cookie),
and the browser's `data/add` loads them, which opens the start tabs again while no agent has acted. A task runs it
after the load that seeds the site and again after `reset_env`, which starts the browser afresh. Preflight refuses a
step on an env with no browser, none of the name given, or several and none named. On a multi env, the steps reach
the browser apps of its computers, each through the computer's own record and control token.
`browser_connect_cdp` hands an agent a CDP address to the same browser, under the origin where the agent reaches the
env, so through a tunnel that ends TLS it's https and wss; on a multi env, one at its gateway, asked through the
gateway's `/svc/<computer>/` route, so the agent reaches the browser through the gateway's CDP relay as it reaches the
rest of the env, with the multi env's tools hidden from its role; a computer there with several website browsers is
refused, as the relay doesn't tell them apart yet, and so, behind a tunnel that ends TLS, is a browser that takes no
origin, one composed before agentenv-framework 0.9.1505. A whole state, Playwright's `storageState`
included, loads through `load_artifact` or `data/add` instead.
