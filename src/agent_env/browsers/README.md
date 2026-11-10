The website browsers that come with agent-env, today `chromium`, a `computer_app` env a computer env lists in its
`apps`. Deployed alone, `chromium` is a browser computer, `chromium-driver`: the `ubuntu-24.04-headless` base, with no
display, and the `page` driver, whose screen is the browser's active tab, headless; the browser on it keeps its name,
`chromium`.

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
`geolocation` and `notifications`, `deny` or `ask`; and `dialogs`, `agent`, `accept` or `dismiss`. A computer env sets its own beside it
(`apps = [{ env = "chromium", settings = { start_urls = ["https://example.test/"] } }]`), and a deploy overrides any of
them (`env_params.app_settings = {chromium = {start_urls = [...]}}`). Its data is its profile, `~/.config/chromium`, which
the image has none of, so each deploy and each reset starts with a clean one. Name it from another bundle as `chromium`,
the alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/browsers/chromium`.

A task changes a browser's state with the steps in `agent_env.task_step.task_steps.browser`, each a call to one
`browser/v1` operation on the browser in the env `env_id`, the one `browser` names when the env has several:
`browser_set_cookies {cookies}`, in the shape `cookies/set` takes; `browser_delete_cookies {name?, domain?, path?}`,
every cookie matching all of those given, a domain as the cookie stores it (`.shop.test` for a domain cookie), every
cookie with none, the count deleted recorded; `browser_set_local_storage {origin, items}`, `items` a map of keys to
values, the origin's other keys left as they are; `browser_clear_local_storage {origin?}`, every origin seen with none;
`browser_open_tab {url, background?}`, in front, or with `background` behind the active tab, which stays active; and
`browser_close_tab {url}` and `browser_activate_tab {url}`, on the one open tab at that URL as the browser shows it
(scheme and host lowercase, no default port, `/` for an empty path), none or several failing the step with the tabs
open listed. `browser_sign_in {site, user}` signs the browser in to `site`, a website env deployed in the run on its
own or in a multi env, by its env id: the site's `website/v1` `session {user}` gives the cookies and local storage,
each cookie is set for every name the site is reached by (an exact name by its URL, a `*.` name as a domain cookie),
and the browser's `data/add` loads them, which opens the start tabs again while no agent has acted. A task runs it
after the load that seeds the site and again after `reset_env`, which starts the browser afresh. Preflight refuses a
step on an env with no browser, none of the name given, or several and none named. On a multi env, the steps reach
the browser apps of its computers, each through the computer's own record and control token; `browser_connect_cdp`
doesn't reach those yet.
`browser_connect_cdp` hands an agent a CDP address to the same browser. A whole state, Playwright's `storageState`
included, loads through `load_artifact` or `data/add` instead.
