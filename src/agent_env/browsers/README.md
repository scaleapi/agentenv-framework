The website browsers that come with agent-env, today `chromium`, a `computer_app` env a computer env lists in its
`apps`. Deployed alone, `chromium` is a browser computer: the `browser-24.04` base, with no display, and the `page`
driver, whose screen is the browser's active tab, headless.

`envs/chromium/` is Chromium for an X11 desktop or a computer with no display on Linux (`amd64` and `arm64`): Chrome for Testing 156.0.8078.4, the
build Playwright 1.64.1 pins, fetched by sha256 when a computer env that lists it is put and unpacked into
`/opt/agentenv/apps/chromium/` by its install step. It's a website browser
(`agentenv_protocol.website_browser.WebsiteBrowserEnvironment`): agentenv launches it on a debugging pipe with its own
flags, and writes managed policies to `/etc/opt/chrome_for_testing/policies/managed/` for unattended use. It runs
without Chromium's sandbox, which a container can't give it. Its env.toml spells out its settings and their defaults
under `[settings]`: `start_urls`, opened at boot, the first tab active; `window`, alone only; `locale`; `time_zone`; and
`geolocation` and `notifications`, `deny` or `ask`; and `dialogs`, `agent`, `accept` or `dismiss`. A computer env sets its own beside it
(`apps = [{ env = "chromium", settings = { start_urls = ["https://example.test/"] } }]`), and a deploy overrides any of
them (`env_params.app_settings = {chromium = {start_urls = [...]}}`). Its data is its profile, `~/.config/chromium`, which
the image has none of, so each deploy and each reset starts with a clean one. Name it from another bundle as `chromium`,
the alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/browsers/chromium`.
