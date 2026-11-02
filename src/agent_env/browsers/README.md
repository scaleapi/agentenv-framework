The website browsers that come with agent-env, today `chromium`, a `computer_app` env a computer env lists in its
`apps`.

`envs/chromium/` is Chromium for an X11 desktop on Linux (`amd64` and `arm64`): Chrome for Testing 156.0.8078.4, the
build Playwright 1.64.1 pins, fetched by sha256 when a computer env that lists it is put and unpacked into
`/opt/agentenv/apps/chromium/` by its install step, with managed policies in
`/etc/opt/chrome_for_testing/policies/managed/` for unattended use: no first-run screens, sign-in, sync, promotional
tabs, default-browser prompt, telemetry, password, autofill or translation prompts, updates, location or notification
prompts, and no warning bar. It runs without Chromium's sandbox and `/dev/shm`, which a container can't give it, and
with no proxy. It opens at boot on its `start_urls` setting, a list, the first tab active, none by default
(`env_params.app_settings = {chromium = {start_urls = [...]}}`). Its data is its profile, `~/.config/chromium`, which the
image has none of, so each deploy and each reset starts with a clean one. Name it from another bundle as `chromium`,
the alias `aliases.toml` gives it, or by its id, `@local/agentenv-framework/browsers/chromium`.
