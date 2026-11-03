# Notes: a reference website/v1 site

Notes is a small notes app built on `agentenv_protocol.website.WebsiteEnvironment`. It is the reference site for the
`urn:agentenv:website/v1` server contract. Each user keeps their own notes and can create, edit, mark done and delete
them. The page shows the universe's time.

| File | What it is |
| --- | --- |
| `website.toml` | The site's spec: its name, domains, health path, source, seed, collections, personas, roles, routes, state read and clock (see `docs/website-spec.md`). |
| `site.py` | The server: the `Notes` WebsiteEnvironment, declared from `website.toml`, its `@api` routes, and `serve()` on `PORT` (8000). |
| `data/seed.json` | The seed data: two users (`ann`, `ben`), their notes, and `current_user`, the acting user. |
| `web/` | A minimal Vite app (`index.html`, `src/main.js`, `src/style.css`). Its assets (`base: "./"`) and API calls are relative to the page, so it works at `/` and under the gateway's `/website/notes/` alike; opened at a path without its final slash, such as `/website/notes`, the page first reloads itself with it. |
| `Dockerfile` | The server image: Node builds `web/`, then Python installs this repo's `packages/agentenv-protocol` with uvicorn. It installs `curl`, which the gateway's compose health check runs. |
| `env.toml` | The website env as a bundle entry, pointing at `website.toml`. |

## What it exercises

- **Site declaration.** `@website(spec="website.toml")`. The spec gives the name, domains, static folder, health path
  and seed, and the class is refused when its `@api` routes aren't the spec's. The card's website/v1 params are the
  spec's `to_card_extension()`: the domains, the health path, every route by name, the session, the state read and
  the clock.
- **Routes.** `list_users`, `sign_in`, `list_notes` (with an optional `?done=`), `create_note` (201), `get_note`,
  `edit_note`, `toggle_note` and `delete_note`. Errors are raised as `NotFound`, `BadRequest` and `Unauthorized`.
- **Roles.** `all_notes` (`GET /api/admin/notes`) is declared with `roles=["admin"]`, and the spec's `[roles]` grants
  it to `admin`. It answers only a caller that
  sends `AgentEnv-Role: admin` with `AgentEnv-Role-Key` equal to the container's `WEBSITE_ROLE_KEY`; any other caller
  gets 403. A deploy with `keyed_roles` gives the site its own role key as `WEBSITE_ROLE_KEY` and its gateway sends it
  on only beside a role it verified with the same key, so a page cannot claim `admin` through it. Any other deploy sets
  no key, and its gateway passes on whatever role a browser sends; with no `WEBSITE_ROLE_KEY` set, every caller is
  `default` and the route is closed.
- **Users.** `ApiContext.user` is the user in the signed session cookie, `agentenv_session_notes` (the site's own, so
  it doesn't collide with another site's under the same gateway host; the card names it). With no cookie, it is the acting user (the
  store's `current_user`), and `GET /api/session` signs that user in. `POST /api/sign-in` switches to another seeded
  user. A user's notes are invisible to every other user, and another user's note answers 404.
- **State.** The site writes no data hooks, so it uses the default store. `data/add` loads the seed (as a data part, or
  as a file part by bytes or `file://`), `data/get` exports it, and `data/reset` empties it. `data/get` is how
  agentenv reads what a task left the site with.
- **Time.** Notes are stamped with `self.now()`, which follows the gateway's clock once `clock/v1`'s `sync_time` names
  it. The frontend reads `X-Universe-Now` from every API response into `universeNow()` and shows it in the header.
  It uses this helper rather than `clock_patch=True`, so the page works the same against a server written without the
  SDK.
- **Readiness.** `@ready_check seeded` holds `/healthz` at 503 until the store has users, for example after a
  `data/reset` and before the next `data/add`.

## Run it

From this folder:

```bash
cd web && npm ci && npm run build && cd ..
PYTHONPATH=<repo>/packages/agentenv-protocol/src python site.py
```

Open http://localhost:8000. To work on the frontend, run `npm run dev` in `web/` while `site.py` runs; Vite proxies
`/api` to port 8000.

## Test it

```bash
python -m pytest packages/agentenv-protocol/tests/test_website_notes_example.py
```

The test imports `site.py`, runs the conformance checker, and checks sign-in, per-user notes, the admin-only route,
clock-synced timestamps, and the data plane's reset and seed load. It serves a stand-in page, so it needs no Vite
build.

## Build and verify the image

The build context is the repo root, because the image installs the protocol package from this repo's source:

```bash
docker build -f packages/agentenv-protocol/examples/website/notes/Dockerfile -t notes-site .
docker run -d --name notes-site -p 18080:8000 notes-site
python -m agentenv_protocol.website.conformance http://localhost:18080 --clock-host host.docker.internal --strict
curl -s http://localhost:18080/api/notes
```

Use `--clock-host` so the container can reach the checker's clock, and add
`--spec packages/agentenv-protocol/examples/website/notes/website.toml` to check the server against its spec too.
`--strict` fails on a recommended check's warning too (the card, `X-Universe-Now`, the clock, forwarded headers,
cookies); without it only the contract's required checks, and the spec's when given, fail the run, and Notes passes
every check either way. The clock check leaves the server following a
clock that is gone once the check ends: later timestamps start from 2031 until the server is synced again, so restart
the container before using it.

## Register it

```bash
agent-env env website put --id notes-site \
  --server-dockerfile packages/agentenv-protocol/examples/website/notes/Dockerfile \
  --server-docker-context . --health-path /healthz
```

Run it from the repo root. `environment_name` comes from the card, which the spec names `notes`. The stored env
has a server image and no frontend image. Pass `--skip-validation` to register without deploying.

`env.toml` describes the same env for a bundle. Its keys are `type`, `spec` and `backend_image`; the env takes its
`environment_name`, health path and domains from the spec. A bundle builds `backend_image` with this
folder as the build context, but this Dockerfile needs the repo root. Until the protocol package is installed from a
published release instead, register the site with `put` as above.
