# Notes: a reference website/v1 site

Notes is a small notes app built on `agentenv_protocol.website.WebsiteEnvironment`. It is the reference site for the
`urn:agentenv:website/v1` server contract. Each user keeps their own notes and can create, edit, mark done and delete
them. The page shows the universe's time.

| File | What it is |
| --- | --- |
| `site.py` | The server: the `Notes` WebsiteEnvironment, its `@api` routes, and `serve()` on `PORT` (8000). |
| `data/seed.json` | The seed data: two users (`ann`, `ben`), their notes, and `current_user`, the acting user. |
| `web/` | A minimal Vite app (`index.html`, `src/main.js`, `src/style.css`). Its assets (`base: "./"`) and API calls are relative to the page, so it works at `/` and under the gateway's `/website/notes/` alike; opened at a path without its final slash, such as `/website/notes` or `/website/notes/index.html`, the page first reloads itself at its folder's address. |
| `Dockerfile` | The server image, built with this folder as its context: Node builds `web/`, then Python installs the released protocol (`agentenv-framework-protocol`) with uvicorn. It installs `curl`, which the gateway's compose health check runs against the card. |
| `env.toml` | The website env as a bundle entry. |

## What it exercises

- **Site declaration.** `@website(domains=["notes.test"], static="web/dist")`. The card advertises website/v1 with
  the domains and every route by name.
- **Routes.** `list_users`, `sign_in`, `list_notes` (with an optional `?done=`), `create_note` (201), `get_note`,
  `edit_note`, `toggle_note` and `delete_note`. Errors are raised as `NotFound`, `BadRequest` and `Unauthorized`.
- **Roles.** `all_notes` (`GET /api/admin/notes`) is declared with `roles=["admin"]`. It answers only a caller that
  sends `AgentEnv-Role: admin` with `AgentEnv-Role-Key` equal to the container's `WEBSITE_ROLE_KEY`; any other caller
  gets 403. The gateway passes on whatever role a browser sends, so without the key a page could claim `admin`; with no
  `WEBSITE_ROLE_KEY` set, every caller is `default` and the route is closed.
- **Users.** `ApiContext.user` is the user in the signed session cookie. With no cookie, it is the acting user (the
  data's `current_user`, which `acting_user()` returns), and `GET /api/session` signs that user in. `POST /api/sign-in` switches to another seeded
  user. A user's notes are invisible to every other user, and another user's note answers 404.
- **State.** The site keeps its users and notes itself, starting from `data/seed.json`, and serves them through its
  data hooks: `data/add` adds a part's records by id (a data part, or a file part by bytes or `file://`), `data/get`
  exports them, and `data/reset` empties them. `data/get` is how agentenv reads what a task left the site with.
- **Time.** Notes are stamped with `self.now()`, which follows the gateway's clock once `clock/v1`'s `sync_time` names
  it. The frontend reads `X-Universe-Now` from every API response into `universeNow()` and shows it in the header.
  It uses this helper rather than `clock_patch=True`, so the page works the same against a server written without the
  SDK.
- **Readiness.** website/v1's `ready`, through the card, says not ready (`@ready_check seeded`) until the site has
  users, for example after a `data/reset` and before the next `data/add`. The site serves no health path.

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

The build context is this folder; the image installs the released protocol, so it runs the SDK as published, while
the test above runs this repo's.

```bash
docker build -t notes-site .
docker run -d --name notes-site -p 18080:8000 notes-site
python -m agentenv_protocol.website.conformance http://localhost:18080 --clock-host host.docker.internal
curl -s http://localhost:18080/api/notes
```

Use `--clock-host` so the container can reach the checker's clock. The clock check leaves the server following a
clock that is gone once the check ends: later timestamps start from 2031 until the server is synced again, so restart
the container before using it.

## Register it

In a bundle, `env.toml` declares the env: copied to a bundle's `envs/notes/`, `agent-env bundle register <bundle>`
builds the image with that folder as its context. Its keys are `type`, `environment_name` and `backend_image`; domains
are not one of them, as they come from the card. Without a bundle:

```bash
agent-env env website put --id notes-site --server-dockerfile Dockerfile --server-docker-context .
```

Run it from this folder. `environment_name` comes from the card (`@environment_card(name="notes")`). The stored env
has a server image and no frontend image. Pass `--skip-validation` to register without deploying.
