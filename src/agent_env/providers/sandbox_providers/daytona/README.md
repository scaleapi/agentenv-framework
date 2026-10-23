# `daytona` sandbox provider

Runs agent-env sandboxes on [Daytona](https://www.daytona.io). Each environment and each agent gets its own Daytona
sandbox running a Docker daemon, with public HTTPS URLs for its ports. It is a VM provider: agent-env's docker-in-VM
flows run on it unchanged. That covers the gateway's docker-compose, an agent's `docker run`, image loading and
building, and artifact collection.

Adapted from Daytona's [`agentenv-daytona`](https://github.com/daytona/integrations/tree/main/packages/agentenv-daytona)
plugin (Apache-2.0).

## Set up

### 1. Install the extra

The Daytona SDK is an optional extra:

```bash
pip install 'agentenv-framework[daytona]'      # or: uv add 'agentenv-framework[daytona]'
```

Without it, selecting `daytona` fails with a `ConfigError` that names this extra. Nothing else needs it.

### 2. Get a Daytona API key

Create a key in the [Daytona dashboard](https://app.daytona.io/dashboard/keys). A key belongs to one organization,
and sandboxes are created and billed there. For local work, export it:

```bash
export DAYTONA_API_KEY=dtn_...
```

For a shared deployment, put it in your secret store instead, for example as `daytona_api_key`, and reference it with
`secret:daytona_api_key`. Never put the key itself in `config.toml`. The provider reads the key only from its config,
never from the process environment.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on Daytona, with the local stores:

```toml
[sandbox]
default       = "daytona"   # environments and sandboxes
agent_default = "daytona"   # agents

[sandbox.providers.daytona.config]
api_key = "env:DAYTONA_API_KEY"   # or "secret:daytona_api_key"

[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use Daytona per run instead, add only the `[sandbox.providers.daytona.config]` table and
pass `--sandbox daytona`, or `--env-sandbox daytona` / `--agent-sandbox daytona` on `agent-env task run`.
`agent-env config show` prints the file in effect and masks the key.

### 4. Run something

The bundled `hello` task deploys a sandbox, loads a file into it and checks it, with no model needed:

```bash
agent-env run hello --sandbox daytona
```

The first sandbox an organization creates also builds the host image (see below). `agent-env -v run …` logs each
sandbox as it starts (`Daytona sandbox started: sandbox_id=… cpu=… memory=… disk=… attribution=…`).

## Configuration reference

All keys go under `[sandbox.providers.daytona.config]`. Unknown keys are refused.

| Key            | Default                      | Meaning                                                                          |
| -------------- | ---------------------------- | -------------------------------------------------------------------------------- |
| `api_key`      | required                     | The Daytona API key, as an `env:` or `secret:` reference.                        |
| `api_url`      | `https://app.daytona.io/api` | The Daytona API URL.                                                             |
| `target`       | the organization's default   | The Daytona region sandboxes are created in.                                     |
| `snapshot`     | none                         | A Docker-capable snapshot in your organization to boot instead of the host image. |
| `base_image`   | `daytonaio/sandbox:0.9.0`    | The Debian or Ubuntu image the host image is built on. Not with `snapshot`.      |
| `egress_hosts` | `[]`                         | Extra hostnames every restricted sandbox may reach, e.g. `["*.internal.example"]`. |

## The host image

Sandboxes boot from a Docker host image the provider defines in code: `base_image` with Docker Engine, the Compose v2
plugin and a passwordless-sudo `daytona` user added. Daytona builds it the first time a sandbox asks for it and caches
it as a snapshot in the organization, so nothing is hosted outside Daytona, and later sandboxes start from the cache.
The provider starts `dockerd`, which Daytona leaves to the sandbox, and waits for it before handing the sandbox over.

That first build happens inside a sandbox create, which is allowed up to 30 minutes. A fallback chain gives each
provider 180 seconds to create a sandbox, so run `agent-env run hello --sandbox daytona` once before putting `daytona`
in a chain, or a first create there falls through to the next provider.

With `snapshot`, the provider boots that snapshot instead. It must ship Docker Engine and the Compose v2 plugin, with
`/var/lib/docker` empty, and a user with passwordless sudo. The snapshot fixes the size of what it boots, so a request
bigger than the snapshot's CPU, memory or disk is refused before anything is created; create the snapshot at least
that size.

## Resources and lifetime

- **Size:** CPU, memory and disk follow the request, rounded up to whole vCPUs and GiB and raised to Daytona's
  Docker-in-Docker floor of 2 vCPU, 4 GiB memory and 10 GiB disk.
- **Ignored requests:** `boot_mode` has no Daytona equivalent, so a `boot_mode` that is set logs a warning and has no
  effect. `create_vm(image=...)` is refused.
- **Lifetime:** the sandbox's `timeout` becomes Daytona's hard time-to-live, in whole minutes. Sandboxes are
  ephemeral: they are deleted when they stop, and never auto-stop on idleness before that.

## Networking

- **Ports:** each exposed port gets a public URL, `https://<port>-<sandbox-id>.<preview domain>`, which fills
  `tunnel_urls`. Sandboxes are created public, so the URLs need no token.
- **Daytona's own hosts:** `*.daytonaproxy01.net` and `*.daytonaproxy01.eu`, Daytona Cloud's preview domains, are
  added to every restricted policy, so a restricted agent can reach the environments it talks to. On a self-hosted
  Daytona, add its preview domain with `egress_hosts`.
- **Egress:** allow-all, or an allowlist of up to 100 hostnames with `*.` wildcards. Daytona keeps hostnames and CIDRs
  in separate, mutually exclusive lists, and agent-env always adds its platforms' hostnames to an allowlist, so
  policies with CIDRs or IP addresses are refused with `NetworkPolicyUnsupportedError` before anything is created, and
  a fallback chain skips to its next provider.
- **Organization tier:** sandbox-level egress lists need a Daytona organization on Tier 3 or above. On Tier 1 and 2
  Daytona refuses them, so a restricted policy fails with `NetworkPolicyUnsupportedError` quoting Daytona's message,
  and no sandbox is created. Unrestricted sandboxes there get the tier's own policy, which blocks Daytona preview URLs
  among others, so an agent in its own sandbox can't reach an environment in another one; tasks that keep everything
  in one sandbox, like `hello`, work.
- **Image and object downloads:** with a store that signs URLs (S3, Cloud Storage), the sandbox fetches objects and
  image tarballs itself when its policy allows the URL's host. Everything else, and everything in the local stores, is
  uploaded through Daytona's file API, which needs no egress. A restricted policy is never widened. Image pulls and
  builds inside the sandbox still need their registries allowed.
- **Reconnect:** `get_sandbox` restores the ports, their URLs and the applied egress policy, and changes nothing.

## Attribution and cost

Each sandbox carries labels: `agentenv.managed=true`, `agentenv.ports`, one `agentenv.attr.<key>` per attribution
value (slugged, at most 255 characters) and, when it boots from the host image, `agentenv.host-image` with the recipe
version. Its name is `ae-<random>-<attribution values>`. The provider logs one `agent_env.daytona_sandbox_started`
event per sandbox with its id, size and attribution. Daytona's analytics API reports usage and price per sandbox id,
deleted sandboxes included but without their labels, so join those events' `daytona_sandbox_id` with its `sandboxId`
to group cost by attribution.

## Cleanup

- A sandbox whose setup fails after it is created (Docker readiness, preview links) is deleted before the error is
  raised, also when the caller is cancelled.
- If the caller is cancelled while the create is in flight, the provider deletes the sandbox that create makes once it
  answers.
- If a create fails after Daytona may have made the sandbox, the provider finds it by its unique name and deletes it.
- A rate-limited create gets up to 5 attempts with backoff.
- Whatever else happens, the time-to-live removes the sandbox at the sandbox's `timeout`.
- `terminate` is idempotent: a sandbox that is already gone, or already being deleted, is not an error.

## Things to know

- **Commands** run as the `daytona` user through one-shot Daytona sessions, which keep stdout and stderr apart;
  agent-env's own `sudo` prefixes give root.
- **Lost connections:** when a command's transport fails without an exit status, `exec` returns exit code -1.
  `exec_script` retries idempotent scripts on -1 only.
- **Uploads** go through Daytona's file API in 4 MiB parts, joined as root and checked against their sha256.
- **No private network:** containers on different sandboxes reach each other through their public URLs.
- **Platform:** linux/amd64.
- **The model key enters the sandbox:** an agent's `LITELLM_API_KEY` is passed in, as on other providers.

## Testing

`tst/integration/providers/sandbox_providers/daytona_sandbox_smoke_test.py` runs against a real Daytona organization
when `[sandbox.providers.daytona.config]` resolves, for example with `make int-test-slow`. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. The unit tests are
`tst/unit/providers/sandbox_providers/daytona_sandbox_provider_test.py`, `daytona_sandbox_test.py` and
`daytona_network_test.py`, and run offline against in-memory stand-ins for the SDK.
