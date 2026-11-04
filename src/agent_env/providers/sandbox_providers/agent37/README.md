# `agent37` sandbox provider

Runs agent-env sandboxes on [Agent37](https://www.agent37.com) VMs. Each environment and each agent gets its own
Linux virtual machine with its own kernel, Docker and Compose v2 running inside, and public HTTPS URLs for its ports.
It is a VM provider: agent-env's docker-in-VM flows run on it unchanged. That covers the gateway's docker-compose, an
agent's `docker run`, image loading and artifact collection.

## Set up

### 1. Nothing to install

The provider talks to Agent37's REST API with `httpx`, which agent-env already depends on, so there is no extra.

### 2. Get an Agent37 API key

Sign up at [agent37.com](https://www.agent37.com), add funds (VMs need a funded workspace), and create a key under
**API keys** in the dashboard. A key belongs to one workspace, and VMs are created and billed there. For local work,
export it:

```bash
export AGENT37_API_KEY=sk_live_...
```

For a shared deployment, put it in your secret store instead, for example as `agent37_api_key`, and reference it with
`secret:agent37_api_key`. Never put the key itself in `config.toml`. The provider reads the key only from its config,
never from the process environment.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on Agent37, with the local stores:

```toml
[sandbox]
default       = "agent37"   # environments and sandboxes
agent_default = "agent37"   # agents

[sandbox.providers.agent37.config]
api_key = "env:AGENT37_API_KEY"   # or "secret:agent37_api_key"

[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use Agent37 per run instead, add only the `[sandbox.providers.agent37.config]` table and
pass `--sandbox agent37`, or `--env-sandbox agent37` / `--agent-sandbox agent37` on `agent-env task run`.
`agent-env config show` prints the file in effect and masks the key.

### 4. Run something

The bundled `hello` task deploys a sandbox, loads a file into it and checks it, with no model needed:

```bash
agent-env run hello --sandbox agent37
```

`agent-env -v run …` also logs each VM as it starts
(`Agent37 instance started: id=… name=… region=… resources=… ports=… attribution=…`).

## Configuration reference

All keys go under `[sandbox.providers.agent37.config]`. Unknown keys are refused.

| Key       | Default  | Meaning                                                                              |
| --------- | -------- | ------------------------------------------------------------------------------------ |
| `api_key` | required | The Agent37 API key, as an `env:` or `secret:` reference.                            |
| `region`  | `us`     | Where VMs run: `us`, `eu` (France) or `ap-south` (India). The price is the same in all three. |

## Resources and lifetime

- **Sizes:** Agent37 VMs come in four shapes: 2 vCPU / 4 GiB, 4 vCPU / 8 GiB, 4 vCPU / 16 GiB and 4 vCPU / 32 GiB.
  A request gets the smallest that covers its CPU, memory and disk, so agent-env's default (1 vCPU, 8 GiB, 10 GB) runs
  on 4 vCPU / 8 GiB.
- **Disk:** every VM gets its shape's largest disk: 12, 20, 40 or 80 GB. Docker keeps its images and containers on it.
- **Refused requests:** a request beyond 4 vCPU, 32 GiB or 80 GB is refused before anything is created, not shrunk.
- **Ignored requests:** `boot_mode` has no Agent37 equivalent, so a `boot_mode` that is set logs a warning and has no
  effect.
- **Lifetime:** Agent37 has no delete-after timer, so the sandbox's `timeout` is not enforced and a VM runs until it is
  terminated. agent-env tears its sandboxes down when a run ends; a run that crashes leaves its VMs running. They are
  named `agentenv-<random>` and carry the run's attribution as metadata, so they are easy to find in the dashboard or
  with `GET https://api.agent37.com/v1/instances`, and `DELETE /v1/instances/{id}` removes one.
- **Idle VMs:** auto-sleep is off, so a VM never sleeps on its own while a run waits on it.

## Networking

- **Ports:** each exposed port is an Agent37 [public port](https://www.agent37.com/docs/agents-api/public-ports), an
  `https://<random>.agent37.app` URL that fills `tunnel_urls`. It serves HTTP, streaming responses and WebSocket with
  no platform authentication; the random hostname is the only credential.
- **`/health`:** Agent37's edge answers the path `/health` on every URL itself, so a service behind a public port never
  sees a request to exactly `/health`. Every other path reaches it.
- **Egress:** allow-all only. Agent37 can restrict a VM to a list of IPv4 addresses and ranges but not to hostnames, so
  an allowlist is refused with `NetworkPolicyUnsupportedError` before anything is created, and a fallback chain skips
  to its next provider.
- **Agent37's own hosts:** `*.agent37.app` is added to every other provider's allowlists, so a restricted agent
  elsewhere can reach an environment running here.
- **Image and object downloads:** with a store that signs URLs (S3, Cloud Storage), the VM fetches objects and image
  tarballs itself. Everything in the local stores goes over exec from this machine instead, which runs at this
  machine's upload speed.
- **Reconnect:** `get_sandbox` restores the ports and the applied egress policy.

## Attribution and cost

agent-env's attribution, including `[sandbox.attribution]` defaults, is stored as the instance's `metadata`, so the
dashboard and `GET /v1/instances` show which run each VM belongs to. VMs are billed per minute from the workspace's
prepaid balance while they run, and deleting one ends its billing.

## Cleanup

- A VM whose setup fails after it is created (Docker never readies) is deleted.
- If the caller is cancelled while the create is in flight, the provider deletes the VM that create makes once it
  answers.
- If a create fails after Agent37 may have made the VM, the provider finds it by its unique name and deletes it.
- `terminate` is idempotent: a VM that is already gone is not an error.

## Things to know

- **Commands run as root,** so the `sudo` agent-env prefixes is dropped.
- **Commands read no input:** each runs with an empty standard input.
- **Long commands:** Agent37 stops a command after 15 minutes, with exit code 137. Start anything longer in the
  background (`nohup … > log 2>&1 &`) and poll it.
- **Resends:** each command carries a random run name, and a connection lost mid-command is resent under it. Agent37
  answers a resend with the first run's result, waiting for it if it is still running, so a command never runs twice.
- **Lost connections:** when no resend gets an answer, `exec` returns exit code -1. `exec_script` retries idempotent
  scripts on -1 only.
- **Output:** exec returns at most 512 KB of each of stdout and stderr, and logs a warning when it cuts one. Copying a
  file off the VM to a store that cannot sign URLs reads it 4 MiB at a time as base64 through stdout, which is past that
  cap, so that copy fails rather than return a truncated file; use a store that signs URLs (S3, Cloud Storage) to
  collect snapshots and images built on the VM.
- **Images:** `create_vm(image=...)` is refused. VMs boot Agent37's `agent37-ubuntu` image (Ubuntu with Docker, Compose
  and buildx, with `dockerd` keeping its data under `/root/docker`), and OCI images run under Docker inside them.
- **Platform:** linux/amd64. GPUs are not supported.
- **The model key enters the sandbox:** an agent's `LITELLM_API_KEY` is passed in, as on other providers.

## Testing

`tst/integration/providers/sandbox_providers/agent37_sandbox_smoke_test.py` runs against a real Agent37 workspace when
`[sandbox.providers.agent37.config]` resolves, for example with `make int-test-slow`. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. The unit tests are
`tst/unit/providers/sandbox_providers/agent37_sandbox_provider_test.py` and `agent37_sandbox_test.py`, and run offline
against an in-memory Agent37 API behind an `httpx.MockTransport`.
