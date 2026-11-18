# `kosa8` sandbox provider

Runs agent-env sandboxes on [kosa8 Cloud](https://kosa8.com). Each environment and each agent gets its own microVM,
booted with Docker inside and a public HTTPS URL for each of its ports. It is a VM provider: agent-env's docker-in-VM
flows run on it unchanged. That covers the gateway's docker-compose, an agent's `docker run`, image loading and
artifact collection.

## Set up

### 1. No extra to install

The provider drives kosa8's REST API with `httpx`, which agent-env already depends on.

### 2. Get a kosa8 API key

Create a key in the [dashboard](https://app.kosa8.com) under API keys, with the `sandboxes:read` and `sandboxes:write`
scopes. Sandboxes booted at a size, as these are, need a payment method on the organization. For local work, export
it:

```bash
export KOSA8_API_KEY=...
```

For a shared deployment, put it in your secret store instead, for example as `kosa8_api_key`, and reference it with
`secret:kosa8_api_key`. Never put the key itself in `config.toml`. The provider reads the key only from its config,
never from the process environment.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on kosa8, with the local stores:

```toml
[sandbox]
default       = "kosa8"   # environments and sandboxes
agent_default = "kosa8"   # agents

[sandbox.providers.kosa8.config]
api_key = "env:KOSA8_API_KEY"   # or "secret:kosa8_api_key"

[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use kosa8 per run instead, add only the `[sandbox.providers.kosa8.config]` table and
pass `--sandbox kosa8`, or `--env-sandbox kosa8` / `--agent-sandbox kosa8` on `agent-env task run`.

### 4. Run something

```bash
agent-env run hello --sandbox kosa8
```

`agent-env -v run …` also logs each sandbox as it starts
(`kosa8 sandbox started: id=… name=… size=… disk_gb=… ttl_s=… ports=… attribution=…`).

## Configuration reference

All keys go under `[sandbox.providers.kosa8.config]`. Unknown keys are refused.

| Key        | Default                                    | Meaning                                                                    |
| ---------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| `api_key`  | required                                   | The kosa8 API key, as an `env:` or `secret:` reference.                    |
| `base_url` | `https://api.kosa8.com`                    | The kosa8 control plane.                                                   |
| `region`   | the organization's default                 | `us`, `eu`, or a region such as `us-ash1` or `eu-hel1`.                    |
| `image`    | `kosa8-template/universal-24.04-20261115.2` | The sandbox's own image. It must ship `dockerd` and the Compose v2 plugin. |

## Resources and lifetime

- **Sizes:** kosa8 has four: 2 vCPU / 4 GiB, 4 / 8, 8 / 16 and 16 / 32. A request gets the smallest that covers both
  its CPU and its memory, so agent-env's default (1 vCPU, 8 GiB) runs on 4 vCPU / 8 GiB. A larger request is refused
  before anything is created, not shrunk.
- **Disk:** the request's disk, at least 32 GB (kosa8's default, which holds the images Docker pulls) and at most
  200 GB; a larger request is refused.
- **Ignored requests:** `boot_mode` and `env` have no kosa8 equivalent here; one that is set logs a warning.
- **Lifetime:** the sandbox's `timeout` is its `ttl_s` (60 s to 90 days), so kosa8 deletes it then even if a crashed
  run never tears it down.
- **Sleep:** `sleep_after_s` is 0, so a sandbox never sleeps while agent-env uses it.

## Networking

- **Ports:** each exposed port is a public kosa8 preview, `https://<port>-<random>.kosa8labs.com`, which fills
  `tunnel_urls`. It serves HTTP and WebSocket with no platform authentication, like other providers' tunnels; the
  random part is the only secret.
- **Egress:** allow-all, or an allow list of at most 256 hostnames, `*.` wildcards and IPv4 addresses or ranges. kosa8
  enforces it on the host, outside the VM, so one list covers `dockerd` and every container. The list goes with the
  create, so it is in force before kosa8 answers and the sandbox never runs more open than asked. IPv6 entries are
  refused with `NetworkPolicyUnsupportedError` before anything is created; a list kosa8 itself refuses (or an
  organization policy that does not allow it) fails the create with the same error, so a fallback chain moves on.
- **Wildcards:** `*.example.com` covers every name under `example.com` but not `example.com` itself.
- **kosa8's own hosts:** `*.kosa8labs.com` is added to every allow list, so a restricted agent can reach the
  environments it talks to.
- **Organization policy:** an organization's default egress policy applies to sandboxes that set none, and the
  sandbox's `network_policy` reports what kosa8 enforces. A deny-only policy reads as allow-all.
- **Reconnect:** `get_sandbox` restores the ports and the applied policy from kosa8.

## Cleanup

- A sandbox whose setup fails after it is created (boot, ports, Docker readiness) is deleted.
- If a create fails after kosa8 may have made the sandbox, or its caller is cancelled, the provider finds it by its
  unique `agentenv-<random>` name and deletes it.
- Whatever else happens, `ttl_s` removes the sandbox at the sandbox's `timeout`.
- `terminate` is idempotent: a sandbox that is already gone is not an error.

## Things to know

- **Commands run as root**, so the `sudo` agent-env prefixes is dropped, and read no input (standard input is
  `/dev/null`).
- **Long commands:** exec is streamed, which no time limit holds; kosa8 sends a heartbeat every 15 s while a command
  is quiet.
- **Lost connections:** when a command's stream ends without an exit status, `exec` returns exit code -1.
  `exec_script` retries idempotent scripts on -1 only. A call kosa8 refused before the command started (`409
  starting`, `429`, `503`) is resent.
- **Large commands:** kosa8 takes at most 256 arguments of 8 KiB each. A longer command is uploaded to
  `/workspace/.agentenv` through the files API and read back as its argument list.
- **Files:** `write_host_file` uploads through the files API, which reaches only `/workspace`, in 4 MiB pieces, then
  moves the bytes into place, so a secret never enters a command line.
- **Images:** `create_vm(image=...)` is refused. Sandboxes boot the configured image, and OCI images run under Docker
  inside them.
- **Platform:** linux/amd64. GPUs are not supported.
- **The model key enters the sandbox:** an agent's `LITELLM_API_KEY` is passed in, as on other providers.

## Testing

`tst/integration/providers/sandbox_providers/kosa8_sandbox_smoke_test.py` runs against a real kosa8 organization when
`[sandbox.providers.kosa8.config]` resolves. Otherwise it skips with `agentenv-capability-missing: remote_sandbox`.
The unit tests are `tst/unit/providers/sandbox_providers/kosa8_sandbox_provider_test.py` and `kosa8_sandbox_test.py`,
and run offline against an in-memory kosa8 API behind `httpx.MockTransport`.
