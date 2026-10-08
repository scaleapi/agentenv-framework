# `sail_vm` sandbox provider

Runs agent-env sandboxes on [Sail Research Sailboxes](https://docs.sailresearch.com/sailboxes). These are
Linux VMs booted from Sail's `devbox` image, which ships Docker and Compose v2 and runs as root. Like
`modal_vm` and `e2b`, it is a VM provider: agent-env's docker-in-VM flows run on it unchanged. That covers
the gateway's docker-compose, an agent's `docker run`, image loading and artifact collection.

## Set up

### 1. Install the extra

The Sail SDK is an optional extra:

```bash
pip install 'agentenv-framework[sail]'      # or: uv add 'agentenv-framework[sail]'
```

Without it, selecting `sail_vm` fails with a `ConfigError` that names this extra. Nothing else needs it.

### 2. Get a Sail API key

Create a key in the [Sail dashboard](https://app.sailresearch.com). For local work, export it:

```bash
export SAIL_API_KEY=sk_...
```

For a shared deployment, put it in your secret store instead, for example as `sail_api_key`, and reference
it with `secret:sail_api_key`. Never put the key itself in `config.toml`.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on Sailboxes, with the local stores:

```toml
[sandbox]
default       = "sail_vm"   # environments and sandboxes
agent_default = "sail_vm"   # agents

[sandbox.providers.sail_vm.config]
api_key = "env:SAIL_API_KEY"   # or "secret:sail_api_key"

# The model endpoint agents call. Sail injects its key into their requests (see below), so it must be HTTPS.
[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use Sail per run instead, add only the `[sandbox.providers.sail_vm.config]`
table and pass `--sandbox sail_vm`. `agent-env config show` prints the file in effect and masks the key.

### 4. Run something

The bundled `hello` task deploys a Sailbox, loads a file into it and checks it, with no model needed:

```console
$ agent-env run hello --sandbox sail_vm
[tasks/hello.json] step 1/3 box (deploy_sandbox)
[tasks/hello.json] step 1/3 box done in 2.4s
[tasks/hello.json] step 2/3 load (load_artifact)
[tasks/hello.json] step 3/3 hello (verify_sandbox)
[tasks/hello.json] passed in 3.1s

Tasks:
  tasks/hello.json v1: passed (hello: 1), 3.1s

Tore down 1 sandbox.
```

`agent-env -v run …` also logs each Sailbox as it starts
(`Sail VM sandbox started: sailbox_id=sb_… app=agent-env size=s …`).

## Configuration reference

All keys go under `[sandbox.providers.sail_vm.config]`. Unknown keys are refused.

| Key | Default | Meaning |
|---|---|---|
| `api_key` | required | The Sail API key, as an `env:` or `secret:` reference. |
| `app` | `"agent-env"` | The Sail App every Sailbox belongs to; Sail groups and bills by App. |
| `min_size` | `"s"` | The smallest Sailbox size to pick: `s`, `m` or `l`. |
| `auto_sleep` | `false` | Let Sail sleep an idle Sailbox; the first request after waking waits a few seconds. |
| `auto_sleep_min_idle_seconds` | unset | 1–3600 seconds of idleness before Sail may sleep a Sailbox; turns `auto_sleep` on. |
| `runtime_threads` | the SDK's own | The size of the SDK's network thread pool (1–256). |
| `inject_model_key` | `true` | Keep an agent's model key out of its Sailbox (see "The agent's model key"). |

A process uses one Sail API key: the SDK reads it from `SAIL_API_KEY` when it builds its process-wide
client. The provider sets that variable only for that one build and then restores it. Workloads never
see the key.

## Examples

**A shared deployment.** The key comes from the secret store, and costs are grouped under their own App:

```toml
[sandbox]
default       = "sail_vm"
agent_default = "sail_vm"
attribution   = { team = "env-pod", project_id = "env:PROJECT_ID?unassigned" }

[sandbox.providers.sail_vm.config]
api_key  = "secret:sail_api_key"
app      = "agent-env-prod"
min_size = "m"

[model]
base_url = "https://litellm.example.com"
api_key  = "secret:litellm_api_key"
```

**A per-run model key.** A run's override key is injected the same way as the configured one, so short-lived
per-run keys never reach a Sailbox either:

```bash
agent-env task run --id my-task --agent-sandbox sail_vm --env-sandbox sail_vm \
  --litellm-api-key "$RUN_SCOPED_KEY" --judge-litellm-api-key "$JUDGE_SCOPED_KEY"
```

**A sandbox with restricted egress.** In a task, `deploy_sandbox` and `deploy_agent` take a
`network_policy`; Sail enforces it for the VM and its containers:

```json
{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "sail_vm",
 "network_policy": {"mode": "allowlist", "allow_hosts": ["pypi.org", "*.github.com"], "allow_cidrs": ["10.0.0.0/8"]}}
```

**Long, mostly idle runs.** Let Sail sleep a Sailbox after 10 idle minutes; it wakes on traffic or a command:

```toml
[sandbox.providers.sail_vm.config]
api_key = "secret:sail_api_key"
auto_sleep_min_idle_seconds = 600
```

**An internal, non-HTTPS model endpoint.** Injection needs HTTPS, so pass the key into the Sailbox as other
providers do:

```toml
[sandbox.providers.sail_vm.config]
api_key = "secret:sail_api_key"
inject_model_key = false
```

**A fallback chain.** Try Sail first and fall back to E2B when a Sailbox can't be created in time:

```toml
[sandbox]
default = "sail_vm,e2b"
```

**From Python.** Build the configured provider and create a VM directly:

```python
import asyncio

from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider


async def main() -> None:
    provider = build_sandbox_provider("sail_vm")
    sandbox = await provider.create_vm(cpu=1, memory=2048, exposed_ports=[8080], timeout=900)
    try:
        print(await sandbox.exec_with_output("docker", "info", "--format", "{{.ServerVersion}}"))
        print(sandbox.tunnel_urls[8080])   # https://sb-<id>-8080.sail.box
    finally:
        await sandbox.terminate()


asyncio.run(main())
```

## Resources and lifetime

- **Size:** a Sailbox's size fixes its vCPU (`s`, `m`, `l` = 1, 4, 8). The provider picks the smallest
  size, no smaller than `min_size`, that covers the requested CPU.
- **Memory and disk:** these are ceilings, not reservations, since Sail bills observed usage. Requests
  are rounded up to whole GiB, into the size's range: memory 2–64, 8–128 or 16–256 GiB, and disk 8–128,
  32–512 or 64–1024 GiB.
- **Refused requests:** a request no size can meet is refused before anything is created.
- **Lifetime:** the sandbox's `timeout` is the Sailbox's hard maximum lifetime.

## Networking

- **Ports:** each exposed port gets a public URL, `https://sb-<id>-<port>.sail.box`, which fills
  `tunnel_urls`. It serves HTTP and WebSocket, with no platform authentication, like Modal's tunnels.
- **Egress:** allow-all, or an allowlist of hostnames, `*.domain` wildcards, IPv4 addresses and IPv4
  CIDRs, up to Sail's 128 entries. IPv6 entries are refused. The provider adds the hosts of signed
  download URLs to an allowlist before it loads images or objects.
- **Reconnect:** `get_sandbox` restores the ports and the applied egress policy. A policy it can't
  represent leaves `network_policy` unknown, and image loading then fails closed.

## The agent's model key

With `inject_model_key` on, an agent's `LITELLM_API_KEY` never enters its Sailbox:

1. The key is stored as a Sail secret named `AGENTENV_LITELLM_<sha256(key)[:32]>`, one per distinct key.
2. The Sailbox is created with a saved egress policy. On requests to `LITELLM_BASE_URL`'s host, the policy
   sets the `authorization: Bearer …` and `x-api-key` headers from that secret. Sail adds the key as each
   request leaves the box.
3. Inside the box, every command and file agent-env sends carries the placeholder
   `sail-injected-model-key` in place of the key.
4. A `docker` shim on the box gives every container the VM's CA bundle (`SSL_CERT_FILE`,
   `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`, `CURL_CA_BUNDLE`). Sail terminates TLS for the model host
   with its own CA, so containers need it to trust those requests.

Requirements and behaviour:

- **Endpoint:** the model endpoint must be HTTPS and reachable from Sail.
- **Key length:** keys shorter than 16 characters are refused, because scrubbing replaces the key
  wherever it appears.
- **Other keys are refused:** a Sailbox refuses any command or file that would carry a model key Sail
  doesn't inject for it. One example is an agent deployed into a sandbox it didn't create.
- **Turning it off:** set `inject_model_key = false` to pass keys in as other providers do.
- **Cleanup:** terminate deletes the Sailbox's policy. The secret is kept, so a concurrent launch with the
  same key never loses it. To remove secrets for retired keys, sweep `AGENTENV_LITELLM_*` with
  `sail secret list`.

## Attribution and cost

Sailboxes have no labels. The provider names each one `ae-<random>-<attribution values>`, which you can
find with the SDK's `Sailbox.list(search=...)`. It also logs one `agent_env.sail_vm_sandbox_started` event per box,
with `sailbox_id`, the App and the full attribution. To attribute cost, join Sail's per-Sailbox spend
(`GET /sailboxes/spend`) on `sailbox_id`.

## Things to know

- **Automatic checkpoints:** Sail checkpoints every Sailbox's disk for host-failure recovery, and this
  can't be turned off. Anything a workload's container env holds lands there, apart from the injected
  model key. Agent containers never receive the object store's credentials; with the S3 object store,
  keep `share_credentials` off for Sail runs too, so `snapshot_env` pushes none to env services.
- **Docker-in-Docker:** containers an agent starts with its own Docker-in-Docker don't get the CA bundle.
- **No `SAIL_MODE`:** the provider talks to Sail's production endpoints.

## Testing

`tst/integration/providers/sandbox_providers/sail_vm_sandbox_smoke_test.py` runs against a real Sail
account when `[sandbox.providers.sail_vm.config]` resolves. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. `gateway_test.py` and `task_steps_test.py` include a
`sail_vm` case.
