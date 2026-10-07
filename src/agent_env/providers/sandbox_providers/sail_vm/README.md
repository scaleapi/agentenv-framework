# `sail_vm` sandbox provider

Runs agent-env sandboxes on [Sail Research Sailboxes](https://docs.sailresearch.com/sailboxes). These are
Linux VMs booted from Sail's `devbox` image, which ships Docker and Compose v2 and runs as root. Like
`modal_vm` and `e2b`, it is a VM provider: agent-env's docker-in-VM flows run on it unchanged. That covers
the gateway's docker-compose, an agent's `docker run`, image loading and artifact collection.

## Install

The Sail SDK is an optional extra:

```bash
pip install 'agentenv-framework[sail]'
```

Without it, selecting `sail_vm` fails with a `ConfigError` that names this extra. Nothing else needs it.

## Configure

Put the Sail API key in your secret store, then add the provider to `.agentenv/config.toml`:

```toml
[sandbox.providers.sail_vm.config]
api_key = "secret:sail_api_key"   # required; env:SAIL_API_KEY for local development
app = "agent-env"                 # the Sail App every Sailbox belongs to
min_size = "s"                    # the smallest Sailbox size to pick: s, m or l
auto_sleep = false                # let Sail sleep idle Sailboxes; off by default
# auto_sleep_min_idle_seconds = 600   # 1-3600; turns auto_sleep on
# runtime_threads = 16                # the SDK's network thread pool (1-256)
inject_model_key = true           # keep an agent's model key out of its Sailbox (below)
```

Use it for a run with `agent-env run <bundle> --sandbox sail_vm`, or by default with:

```toml
[sandbox]
default = "sail_vm"
agent_default = "sail_vm"
```

A process uses one Sail API key: the SDK reads it from `SAIL_API_KEY` when it builds its process-wide
client. The provider sets that variable only for that one build and then restores it. Workloads never
see the key.

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
  model key. With the S3 object store, keep `share_credentials` off for Sail runs.
- **Docker-in-Docker:** containers an agent starts with its own Docker-in-Docker don't get the CA bundle.
- **No `SAIL_MODE`:** the provider talks to Sail's production endpoints.

## Testing

`tst/integration/providers/sandbox_providers/sail_vm_sandbox_smoke_test.py` runs against a real Sail
account when `[sandbox.providers.sail_vm.config]` resolves. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. `gateway_test.py` and `task_steps_test.py` include a
`sail_vm` case.
