# `boxd` sandbox provider

Runs agent-env sandboxes on [boxd](https://docs.boxd.sh) machines. Each environment and each agent gets its own
hardware-isolated Linux VM, with Docker running inside and public HTTPS URLs for its ports. It is a VM provider:
agent-env's docker-in-VM flows run on it unchanged. That covers the gateway's docker-compose, an agent's `docker run`,
image loading and artifact collection.

## Set up

### 1. Install the extra

The boxd SDK is an optional extra:

```bash
pip install 'agentenv-framework[boxd]'      # or: uv add 'agentenv-framework[boxd]'
```

Without it, selecting `boxd` fails with a `ConfigError` that names this extra. Nothing else needs it.

### 2. Get a boxd API key

Create a key with `boxd auth keys create agentenv --org <your-org>` or in the [console](https://app.boxd.sh). A key
belongs to one organization, and machines are created and billed there. For local work, export it:

```bash
export BOXD_API_KEY=bxd_...
```

For a shared deployment, put it in your secret store instead, for example as `boxd_api_key`, and reference it with
`secret:boxd_api_key`. Never put the key itself in `config.toml`. The provider reads the key only from its config,
never from the process environment.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on boxd, with the local stores:

```toml
[sandbox]
default       = "boxd"   # environments and sandboxes
agent_default = "boxd"   # agents

[sandbox.providers.boxd.config]
api_key = "env:BOXD_API_KEY"   # or "secret:boxd_api_key"

[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use boxd per run instead, add only the `[sandbox.providers.boxd.config]` table and pass
`--sandbox boxd`, or `--env-sandbox boxd` / `--agent-sandbox boxd` on `agent-env task run`. `agent-env config show`
prints the file in effect and masks the key.

### 4. Run something

The bundled `hello` task deploys a sandbox, loads a file into it and checks it, with no model needed:

```bash
agent-env run hello --sandbox boxd
```

`agent-env -v run …` also logs each machine as it starts
(`boxd machine started: machine_id=… name=… vcpu=… memory=… ports=… attribution=…`).

## Configuration reference

All keys go under `[sandbox.providers.boxd.config]`. Unknown keys are refused.

| Key        | Default                | Meaning                                                                                    |
| ---------- | ---------------------- | ------------------------------------------------------------------------------------------ |
| `api_key`  | required               | The boxd API key, as an `env:` or `secret:` reference.                                     |
| `org`      | the key's organization | The organization machines are created in. A key works in its own organization only.        |
| `base_url` | `https://boxd.sh:9443` | The boxd API endpoint.                                                                     |
| `snapshot` | none                   | Boot every machine from this boxd snapshot instead of boxd's image (see below).           |

## Faster startup

A machine booted from boxd's image builds or pulls every image an environment or agent needs. Boot it from a snapshot
of a machine that already has them, and those builds become cache hits: the `echo-mcp` bundle's `deploy_env` went from
about 117 s to 47 s, with its five image builds taking 12 s in all.

Keep one run's sandboxes up, clear the env machine down to its images, and save it:

```bash
agent-env -v run my-bundle --sandbox boxd --keep     # the "boxd machine started" line names the env's machine
boxd machine exec agentenv-… -- sudo sh -c 'docker rm -f $(docker ps -aq); docker volume prune -af; docker network prune -f'
boxd snapshots save agentenv-… agentenv-warm
```

Then stop the run with Ctrl-C and set `snapshot = "agentenv-warm"`. A snapshot holds the machine's memory and disk, so
clear its containers and volumes first, or every machine restores them, data included. The images it keeps are the
ones that machine built, so an agent whose image was built on another machine still builds it.

## Resources and lifetime

- **Sizes:** boxd has four: 1 vCPU / 4 GiB, 2 vCPU / 8 GiB, 4 vCPU / 16 GiB and 8 vCPU / 32 GiB. A request gets the
  smallest that covers both its CPU and its memory, so agent-env's default (1 vCPU, 8 GiB) runs on 2 vCPU / 8 GiB.
  With `snapshot`, every machine has the size the snapshot was saved at, and a request bigger than that is refused
  before anything is created.
- **Disk:** every machine has a fixed 100 GB copy-on-write disk.
- **Refused requests:** a request beyond 8 vCPU, 32 GiB or 100 GB is refused before anything is created, not shrunk.
- **Ignored requests:** `boot_mode` has no boxd equivalent, so a `boot_mode` that is set logs a warning and has no
  effect.
- **Lifetime:** the sandbox's `timeout` is the machine's delete-after timer, so boxd deletes it at that time even if
  a crashed run never tears it down.
- **Idle machines:** `auto_suspend_timeout` is 0, so a machine doesn't go to standby when idle. boxd's own idle
  hibernation (after 4 hours by default) keeps memory and processes, and the next exec or request wakes the machine.

## Networking

- **Ports:** each exposed port gets a subdomain proxy, `https://p<port>.<machine>.boxd.sh`, which fills
  `tunnel_urls`. It serves HTTP and WebSocket with no platform authentication. A machine restored from a snapshot
  starts with the snapshot's proxies; the provider removes the ones for ports it wasn't asked to expose.
- **Isolation:** every machine is created `isolated`. It has no peers on the organization's network, and boxd removes
  the in-machine `boxd` CLI, the metadata endpoint and account integrations from it. It keeps outbound internet unless
  an allowlist is set.
- **Egress:** allow-all, or an allowlist of hostnames, `*.` wildcards and public IPv4 addresses or ranges. boxd
  enforces it outside the machine, so one allowlist covers `dockerd` and every container. It is applied right after
  the machine is created, before anything runs in it. IPv6, private, loopback, link-local and multicast addresses are
  refused with `NetworkPolicyUnsupportedError` before anything is created, as is an entry with a scheme, port or path,
  and a fallback chain skips to its next provider. An allowlist boxd itself refuses, such as a wildcard on a
  shared-hosting domain like `*.amazonaws.com`, deletes the machine and raises the same error.
- **Wildcards:** `*.example.com` covers every name under `example.com`, at any depth, but not `example.com` itself.
- **boxd's own hosts:** `*.boxd.sh` is added to every allowlist, so a restricted agent can reach the environments it
  talks to.
- **Image and object downloads:** with a store that signs URLs (S3, Cloud Storage), the machine fetches objects and
  image tarballs itself. Those whose host the allowlist doesn't cover, and everything in the local stores, go over
  exec from this machine instead, which needs no egress but runs at this machine's upload speed. Image pulls and
  builds inside the machine still need their registries allowed.
- **Reconnect:** `get_sandbox` restores the ports and the applied allowlist.

## Attribution and cost

boxd machines have no labels, and a machine's name is its public DNS name, so attribution stays out of it: machines
are named `agentenv-<random>`. agent-env's attribution, including `[sandbox.attribution]` defaults, is in the
`boxd machine started` log line, keyed by machine id. `boxd machine list --json` lists the machines.

## Cleanup

- A machine whose setup fails after it is created (allowlist, ports, Docker readiness) is deleted.
- If the caller is cancelled while the create is in flight, the provider deletes the machine that create makes once
  it answers.
- If a create fails after boxd may have made the machine, the provider finds it by its unique name and deletes it.
- Whatever else happens, the delete-after timer removes the machine at the sandbox's `timeout`.
- `terminate` is idempotent: a machine that is already gone is not an error.

## Things to know

- **Commands run as the `boxd` user**, who has passwordless sudo and is in the `docker` group, so the `sudo`
  agent-env prefixes is kept.
- **Commands read no input:** each runs with standard input from `/dev/null`.
- **Quiet commands:** boxd stops relaying a command that prints nothing for 30 minutes, so each command runs beside a
  loop that writes a NUL byte to its stderr every minute. `exec` removes those bytes from what it returns.
- **Lost connections:** when a command's stream ends without an exit status, `exec` returns exit code -1.
  `exec_script` retries idempotent scripts on -1 only.
- **Images:** `create_vm(image=...)` is refused. Machines boot boxd's Ubuntu 24.04 image, or the configured
  `snapshot`, and OCI images run under Docker inside them.
- **Platform:** linux/amd64. GPUs are not supported.
- **The model key enters the sandbox:** an agent's `LITELLM_API_KEY` is passed in, as on other providers.

## Testing

`tst/integration/providers/sandbox_providers/boxd_sandbox_smoke_test.py` runs against a real boxd organization when
`[sandbox.providers.boxd.config]` resolves, for example with `make int-test-slow`. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. The unit tests are
`tst/unit/providers/sandbox_providers/boxd_sandbox_provider_test.py` and `boxd_sandbox_test.py`, and run offline
against in-memory stand-ins for the SDK.
