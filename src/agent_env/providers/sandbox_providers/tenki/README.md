# `tenki` sandbox provider

Runs agent-env sandboxes on [Tenki Sandbox](https://tenki.cloud/docs/sandbox) VMs. Each environment and each agent
gets its own Linux VM, with Docker inside and public HTTPS URLs for its ports. It is a VM provider: agent-env's
docker-in-VM flows run on it unchanged. That covers the gateway's docker-compose, an agent's `docker run`, image loading
and artifact collection.

## Set up

### 1. Install the extra

The Tenki SDK is an optional extra:

```bash
pip install 'agentenv-framework[tenki]'      # or: uv add 'agentenv-framework[tenki]'
```

Without it, selecting `tenki` fails with a `ConfigError` that names this extra. Nothing else needs it.

### 2. Get a Tenki API key

Create a key as described in [API keys](https://tenki.cloud/docs/account/api-keys). For local work, export it:

```bash
export TENKI_API_KEY=tk_...
```

For a shared deployment, put it in your secret store instead, for example as `tenki_api_key`, and reference it with
`secret:tenki_api_key`. Never put the key itself in `config.toml`. The provider reads the key only from its config,
never from the process environment.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on Tenki, with the local stores:

```toml
[sandbox]
default       = "tenki"   # environments and sandboxes
agent_default = "tenki"   # agents

[sandbox.providers.tenki.config]
api_key = "env:TENKI_API_KEY"   # or "secret:tenki_api_key"

[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use Tenki per run instead, add only the `[sandbox.providers.tenki.config]` table and
pass `--sandbox tenki`, or `--env-sandbox tenki` / `--agent-sandbox tenki` on `agent-env task run`.
`agent-env config show` prints the file in effect and masks the key.

### 4. Run something

The bundled `hello` task deploys a sandbox, loads a file into it and checks it, with no model needed:

```bash
agent-env run hello --sandbox tenki
```

`agent-env -v run …` also logs each VM as it starts
(`Tenki VM started: session_id=… cpu=… memory=…MB disk=…GB ports=… attribution=…`).

## Configuration reference

All keys go under `[sandbox.providers.tenki.config]`. Unknown keys are refused.

| Key           | Default                   | Meaning                                                                                                     |
| ------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `api_key`     | required                  | The Tenki API key, as an `env:` or `secret:` reference.                                                     |
| `base_url`    | `https://api.tenki.cloud` | The Tenki API endpoint.                                                                                     |
| `snapshot_id` | none                      | Boot every VM from this snapshot, for example one with Docker already installed (see below).                |
| `image`       | the Tenki base image      | Boot every VM from this Tenki registry image instead. At most one of `image` and `snapshot_id`.             |
| `min_disk_gb` | `20`                      | The smallest VM disk. Docker keeps compressed and extracted layers, so agent-env's 10 GB default is raised. |

## Faster startup

A VM booted from the Tenki base image installs Docker and Compose v2 the first time agent-env uses it. Snapshot a VM
that has them once, and every VM booted from the snapshot skips that step:

```bash
tenki sandbox create --name agentenv-docker --cpu 2 --memory-mb 4096 --disk-size-gb 20 --max-duration 30m
tenki sandbox exec --session agentenv-docker -c 'sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io docker-compose-v2 docker-buildx curl'
tenki sandbox exec --session agentenv-docker -c 'sudo systemctl start docker && sudo docker info --format "{{.ServerVersion}}"'
tenki sandbox snapshot create agentenv-docker --name agentenv-docker --wait-durable
tenki sandbox terminate agentenv-docker
```

Then set `snapshot_id` to the snapshot's id. A VM can be restored larger than its snapshot but not smaller, so keep
the snapshot's VM at 2 vCPU and 4 GiB, the smallest the provider starts. Images your environments and agents pull
from a registry can be pulled into the same VM before the snapshot; a VM booted from it then finds them in place.

## Resources and lifetime

- **CPU and memory:** whole vCPUs, memory rounded up to 2 MiB. A request below 2 vCPU or 4 GiB is raised to it, the
  smallest VM Tenki starts from a warm snapshot.
- **Disk:** at least `min_disk_gb`.
- **Refused requests:** a request beyond Tenki's limits (128 vCPU, 512 GiB memory, 100 GB disk, or the workspace's own
  lower limit) is refused before anything is created, not shrunk.
- **Ignored requests:** `boot_mode` has no Tenki equivalent, so a `boot_mode` that is set logs a warning and has no
  effect.
- **Lifetime:** the sandbox's `timeout` is the VM's hard maximum lifetime, so a crashed run can't leave a VM running
  past it. It is also every command's execution budget. A workspace with a lower maximum session duration grants that
  instead; the provider logs a warning and budgets commands by the lifetime granted.

## Networking

- **Ports:** each exposed port gets a public HTTPS URL under `sb.tenki.sh`, which fills `tunnel_urls`. It serves HTTP
  and WebSocket with no platform authentication.
- **Egress:** allow-all, or an allowlist of up to 256 hostnames and 64 IPv4 CIDRs. An IP address listed as a host
  becomes a /32. A policy with IPv6 or malformed entries is refused with `NetworkPolicyUnsupportedError` before
  anything is created, and a fallback chain skips to its next provider. Tenki applies the policy outside the VM, so
  one policy covers `dockerd` and every container.
- **Wildcards:** `*.example.com` covers exactly one label: `files.example.com`, not `a.b.example.com`. List deeper names
  explicitly.
- **Tenki's own hosts:** `*.sb.tenki.sh` and `*.us.sb.tenki.sh` are added to every allowlist, so a restricted agent can
  reach the environments it talks to. A workspace with wildcard preview domains gets URLs one label deeper
  (`<token>.<workspace>.us.sb.tenki.sh`); the provider adds `*.<workspace>.us.sb.tenki.sh` once it has exposed a port
  there. A restricted VM created before any environment in the run needs that entry in its own `allow_hosts`.
- **Image and object downloads:** Tenki sets a VM's egress when it is created. Objects and image tarballs whose host the
  allowlist doesn't cover go over the exec's stdin instead, which needs no egress. Image pulls and builds inside the
  VM still need their registries allowed, and so does installing Docker on the base image, so boot allowlisted VMs
  from a snapshot that has Docker.
- **Reconnect:** `get_sandbox` restores the ports and the applied egress policy.

## Attribution and cost

Every VM is tagged `agentenv` and carries agent-env's attribution, including `[sandbox.attribution]` defaults, in its
session metadata. `tenki sandbox list --tags agentenv` lists them, and `tenki sandbox get <session id> --json` shows a
VM's metadata.

## Cleanup

- A VM whose setup fails after it is created (readiness, ports, Docker) is terminated.
- If the caller is cancelled while the create is in flight, the provider terminates the VM that create makes once it
  answers.
- If a create fails after Tenki may have made the VM, the provider finds that VM by its unique name and terminates it.
- `terminate` is idempotent: a VM that is already gone or terminating is not an error.

## Things to know

- **Commands run as root** through Tenki's privileged exec, so the `sudo` agent-env prefixes is dropped.
- **Lost connections:** when a command's stream ends without an exit status, `exec` returns exit code -1.
  `exec_script` retries idempotent scripts on -1 only.
- **Images:** `create_vm(image=...)` is refused. VMs boot the configured `image` or `snapshot_id`, and OCI images run
  under Docker inside them.
- **GPUs** are not supported.
- **The model key enters the sandbox:** an agent's `LITELLM_API_KEY` is passed in, as on other providers.

## Testing

`tst/integration/providers/sandbox_providers/tenki_sandbox_smoke_test.py` runs against a real Tenki workspace when
`[sandbox.providers.tenki.config]` resolves, for example with `make int-test-slow`. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. The unit tests are
`tst/unit/providers/sandbox_providers/tenki_sandbox_provider_test.py` and `tenki_sandbox_test.py`, and run offline
against in-memory stand-ins for the SDK.
