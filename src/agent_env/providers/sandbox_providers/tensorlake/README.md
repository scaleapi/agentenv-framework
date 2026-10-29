# `tensorlake` sandbox provider

Runs agent-env sandboxes on [Tensorlake](https://tensorlake.ai) sandboxes: MicroVMs booted from a
Docker-capable host image. Like `sail_vm`, `modal_vm` and `e2b`, it is a VM provider: agent-env's
docker-in-VM flows run on it unchanged. That covers the gateway's docker-compose, an agent's `docker run`,
image loading and artifact collection.

## Set up

### 1. Install the extra

The Tensorlake SDK is an optional extra:

```bash
pip install 'agentenv-framework[tensorlake]'      # or: uv add 'agentenv-framework[tensorlake]'
```

Without it, selecting `tensorlake` fails with a `ConfigError` that names this extra. Nothing else needs it.

### 2. Get a Tensorlake API key

Create a key in the Tensorlake dashboard. For local work, export it:

```bash
export TENSORLAKE_API_KEY=...
```

For a shared deployment, put it in your secret store instead, for example as `tensorlake_api_key`, and
reference it with `secret:tensorlake_api_key`. Never put the key itself in `config.toml`. The provider
reads the key only from its config, never from the process environment.

### 3. Configure

A complete `.agentenv/config.toml` that runs every environment and agent on Tensorlake, with the local stores:

```toml
[sandbox]
default       = "tensorlake"   # environments and sandboxes
agent_default = "tensorlake"   # agents

[sandbox.providers.tensorlake.config]
api_key = "env:TENSORLAKE_API_KEY"   # or "secret:tensorlake_api_key"

[model]
base_url = "https://litellm.example.com"
api_key  = "env:LITELLM_API_KEY"
```

To keep the local default and use Tensorlake per run instead, add only the
`[sandbox.providers.tensorlake.config]` table and pass `--sandbox tensorlake`. `agent-env config show` prints
the file in effect and masks the key.

### 4. Run something

The bundled `hello` task deploys a sandbox, loads a file into it and checks it, with no model needed:

```bash
agent-env run hello --sandbox tensorlake
```

`agent-env -v run …` also logs each sandbox as it starts
(`Tensorlake sandbox started: sandbox_id=… image=agentenv-dind-host-v1 attribution=…`).

## Configuration reference

All keys go under `[sandbox.providers.tensorlake.config]`. Unknown keys are refused.

| Key | Default | Meaning |
|---|---|---|
| `api_key` | required | The Tensorlake API key, as an `env:` or `secret:` reference. |
| `image` | `"agentenv-dind-host-v1"` | A registered Docker-capable Tensorlake image to boot from. |
| `api_url` | `"https://api.tensorlake.ai"` | The Tensorlake API. Every SDK call uses it, whatever `TENSORLAKE_API_URL` says. |

## The host image

Sandboxes boot from `agentenv-dind-host-v1` by default, a public image built from `host_image.Dockerfile`
next to this file: `tensorlake/ubuntu-systemd` with Docker Engine and the Compose v2 plugin. Tensorlake
runs `dockerd` only as the systemd `docker.service`, so the provider starts it with `systemctl` and does
not start `dockerd` by hand.

To use your own image, publish it once and set `image`:

```bash
tl sbx image create host_image.Dockerfile -n my-docker-host \
  --disk_mb 30720 --builder_disk_mb 24576 --docker_compat
```

An image that is not registered fails the create with a `ValueError` that contains this command. The
provider never boots a different image. A changed Dockerfile is
published under a new version, never over the old one.

## Resources and lifetime

- **CPU and memory:** Tensorlake hosts run without KVM, so cores are slower than the request suggests.
  The provider raises a request to at least 2 CPUs, 4 GiB of memory and 1 GiB of memory per CPU. More
  than 8 GiB of memory per CPU is refused.
- **Disk:** at least 30 GiB, the disk the host image is published with, and at most 100 GiB.
- **Changed or ignored requests:** a raised CPU, memory or disk request logs a warning. `boot_mode` has no
  Tensorlake equivalent, so a `boot_mode` that is set logs a warning and has no effect.
- **Refused requests:** a request Tensorlake can't meet is refused before anything is created.
- **Lifetime:** the sandbox's `timeout` is Tensorlake's idle timeout. After it, the sandbox terminates.

## Networking

- **Ports:** each exposed port gets a public URL, `https://<port>-<sandbox id>.sandbox.tensorlake.ai`,
  which fills `tunnel_urls`. It has no platform authentication, like Modal's tunnels. A Bearer header
  would hand the Tensorlake API key to every workload that calls the URL.
- **Egress:** allow-all, or an allowlist of hostnames, IPv4 addresses and IPv4 CIDRs. A policy with IPv6
  entries is refused with `NetworkPolicyUnsupportedError` before anything is created, and a fallback chain
  skips to its next provider. Tensorlake applies the policy outside the VM, so one policy covers `dockerd` and every
  container. Tensorlake refuses a policy update with a host that does not resolve.
- **Image downloads:** before it loads images under an allowlist, the provider adds the hosts of the
  signed download URLs, then waits up to 60 s for the change to reach the host firewall.
- **Reconnect:** `get_sandbox` restores the ports and the applied egress policy. A policy it can't
  represent leaves `network_policy` unknown, and image loading then fails closed.

## Attribution and cost

Tensorlake sandboxes have no labels. The provider logs one `agent_env.tensorlake_sandbox_started` event per
sandbox, with `tensorlake_sandbox_id`, the full attribution (including `[sandbox.attribution]` defaults),
`cpus`, `memory_mb` and `disk_mb`. To attribute cost, join Tensorlake's per-sandbox usage on the sandbox id.

## Cleanup

- A sandbox that fails to start is deleted.
- A sandbox whose setup fails after it starts (ports, Docker) is terminated.
- If the caller is cancelled while the create request is in flight, the provider deletes the sandbox
  that request makes, once it answers.
- `terminate` is idempotent: a sandbox that is already gone is not an error.

## Things to know

- **Commands run as `tl-user`:** agent-env's `sudo` prefixes stay, and file uploads land in `/tmp` first,
  then a root command joins them at the target path.
- **Lost connections:** when the command stream ends without an exit status, `exec` returns exit code -1.
  `exec_script` retries idempotent scripts on -1 only.
- **Uploads:** the ingress proxy refuses request bodies above about 4 MB, so uploads go through the file
  API in 512 KiB parts, 8 at a time, and the joined file is checked against its sha256. An image tarball
  from an object store that can't sign URLs, such as the local default, is streamed the same way.
- **The model key enters the sandbox:** an agent's `LITELLM_API_KEY` is passed in, as on other providers.

## Testing

`tst/integration/providers/sandbox_providers/tensorlake_sandbox_smoke_test.py` runs against a real
Tensorlake account when `[sandbox.providers.tensorlake.config]` resolves. Otherwise it skips with
`agentenv-capability-missing: remote_sandbox`. The unit tests are
`tst/unit/providers/sandbox_providers/tensorlake_sandbox_provider_test.py` and `tensorlake_sandbox_test.py`.
