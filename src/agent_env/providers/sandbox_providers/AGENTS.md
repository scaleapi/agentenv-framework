# AGENTS.md: writing a sandbox provider

How to add a compute backend that agent-env runs its envs, agents and verifiers on. The repository-wide rules are in
the root `AGENTS.md`.

A sandbox provider hands agent-env a machine; agent-env then loads Docker images, starts containers, copies files in
and runs verifiers over that machine's shell. A provider is two classes:

- a `SandboxProvider` (`sandbox_provider.py`) that creates sandboxes and reconnects to them by id;
- a `Sandbox`, almost always a `VmSandbox` (`sandbox.py`), the handle to one machine: `exec`, `terminate`,
  `tunnel_urls`.

Read `boxd/` (the smallest complete VM provider) and `e2b/` before writing anything. `tensorlake/`, `vercel/` and `tenki/`
are further VM backends; `modal_sandbox.py` is a container backend; `local_sandbox.py` runs on this machine.

## Workflow

1. Ask which platform it is for. Pick a registry name: lowercase letters, digits and underscores (`acme_cloud`). It is the
   entry-point name, the `[sandbox.providers.<name>]` table and `Sandbox.type`.
2. Decide whether it ships in its own package or in this repository (below). For this repository, open an issue first, as
   `CONTRIBUTING.md` says.
3. Implement a VM backend with the platform's Python SDK.
4. Write unit tests with the SDK injected and no network.
5. The user puts credentials in the environment or the secrets file. Never ask for one in the conversation and never
   print one.
6. Run `agent-env plugin check` (out of tree), then `agent-env run hello --sandbox <name>`. If either fails and you
   cannot fix it, stop and show the output.
7. Report each item of the checklist as passed or failed, with command output.

## Pick a shape

| Shape | The platform gives you | You implement | You inherit |
|---|---|---|---|
| VM backend (recommended) | a Linux machine you can `exec` into, with Docker | `create_vm`, `create_sandbox` (a wrapper over `create_vm`), `get_sandbox`, a `VmSandbox` | `create_container` (create a VM, log in, pull, run), image loading and building, file staging |
| Container backend | an image booted directly, with a port exposed | `create_sandbox`, `get_sandbox`, a `Sandbox` with `exec` and the `write_file_*` methods | nothing |

Nothing on `SandboxProvider` is abstract, but registration still refuses a class that overrides neither `create_sandbox`
(container) nor `create_vm` (vm) with `must implement create_sandbox or create_vm`, through an entry point and through
config alike; a provider missing `get_sandbox` registers and fails on its first reconnect. `SUPPORTED_SANDBOX_MODES` is
derived from which of the two you override. A container backend cannot run envs that deploy onto a VM or
`deploy_sandbox` steps with `sandbox_mode: vm`.

A VM backend must override `create_sandbox` too, though registration does not ask for it: an agent's sandbox is made by
`create_sandbox`, whose base raises `NotImplementedError`, so a provider with only `create_vm` fails the first time an agent
is deployed on it. Most VM providers here do the same: `create_sandbox` ignores `image_name` and `env`, calls `create_vm`
with `exposed_ports=[port]` and returns the VM, and the deploy then loads and starts the image on it. See
`boxd/provider.py`. The rest of this file assumes a VM backend.

## Provider contract

| Method | Required | Called by |
|---|---|---|
| `create_vm(*, image, boot_mode, cpu, memory, disk_size_gb, timeout, exposed_ports, setup_for_gateway, attribution, network_policy) -> VmSandbox` | VM backend | env deploys, `deploy_sandbox` in VM mode, `create_container`, and your `create_sandbox` |
| `create_sandbox(*, image_name, port, env, cpu, memory, disk_size_gb, timeout, attribution, network_policy) -> Sandbox` | yes (a VM backend wraps `create_vm`) | an agent's sandbox (`provider.create_sandbox`); `create_container` of a container-only backend |
| `get_sandbox(sandbox_id) -> Sandbox` | yes | every step after the one that created the sandbox, and teardown |
| `terminate_sandbox(sandbox_id)` | no | teardown. The default calls `get_sandbox` then `terminate()`, and treats `SandboxNotFoundError` as done; override when you can terminate without a live handle |
| `create_container(...)` | inherited | containers that run an env or agent image. Override only when the platform boots images natively |
| `prepare_image(image, *, attribution)` | no | before a sandbox is asked for an image, outside any creation deadline; build a context-only image here, and pull one with an `image_digest` by it, building it from its context when it can't be pulled |
| `from_config(**config)` | no | receives `[sandbox.providers.<name>.config]` with `secret:` references resolved. Override to fail early with a `ConfigError` that names the missing key (`boxd` also rejects unknown keys) |
| `close()` | no | release SDK clients |

- Units: `cpu` in cores, `memory` in MiB, `disk_size_gb` in GB, `timeout` is the sandbox lifetime in seconds.
- `attribution` is an open `dict[str, str]` of cost-attribution tags. Pass it through `apply_default_attribution` and carry
  every key onto the sandbox as tags or metadata.
- A parameter the platform cannot honour is logged as ignored, never approximated. A request it cannot satisfy (an image it
  cannot boot, a size above its largest) raises `ValueError`; do not boot something else.
- `get_sandbox` may run in another process, on a provider built fresh from config. Whatever the handle needs that the
  platform cannot report back (exposed ports, applied network policy) must be stored with the sandbox at creation, in
  platform metadata or tags, and read back there. `e2b` keeps its exposed ports in sandbox metadata for this reason.
- A sandbox that is gone raises `SandboxNotFoundError` from `get_sandbox`.

### Declarations

Class variables the rest of agent-env reads instead of asking which provider it has. A provider that declares nothing has its
sandboxes off this machine, runs containers' images by name, and has no private network.

| Variable | Set it when |
|---|---|
| `ON_THIS_MACHINE`, `REACHES_HOST_AT` | the sandboxes run on this machine (local Docker, a local VM) |
| `CONTAINER_ACCEPTS` | `create_container` runs images other than by name: `Accepts.LOADABLE` (tar.gz, registry name or build context), `Accepts.NAME_OR_CONTEXT` |
| `SANDBOX_ACCEPTS` | an agent's sandbox runs different images than the default (a provider that implements `create_vm` defaults to `LOADABLE`) |
| `PRIVATE_NETWORK` | `create_container(private_network=True)` puts containers on a network they share |

## Sandbox handle contract

| Member | Contract |
|---|---|
| `type` | must equal the registry name. agent-env terminates a sandbox whose type differs and raises `SandboxProviderTypeError` |
| `sandbox_id` | the platform's id; `get_sandbox` must accept it |
| `mode` | `"vm"` for a VM sandbox |
| `tunnel_urls` | `{port: url}` for every exposed port, filled in by the time `create_*` returns. Each URL must be reachable from wherever agent-env and the agent run; `https://` for a remote backend |
| `vnc_url` | `None` unless the platform serves one |
| `network_policy` | the policy actually applied, or `None` when you cannot tell |
| `platform` | the image platform it runs; `linux/amd64` by default, set another when the provider creates it |
| `exec(*argv)` | returns an object with async `stdout.read()` and `stderr.read()` (bytes) and async `wait()` (exit code). If the platform takes one command string, build it with `shlex.join(argv)` |
| `terminate()` | abstract; idempotent: terminating a sandbox that is already gone is not an error |

Everything else is built on `exec`. `exec_script` runs `sudo bash -c <script>`, so the exec user needs passwordless sudo.
When the transport drops without an exit status, return exit code `-1`: callers retry idempotent commands on `-1` and on
nothing else.

## The machine

A VM sandbox must provide:

- a Docker daemon and the Docker Compose v2 plugin (`docker compose version` succeeds): the gateway starts every service of
  an env with `docker compose up`;
- `bash`, `curl`, `base64`, `tail`, `head` and passwordless sudo;
- outbound HTTPS to the configured object store, whose signed URLs deliver image tarballs and files (a store that cannot
  sign sends bytes over `exec`, which works but is slower);
- outbound access to the configured image registry, which `load_docker_images` and `create_container` pull from;
- inbound reachability of each exposed port through `tunnel_urls`.

`VmSandbox.wait_for_vm` polls until `exec` works and `/var/lib/docker` exists. `VmSandbox.setup_vm_for_gateway` calls it,
then opens each exposed port with `iptables`. Override both when the platform differs, as `e2b` does: it starts `dockerd`,
checks `docker compose version` up front so a bad base image fails at create time, and skips the firewall step.

## Lifecycle

1. A step calls `create_vm(exposed_ports=[...], timeout=..., attribution=...)`.
2. With `setup_for_gateway=True` (the default), `create_vm` calls `setup_vm_for_gateway` before it returns. If setup fails,
   terminate the sandbox, then re-raise.
3. agent-env gets the env's images onto the VM with `load_docker_images` (a tar.gz is loaded, a registry image pulled, a
   build context built), writes `docker-compose.yml` over `exec` and runs `docker compose up -d`.
4. `tunnel_urls[port]` goes to the agent.
5. Later steps, possibly in other processes, call `get_sandbox(sandbox_id)`.
6. Teardown calls `terminate_sandbox(sandbox_id)`.

## Network policy and reachability

| Hook | Default | Set it when |
|---|---|---|
| `supports_network_policy(policy)` | unrestricted egress only | the platform can enforce an outbound allowlist; apply it at create time and set `sandbox.network_policy` |
| `EGRESS_HOSTS` | none | tunnel URLs live on a domain (`*.acme.app`) that allowlisted workloads must still reach |
| `URL_REWRITES` | none | the platform issues internal URLs that callers outside it must reach by another host |
| `CONTAINER_ENV`, `REQUEST_HEADERS` | none | URLs need a credential: name it in `CONTAINER_ENV` (an `env:` or `secret:` reference) and attach it per host in `REQUEST_HEADERS` |
| `shares_network_with(sandbox_type)` | same provider only | sandboxes on another provider can reach yours directly |

When asked for a policy it cannot enforce, a provider refuses rather than creating an unrestricted sandbox: call
`refuse_unenforceable_policy(self, network_policy)` at the top of `create_vm`, and of `create_sandbox` when it does not delegate to `create_vm`. A fallback chain skips a provider whose
`supports_network_policy` says no.

## Register and configure

Out of tree, ship the provider in its own distribution; no fork is needed.

```toml
[project]
dependencies = ["agentenv-framework>=<oldest release you test against>"]

[project.entry-points."agent_env.sandbox_providers"]
acme = "agentenv_acme.provider:AcmeSandboxProvider"
```

```toml
# .agentenv/config.toml
[sandbox]
default = "acme"              # or "acme,local" for a fallback chain
agent_default = "acme"

[sandbox.providers.acme.config]
api_key = "secret:acme_api_key"
region = "us-east-1"
```

- Every key in the config table goes to `from_config` as a keyword argument.
- `secret:NAME` is resolved by the configured secret store; the default reads the environment variable `NAME`, or the same
  key in its secrets file. Credentials reach the provider only this way; it never reads `os.environ`.
- The entry-point name, the config table name and `Sandbox.type` are one string.
- `agent-env plugin list` and `agent-env plugin check` show whether the provider loaded and why not.
- While developing, skip packaging with `impl = "agentenv_acme.provider:AcmeSandboxProvider"` in
  `[sandbox.providers.acme]`.

In this repository, a built-in also needs: its package under `providers/sandbox_providers/<name>/` (`provider.py`,
`sandbox.py`); an entry in `_BUILTIN_SANDBOX_PROVIDERS` (`sandbox_provider.py`); an optional-dependency extra in
`pyproject.toml` for the SDK, added to the `dev` extra so CI exercises it, with the SDK imported only where it is used; and
a line for it in the root `AGENTS.md` provider lists. `network_policy_stamping_test.py` fails until you add a factory for
the new one to its `FACTORIES`; also add a row to the table in `provider_capabilities_test.py`, which nothing forces. Say in
the PR what the website docs should cover.

## Tests

1. Unit tests, SDK injected, no network, in `tst/unit/providers/sandbox_providers/`. `E2BSandboxProvider` takes its SDK
   class as a constructor argument, and `boxd_fakes.py` and `tenki_fakes.py` show fake SDKs. Cover `create_vm`, `get_sandbox`
   from a fresh provider, idempotent `terminate` and the network-policy paths.
2. The smallest live check: `agent-env run hello --sandbox <name>`.
3. A live smoke test in `tst/integration/providers/sandbox_providers/`, modelled on `boxd_sandbox_smoke_test.py`: create a
   VM, run a container on an exposed port and reach it through `tunnel_urls`, reconnect with `get_sandbox` from a fresh
   provider and check ports and policy survive, then terminate twice. In this repository, add `<name>` to `remote_sandbox_is_available`
   in `tst/util/capabilities.py` first (it raises `ValueError` for an unknown name, which stops collection), then gate the
   test with `skip_without_remote_sandbox("<name>")` so it declares its skip instead of failing.
4. An env deploy, last: it exercises the gateway, image loading and `docker compose`, so it needs a gateway image and a
   registry your sandboxes can pull from.

## Checklist

- [ ] `create_vm`, `create_sandbox` (wrapping `create_vm`) and `get_sandbox` are implemented and `Sandbox.type` equals the registry name.
- [ ] A sandbox that fails setup is terminated before the error is re-raised.
- [ ] `get_sandbox` on a freshly built provider restores exposed ports and applied network policy.
- [ ] `terminate` is idempotent and a gone sandbox raises `SandboxNotFoundError` from `get_sandbox`.
- [ ] `docker compose version` succeeds on the base image and a container on an exposed port is reachable through
      `tunnel_urls`.
- [ ] Ignored parameters are logged, unsatisfiable requests rejected, unenforceable policies refused.
- [ ] Credentials reach the provider only through `from_config`.
- [ ] `agent-env plugin check` passes and `agent-env run hello --sandbox <name>` succeeds.
