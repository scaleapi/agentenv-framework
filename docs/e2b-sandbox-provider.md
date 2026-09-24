# E2B sandbox provider

The built-in `e2b` provider runs agent-env workloads in an E2B VM with Docker
and Docker Compose. The E2B API key stays in the host process; it is not sent
to workload containers.

## Configuration

Add the E2B key to your configured secret store as `e2b_api_key`, then set:

```toml
[sandbox]
default = "e2b"       # environments
agent_default = "e2b" # agents

[sandbox.providers.e2b.config]
api_key = "secret:e2b_api_key"
base_template = "agent-env-docker-v2"
```

For local development, `api_key = "env:E2B_API_KEY"` can be used instead.
`base_template` must be an immutable, versioned E2B template containing Docker
Engine and the Docker Compose v2 plugin. Do not use a moving alias such as
`latest`.

Use `--sandbox e2b` to override the configured default for one command:

```bash
agent-env env deploy --id <env-id> --version <version> --sandbox e2b
agent-env a2a-agent deploy --id <agent-id> --version <version> --sandbox e2b
```

## Resources and templates

E2B fixes CPU and memory at template-build time. For each requested size,
agent-env deterministically resolves `<base>-<cpu>c-<memory>m`; for example,
`agent-env-docker-v2-2c-4096m`. An existing template is reused, otherwise it is
built from `base_template`. CPU must be a positive integer; memory is specified
in MB.

Disk size is not configurable through this provider. E2B determines disk
capacity, and agent-env does not forward `disk_size_gb`.

## Networking and lifecycle

Ingress URLs are public `*.e2b.app` endpoints. In `ALLOWLIST` mode, outbound
traffic uses E2B's deny-all rule (`deny_out = ["0.0.0.0/0"]`) plus the policy's
hostname and CIDR exceptions in `allow_out`. Agent-env includes `*.e2b.app` and
adds signed object-store hostnames when loading Docker images.

Sandbox reconnect restores exposed ports and the applied network policy. If
the policy cannot be reconstructed safely, image loading fails closed.
Artifact collection reads files over the sandbox command channel, as with
the other VM providers. Normal agent-env teardown terminates the E2B sandbox; callers creating a sandbox
directly remain responsible for calling `terminate()`.
