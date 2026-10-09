# Freestyle VM sandboxes

Select `freestyle` for environments, agents, or a fallback chain. The provider uses
the Freestyle v5 HTTP API through the existing `httpx` dependency; no extra SDK is needed.

```toml
[sandbox]
default = "freestyle"
agent_default = "freestyle"

[sandbox.providers.freestyle.config]
api_key = "env:FREESTYLE_API_KEY" # or secret:freestyle_api_key
snapshot_id = "freestyle/ubuntu"
# exec_timeout_seconds = 300
# api_url = "https://beta-api.freestyle.sh"
```

Get a key through the [Freestyle dashboard](https://dash.freestyle.sh) or the
[Freestyle CLI](https://www.freestyle.sh/docs/cli). Set it in your environment or secret
store; the provider resolves it from config and never injects it into the guest.
Then use `agent-env env deploy <env-id> --sandbox freestyle` or select the backend in config.

The snapshot must contain Bash, Docker, Docker Compose v2, curl, and the usual GNU
utilities. `freestyle/ubuntu` supplies those. To pin a prepared environment, use its
`sh-…` snapshot id; public snapshot names can be updated by Freestyle. `create_vm(image=…)`
selects another snapshot, and `boot_mode` overrides are rejected.

CPU requests round up to whole vCPUs, memory is MiB, and disk requests are converted
from GiB to MiB. A VM starts with its snapshot's resources; requested dimensions grow
it when larger. Requests below the snapshot's minimum retain that minimum, because
Freestyle resizing only grows resources. `timeout` becomes an absolute VM lifetime
(`ttlSeconds`), including provisioning. Always terminate completed sandboxes.

Each requested port receives its own public HTTPS `style.dev` URL through a TLS
rule created with the VM. Deleting the VM deletes those rules. Reconnection reads
the VM and its TLS rules to recover published ports; it reports an unknown network
policy rather than assuming that the firewall has not changed.

This adapter supports `allow_all` egress only. Freestyle's firewall matches addresses
and VM/network identities, while agent-env's allowlists include hostnames. Restricted
policies are rejected before provisioning; fallback chains can select another backend.
`*.style.dev` joins the framework's platform egress hosts so other restricted sandboxes
can reach a Freestyle environment. See the [firewall docs](https://www.freestyle.sh/docs/vms/network/firewall).

Commands run as root through the buffered exec API. Each command has a maximum
five-minute runtime (`exec_timeout_seconds`, 1–300); a guest timeout reports exit 124
and is not retried. Run longer services in Docker or as background guest processes.
Live stdout/stderr streaming is not supplied by this adapter. File writes use the
binary guest-filesystem API; shared VM helpers handle image artifacts and container files.
Attribution is forwarded as metadata and must fit Freestyle's 64-entry limit with
keys and values at most 63 characters; `freestyle.sh/` keys are reserved.

With credentials configured, run the real provider smoke test:

```bash
.venv/bin/python -m pytest tst/integration/providers/sandbox_providers/freestyle_sandbox_smoke_test.py -v
```

It creates a Docker container, checks HTTPS and container-file writes, verifies a
binary host-file checksum, reconnects, and confirms deletion. Its VM has a 900-second
TTL and is terminated in `finally`.
