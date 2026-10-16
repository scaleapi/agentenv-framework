<h1><img alt="AgentEnv Framework" src="https://raw.githubusercontent.com/scaleapi/agentenv-framework/main/assets/brand/readme-banner.png"></h1>

[![PyPI version](https://img.shields.io/pypi/v/agentenv-framework)](https://pypi.org/project/agentenv-framework/)
[![Python versions](https://img.shields.io/pypi/pyversions/agentenv-framework)](https://pypi.org/project/agentenv-framework/)
[![License](https://img.shields.io/pypi/l/agentenv-framework)](https://github.com/scaleapi/agentenv-framework/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/scaleapi/agentenv-framework/local-backends.yml?branch=main&label=CI)](https://github.com/scaleapi/agentenv-framework/actions/workflows/local-backends.yml)

AgentEnv Framework is a Python SDK and CLI for building, deploying and running agentic environments and the tasks that grade agents inside them. Environments are containerized servers that speak the open `agentenv-framework-protocol`; AgentEnv Framework builds them into versioned images, deploys them behind a gateway, points an agent at them, and scores what the agent did.

**Documentation: [www.agentenvframework.com/docs](https://www.agentenvframework.com/docs)** covers environments, artifacts, agents, tasks, the registry and plugins.

## Install

Both packages are on PyPI. You need Python 3.11.4 or newer and, to run environments locally, a running Docker daemon. With [uv](https://docs.astral.sh/uv/), install the `agent-env` command as a tool:

```bash
uv tool install agentenv-framework
agent-env run hello
```

`agent-env run hello` runs the built-in hello task, which needs no Docker, model or configuration. `agent-env run echo-mcp` needs Docker: it builds an MCP server from a Dockerfile, deploys it through the gateway and checks it by calling its tool. `agent-env run` lists both. `agent-env bundle init <folder> --from-task <id>` writes a task a store holds as a bundle folder, with each reference it makes pinned at the version the store holds, so you can run and edit it locally. To try hello without installing anything, run it with uvx:

```bash
uvx --from agentenv-framework agent-env run hello
```

Or install it with pip, into a virtualenv:

```bash
pip install agentenv-framework
```

To use the SDK in your own project, add it as a dependency with `uv add agentenv-framework`. The explorer, `agent-env up`, needs an `.agentenv/config.toml` in the current folder or above it; an empty one keeps every local default. The `tensorlake` sandbox backend needs the `tensorlake` extra (`pip install 'agentenv-framework[tensorlake]'`) and an `api_key` in `[sandbox.providers.tensorlake.config]`. Its sandboxes boot from the public image `agentenv-dind-host-v1`; set `image` there to use another registered Docker-capable image. The `tenki` sandbox backend needs the `tenki` extra (`pip install 'agentenv-framework[tenki]'`) and an `api_key` in `[sandbox.providers.tenki.config]`; see [its README](src/agent_env/providers/sandbox_providers/tenki/README.md).

A plain install runs on local stores and needs no cloud SDK. The cloud store backends are extras: `aws` for S3, Secrets Manager, DynamoDB and ECR (`pip install 'agentenv-framework[aws]'`), and `gcp` for Cloud Storage, Secret Manager, Firestore and Artifact Registry (`pip install 'agentenv-framework[gcp]'`). A config that names a backend without its extra fails, naming the extra to install.

The distribution is named `agentenv-framework`, the import package is `agent_env` and the command is `agent-env`. It depends on `agentenv-framework-protocol`, whose import package is `agentenv_protocol`, and installs it too.

### Sandbox providers

For a complete evaluation using your existing Vercel login, follow the [Vercel Sandbox demo](https://github.com/vercel-labs/agentenv-framework/tree/vercel-demo/examples/vercel-sandbox).

The default provider runs environments and agents as containers on the local Docker daemon. The `vercel` provider runs each environment or agent in an ephemeral, Docker-capable Vercel Sandbox and exposes declared ports through public `*.vercel.run` routes. From this source checkout, install the optional extra and select it in `.agentenv/config.toml`. This provider is not yet in a published PyPI release:

```bash
uv sync --frozen --extra vercel
```

```toml
[sandbox]
default = "vercel"
agent_default = "vercel"

[sandbox.providers.vercel.config]
# Optional. The default managed image installs Docker and Compose v2 when apt is available;
# a prepared image avoids that download under a restrictive network policy.
image = "vercel/sandbox/universal"
# region = "iad1"
# failover_regions = ["cle1", "sfo1"]
# network_id = "your-secure-compute-network"
# Omit all three credential fields to use the SDK's default OIDC credentials.
# token = "secret:vercel_token"
# team_id = "your-team"
# project_id = "your-project"
```

Credentials are resolved by AgentEnv's `env:` and `secret:` references before the provider builds its scoped Vercel SDK client; the provider never changes the process environment or makes its SDK session the active one. Each sandbox is ephemeral, its execution time follows AgentEnv's timeout, and teardown destroys the sandbox and its orphan snapshots. Vercel currently provisions 64 GB of ephemeral disk: requests above 64 GB are rejected rather than silently ignored. Resource sizing uses the smallest supported even vCPU count (or one vCPU) that covers both the requested CPU and memory (2048 MB per vCPU), up to 32 vCPUs and 64 GB at the provider-wide ceiling; the account's plan may impose lower limits.

See the [Vercel resource limits](https://vercel.com/docs/sandbox/pricing#resource-limits) and [firewall policy](https://vercel.com/docs/sandbox/concepts/firewall) documentation. The provider treats CPU and memory requests as minimums. If a request does not already match a supported shape, AgentEnv logs both the requested and allocated CPU and memory before provisioning. For example, `cpu = 0.5` and `memory = 5000` allocate four vCPUs and 8192 MB. A disk request below 64 GB is also ignored: every sandbox receives the fixed 64 GB disk, and the provider logs the ignored request. Configure all three of `token`, `team_id`, and `project_id` for an explicit scoped token, or omit all three to continue using the SDK's OIDC credentials.

Attribution is limited to five tags. When more are supplied, the provider keeps `pipeline_step` and `run_id` first, then fills the remaining slots by key name in alphabetical order. It warns with the omitted key names without logging their values. Invalid empty keys or values are still rejected.

A missing sandbox raises `SandboxNotFoundError` when reconnecting. Run cleanup and the `teardown_sandboxes` step treat this condition as already cleaned up; authentication and transport errors still fail cleanup.

Outbound policy supports AgentEnv's allow-all and allowlist forms, including hostname and CIDR entries. Vercel's custom transforms, forwarding rules, and denied CIDR exceptions are not reconstructed as ordinary allowlists: reconnect leaves such a policy unknown and image loading fails closed. At creation, public Vercel routes are added to a restricted workload's platform egress floor so it can reach another Vercel sandbox's declared route. Image and file downloads add their exact signed-URL hostnames to the current allowlist; these hosts remain allowed until the policy is replaced. A CIDR-only Vercel policy allows unrestricted DNS resolution; use domain rules to restrict DNS, or deny-all to block it.

To work on agent-env itself, install from a clone with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/scaleapi/agentenv-framework && cd agentenv-framework
uv sync --extra dev
source .venv/bin/activate
```

## Run a task from your own loop

An episode is one run of a task in which some of its agents are played from outside AgentEnv, by a training loop's model or any code of your own. AgentEnv runs the rest exactly as `agent-env run` does, deploying the envs, running the other agents and scoring, and waits at each of your agent's turns:

```python
from agent_env.task import EpisodeSpec, Task

spec = EpisodeSpec.from_task(Task.get("my-task"), turn_timeout=900)
episode = await spec.start()
while (turn := await episode.next_turn()) is not None:
    async with turn.tools() as tools:  # the agent's env tools, called as its role
        ...  # your model reads turn.system_prompt and turn.prompt, and acts with tools.list() and tools.call()
    await turn.end(response="Done.")
scores = await episode.result()  # every score the run recorded, by step id
```

The scores are the ones `agent-env run` reports, so making a reward of them is yours. `trainer_agents=` names the agents you play when the task prompts several, `turn_timeout` is each turn's deadline in seconds (the prompt's own `timeout_seconds` otherwise), `episode.abandon(reason)` stops a run early, and `from_task` names any step an outside player can't stand behind. [Running tasks](https://www.agentenvframework.com/docs/tasks/running) covers the rest.

## Serve tasks as OpenEnv envs

Every task in a bundle is an [OpenEnv](https://github.com/huggingface/OpenEnv) env: any OpenEnv client resets it, acts through its tools and gets its reward.

```python
# app.py, served with `uvicorn app:app`
from agent_env.openenv import build_app

app = build_app("portsim")  # a folder or an installed bundle, or {"train": "dock-v1-train/", "eval": "dock-v1-eval/"}
```

```python
from openenv.core.generic_client import GenericEnvClient

async with GenericEnvClient(base_url="http://localhost:8000") as env:
    await env.reset(split="all", index=0)  # deploys the task's envs; the observation carries the prompt
    await env.step({"type": "call_tool", "tool_name": "get_situation", "arguments": {}})
    result = await env.step({"type": "call_tool", "tool_name": "end_turn", "arguments": {"response": "Done."}})
    result.reward  # the task's verifier score, as agent-env run reports it
```

Each session plays one episode at a time: reset deploys the task's envs and opens its agent's turn, the envs' MCP tools are the actions, and `end_turn` hands the reply back, after which the task's verifiers score it. A bundle's evals are splits beside `all`, which OpenEnv's Task API lists with each task's prompt. The bundle is prepared as `agent-env run` prepares it, except that the agent the client plays is never deployed, so it needn't exist. A task that records several scores takes `reward=<step id>`, `sandbox=` is where envs deploy, and `max_concurrent_envs` caps the sessions, each holding a deployment. A failure that ends an episode is an observation with `done`, no reward and `metadata["status"] == "error"`; a failed tool call is a tool error, and the episode goes on.

## Run agents in local microVMs

Install `agentenv-framework[smol]` and select the `smol_vm` sandbox provider in
`.agentenv/config.toml`:

```toml
[sandbox]
default = "smol_vm"
agent_default = "smol_vm"
```

Each agent and environment gets its own local SmolVM machine with a Docker daemon,
so workloads are isolated from the host Docker daemon. This needs a supported
Apple Silicon macOS or Linux host; on Linux, `/dev/kvm` must be available. On Apple Silicon,
build images for `linux/arm64` (or provide their build contexts so SmolVM can rebuild them);
this provider does not install amd64 emulation in the guest. AgentEnv can still
use its default local object and image stores: guest containers can fetch local
object grants, and the VM forwards pulls from the host's loopback registry.
Cloud image and object stores continue to work through the same provider.

## Plugins

A package of your own can add env types, providers, task steps and stores to agent-env, and `agent-env plugin` lists, checks, adds and removes installed plugins. The [plugin docs](https://www.agentenvframework.com/docs/plugins) show how.

## Contribute, release, license

[CONTRIBUTING.md](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md) is the contributor guide: open an issue before anything larger than a bug fix, one logical change per pull request with tests, pull request titles in the form `type(scope): summary`, an approving review from a code owner (`CODEOWNERS`) and green CI. [AGENTS.md](https://github.com/scaleapi/agentenv-framework/blob/main/AGENTS.md) is the repository map and conventions file for contributors and coding agents.

### Documentation map

| Document | Covers |
|---|---|
| [www.agentenvframework.com/docs](https://www.agentenvframework.com/docs) | the user guide: environments, artifacts, agents, tasks, the registry and plugins |
| [`packages/agentenv-protocol/README.md`](https://github.com/scaleapi/agentenv-framework/blob/main/packages/agentenv-protocol/README.md) | wire contract, environment server SDK, A2A agent framework |
| [`CONTRIBUTING.md`](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md) | development setup, test tiers, CI jobs, pull request rules |
| [`AGENTS.md`](https://github.com/scaleapi/agentenv-framework/blob/main/AGENTS.md) | repository map, configuration and extension-point summary, conventions for contributors and coding agents |
| [`SECURITY.md`](https://github.com/scaleapi/agentenv-framework/blob/main/SECURITY.md) | private vulnerability reporting and supported versions |
| [`CODE_OF_CONDUCT.md`](https://github.com/scaleapi/agentenv-framework/blob/main/CODE_OF_CONDUCT.md) | Contributor Covenant |
| [`.agentenv/config.example.toml`](https://github.com/scaleapi/agentenv-framework/blob/main/.agentenv/config.example.toml) | the all-local configuration to copy |
| [`.env.example`](https://github.com/scaleapi/agentenv-framework/blob/main/.env.example) | a commented reference of the `AGENT_ENV_*` variables; agent-env never loads this file, export what you need yourself. Its `AGENT_ENV_ENVIRONMENT` line is read only by an installed plugin, never by agent-env |

### Versioning and compatibility

The `agentenv-framework` distribution and `agentenv-framework-protocol` are versioned separately (`0.9.x` and `0.1.x` today), both in their `pyproject.toml`; agent-env releases carry a `vX.Y.Z` tag, and agentenv-framework-protocol is bumped in the same commit and has no separate tag today. There is no `agent_env.__version__` attribute; `agent-env --version` prints the installed version. Protocol extensions carry their version in the URI (`urn:agentenv:clock/v1`, `urn:agentenv:agent-config/v1`, `urn:agentenv:trajectory/v1`). One version can take more than one request shape: under `v1` the skill, trajectory, snapshot and changelog extensions accept object-transfer requests next to their older shapes, and the request field lists on an agent's card say which ones that agent takes. Renamed CLI commands are removed outright; no deprecated aliases exist at this version. A pull request that breaks the classes and entry points plugins build on fails CI unless its title marks the break with `!`; beyond that, a written compatibility and deprecation policy does not exist yet.

### Releases

A release is a version bump in both `pyproject.toml` files plus a `vX.Y.Z` tag. Maintainers cut releases: the bump is automated when a labelled pull request merges, so contributors do not edit `version` or push tags (see the Releases section of [CONTRIBUTING.md](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md)). Each release publishes both packages to PyPI and gets a [GitHub release](https://github.com/scaleapi/agentenv-framework/releases) whose notes list the pull requests merged since the previous one; there is no `CHANGELOG.md`.

### Support, security, license

Ask questions in [Discussions Q&A](https://github.com/scaleapi/agentenv-framework/discussions/categories/q-a), and report bugs with the [bug report form](https://github.com/scaleapi/agentenv-framework/issues/new?template=bug_report.yml). Report vulnerabilities privately through the contact in [SECURITY.md](https://github.com/scaleapi/agentenv-framework/blob/main/SECURITY.md), not in public issues; only the latest release is supported, so reproduce against it first. Contributors follow [CONTRIBUTING.md](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/scaleapi/agentenv-framework/blob/main/CODE_OF_CONDUCT.md); every pull request needs a code-owner review. agent-env and agentenv-framework-protocol are licensed under the Apache License 2.0; see [`LICENSE`](https://github.com/scaleapi/agentenv-framework/blob/main/LICENSE) and [`NOTICE`](https://github.com/scaleapi/agentenv-framework/blob/main/NOTICE). Their third-party dependencies and those dependencies' licenses are listed in [`THIRD_PARTY_NOTICES.md`](https://github.com/scaleapi/agentenv-framework/blob/main/THIRD_PARTY_NOTICES.md).

## Citation

To cite AgentEnv Framework, use the metadata in [`CITATION.cff`](https://github.com/scaleapi/agentenv-framework/blob/main/CITATION.cff). GitHub's **Cite this repository** button in the repository sidebar exports it as APA or BibTeX:

```bibtex
@software{agentenv_framework,
  author = {Arakelyan, Edgar and Polakam, Tejas and Singhal, Pratyush},
  title = {AgentEnv Framework},
  year = {2026},
  url = {https://github.com/scaleapi/agentenv-framework},
  license = {Apache-2.0}
}
```
