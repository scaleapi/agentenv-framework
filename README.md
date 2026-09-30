<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/lockup-dark.png">
  <img alt="AgentEnv Framework" src="assets/brand/lockup-light.png" width="360">
</picture>

# agent-env

agent-env is a Python SDK and CLI for building, deploying and running agentic environments and the tasks that grade agents inside them. Environments are containerized servers that speak the open `agentenv-protocol`; agent-env builds them into versioned images, deploys them behind a gateway, points an agent at them, and scores what the agent did.

## Contents

1. [What agent-env is](#what-agent-env-is)
2. [Install](#install)
3. [Quickstart: a local environment you can call, then a graded run](#quickstart-a-local-environment-you-can-call-then-a-graded-run)
4. [Build an environment](#build-an-environment)
5. [Register, deploy, connect and load an environment](#register-deploy-connect-and-load-an-environment)
6. [Define a task](#define-a-task)
7. [Agents](#agents)
8. [Reward and verification](#reward-and-verification)
9. [Run tasks: locally, then at scale](#run-tasks-locally-then-at-scale)
10. [Read results, evaluate and iterate](#read-results-evaluate-and-iterate)
11. [Configuration reference](#configuration-reference)
12. [Run on Google Cloud](#run-on-google-cloud)
13. [CLI reference](#cli-reference)
14. [Extend agent-env](#extend-agent-env)
15. [Contribute, release, license](#contribute-release-license)

## What agent-env is

agent-env gives you five primitives: an **environment** an agent acts in, an **artifact** that seeds or captures state, a **task** that strings deploy, prompt and grading steps into a DAG, an **agent** that speaks the A2A protocol, and an **eval** that groups tasks. Every primitive is a versioned document in a store you choose; the default store is a local SQLite file, so a laptop with Docker is a complete installation.

### The environment model

- **Environment card.** Every environment server publishes `GET /.well-known/agent-env.json`: its `name`, `protocolVersion` (`1.0`), the JSON-RPC data-plane URL, its `capabilities` (`tools`, `operations`, `extensions`) and any `children_environments`. The name comes from the `ENVIRONMENT_NAME` variable (agent-env sets it to the env's registered name), then `@environment_card(name=...)`, then the class name.
- **MCP tools.** `@tool` methods become real MCP tools served over streamable HTTP at `/mcp`. Tools are what the agent calls.
- **Data plane.** `POST /agentenv` is a JSON-RPC endpoint with three operations: `data/reset`, `data/add` and `data/get`. Seeding, loading and exporting world state go through it, so a run can start from a known state and end with an exportable one. This replaces a gym-style `reset()`/`step()` API: the agent acts through tools, and the harness controls state through the data plane.
- **Universe data.** Seed files are registered as versioned artifacts (`EnvironmentArtifact`, bundled into an `EnvironmentUniverseArtifact`) and loaded into a running instance with a reset followed by an add.
- **Snapshots.** A `snapshot_env` step exports the state of a running instance as a new universe artifact. Baking a snapshot into a pre-loaded image is limited to multi-server environments not on a container-mode sandbox, with the built-in local Postgres state backend and an object store that can presign uploads.
- **Gateway.** Every deploy places a gateway in front of the environment. It composes the card (children under `/svc/mcp-<name>/...`), proxies MCP and the data plane, and adds control endpoints: `/state`, `/step`, per-role tool gating (`/tools/disable`, `/tools/enable`), a virtual clock, triggers and an NDJSON `/trajectory` of tool calls.

### The reward loop in one sentence

A task deploys an environment and an agent, prompts the agent, runs one or more verifiers over the response and the captured trajectory, and optionally snapshots the resulting environment state; the verifiers write per-criterion results and a score into the run context. The chain is drawn once in [The canonical chain](#the-canonical-chain-the-one-run-diagram).

### Two packages and the seams

| Package | What it is | Who installs it |
|---|---|---|
| `agentenv-protocol` (`packages/agentenv-protocol/`) | The wire contract plus the server and agent SDKs: `AgentEnvEnvironment`, the card, data-plane and extension decorators, and the A2A (agent-to-agent protocol) agent framework behind the `agent` extra. Depends only on `pydantic`, `starlette` and `httpx`; `mcp` is imported lazily. | Environment images and agent images. An environment image does not need `agentenv-framework`. |
| `agentenv-framework` (this repository; import package `agent_env`, command `agent-env`) | The runtime and CLI: stores, image builds, the gateway, sandbox providers, task steps, verifiers, evals, the local explorer. | The machine that builds, deploys and runs. |

Every backend in `agent-env` is a seam: a TOML table naming `impl = "module.path:ClassName"` plus a `config` table. Stores (document, object, image, secret), sandbox providers, state providers, the runner, task steps, artifacts, environment kinds, CLI groups and explorer routes all plug in this way, and a bad pointer fails loudly with `ConfigError` the first time the seam is used. An installed package can also register its types through entry points, with no config at all. Conformance suites exist for the four store seams only. Details are in [Extend agent-env](#extend-agent-env); server and agent authoring depth is in [`packages/agentenv-protocol/README.md`](packages/agentenv-protocol/README.md).

## Install

### Prerequisites and trust model

- **Python 3.11 or newer** for `agentenv-framework` (the protocol package alone accepts 3.10). Verified on CPython 3.11.9, 3.11.14 and 3.12.12; the public CI runs 3.12.
- **uv** for the one-command install below. `pip` works too; see the alternative in [Install the packages](#install-the-packages).
- **Docker** with a running daemon, needed at the point of use: the `local` sandbox, every `put` that builds an image, and the first-run bootstrap of `agent-env up`. The unit tests need neither Docker nor network.
- **A model endpoint** for anything that prompts or judges an agent: any OpenAI-compatible base URL and key, or provider-prefixed model names routed natively. See [Point at a model](#point-at-a-model).
- **Trust model.** The `local` sandbox runs environments and agents as ordinary containers on your Docker daemon, one compose stack per deploy, with host ports published on the machine. Treat it as a development convenience, not a security boundary. The explorer served by `agent-env up` binds `127.0.0.1` only, has no `host` setting, and rejects foreign `Host` headers with HTTP 421. Configuration values hold references (`env:NAME`, `secret:KEY`), not secrets; the run context written to disk strips API keys. agent-env does not copy your AWS credentials into agent or environment containers unless the S3 object store opts in with `share_credentials` (see [Stores and secrets](#stores-and-secrets)). Every HTTP client agent-env makes verifies TLS; there is no setting or variable that turns that off.

### Install the packages

Neither package is on PyPI yet. Install both from a clone. The distribution is named `agentenv-framework`; the import package is `agent_env` and the command is `agent-env`. The repository is a uv workspace, so one command installs `agentenv-framework` and `agentenv-protocol` as editable packages with their dependencies from PyPI:

```bash
git clone <repo-url> agent-env && cd agent-env
uv sync --extra dev
```

`uv sync` creates `.venv` (it picked CPython 3.12.12 on the verification host) and installs about 120 packages from PyPI, roughly 250 MB, or 280 MB with `--extra dev`. Plain `uv sync` installs the runtime dependencies only: no `pytest`, no explorer. Add `--extra dev` for the test suite or `--extra explorer` for `agent-env up`. Run the CLI as `.venv/bin/agent-env` or activate the environment.

The committed `uv.lock` is kept current: the release bump writes both packages' new versions into it, and CI fails a pull request that leaves it stale, so `uv sync --locked --extra dev` installs exactly what it records.

With plain `pip`, install the protocol package first, because no public index has it and `pip install -e '.[dev]'` on its own fails with `No matching distribution found for agentenv-protocol>=0.1.0`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ./packages/agentenv-protocol
.venv/bin/pip install -e '.[dev]'
```

`make install` runs those two installs (about 50 s, 410 MB); it is what [CONTRIBUTING.md](CONTRIBUTING.md) and CI use.

### Install the command on its own

To use `agent-env` rather than develop it, install it as a uv tool from an index that has both packages, then add plugins to it with `agent-env plugin add` (see [Manage plugins](#manage-plugins)):

```bash
uv tool install agentenv-framework
agent-env --version
```

Upgrade it with `uv tool upgrade agentenv-framework`, which keeps the plugins. Running `uv tool install agentenv-framework` again replaces the tool with what that one command names and drops them. An exact `==` pin leaves `uv tool upgrade` nothing to move to, so pin only when you mean to stay.

### Optional extras and bundled cloud SDKs

| Extra | Contents | Needed for |
|---|---|---|
| `agentenv-framework[explorer]` | `fastapi`, `uvicorn` | `agent-env up` (refuses to start without it) |
| `agentenv-framework[gcp]` | `google-api-core`, `google-auth`, `google-cloud-secret-manager`, `google-cloud-storage`, `requests` | `GcsObjectStore`, `GcpSecretManagerSecretStore`, `FirestoreMongoDocumentStore`, `GoogleAccessTokenCredentials` |
| `agentenv-framework[dev]` | `explorer` and `gcp` plus `moto`, `psycopg2-binary`, `pytest` and its plugins (`pytest-asyncio`, `pytest-dependency`, `pytest-socket`, `pytest-timeout`, `pytest-xdist`) | running the test suite |
| `agentenv-protocol[agent]` | `a2a-sdk[http-server]`, `uvicorn`, `regex` | authoring and serving your own A2A agent |

The runtime dependencies of `agentenv-framework` are `agentenv-protocol`, `boto3`, `click`, `httpx`, `litellm` (the 1.96 line), `mcp` (below 2.0), `a2a-sdk` (pinned to 0.3.26), `pydantic`, `pymongo`, `pyyaml`, `modal` and `e2b` (the 2.46 line). The cloud SDKs install every time and stay inert until a config table or a `--sandbox` flag selects them; the default sandbox and stores are local. `google-cloud-storage` and `google-cloud-secret-manager`, and what they pull in, come only with the `gcp` extra; the extra also names `google-api-core`, `google-auth` and `requests`, which core already installs, because the stores import them.

An extra belongs to `agentenv-framework` itself, so `agent-env plugin add` does not install one: name it where you install agent-env, as in `uv tool install 'agentenv-framework[gcp]'`, `pipx install 'agentenv-framework[gcp]'` or `pip install 'agentenv-framework[gcp]'`. Re-running `uv tool install` drops the plugins it does not name. A store impl whose extra is missing fails to load with an error that names the extra. There is no `asyncpg`; `psycopg2-binary` comes only with the `dev` extra.

### Verify and platform notes

```bash
agent-env --help
```

```
Usage: agent-env [OPTIONS] COMMAND [ARGS]...
  Agent environment CLI.
Options:
  -v, --verbose  Enable verbose logging (DEBUG level)
  --help         Show this message and exit.
Commands:
  a2a-agent  A2A agent commands.
  artifact   Artifact commands.
  config     Inspect the resolved agent-env configuration.
  env        Environment commands.
  eval       Eval commands.
  plugin     Inspect, add and remove installed plugins.
  run        Run a bundle's tasks and evals, or list the installed bundles.
  task       Task commands.
  up         Start the local agent-env stack (stores, runner, explorer)...
```

Installed plugins can add groups and root options; they appear in this listing. `agent-env --version` prints the installed version.

The second check is `agent-env config show`: it prints which config file won, or `(none)`, and what every section resolved to (see [Inspect the resolved configuration](#inspect-the-resolved-configuration)). Running `--help`, `config show`, `config explain`, `config sources`, `config debug`, `plugin list`, `plugin show` or `plugin check` writes nothing to disk, though a plugin's own import code may. The first command that writes a local store (for example `task create` or `run hello`) creates your per-user state root, `~/.local/state/agent-env/` (see [Config discovery and precedence](#config-discovery-and-precedence)).

The third check runs `hello`, the bundle agent-env ships. It needs `bash` and the usual shell tools, and no Docker, model or configuration.

```bash
agent-env run          # lists the installed bundles and their folders; hello is agent-env's own
agent-env run hello
```

```
artifacts/greeting: v1 (new)
tasks/hello.json: v1 (new)
...
Tasks:
  tasks/hello.json v1: passed (hello: 1), 0.1s, instance @local/agentenv-framework/hello/hello-5aii6zzz
```

It writes to the per-user local stores, and removes the sandbox it deployed, a work folder under `~/.agent-env-sandboxes/`, when the run ends; [Run a bundle folder](#run-a-bundle-folder) says what it does and how to run an edited copy.

**Apple Silicon.** Every image-building `put` (`env mcp-server|gateway|service-db|website|website-browser put` and `a2a-agent put`) builds for `linux/amd64` by default, which is the right target for remote VM sandboxes but runs emulated under the `local` sandbox. Pass `--platform linux/arm64` on every `put` you intend to deploy locally, including the two bootstrap environments and the agent image. `agent-env up` has no platform flag and builds its bootstrap images for `linux/amd64`; register those two environments by hand instead (see [Start the local stack](#start-the-local-stack)). Local sandbox work directories live under `~/.agent-env-sandboxes/`; if you override that with `AGENT_ENV_LOCAL_SANDBOX_DIR`, pick a path your Docker Desktop file sharing includes.

## Quickstart: a local environment you can call, then a graded run

Steps 2 to 5 run cold on a laptop with Docker: a bundled example environment is built, deployed behind the gateway and called over MCP, then torn down. Step 6 runs a graded task fully locally too, with the bundled echo agent from step 3; it needs the model endpoint from step 1, a task file you write yourself and one workaround, all stated inline. Run every command from one project directory; the local stores live in its `.agentenv/`, and every id a run references must be in the same store. The commands below pass `--platform linux/arm64` because they were verified on Apple Silicon; the flag defaults to `linux/amd64`, so omit it on x86-64 hosts.

### Point at a model

Export an OpenAI-compatible endpoint and key. `deploy_agent` injects both into every agent container it starts, and the LLM judge reads them in-process:

```bash
export LITELLM_BASE_URL=https://<your-openai-compatible-endpoint>/v1
export LITELLM_API_KEY=<your-key>
```

No endpoint is defaulted. With neither set, a run stops at `deploy_agent` with `ConfigError: No model API key configured`; with only the key set, with `ConfigError: No model endpoint configured`. The check runs when the agent container is created, before any prompt, so even the model-free echo agent from step 3 needs both values to exist. The judge model is defaulted: with nothing configured a `rubrics_verifier` uses `claude-sonnet-4-6`, and its direct judge fails without an endpoint with `ConfigError: No model endpoint configured for 'claude-sonnet-4-6'`. If your endpoint does not serve that name, set `default_model` on the step or `[model.roles] judge` in `.agentenv/config.toml`. The agent's model comes from `--agent-model <name>` on `task run`, `model` on the `prompt_agent` step, or `[model.roles] agent`; the `[model]` table and the full precedence list are in [Model configuration](#model-configuration-owns-precedence). Steps 2 to 5 do not call a model.

### Start the local stack

Copy the all-local example config into the project directory, then check what every command run from here will read:

```bash
mkdir -p .agentenv
cp <repo>/.agentenv/config.example.toml .agentenv/config.toml
agent-env config show
```

The first two lines name the file that won and how it was found (`discovered by walking up from the working directory`). Each block after that names a section, its resolved backend and the layer that supplied it: `document: LocalSqliteDocumentStore ... from [stores.document]`, `agents: default_a2a_agent_id=a2a-default from [agents]`, `explorer: port=8234 from [explorer]`. The file is optional: without it every command falls back to the same local defaults and `config show` reports `config: (none)` with every store `from built-in default`. `agent-env config debug` lists the paths discovery checked. Both commands are read-only and create nothing; see [Inspect the resolved configuration](#inspect-the-resolved-configuration).

Every local deploy resolves two bootstrap environments by id: the service database `default-db` and the gateway `default`. Register them once: every project you run from shares one local store. The first `service-db` build takes minutes:

```bash
agent-env env service-db put --id default-db --platform linux/arm64
agent-env env gateway put --id default --platform linux/arm64
```

The first `put` is the first store write: it creates `document_store/documents.db` and `object_store/` under `~/.local/state/agent-env/`, your per-user state root (see [Config discovery and precedence](#config-discovery-and-precedence)), and writes nothing into the project. Images push to a local OCI registry on `127.0.0.1:5000`. The first push starts the `agentenv-registry` container (`registry:2`, `--restart unless-stopped`) that hosts it; nothing is started if a registry already answers there.

Optional: `agent-env up` registers the same two environments (as `linux/amd64` images; it has no platform flag) and serves the explorer API at `http://127.0.0.1:8234`. It needs the `explorer` extra and the config file copied above, and it runs in the foreground until Ctrl-C, so start it in a second terminal:

```bash
agent-env up
```

`up` prints the resolved backends (all local with this profile), a `bootstrap` block, and `explorer  http://127.0.0.1:8234`. The first run builds four images and takes a few minutes; later runs report `already registered` and are ready in about a second. `up` runs the explorer as a host process and starts no environment containers. Nothing in this quickstart calls the explorer; its API, port setting and `--no-bootstrap` are in [The local explorer and runner](#the-local-explorer-and-runner).

### Register an agent (the bundled echo agent)

Tasks (see [Define a task](#define-a-task)) deploy an agent by id; the default id is `a2a-default` (`[agents] default_a2a_agent_id` in `.agentenv/config.toml` changes it). An A2A agent is a container that agent-env prompts over the agent-to-agent protocol and configures through the `urn:agentenv:*` extensions; see [Agents](#agents). The repository ships no model-backed agent. It ships the deterministic agent its own test suites deploy, at `tst/data/a2a_agent/` (`Dockerfile`, `agent.py`): it answers `Echo: <prompt>`, records a trajectory, and calls no tool and no model. Its Dockerfile installs `agentenv-protocol[agent]` from a copy of the protocol package in the build context, so assemble a context and register it under the default id:

```bash
mkdir echo_agent
cp <repo>/tst/data/a2a_agent/{Dockerfile,agent.py} echo_agent/
cp -R <repo>/packages/agentenv-protocol echo_agent/agentenv-protocol
agent-env a2a-agent put --id a2a-default --dockerfile echo_agent/Dockerfile --context echo_agent --skip-validation --platform linux/arm64
```

```
Building Docker image...
Creating DockerImageArtifact...
Created artifact: id=a2a-agent-a2a-default version=1
Registering A2A agent...
Created A2A agent: id=a2a-default version=1 image=a2a-agent-a2a-default:1
```

Copy the whole package directory (it needs its `pyproject.toml`), leaving out any `.venv`, `dist` or `__pycache__` inside it. The image is `python:3.12-slim` plus the protocol package; the build took about 20 seconds with cached layers. `--skip-validation` is required on a fresh store (see [Validate an agent](#validate-an-agent)). `agent-env a2a-agent get --id a2a-default` prints the stored document: the image artifact `a2a-agent-a2a-default`, empty `default_env_vars`, and no agent card, because the card is read from the running container. A run with the echo agent proves the pipeline, not an agent: it makes no tool calls, so a rubric that requires one scores 0. To bring your own agent image, see [Register and deploy an A2A agent](#register-and-deploy-an-a2a-agent).

### Build and register the example environment

The repository ships an example server at `tst/data/agentenv_mcp/` (`server.py`, `Dockerfile`, `seed.json`): an in-memory item store with two tools, `items_add_item` and `list_items`, plus the data plane. Its Dockerfile copies the protocol package from the build context, so assemble a context first:

```bash
mkdir items_env
cp -R <repo>/packages/agentenv-protocol/src/agentenv_protocol items_env/
cp <repo>/tst/data/agentenv_mcp/{server.py,Dockerfile,seed.json} items_env/
agent-env env mcp-server put --id <env-id> --dockerfile items_env/Dockerfile --context items_env --platform linux/arm64
```

```
Derived environment_name='items' from the environment card.
Building MCP server Docker image...
Creating DockerImageArtifact...
Created artifact: id=mcp-server-<env-id> version=1
Creating MCPServerEnv...
Created MCPServerEnv: id=<env-id> version=1 environment_name=items env_provider_type=gateway
```

Passing `--context <repo>/tst/data/agentenv_mcp` directly fails at `COPY agentenv_protocol`; that directory is not a self-contained build context. `put` only registers the environment; `--validate` also runs the release gate, which needs a registered `a2a-default` agent and a model endpoint; see [Build, register and the release gate](#build-register-and-the-release-gate). Repeating `put` on the same id appends a new version. The image lands in the local registry as `localhost:5000/mcp-server-<env-id>:v1`; a `registry:2` container named `agentenv-registry` is started on first push if none is listening.

### Deploy it and call a tool

```bash
agent-env env deploy --id <env-id> --ttl-seconds 1800
```

```
Sandbox backend: config default
Found env: id=<env-id> version=1 type=mcp_server
Deploying (ttl=1800s, gateway_mode=performance, disk=10.0GB)...
Deployed!
Instance ID: <instance-id>
Env MCP Url: http://localhost:<port>/mcp
Env Gateway Url: http://localhost:<port>
Env DB Web Url: http://localhost:<port>/
Env DB MCP Url: http://localhost:<port>/mcp
Expires At (UTC): <timestamp>
```

A local deploy took about 30 seconds; `Sandbox backend: config default` says the provider came from `[sandbox] default` in the config file. Keep the instance id: the local sandbox records the TTL but does not enforce it, so you tear down by hand (see [Tear down](#tear-down)); what the deploy started, the deploy flags and how to read the record back are in [Deploy and the instance record](#deploy-and-the-instance-record). Call a tool through the gateway from any MCP client that speaks streamable HTTP (the `mcp` client library is a dependency of agent-env, so no extra install is needed):

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("<env-mcp-url>") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()                                            # items_add_item, list_items
            res = await s.call_tool("items_add_item", {"item": "from-mcp-client", "times": 2})   # {"count": 2}

asyncio.run(main())
```

IDE MCP clients take the same URL in their config (`{"mcpServers": {"agent-env": {"url": "<env-mcp-url>"}}}`); no auth header is needed locally. `curl -sS <env-gateway-url>/.well-known/agent-env.json` returns the composed card with `items` under `children_environments`; raw `curl` against `/mcp` must send `Accept: application/json, text/event-stream` or it gets HTTP 406. Loading `seed.json` into the instance and the gateway control endpoints are in [Register, deploy, connect and load an environment](#register-deploy-connect-and-load-an-environment).

### Create, run and read the example task (gated)

The repository does not yet ship an example task file, so write this one as `task.json`, substituting your environment id:

```json
[
  {"id": "deploy-env", "type": "deploy_env", "env_id": "<env-id>", "ttl_seconds": 1800},
  {"id": "deploy-agent", "type": "deploy_agent", "env_ids": ["<env-id>"], "a2a_agent_id": "a2a-default", "ttl_seconds": 1800},
  {"id": "prompt", "type": "prompt_agent", "prompt_id": "p1", "timeout_seconds": 600,
   "prompt": "Using the tools available to you, add one item named 'readme' to the item store, then list all stored items and reply with the exact list of items returned by the tool."},
  {"id": "verify", "type": "rubrics_verifier", "prompt_id": "p1", "verifier_id": "rubric", "use_agent_judge": false,
   "output_format": "rubric_binary", "score_aggregator": "all_pass",
   "criteria": [{"id": "item_added_and_listed", "weight": 1,
                 "criterion": "The agent called a tool to add an item named 'readme' and its final response reports the stored items, including 'readme'."}]},
  {"id": "snapshot", "type": "snapshot_env", "env_id": "<env-id>", "fail_task_on_error": false}
]
```

One line is a workaround, which is why this step is gated: `use_agent_judge: false` grades with a direct model call instead of deploying a third sandbox for a judge agent, which here would be the echo agent registered as `a2a-default`; that judge call is the only model call in this chain. The trajectory needs no field: `prompt_agent` stores it below `prompt_agent_trajectories/prompt_id=<prompt-id>/` in the configured object store. Then create and run:

```bash
agent-env task create task.json --id <task-id> --project-id <label>
agent-env task run --id <task-id> --k 1 --env-sandbox local --agent-sandbox local --output-dir <output-dir>
```

`--project-id` is required by `task create` and optional on `task run`, which warns when it is omitted and proceeds; see [Task document and DAG semantics](#task-document-and-dag-semantics). The run took 41 seconds locally, printed `Task instance: <instance-id>` first, one block per completed step, and ended with:

```
Task instance: <instance-id>
...
prompt-response:  prompt_id: p1  response: Echo: Using the tools available to you, add one item named 'readme' ...  tool_call_count: 0
...
score (verifier_id=rubric): 0.0
Task completed!
Task context written to: <output-dir>/<task-id>_<hex>.json
```

The score is 0.0 because the echo agent called no tool: the judge's justification says so, and `result` is `false`. That is the expected outcome with the bundled agent; the run proves the pipeline. A passing score needs an agent image that calls tools, registered under `a2a-default` (see [Agents](#agents)). A verifier that needs no model, `agent_prompt_response_verifier` with a `response_contains` criterion, would grade the echo reply itself; it is described in [Programmatic verifiers](#programmatic-verifiers) and was not run for this guide.

In the context JSON, `metadata.verifications.rubric` holds the per-criterion rows (`id`, `weight`, `criterion`, `score`, `justification`, `result`) and the aggregated `score`; `prompt_responses[0].agent_trajectory_s3_uri` is the `file://` URL of the trajectory the agent reported, for the echo agent a one-entry JSON list `{"type": "echo", "input": ..., "output": ...}` (the format is agent-defined; see [Trajectories](#trajectories)); `metadata.env_snapshotted_universes.snapshot` names the exported universe artifact, keyed by the step id. `agent-env task get-instance --id <instance-id>` shows `Status: completed` and `Progress: 5/5` for the instance id printed at run start. Interpretation and evals are in [Read results, evaluate and iterate](#read-results-evaluate-and-iterate).

### Tear down

After step 6 three things are running: the step 5 environment stack, the environment stack the task's `deploy_env` started, and the agent container. The `local` sandbox does not enforce `--ttl-seconds`, `task run` never tears down (a fatal step leaves everything deployed so far running too), and there is no teardown command. Step 5 printed its instance id; the task's is `deployed_envs[0].instance_id` and the agent's sandbox id is `deployed_agents[0].sandbox_id` in the context JSON, both also printed in the run's `deployed-env:` and `deployed-agent:` blocks. Close each environment instance from Python in the project directory, once per id:

```python
import asyncio
from agent_env.env import Env

async def main():
    env = await Env.from_instance_id("<instance-id>")
    await env.close()

asyncio.run(main())
```

Terminating the agent container, what `close()` runs and what is left behind are in [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down). If you started `agent-env up`, stop it with Ctrl-C.

## Build an environment

An environment is an MCP server that also speaks the agent-env data plane: one Python class with decorated methods, run as a process, then packaged as a Docker image that contains `agentenv-protocol` but not `agent-env`. This section walks the bundled example in `tst/data/agentenv_mcp/`; decorator and wire-format depth is in [`packages/agentenv-protocol/README.md`](packages/agentenv-protocol/README.md).

### Environment card, naming and tools

Abridged from `tst/data/agentenv_mcp/server.py` (the full file runs as shown below):

```python
from agentenv_protocol import AgentEnvEnvironment, DataPart, add_data, environment_card, extension, get_data, reset_data, tool

@environment_card(name="items")
class ItemsEnv(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.store: list = []
        self.create_app()
        self.mcp.tool(name="list_items")(self.list_items)  # imperative registration also works

    @reset_data
    async def _reset(self) -> None:
        self.store.clear()

    @add_data
    async def _add(self, parts: list) -> None: ...

    @get_data
    async def _state(self) -> list:
        return [DataPart(data={"items": self.store})]

    @extension(uri="urn:agentenv:set-errors/v1", description="Make a tool start raising at a given error rate.")
    async def set_errors(self, tool_name: str, error_rate: float) -> dict: ...

    @tool(name="{environment_name}_add_item")
    async def add_item(self, item: str, times: int = 1) -> str:
        """Add an item to the store."""
        ...

if __name__ == "__main__":
    ItemsEnv().serve()
```

`GET /.well-known/agent-env.json` serves the card: `name`, `protocolVersion` `1.0`, `url` `/agentenv`, and `capabilities.{tools,extensions,operations}`. The name resolves from the `ENVIRONMENT_NAME` variable, then `@environment_card(name=...)`, then the class name; `{environment_name}` in a tool name resolves to it at mount. Each `@tool` method becomes a real MCP tool with an `inputSchema` derived from the signature. Only `@tool` methods appear on the card; tools registered through `self.mcp.tool(...)` are served by MCP `tools/list` but not advertised.

### Data plane: reset, add, get

With the example server running (`MCP_PORT=18765 python tst/data/agentenv_mcp/server.py` from the repo root; see [Run and smoke-test locally](#run-and-smoke-test-locally)), JSON-RPC 2.0 at `POST /agentenv` with methods `data/reset`, `data/add` and `data/get`:

```bash
curl -sS -H 'Content-Type: application/json' -X POST http://127.0.0.1:18765/agentenv \
  -d '{"jsonrpc":"2.0","id":2,"method":"data/add","params":{"parts":[{"kind":"data","data":{"items":["a","b"]}}]}}'
```

`data/add` receives parts in three shapes: a `DataPart` (`{"kind":"data","data":{...}}`), a `FilePart` with a `file://` URI, and a `FilePart` with inline base64 `bytes`. The example handles the first two; an inline-bytes part returns `-32000 add_failed`. The protocol README example handles all three. `data/add` is additive, and the artifact loader calls `data/reset` before `data/add`, so `reset` must clear all state.

| Condition | JSON-RPC error |
|---|---|
| Unknown method | `-32601` `method not found: <method>` |
| Invalid params (pydantic detail in `data.detail`) | `-32602` `invalid_request` |
| Malformed JSON / not a JSON-RPC object | `-32700` `parse error` / `-32600` `invalid request` |
| Handler raised | `-32000` with `data.code` `reset_failed`, `add_failed` or `get_failed` |

Error envelopes come back with HTTP 200; `GET /agentenv` is 405.

### Extensions, intake declaration and interface manifest

`@extension(uri="urn:agentenv:<name>/v1", description=...)` on an async method publishes `POST /agentenv/ext/<method-name>`. Body keys bind to keyword arguments and coerce to the annotated types; the card lists the route under `capabilities.extensions` with a signature-derived schema. A missing or mistyped argument returns HTTP 400 `{"ok": false, "error": {"code": "invalid_params", ...}}`; unknown routes return 404. The example server defines two such methods, `set_errors` (shown in the listing and called below) and a clock-sync extension, `sync_time`, which the listing omits.

```bash
curl -sS -X POST http://127.0.0.1:18765/agentenv/ext/set_errors -H 'Content-Type: application/json' -d '{"tool_name":"list_items","error_rate":0.0}'
```

Behind the gateway the route becomes `/svc/mcp-<name>/agentenv/ext/<method-name>` and the composed card rewrites the endpoint. The well-known `urn:agentenv:*` operation sets (clock, snapshot, triggers, trajectory) and `@custom_extension` are documented in the protocol README. `env mcp-server create-cli --id <env-id>` deploys the environment itself, reads its live tools and, when the server serves one, its interface manifest (`urn:agentenv:get-interfaces/v1`, `GET /agentenv/interface-manifest`), stores a CLI artifact and closes that deploy (described from the code, not exercised); see [Register and deploy an A2A agent](#register-and-deploy-an-a2a-agent).

### Run and smoke-test locally

```bash
cd <repo>
MCP_PORT=18765 python tst/data/agentenv_mcp/server.py
curl -sS http://127.0.0.1:18765/.well-known/agent-env.json
```

`serve()` binds `MCP_HOST` (`0.0.0.0`) and `MCP_PORT` (`18765`) with streamable-http and needs only `starlette`, `pydantic`, `httpx`, `mcp` and an importable `agentenv_protocol`. Check the card, the data plane and the error codes, then call a tool over MCP with the client snippet from [Deploy it and call a tool](#deploy-it-and-call-a-tool), pointed at `http://127.0.0.1:18765/mcp`. Tool calls and the data plane share one store, so `data/get` shows the added items. Raw `curl` against `/mcp` needs `Accept: application/json, text/event-stream` or the server answers 406.

### What goes in the image

`tst/data/agentenv_mcp/Dockerfile`, trimmed (the original pins the base image by digest and adds a non-root user with `RUN useradd --create-home --uid 10001 appuser` and `USER appuser`):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir starlette pydantic httpx "mcp>=1.25,<2"
COPY agentenv_protocol /app/agentenv_protocol
COPY server.py /app/server.py
COPY seed.json /data/seed.json
ENV PYTHONPATH=/app
CMD ["python", "server.py"]
```

The Dockerfile expects `agentenv_protocol/` in the build context, and `tst/data/agentenv_mcp/` does not ship it, so building that directory directly fails at the `COPY`. Assemble a context first, as shown in [Build and register the example environment](#build-and-register-the-example-environment) (or `pip install agentenv-protocol` in the Dockerfile once published). On Apple Silicon, build with `--platform linux/arm64` for the local sandbox (see [Verify and platform notes](#verify-and-platform-notes)).

### Environment kinds

| Kind | Register with | Fits when | Limits |
|---|---|---|---|
| `mcp_server` | `agent-env env mcp-server put --id <env-id> --dockerfile <Dockerfile> --context <dir>` (or `--dockerfile-github-url`) `[--env-provider-type <type>] [--validate]` | One protocol-conformant server image | Seed loads need the data plane implemented |
| `multi` | `agent-env env multi put --id <env-id> --mcp-server <id>[:version] --mcp-server <id>[:version] [--website <id>[:version]] [--name <label>] [--env-provider-type <type>] [--validate]` | Several servers behind one gateway and one service database; universes and snapshots | Refs pin the latest version at put time; only `mcp_server` and `website` ids. `--name` fixes the name agents see the MCP server under (`mcp__<label>__<tool>`); without it each deploy draws `env` plus four random digits, so declare one when rubric text or tooling needs a stable prefix |
| `website` | `agent-env env website put --id <env-id> --backend-dockerfile <f> --frontend-dockerfile <f> [--env-provider-type <type>]` | A browsable web app proxied at `<gateway>/website/<name>/` | Register `agent-env env website-browser put` once first (not exercised for this guide) |
| live endpoint (custom) | a custom `Env` subclass whose documents carry an http(s) `mcp_url` attribute, registered through `[envs] impls` and saved with `put` from Python; no CLI | An MCP server that already runs elsewhere | Never deployed: a `deploy_agent` step that lists the id in `env_ids` reads `mcp_url` from the env document; no seed, reset or snapshot; no bearer token is sent, so the endpoint must accept requests without one (see [Custom envs, task steps, artifacts](#custom-envs-task-steps-artifacts)) |
| custom | `.agentenv/config.toml`: `[envs] impls = ["mypkg.envs:MyEnv"]` (subclass `Env`, set `type`) | A runtime the built-ins do not cover | No CLI `put` unless your plugin adds one |

### Author seed data as artifacts

An `EnvironmentArtifact` wraps one file (a `FileArtifact` named `<artifact-id>-file` is created for it) and pins it to an environment name; an `EnvironmentUniverseArtifact` bundles several with pinned `id:version` refs.

```bash
agent-env artifact environment put --id <artifact-id> --description 'items seed' --environment-name items items_env/seed.json
agent-env artifact environment-universe put --id <universe-id> --environment-artifact <artifact-id>:1
```

`--environment-name` is required here (the CLI does not derive it from a card for artifacts) and must equal the environment's card name. Files land in `~/.local/state/agent-env/object_store/artifacts/file/<artifact-id>-file/<version>/`. Loading needs a deployed instance; see [Load universe data, snapshot and export](#load-universe-data-snapshot-and-export).

## Register, deploy, connect and load an environment

Registering turns an image into a versioned environment document; deploying provisions it behind a gateway that composes the card, proxies MCP and REST, and records tool calls. Every command below ran on the local stack (SQLite documents, filesystem objects, a local OCI registry on `127.0.0.1:5000`, the `local` Docker sandbox). A configured object store, remote registry and remote sandbox provider take the same commands and return `https://` URLs; see [Choose compute, network policy and state](#choose-compute-network-policy-and-state).

### Bootstrap prerequisites

Deploy resolves the gateway environment `default` and the service-db environment `default-db` by id. Register them once with the two `put` commands in [Start the local stack](#start-the-local-stack). `agent-env up` runs the same two puts when missing, always for `linux/amd64` (see [Verify and platform notes](#verify-and-platform-notes)). The first `service-db` build compiles a Postgres MCP sidecar from source and takes minutes. Images push to `localhost:5000/<artifact-id>:v<version>` (the `agentenv-registry` container starts on first push) plus a tarball in the local object store; documents go to the local document store. Website environments also need `agent-env env website-browser put`.

### Build, register and the release gate

The build-and-register command is shown in [Build and register the example environment](#build-and-register-the-example-environment). The output reports `Derived environment_name='items' from the environment card`, the image artifact `mcp-server-<env-id>` and the environment version. `--environment-name` is optional when exactly one `@environment_card(name=...)` is in the build source. `put` on an existing id appends a version; nothing is overwritten.

**GitHub-sourced builds.** `env mcp-server put --dockerfile-github-url https://github.com/<owner>/<repo>/tree/<ref>/<path>/Dockerfile` (optionally `--docker-context-github-url` for a context directory other than the Dockerfile's parent, same repository and ref) and `env website put --backend-dockerfile-github-url ... --frontend-dockerfile-github-url ...` clone the repository instead of reading a local build context. The Python equivalents are `MCPServerEnv.put_from_github(...)` and `WebsiteEnv.put_from_github(...)`; the resulting image artifact and environment are the same as from a local build. Public repositories need no credentials. For a private one, pass `github_token=` (any token GitHub accepts as the password for `git clone`); the CLI reads it from `GITHUB_TOKEN`. The clone, `docker build`, `docker push` and `docker save` run through the `[sandbox] default` provider's VM path; on the `local` provider that is your host shell as the invoking user (no VM, no container; `sudo` is stripped), starting with `apt-get install git`, so it needs a Debian-like Linux host with root, Docker, and push access to the configured image registry (on macOS or without root it fails earlier with `Script failed (exit 127)`). The image is always built for `linux/amd64` (`--platform` is ignored with a warning) and then uploaded through a presigned URL, which the local filesystem object store cannot issue (`GitHub image builds need a signable object store`), so the all-local stack cannot complete a GitHub build. Use a local Dockerfile there, or configure a VM-capable sandbox provider plus an object store that can presign uploads (the bundled S3 store does). Described from the code; not exercised for this guide.

`put` registers the environment and stops there. `--validate` then runs the release gate: a nine-step validation task (deploy, card, core protocol, tool schema and conformance checks, then an agent probe and assessment) that exits non-zero when the environment fails it. The gate needs the default agent (`[agents] default_a2a_agent_id`, `a2a-default` when unset; see [Agents](#agents)) to be registered, and a model endpoint; on a fresh store with the default config it aborts at `deploy_agent` with `A2AAgent a2a-default not found`, stamps no verdict, and leaves the validation deploy's containers running. `--override` runs the gate and publishes despite a failure, recording the override; it implies `--validate` and still needs the agent. `env mcp-server validate <env-id>` runs the same validation on an environment that is already registered and prints the results without a release decision; for a multi-server environment use `env multi put --validate` or `env multi validate --id <env-id>`.

**Without a gateway.** `--env-provider-type server` (default `gateway`) registers an env that deploys as its server alone, with no gateway or service database, on whatever sandbox provider is configured. The release gate deploys it as declared on `[sandbox].default`, so the card it records in `metadata.environment_card` is the server's own. A `deploy_env` step on such an env can't take `gateway_mode = "consistent"` or `env_state_*`, and `task create` refuses them at save. Composed into a `multi` env, it is deployed behind that env's gateway like any other child.

There is no `env get` or `env list`. Read back in Python: `Env.get("<env-id>")` (latest), `Env.get("<env-id>", 1)`, or `Env.query().type("mcp_server").latest().execute()` from `agent_env.env`.

### Deploy and the instance record

The deploy command and the record it prints are shown in [Deploy it and call a tool](#deploy-it-and-call-a-tool). A local deploy takes about 30 seconds and starts one Compose stack of five containers (gateway, your server, service database, database web UI, database MCP sidecar) on random free ports, under `~/.agent-env-sandboxes/agent-env-local-<id>-*` (override with `AGENT_ENV_LOCAL_SANDBOX_DIR`, a path Docker Desktop shares). `agent-env env get-instance --id <instance-id>` prints the id, env id and version, the four URLs, the sandbox id and the timestamps; the stored record additionally holds `sandbox_type`, `gateway_mode`, `env_state_instance_ids` and `metadata` (read it with `agent_env.env.store.get_env_instance_store().get("<instance-id>")`). Keep the id: nothing lists instances.

| Flag | Effect |
|---|---|
| `--version` | Environment version; defaults to latest |
| `--ttl-seconds` | 60 to 1209600, default 10800; stamped as `expires_at_utc`; enforced only by remote providers |
| `--gateway-mode performance\|consistent` | `consistent` serializes tool calls through one lock |
| `--sandbox` | `local`, `modal`, `modal_vm`, `e2b` or a `[sandbox.providers]` name; comma-separated fallback chain; default `[sandbox].default` |
| `--service-db`, `--gateway` | Override the bootstrap ids (`default-db`, `default`) |
| `--priority` | Integer, default `0`; passed through to the sandbox provider unchanged. None of the built-in providers (`local`, `modal`, `modal_vm`, `e2b`) use it; a custom `[sandbox.providers]` implementation may map it to a scheduling tier |
| `--disk-size-gb`, `--cpu`, `--memory-mb` | VM sizing (defaults 10 GB, 1.0 CPU, about 8192 MB) |
| `--env-state-type`, `--env-state-instance-id` | State backend for a fresh store, or attach to an existing `esi-...` store |

### Choose compute, network policy and state

Environments and agents have separate sandbox slots, `[sandbox].default` and `[sandbox].agent_default`, overridden per command by `env deploy --sandbox`, `a2a-agent deploy --sandbox` and `task run --env-sandbox` / `--agent-sandbox`. Built-ins are `local`, `modal`, `modal_vm` and `e2b`; custom providers register under `[sandbox.providers.<name>]`, and any slot takes a comma-separated chain that falls through on failure (see [Configuration reference](#configuration-reference)). The `local` provider runs on the host Docker daemon with no isolation; remote providers need an object store and remote registry and return the same record over HTTPS (TTL enforcement: [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down)). Sandbox mode comes from how the sandbox is started: `modal` deploys an environment as separate containers (container-mode); `modal_vm` and `e2b` give a VM; `local` runs an environment as a Compose stack on the host daemon (VM-mode) but starts an agent, or a `deploy_sandbox` sandbox, as a single container (container-mode). `env snapshot` refuses container-mode sandboxes, so it excludes `modal`, not `local`; on any provider it also needs an object store that signs upload URLs (`signed_put_url`), as `S3ObjectStore` does and `GcsObjectStore` does with a signer; the bundled filesystem store signs none, so the zero-config local setup fails at the upload step.

`local_postgres` is the only built-in state provider; every deploy creates an `esi-...` state record. Extra backends register under `[state.providers.<name>]`. `agent-env env init-env-state` pre-creates a store for external providers only (`local_postgres` exits with `cannot be pre-initialized out-of-band`); reuse one with `--env-state-instance-id`.

### Connect an MCP client, inspect the gateway and data

Point any MCP client that speaks streamable HTTP at `Env MCP Url`; the local URL needs no headers, and remote deploys give `https://` URLs. IDE clients typically take the same URL in a JSON config file (for example `.mcp.json` or `.cursor/mcp.json`):

```json
{"mcpServers": {"agent-env": {"url": "<Env MCP URL>"}}}
```

The IDE config above was not tested from an IDE. At `Env Gateway Url`:

| Path | Returns |
|---|---|
| `/.well-known/agent-env.json` | Composed card; `children_environments[]` lists each server at `/svc/mcp-<name>/agentenv` with rewritten extension endpoints |
| `/mcp` | MCP over streamable-http; tool names pass through unprefixed |
| `/svc/mcp-<name>/...` | Per-server proxy (card, `/agentenv`, `/agentenv/ext/...`); unknown names return 404 |
| `/agentenv` | Environment-level data plane, forwarded when exactly one server sits behind the gateway (`-32000` otherwise) |

`Env DB Web Url` and `Env DB MCP Url` point at the shared service database through a web UI and an MCP sidecar.

### Gateway and agent controls for experiments

| Endpoint | Purpose |
|---|---|
| `GET /state` | Servers, their tools, `changelog_id`, per-role rules |
| `POST /step` | `{"action":"list_tools"}` or `{"action":"call_tool","tool_name":...,"arguments":{...}}` without an MCP client |
| `POST /tools/disable`, `/tools/enable` | `{"role": <role>, "tools": [...] or "*"}`; rules are in-memory per gateway |
| `GET /trajectory` | NDJSON `tool_call` / `tool_call_result` events with `event_id` and `timestamp_utc` |
| `GET /clock/state`, `GET /triggers/state` | Virtual clock (`{"armed": false}` until set) and trigger engine state |

The role comes from the `AgentEnv-Role` request header; requests without it use `default`:

```bash
curl -sS -X POST <env-gateway-url>/tools/disable -d '{"role":"restricted","tools":["list_items"]}'
curl -sS -H 'AgentEnv-Role: restricted' -X POST <env-gateway-url>/step -d '{"action":"call_tool","tool_name":"list_items","arguments":{}}'
```

The second call returns 403 `Tool 'list_items' is disabled for role 'restricted'`; over MCP the same call returns `isError=true`. `PUT /clock/set-time`, `/triggers/register` and `--gateway-mode consistent` exist; not exercised for this guide.

### Load universe data, snapshot and export

```bash
agent-env env mcp-server load-environment-artifact --instance-id <instance-id> --environment-artifact-id <artifact-id>
```

The loader copies the file to `/data/<filename>` in the server container, then calls `data/reset` and `data/add` with one `file://` `FilePart` (replace semantics; the server must implement that branch). The artifact's `environment_name` must match the environment's. For a `multi` environment, load a universe per instance:

```bash
agent-env env multi load-environment-universe-artifact --env-instance-id <instance-id> --environment-universe-artifact-id <universe-id>
```

Against a single-server instance the data loads but the command exits 1 on a `snapshot_baked` attribute error; use the per-artifact loader there. `agent-env env snapshot --instance-id <instance-id>` bakes the loaded database into an `env-snapshot-<env>` image for clean resets; it supports `multi` environments only, after a universe load, not on a container-mode sandbox, with `local_postgres` and an object store that signs upload URLs (the filesystem store does not), and `--snapshot-after-load` on the universe load bakes right after ingest. The `snapshot_env` task step exports a run's end state as an `EnvironmentUniverseArtifact`; see [Run tasks: locally, then at scale](#run-tasks-locally-then-at-scale). `env multi validate-universe-compatibility` and `artifact environment-universe compatible-envs` exist; not exercised for this guide.

### Reattach, look up a known instance, tear down

`agent-env env get-instance --id <instance-id>` prints a known instance. No command lists instances or tears one down, and the TTL does not reap local stacks. Tear down in Python from the project directory (so config discovery finds `.agentenv/`):

```python
import asyncio
from agent_env.env import Env

async def main():
    env = await Env.from_instance_id("<instance-id>")
    await env.close()

asyncio.run(main())
```

`close()` runs `docker compose down -v --remove-orphans` for the local sandbox and terminates the VM for remote providers (the gateway then answers 404); the same command in the work directory is equivalent locally. An env a plugin's environment provider deployed can't be closed this way; see [Custom sandbox, state, environment and runner providers](#custom-sandbox-state-environment-and-runner-providers).

For an agent deployed by a task run, take `deployed_agents[0].instance_id` from the context JSON and terminate its sandbox through the provider:

```python
import asyncio
from agent_env.a2a_agent.store import get_a2a_agent_instance_store
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

async def main():
    dep = get_a2a_agent_instance_store().get("<agent-instance-id>")
    sb = await build_sandbox_provider(dep.sandbox_type).get_sandbox(dep.sandbox_id)
    await sb.terminate()

asyncio.run(main())
```

With the `sandbox_id` from the run output's `deployed-agent:` block (or `deployed_agents[0].sandbox_id` in the context JSON) the store lookup can be skipped: `await (await build_sandbox_provider("local").get_sandbox("<sandbox-id>")).terminate()`. To find instances you did not keep ids for, query the document store: `from agent_env.store import Filter, Sort`, then `get_config().get_document_store().query("env_instances", Filter.of(env_id="<env-id>"), sort=Sort.by("created_at_utc", descending=True))` (and `a2a_agent_instances` with `Filter.of(agent_id=...)`). Closed instances stay listed with no status flag, so keep the ids from the run output when you can.

Every sandbox a task run recorded goes at once through `teardown_run`, given the run's context: `agent-env run` and `eval run` call it after each run. From a context JSON that `--output-dir` wrote:

```python
import asyncio
import json
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

context = TaskStepContext.from_dict(json.load(open("<context>.json")))
report = asyncio.run(teardown_run(context))
print(len(report.terminated), report.still_up)
```

It terminates each sandbox within 120 seconds, removes a local sandbox's work folder once it is down, and reports a failure rather than raising it.

`task run` leaves its environment and agent running (see [What every run needs](#what-every-run-needs)). Left behind: the instance record (no status field), the `esi-...` state record, the work directory, images in the registry and daemon, and documents and tarballs in the local stores. `agent-env env teardown-env-state --instance-id <esi-id>` retires a state record by stamping its expiry. Reset means re-seeding through the loader or restoring a snapshot; there is no reset command.

## Define a task

A task is a versioned JSON document: a list of step definitions that agent-env runs as a DAG. Steps deploy environments and agents, prompt, verify, and snapshot. They communicate through one shared run context.

### Task document and DAG semantics

Write the steps as a JSON array and register it. The first `create` stores version 1; every later `create` under the same id appends a new version and never overwrites.

```bash
agent-env task create task.json --id <task-id> --project-id <label>
```

`--project-id` is required by the CLI. Any string is accepted. The value is stored on the task and forwarded as a user/tag label on every model call; nothing else in a local run reads it.

Ordering is declared per step. With `depends_on` omitted, a step runs after every earlier step (a sequential chain). With `depends_on: [{"task_step_id": "<id>"}]` the steps form a DAG and steps with no edge between them run concurrently; forward references and duplicate step ids are rejected at create time. `fail_task_on_error` defaults to `true`: a failure cancels in-flight steps and fails the run. Set it to `false` on a tolerant step and the failure is recorded in `context.metadata.failed_steps` (step id, type, error, `is_fatal: false`), dependents still run, and the run can complete with exit code 0.

Every step reads and writes the same `TaskStepContext`: `deployed_envs`, `deployed_agents`, `deployed_sandboxes`, `prompt_responses`, and a free-form `metadata` map. The run writes this context to `--output-dir` as one JSON file per run; see [The run record](#the-run-record).

Each step is also stored as a versioned document keyed by its `id` in a collection shared by every task in the store; tasks pin their steps inline, so a step id shared by two tasks only bumps a counter. If many tasks share one store, prefix step ids with the task name so the collection stays readable.

### The canonical chain (the one run diagram)

```text
[load_artifact] -> deploy_env -> deploy_agent -> prompt_agent -> <verifier> -> snapshot_env | collect_artifacts
                   deployed_envs  deployed_agents  prompt_responses  metadata.verifications  metadata.env_snapshotted_universes
```

The quickstart task in [Create, run and read the example task](#create-run-and-read-the-example-task-gated) ran end to end on the local Docker sandbox. Steps join through context keys, not through return values: `deploy_env` appends to `deployed_envs`; `deploy_agent` takes `env_ids` (or `env_step_id`), pushes each environment's MCP URL to the agent together with the name the agent should file its tools under (a `multi` environment's `--name`, otherwise `env` plus four random digits drawn at deploy; agents that accept a name expose the tools as `mcp__<name>__<tool>`), and appends to `deployed_agents` (an id in `env_ids` that no `deploy_env` step deployed must be an environment with a live http(s) `mcp_url`; see [Custom envs, task steps, artifacts](#custom-envs-task-steps-artifacts)); `prompt_agent` targets `agent_name` (default `default-agent`) and appends a `prompt_responses` entry keyed by `prompt_id`; the verifier reads that `prompt_id` and writes `metadata.verifications[<verifier_id>]`; `snapshot_env` exports the environment and records the universe artifact id under `metadata.env_snapshotted_universes[<step_id>]`.

`prompt_agent` stores each trajectory below `prompt_agent_trajectories/prompt_id=<prompt-id>/` in the configured object store, after the `AGENT_ENV_FIXTURE_PREFIX` prefix when one is set. Set `trajectory_output_prefix` on the step to store it elsewhere; it must be a URL of that store, since a prefix the store does not accept fails the upload rather than writing somewhere else.

### Tasks without an environment server

Coding-style tasks need a machine and a container, not an MCP server. Three steps cover that: `deploy_sandbox` provisions a bare sandbox (`sandbox_name`, `image`, `cpu`, `memory_mb`, `disk_size_gb`, `ttl_seconds`, `exposed_ports`); `run_docker_container` builds an image from a docker-context artifact or URL and starts it on that sandbox (`container_name` defaults to `task-container`, with `ports`, `env_vars`, `build_args`, `command_override`, `ready_command`, `volumes`); `deploy_agent` with the same `sandbox_name` places the agent on that machine. Reward comes from `run_container_unit_tests_verifier` (a command's exit code plus optional reward and result files), from `verify_sandbox` with the same `sandbox_name` (file and shell probes on the machine itself), or from `run_code`, which runs a script artifact against an environment or agent and stores its output under `metadata.script_results[<result_id>]`. Described from their definitions; not exercised for this guide.

A long pipeline keeps every sandbox until the run ends or its TTL expires. `teardown_sandboxes` terminates the sandboxes behind named targets mid-run (`agent_names`, `env_ids` — every sandbox of the env — and `sandbox_names`); give it a `depends_on` on the last step that uses them. It is strictly best-effort: an already-gone sandbox or a failed terminate is logged and the run continues, and `fail_task_on_error` must stay `false`. Terminated ids are appended to `metadata.torn_down_sandbox_ids`, and a re-run of the step skips them.

### Step catalogue by family

The runtime ships 48 built-in step types. Print the live list from Python:

```bash
python -c "from agent_env.task_step.registry import get_task_step_registry as g; r=g(); print(len(r)); print(sorted(r))"
```

| Family | Step types | Needs a model |
|---|---|---|
| Deploy and provision | `deploy_env`, `deploy_sandbox`, `run_docker_container`, `deploy_agent`, `install_agent`, `reset_env`, `sync_env_clock`, `apply_server_config`, `modify_env_tool_access`, `register_env_triggers`, `register_agent_triggers`, `peer_agents`, `add_skills`, `build_mcp_cli`, `teardown_sandboxes` | no |
| Load, snapshot, collect | `load_artifact`, `snapshot_env`, `snapshot_agent_state`, `collect_artifacts` | no |
| Agent interaction | `prompt_agent` | only if the agent calls one; the bundled echo agent does not |
| Scripts | `run_code` | no |
| Verifiers (reward) | `rubrics_verifier`, `agent_prompt_response_verifier`, `verify_sandbox`, `env_outcome_verifier`, `run_container_unit_tests_verifier`, `aggregate_verifiers` | only `rubrics_verifier` |
| Conformance validators (environment, universe) | `verify_env_card`, `verify_env_core_protocol`, `verify_mcp_tool_schema`, `verify_spec_conformance`, `verify_mcp_env_assessment`, `validation_gate_aggregator`, `verify_universe_load_export_roundtrip`, `combine_universe_verdicts` | some |
| Conformance validators (agent) | `verify_a2a_agent_card`, `verify_a2a_agent_config_identity`, `verify_a2a_agent_mcp`, `verify_a2a_core_protocol`, `verify_a2a_install`, `verify_a2a_modalities`, `verify_a2a_peer_agents`, `verify_a2a_role`, `verify_a2a_skill_config`, `verify_a2a_snapshot`, `verify_a2a_system_prompt`, `verify_a2a_trajectory` | some |
| Human in the loop | `review`, `deploy_human_agent` | no |

Validators gate registration; verifiers grade behavior; [Reward and verification](#reward-and-verification) explains the split. Custom steps register through `[task_steps] impls` in `.agentenv/config.toml` (see [Custom envs, task steps, artifacts](#custom-envs-task-steps-artifacts)) and must declare their own `type`.

### Artifacts between steps

Artifacts are immutable, versioned documents referenced as `id:version`; a bare `id` resolves to the latest version, and `put` on an existing id appends a version. Built-in kinds: file artifacts (a single uploaded file), environment and environment-universe artifacts (seed data and bundles of it, see [Author seed data as artifacts](#author-seed-data-as-artifacts)), file-artifact universes, docker-image artifacts (created by `env mcp-server put` and `a2a-agent put`), skill artifacts (loaded into an agent with `a2a-agent add-skill`), and CLI artifacts (built from an environment's live tools by `env mcp-server create-cli`, which deploys it itself). Steps move data through them in both directions: `load_artifact` seeds a deployed environment, `run_code` executes a script artifact, and `snapshot_env` produces a new environment-universe artifact whose id is recorded in the run context. Download one with `agent-env artifact environment-universe get --id <id> --output-dir <dir>`.

### Seeds and templating

One task can run against many worlds. Any string field of any step may contain `<name>` placeholders; `task run-batch` fills them from the columns of a CSV, one run per row.

```text
name,item_name
seed-alpha,alpha
seed-beta,beta
```

The header row names the placeholders. Substitution applies to prompts, rubric criteria and every other string field. The command, the per-run output files and the seed metadata each run records are in [Fan-out](#fan-out).

### Step-level retry (`retry_config`)

Any step may declare `retry_config`, so that when it fails the scheduler rebuilds a clean world from an earlier point instead of re-running on whatever the failed attempt left behind:

```json
{"id": "prompt", "type": "prompt_agent", "prompt_id": "p1", "prompt": "...",
 "depends_on": [{"task_step_id": "deploy-agent"}],
 "retry_config": {"retry_from_step_id": "deploy-env", "max_retries": 1}}
```

- `retry_from_step_id` is required and must be the step itself or one of its dependency ancestors; `task create` rejects anything else. Pointing it at the step's own id re-runs only that step. `go_to_step_id` is accepted as an alias on input.
- `max_retries` (an integer, default `1`) bounds how many times the step may trigger a retry before its failure is terminal. The budget is per step.
- The span is `retry_from_step_id` plus every step that transitively depends on it, plus any sibling still in flight when the failure landed, since it may hold partial writes. Completed independent branches are untouched.
- Retry is consulted before `fail_task_on_error`: a tolerant step is retried first and tolerated only once its budget is spent. If two retryable steps fail in the same scheduler batch, the task fails instead of rebuilding overlapping spans.

The rollback works through the run's step journal. Every completed step's context diff is recorded against the task instance (collection `task_step_journal`), and the store rebuilds the context by replaying the surviving steps' diffs over the run's seed in one write. Everything the span wrote goes, including its `deployed_envs`, `deployed_agents`, `deployed_sandboxes` and `prompt_responses` entries and any metadata it set; secrets are never stored and are grafted back into the live context; then the span is dispatched again as ordinary steps. A retry is refused, and the task fails with the original error and the live context untouched, when the run has no task-instance record, a surviving step has no journal entry (a run that predates the journal), the write fails, or a surviving step ran concurrently with a span member, because its diff may contain the span's writes. Sequential tasks (`depends_on` unset) never hit that last case.

The sandboxes the rolled-back span deployed are not terminated by the rollback. Each entry in `context.metadata.retry_resets[]` (`attempt`, `step_id`, `retry_from_step_id`, `replayed`, `at_utc`) records them under `rolled_back_sandboxes`, shaped like the context's `deployed_envs` / `deployed_agents` / `deployed_sandboxes` slices, so you can tear them down as in [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down). The failure that triggered the retry stays in `metadata.failed_steps[]` flagged `retried: true`, and the `LocalRunner` ignores such entries when deciding whether a run completed. The `retried` flag and the `retry_resets` entry are written in the same store write as the rollback and journaled as the scheduler's own entry, which replays right after the seed and is never undone; when the rollback is refused or cannot be persisted, neither is written, so `failed_steps` records the failure as terminal. Described from the scheduler's code; not exercised for this guide.

### Validate before you deploy

```bash
agent-env task validate --id <task-id>
```

`validate` runs each step's `preflight()` and exits non-zero on a problem. Only steps that implement a preflight are checked: `run_code` verifies that its script artifact resolves to the right type, `deploy_env` that its `gateway_mode` is valid and its env loads and takes the step's options (a `server` env refuses `gateway_mode = "consistent"` and `env_state_*`), and custom steps may add their own. It does not resolve agent ids, so a missing `a2a-default` surfaces at run time, not here. `task create` runs the same checks and refuses to save a failing task unless `--skip-validation` is passed. For partial runs, `task run --start-step <n> --context-json <saved-context>` resumes from a persisted context (see [Resume and partial runs](#resume-and-partial-runs)); the Python `Task.run()` also accepts `end_step`, which the CLI does not expose.

## Agents

An agent is a container that speaks the A2A (agent-to-agent) protocol and implements the `urn:agentenv:*` extensions agent-env uses to hand it environments, skills and configuration. Registering one is a prerequisite for every `deploy_agent` step and for the environment release gate.

### Two ways to get an agent

A fresh store contains no agent; the first `deploy_agent` fails with `A2AAgent a2a-default not found` until you register one.

1. **The bundled echo agent.** `tst/data/a2a_agent/` is the deterministic agent the repository's own integration tests deploy: it answers `Echo: <prompt>`, records a trajectory, makes no tool calls and calls no model. Its card advertises the `agent-config`, `mcp-config`, `trajectory` and `triggers` extensions, and its agent-config accepts `model`, `system_prompt`, `project_id`, `task_id` and `model_params`. Register it with the commands in [Register an agent](#register-an-agent-the-bundled-echo-agent). It proves a task pipeline end to end; it cannot pass a rubric that requires tool use, and it cannot pass `a2a-agent validate` (no tool call for the MCP check, no snapshot extension).

2. **Your own A2A agent image.** Any container that serves an A2A card at `/.well-known/agent-card.json` (see [Author your own agent](#author-your-own-agent)) and the extensions in the protocol package can be registered with `a2a-agent put` (see [Author your own agent](#author-your-own-agent)).

`deploy_agent` resolves the id from `[agents] default_a2a_agent_id` when a step names no `a2a_agent_id`; the built-in default is `a2a-default`. Precedence: `configure(default_a2a_agent_id=...)` in code, then `[agents]`, then the built-in; there is no environment variable, and a blank value or an unknown key under `[agents]` raises `ConfigError`. Override per step with `a2a_agent_id`, or per run with `task run --a2a-agent-id`. The same key names the judge agent a `rubrics_verifier` deploys when `judge_a2a_agent_id` is unset.

```toml
[agents]
default_a2a_agent_id = "a2a-default"
```

### Register and deploy an A2A agent

| Command | Purpose |
|---|---|
| `a2a-agent put --id <id> --dockerfile <path> [--context <dir>] [--default-model <m>] [--env-var K=V] [--platform linux/amd64\|linux/arm64] [--skip-validation]` | Build the image (artifact `a2a-agent-<id>`) and register the agent. Pass `--skip-validation` (see [Validate an agent](#validate-an-agent)). `--env-var K=V` values are stored as the agent's default environment; per the deploy code, if they include both `LITELLM_API_KEY` and `LITELLM_BASE_URL`, deploys no longer resolve a model endpoint (not exercised for this guide). `--default-model` is stored as metadata and read only by the `install_agent` step. |
| `a2a-agent get --id <id> [--version]` | Print the agent document as JSON. |
| `a2a-agent deploy --id <id> [--version] [--env-var K=V] [--ttl-seconds 60..1209600] [--sandbox <provider>[,<fallback>]] [--priority N]` | Deploy on a sandbox. TTL defaults to 7200 s; the provider defaults to `[sandbox] agent_default`, else `local`. |
| `a2a-agent get-instance --id <instance-id>` | Read a deployed instance record. It cannot stop the instance. |
| `a2a-agent add-skill --instance-id <instance-id> (--skill-artifact-id <id> \| --skill-md-path <file> \| --skill-s3-url <url>)` | Push a skill to a running agent's `/ext/skill-config`. |
| `a2a-agent validate --id <id> [--sandbox <provider>]` | Deploy and run the agent conformance checks (see [Validate an agent](#validate-an-agent)). |

Inside a task, `deploy_agent` does the same deploy and then configures the instance: it POSTs each environment's MCP URL to the agent's `urn:agentenv:mcp-config/v1` endpoint, `skills` to skill-config, and `system_prompt` through agent-config. `add_skills` does the skill push as a separate step, and `env mcp-server create-cli` turns a deployed environment's tools into a CLI artifact an agent can install. There is no command to stop a deployed agent; terminate its sandbox through the provider from Python (snippet in [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down)), or wait for the TTL. Every deploy path (`a2a-agent deploy`, `deploy_agent`, the judge agent of `rubrics_verifier`) goes through `A2AAgent.deploy`, which is where the model endpoint and key are resolved; see [Model configuration](#model-configuration-owns-precedence).

### Author your own agent

The protocol package contains the agent-side framework and runnable examples (`basic_agent.py`, `streaming_agent.py`, `multimodal_agent.py`, `custom_extensions_agent.py`, `advanced_agent.py`). Install `agentenv-protocol[agent]` and serve one:

```bash
cd packages/agentenv-protocol/examples
A2A_PORT=18777 python basic_agent.py
```

In a second terminal:

```bash
curl -sf http://127.0.0.1:18777/.well-known/agent-card.json
```

Stop the server with Ctrl-C. `serve()` reads `A2A_HOST` and `A2A_PORT`. The card lists `protocolVersion 0.3.0`, the `/a2a` endpoint, and the extensions the agent implements: `urn:agentenv:agent-config/v1` (endpoint `/ext/agent-config`, accepting `model`, `system_prompt`, `max_input_chars`, `provider_token`, `role`, `name`, `description`, `timeout_seconds`) and `urn:agentenv:trajectory/v1`. Model credentials are read only when a prompt arrives, so the card serves without them. The a2a-sdk logs a deprecation for the older `/.well-known/agent.json` alias; use `agent-card.json`. agent-env itself reads the card from `/.well-known/agent.json` (`A2AAgent.deploy` polls that path for up to 300 s); an agent must keep serving it too, or `deploy_agent`, `a2a-agent deploy` and `a2a-agent validate` fail with `did not serve /.well-known/agent.json`. The protocol framework and the a2a-sdk default server apps serve both paths. For the class model, streaming, error semantics and custom extensions, see the [A2A agent framework](packages/agentenv-protocol/README.md#a2a-agent-framework) section of the protocol README.

### Model configuration (owns precedence)

Every step that deploys an agent or judges a response needs a model endpoint and key. Neither is defaulted. `deploy_agent` does not read them itself: it calls `A2AAgent.deploy`, which fills `LITELLM_API_KEY` from `LITELLM_API_KEY` or `[model] api_key`, then `LITELLM_BASE_URL` from `LITELLM_BASE_URL` or `[model] base_url`, for each variable the agent did not register as a default at `a2a-agent put --env-var`, and raises before any container starts: `ConfigError: No model API key configured` when nothing is set, `ConfigError: No model endpoint configured: set [model] base_url in .agentenv/config.toml or the LITELLM_BASE_URL env var.` when only the key is set. The step's `litellm_base_url` field and `task run --litellm-api-key` override what is injected. In-process calls (the direct judge) accept provider-prefixed names such as `openai/gpt-4o` and route them natively without an endpoint; a bare name needs one.

```toml
[model]
base_url = "<openai-compatible-base-url>"
api_key = "secret:MODEL_KEY"
default = "gpt-4o-mini"
[model.roles]
judge = "gpt-4o"
agent = "env:AGENT_MODEL?claude-sonnet"
[model.params]
temperature = 0.2
```

| Setting | Resolution order (first wins) |
|---|---|
| Endpoint | `LITELLM_BASE_URL` env var, then `[model] base_url` |
| API key | `--litellm-api-key` per run, then `LITELLM_API_KEY` env var, then `[model] api_key` (`env:` and `secret:` references resolve lazily) |
| Judge key | `--judge-litellm-api-key`, then `--litellm-api-key`, then the key above |
| Model for a prompt | `--agent-model` per run, then the `prompt_agent` step's `model`, then `[model.roles] agent` (else `[model] default`), which `deploy_agent` copies onto the context; with none set, no `model` is sent and the agent uses its image's own default |
| Judge model | the verifier's `default_model`, then `[model.roles] judge` (else `[model] default`), then `claude-sonnet-4-6` (resolved when the task is created and stored on the step; create a new task version to pick up a changed role) |
| Judge endpoint and key | direct judge: key `--judge-litellm-api-key`, then `--litellm-api-key`, then `LITELLM_API_KEY`, then `[model] api_key`; endpoint a `judge_litellm_base_url` or `litellm_base_url` entry in the run context's `user_overrides` (no `task run` flag; reachable via `--context-json` or the Python API), then the step's `default_model_api_base`, then `LITELLM_BASE_URL`, then `[model] base_url`. When the resolved endpoint is not the configured one (the env var or `[model] base_url`), `[model] api_key` is not sent to it; only `LITELLM_API_KEY` or a per-run key is. A bare model name with no endpoint fails with `ConfigError: No model endpoint configured for '<model>'`. Agent judge: the judge agent (`judge_a2a_agent_id`, else `[agents] default_a2a_agent_id`) is deployed through `A2AAgent.deploy` with the same resolution as `deploy_agent` |
| Call parameters | `[model.params]` merged with the step's `model_params`; the reserved keys `model`, `messages`, `api_key`, `api_base`, `user`, `metadata`, `timeout`, `response_format` are rejected with a `ConfigError` |

`A2AAgent.deploy` injects the resolved `LITELLM_BASE_URL` and `LITELLM_API_KEY` into the agent container, so an agent that honors them uses the same endpoint as the runtime. `[model]` accepts only `api_key`, `base_url`, `default`, `params` and `roles`; unknown keys fail loud.

### Agent configuration and agent-side triggers

`prompt_agent` sends its per-prompt configuration to the agent's `urn:agentenv:agent-config/v1` endpoint before the prompt: `model`, `system_prompt`, `effort`, `harness`, `max_turns`, `max_thinking_tokens`, `output_format`, `timeout_seconds`, `project_id`, `task_id`, `agentenv_tools`, and `model_params` (`[model.params]` merged with the step's `model_params`). The set is negotiated: only the fields the agent card lists as supported are sent, the rest are dropped silently. `agentenv_tools` is therefore an opaque list of tool names for an agent that implements in-process tools of its own; no bundled agent does, and the echo agent accepts only `model`, `system_prompt`, `project_id`, `task_id` and `model_params`. On the agent side, `register_agent_triggers` (`agent_name`, `triggers`) pushes trigger definitions to the agent's `urn:agentenv:triggers/v1` extension; environment-side triggers live on the gateway (see [Gateway and agent controls for experiments](#gateway-and-agent-controls-for-experiments)). The extension contract is in the [A2A agent framework](packages/agentenv-protocol/README.md#a2a-agent-framework) section of the protocol README.

### Validate an agent

`a2a-agent validate` deploys a validation environment and the agent, then runs a validation task built around the `verify_a2a_*` conformance steps (the twelve registered `verify_a2a_*` types in the step table above, plus two LiteLLM call-attribution checks in `task_step/task_steps/a2a_agent_validator/` that are not registered as step types), interleaved with further agent deploys, probe prompts and rubric checks. The registered checks cover the extension surface: `agent_card` and `core_protocol` (the card and A2A message flow), `agent_mcp` (environments arrive through mcp-config), `agent_config_identity` and `system_prompt` (agent-config), `skill_config`, `install`, `modalities`, `peer_agents`, `role`, `snapshot` and `trajectory`. `a2a-agent put` runs the same task unless `--skip-validation` is passed. The validator is described from its code; not exercised for this guide.

Caveat: the validator deploys a validation environment whose id is fixed in the runtime and does not exist in a fresh store, so the suite cannot pass until that id becomes configurable. Until then, register agents with `--skip-validation` and prove them the way this README does: run a task whose verifier checks the agent's response or tool calls (see [The canonical chain](#the-canonical-chain-the-one-run-diagram)). The bundled echo agent would not pass the suite even then: the MCP check needs a tool call and it implements no snapshot extension, which is why the repository's own integration test marks its validation as an expected failure.

### Humans, peers and user simulators

Multi-turn and multi-agent tasks use the same `prompt_agent` step. Set `max_conversation_turns` above 1 and name the other party: `user_agent_name` for a deployed user-simulator agent, or `user_a2a_url` for an A2A endpoint you host. When neither is set, the runtime falls back to `[conversations] default_human_a2a_url` or the `AGENT_ENV_HUMAN_A2A_URL` env var and raises `ConfigError: No human-A2A URL configured` when both are unset. `deploy_human_agent` (`agent_name` default `human_agent`, `a2a_url`) registers such an endpoint in the run context without deploying anything. `peer_agents` pushes a routing table to each agent's `/ext/peer-agents` so agents can address each other. `review` (`label`, `timeout_seconds`) pauses the run until a decision is recorded in the review store; agent-env ships the pause, and you supply the tool that records the decision. Described from their definitions; not exercised for this guide.

## Reward and verification

Reward in agent-env is not a return value. It is whatever verifier steps write into the run context after the agent has acted. A verifier can be an LLM judge, a deterministic check, a shell probe, or a unit-test run, and several can be combined.

### Verifier contract, and why validators are not rewards

Every verifier writes one entry to `context.metadata.verifications[<verifier_id>]` with an aggregated `score` between 0 and 1. Set `verifier_id` explicitly; it defaults to a random hex string. Every verifier except `aggregate_verifiers` also stores `results`, the rows the score was computed from; the aggregator reads three fields of a row, `result` (boolean), `score` (defaults to 1 or 0 from `result`) and `weight` (default 1), and everything else is the verifier's own. `rubrics_verifier` rows carry your criterion fields (`id`, `criterion`, `weight`) plus the judge's `score`, `result` and `justification`; `agent_prompt_response_verifier` and `verify_sandbox` copy every field of your criterion into the row and add `criterion_index`, `score`, `result` and `justification`, so `id` is present only if you gave the criterion one; `run_container_unit_tests_verifier` writes a single row with `id` `exit_code` or `reward`; `env_outcome_verifier` stores the rows your script returns unchanged. Skipped runs write one row `{id, score: 0, result: false, message}` (`prompt_error` when the prompt failed). From the quickstart run with the echo agent:

```json
"verifications": {
  "rubric": {
    "format": "rubric_binary",
    "results": [
      {"id": "item_added_and_listed", "weight": 1, "criterion": "...", "score": 0.0, "justification": "The agent did not call any tool to add an item named 'readme' ...", "result": false}
    ],
    "score": 0.0,
    "compact_trajectory_s3_uri": null,
    "judge_trajectory_s3_uri": null
  }
}
```

Three aggregators turn rows into the score: `all_pass`, `any_pass`, and `weighted_average` (positive weights are averaged, negative weights act as penalties, and the result is clamped to 0..1); a row without `result` counts as failed. A custom verifier is an ordinary custom step that writes the same map (see [Custom envs, task steps, artifacts](#custom-envs-task-steps-artifacts)). Verifiers never set a top-level `metadata.score`; readers of results use the per-verifier entries (see [Verifications and scores](#verifications-and-scores)).

The `verify_env_*`, `verify_mcp_*`, `verify_a2a_*` and `validation_gate_aggregator` steps look similar but answer a different question. They check that an environment or agent conforms to the protocol, run as the release gate at `put` time, and stamp the registered document. They say nothing about how well an agent did on a task. Use validators to admit components; use verifiers to grade behavior.

### LLM judge over trajectory and response

`rubrics_verifier` grades a `prompt_id` against a list of criteria (`id`, `criterion`, `weight`) using a judge model. It reads the agent's response and, with `use_trajectory` (default true), the captured trajectory, so a criterion can require specific tool calls, as the example above does (and fails the echo agent for exactly that reason). The verifier step from the quickstart task:

```json
{"id": "verify", "type": "rubrics_verifier", "prompt_id": "p1", "verifier_id": "rubric",
 "use_agent_judge": false, "output_format": "rubric_binary", "score_aggregator": "all_pass",
 "criteria": [{"id": "item_added_and_listed", "weight": 1,
               "criterion": "The agent called a tool to add an item named 'readme' and its final response reports the stored items, including 'readme'."}]}
```

| Option | Effect |
|---|---|
| `output_format` | `rubric_binary` (default; each row carries a boolean `result` and a 0 or 1 `score`), `rubric_partial`, or `trajectory_mistakes` |
| `use_agent_judge` | `true` (default) deploys a judge agent on the agent sandbox and terminates it afterwards: `judge_a2a_agent_id` when set, else `[agents] default_a2a_agent_id`, else `a2a-default`; `false` calls the model directly, which needs no extra sandbox and can re-grade a persisted trajectory after the run's sandboxes are gone |
| `default_model` | judge model; falls back to `[model.roles] judge` (see [Model configuration](#model-configuration-owns-precedence)) |
| `trajectory_filter` | compacts the trajectory before judging; `compaction_type` `default` or `screenshot`. Screenshot compaction is not supported for multi-turn prompts or agent judges. `--apply-trajectory-filter` / `--no-trajectory-filter` on `task run` override the step |
| `max_thinking_tokens`, `effort`, `judge_timeout_seconds` | judge call limits (timeout default 1000 s) |

Criteria are re-keyed `c1..cN` for the judge and mapped back to your ids in the results. If the prompt step recorded an error, verification is skipped with score 0. The entry also records `compact_trajectory_s3_uri` and `judge_trajectory_s3_uri` when those objects exist.

### Programmatic verifiers

These steps need no model and produce the same `verifications` entry.

| Step | Where it looks | Criteria or inputs |
|---|---|---|
| `agent_prompt_response_verifier` | the response text for `prompt_id` | `{"type": "response_contains", "needles": [...]}` or `{"type": "response_regex_present", "pattern": "...", "flags": "..."}` |
| `verify_sandbox` | under `base_dir` in a deployed agent's sandbox (`agent_name`, inside its container on a VM) or in a `deploy_sandbox` sandbox (`sandbox_name`, directly on it) | `probe_file_exists`, `probe_dir_exists`, `probe_file_contains`, `bash_cmd_succeeds` (with `shell_timeout_seconds`); unknown types are kept as rows flagged `skipped: true` and excluded from the score. On the `local` sandbox, every `/app` in a `bash_cmd`, even inside quotes, is pointed at the sandbox's work directory, so write paths relative to `base_dir` (the command's working directory) |
| `env_outcome_verifier` | the deployed environment's MCP URL for `env_id` | a Python file artifact (`file_artifact_id`) exposing `async def verify(mcp_url)` that runs in the agent-env process and returns the result rows, stored unchanged; the Python `put(verify_script_file_path=...)` helper uploads it as `<id>-verifier-script` |
| `run_container_unit_tests_verifier` | a container started by `run_docker_container` (`sandbox_name`, `container_name`) | `command`, `setup_commands`, `user` (default `root`), `timeout_sec` (default 300), `env_vars`; records `command`, `exit_code`, `timed_out`, `stdout_head`, `stderr_head`, `extracted_files` from `result_paths`, and one row (`id` `reward` when `reward_path` is set and graded from that file, else `exit_code` from the exit status). It stores stdout and stderr as file artifacts below `verifier-outputs/` in the configured object store. |

Each accepts `score_aggregator` and `verifier_id`, and every row copies the criterion you wrote (`agent_prompt_response_verifier`, `verify_sandbox`) or your script's output (`env_outcome_verifier`). Described from their step definitions; not exercised for this guide.

### Combining signals

`aggregate_verifiers` takes `verifier_ids`, merges their `results` rows (dropping `skipped` rows), applies `score_aggregator` (default `weighted_average`), and writes `{score, source_verifier_ids}` under its own `verifier_id`. The merged entry has no `results` of its own; rows without a `result` field (for example an `env_outcome_verifier` script that returned only `score`) count as failed under `all_pass` and `any_pass`. Per-criterion rows stay available in the source entries, so a run can report both a single number and the breakdown behind it. Give it `depends_on` edges to the verifiers it reads when they run on concurrent branches.

Scores stay per run; `eval run` reports completion per task run, not a score (see [Evals](#evals-run-a-task-set-under-different-agents-and-models)). Pass@k, means across seeds, and comparisons between agents or models are computed from the per-run context files.

## Run tasks: locally, then at scale

One primitive executes a task: `Task.run()`. The CLI wraps it four ways: `task run` for one task, `task run-batch` for one task over many seed rows, `eval run` for a task set, and `run` for the tasks and evals of a bundle folder. The first three write one context JSON per run. `task run` and `task run-batch` leave the deployed sandboxes running, where remote providers reclaim them at the TTL and the local sandbox does not (see [Tear down](#tear-down)); `eval run` and `run` tear down each run's sandboxes as it ends.

### Run from the CLI

`task run` executes the latest version of a stored task. Every flag is a per-run override; none creates a new task version.

```bash
agent-env task run --id <task-id> --k 1 --env-sandbox local --agent-sandbox local --output-dir <path>
```

On the local sandbox this needs the prerequisites in [What every run needs](#what-every-run-needs); without them it stops at the first missing piece.

| Flag | Effect |
|---|---|
| `--version` | Run a specific task version (default: latest). |
| `--agent-model`, `--a2a-agent-id`, `--agent-artifact-id` | Model, agent and agent image for this run. Model precedence is owned by [Model configuration](#model-configuration-owns-precedence). |
| `--env-sandbox`, `--agent-sandbox` | Sandbox slot per deploy family: a built-in name, a `[sandbox.providers]` name, or a comma-separated fallback chain. |
| `--gateway-env-id`, `--service-db-env-id` | Bootstrap environments to use instead of `default` and `default-db`. |
| `--env-state-type`, `--env-state-instance-id` | State backend for a fresh store, or attach to an existing one. |
| `--litellm-api-key`, `--judge-litellm-api-key` | Per-run model keys for agent and judge. |
| `--apply-trajectory-filter` / `--no-trajectory-filter` | Force trajectory compaction on or off for every `rubrics_verifier` in the run. |
| `--start-step`, `--context-json` | Resume; see [Resume and partial runs](#resume-and-partial-runs). |
| `--output-dir` | Where the context JSON lands (default: a fresh directory under `/tmp`). |
| `--project-id` | Optional label forwarded to model calls. A warning prints when unset; the run proceeds. |

The run prints `Task instance: <instance-id>` first, then one block per completed step (environment URLs, agent card, prompt response, each verification with its score). It ends with `Task completed!` and `Task context written to: <path>/<task-id>_<8hex>.json`. A fatal step exits non-zero and leaves earlier deployments running.

### Fan-out

`--k N` launches N independent rollouts in parallel; lines carry a `[run N]` prefix and N context JSONs are written. Pass@k is yours to compute from `metadata.verifications` in each file. Larger values are described from the command's help; not exercised for this guide.

`task run-batch` fans out over CSV rows instead. The header names the placeholders, one row is one run, and `<column>` tokens are substituted in every string field of every step; the CSV shape and the rules are in [Seeds and templating](#seeds-and-templating).

```bash
agent-env task run-batch --id <task-id> --seeds seeds.csv --concurrency 2 --env-sandbox local --agent-sandbox local --output-dir <path>
```

Each seed writes `<task-id>-seed<N>_<8hex>.json` with `metadata.seed` (the row) and `metadata.universe_id` (the `name` column); all seeds share one `run_group_id`. Running a task set under several agents or models is the job of [Evals](#evals-run-a-task-set-under-different-agents-and-models).

### Run a bundle folder

A bundle is a folder holding tasks and what they need, which `agent-env run` writes and runs without registering anything first. In this release it can hold:
- `artifacts/<name>/`, whose files become a file artifact, or a file-artifact universe when there are several;
- `agents/<name>/`, an A2A agent: an `agent.toml` whose `image` names a `docker_image` artifact in a store, or a `Dockerfile` the run builds the agent's image from;
- `tasks/<name>.json`, a list of steps that refer to the bundle's entities by name;
- `evals/<name>.toml`, with `tasks = [...]` naming the bundle's tasks.

An `agent.toml` takes `image`, a store id or `{ artifact = "<id>", version = <n> }` to pin one, and optionally `default_env_vars` (string values) and a `[metadata]` table of `default_model` and `min_disk_size_gb`. The agent's card isn't authored: the image serves it when the agent deploys. Leave `image` out, or the whole `agent.toml`, and the folder's `Dockerfile` builds the image instead, as described below.

```toml
# agents/solver/agent.toml
image = { artifact = "solver-image", version = 3 }
default_env_vars = { LOG_LEVEL = "debug" }

[metadata]
default_model = "claude-sonnet-4-6"
```

A task naming `solver`, in a `deploy_agent`'s `a2a_agent_id` or a `rubrics_verifier`'s `judge_a2a_agent_id`, deploys this agent, so its image is the one pinned here.

An agent folder with a `Dockerfile` and no `image` is built by the run, with `docker build` on this machine and the folder as the build context, into the `@local` image `<agent id>__agent_image`: pushed to the local registry, which starts on the first push, and saved as a tarball in the local object store. The run prints a line when a build starts. It is reused while every file of the folder stays the same, the `agent.toml` too, since a Dockerfile can copy it; a changed file rebuilds it, and rewrites the agent over the new image. What the build fetches, such as the base image a `FROM` tag names, isn't an input, so a newer one arrives only with the next rebuild. The image is built for this machine's platform and stays on this machine, so the local sandbox runs it; remote sandbox providers can't use it yet. The run needs `docker` on `PATH`, and refuses before any write without it.

A run that needs anything else written, such as an environment, is refused before any write.

agent-env ships one bundle, `hello`. Its task deploys a local sandbox (a work folder on this machine), loads the two files of `artifacts/greeting/` into it, and checks them with `verify_sandbox`: a file probe, and `bash check.sh`. It needs `bash` and the usual shell tools, and no Docker, model or configuration.

```bash
agent-env run hello
```

```
artifacts/greeting: v1 (new)
tasks/hello.json: v1 (new)
[tasks/hello.json] step 1/3 box (deploy_sandbox)
...
Tasks:
  tasks/hello.json v1: passed (hello: 1), 0.1s, instance @local/agentenv-framework/hello/hello-5aii6zzz
```

A second run writes nothing: it prints `artifacts/greeting: v1, unchanged` and `tasks/hello.json: v1, unchanged`, then runs the task again.

To change hello, copy its folder, which `agent-env run` prints under its row, and run the copy by path. The copy's ids are rooted at its own folder (`@local/~/my-hello/…` for a copy at `~/my-hello`), so it never shares a version with the installed one:

```bash
cp -R <the folder agent-env run prints> ./my-hello
agent-env run ./my-hello
```

`agent-env run NAME` runs a bundle an installed package provides, and `agent-env run` with no argument lists them with what each holds, marking as invalid one that fails the checks `run` makes before it reads a store (see [Bundles from installed packages](#bundles-from-installed-packages)). An argument that is an existing folder, or starts with `.`, `/` or `~`, is always a folder.

What it writes, and the instances it records, have `@local/` ids and land in the local stores, whatever stores are configured.

A rerun writes a new version only of what changed since the bundle last wrote it, and reuses the rest. A reused task keeps whatever its steps took from the configuration when it was first written, such as a `rubrics_verifier`'s default judge model.

- **What runs.** Every eval runs, or every task in a bundle without evals. `--task` and `--eval` select by name or id, and each is repeatable. When every eval runs, a task that no eval names doesn't run, and the command says how to run it.
- **How tasks run.** Each selected task, and each task a selected eval names, runs once. At most four run at once, and they share one `run_group_id`.
- **Overrides.**
  - `--model` sets the agent's model. The judge keeps its own, so scores stay comparable across models.
  - `--sandbox` sets the provider for `deploy_env`, `deploy_agent`, `deploy_sandbox` and the judge's deploy. It takes a name or a comma-separated fallback chain, and an unknown one is refused before anything is written. A chain can't create a VM, so under one a `deploy_sandbox` step in `vm` mode fails, and so does an environment behind a gateway.
- **Results.** The summary gives each task's outcome, its scores labelled by the step that recorded each, its duration and its instance id, then each eval's pass count.
  - A task passes when it recorded a score and every score is at least 1. A task that recorded no score is `unscored`.
  - The command exits 1 when a run raised or left a failed step; a score below 1 doesn't change the exit status. With `agent-env --verbose run`, each run that raised also prints its traceback.
  - `agent_env.bundle.run_bundle()` returns the same results, contexts included, to Python. It runs its own event loop, so from async code call it in a thread: `await asyncio.to_thread(run_bundle, path)`.
- **Teardown.** Each run's sandboxes are torn down as it ends, passed or failed, and a local sandbox's work folder goes with them. Teardown removes compute only: the instance, its outputs and the artifacts it collected stay. The summary ends with `Tore down N sandboxes.`, and lists any sandbox it couldn't terminate as still up.
  - `--keep` holds them up instead: after the summary it prints each run's sandboxes with the endpoints they record (MCP, gateway, pgweb, tunnel URLs, the local work folder), waits, and tears them down on Ctrl-C. From Python, `run_bundle(path, keep=True)` leaves them up and `result.teardown()` removes them; a signal during it raises `RunInterrupted` with what it reached.
  - Ctrl-C or SIGTERM during the runs cancels them: each run that started is marked `cancelled` in the store and torn down, a run that was waiting shows `didn't start`, the summary prints, and the command exits 130 (143 for SIGTERM). A second Ctrl-C stops the teardown and prints what is still up. From Python, `run_bundle()` raises `RunInterrupted`, a `KeyboardInterrupt` whose `result` holds the runs.

Known limits:
- An eval in a bundle runs only the bundle's own tasks. One that names a store task is refused; run that task with `agent-env task run`.
- A step that writes an entity under an id it makes up is refused when it runs, since that id isn't under `@local/`. This refuses `run_container_unit_tests_verifier`'s stdout artifacts and `collect_artifacts`, and `snapshot_env` unless the task names its `snapshot_id`. `verify_sandbox` writes nothing, so it runs.
- A step's reference to one of the bundle's entities names no version, so it reads the latest version when the step runs. Another run of the same folder, made after an edit, can write a newer one first.
- Teardown reaches only the sandboxes a run's context records. A `rubrics_verifier`'s judge, and an agent whose deploy was cancelled, terminate their own sandboxes but leave their local work folders. A run killed outright (`kill -9`) leaves everything it deployed.
- A process a local sandbox's command detaches from itself (a double fork, `setsid`) outlives the command and the run.
- A run still opens the configured stores: it reads store entities through them and sets up their indexes. Under a config whose stores are remote, their credentials must be available, even for a bundle, such as `hello`, that writes nothing there.

### The local explorer and runner

`agent-env up` starts the local control plane: it resolves the configured stores, bootstraps the two environments every deploy needs, then serves a loopback HTTP API. It requires a discovered `.agentenv/config.toml` (or `AGENT_ENV_CONFIG`) and the `explorer` extra (`pip install 'agentenv-framework[explorer]'`), and starts no environment containers itself (the first image push starts the `agentenv-registry` container; see [Start the local stack](#start-the-local-stack)).

```bash
agent-env up
```

The first run builds and registers the `default-db` service-db environment and the `default` gateway environment (four images, a few minutes, Docker required). Later runs report `already registered` and are ready in about a second; `agent-env up --no-bootstrap` skips the build block. Stop with Ctrl-C.

```bash
curl http://127.0.0.1:8234/health
```

`/health` returns the resolved document store class and runner type. The port comes from `[explorer] port` (default 8234); there is no port flag, and a busy port fails only after bootstrap. The host is always `127.0.0.1`; foreign `Host` headers get `421` unless listed in `[explorer] allowed_hosts`.

| Path | Purpose |
|---|---|
| `GET /api/docs`, `GET /openapi.json` | Swagger UI and schema. `/docs` is 404 from a source checkout: no UI is packaged, the explorer is API-only. |
| `GET /api/v1/{envs,tasks,agents,artifacts,evals}[/{id}[/versions]]` | List and read every primitive. The CLI has no `list` commands; this is the list surface. |
| `POST /api/v1/tasks/{task_id}/run`, `.../runs`, `.../cancel-run` | Submit one or N runs to the configured runner; cancel one. |
| `GET /api/v1/tasks/{task_id}/instances[/{id}[/progress]]`, `.../run-groups[/{id}[/stream]]` | Poll status per step; follow a run group. |
| `GET /api/v1/objects/{content,metadata}?object_url=...` | Read any stored object, trajectories included. |

Runs submitted through the API go to the `[runner]` seam. The bundled `LocalRunner` runs them in-process (`workers = 2` by default) and is not durable: runs die with the process. `agent-env task run` bypasses the runner and calls `Task.run()` directly.

### Resume and partial runs

Every context JSON is a restart point. `--context-json` restores it and `--start-step` (0-based) says where to continue; earlier steps are marked successful without executing and a new instance id is minted.

```bash
agent-env task run --id <task-id> --version 2 --start-step 3 --context-json <path>/<task-id>_<8hex>.json --output-dir <path>
```

This re-ran a direct-LLM `rubrics_verifier` from the persisted trajectory in about 8 seconds, then the `snapshot_env` step, which needs the environment named in the context to be running still (it was, so a second snapshot artifact was written). When that environment is gone, the tolerant snapshot step (`fail_task_on_error: false`) lands in `metadata.failed_steps` and the run still exits 0. The new run keeps `deployed_envs[0].instance_id` from the loaded context. Progress is pollable per step at `.../instances/{id}/progress` or with `task get-instance`. There is no `--resume` flag.

### What every run needs

The canonical chain (`deploy_env`, `deploy_agent`, `prompt_agent`, `rubrics_verifier`, `snapshot_env`) ran end to end on the local Docker sandbox in 41 seconds with the echo agent once these were in place. Each item names the error you see when it is missing, in the order a run surfaces them.

1. The bootstrap environments `default-db` and `default` in the same document store; the two `put` commands are in [Start the local stack](#start-the-local-stack). Missing: `NotFoundError: Env default-db not found` at `deploy_env`.
2. An agent registered under the id `deploy_agent` resolves: `[agents] default_a2a_agent_id` in `.agentenv/config.toml`, `a2a-default` when unset, or `--a2a-agent-id` per run; the command is in [Register an agent](#register-an-agent-the-bundled-echo-agent). Missing: `A2AAgent a2a-default not found` at `deploy_agent`.
3. A model key and endpoint: `LITELLM_API_KEY` and `LITELLM_BASE_URL` exported, or `[model] api_key` / `base_url`. `deploy_agent` resolves them when it creates the agent container (`A2AAgent.deploy`), unless the agent was registered with both as `--env-var` defaults. Missing, at `deploy_agent`: `ConfigError: No model API key configured` when nothing is set, `ConfigError: No model endpoint configured` when only the endpoint is missing. A `rubrics_verifier` needs them again: its direct judge stops with `ConfigError: No model endpoint configured for '<model>'` when the endpoint is missing, and its agent judge deploys the judge agent through the same `A2AAgent.deploy` check. `agent_prompt_response_verifier` needs no model.
4. A running Docker daemon for the `local` sandbox, and a `--project-id` label on `task create` (required; any string).

`task run` never tears down, so a fatal step leaves earlier deployments running; close them as described in [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down).

### At scale: bring your own durable runner

agent-env ships one runner, `LocalRunner`. Anything durable, queued or multi-host is a `Runner` subclass registered through the `[runner]` seam; the contract is in [Custom sandbox, state, environment and runner providers](#custom-sandbox-state-environment-and-runner-providers).

```toml
[runner]
impl = "agent_env.runner.local_runner:LocalRunner"

[runner.config]
workers = 2
```

`Runner` has three abstract methods, `submit`, `status` and `cancel`, plus optional `start` and `stop`. `submit` returns a `RunHandle(run_id, instance_id)`, `status` a `RunRecord`, and `RunStatus` is `QUEUED | RUNNING | COMPLETED | FAILED | CANCELED`. The runner's `type` is stamped on every run record, printed by `agent-env up` and returned by `/health`. `AGENT_ENV_RUNNER=local` overrides the table; other alias strings raise a `ConfigError` asking for an explicit `[runner]` table.

A worker needs three things the core already provides. It loads the same config (`AGENT_ENV_CONFIG` or the discovered file) so ids resolve against the same stores. It reconnects by type: `Env.from_instance_id(<instance-id>)` rebuilds a deployed environment, and sandbox and state providers are found by the registry name that must equal the produced `.type`. It resumes with `TaskStepContext.from_dict(...)` and `Task.run(start_step=..., context=...)`, the mechanism behind the resume flags above.

## Read results, evaluate and iterate

### The run record

A run has two records: the persisted `TaskInstance` in the document store and the context JSON in `--output-dir`.

```bash
agent-env task get-instance --id <instance-id>
```

This prints `Instance ID`, `Task ID`, `Task Version`, `Status` (`running | completed | failed`), `Progress` (`<done>/<total>`), `Created At (UTC)`, `Completed At (UTC)` and the full `Context`. The instance id is printed at the top of every run and stored as `instance_id` in the context JSON. There is no instance list command; keep the id, or query `GET /api/v1/tasks/{task_id}/instances` on the [explorer](#the-local-explorer-and-runner). Instance fields also include `current_step`, `total_steps`, `error`, `completed_steps[{step_id, status}]` and `step_attempt_failures`.

The context JSON is `context.to_safe_dict()`, which strips credential keys such as `litellm_api_key` and `cf_access_client_secret` at any depth. Top-level keys:

| Key | Contents |
|---|---|
| `deployed_envs[]` | `env_id`, `env_version`, `gateway_url`, `mcp_url`, `db_web_url`, `db_mcp_url`, `environment_card_url`, `sandbox_id`, `sandbox_type`, `instance_id`, `created_at_utc`, `expires_at_utc`, `gateway_mode`, `env_state_instance_ids` |
| `deployed_agents[]` | `agent_name`, `api_url`, `a2a_url`, `sandbox_id`, `sandbox_type`, `a2a_card`, `instance_id`, `role`, `network_policy` |
| `deployed_sandboxes[]` | Bare sandboxes from `deploy_sandbox` steps |
| `prompt_responses[]` | `prompt_id`, `prompt_text`, `response`, `structured_output`, `tool_call_count`, `model`, trajectory URLs, `error_type` / `error_code` / `error_message`, `agent_name`, `step_id` |
| `metadata` | `run_group_id`, `task_id`, `user_overrides`, `verifications`, `env_snapshotted_universes`, `failed_steps`, `retry_resets`, `seed`, `universe_id` |
| `agent_model`, `default_agent_model`, `agent_artifact_id`, `agent_harness`, `instance_id` | Run-level overrides and identity |

### Verifications and scores

Every verifier step writes `metadata.verifications[<verifier_id>]` (the contract is in [Verifier contract, and why validators are not rewards](#verifier-contract-and-why-validators-are-not-rewards)). Each entry holds `score`, the aggregate over its rows using the step's `score_aggregator` (`all_pass`, `any_pass`, `weighted_average`), plus fields that depend on the verifier:

| Verifier | `results[]` rows | Other entry fields |
|---|---|---|
| `rubrics_verifier` | your criterion (`id`, `criterion`, `weight`) plus the judge's `score`, `result`, `justification`; a skipped run has one row `{id: "prompt_error" \| "no_screenshots", score: 0, result: false, message}` | `format` (`rubric_binary`, `rubric_partial`, `trajectory_mistakes`), `compact_trajectory_s3_uri`, `judge_trajectory_s3_uri`, and `judge_output_retries` when the judge had to be re-asked |
| `agent_prompt_response_verifier` | your criterion dict copied whole (`type`, `needles` or `pattern`, any `id`) plus `criterion_index`, `score`, `result`, `justification`; on a failed prompt one row `{id: "prompt_error", score: 0, result: false, message}` | none |
| `verify_sandbox` | your criterion dict plus `criterion_index`, `score`, `result`, `justification`, `skipped: false`; criteria of unknown type keep `criterion_index`, `justification`, `skipped: true` and no score | none |
| `env_outcome_verifier` | whatever your `verify(mcp_url)` returned | none |
| `run_container_unit_tests_verifier` | one row `{id: "exit_code" \| "reward", score, result, message}` | `command`, `exit_code`, `timed_out`, `stdout_artifact`, `stderr_artifact`, `stdout_head`, `stderr_head`, `extracted_files` |
| `aggregate_verifiers` | none | `source_verifier_ids` |

The aggregator reads `result`, `score` and `weight`; rows flagged `skipped` are excluded and rows without `result` count as failed. A prompt that errored skips verification with score 0. Failed steps land in `metadata.failed_steps[]` with `step_id`, `step_type`, `error`, `error_type`, `started_at_utc`, `duration_seconds`, `is_fatal` and, after a retry, `retried: true` (see [Step-level retry](#step-level-retry-retry_config)).

`snapshot_env` records `metadata.env_snapshotted_universes[<step_id>] = {id, version}`, keyed by step id, so a task with several snapshot steps yields one entry per step.

### Trajectories

The agent trajectory is written to the object store and referenced from `prompt_responses[i].agent_trajectory_s3_uri`. The field names keep their legacy `s3` spelling, but the value is an object URL for whatever store is configured: `file://...` on the local store, `s3://...` on an S3 store. The object is whatever the agent's `urn:agentenv:trajectory/v1` extension reports for the prompt, as JSON; the format is agent-defined. An agent that advertises the object form uploads it itself through a signed grant when the store issues grants, as `S3ObjectStore` does; otherwise, and always on the local default store, the agent returns it inline and agent-env writes it (see [Object transfer](packages/agentenv-protocol/README.md#object-transfer)). The bundled echo agent reports its native format, one `{"type": "echo", "input": ..., "output": ...}` entry per prompt. An agent that reports OpenTelemetry spans stores a list of spans, each with `name`, `context`, `kind`, `parent_id`, `start_time`, `end_time`, `status`, `attributes`, `events`, `links` and `resource`; the LLM judge reads them through the GenAI semantic conventions: it selects spans by `gen_ai.operation.name` (`chat` for model turns, `execute_tool` for tool calls, `chain` for the conversation root), takes the tool label from the span `name`, the arguments from `gen_ai.prompt` (`input`) and the result from `gen_ai.completion` (`output`). No span-name convention is required; the judge compacts such trajectories before grading, and a list with no `gen_ai.operation.name` attribute is passed through unchanged.

No CLI downloads a trajectory. Read it through the object store in Python, or via the explorer's `GET /api/v1/objects/content?object_url=...`:

```python
import json
from agent_env.config import get_config

context = json.load(open("<output-dir>/<task-id>_<hex>.json"))
data = get_config().get_object_store().get(context["prompt_responses"][0]["agent_trajectory_s3_uri"])
```

When a trajectory filter runs, `compact_trajectory_s3_uri` points at the compacted form the judge saw. The gateway keeps its own environment-side trajectory of tool calls and results, served as NDJSON at `<gateway-url>/trajectory`; see [Gateway and agent controls for experiments](#gateway-and-agent-controls-for-experiments).

### Collected artifacts and snapshots

`snapshot_env` exports the environment's state after the run into an `EnvironmentUniverseArtifact` named `snapshot-<env-id>-<instance-suffix>`, with one `EnvironmentArtifact` per environment name inside. Its id and version are in `metadata.env_snapshotted_universes[<step_id>]`. Fetch it by id:

```python
from agent_env.artifact import Artifact

art = Artifact.get("<snapshot-artifact-id>", 1)
```

Its `type` is `environment_universe`, and it lists its members under `environment_artifact_refs`. The built-in artifact types are `cli`, `docker_image`, `environment`, `environment_universe`, `file`, `file_artifact_universe`, `skill` and `vm_image`. A store written by an older release under a renamed spelling is read through `[artifacts] type_aliases` in `.agentenv/config.toml` (`legacy_name = "canonical_name"`): the stored string is mapped on read and `type` filters match both spellings. An alias must point at a registered type, may not chain, and may not rename a type that is itself registered to a different class; agent-env ships no aliases of its own. The CLI download is `agent-env artifact environment-universe get --id <id> --output-dir <dir>` (from `--help`). A snapshot is a universe artifact like any other; loading universes is covered in [Load universe data, snapshot and export](#load-universe-data-snapshot-and-export). `collect_artifacts` (file universes from the agent sandbox) and `snapshot_agent_state` (agent-side captures) are further steps in the registry that record their outputs the same way; not exercised for this guide.

### Evals: run a task set under different agents and models

An eval is a versioned document `{id, version, tasks: [{task_id, task_version?}]}`. Create it from a JSON list of task references; every task must already exist.

```json
[
  {"task_id": "<task-id>", "task_version": 1}
]
```

```bash
agent-env eval create eval_tasks.json --id <eval-id>
```

`eval add-tasks <file> --id <eval-id>` merges more references by `task_id` and writes a new eval version. `eval run` runs every task `--k` times concurrently:

```bash
agent-env eval run --id <eval-id> --k 1 --output-dir <path>
```

Output is compact: one `[<task-id>] Completed step ...` line per step, `Output written to: <path>/<task-id>_<8hex>.json` per run, then a `Results:` block with `[<task-id>] PASSED` or `FAILED: <error>`, and exit 1 if any run failed. `PASSED` means the task completed; it carries no score because `eval run` reads `metadata.score`, which verifiers do not set. Scores live in each run's JSON under `metadata.verifications`, or in `task get-instance` using the `instance_id` inside the JSON.

To compare agents or models, run the same eval again with `--agent-model <model>` or `--agent-artifact-id <artifact-id>` (and `--max-concurrency N`). Aggregation across runs is yours. `eval run` has no `--a2a-agent-id` and no sandbox overrides, and there is no `eval get` or `eval list`; read an eval through the explorer API. It tears down each run's sandboxes as the run ends, whether it passed, failed or was interrupted, and prints `[<task-id>] Tore down N sandboxes`; a run's context JSON still records them. Ctrl-C or SIGTERM cancels the runs still going, lets a teardown already under way finish, lists the cancelled runs as `CANCELLED` and exits 130 (143 for SIGTERM); a second one stops the teardown.

### Iterate safely

Every `put` and `create` appends a version; nothing is overwritten. Creating a task again under the same id yields version 2, and `task run --version` picks any earlier one. Tasks embed their steps inline, so later edits never change a stored version. Step documents are also versioned under their own step id in a shared collection; if many tasks share one store, prefix step ids with the task name.

Use per-run overrides for anything the flags in [Run from the CLI](#run-from-the-cli) cover: model, agent, sandbox slots, state, tokens. Change a step field (prompt text, criteria, `trajectory_output_prefix`) by creating a new task version. Check it with `task validate` before running; see [Validate before you deploy](#validate-before-you-deploy).

To re-grade without redeploying, resume at the verifier step from a saved context: the direct-LLM judge reads the persisted trajectory alone. A `snapshot_env` artifact preserves the end state of a run as a universe you can load again, so a follow-up experiment can start from where an earlier one stopped rather than from the seed.

## Configuration reference

agent-env reads exactly one TOML file, `.agentenv/config.toml`, plus a few environment variables. Every backend is a seam: a table naming a Python class that agent-env builds on first use. With no file, every seam falls back to a local implementation.

### Config discovery and precedence

Exactly one file is read, and it is used whole. There is no layering: no user-level file, no merge of a project file over anything else. Resolution is:

1. If `AGENT_ENV_CONFIG` is set, that path is the file. It must exist: every command fails at once with `ConfigError: AGENT_ENV_CONFIG='<path>' does not point to an existing file` rather than falling back to the local defaults (`config show` reports the same text as an error instead of failing).
2. Otherwise the nearest `.agentenv/config.toml`, walking up from the working directory through every parent to `/`.
3. Otherwise no file, which parses as an empty table.

The `local` document and object stores keep their state in one per-user root, whatever config file was found: `$XDG_STATE_HOME/agent-env/` when `XDG_STATE_HOME` is an absolute path, else `~/.local/state/agent-env/` (`%LOCALAPPDATA%\agent-env\` on Windows). Every project you run from shares that one store, and `.agentenv/` holds only config. A `[stores.document]` table with its own `path`, or a `[stores.object]` table with its own `root`, puts that store somewhere else. Each store directory is created on the first write, readable only by you, with a `.gitignore` of `*`.

A section the file does not define falls back to a code default, never to another file: a file that sets only `[stores.document]` leaves the other three stores on `local`. Several of those defaults are a real selection rather than "off", so read the last column as what you get. Where a section has an environment variable, the variable wins over the file:

| Section | Env var | Fallback when the section is absent |
|---|---|---|
| `[stores.document]`, `[stores.object]`, `[stores.image]`, `[stores.secret]` | `AGENT_ENV_DOCUMENT_STORE`, `AGENT_ENV_OBJECT_STORE`, `AGENT_ENV_IMAGE_STORE`, `AGENT_ENV_SECRET_STORE` | the `local` alias |
| `[runner]` | `AGENT_ENV_RUNNER` | the `local` alias: the in-process `LocalRunner` |
| `[conversations]` | `AGENT_ENV_HUMAN_A2A_URL` | unconfigured; `ConfigError` where a human A2A endpoint is consumed |
| `[agents]` | none | `default_a2a_agent_id = "a2a-default"`, the agent a `deploy_agent` step without an id deploys |
| `[model]` | `LITELLM_BASE_URL`, `LITELLM_API_KEY` | unconfigured; `ConfigError` where a model endpoint or key is needed |
| `[sandbox]` | none | `default = "local"`, `agent_default = "local"`; built-in providers only |
| `[state]` | none | a fresh environment state store is `local_postgres`; built-in providers only |
| `[envs]`, `[artifacts]`, `[task_steps]` | none | built-in types only; no plugin classes registered |
| `[explorer]` | none | `port = 8234`; no explorer plugins |
| `[plugins.<package>]` | none | nothing: agent-env reads none of it; each plugin reads its own table (see [Plugin settings](#plugin-settings)) |

`agent-env config show` answers the same question for the install you are running and names the module each fallback comes from. Prefer it over this table, which is a snapshot of one release's defaults; see [Inspect the resolved configuration](#inspect-the-resolved-configuration).

Precedence per seam, lowest to highest: built-in default, config table, `AGENT_ENV_*` variable, an explicit `configure(...)` or `set_*_store(...)` call in code. Environment variables accept only alias strings such as `local`; hosted backends need a table. An alias string selects a built-in with default coordinates. A table with `impl = "module.path:ClassName"` and an optional `config` sub-table selects any class:

```toml
[stores]
document = "local"   # SQLite      ~/.local/state/agent-env/document_store/documents.db
object   = "local"   # filesystem  ~/.local/state/agent-env/object_store/

[stores.image]
impl = "agent_env.store.image_store:LocalRegistryImageStore"
config = { registry_host = "localhost:5000", repository_prefix = "<prefix>" }
```

A key may use only one shape; `image = "local"` plus a `[stores.image]` table is a TOML duplicate-key error. Any string value may be a reference: `env:NAME` reads an environment variable, `secret:KEY` reads the secret store, and `?default` supplies a fallback. An unresolved reference without a default is a `ConfigError`. Errors surface when the seam is first consumed, not at process start.

Re-pointing `AGENT_ENV_CONFIG` inside a running process takes effect only after `agent_env.config.reset_config()`. The `Config` resolves the file once, on first use, and holds that document for its life; every reader, the sandbox, state, environment, artifact and task-step registries included, reads it rather than discovering the file again. Set the variable before anything resolves, which is what a CLI plugin's root option does (see [CLI plugins, root options, explorer routes](#cli-plugins-root-options-explorer-routes)). `configure(...)` rebuilds the process-wide config from the file `AGENT_ENV_CONFIG` currently resolves to; it also accepts `default_a2a_agent_id`, `default_human_a2a_url` and explicit `document_store` / `object_store` / `image_store` / `secret_store` objects.

### Inspect the resolved configuration

Every layer resolves silently, including the bottom one: with no config file above the working directory, agent-env picks the four `local` backends and says nothing. `agent-env config show` prints the file that won, how it was found, and for every section the resolved value and the layer that supplied it. With the example config in place:

```console
$ agent-env config show
config:         <project>/.agentenv/config.toml
                discovered by walking up from the working directory

document:       LocalSqliteDocumentStore  path=<home>/.local/state/agent-env/document_store/documents.db
                from [stores.document]
object:         LocalFilesystemObjectStore  root=<home>/.local/state/agent-env/object_store
                from [stores.object]
image:          LocalRegistryImageStore  registry_host=localhost:5000
                from [stores.image]
...
model:          (unset)
                from default in config.runtime
agents:         default_a2a_agent_id=a2a-default
                from [agents]
sandbox:        default=local  agent_default=local
                from [sandbox]
...
```

Fourteen sections are reported, followed by a `plugins:` block that lists each `[plugins.<package>]` table under the distribution it belongs to (see [Plugin settings](#plugin-settings)). With no file the header reads `config: (none)` and every store `from built-in default`. When an environment variable beats the file, the section says so and names what it shadowed: with `AGENT_ENV_DOCUMENT_STORE=local` exported against a file whose `[stores.document]` names another class, the `document` block reads `from $AGENT_ENV_DOCUMENT_STORE` followed by `; <class> in [stores.document], shadowed`. That line is the point: one variable can downgrade a single store while every other section stays on the file, and nothing else in the system says so. Sections resolved through an alias report the class the name became; the rest report what the file contributes and, when absent, the module their fallback lives in. A section that fails validation is reported in place as `(unresolved)` with the `ConfigError` text, for example an empty `[agents] default_a2a_agent_id`.

`config show` also warns, under the header, about anything in the file that nothing reads: a top-level section's name bound to the table above it by a bare key, a key set both beside a `config` table and inside it, and a table or key agent-env does not have. The last names the one meant when one is close, so `[sanbox]` gets `Did you mean [sandbox]?` and `[envs] implz` gets `Did you mean 'impls'?`; otherwise it says what the table takes, that a seam's class takes its settings under `config`, or, for a top-level table, that a plugin's own settings go under `[plugins.<package>]`. The file is checked as written, so a typo in a table an environment variable replaces is still reported. Tables whose keys a class, a plugin or you choose are not checked: a seam's `config`, provider names and their `config`, `[sandbox.attribution]`, `[model.roles]`, `[model.params]`, `[artifacts.type_aliases]` and everything under `[plugins]`. The warnings are advice: `config show` still exits 0, `--json` lists them in `warnings`, a service that logs the report logs them, and `plugin check` does not fail on them.

`agent-env config debug` answers the other question, why that file: it prints every path discovery considers, in order, marking the winner with `->` and each with `(walk-up, exists: yes|no)`; with `AGENT_ENV_CONFIG` set, the single line reads `($AGENT_ENV_CONFIG, exists: yes)` and `config show` reports `via $AGENT_ENV_CONFIG`. Every command in the group takes `--json`; the JSON keeps the provenance (`winner`, `shadowed`, `impl`, `config`, `error` per section, plus `config_path` and `config_source`) rather than flattening to effective values. `config show --json` also has a `plugins` object, `{error, tables}`, where each table is `{name, installed, version, plugin, keys, value, error}`: `installed` says a distribution of that name is installed, `plugin` that it declares an `agent_env.*` entry point, and `keys` lists the table's keys as the file spells them. All are read-only: no store is constructed, nothing is fetched, and no `.agentenv/` directory is created. Secret values never appear: an `env:` or `secret:` reference prints as the reference (its `?default`, a literal, is masked like one), and a literal under a secret-shaped key (`api_key`, `token`, `password` and similar) prints as `***`, as does anything nested beneath one.

`agent-env config explain <path>` narrows `show` to one value. `<path>` is a section name or a TOML path — `document` and `stores.document` both work — and the output is that section's block on its own: the resolved value, `from <layer>`, and any `; ... shadowed` lines. A path *above* a section — `stores`, say — has no single winner, so it lists the sections under it and how each resolved rather than echoing a file table a higher layer may already have replaced. A path *below* a section is read out of whichever layer won that section, not out of the file — so with `AGENT_ENV_DOCUMENT_STORE=local` set, `config explain stores.document.config.database` reports what the local backend resolves to and names the shadowed file table, rather than echoing a database name the process never reads. Keys the file is not the only source for (`model.api_key`, `agents.default_a2a_agent_id` and the like) resolve through their own layers, so an environment override or a built-in default is reported as the winner. A path nothing supplies reads `(unset)`, and one only the file knows about says so. A `plugins.<package>` path is read from that plugin's table, with the package matched by canonical name. Its last line says whose table it is and that agent-env does not read it; agent-env cannot tell whether the plugin reads that key. `--json` puts the owner in `plugin`, `{name, installed, version, plugin}`, which is `null` for every other path. A package segment may be quoted as `config show` prints it, `plugins."AgentEnv.Toy".region`, or dotted as `plugin list` prints it.

`agent-env config sources` answers the question between the other two — not which file, and not what each section became, but which *layers* are in play at all. It lists them lowest precedence first, marking each present one with `->` and naming what each shadows:

```
-> default  built-in defaults
-> file     /home/you/.agentenv/config.toml  (via $AGENT_ENV_CONFIG)
   env      $AGENT_ENV_DOCUMENT_STORE -> [stores.document]
-> env      $AGENT_ENV_OBJECT_STORE -> [stores.object]  (s3)
   env      $AGENT_ENV_IMAGE_STORE -> [stores.image]
   env      $AGENT_ENV_SECRET_STORE -> [stores.secret]
   env      $AGENT_ENV_RUNNER -> [runner]
-> env      $LITELLM_API_KEY -> [model.api_key]  (***)
   env      $LITELLM_BASE_URL -> [model.base_url]
   env      $AGENT_ENV_HUMAN_A2A_URL -> [conversations.default_human_a2a_url]
   env      $AGENT_SANDBOX_MODE  (no file equivalent)
   env      $AGENT_ENV_MODAL_REGION  (no file equivalent)
   env      $AGENT_ENV_MODAL_APP_NAME  (no file equivalent)
   env      $AGENT_ENV_FIXTURE_PREFIX  (no file equivalent)
   env      $MODAL_TOKEN_ID  (credential, no file equivalent)
   env      $MODAL_TOKEN_SECRET  (credential, no file equivalent)
```

The layers contributing nothing are listed on purpose: "the file I edited is not being read" and "an environment variable I forgot about is overriding it" are the two questions this answers, and neither is visible from a list of what won. The file line says *how* it was found, because `$AGENT_ENV_CONFIG` is terminal over a discovered file and conflating the two would hide which layers could still apply. An unreadable file is reported in place rather than raising, and an override's value is masked on the same terms as everywhere else. `env:NAME` references the active file makes are listed too — they are not a precedence layer, since a reference resolves inside the document that wrote it, but they are still variables supplying values, and only their names are printed, never their values. A variable that is both a declared override and referenced by the file gets **one** row carrying both facts, and a variable referenced from more than one key names every one of them — one variable feeding two keys is the coupling worth seeing. The list is not numbered: the precedence chain is the array's own order, and publishing a layer number that later renumbers would be worse than publishing none.

### Stores and secrets

| Seam | Alias | Built-in implementations | Env var |
|---|---|---|---|
| `[stores.document]` | `local` (SQLite) | `LocalSqliteDocumentStore`, `MongoDocumentStore`, `DynamoDbDocumentStore` (table only), `FirestoreMongoDocumentStore` (`gcp` extra, table only) | `AGENT_ENV_DOCUMENT_STORE` |
| `[stores.object]` | `local` (filesystem) | `LocalFilesystemObjectStore`, `S3ObjectStore` (table only), `GcsObjectStore` (`gcp` extra, table only) | `AGENT_ENV_OBJECT_STORE` |
| `[stores.image]` | `local` (registry at `localhost:5000`) | `LocalRegistryImageStore`, `OciRegistryImageStore`, `EcrImageStore` | `AGENT_ENV_IMAGE_STORE` |
| `[stores.secret]` | `local` (process env vars) | `LocalSecretStore`, `AwsSecretsManagerSecretStore`, `GcpSecretManagerSecretStore` (`gcp` extra, table only) | `AGENT_ENV_SECRET_STORE` |

`DynamoDbDocumentStore` keeps each collection in its own on-demand table, `{table_prefix}{collection}`, created on first use. It authenticates through the standard boto3 credential chain; that identity needs `dynamodb:CreateTable` (only if agent-env creates the tables), `DescribeTable`, `GetItem`, `PutItem`, `DeleteItem`, `Query` and `Scan` on the prefixed tables. Items are keyed by the collection's unique index, and the table's sort key is named for that index's fields as a JSON list (`sk` when it has none), so every process reads the keying from the table itself; a pre-created table needs a string hash key `pk` and a string sort key named that way. The index must exist before the collection's first write; agent-env ensures each collection's index before writing it, except conversations, so call `agent_env.a2a_agent.conversation_store.ensure_indexes()` once before the first multi-turn run. A read that pins every index field is a GetItem, one that pins the first is a Query, and any other read scans the whole table. A document is limited to DynamoDB's 400 KB item size:

```toml
[stores.document]
impl = "agent_env.store.document_store:DynamoDbDocumentStore"
config = { table_prefix = "agentenv_", region = "<region>" }
```

`FirestoreMongoDocumentStore`, from the `gcp` extra, keeps documents in a Firestore database with MongoDB compatibility (Enterprise edition). It takes `host`, which is `<uid>.<location>.firestore.goog` (`gcloud firestore databases describe --database=<database> --format='value(uid)'` prints the uid), and `database`. It signs in with an access token of the Application Default Credentials, so it runs wherever those do, on Google Cloud or off it. That identity needs `roles/datastore.user` for the documents and `roles/datastore.indexAdmin` for the index creation every process runs when it starts:

```toml
[stores.document]
impl = "agent_env.store.document_store.firestore_mongo_document_store:FirestoreMongoDocumentStore"
config = { host = "<uid>.<location>.firestore.goog", database = "<database>" }
```

It is `MongoDocumentStore` with the changes Firestore needs. Retryable writes are off. When two guarded writes race on one document, Firestore tells both that they matched, so every update and replace stamps an `_agentenv_write` field with a fresh value and counts as applied only when its own stamp landed, and an update that returns the document's previous state runs in a transaction, retried on write conflicts. Reads through the store drop the field. Raw pymongo access to the same collections (`Config.db`) gets none of this: it sees the field, and a raw guarded write that loses a race is still reported as matched. An upsert that loses an insert race is retried once, as MongoDB's server does for itself. Building an index takes about a minute even on an empty collection, and `ensure_index` waits for it, so the first process on a new database takes that long for each collection it touches. Threads in that process wait for the same build; let the process finish before starting others, because an index another process is still building does not yet reject duplicates, and duplicates written then leave a unique index needing repair. Documents nest at most 20 levels deep.

The aliases `mongo`, `s3`, `ecr` and `aws` are recognized but have no built-in coordinates; using one raises `ConfigError` until you supply the table. The local image store starts a `registry:2` container named `agentenv-registry` on first push. Private registry credentials go in `[stores.image.config] credentials`.

`S3ObjectStore` takes `bucket`, an optional `region` and `share_credentials` (default `false`), and authenticates through the standard boto3 credential chain (environment variables, shared config or SSO profile, instance or task role):

```toml
[stores.object]
impl = "agent_env.store.object_store:S3ObjectStore"
config = { bucket = "<bucket>", region = "<region>" }
```

With `share_credentials = true`, the AWS credentials boto3 resolves are frozen and handed to every agent container agent-env deploys and to every environment service that advertises `urn:agentenv:add-s3-credentials/v1` during `snapshot_env`. They carry that identity's whole IAM scope, not just the bucket.

`GcsObjectStore`, from the `gcp` extra, takes `bucket` and an optional `project` and `signing_service_account`, and authenticates through Application Default Credentials (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, or the attached service account). That identity needs to create, read, list and overwrite objects in the bucket (overwriting takes `storage.objects.delete`), as `roles/storage.objectUser` allows. Its object URLs are `gs://<bucket>/<key>`, and like the S3 store's they reach any bucket those credentials can read. The store's reads return an object's bytes as stored, so a gzip-encoded object comes back compressed, matching its reported size; Cloud Storage decompresses one fetched through a signed URL or grant, so the protocol client refuses those as oversize. Each write-once object carries an `agentenv-write-id` metadata entry, which lets a create the client retried after a lost response recognize its own write:

```toml
[stores.object]
impl = "agent_env.store.object_store.gcs_object_store:GcsObjectStore"
config = { bucket = "<bucket>", signing_service_account = "<name>@<project>.iam.gserviceaccount.com" }
```

Signed URLs and grants need something to sign with. With `signing_service_account` set, the store always signs as that account through IAM, which needs the same object access, while the store's identity needs `iam.serviceAccounts.signBlob` on it (`roles/iam.serviceAccountTokenCreator`); Google guarantees those signatures for twelve hours, and each is a network round trip. Without it, credentials that hold a service-account key sign locally, for up to seven days, and ADC of the `impersonated_service_account` type sign through IAM as their account; other ADC, such as a user login, an attached service account or Workload Identity Federation, sign nothing. `from_config` needs the network: it lists the bucket, and signs once when IAM signs, so a missing bucket or an unusable signer fails there. Without a signer the store signs nothing: a remote sandbox receives an object over its sandbox connection instead of fetching it, trajectories come back inline, and what needs a signed upload or a grant is unavailable, which includes `env snapshot`, GitHub image builds and the skill bundles, snapshots and changelogs agents move themselves. The object store never hands workloads Google credentials.

For Artifact Registry, point `OciRegistryImageStore` at the registry and give it `GoogleAccessTokenCredentials`, from the `gcp` extra. It hands out access tokens of a service account that the Application Default Credentials impersonate, so the identity behind them needs `iam.serviceAccounts.getAccessToken` on that account (`roles/iam.serviceAccountTokenCreator`):

```toml
[stores.image]
impl = "agent_env.store.image_store:OciRegistryImageStore"

[stores.image.config]
registry_host = "<region>-docker.pkg.dev"
repository_prefix = "<project>/<repository>"

[stores.image.config.credentials]
impl = "agent_env.store.image_store.google_credentials:GoogleAccessTokenCredentials"
service_account = "<registry-account>@<project>.iam.gserviceaccount.com"
```

The token is used for `docker login` inside sandboxes and handed to Modal as a registry secret, so give that account nothing but `roles/artifactregistry.writer` on the repository: writer rather than reader, because images built from GitHub are pushed from inside a sandbox. A token lasts an hour and is replaced once less than 45 minutes of it are left, so an image build can push at its end with the token it was given at its start. Create the repository beforehand; the store does not create repositories. The token can move any tag in the repository. The store logs in with the token for every image on the registry host, which Artifact Registry shares among all projects in a region. Without impersonation, `agent_env.store.image_store:SecretStoreCredentials` can read the registry's entry from a Docker config held in the secret store (under `registry_auths`), with the username `_json_key` and a service-account key as the password. That key does not expire and reaches every sandbox, so prefer impersonation.

`LocalSecretStore` reads process environment variables first (`use_env = true`), then an optional flat YAML or JSON file:

```toml
[stores.secret]
impl = "agent_env.store.secret_store:LocalSecretStore"
config = { file_path = "<path>/secrets.yaml", use_env = true }
```

`file_path` is resolved against the process working directory, not the config file, and must exist. `secret:` references cannot appear inside `[stores.secret]` itself. Store implementations are validated by the conformance suites in `tst/store/` (see [Conformance suites](#conformance-suites)).

`AwsSecretsManagerSecretStore` reads one AWS Secrets Manager secret holding a flat JSON (or YAML) mapping, so `secret:KEY` resolves to that mapping's `KEY`, read verbatim as a string. It authenticates through the standard boto3 credential chain, needs `secretsmanager:GetSecretValue` on the secret, and re-reads it every `ttl_seconds` (default 300):

```toml
[stores.secret]
impl = "agent_env.store.secret_store:AwsSecretsManagerSecretStore"
config = { secret_name = "<secret name or ARN>", region = "<region>" }
```

`GcpSecretManagerSecretStore`, from the `gcp` extra, reads one Google Cloud Secret Manager secret version holding the same kind of mapping, and re-reads it on the same schedule. It takes `secret_name` (the secret's id), an optional `project`, which defaults to the one Application Default Credentials resolve (`GOOGLE_CLOUD_PROJECT`, a key file's or the metadata server's), and an optional `version`, which defaults to `latest`. It authenticates through Application Default Credentials and needs `secretmanager.versions.access` on the secret, which `roles/secretmanager.secretAccessor` granted on that secret alone allows:

```toml
[stores.secret]
impl = "agent_env.store.secret_store.gcp_secret_manager_secret_store:GcpSecretManagerSecretStore"
config = { secret_name = "<secret id>", project = "<project>" }
```

`latest` is the newest version even when it is disabled, so undo a bad rotation by adding a corrected version: a disabled `latest` fails every new process's first read.

Both stores keep serving the last mapping they read when a re-read fails; only the first read, and an explicit `refresh()`, raise.

### Compute, state and runner

| Seam | Purpose | Default | Notes |
|---|---|---|---|
| `[sandbox] default` / `agent_default` | where environments and agents run | `local` / `local` | built-ins `local`, `modal`, `modal_vm`, `e2b`; comma-separated names form a fallback chain |
| `[sandbox.providers.<name>]` | extra or configured providers | none | `"module:Class"` string or `{impl, config}` table; built-in names accept a `config` table only |
| `[sandbox.attribution]` | optional labels copied into every sandbox request | empty | free-form key/value; ignored by the local sandbox |
| `[state.providers.<name>]` | durable environment state stores | `local_postgres` | the table name must equal the class's `type`; `local_postgres` is the only built-in and accepts no override |
| `[runner]` / `[runner.config]` | who executes runs submitted through the explorer API | `local` (`LocalRunner`, `workers = 2`) | `AGENT_ENV_RUNNER=local` overrides the table; see [The local explorer and runner](#the-local-explorer-and-runner) |

A chain such as `default = "my_cloud_flaky,my_cloud"` tries each provider in order and raises `RuntimeError: All N providers failed` when every member fails. Unknown provider names raise `ValueError`. Putting `impl` on a built-in name (`[sandbox.providers.local]`) is a `ConfigError`. The `e2b` provider's `config` table needs two keys: `api_key` (a secret reference such as `secret:e2b_api_key` or `env:E2B_API_KEY`) and `base_template` (an immutable, versioned E2B template with Docker Engine and the Docker Compose v2 plugin).

### Model, human endpoint, explorer, registries

An example `[model]` table and the per-run precedence order are in [Model configuration](#model-configuration-owns-precedence).

| Table | Keys | Failure mode |
|---|---|---|
| `[model]` | `base_url`, `api_key`, `default`, `roles`, `params` | unknown key: `ConfigError: [model] has unknown keys [...]`; `params` may not set `model`, `messages`, `api_key`, `api_base`, `user`, `metadata`, `timeout`, `response_format` |
| `[conversations]` | `default_human_a2a_url` | `ConfigError` when a human-in-the-loop step needs it and nothing is set |
| `[agents]` | `default_a2a_agent_id` | the agent a `deploy_agent` step without `a2a_agent_id` (and a `rubrics_verifier` without `judge_a2a_agent_id`) deploys; precedence `configure(default_a2a_agent_id=...)`, then `[agents]`, then the built-in `a2a-default`; no env var; the value may be an `env:` / `secret:` reference; a blank value or any other key under `[agents]` is `ConfigError` (`[agents] has unknown keys [...]; allowed: ['default_a2a_agent_id']`) |
| `[explorer]` | `port` (8234), `cors_origins`, `allowed_hosts`, `static_dir` | host is always `127.0.0.1`; foreign `Host` headers get HTTP 421 unless listed |
| `[explorer.plugins]` | `impls` list | mounted before core routers; a plugin's own settings go in `[plugins.<package>]` |
| `[task_steps]`, `[artifacts]`, `[envs]` | `impls` list of `"module:Class"` | each class needs its own unique `type`; duplicates, inherited base types and a class that leaves a method its base requires unimplemented are `ConfigError` |
| `[artifacts] type_aliases` | `legacy = "canonical"` string pairs | maps an artifact `type` string found in stored documents to a registered type; core ships no aliases. An alias that maps to itself, to a non-string, or to another alias is `ConfigError` |

`LITELLM_BASE_URL` and `LITELLM_API_KEY` override `[model] base_url` and `api_key`; endpoint requirements and the full per-run precedence are in [Model configuration](#model-configuration-owns-precedence).

### Environment variables

| Variable | Effect |
|---|---|
| `AGENT_ENV_CONFIG` | path of the config file; skips discovery; must exist or every command fails with `ConfigError`; its directory becomes the state root |
| `AGENT_ENV_DOCUMENT_STORE`, `AGENT_ENV_OBJECT_STORE`, `AGENT_ENV_IMAGE_STORE`, `AGENT_ENV_SECRET_STORE`, `AGENT_ENV_RUNNER` | select a backend by alias (`local`); win over the config table |
| `LITELLM_BASE_URL`, `LITELLM_API_KEY` | OpenAI-compatible model endpoint and key; win over `[model]` |
| `AGENT_ENV_HUMAN_A2A_URL` | human-in-the-loop A2A base URL; wins over `[conversations]` |
| `AGENT_ENV_LOCAL_SANDBOX_DIR` | work directory for local compose stacks (default `~/.agent-env-sandboxes`); must be a path Docker Desktop shares |
| `AGENT_ENV_SNAPSHOT_AFTER_LOAD` | default for `--snapshot-after-load` on universe loads; `1`, `true` or `yes` turns it on |
| `AGENT_ENV_FIXTURE_PREFIX` | prefix prepended, in a shared bucket, to every object key core builds: artifact objects, image builds, env and agent snapshots, changelogs, default agent and judge trajectories, verifier outputs and the A2A validator's fixtures |
| `GITHUB_TOKEN` | auth for `--dockerfile-github-url` builds |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `E2B_API_KEY` | provider credentials; `E2B_API_KEY` is read as `env:E2B_API_KEY` from `[sandbox.providers.e2b.config]` |
| `AGENT_ENV_MODAL_REGION` | region for the `modal` / `modal_vm` providers (default `us-east-1`) |

No variable relaxes TLS verification; every client verifies certificates. Variables injected into containers are separate: the environment server reads `MCP_HOST`, `MCP_PORT` and `ENVIRONMENT_NAME`; an A2A agent reads `A2A_HOST`, `A2A_PORT`, and `A2AAgent.deploy` (called by `deploy_agent`) injects `LITELLM_BASE_URL` and `LITELLM_API_KEY`.

## Run on Google Cloud

agent-env runs on Google Cloud with no AWS account: Cloud Storage holds objects, Artifact Registry holds images, Secret Manager holds secrets, and MongoDB, as Atlas on Google Cloud or your own deployment, or Firestore with MongoDB compatibility holds documents. Each is an ordinary store table, described with its options in [Stores and secrets](#stores-and-secrets). This section puts them together: one configuration, the identities it needs, a first-day checklist, what does not work yet, and a run you can reproduce on a laptop.

### Configuration

Install agent-env with the `gcp` extra (see [Optional extras and bundled cloud SDKs](#optional-extras-and-bundled-cloud-sdks)): `uv sync --extra gcp` in a clone, or `uv tool install 'agentenv-framework[gcp]'`. Then point every store at Google Cloud in `.agentenv/config.toml`:

```toml
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { uri = "secret:mongodb_uri", database = "<database>" }

[stores.object]
impl = "agent_env.store.object_store.gcs_object_store:GcsObjectStore"
config = { bucket = "<bucket>", signing_service_account = "<signer>@<project>.iam.gserviceaccount.com" }

[stores.image]
impl = "agent_env.store.image_store:OciRegistryImageStore"

[stores.image.config]
registry_host = "<region>-docker.pkg.dev"
repository_prefix = "<project>/<repository>"

[stores.image.config.credentials]
impl = "agent_env.store.image_store.google_credentials:GoogleAccessTokenCredentials"
service_account = "<registry>@<project>.iam.gserviceaccount.com"

[stores.secret]
impl = "agent_env.store.secret_store.gcp_secret_manager_secret_store:GcpSecretManagerSecretStore"
config = { secret_name = "<secret id>", project = "<project>" }

[model]
base_url = "https://<your-openai-compatible-endpoint>/v1"
api_key = "secret:litellm_api_key"

[sandbox]
default = "local"
agent_default = "local"
```

The Secret Manager secret holds one YAML or JSON mapping, here with `mongodb_uri` and `litellm_api_key` among its keys, and each `secret:` reference reads one key, so no credential is written in the file; `[stores.secret]` itself cannot use `secret:` references, though `env:` works there. With Atlas, `uri` is the cluster's `mongodb+srv://` connection string, and the cluster must accept connections from wherever agent-env runs. To keep Secret Manager out of the process, mount the secret as a file instead (a Cloud Run secret volume, or the Secret Manager add-on on GKE) and read it with `LocalSecretStore(file_path=...)`. That store reads the file once, when it is built, and parses it as typed YAML, so quote a value such as `0123` or `on` that must stay a string; the Secret Manager store reads every value verbatim. `agent-env config show` confirms each store resolved to the class you meant without building any of them.

To keep documents on Google Cloud as well, use a Firestore database with MongoDB compatibility (Enterprise edition) in place of MongoDB: `FirestoreMongoDocumentStore` signs in with the same Application Default Credentials, so the secret needs no `mongodb_uri`. Its table and what it changes are in [Stores and secrets](#stores-and-secrets):

```toml
[stores.document]
impl = "agent_env.store.document_store.firestore_mongo_document_store:FirestoreMongoDocumentStore"
config = { host = "<uid>.<location>.firestore.goog", database = "<database>" }
```

### Identities and IAM

Three identities are involved, and each needs only what its row names:

| Identity | Grant | Why |
|---|---|---|
| The one agent-env runs as, its Application Default Credentials | `roles/storage.objectUser` on the bucket | reads, lists and writes objects; replacing one takes `storage.objects.delete`, which this role has |
| | `roles/iam.serviceAccountTokenCreator` on the signer account | `iam.serviceAccounts.signBlob`, to sign URLs and grants as that account |
| | `roles/iam.serviceAccountTokenCreator` on the registry account | `iam.serviceAccounts.getAccessToken`, to mint registry tokens |
| | `roles/secretmanager.secretAccessor` on the secret alone | `secretmanager.versions.access`, to read the bundle |
| | `roles/datastore.user` and `roles/datastore.indexAdmin` on the project, with Firestore | reads and writes documents, and creates the indexes every process asks for when it starts |
| The signer account (`signing_service_account`) | `roles/storage.objectUser` on the bucket | a signed request acts with the signer's access: a read needs `storage.objects.get`, an upload `create`, and a replacement `delete` |
| The registry account (`service_account` under the image store's credentials) | `roles/artifactregistry.writer` on the one repository, nothing else | its token is used for `docker login` inside sandboxes and handed to Modal as a registry secret |

MongoDB takes no Google IAM: its credentials are in the connection string, which is why the example keeps that in the secret. Firestore takes the `roles/datastore.*` row instead.

- **Signing needs a service account.** A user login (`gcloud auth application-default login`), the attached service account of a VM, GKE or Cloud Run workload, and Workload Identity Federation cannot sign on their own, so without `signing_service_account` the store signs nothing and every feature that needs a signed URL or grant is unavailable (the list is in [Stores and secrets](#stores-and-secrets)). A workload may also sign as its own attached service account by setting `signing_service_account` to that account, which then needs `iam.serviceAccounts.signBlob` on itself (`roles/iam.serviceAccountTokenCreator` with the account as its own principal). IAM signatures are guaranteed for twelve hours; a service-account key signs locally for up to seven days, but it is a long-lived secret.
- **Keep the signer account to the bucket.** `signBlob` on an account is enough to obtain that account's access tokens, so whoever can sign as the signer can act as it. Grant it nothing beyond the bucket.
- **Keep the registry account dedicated.** Its token reaches every sandbox that pulls an image and Modal's registry secret, for up to an hour, and it can push or move any tag in the repository. An account with more than `artifactregistry.writer` on one repository would hand all of that to every workload.
- **Set a quota project on a user login.** User credentials bill their API calls to a quota project, and the APIs must be enabled there. When the project `gcloud` recorded at login is not yours, signing fails at the first store use with `ConfigError: Cannot sign as <signer> (requests are billed to quota project '<other-project>'): ... IAM Service Account Credentials API has not been used in project <other-project> before or it is disabled`. Fix it with `gcloud auth application-default set-quota-project <project>`, or export `GOOGLE_CLOUD_QUOTA_PROJECT=<project>` for one shell.

### Topology

Put the bucket, the Artifact Registry repository and the machines that run agent-env in one region. With the `local` sandbox provider on a Compute Engine VM in that region, environment loads download their image tarballs from the bucket and agent containers pull from the registry without leaving it. Modal and E2B sandboxes run outside your project, on a cloud and region agent-env does not choose beyond Modal's `AGENT_ENV_MODAL_REGION`. A VM-mode deploy (`modal_vm`, `e2b`) downloads its image tarballs from the bucket through signed URLs, while the container-mode `modal` provider pulls every image, and every agent image, from the registry with the registry token, so expect that traffic to be billed as egress. The providers themselves are in [Choose compute, network policy and state](#choose-compute-network-policy-and-state).

### Day-one checklist

1. Enable the Cloud Storage (`storage.googleapis.com`), IAM Service Account Credentials (`iamcredentials.googleapis.com`), Artifact Registry (`artifactregistry.googleapis.com`) and Secret Manager (`secretmanager.googleapis.com`) APIs, and Firestore (`firestore.googleapis.com`) if documents go there, in your project and in the quota project of any user login.
2. Create the bucket with uniform bucket-level access, since agent-env sets no object ACLs, and public access prevention enforced.
3. Add lifecycle rules, because the object store never deletes anything. `a2a_validator/` holds validator probes and is disposable. Run outputs can expire as far as your run retention allows: `prompt_agent_trajectories/`, `judge_trajectories/`, `compacted-trajectories/`, `env_trajectory/`, `env_trigger_state/` and `verifier-outputs/`. Run records point at them, and each `verifier-outputs/` object is also registered as a per-run `FileArtifact`, which is left pointing at nothing once the object expires. `agent_changelog/` is a run output too, except for a capture a task replays through a `deploy_agent` step's `agent_changelog_object_url`; keep those. Keep `artifacts/`, `agent_snapshots/`, `env-snapshots/` and `github-builds/`: versioned artifacts point at them. With `AGENT_ENV_FIXTURE_PREFIX` set, every one of these keys starts with `<prefix>/`, so write the rules for the prefixed paths.
4. Create the Docker repository in Artifact Registry; the image store does not create repositories. Every artifact version pushes its own `v<N>` tag, so the repository grows with every `put`. A cleanup policy that deletes old tags leaves those versions deployable only where the image is loaded from its tarball in the bucket (VM-mode environment deploys, `local` included, and agents on `modal_vm` and `e2b`), not where it is pulled (agents on `local` and `modal`, and environments on the container-mode `modal` provider).
5. Create the signer and registry service accounts and make the grants in [Identities and IAM](#identities-and-iam).
6. Create the secret and add its first version holding the mapping.
7. Run `agent-env config show`, then any command that uses the object store. When `GcsObjectStore` is built it lists the bucket as your identity and, with `signing_service_account`, signs once as that account, so a missing bucket, a missing grant for your identity on the bucket or on the signer, or the quota-project trap fails there with a `ConfigError` rather than mid-run. The signer account's own grant on the bucket is first used by a signed URL, so a missing one fails later, with `403 AccessDenied` from Cloud Storage.

### What works, and what does not yet

- **Agents that take the object-transfer forms** (see [Object transfer](packages/agentenv-protocol/README.md#object-transfer)) get HTTPS grants for everything they move: skill bundles, trajectories, snapshot saves and loads, and changelog capture and replay. None needs Google credentials. A grant lasts at most as long as the signer can sign, twelve hours through IAM, so a changelog capture configured to last longer fails its step for an agent that takes only the object form, and sends an agent that also takes `s3_prefix` that form, which it cannot use here.
- **Agents that take only the older S3-named forms** are handed `gs://` URLs as `skill_s3_url` or `s3_prefix`, which they cannot use without their own Google credentials. A legacy snapshot save also carries a `presigned_post` that Cloud Storage accepts; an agent that uploads through it instead of `s3_prefix` works. A legacy agent that returns its trajectory inline works, since agent-env uploads it; one that answers with a `trajectory_s3_prefix` it wrote itself does not.
- **No credentials are shared.** agent-env hands agents and environment services no Google credentials: `GcsObjectStore` shares none, and `urn:agentenv:add-s3-credentials/v1` applies to S3 only. What they move goes through grants and signed URLs.
- **Without a signer**, the store still reads and writes, but a remote sandbox receives each object over its sandbox connection instead of fetching it, trajectories come back inline, and `env snapshot`, GitHub image builds, and the skill bundles, snapshots and changelogs agents move themselves are unavailable; see [Stores and secrets](#stores-and-secrets).

### A local run on Google Cloud

This reproduces the [Quickstart](#quickstart-a-local-environment-you-can-call-then-a-graded-run) with objects and images on Google Cloud and no AWS account or credentials; documents and secrets stay local, so the project needs only the bucket, the repository, the two accounts and their grants. Log in and set the quota project:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <project>
```

Write this as `.agentenv/config.toml` in an empty project directory, in place of the Quickstart's all-local example:

```toml
[stores]
document = "local"
secret = "local"

[stores.object]
impl = "agent_env.store.object_store.gcs_object_store:GcsObjectStore"
config = { bucket = "<bucket>", signing_service_account = "<signer>@<project>.iam.gserviceaccount.com" }

[stores.image]
impl = "agent_env.store.image_store:OciRegistryImageStore"

[stores.image.config]
registry_host = "<region>-docker.pkg.dev"
repository_prefix = "<project>/<repository>"

[stores.image.config.credentials]
impl = "agent_env.store.image_store.google_credentials:GoogleAccessTokenCredentials"
service_account = "<registry>@<project>.iam.gserviceaccount.com"

[sandbox]
default = "local"
agent_default = "local"
```

`agent-env config show` should report `object: GcsObjectStore` and `image: OciRegistryImageStore`, each `from [stores.object]` and `from [stores.image]`. Unset every `AWS_*` variable to confirm nothing needs AWS credentials; the bootstrap images' base images and the local state Postgres still come, anonymously, from Amazon ECR Public (`public.ecr.aws`). Then run the Quickstart from [Point at a model](#point-at-a-model) through [Tear down](#tear-down), with four differences:

- Skip the `cp` in [Start the local stack](#start-the-local-stack), which would replace this config with the all-local example.
- Every `put` pushes its image to `<region>-docker.pkg.dev/<project>/<repository>/` and uploads the image's tarball to `gs://<bucket>/artifacts/docker_image/`. The push runs `docker login` with a registry token, which Docker keeps in its credential store after the token expires within the hour; `docker logout <region>-docker.pkg.dev` removes the entry.
- Trajectories go to `gs://<bucket>/prompt_agent_trajectories/prompt_id=<prompt-id>/`; `task.json` needs no change.
- Environment deploys download each image tarball through a signed URL, and the agent container is pulled from Artifact Registry.

The context JSON's `prompt_responses[0].agent_trajectory_s3_uri` is then a `gs://` URL; the field keeps its older name. Nothing agent-env wrote is deleted by tear-down: the images stay in the repository and the objects in the bucket, which the lifecycle rules and cleanup policy above take care of.

## CLI reference

### Conventions

The root `agent-env --help` listing is shown under [Verify and platform notes](#verify-and-platform-notes).

- `-v/--verbose` is the only root option; installed plugins may add more.
- Verbs: `put` builds and registers a new version; `deploy` starts an instance; `get-instance` reads an instance record; `validate` checks without deploying.
- `put` never overwrites: repeating it on an existing id appends a version.
- References are `id` or `id:version`; `--version` and bare ids default to the latest version.
- `--platform` on every image-building `put` defaults to `linux/amd64`; see [Verify and platform notes](#verify-and-platform-notes) for Apple Silicon.
- `--sandbox`, `--env-sandbox` and `--agent-sandbox` accept a name or a comma-separated fallback chain.
- `--output-dir` on `task run`, `task run-batch`, `eval run` and the `artifact ... get` commands selects where files land.
- `--help`, `config show`, `config explain`, `config sources`, `config debug`, `plugin list`, `plugin show` and `plugin check` never touch `.agentenv/`; every other command may create it.

### Command groups

| Group | Purpose | Notable subcommands |
|---|---|---|
| `a2a-agent` | A2A agents | `put`, `get`, `validate`, `deploy`, `get-instance`, `add-skill` |
| `artifact` | files, seeds, universes, skills, CLIs | `environment put`, `environment-universe put/get/compatible-envs`, `file-artifact-universe put/put-bundled/get/get-many/list`, `skill put/get`, `cli put/get` |
| `config` | inspect the resolved configuration, read-only | `show [--json]`, `explain PATH [--json]`, `sources [--json]`, `debug [--json]` |
| `env` | build, register, deploy and inspect environments | `mcp-server put/validate/create-cli/load-environment-artifact`, `multi put/validate/load-environment-artifact/load-environment-universe-artifact/compatible-universes/validate-universe-compatibility`, `website put/load-environment-artifact`, `website-browser put`, `service-db put`, `gateway put`, `deploy`, `get-instance`, `snapshot`, `init-env-state`, `teardown-env-state` |
| `eval` | groups of tasks | `create`, `add-tasks`, `run` |
| `plugin` | installed plugins: what each contributes and whether it took effect; add and remove them | `list [--json] [--no-load]`, `show PACKAGE [--json] [--no-load]`, `check [--json]`, `add SPEC...`, `remove PACKAGE...` |
| `run` | write and run the tasks and evals of a bundle folder or an installed bundle, such as the `hello` agent-env ships; with no argument, list the installed bundles and their folders | `--task`, `--eval`, `--model`, `--sandbox`, `--keep`; see [Run a bundle folder](#run-a-bundle-folder) |
| `task` | define and run tasks | `create`, `get`, `validate`, `run`, `run-batch`, `get-instance` |
| `up` | local stack: resolves backends, bootstraps `default-db` and `default`, serves the explorer API | `--no-bootstrap` |

Plugin groups appear in `--help` next to the built-ins; `agent-env plugin list` shows which package each comes from.

### Deprecated aliases and legacy paths

No renamed-command aliases remain. `artifact service`, `artifact service-universe`, `load-service-artifact`, `load-service-universe-artifact` and the `--service-artifact*` options answer `No such command` or `No such option`; the names are `artifact environment`, `artifact environment-universe`, `load-environment-artifact`, `load-environment-universe-artifact` and `--environment-artifact-id`. `--service-version` is gone from every command, along with the field it set. There is no `agent` command group either (the former `agent put-image` and `agent deploy` are gone): `a2a-agent put` is the only way to register an agent image, and `deploy_agent` resolves it by id through `[agents] default_a2a_agent_id`. Stored documents that still carry a legacy artifact `type` string load only if `[artifacts] type_aliases` maps it (see [Configuration reference](#configuration-reference)).

### Known gaps

| Missing | Workaround |
|---|---|
| `init` or config scaffolding | copy `.agentenv/config.example.toml` to `.agentenv/config.toml`, then `agent-env config show` to confirm it is the file in effect |
| `list` for environments, tasks, evals, agents, instances | `Env.query().execute()` in Python, or `GET /api/v1/{envs,tasks,agents,artifacts,evals}` on the explorer; only `artifact file-artifact-universe list` exists |
| `env get` | `Env.get(id[, version])` in Python |
| teardown of a deployed environment instance | `await (await Env.from_instance_id("<instance-id>")).close()`; `env teardown-env-state` retires state stores only (see [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down)) |
| `eval get`, `eval list`, sandbox or agent-id overrides on `eval run` | read the per-run JSON in `--output-dir`; re-run with `--agent-model` / `--agent-artifact-id` |
| `--resume` | `task run --start-step N --context-json <saved context>` |
| trajectory download | read `prompt_responses[i].agent_trajectory_s3_uri` through the object store in Python |
| `up` flags for port or build platform | `[explorer] port`; bootstrap images are always `linux/amd64` |

Report gaps as issues against this repository.

## Extend agent-env

Every backend and primitive plugs in through the same pattern, so a company or lab can ship one package that supplies its own stores, compute, steps and CLI without patching core.

### The seam pattern

A config table names an implementation with `impl = "module.path:ClassName"`. agent-env imports the class, checks that it subclasses the seam's base class, resolves `env:` and `secret:` references in the `config` table, and calls `Class.from_config(**config)` (default: `Class(**config)`). Primitives register under their own `type`. Misconfiguration raises `agent_env.config.ConfigError` naming the impl and the reason: cannot import, not a subclass, malformed pointer, missing `impl`, unresolved reference, duplicate `type`, inherited base `type`, or, for a store or the runner, a `config` key the class does not take or a required one it lacks. The error appears when the seam is first used, so a broken `[task_steps]` entry breaks every task load.

Conformance suites exist for the four store seams only. Sandbox, state, runner, step, artifact, environment and explorer-plugin seams have none yet; the unit tests under `tst/unit/providers/` and `tst/unit/cli/plugins_test.py` are the closest contract specification.

### Custom stores

Subclass `DocumentStore`, `ObjectStore`, `SecretStore` or `ImageStore` from `agent_env.store.*` and implement the abstract methods (nine for `DocumentStore`, `get(name)` for `SecretStore`). Wire it in:

```toml
[stores.document]
impl = "mycorp_demo.stores:JsonFileDocumentStore"
[stores.document.config]
path = "env:MYCORP_DOCS_PATH?.agentenv/mycorp_docs.json"
```

A custom store may resolve relative paths in its `config` table as it likes; only the built-in `local` aliases anchor to the config file's directory. Run the conformance suites from a source checkout, as described in [Conformance suites](#conformance-suites).

An `ObjectStore`'s `get`, `open` and `download_to_file` must raise `ObjectNotFoundError` (from `agent_env.store`; both a `NotFoundError` and a `FileNotFoundError`) for a missing object, and `get_object_key` must raise `ValueError` for a URL that is not one of its own. Core never checks for a particular backend's URL scheme. It hands object URLs to the store, which reads whatever its backend can reach, and asks `owns(url)` where only the store's own objects may be served; `owns` is by default true exactly when `get_object_key` accepts the URL.

An `ImageStore`'s `owns(ref)` says whether a ref is on the store's registry, the refs `auth(ref)` logs in for. The default owns nothing; `OciRegistryImageStore`, and so the ECR and local stores, owns every ref on its host. The Modal container path runs a configured service-db, pgweb or db-mcp image only when the store serving it owns it, and otherwise stock Postgres, or no sidecar.

Core calls a store from worker threads, several at once: every sign, registry login and grant runs off the event loop, since for a remote backend each is a network round trip. A store must be safe to call concurrently.

An `ObjectStore` has three optional methods that sign the HTTPS grants through which A2A agents move skill bundles, trajectories, snapshots and changelog increments without storage credentials (see [Object transfer](packages/agentenv-protocol/README.md#object-transfer)). To offer them, set `supports_transfer_grants = True`, on the class or per instance, and implement all three; agent-env requests grants only from a store that sets the flag. The defaults raise `NotImplementedError`, so one the store cannot offer raises `GrantUnavailableError` instead. `S3ObjectStore` sets the flag unless its endpoint is plain HTTP, as a local S3 emulator's usually is, because grants are HTTPS URLs. A store whose provider caps what one upload can create sets `max_single_upload_bytes`, and no write grant promises more.

| Method | Returns |
|---|---|
| `issue_read_grant(object_url, *, expires_in=3600)` | `HttpGetGrant` for one existing object |
| `issue_write_grant(object_url, *, media_type, max_bytes, expires_in=3600)` | `HttpPutGrant` for one object, signed for `media_type` |
| `issue_upload_policy(prefix_url, *, max_object_bytes, expires_in)` | `UploadPolicy`: a multipart POST (`HttpPostPolicyGrant`) for uploads below `prefix_url`, and its `expires_at` |

A grant's `expires_at` is when it stops working: at most `expires_in` from now, and earlier if the store's signing credentials expire first. agent-env wraps an upload policy in the changelog's object-count and total-size limits, which the agent's uploader enforces, and hands it over once for a whole capture, so the policy must last all of `expires_in`: raise `GrantUnavailableError` (from `agent_env.store`) when it could be cut short, and for any `expires_in` longer than the store can sign. Decide that from what signs the grant, not from how long the current credentials happen to have left, so that one configuration always gets the same answer. `deploy_agent` then falls back to the older `s3_prefix` form for an agent that also takes it, and fails the step for an agent that takes only `write_namespace`. Have the provider enforce the prefix and the per-object size. `S3ObjectStore` implements all three with presigned GET and PUT URLs and POST policies, which SigV4 caps at seven days, and states S3's 5 GiB single-PUT cap. It issues upload policies only when it signs with long-term credentials: temporary ones, such as an assumed role's, SSO's or an instance profile's, are replaced during a capture and end the grants they signed. `GcsObjectStore` sets the flag when it has a signer and an HTTPS endpoint, and implements all three with V4 signed URLs and POST policies, for as long as its signer can sign (above); a write grant signs `x-goog-content-length-range` and an upload policy a `content-length-range`, so Cloud Storage itself refuses an upload over the limit, and a policy's signed `starts-with` condition refuses one whose key falls outside its prefix. Its signature does not depend on how long the caller's own token lasts, so every signer counts as long-term within its cap. A capture longer than that cap sends an agent that also takes `s3_prefix` that form instead, with a `gs://` prefix it cannot write to.

### Custom sandbox, state, environment and runner providers

Sandbox: subclass `agent_env.providers.sandbox_providers.sandbox_provider.SandboxProvider`, implement `create_sandbox`, and register it under `[sandbox.providers.<name>]` or as an `agent_env.sandbox_providers` entry point named `<name>`. The `<name>` must equal the `.type` of the `Sandbox` objects it produces. That check runs on the first `create_*` call, terminates the mis-typed sandbox, and raises `SandboxProviderTypeError`, a `ConfigError`. This name-equals-type rule is what lets agent-env reconnect to an instance from its stored record.

State: subclass `agent_env.providers.env_state.env_state_provider.EnvStateProvider`, set the `type` ClassVar, implement `acquire`, `_teardown` and `deploy_state_context(ttl_seconds=, name_hint=)`. Register under `[state.providers.<name>]` or as an `agent_env.state_providers` entry point; here the name-equals-type check runs at registration. Once a non-local provider exists, `env init-env-state --env-state-type <name>` can pre-create a store out of band; `local_postgres` refuses that.

Environment: subclass `agent_env.providers.env_providers.env_provider.EnvironmentProvider`, set `type`, and implement `deploy(env, sandbox_provider, **options)` and `close()`. `deploy` returns the env's record unregistered, a `DeployedEnv`, `DeployedSandboxEnv` or `DeployedGatewayEnv` whose `env_provider_type` is `type`, and `close` tears down everything the deploy created: agent-env's reapers find only its own sandboxes, by the ids a record carries. Register it as an `agent_env.env_providers` entry point named `type`; the name-equals-type check runs at registration, and `gateway` (a gateway and service database in front of the server) and `server` (the server on its own) are built in.

A stock `MCPServerEnv`, `WebsiteEnv` or `MultiEnv` deploys through the provider its `env_provider_type` names, so it runs on yours. Set the type with `--env-provider-type <type>` on `agent-env env mcp-server put`, `env website put` or `env multi put`, which check that the type is installed, or with `env_provider_type="<type>"` on the env's `put`; a task's `deploy_env` then builds your provider when it deploys the env. Reading a stored env needs no provider installed, but saving a task that deploys it does: `task create` checks the step's options against your `deploy`, and reports a type it can't find.

```python
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers import EnvironmentProvider

class PodProvider(EnvironmentProvider):
    type = "pod"

    async def deploy(self, env, sandbox_provider, **options) -> DeployedEnv:
        if isinstance(env, MultiEnv) and not env.website_envs:
            servers = env.mcp_server_envs
        elif isinstance(env, MCPServerEnv):
            servers = [env]
        else:
            raise TypeError(f"{self.type} doesn't deploy a {env.type} env")  # before creating anything
        # One server serves its own card; several run behind a gateway whose card lists each of them.
        url = await start_pods({s.environment_name: s.docker_image_artifact.image_name for s in servers},
                               ttl_seconds=options["ttl_seconds"])
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type,
                           environment_card_url=f"{url}/.well-known/agent-env.json", environment_card=await read_card(url))

    async def close(self) -> None:
        await stop_pods()
```

- **What it deploys.** `deploy` receives an `MCPServerEnv`, a `WebsiteEnv` or a `MultiEnv`, and reads what to run from the attributes [the plugin surface](#plugin-compatibility) lists. A provider handed an env it doesn't host raises `TypeError` before it creates anything. The built-in `server` deploys only an `MCPServerEnv`, so a `WebsiteEnv` or `MultiEnv` of that type is refused at `put`, when a task is saved, and at deploy.
- **A `MultiEnv`** is deployed by its own `env_provider_type`, which hosts all of its MCP servers and websites; the children's own types are ignored. A `MultiEnv` whose MCP server and website share a name is refused for a plugin's type, at `put` as at deploy, since one card can't tell them apart.

- **Options.** Take `**options`: it receives every option the env's deploy was given, `ttl_seconds`, `disk_size_gb`, `gateway_mode`, `cpu`, `memory_mb`, `priority`, `env_state_type`, `env_state_instance_id` and `attribution`. A `deploy` that names its options instead gets those, and the deploy is refused any other option set away from its default: `deploy_env` always sets `ttl_seconds` and `attribution`, and `agent-env env deploy` and a run's overrides set `priority`.
- **The record** must say `env_provider_type` is your `type` and carry the server's MCP URL, in its env card (`environment_card` and `environment_card_url`, which come together and which the URL derives from) or in `mcp_url`. A card must be named the env's `environment_name`, or list a child env of that name; a `MultiEnv`'s lists each of its MCP servers and websites as a child env by `environment_name`, and needn't list the website browser our gateway adds. A child env's `url` is a path under the record's address, such as `/svc/<name>/agentenv`. Agents see a `MultiEnv`'s MCP server under the record's `mcp_server_name`, which defaults to its card's `name`; when the env has a `name`, it must be that. A record that breaks one of these fails the deploy, which closes the provider.
- **Loads** (`load_artifact`, `env mcp-server load-environment-artifact`) hand the server the data file as a signed URL in the data plane's `add_data`, at the address its record's env card gives; each of a `MultiEnv`'s children, websites included, loads this way. The server must fetch an `http(s)` `FileWithUri` part, and the object store must sign URLs: S3 does, while the local filesystem store can't. A load the store can't sign, of an object that is gone, or into a card whose operations leave out `data/reset` or `data/add` is refused before it touches the env's data. A universe loads service by service, with no resume and no snapshot restore or bake, as on an external state store. Staging files onto the env's host (`load_file_artifact_universe`, or a universe's metadata files) is refused before anything is loaded, and so is snapshot capture before it reads anything.
- **Lifetime.** `close()` runs when the deploy fails, or when a caller holding the env object that deployed closes it; a task doesn't. `teardown_sandboxes`, and the cleanup after an env's `validate()`, terminate the sandboxes a `DeployedSandboxEnv` record names; a deployment outside agent-env's sandboxes must end by itself, for example at `ttl_seconds`. An env reattached from a plugin's record, as `Env.from_instance_id` gives, holds no provider, so its `close()` does nothing.
- **A provider that subclasses a built-in** (`EnvironmentGatewayProvider`, `EnvironmentServerProvider`) is handled as one: loads stage into its sandboxes, and its records reattach to them. Its `deploy` must return the built-in's kind of record, a `DeployedSandboxEnv`, or the deploy fails.

A custom env can use a provider itself: it builds it with `build_env_provider(type)` and registers the record with `agent_env.env.store.register_env_instance`, and it may list the types it accepts in its `env_provider_types`. `register_env_instance` refuses a record that would load back as another class and drop the fields its own class adds, so a deploy keeps what it learns on the env card and the containers it created in `sandbox_ids`.

Runner: subclass `agent_env.runner.runner.Runner` (`submit`, `status`, `cancel` abstract) and set `[runner] impl` plus `[runner.config]`; the contract and what the runner serves are in [At scale: bring your own durable runner](#at-scale-bring-your-own-durable-runner). Only `LocalRunner` ships. None of these four seams has a conformance suite.

### Custom envs, task steps, artifacts

| Primitive | Base class | Contract | Registration |
|---|---|---|---|
| task step | `agent_env.task_step.task_step.TaskStep` | `type` ClassVar, `to_dict`/`from_dict`, `async execute(context)`, optional `preflight()` | `[task_steps] impls`, or an `agent_env.task_steps` entry point |
| artifact | `agent_env.artifact.artifact.Artifact` (pydantic) | `type` field default is the registry key | `[artifacts] impls`, or an `agent_env.artifacts` entry point |
| environment | `agent_env.env.env.Env` (subclass it directly; there are no generic parameters) | `type` ClassVar, `from_dict`, `async deploy(**kwargs) -> DeployedEnv` and the async classmethod `from_deployed_env(deployed)` (all three raise `NotImplementedError` on the base); `to_dict` is inherited | `[envs] impls`, or an `agent_env.envs` entry point |

```toml
[task_steps]
impls = ["mycorp_demo.steps:GradeEssayTaskStep"]
```

A custom step's `preflight()` participates in `task create` and `task validate`. Verifiers are ordinary steps that write `context.metadata["verifications"][verifier_id] = {"score", "results"}`. A task made only of custom steps that need no sandbox runs in-process with no Docker, model or remote backend. Custom artifacts and environments round-trip through `put`, `get` and `query().type(...)`, and `Artifact.get` / `Env.get` return the registered subclass. There are no CLI subcommands for custom artifact or environment types; those round trips are Python-only. A custom environment's full contract is `type`, `from_dict`, `deploy` and `from_deployed_env`, which core calls to drive a running deployment, as `load_artifact` and `Env.from_instance_id` do:

```python
from agent_env.env.env import DeployedEnv, Env

class MyCustomEnv(Env):
    type = "my_custom_env"                       # the portable identity stored in the document store

    @classmethod
    def from_dict(cls, data: dict) -> "MyCustomEnv":
        ...                                      # rebuild from the stored document

    async def deploy(self, **kwargs) -> DeployedEnv:
        ...                                      # provision and return its record, e.g. a DeployedSandboxEnv

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> "MyCustomEnv":
        ...                                      # the env that record serves, connected to it
```

An environment can also be referenced without ever being deployed. When a `deploy_agent` step lists its id in `env_ids` and no `deploy_env` step deployed it, the step reads an http(s) `mcp_url` attribute from the stored environment document and registers that live endpoint with the agent, so such a class may leave `deploy()` unimplemented. The step sends the endpoint no bearer token, so it must accept requests without one. This is how an MCP server that already runs elsewhere joins a task. Any other undeployed id fails with `Env '<id>' is not in context.deployed_envs; add a DeployEnvTaskStep for it, or reference an env that exposes a live http(s) 'mcp_url'`.

### Register types from an installed package

An installed distribution registers envs, task steps, artifacts, sandbox, state and environment providers and explorer plugins by declaring entry points, with no config. The group says what kind of thing it is, the entry-point name is the registry key, and the value is the class:

```toml
[project.entry-points."agent_env.envs"]
browser = "agentenv_browser.env:BrowserEnv"

[project.entry-points."agent_env.task_steps"]
browser_navigate = "agentenv_browser.steps:NavigateTaskStep"
```

| Group | Value | Name |
|---|---|---|
| `agent_env.envs` | `Env` subclass with its own `type`, implementing `from_dict` | must equal the class's `type`, the spelling its documents are written under |
| `agent_env.task_steps` | `TaskStep` subclass with its own `type`, implementing `execute` and `from_dict` | must equal the class's `type` |
| `agent_env.artifacts` | `Artifact` subclass with its own `type` field default | the registry key; a class may also register under extra names, such as a legacy spelling, as long as its own `type` default resolves to it and its `type` field accepts each extra name |
| `agent_env.sandbox_providers` | `SandboxProvider` subclass implementing `create_sandbox` | the registry key; it must equal the `.type` of the sandboxes the provider produces, checked on the first `create_*` call |
| `agent_env.state_providers` | `EnvStateProvider` subclass implementing `acquire` and `_teardown` | must equal the class's `type`, checked at registration |
| `agent_env.env_providers` | `EnvironmentProvider` subclass implementing `deploy` and `close` | must equal the class's `type`, which its records carry as `env_provider_type`; checked at registration |
| `agent_env.explorer_plugins` | `ExplorerPlugin` subclass with its own `type`, implementing `router` | must equal the class's `type` |

- The class must implement every abstract method of its base, and, for envs and task steps, `from_dict` as a classmethod, which the base defines only to raise. A plugin that does not is `failed` with `invalid-plugin`, naming what is missing, for example `GradeTaskStep must implement execute and from_dict`. An `impls` entry or provider table naming such a class is a `ConfigError`.
- Each registry takes the built-ins first, then plugins, then config. A plugin cannot replace a built-in: it is skipped with a warning, however many distributions claim that name.
- A name that two installed distributions register in one group, or that one declares twice, is left out of the registry with a warning, and the rest of the group, built-ins included, loads. An explorer plugin under that name is not mounted, so its routes are absent. Resolving the name fails like a plugin that did not load, naming each claimant and its version, for example `Unknown env type: browser (2 installed plugins register 'browser' in agent_env.envs: …)`, and says to remove all but one with `agent-env plugin remove PACKAGE`, or, when one package declares the name twice, to report it to the package's author. Neither claimant is picked, because the order entry points are found in is not fixed.
- Config cannot settle a conflicted name. An `impls` entry, a provider table, an `[explorer.plugins]` impl or an `[artifacts] type_aliases` entry mapping to that name is checked as usual, then skipped with a warning. A `type_aliases` entry whose old spelling is that name is a `ConfigError`, as it is when one plugin registers it.
- A plugin that fails to import, fails the checks above, or (for explorer plugins) fails to construct is skipped with a warning and the others still load. So is one whose requirement on agent-env excludes the installed version, before it is imported (see [Plugin compatibility](#plugin-compatibility)). Resolving its name then reports the recorded error, for example `Unknown env type: browser (registered by 'browser' from agentenv-web 2.1.0 (…) but failed to load: ModuleNotFoundError(…))`.
- Config replaces a plugin's class with a warning, except under a conflicted name: an `impls` entry of the same `type`, or a `[sandbox.providers.<name>]` / `[state.providers.<name>]` table carrying `impl`. Naming the plugin's own class is silent. A provider table without `impl` configures the plugin's provider the way it configures a built-in's, which is where per-deployment settings such as a secret name belong. A config-only provider table, or an `[artifacts] type_aliases` entry, that points at a plugin which failed to load is skipped with a warning rather than failing the whole registry.
- `agent_env.plugins.load_failures()` lists every plugin that did not take effect in the registries the current `Config` has built (failed to load, validate or construct, clashed with a built-in, or claims a name another entry point claims), by group and name, with the reason. The record belongs to the `Config`, so `reset_config()` starts a new one. A deployment that ships its plugins can build its registries at startup and assert it is empty.
- `agent_env.plugins.inventory()` lists every installed distribution that declares entry points in these groups, with a status for each contribution and a `code` saying why (see [Plugin report format](#plugin-report-format)): `active`; `replaced` (config names a different class, and `replaced_by` says where); `failed`; `skipped` (a built-in owns the name); `conflict`; `blocked` (the group's real build fails, for example because its config table is invalid, so nothing in the group loads); or `unloaded`. It builds the registries on a throwaway `Config` that reads the same document, so the `Config` in use keeps its registries and its `load_failures()`. `inventory(load=False)` reads installed metadata only and runs no plugin code, so a status that needs a build is `unloaded`. With the default `load=True` the plugins are imported and the explorer plugins constructed, with whatever process-wide effects that has. `discovery_errors` names each group whose installed entry points could not be read at all, so none of its plugins is listed.
- Discovery reads installed metadata and imports nothing. When entry points are loaded is unspecified: today each group is imported when its registry is first built, but a plugin must not depend on that.
- Loading a plugin never changes which config a `Config` reads: the `Config` resolves its document before it imports any plugin. A plugin package that sets `AGENT_ENV_CONFIG` on import affects only configs built afterwards, such as after `reset_config()`.

The distribution that declares the entry points is the plugin, so `pip uninstall` removes it completely, apart from any `[plugins.<package>]` table in the config file. CLI commands are a separate contribution, described next.

### CLI plugins, root options, explorer routes

Installed packages add commands through two entry-point groups:

```toml
[project.entry-points."agent_env.cli_plugins"]
my-tools = "mycorp_demo.cli:my_tools"
[project.entry-points."agent_env.cli_root_options"]
tenant = "mycorp_demo.cli:tenant_option"
```

`agent_env.cli_plugins` entries are click commands or groups added next to the built-ins, under the entry-point name whatever the command object is called: `my-tools` above is `agent-env my-tools`. `agent_env.cli_root_options` entries are optional `click.Option` instances with `expose_value=False`; their callback runs before the subcommand, so it can set `AGENT_ENV_CONFIG` and call `agent_env.config.reset_config()` to select the config for the whole process. Plugins load when `agent_env.cli` is imported, after the built-ins. Clash rules: a plugin command whose entry-point name core already uses, or a flag core owns, is skipped with a warning on stderr, and core wins. Two different root options on the same flag are both left off with a warning, and `agent-env plugin list` reports each as a `conflict`; the CLI still starts, so `agent-env plugin remove` can settle it. The same `click.Option` object exported by two entry points, from one package or two, is attached once. Nothing grafts onto existing groups.

Explorer routes: subclass `agent_env.explorer.plugin.ExplorerPlugin`, set the `type` ClassVar, and return a FastAPI `APIRouter` from the `router` property. List it under `[explorer.plugins] impls` or declare an `agent_env.explorer_plugins` entry point; `from_config()` takes no arguments, and a plugin reads its own settings with `agent_env.plugins.settings` (see [Plugin settings](#plugin-settings)). `agent-env up --no-bootstrap` mounts it before the core routers and needs no Docker.

### Bundles from installed packages

A package ships bundles by naming the package that holds their folders:

```toml
[project.entry-points."agent_env.bundles"]
triage = "mycorp_demo.bundles"
```

- The bundle `triage` is the folder `mycorp_demo/bundles/triage/`. The entry-point name is both the bundle's name and its folder's name, so it may contain `-`.
- The value names a package, never an object. The folder is found without importing any of the package's code, so listing bundles runs nothing.
- The folder ships in the wheel as package data, and the package must be installed unpacked.
- An installed bundle's ids are rooted at its distribution, wherever the package is installed: `triage` above writes `@local/mycorp-demo/triage/...`.
- `agent-env run triage` runs it, and `agent-env run` with no argument lists the installed bundles, each with its folder.
- When two packages install bundles of one name, run each as `<package>/<name>`, using the canonical distribution name, for example `mycorp-demo/triage`. agent-env's own bundles keep their bare names, so a package's `hello` runs only as `<package>/hello`.
- `agent-env plugin list` shows each bundle as a contribution of its package, agent-env's own `hello` as `agentenv-framework … bundle hello`. `plugin check` fails when a bundle doesn't resolve to a folder, fails a check `agent-env run` makes before it reads a store (it isn't a valid bundle, a type it names doesn't resolve, or a step's fields don't build), is registered twice by one package (`conflict`), or comes from a package whose agent-env requirement isn't met (`incompatible-core`). `list --no-load` and `show --no-load` only parse it.

### Plugin settings

A plugin that needs settings of its own reads them from `[plugins.<package>]`, where `<package>` is its distribution name: the name `pip install` takes and `agent-env plugin list` prints. That table belongs to the plugin. agent-env reads nothing in it and checks none of its keys. Every other top-level table belongs to agent-env, which warns about one it does not read (see [Inspect the resolved configuration](#inspect-the-resolved-configuration)), so a plugin keeps nothing of its own anywhere else.

```toml
[plugins.acme-agentenv-browser]
endpoint = "env:BROWSER_URL?http://localhost:9222"
timeout = 30

[plugins.acme-agentenv-browser.viewport]
width = 1280
```

```python
from agent_env.plugins import settings

def browser_timeout() -> int:
    browser = settings("acme-agentenv-browser")   # {} when the file has no such table
    return browser.get("timeout", 10)             # the plugin keeps its own defaults
```

- Pass the distribution name, not the module's `__name__` or `__package__`: `acme-agentenv-browser` may import as `acme_browser`.
- Call `settings()` where the value is used, in `from_config`, `execute` or a command's body, not at import. Entry points are imported before a root option can select the config file, so a value read at import may come from another file.
- The key is matched by canonical name (PEP 503: lowercase, with each run of `-`, `_` and `.` read as one `-`), so `[plugins.acme_agentenv_browser]` is the same table. Write the canonical form. It never needs quoting, whereas a dotted name written bare (`[plugins.acme.browser]`) nests. Two tables that name one package are a `ConfigError`, not a guess.
- `settings(name, *, config=None)` returns a copy of the table, read from `config`'s document (default: the process `Config`), with `env:` and `secret:` references resolved as in every other table. It raises `agent_env.config.ConfigError` when `[plugins]` or the plugin's entry is not a table, when two keys name the package, or when a reference without a `?default` cannot be resolved; the message names the table. agent-env never reads the table itself, so a broken entry fails only its own plugin's read, and `config show` reports it in place.
- No environment variable overrides a plugin setting. For a value that differs per deployment, write an `env:` reference in the table.
- `agent-env config show` lists each table under its package: with the installed version, with `(declares no agent_env entry point)` when the distribution is installed but is not a plugin, which usually means a misspelled entry-point group, with `(agent-env itself)` for `agentenv-framework`, whose table agent-env does not read, or with `(not installed)`. `config explain plugins.<package>.<key>` says whose table holds the key; agent-env does not read or check it, so it cannot tell whether the plugin reads that key, and a misspelled key is still shown (see [Inspect the resolved configuration](#inspect-the-resolved-configuration)). A table for a plugin that is not installed is reported, never an error, because one config file is often shared by processes that install different plugins.
- Both commands mask a plugin's table the way they mask agent-env's, and a service may log what they report. A literal value is printed as `***` when its key, or a key above it in the table, contains `password`, `passwd`, `passphrase`, `secret`, `token`, `credential`, `api_key`, `private_key`, `auth` or `bearer` in any case, and a connection URI's userinfo is masked. Any other literal is printed, so keep credentials behind `secret:` or `env:` references. A reference prints as written, with any `?default` masked like a literal.
- A provider the plugin registers is still configured in `[sandbox.providers.<name>]` or `[state.providers.<name>]`, under the provider's name, and an explorer plugin class is still listed in `[explorer.plugins]`. agent-env reads those tables itself. `[plugins.<package>]` holds only what the plugin reads.
- Uninstalling a plugin leaves its table in the config file.
- A top-level table one letter from `[plugins]`, such as `[plugin]`, gets a warning in `config show`: nothing reads it, so the plugin would get no settings.

### Manage plugins

<!-- tst/installer/test_container_journey.py runs the commands in the first bash block below. -->

`agent-env plugin` shows what the installed plugins contribute and whether each piece took effect, and adds or removes them.

```bash
agent-env plugin list            # every plugin package, what it provides, and its status
agent-env plugin show PACKAGE    # each contribution and why it is in that state
agent-env plugin check           # exit 1 if any contribution did not take effect
agent-env plugin add SPEC...     # install through this environment's installer, then check
agent-env plugin remove PACKAGE  # uninstall through the same installer, after safety checks
```

```
agent-env 0.9.1193 · uv tool at ~/.local/share/uv/tools/agentenv-framework
config: (none)

PACKAGE             VERSION   PROVIDES                                 STATUS
agentenv-browser    1.0.0     env browser, task step browser_navigate  ok
agentenv-framework  0.9.1193  bundle hello                             ok
agentenv-grader     0.3.1     task step grade_essay                    1 failed
```

| Status | Meaning |
|---|---|
| `active` | registered and in use |
| `replaced` | config names a different class for the name; `show` says where. Not a failure |
| `failed` | failed to import, validate or construct |
| `skipped` | a built-in, a core command or root option, or (for CLI commands) a plugin loaded first owns the name |
| `conflict` | another entry point, from another package or this one, claims the same name, so none of them is used. For a bundle, only a second claim from the same package is a conflict; another package's bundle of that name is `qualified-only` |
| `blocked` | the group cannot load at all, for example because its config table is invalid, so this contribution does not either |
| `unloaded` | not loaded (`--no-load`). When loading, it means the status could not be determined, and `check` fails on it |

A contribution that is not `active` also has a code saying why, shown in brackets after its reason; the codes are listed under [Plugin report format](#plugin-report-format).

- The header says how agent-env is installed (uv tool, pipx, uv project, virtualenv or system Python) and where: a plugin has to be installed into that same environment.
- `list` and `show` take `--json` and `--no-load`. `--no-load` reads installed metadata only and imports no type plugin; CLI plugins are already loaded, because the CLI loads them when it starts. Without it, `list`, `show` and `check` import every type plugin and construct the explorer plugins, so that plugin code runs.
- `show` also reports whether importing the package sets `AGENT_ENV_CONFIG`, checked in a fresh interpreter.
- `check` builds every registry the way a process does, adds the CLI's own plugins, and fails on `failed`, `skipped`, `conflict`, `blocked` or `unloaded`, or when the config file or the installed entry points cannot be read (a malformed `entry_points.txt` hides every plugin, so it fails rather than passing empty). `check --json` prints the `list --json` report plus `ok` and `problems`. A `replaced` contribution passes: config chose it.
- `plugin` is a core command: a CLI plugin that names a command `plugin` is skipped, like any clash with a core command.
- The Python equivalent is `agent_env.plugins.inventory()` (see [Register types from an installed package](#register-types-from-an-installed-package)).

#### Plugin report format

`plugin list --json`, `plugin show --json` and `plugin check --json`, and the `agent_env.plugins` types they are built from, are the interface for scripts, CI and image builds. The human output of these commands is not: parse the JSON.

```json
{
  "format_version": 1,
  "agent_env": {"version": "0.9.1217", "environment": "uv tool", "location": "/home/me/.local/share/uv/tools/agentenv-framework"},
  "config": {"path": "/work/.agentenv/config.toml", "error": null},
  "loaded": true,
  "group_errors": {},
  "discovery_errors": {},
  "plugins": [
    {"name": "agentenv-grader", "version": "0.3.1", "contributions": [
      {"group": "agent_env.task_steps", "name": "grade_essay", "value": "agentenv_grader.steps:GradeEssay",
       "status": "failed", "code": "load-failed", "reason": "failed to load: ModuleNotFoundError(\"No module named 'openai'\")",
       "replaced_by": null, "conflicts_with": []}
    ]}
  ]
}
```

- Every key is always present; an empty value is `null`, `[]` or `{}`. `group_errors` and `discovery_errors` map an entry-point group to `{code, reason}`, and `config.error` is `{code, reason}` or `null`.
- A distribution whose metadata cannot be read is listed under the label `(unreadable metadata: DIR)`, where `DIR` is its `.dist-info` directory. The label is not a package name, so `plugin remove` cannot take it: reinstall the package with its installer, or delete that directory.
- A contribution's `status` says what happened to it (the table above) and its `code` says why. `code` and `reason` are set together: on every contribution that is not `active`, and on an `active` one only for information. `replaced_by` is set exactly when the status is `replaced`: `{file, table, impl}`, where `table` is the config table naming the other class (`envs`, `task_steps`, `artifacts`, `explorer.plugins`, `sandbox.providers.NAME` or `state.providers.NAME`). `conflicts_with` lists the other entry points that claim the name, as `{package, version, value}`.
- `show --json` adds `config_effect`, a sentence saying whether importing the package sets `AGENT_ENV_CONFIG` (`null` with `--no-load`). `check --json` adds `ok` and `problems`, each a contribution with its `package` and `version`.
- `check` exits 0 when `ok` and 1 otherwise; a usage error exits 2. `show` exits 1 with no report when no installed plugin package has that name. A run that does not print exactly one JSON document on stdout, such as a crash while agent-env starts, is not a report: count it as a failure whatever its exit status.

`format_version` changes only for a change a consumer cannot ignore: a key removed, renamed or given another type, a new status, a code redefined or reused, or a new meaning for an exit status. A new key, a new code, reworded `reason` or `config_effect` text, and a change in list order leave it as it is. So a consumer should ignore keys it does not know, handle a code it does not know by its `status`, refuse a `format_version` it does not know, and never parse `reason`. Which situation gets which status and code, and which group error is reported when several apply, is behaviour rather than format: a release that changes one says so in its notes. A code is never removed or reused.

| Code | Where | Meaning | What to do |
|---|---|---|---|
| `load-failed` | `failed` | The plugin's code raised: when it was imported (`SystemExit` included), when an explorer plugin was constructed, or in a root option's callback with the flag absent | Install what it needs, or report it to the plugin's author |
| `invalid-plugin` | `failed` | It imported but does not fit its group: not a subclass of the group's base class; a `type` that is missing, inherited or not the entry-point name; a method its base requires left unimplemented; not a `click.Command` or `click.Option`; a root option that is required or exposes a value; a command click refused; an extra artifact name that reads no document. A bundle is never imported: it is invalid when its value isn't an installed, unpacked package holding that folder, when its metadata can't be read, or when the folder isn't a valid bundle (with plugins loading, one whose step, env or artifact types don't resolve) | Report it to the plugin's author; for a bundle whose package is installed zipped, reinstall it unpacked. For an extra name whose class's own type is in conflict, settle that conflict |
| `incompatible-core` | `failed` | The plugin's requirement on `agentenv-framework` or `agentenv-protocol` excludes the installed version, so it was not imported. See [Plugin compatibility](#plugin-compatibility) | Upgrade agent-env, or install a version of the plugin that fits |
| `builtin-name` | `skipped` | agent-env owns the name: a built-in type, a core command or a core root option | The plugin has to rename it |
| `name-conflict` | `conflict`, `skipped` | More than one entry point claims the name, from two packages or twice from one; for bundles, only twice from one package (see `qualified-only`). In a type group none of them is registered, and the rest of the group loads; two different root options on one flag are each a `conflict`, and neither is attached. A CLI command whose name a plugin loaded earlier took is `skipped`, and the earlier one stays | Remove all but one: `agent-env plugin remove PACKAGE`. A package that declares a name twice has to be fixed by its author |
| `replaced-by-config` | `replaced` | The config registers another class under the name | Nothing, unless you did not mean it |
| `already-attached` | `active` | A second entry point, from the same package or another, exports the same `click.Option` object, which is attached once | Nothing |
| `not-loaded` | `unloaded` | `--no-load`, or `inventory(load=False)`: whether it takes effect needs an import | Run without `--no-load` |
| `status-unknown` | `unloaded` | It was loaded, but nothing was recorded for it. This should not happen | Report it as an agent-env bug |
| `config-not-found` | config error, group error, `blocked` | `AGENT_ENV_CONFIG` names a file that does not exist | Fix or unset `AGENT_ENV_CONFIG` |
| `config-unreadable` | config error, group error, `blocked` | The config file could not be read or parsed | Fix the file |
| `config-invalid` | group error, `blocked` | The file parsed, but what it says for this group is invalid | Fix the table the reason names |
| `group-build-failed` | group error, `blocked` | Building the group raised something else | See the reason |
| `entry-points-unreadable` | discovery error | The group's installed entry points could not be read, so none of its plugins is listed | Reinstall the package the reason names |
| `qualified-only` | `active` | Another package installs a bundle of the same name, so this one runs only as `<package>/<name>`. agent-env's own bundles keep the bare name | Run it by its qualified name |

A `blocked` contribution carries its group error's code, so it says what to fix rather than only that the group failed.

#### Add and remove plugins

agent-env never installs anything itself. `add` and `remove` run the installer that owns the environment agent-env runs in, so that installer's next rebuild keeps the change:

| agent-env installed as | `add` runs | `remove` runs |
|---|---|---|
| uv tool | `uv tool install agentenv-framework --with ... --with SPEC`, passing back the receipt's requirements (editable and git ones included), Python and options | the same, without the package |
| pipx | `pipx inject VENV SPEC`, with the venv's own pip arguments (and `--force` to upgrade a package it already has) | `pipx uninject --leave-deps VENV PACKAGE` |
| uv project | `uv add --no-sync --project ROOT SPEC` and `uv sync --inexact`, then shows the `pyproject.toml` diff | `uv remove --no-sync --project ROOT PACKAGE` and `uv pip uninstall PACKAGE` |
| virtualenv | `python -m pip install SPEC` (or `uv pip install` when the venv has no pip) | `python -m pip uninstall -y PACKAGE` |
| Poetry or PDM project | prints `poetry add` / `pdm add` to run yourself | prints `poetry remove` / `pdm remove` |
| Hatch environment | says to add it to the environment's dependencies | the same, for removal |
| system Python with the PEP 668 marker, or read-only site-packages | refuses, with guidance | refuses |

`SPEC` is anything the installer accepts, so a plugin can come from an index, a git repository or a file:

| From | `SPEC` |
|---|---|
| PyPI, or the index the installer is configured with | `agentenv-grader`, `'agentenv-grader==0.3.0'`, `'agentenv-grader>=0.3,<0.4'` |
| a git tag, commit or branch | `'agentenv-grader @ git+https://github.com/mycorp/agentenv-grader@v0.3.0'` |
| one package in a monorepo | `'agentenv-grader @ git+https://github.com/mycorp/plugins@v1#subdirectory=agentenv-grader'` |
| a private repository | `'agentenv-grader @ git+ssh://git@github.com/mycorp/agentenv-grader.git@v0.3.0'` |
| a release asset | `https://github.com/mycorp/agentenv-grader/releases/download/v0.3.0/agentenv_grader-0.3.0-py3-none-any.whl` |
| a local wheel or checkout | `./dist/agentenv_grader-0.3.0-py3-none-any.whl`, `./agentenv-grader` |

- Write a git URL as `NAME @ URL`, and a local checkout as `NAME @ file:///absolute/path`. agent-env cannot read the name from a bare `git+https://…` or a directory, so it cannot tell pipx or a uv tool which installed plugin the spec replaces: pipx leaves the installed one as it is, and `add` says so; a uv tool reports conflicting URLs, and the add is rolled back. A git install records the commit it resolved to, so a rollback reinstalls that commit.
- `add` never replaces agent-env itself. A spec that names `agentenv-framework`, as a name, a wheel or `NAME @ URL`, is refused before anything runs. A bare URL or directory that turns out to build it is rolled back after the install: agent-env may move to a newer release from the index when a plugin requires one, but not to a URL, git or directory source. Upgrade agent-env with the installer that owns the environment.
- Credentials stay with git and the installer: an SSH key or a git credential helper (`gh auth setup-git`) for a private repository, and the installer's own config for a private index. A token written into `SPEC` is kept in the installer's record of the environment.
- `--installer` overrides the detected installer, `--index-url` passes an index through to it (a uv tool keeps it as its default index), `--dry-run` shows what would run, and `--yes` skips the prompt.
- uv takes a pre-release or a yanked version only when you name it. If the installer reports that a plugin needs one, name that exact version in the same command, for example `agent-env plugin add agentenv-grader 'agentenv-grader-core==0.2.0b1'`; `plugin remove` removes it later the same way.
- A uv project whose environment is set by `UV_PROJECT_ENVIRONMENT` is found from the current directory, as uv finds it, so run `add` and `remove` inside the project. In a uv workspace whose root has no `[project]` table, `add` goes to the one member that declares `agentenv-framework` (`uv add --package MEMBER`). `remove` drops a package from every member and group that declares it (`--package`, `--group`, `--optional`, `--dev`).
- A uv project is never synced exactly, so packages from extras and anything installed outside the lock, agent-env included, stay installed.
- pipx and a virtualenv leave a removed plugin's own dependencies installed, as `pip uninstall` does. Left to itself, `pipx uninject` would also uninstall everything nothing else requires, agent-env included.
- A uv tool installed with constraints, overrides or executables from another package is refused: `uv tool install` cannot take those back from agent-env, so make that change with uv.
- One `add` or `remove` runs at a time per environment; a second one is refused until the first finishes. The lock is a file in `$XDG_STATE_HOME/agent-env/locks` (by default `~/.local/state/agent-env/locks`). If the installer's record changes some other way while a change waits for confirmation, the change is refused rather than run from its outdated plan.
- A plain virtualenv first shows the installer's own dry run of what would change. Every mode reports what did change afterwards, calling out a change to agent-env itself.
- `add` then checks each new plugin package in a fresh interpreter: it must declare at least one `agent_env.*` entry point, and every contribution must be `active` or `replaced`. `show` reports whether importing it sets `AGENT_ENV_CONFIG`. If the check fails, the installer fails or removes another plugin, or the change is interrupted (Ctrl-C or SIGTERM), `add` puts the environment back as it was: every package at its old version from its old source, and the uv tool receipt, pipx metadata, or `pyproject.toml` and `uv.lock`, byte for byte. The installer runs with Ctrl-C ignored, since one stopped halfway leaves packages no installer can read: it finishes, and then the change is rolled back. A second Ctrl-C stops it at once. It keeps the terminal, so its own prompts, such as git's for credentials, still work. When the environment cannot be put back exactly, `add` says which packages differ and the installer command that reinstalls them. Adding what is already installed changes nothing and succeeds, and a bare name that a uv tool already lists keeps its version pin. `--keep` keeps a change that failed the check instead.
- `remove` refuses, unless `--force`:
  - while another installed package requires the package;
  - while the config file names something only it provides (`[sandbox] default` or `agent_default`, a provider table, a `type_aliases` target), or an `impls` list, provider, store or runner `impl` names one of its modules;
  - while stored documents use env, artifact or task-step types only it provides. This is checked for a local document store, and for a remote one only with `--check-usage`; a store that cannot be read also blocks, until `--force`.

  It never removes agent-env itself. A package that declares no `agent_env.*` entry point is removed only when the installer records it on its own (a uv tool's `--with`, a pipx injection, a project dependency), as it does for a package named next to a plugin in `add`; after removing a plugin, `remove` names any such package that only the plugin needed. In a uv tool or a uv project, removing a plugin also removes the plugins that only it needed, as the installer's rebuild would, so `remove` names them first and checks them like the package itself. If the installer fails, removes a plugin it did not name, or the change is interrupted, `remove` puts the environment back the same way `add` does.
- Plugins are trusted code: nothing installs one implicitly, from a task document, a config file or anywhere else.
- A plugin that breaks CLI startup does not lock you out. A root option whose callback raises or exits while its flag is absent is skipped, with a warning, and reported as `failed`; a flag you pass still fails, naming the plugin. A CLI plugin that exits while being imported is skipped like any load failure. For a plugin that hangs or kills the interpreter while being imported, uninstall it with the installer directly: the command `plugin remove --dry-run` would print.

### Building a platform plugin

One installable package can combine all of the above: bundled config files, a root option that selects one per invocation, store and provider classes, a `Runner`, custom steps, and explorer routers. The `--tenant` demo above is that pattern in miniature. Its callback points `AGENT_ENV_CONFIG` at a bundled `tenants/<name>.toml`; a name that does not exist fails loud with `ConfigError` on first use, while `--help` still works. A hosted control plane serves the explorer app from its own server process and lists its public hostnames under `[explorer] allowed_hosts` (see the comments in `.agentenv/config.example.toml`). A durable runner is the plugin's responsibility: this repository ships only `LocalRunner`.

### Plugin compatibility

What a plugin can build on, how to declare the agent-env it needs, and what agent-env does when the installed one does not fit.

**The plugin surface.** A plugin may rely on these. Anything else, including every underscored name other than a method a subclass must implement (`EnvStateProvider._teardown`), is internal and can change in any release.

- The entry-point groups and their rules, in [Register types from an installed package](#register-types-from-an-installed-package) and [CLI plugins, root options, explorer routes](#cli-plugins-root-options-explorer-routes).
- The base class each group names, with its public methods and attributes: `agent_env.env.env.Env`, `agent_env.task_step.task_step.TaskStep` and the `agent_env.task_step.context.TaskStepContext` a step runs with, `agent_env.artifact.artifact.Artifact`, `agent_env.providers.sandbox_providers.sandbox_provider.SandboxProvider`, `agent_env.providers.env_state.env_state_provider.EnvStateProvider`, `agent_env.providers.env_providers.env_provider.EnvironmentProvider`, and `agent_env.explorer.plugin.ExplorerPlugin`.
- The two functions an env that uses an environment provider calls: `agent_env.providers.env_providers.env_provider.build_env_provider` and `agent_env.env.store.register_env_instance`.
- What an environment provider reads to deploy a built-in env: `agent_env.env.envs.mcp_server.MCPServerEnv.docker_image_artifact` and `agent_env.env.envs.mcp_server.MCPServerEnv.environment_name`; `agent_env.env.envs.website.WebsiteEnv.backend_docker_image_artifact`, `agent_env.env.envs.website.WebsiteEnv.frontend_docker_image_artifact` and `agent_env.env.envs.website.WebsiteEnv.environment_name`; `agent_env.env.envs.multi_env.MultiEnv.mcp_server_envs`, `agent_env.env.envs.multi_env.MultiEnv.website_envs` and `agent_env.env.envs.multi_env.MultiEnv.name`; and each image's `agent_env.artifact.artifacts.docker_image.DockerImageArtifact.image_name`.
- The top level of `agent_env.plugins`, including `settings`, and the `[plugins.<package>]` table it reads ([Plugin settings](#plugin-settings)).
- The `plugin --json` output, which has its own rules: [Plugin report format](#plugin-report-format).

The classes a config `impl` names, such as stores and runners, are not on the list yet.

**Changes before 1.0.** Every merged change can ship as a release, several a day. A change that breaks the plugin surface is marked with `!` after the scope in its pull request title, which becomes its commit title, as in `feat(plugins)!: …`. Where the old behaviour can be kept for a while, it is deprecated first: it keeps working and emits a `DeprecationWarning` that names what replaces it. How long that lasts is not fixed before 1.0.

The `plugin-api` CI job holds pull requests to this. It compares the listed base classes, `TaskStepContext`, the two functions, the env attributes and the top level of `agent_env.plugins` with the pull request's base (`.github/scripts/check_plugin_api.py`), and fails on a break the title does not mark; with the `!`, it lists the breaks and passes, and editing the title re-runs it. A break is what fails code written against the old surface:

- for a caller, a name, parameter or `__all__` entry that is removed or renamed, a new required parameter, a parameter that can no longer be passed as before, or a changed default or constant, including the group-name constants `agent_env.plugins` exports, compared by value;
- for a subclass of a base class, a new abstract or required method, a method that becomes abstract or required, a new `ClassVar` with no value, a method that changes between plain, `async`, `classmethod`, `staticmethod` and property, and a base-class method that accepts more than before: a new parameter, even an optional one, a parameter that stops being required, a new `*args` or `**kwargs`, or a keyword-only parameter that can now be passed by position, since an override written for the old signature fails when core passes it;
- for construction, a changed `@dataclass(...)` option or pydantic `model_config`, a new metaclass, or a new `__init_subclass__` that can raise, which refuses a subclass.

A member every subclass must implement is declared `@abstractmethod`, or listed in its registry's `_MUST_IMPLEMENT`, so that the check enforces it; a new method that only raises `NotImplementedError` gets a note but does not fail the job.

agent-env ships `py.typed`, so mypy and pyright check a plugin against its annotations. Import each name from the module the list above gives, such as `TaskStepContext` from `agent_env.task_step.context`: pyright, and mypy under `--strict`, treat a name a module only imports as private to that module. Type annotations are not compared, so a release can correct one without a `!`. The check can report a change no plugin notices, such as a new parameter core never passes to an override; mark it all the same. It does not see a change in behaviour, in a type the surface only names in a signature, such as `DeployedEnv`, in a pydantic field's requirements or the order of dataclass fields, in an abstract method a new base from outside agent-env brings, or in a registration rule other than abstract methods and `_MUST_IMPLEMENT`; review covers those. A `!` can also mark a break outside the plugin surface.

**Declaring the agent-env your plugin needs.** Declare a floor, `agentenv-framework>=X`, where `X` is the oldest release you test against. Leave out a ceiling such as `<1`: before 1.0 it would not guard against a change in a 0.9 release, and it would keep your plugin from installing with the next major one. Pin exact versions in the application or image that installs your plugin, not in the plugin. If your plugin imports `agentenv_protocol` itself, declare that as well.

**What agent-env checks.** Before it imports a plugin, agent-env compares the plugin's requirements on `agentenv-framework` and `agentenv-protocol` with the versions installed. A plugin they exclude is not loaded:

- `plugin list`, `show` and `check` report each of its contributions as `failed` with the code `incompatible-core`, with or without `--no-load`, and `check` exits 1;
- using one of its types names the requirement, as in `needs agentenv-framework>=0.9.1220 (installed: 0.9.1218)`;
- it claims none of its names, so a plugin that can load and registers the same name does not conflict with it.

An installer that resolves dependencies never gets you there; `pip install --no-deps` or a forced install can. There is no override: upgrade agent-env, or install a version of the plugin that fits. A requirement under an extra, or whose environment marker is false, does not count, and an agent-env with no installed metadata, such as a source tree on `sys.path`, is not checked. Requirements between plugins are the installer's to check; `pip check` lists any that are unmet.

## Contribute, release, license

### Development setup and test tiers

[CONTRIBUTING.md](CONTRIBUTING.md) is the contributor guide: open an issue before anything larger than a bug fix, one logical change per pull request with tests, pull request titles in the form `type(scope): summary`, an approving review from a code owner (`CODEOWNERS`) and green CI. [AGENTS.md](AGENTS.md) is the repository map and conventions file for contributors and coding agents; `CLAUDE.md` imports it.

Set up with `uv sync --extra dev` or `make install` (both in [Install the packages](#install-the-packages)). Tests are tiered by path:

| Command | Runs | Needs |
|---|---|---|
| `make unit-test`, or `python -m pytest tst/unit packages/agentenv-protocol/tests -n auto -q` | 3,086 unit tests (3,084 pass, 2 skip) in about 20 seconds; IP sockets are blocked by `pytest-socket` and AWS calls go to `moto` | nothing external |
| `make int-test-fast` | the 129 integration tests not marked `int_test_slow`, in parallel | Docker and a local OCI registry on `:5000` |
| `make clean-install-test` | both distributions built as the release builds them, installed into a fresh venv from public PyPI with nothing else, and `agent-env run hello` run twice by name and checked through the local store | Python 3.11, uv and git |
| `make int-test-slow` | the 285 `int_test_slow` tests (12 of the 27 integration modules; real image builds and sandboxes), serially | Docker and the registry; a model endpoint, a remote sandbox or a registered default agent for some |

Run the unit tier from the checkout root with `AGENT_ENV_CONFIG` unset; the Makefile targets hardcode `.venv/bin/python`. The integration tiers were not run for this guide. A test may skip only for a declared capability gap, with the reason `agentenv-capability-missing: <name>` where `<name>` is `model_endpoint_configured`, `remote_sandbox`, `default_a2a_agent` or `mcp_server_sources` (see `tst/util/capabilities.py`); CI rejects any other skip reason.

CI is GitHub Actions. `.github/workflows/local-backends.yml` runs the `unit`, `integration-local` and `integration-local-slow` jobs on Python 3.12, installed from public PyPI with no secrets and a `registry:2` service container for the integration jobs. The `unit` job also fails if `uv.lock` resolves anything from a registry other than PyPI, if `agentenv-protocol` is not the editable workspace member, or if `uv.lock` is out of date, before or after a trial run of the release bump (`scripts/bump_version.py`). The public jobs skip `tst/integration/env/gateway/gateway_test.py`, which needs an x86 Chromium build and, for its virtual-clock tests, MCP server sources named by `AGENT_ENV_TEST_MCP_SERVERS_DIR`; `make int-test-slow` runs it, so run that locally when a change touches the gateway and say so in the pull request. `.github/workflows/plugin-api.yml` runs the `plugin-api` job on every pull request, including a title edit, and on `main`; it fails a break to the plugin surface that the title does not mark (see [Plugin compatibility](#plugin-compatibility)). `.github/workflows/clean-install.yml` runs the `clean-install` job on every pull request and on `main`, on Python 3.11: it builds both distributions as the release does, fails if the wheel leaves out a tracked example file or bundle entry point, installs the two wheels into a fresh venv from public PyPI with an allowlisted environment (no AWS credentials, config or plugin), runs `agent-env run hello` twice by name, and checks through the local store that both runs completed with a score of 1, left no sandbox work folder, and the second changed no artifact, task, ledger row or stored object the first wrote. The required checks on `main` are `unit`, `integration-local`, `integration-local-slow`, `installer`, `plugin-api` and `clean-install`. Dependabot (`.github/dependabot.yml`) opens weekly updates for the uv lock and the pinned actions.

### Conformance suites

`tst/store/` holds four backend-neutral suites: `conformance.py` (`DocumentStore`, 35 cases), `object_conformance.py` (`ObjectStore`, 17, plus 3 in `GRANT_CASES` for a store that sets `supports_transfer_grants`), `secret_conformance.py` (`SecretStore`, 3; seed the backend with `FIXTURE` first) and `image_conformance.py` (`ImageStore`, 5; push and pull need Docker). Each exposes a `CASES` list; a backend test builds its store fixture and parametrizes over `CASES`, exactly as `tst/unit/store/sqlite_document_store_test.py` does. `GRANT_CASES` send each grant to the provider over HTTPS, so they need a real store rather than a local stand-in, and on `S3ObjectStore` the namespace case needs long-term credentials. The unit tier runs `CASES` for the local stores, for `DynamoDbDocumentStore` through moto and for `GcsObjectStore` against a stand-in client; the MongoDB-protocol stores need a live database, so it does not run theirs.

The suites are not packaged in the wheel. Run them from a source checkout with `PYTHONPATH=.`:

```
cd <agent-env checkout> && PYTHONPATH=. python -m pytest -p no:cacheprovider -q <path>/tests/test_store_conformance.py
```

`pytest tst/store` alone collects nothing; the cases only run through a consumer test. Protocol package tests live in `packages/agentenv-protocol/tests/` and are part of the unit tier.

### Documentation map

| Document | Covers |
|---|---|
| [`packages/agentenv-protocol/README.md`](packages/agentenv-protocol/README.md) | wire contract, environment server SDK, A2A agent framework |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | development setup, test tiers, CI jobs, pull request rules |
| [`AGENTS.md`](AGENTS.md) | repository map, configuration and extension-point summary, conventions for contributors and coding agents |
| [`SECURITY.md`](SECURITY.md) | private vulnerability reporting and supported versions |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Contributor Covenant |
| [`.agentenv/config.example.toml`](.agentenv/config.example.toml) | the all-local configuration to copy |
| [`.env.example`](.env.example) | a commented reference of the `AGENT_ENV_*` variables; agent-env never loads this file, export what you need yourself. Its `AGENT_ENV_ENVIRONMENT` line is read only by an installed plugin, never by agent-env |

Full configuration, CLI, step and extension references, and the [Run on Google Cloud](#run-on-google-cloud) guide, have not been split out of this README yet.

### Versioning and compatibility

The `agentenv-framework` distribution and `agentenv-protocol` are versioned separately (`0.9.x` and `0.1.x` today), both in their `pyproject.toml`; agent-env releases carry a `vX.Y.Z` tag, and agentenv-protocol is bumped in the same commit and has no separate tag today. There is no `agent_env.__version__` attribute; `agent-env --version` prints the installed version. Protocol extensions carry their version in the URI (`urn:agentenv:clock/v1`, `urn:agentenv:agent-config/v1`, `urn:agentenv:trajectory/v1`). One version can take more than one request shape: under `v1` the skill, trajectory, snapshot and changelog extensions accept object-transfer requests next to their older shapes, and the request field lists on an agent's card say which ones that agent takes. Renamed CLI commands are removed outright; no deprecated aliases exist at this version (see [Deprecated aliases and legacy paths](#deprecated-aliases-and-legacy-paths)). What plugins may rely on, and how changes to it are made, is in [Plugin compatibility](#plugin-compatibility); the plugin commands' `--json` output has its own format version and rules ([Plugin report format](#plugin-report-format)). Beyond those, a written compatibility and deprecation policy does not exist yet.

### Releases

A release is a version bump in both `pyproject.toml` files plus a `vX.Y.Z` tag. Maintainers cut releases: the bump is automated when a labelled pull request merges, so contributors do not edit `version` or push tags (see the Releases section of [CONTRIBUTING.md](CONTRIBUTING.md)). Neither package is published to a public index yet, and there is no `CHANGELOG.md`.

### Support, security, license

Report bugs and gaps as issues against this repository, with the installed `agentenv-framework` version and the sandbox backend in use. Report vulnerabilities privately through the contact in [SECURITY.md](SECURITY.md), not in public issues; only the latest release is supported, so reproduce against it first. Contributors follow [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md); every pull request needs a code-owner review. agent-env and agentenv-protocol are licensed under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Their third-party dependencies and those dependencies' licenses are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
