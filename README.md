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
12. [CLI reference](#cli-reference)
13. [Extend agent-env](#extend-agent-env)
14. [Contribute, release, license](#contribute-release-license)

## What agent-env is

agent-env gives you five primitives: an **environment** an agent acts in, an **artifact** that seeds or captures state, a **task** that strings deploy, prompt and grading steps into a DAG, an **agent** that speaks the A2A protocol, and an **eval** that groups tasks. Every primitive is a versioned document in a store you choose; the default store is a local SQLite file, so a laptop with Docker is a complete installation.

### The environment model

- **Environment card.** Every environment server publishes `GET /.well-known/agent-env.json`: its `name`, `protocolVersion` (`1.0`), the JSON-RPC data-plane URL, its `capabilities` (`tools`, `operations`, `extensions`) and any `children_environments`. The name comes from `@environment_card(name=...)`, then the `ENVIRONMENT_NAME` variable, then the class name.
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
- **Trust model.** The `local` sandbox runs environments and agents as ordinary containers on your Docker daemon, one compose stack per deploy, with host ports published on the machine. Treat it as a development convenience, not a security boundary. The explorer served by `agent-env up` binds `127.0.0.1` only, has no `host` setting, and rejects foreign `Host` headers with HTTP 421. Configuration values hold references (`env:NAME`, `secret:KEY`), not secrets; the run context written to disk strips API keys. Every HTTP client agent-env makes verifies TLS; there is no setting or variable that turns that off.

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

### Optional extras and bundled cloud SDKs

| Extra | Contents | Needed for |
|---|---|---|
| `agentenv-framework[explorer]` | `fastapi`, `uvicorn` | `agent-env up` (refuses to start without it) |
| `agentenv-framework[dev]` | `explorer` plus `moto`, `psycopg2-binary`, `pytest` and its plugins (`pytest-asyncio`, `pytest-dependency`, `pytest-socket`, `pytest-timeout`, `pytest-xdist`) | running the test suite |
| `agentenv-protocol[agent]` | `a2a-sdk[http-server]`, `uvicorn`, `boto3`, `regex` | authoring and serving your own A2A agent |

The runtime dependencies of `agentenv-framework` are `agentenv-protocol`, `boto3`, `click`, `httpx`, `litellm` (the 1.96 line), `mcp` (below 2.0), `a2a-sdk` (pinned to 0.3.26), `pydantic`, `pymongo`, `pyyaml`, `modal` and `e2b` (the 2.46 line). The cloud SDKs install every time and stay inert until a config table or a `--sandbox` flag selects them; the default sandbox and stores are local. There is no `asyncpg`; `psycopg2-binary` comes only with the `dev` extra.

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
  plugin     Inspect installed plugins.
  task       Task commands.
  up         Start the local agent-env stack (stores, runner, explorer)...
```

Installed plugins can add groups and root options; they appear in this listing. There is no `--version` flag (`agent-env --version` answers `No such option '--version'. Did you mean '--verbose'?`). To read the installed version:

```bash
python -c "import importlib.metadata as m; print(m.version('agentenv-framework'))"
```

The second check is `agent-env config show`: it prints which config file won, or `(none)`, and what every section resolved to (see [Inspect the resolved configuration](#inspect-the-resolved-configuration)). Running `--help`, `config show`, `config explain`, `config sources`, `config debug`, `plugin list`, `plugin show` or `plugin check` writes nothing to disk, though a plugin's own import code may. The first command that resolves a store (for example `task get`) creates `.agentenv/` in the working directory (or next to the discovered config file) with an auto-managed `.gitignore`.

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

Every local deploy resolves two bootstrap environments by id: the service database `default-db` and the gateway `default`. Register them once per project directory; the first `service-db` build takes minutes:

```bash
agent-env env service-db put --id default-db --platform linux/arm64
agent-env env gateway put --id default --platform linux/arm64
```

The first `put` is the first store write: it creates `.agentenv/document_store/documents.db`, `.agentenv/object_store/` and `.agentenv/.gitignore`. Images push to a local OCI registry on `127.0.0.1:5000`. The first push starts the `agentenv-registry` container (`registry:2`, `--restart unless-stopped`) that hosts it; nothing is started if a registry already answers there.

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
agent-env env mcp-server put --id <env-id> --dockerfile items_env/Dockerfile --context items_env --platform linux/arm64 --skip-validation
```

```
Derived environment_name='items' from the environment card.
Building MCP server Docker image...
Created artifact: id=mcp-server-<env-id> version=1
Created MCPServerEnv: id=<env-id> version=1 environment_name=items service_version=1
```

Passing `--context <repo>/tst/data/agentenv_mcp` directly fails at `COPY agentenv_protocol`; that directory is not a self-contained build context. `--skip-validation` skips the release gate, which needs a registered `a2a-default` agent and a model endpoint; see [Build, register and the release gate](#build-register-and-the-release-gate). Repeating `put` on the same id appends a new version. The image lands in the local registry as `localhost:5000/mcp-server-<env-id>:v1`; a `registry:2` container named `agentenv-registry` is started on first push if none is listening.

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

The repository does not yet ship an example task file, so write this one as `task.json`, substituting your environment id and the absolute path of your project directory:

```json
[
  {"id": "deploy-env", "type": "deploy_env", "env_id": "<env-id>", "ttl_seconds": 1800},
  {"id": "deploy-agent", "type": "deploy_agent", "env_ids": ["<env-id>"], "a2a_agent_id": "a2a-default", "ttl_seconds": 1800},
  {"id": "prompt", "type": "prompt_agent", "prompt_id": "p1", "timeout_seconds": 600,
   "prompt": "Using the tools available to you, add one item named 'readme' to the item store, then list all stored items and reply with the exact list of items returned by the tool.",
   "trajectory_output_prefix": "file:///<absolute-path-to-project>/.agentenv/object_store/prompt_agent_trajectories/p1/"},
  {"id": "verify", "type": "rubrics_verifier", "prompt_id": "p1", "verifier_id": "rubric", "use_agent_judge": false,
   "output_format": "rubric_binary", "score_aggregator": "all_pass",
   "criteria": [{"id": "item_added_and_listed", "weight": 1,
                 "criterion": "The agent called a tool to add an item named 'readme' and its final response reports the stored items, including 'readme'."}]},
  {"id": "snapshot", "type": "snapshot_env", "env_id": "<env-id>", "fail_task_on_error": false}
]
```

Two lines are workarounds, which is why this step is gated. `trajectory_output_prefix` must be a `file://` URL inside `.agentenv/object_store/`; the reason is in [What every run needs](#what-every-run-needs). `use_agent_judge: false` grades with a direct model call instead of deploying a third sandbox for a judge agent; that judge call is the only model call in this chain. Then create and run:

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

`GET /.well-known/agent-env.json` serves the card: `name`, `protocolVersion` `1.0`, `url` `/agentenv`, and `capabilities.{tools,extensions,operations}`. The name resolves from `@environment_card(name=...)`, then the `ENVIRONMENT_NAME` variable, then the class name; `{environment_name}` in a tool name resolves to it at mount. Each `@tool` method becomes a real MCP tool with an `inputSchema` derived from the signature. Only `@tool` methods appear on the card; tools registered through `self.mcp.tool(...)` are served by MCP `tools/list` but not advertised.

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
| `mcp_server` | `agent-env env mcp-server put --id <env-id> --dockerfile <Dockerfile> --context <dir>` (or `--dockerfile-github-url`) | One protocol-conformant server image | Seed loads need the data plane implemented |
| `multi` | `agent-env env multi put --id <env-id> --mcp-server <id>[:version] --mcp-server <id>[:version] [--website <id>[:version]] [--name <label>]` | Several servers behind one gateway and one service database; universes and snapshots | Refs pin the latest version at put time; only `mcp_server` and `website` ids. `--name` fixes the name agents see the MCP server under (`mcp__<label>__<tool>`); without it each deploy draws `env` plus four random digits, so declare one when rubric text or tooling needs a stable prefix |
| `website` | `agent-env env website put --id <env-id> --backend-dockerfile <f> --frontend-dockerfile <f>` | A browsable web app proxied at `<gateway>/website/<name>/` | Register `agent-env env website-browser put` once first (not exercised for this guide) |
| live endpoint (custom) | a custom `Env` subclass whose documents carry an http(s) `mcp_url` attribute, registered through `[envs] impls` and saved with `put` from Python; no CLI | An MCP server that already runs elsewhere | Never deployed: a `deploy_agent` step that lists the id in `env_ids` reads `mcp_url` from the env document; no seed, reset or snapshot; a bearer token is passed per run with `task run --remote-token <env-id>=<token>` (see [Custom envs, task steps, artifacts](#custom-envs-task-steps-artifacts)) |
| custom | `.agentenv/config.toml`: `[envs] impls = ["mypkg.envs:MyEnv"]` (subclass `Env`, set `type`) | A runtime the built-ins do not cover | No CLI `put` unless your plugin adds one |

### Author seed data as artifacts

An `EnvironmentArtifact` wraps one file (a `FileArtifact` named `<artifact-id>-file` is created for it) and pins it to an environment name; an `EnvironmentUniverseArtifact` bundles several with pinned `id:version` refs.

```bash
agent-env artifact environment put --id <artifact-id> --description 'items seed' --environment-name items items_env/seed.json
agent-env artifact environment-universe put --id <universe-id> --environment-artifact <artifact-id>:1
```

`--environment-name` is required here (the CLI does not derive it from a card for artifacts) and must equal the environment's card name. The seed-schema version lives on the environment (`env mcp-server put --service-version`, default `1`), not on the artifact. Files land in `.agentenv/object_store/artifacts/file/<artifact-id>-file/<version>/`. Loading needs a deployed instance; see [Load universe data, snapshot and export](#load-universe-data-snapshot-and-export).

## Register, deploy, connect and load an environment

Registering turns an image into a versioned environment document; deploying provisions it behind a gateway that composes the card, proxies MCP and REST, and records tool calls. Every command below ran on the local stack (SQLite documents, filesystem objects, a local OCI registry on `127.0.0.1:5000`, the `local` Docker sandbox). A configured object store, remote registry and remote sandbox provider take the same commands and return `https://` URLs; see [Choose compute, network policy and state](#choose-compute-network-policy-and-state).

### Bootstrap prerequisites

Deploy resolves the gateway environment `default` and the service-db environment `default-db` by id. Register them once per project with the two `put` commands in [Start the local stack](#start-the-local-stack). `agent-env up` runs the same two puts when missing, always for `linux/amd64` (see [Verify and platform notes](#verify-and-platform-notes)). The first `service-db` build compiles a Postgres MCP sidecar from source and takes minutes. Images push to `localhost:5000/<artifact-id>:v<version>` (the `agentenv-registry` container starts on first push) plus a tarball in `.agentenv/object_store/`; documents go to `.agentenv/document_store/documents.db`. Website environments also need `agent-env env website-browser put`.

### Build, register and the release gate

The build-and-register command is shown in [Build and register the example environment](#build-and-register-the-example-environment). The output reports `Derived environment_name='items' from the environment card`, the image artifact `mcp-server-<env-id>` and the environment version. `--environment-name` is optional when exactly one `@environment_card(name=...)` is in the build source. `put` on an existing id appends a version; nothing is overwritten.

**GitHub-sourced builds.** `env mcp-server put --dockerfile-github-url https://github.com/<owner>/<repo>/tree/<ref>/<path>/Dockerfile` (optionally `--docker-context-github-url` for a context directory other than the Dockerfile's parent, same repository and ref) and `env website put --backend-dockerfile-github-url ... --frontend-dockerfile-github-url ...` clone the repository instead of reading a local build context. The Python equivalents are `MCPServerEnv.put_from_github(...)` and `WebsiteEnv.put_from_github(...)`; the resulting image artifact and environment are the same as from a local build. Public repositories need no credentials. For a private one, pass `github_token=` (any token GitHub accepts as the password for `git clone`); the CLI reads it from `GITHUB_TOKEN`. The clone, `docker build`, `docker push` and `docker save` run through the `[sandbox] default` provider's VM path; on the `local` provider that is your host shell as the invoking user (no VM, no container; `sudo` is stripped), starting with `apt-get install git`, so it needs a Debian-like Linux host with root, Docker, and push access to the configured image registry (on macOS or without root it fails earlier with `Script failed (exit 127)`). The image is always built for `linux/amd64` (`--platform` is ignored with a warning) and then uploaded through a presigned URL, which the local filesystem object store cannot issue (`GitHub image builds need a signable object store`), so the all-local stack cannot complete a GitHub build. Use a local Dockerfile there, or configure a VM-capable sandbox provider plus an object store that can presign uploads (the bundled S3 store does). Described from the code; not exercised for this guide.

Without `--skip-validation`, `put` runs the release gate: a nine-step validation task (deploy, card, core protocol, tool schema and conformance checks, then an agent probe and assessment). The environment is registered before the gate. The gate needs the default agent (`[agents] default_a2a_agent_id`, `a2a-default` when unset; see [Agents](#agents)) to be registered, and a model endpoint; on a fresh store with the default config it aborts at `deploy_agent` with `A2AAgent a2a-default not found`, stamps no verdict, and leaves the validation deploy's containers running. Use `--skip-validation` until the agent exists. `--override` publishes despite a failed gate and records the override, but still needs the agent.

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

Environments and agents have separate sandbox slots, `[sandbox].default` and `[sandbox].agent_default`, overridden per command by `env deploy --sandbox`, `a2a-agent deploy --sandbox` and `task run --env-sandbox` / `--agent-sandbox`. Built-ins are `local`, `modal`, `modal_vm` and `e2b`; custom providers register under `[sandbox.providers.<name>]`, and any slot takes a comma-separated chain that falls through on failure (see [Configuration reference](#configuration-reference)). The `local` provider runs on the host Docker daemon with no isolation; remote providers need an object store and remote registry and return the same record over HTTPS (TTL enforcement: [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down)). Sandbox mode comes from how the sandbox is started: `modal` deploys an environment as separate containers (container-mode); `modal_vm` and `e2b` give a VM; `local` runs an environment as a Compose stack on the host daemon (VM-mode) but starts an agent, or a `deploy_sandbox` sandbox, as a single container (container-mode). `env snapshot` refuses container-mode sandboxes, so it excludes `modal`, not `local`; on any provider it also needs an object store that can presign uploads (S3-compatible), which the bundled filesystem store cannot, so the zero-config local setup fails at the upload step. Egress allowlists and template rules for `e2b` are in [`docs/e2b-sandbox-provider.md`](docs/e2b-sandbox-provider.md).

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

Against a single-server instance the data loads but the command exits 1 on a `snapshot_baked` attribute error; use the per-artifact loader there. `agent-env env snapshot --instance-id <instance-id>` bakes the loaded database into an `env-snapshot-<env>` image for clean resets; it supports `multi` environments only, after a universe load, not on a container-mode sandbox, with `local_postgres` and a presign-capable object store (the filesystem store is not), and `--snapshot-after-load` on the universe load bakes right after ingest. The `snapshot_env` task step exports a run's end state as an `EnvironmentUniverseArtifact`; see [Run tasks: locally, then at scale](#run-tasks-locally-then-at-scale). `env multi validate-universe-compatibility` and `artifact environment-universe compatible-envs` exist; not exercised for this guide.

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

`close()` runs `docker compose down -v --remove-orphans` for the local sandbox and terminates the VM for remote providers (the gateway then answers 404); the same command in the work directory is equivalent locally.

For an agent deployed by a task run, take `deployed_agents[0].instance_id` from the context JSON and terminate its sandbox through the provider:

```python
import asyncio
from agent_env.a2a_agent.store import get_a2a_agent_instance_store
from agent_env.providers.sandbox_provider import build_sandbox_provider

async def main():
    dep = get_a2a_agent_instance_store().get("<agent-instance-id>")
    sb = await build_sandbox_provider(dep.sandbox_type).get_sandbox(dep.sandbox_id)
    await sb.terminate()

asyncio.run(main())
```

With the `sandbox_id` from the run output's `deployed-agent:` block (or `deployed_agents[0].sandbox_id` in the context JSON) the store lookup can be skipped: `await (await build_sandbox_provider("local").get_sandbox("<sandbox-id>")).terminate()`. To find instances you did not keep ids for, query the document store: `from agent_env.store import Filter, Sort`, then `get_config().get_document_store().query("env_instances", Filter.of(env_id="<env-id>"), sort=Sort.by("created_at_utc", descending=True))` (and `a2a_agent_instances` with `Filter.of(agent_id=...)`). Closed instances stay listed with no status flag, so keep the ids from the run output when you can.

`task run` leaves its environment and agent running (see [What every run needs](#what-every-run-needs)). Left behind: the instance record (no status field), the `esi-...` state record, the work directory, images in the registry and daemon, and `.agentenv/` documents and tarballs. `agent-env env teardown-env-state --instance-id <esi-id>` retires a state record by stamping its expiry. Reset means re-seeding through the loader or restoring a snapshot; there is no reset command.

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

On the default local object store the `prompt_agent` step also needs a `trajectory_output_prefix`; the reason and the exact field are in [What every run needs](#what-every-run-needs). A configured object store needs no such field.

### Tasks without an environment server

Coding-style tasks need a machine and a container, not an MCP server. Three steps cover that: `deploy_sandbox` provisions a bare sandbox (`sandbox_name`, `image`, `cpu`, `memory_mb`, `disk_size_gb`, `ttl_seconds`, `exposed_ports`); `run_docker_container` builds an image from a docker-context artifact or URL and starts it on that sandbox (`container_name` defaults to `task-container`, with `ports`, `env_vars`, `build_args`, `command_override`, `ready_command`, `volumes`); `deploy_agent` with the same `sandbox_name` places the agent on that machine. Reward comes from `run_container_unit_tests_verifier` (a command's exit code plus optional reward and result files) or from `run_code`, which runs a script artifact against an environment or agent and stores its output under `metadata.script_results[<result_id>]`. Described from their definitions; not exercised for this guide.

### Step catalogue by family

The runtime ships 48 built-in step types. Print the live list from Python:

```bash
python -c "from agent_env.task_step.registry import get_task_step_registry as g; r=g(); print(len(r)); print(sorted(r))"
```

| Family | Step types | Needs a model |
|---|---|---|
| Deploy and provision | `deploy_env`, `deploy_sandbox`, `run_docker_container`, `deploy_agent`, `install_agent`, `reset_env`, `sync_env_clock`, `apply_server_config`, `modify_env_tool_access`, `register_env_triggers`, `register_agent_triggers`, `peer_agents`, `add_skills`, `build_mcp_cli` | no |
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

`validate` runs each step's `preflight()` and exits non-zero on a problem. Only steps that implement a preflight are checked: `run_code` verifies that its script artifact resolves to the right type, and custom steps may add their own. It does not resolve environment or agent ids, so a missing `a2a-default` surfaces at run time, not here. `task create` runs the same checks and refuses to save a failing task unless `--skip-validation` is passed. For partial runs, `task run --start-step <n> --context-json <saved-context>` resumes from a persisted context (see [Resume and partial runs](#resume-and-partial-runs)); the Python `Task.run()` also accepts `end_step`, which the CLI does not expose.

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
| `verify_sandbox` | the agent's live sandbox (`agent_name`, `base_dir`) | `probe_file_exists`, `probe_dir_exists`, `probe_file_contains`, `bash_cmd_succeeds` (with `shell_timeout_seconds`); unknown types are kept as rows flagged `skipped: true` and excluded from the score |
| `env_outcome_verifier` | the deployed environment's MCP URL for `env_id` | a Python file artifact (`file_artifact_id`) exposing `async def verify(mcp_url)` that runs in the agent-env process and returns the result rows, stored unchanged; the Python `put(verify_script_file_path=...)` helper uploads it as `<id>-verifier-script` |
| `run_container_unit_tests_verifier` | a container started by `run_docker_container` (`sandbox_name`, `container_name`) | `command`, `setup_commands`, `user` (default `root`), `timeout_sec` (default 300), `env_vars`; records `command`, `exit_code`, `timed_out`, `stdout_head`, `stderr_head`, `extracted_files` from `result_paths`, and one row (`id` `reward` when `reward_path` is set and graded from that file, else `exit_code` from the exit status). It uploads stdout and stderr as file artifacts by `s3://` URL, so it needs an S3-backed `[stores.object]`; on the local object store it fails with `ConfigError: This caller needs an S3 object store` |

Each accepts `score_aggregator` and `verifier_id`, and every row copies the criterion you wrote (`agent_prompt_response_verifier`, `verify_sandbox`) or your script's output (`env_outcome_verifier`). Described from their step definitions; not exercised for this guide.

### Combining signals

`aggregate_verifiers` takes `verifier_ids`, merges their `results` rows (dropping `skipped` rows), applies `score_aggregator` (default `weighted_average`), and writes `{score, source_verifier_ids}` under its own `verifier_id`. The merged entry has no `results` of its own; rows without a `result` field (for example an `env_outcome_verifier` script that returned only `score`) count as failed under `all_pass` and `any_pass`. Per-criterion rows stay available in the source entries, so a run can report both a single number and the breakdown behind it. Give it `depends_on` edges to the verifiers it reads when they run on concurrent branches.

Scores stay per run; `eval run` reports completion per task run, not a score (see [Evals](#evals-run-a-task-set-under-different-agents-and-models)). Pass@k, means across seeds, and comparisons between agents or models are computed from the per-run context files.

## Run tasks: locally, then at scale

One primitive executes a task: `Task.run()`. The CLI wraps it three ways: `task run` for one task, `task run-batch` for one task over many seed rows, and `eval run` for a task set. Each writes one context JSON per run and leaves the deployed sandboxes running; remote providers reclaim them at the TTL, the local sandbox does not (see [Tear down](#tear-down)).

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
| `--remote-token ENV_ID=TOKEN` | Bearer token carried to an environment referenced by its live `mcp_url` without a `deploy_env` step. |
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
4. On the default local object store, `trajectory_output_prefix` on every `prompt_agent` step, as a `file://` URL inside the object store root. The default prefix assumes an S3-style store. Missing: `ConfigError: This caller needs an S3 object store, but the configured store is LocalFilesystemObjectStore` at `prompt_agent`; fix the step and create a new task version.
   ```json
   "trajectory_output_prefix": "file:///<path-to-project>/.agentenv/object_store/prompt_agent_trajectories/<prompt-id>/"
   ```
5. A running Docker daemon for the `local` sandbox, and a `--project-id` label on `task create` (required; any string).

`task run` never tears down, so a fatal step leaves earlier deployments running; close them as described in [Reattach, look up a known instance, tear down](#reattach-look-up-a-known-instance-tear-down).

### At scale: bring your own durable runner

agent-env ships one runner, `LocalRunner`. Anything durable, queued or multi-host is a `Runner` subclass registered through the `[runner]` seam; the contract is in [Custom sandbox, state and runner providers](#custom-sandbox-state-and-runner-providers).

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

The context JSON is `context.to_safe_dict()`: model API keys and remote tokens are stripped. Top-level keys:

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

The agent trajectory is written to the object store and referenced from `prompt_responses[i].agent_trajectory_s3_uri`. The field names keep their legacy `s3` spelling, but the value is an object URL for whatever store is configured: `file://...` on the local store, `s3://...` on an S3 store. The object is whatever the agent's `urn:agentenv:trajectory/v1` extension returned for the prompt, written as JSON; the format is agent-defined. The bundled echo agent reports its native format, one `{"type": "echo", "input": ..., "output": ...}` entry per prompt. An agent that reports OpenTelemetry spans stores a list of spans, each with `name`, `context`, `kind`, `parent_id`, `start_time`, `end_time`, `status`, `attributes`, `events`, `links` and `resource`; the LLM judge reads them through the GenAI semantic conventions: it selects spans by `gen_ai.operation.name` (`chat` for model turns, `execute_tool` for tool calls, `chain` for the conversation root), takes the tool label from the span `name`, the arguments from `gen_ai.prompt` (`input`) and the result from `gen_ai.completion` (`output`). No span-name convention is required; the judge compacts such trajectories before grading, and a list with no `gen_ai.operation.name` attribute is passed through unchanged.

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

To compare agents or models, run the same eval again with `--agent-model <model>` or `--agent-artifact-id <artifact-id>` (and `--max-concurrency N`). Aggregation across runs is yours. `eval run` has no `--a2a-agent-id` and no sandbox overrides, and there is no `eval get` or `eval list`; read an eval through the explorer API. Like `task run`, it leaves sandboxes running; only remote providers reclaim them at TTL.

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

Local state (`document_store/`, `object_store/`, an auto-managed `.gitignore`) lives next to the discovered config file; with no file it is created under `./.agentenv/`.

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

`agent-env config show` answers the same question for the install you are running and names the module each fallback comes from. Prefer it over this table, which is a snapshot of one release's defaults; see [Inspect the resolved configuration](#inspect-the-resolved-configuration).

Precedence per seam, lowest to highest: built-in default, config table, `AGENT_ENV_*` variable, an explicit `configure(...)` or `set_*_store(...)` call in code. Environment variables accept only alias strings such as `local`; hosted backends need a table. An alias string selects a built-in with default coordinates. A table with `impl = "module.path:ClassName"` and an optional `config` sub-table selects any class:

```toml
[stores]
document = "local"   # SQLite      .agentenv/document_store/documents.db
object   = "local"   # filesystem  .agentenv/object_store/

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

document:       LocalSqliteDocumentStore  path=<project>/.agentenv/document_store/documents.db
                from [stores.document]
object:         LocalFilesystemObjectStore  root=<project>/.agentenv/object_store
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

Fourteen sections are reported. With no file the header reads `config: (none)` and every store `from built-in default`. When an environment variable beats the file, the section says so and names what it shadowed: with `AGENT_ENV_DOCUMENT_STORE=local` exported against a file whose `[stores.document]` names another class, the `document` block reads `from $AGENT_ENV_DOCUMENT_STORE` followed by `; <class> in [stores.document], shadowed`. That line is the point: one variable can downgrade a single store while every other section stays on the file, and nothing else in the system says so. Sections resolved through an alias report the class the name became; the rest report what the file contributes and, when absent, the module their fallback lives in. A section that fails validation is reported in place as `(unresolved)` with the `ConfigError` text, for example an empty `[agents] default_a2a_agent_id`.

`agent-env config debug` answers the other question, why that file: it prints every path discovery considers, in order, marking the winner with `->` and each with `(walk-up, exists: yes|no)`; with `AGENT_ENV_CONFIG` set, the single line reads `($AGENT_ENV_CONFIG, exists: yes)` and `config show` reports `via $AGENT_ENV_CONFIG`. Every command in the group takes `--json`; the JSON keeps the provenance (`winner`, `shadowed`, `impl`, `config`, `error` per section, plus `config_path` and `config_source`) rather than flattening to effective values. All are read-only: no store is constructed, nothing is fetched, and no `.agentenv/` directory is created. Secret values never appear: an `env:` or `secret:` reference prints as the reference, and a literal under a secret-shaped key (`api_key`, `token`, `password` and similar) prints as `***`, as does anything nested beneath one.

`agent-env config explain <path>` narrows `show` to one value. `<path>` is a section name or a TOML path — `document` and `stores.document` both work — and the output is that section's block on its own: the resolved value, `from <layer>`, and any `; ... shadowed` lines. A path *above* a section — `stores`, say — has no single winner, so it lists the sections under it and how each resolved rather than echoing a file table a higher layer may already have replaced. A path *below* a section is read out of whichever layer won that section, not out of the file — so with `AGENT_ENV_DOCUMENT_STORE=local` set, `config explain stores.document.config.database` reports what the local backend resolves to and names the shadowed file table, rather than echoing a database name the process never reads. Keys the file is not the only source for (`model.api_key`, `agents.default_a2a_agent_id` and the like) resolve through their own layers, so an environment override or a built-in default is reported as the winner. A path nothing supplies reads `(unset)`, and one only the file knows about says so.

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
   env      $AGENT_ENV_RDS_HOST -> [state.providers]  (selects the group)
   env      $AGENT_ENV_RDS_PORT -> [state.providers]
   env      $AGENT_ENV_RDS_DBNAME -> [state.providers]
   env      $AGENT_ENV_RDS_USERNAME -> [state.providers]
   env      $AGENT_ENV_RDS_PASSWORD -> [state.providers]
   env      $AGENT_ENV_RDS_SSLMODE -> [state.providers]
   env      $AGENT_ENV_RDS_AUTH -> [state.providers]
   env      $AGENT_ENV_RDS_REGION -> [state.providers]
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
| `[stores.document]` | `local` (SQLite) | `LocalSqliteDocumentStore`, `MongoDocumentStore` (table only) | `AGENT_ENV_DOCUMENT_STORE` |
| `[stores.object]` | `local` (filesystem) | `LocalFilesystemObjectStore`, `S3ObjectStore` (table only) | `AGENT_ENV_OBJECT_STORE` |
| `[stores.image]` | `local` (registry at `localhost:5000`) | `LocalRegistryImageStore`, `OciRegistryImageStore`, `EcrImageStore` | `AGENT_ENV_IMAGE_STORE` |
| `[stores.secret]` | `local` (process env vars) | `LocalSecretStore`, `AwsSecretsManagerSecretStore` | `AGENT_ENV_SECRET_STORE` |

The aliases `mongo`, `s3`, `ecr` and `aws` are recognized but have no built-in coordinates; using one raises `ConfigError` until you supply the table. The local image store starts a `registry:2` container named `agentenv-registry` on first push. Private registry credentials go in `[stores.image.config] credentials`.

`LocalSecretStore` reads process environment variables first (`use_env = true`), then an optional flat YAML or JSON file:

```toml
[stores.secret]
impl = "agent_env.store.secret_store:LocalSecretStore"
config = { file_path = "<path>/secrets.yaml", use_env = true }
```

`file_path` is resolved against the process working directory, not the config file, and must exist. `secret:` references cannot appear inside `[stores.secret]` itself. Store implementations are validated by the conformance suites in `tst/store/` (see [Conformance suites](#conformance-suites)).

### Compute, state and runner

| Seam | Purpose | Default | Notes |
|---|---|---|---|
| `[sandbox] default` / `agent_default` | where environments and agents run | `local` / `local` | built-ins `local`, `modal`, `modal_vm`, `e2b`; comma-separated names form a fallback chain |
| `[sandbox.providers.<name>]` | extra or configured providers | none | `"module:Class"` string or `{impl, config}` table; built-in names accept a `config` table only |
| `[sandbox.attribution]` | optional labels copied into every sandbox request | empty | free-form key/value; ignored by the local sandbox |
| `[state.providers.<name>]` | durable environment state stores | `local_postgres` | the table name must equal the class's `type`; `local_postgres` is the only built-in and accepts no override |
| `[runner]` / `[runner.config]` | who executes runs submitted through the explorer API | `local` (`LocalRunner`, `workers = 2`) | `AGENT_ENV_RUNNER=local` overrides the table; see [The local explorer and runner](#the-local-explorer-and-runner) |

A chain such as `default = "my_cloud_flaky,my_cloud"` tries each provider in order and raises `RuntimeError: All N providers failed` when every member fails. Unknown provider names raise `ValueError`. Putting `impl` on a built-in name (`[sandbox.providers.local]`) is a `ConfigError`. See [`docs/e2b-sandbox-provider.md`](docs/e2b-sandbox-provider.md) for the `e2b` provider's `config` keys.

### Model, human endpoint, explorer, registries

An example `[model]` table and the per-run precedence order are in [Model configuration](#model-configuration-owns-precedence).

| Table | Keys | Failure mode |
|---|---|---|
| `[model]` | `base_url`, `api_key`, `default`, `roles`, `params` | unknown key: `ConfigError: [model] has unknown keys [...]`; `params` may not set `model`, `messages`, `api_key`, `api_base`, `user`, `metadata`, `timeout`, `response_format` |
| `[conversations]` | `default_human_a2a_url` | `ConfigError` when a human-in-the-loop step needs it and nothing is set |
| `[agents]` | `default_a2a_agent_id` | the agent a `deploy_agent` step without `a2a_agent_id` (and a `rubrics_verifier` without `judge_a2a_agent_id`) deploys; precedence `configure(default_a2a_agent_id=...)`, then `[agents]`, then the built-in `a2a-default`; no env var; the value may be an `env:` / `secret:` reference; a blank value or any other key under `[agents]` is `ConfigError` (`[agents] has unknown keys [...]; allowed: ['default_a2a_agent_id']`) |
| `[explorer]` | `port` (8234), `cors_origins`, `allowed_hosts`, `static_dir` | host is always `127.0.0.1`; foreign `Host` headers get HTTP 421 unless listed |
| `[explorer.plugins]` | `impls` list | mounted before core routers; no per-plugin config |
| `[task_steps]`, `[artifacts]`, `[envs]` | `impls` list of `"module:Class"` | each class needs its own unique `type`; duplicates and inherited base types are `ConfigError` |
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
| `AGENT_ENV_FIXTURE_PREFIX` | prefix prepended to artifact object keys in a shared bucket |
| `GITHUB_TOKEN` | auth for `--dockerfile-github-url` builds |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `E2B_API_KEY` | provider credentials; `E2B_API_KEY` is read as `env:E2B_API_KEY` from `[sandbox.providers.e2b.config]` |
| `AGENT_ENV_MODAL_REGION` | region for the `modal` / `modal_vm` providers (default `us-east-1`) |

No variable relaxes TLS verification; every client verifies certificates. Variables injected into containers are separate: the environment server reads `MCP_HOST`, `MCP_PORT` and `ENVIRONMENT_NAME`; an A2A agent reads `A2A_HOST`, `A2A_PORT`, and `A2AAgent.deploy` (called by `deploy_agent`) injects `LITELLM_BASE_URL` and `LITELLM_API_KEY`.

## CLI reference

### Conventions

The root `agent-env --help` listing is shown under [Verify and platform notes](#verify-and-platform-notes).

- `-v/--verbose` is the only root option; installed plugins may add more.
- Verbs: `put` builds and registers a new version; `deploy` starts an instance; `get-instance` reads an instance record; `validate` checks without deploying.
- `put` never overwrites: repeating it on an existing id appends a version.
- References are `id` or `id:version`; `--version` and bare ids default to the latest version.
- `--platform` on every image-building `put` defaults to `linux/amd64`; see [Verify and platform notes](#verify-and-platform-notes) for Apple Silicon.
- `--sandbox`, `--env-sandbox` and `--agent-sandbox` accept a name or a comma-separated fallback chain. `--cua-sandbox` on `task run` / `task run-batch` takes a single backend, no chain; it is forwarded to `deploy_env` as `cua_sandbox_type` and only a custom `Env.deploy()` that accepts that keyword uses it. The built-in `mcp-server`, `website` and `multi` kinds log `Ignoring cua_sandbox=...` and proceed.
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
| `plugin` | installed plugins: what each contributes and whether it took effect | `list [--json] [--no-load]`, `show PACKAGE [--json] [--no-load]`, `check [--json]` |
| `task` | define and run tasks | `create`, `get`, `validate`, `run`, `run-batch`, `get-instance` |
| `up` | local stack: resolves backends, bootstraps `default-db` and `default`, serves the explorer API | `--no-bootstrap` |

Plugin groups appear in `--help` next to the built-ins; `agent-env plugin list` shows which package each comes from.

### Deprecated aliases and legacy paths

No renamed-command aliases remain. `artifact service`, `artifact service-universe`, `load-service-artifact`, `load-service-universe-artifact` and the `--service-artifact*` options answer `No such command` or `No such option`; the names are `artifact environment`, `artifact environment-universe`, `load-environment-artifact`, `load-environment-universe-artifact` and `--environment-artifact-id`. `--service-version` survives on `env mcp-server put` and `env website put` (the environment's seed-schema version, default `1`); no artifact or load command takes it. There is no `agent` command group either (the former `agent put-image` and `agent deploy` are gone): `a2a-agent put` is the only way to register an agent image, and `deploy_agent` resolves it by id through `[agents] default_a2a_agent_id`. Stored documents that still carry a legacy artifact `type` string load only if `[artifacts] type_aliases` maps it (see [Configuration reference](#configuration-reference)).

### Known gaps

| Missing | Workaround |
|---|---|
| `agent-env --version` | `python -c "import importlib.metadata as m; print(m.version('agentenv-framework'))"` |
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

A config table names an implementation with `impl = "module.path:ClassName"`. agent-env imports the class, checks that it subclasses the seam's base class, resolves `env:` and `secret:` references in the `config` table, and calls `Class.from_config(**config)` (default: `Class(**config)`). Primitives register under their own `type`. Misconfiguration raises `agent_env.config.ConfigError` naming the impl and the reason: cannot import, not a subclass, malformed pointer, missing `impl`, unresolved reference, duplicate `type`, or inherited base `type`. The error appears when the seam is first used, so a broken `[task_steps]` entry breaks every task load.

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

### Custom sandbox, state and runner providers

Sandbox: subclass `agent_env.providers.sandbox_provider.SandboxProvider`, implement `create_sandbox`, and register it under `[sandbox.providers.<name>]` or as an `agent_env.sandbox_providers` entry point named `<name>`. The `<name>` must equal the `.type` of the `Sandbox` objects it produces. That check runs on the first `create_*` call, terminates the mis-typed sandbox, and raises `SandboxProviderTypeError`, a `ConfigError`. This name-equals-type rule is what lets agent-env reconnect to an instance from its stored record.

State: subclass `agent_env.providers.state.env_state_provider.EnvStateProvider`, set `type`, implement `acquire`, `_teardown` and `deploy_state_context(ttl_seconds=, name_hint=)`. Register under `[state.providers.<name>]` or as an `agent_env.state_providers` entry point; here the name-equals-type check runs at registration. Once a non-local provider exists, `env init-env-state --env-state-type <name>` can pre-create a store out of band; `local_postgres` refuses that.

Runner: subclass `agent_env.runner.runner.Runner` (`submit`, `status`, `cancel` abstract) and set `[runner] impl` plus `[runner.config]`; the contract and what the runner serves are in [At scale: bring your own durable runner](#at-scale-bring-your-own-durable-runner). Only `LocalRunner` ships. None of these three seams has a conformance suite.

### Custom envs, task steps, artifacts

| Primitive | Base class | Contract | Registration |
|---|---|---|---|
| task step | `agent_env.task_step.task_step.TaskStep` | `type` ClassVar, `to_dict`/`from_dict`, `async execute(context)`, optional `preflight()` | `[task_steps] impls`, or an `agent_env.task_steps` entry point |
| artifact | `agent_env.artifact.artifact.Artifact` (pydantic) | `type` field default is the registry key | `[artifacts] impls`, or an `agent_env.artifacts` entry point |
| environment | `agent_env.env.env.Env` (subclass it directly; there are no generic parameters) | `type` ClassVar, `from_dict`, `async deploy(**kwargs) -> DeployedEnv` (both raise `NotImplementedError` on the base); `to_dict` is inherited | `[envs] impls`, or an `agent_env.envs` entry point |

```toml
[task_steps]
impls = ["mycorp_demo.steps:GradeEssayTaskStep"]
```

A custom step's `preflight()` participates in `task create` and `task validate`. Verifiers are ordinary steps that write `context.metadata["verifications"][verifier_id] = {"score", "results"}`. A task made only of custom steps that need no sandbox runs in-process with no Docker, model or remote backend. Custom artifacts and environments round-trip through `put`, `get` and `query().type(...)`, and `Artifact.get` / `Env.get` return the registered subclass. There are no CLI subcommands for custom artifact or environment types; those round trips are Python-only. A custom environment's full contract is `type`, `from_dict` and `deploy`:

```python
from agent_env.env.env import DeployedEnv, Env

class MyCustomEnv(Env):
    type = "my_custom_env"                       # the portable identity stored in the document store

    @classmethod
    def from_dict(cls, data: dict) -> "MyCustomEnv":
        ...                                      # rebuild from the stored document

    async def deploy(self, **kwargs) -> DeployedEnv:
        ...                                      # provision and return a DeployedEnv
```

An environment can also be referenced without ever being deployed. When a `deploy_agent` step lists its id in `env_ids` and no `deploy_env` step deployed it, the step reads an http(s) `mcp_url` attribute from the stored environment document and registers that live endpoint with the agent (a bearer token comes from `task run --remote-token <env-id>=<token>`), so such a class may leave `deploy()` unimplemented. This is how an MCP server that already runs elsewhere joins a task. Any other undeployed id fails with `Env '<id>' is not in context.deployed_envs; add a DeployEnvTaskStep for it, or reference an env that exposes a live http(s) 'mcp_url'`.

### Register types from an installed package

An installed distribution registers envs, task steps, artifacts, sandbox and state providers and explorer plugins by declaring entry points, with no config. The group says what kind of thing it is, the entry-point name is the registry key, and the value is the class:

```toml
[project.entry-points."agent_env.envs"]
browser = "agentenv_browser.env:BrowserEnv"

[project.entry-points."agent_env.task_steps"]
browser_navigate = "agentenv_browser.steps:NavigateTaskStep"
```

| Group | Value | Name |
|---|---|---|
| `agent_env.envs` | `Env` subclass with its own `type` | must equal the class's `type`, the spelling its documents are written under |
| `agent_env.task_steps` | `TaskStep` subclass with its own `type` | must equal the class's `type` |
| `agent_env.artifacts` | `Artifact` subclass with its own `type` field default | the registry key; a class may also register under extra names, such as a legacy spelling, as long as its own `type` default resolves to it and its `type` field accepts each extra name |
| `agent_env.sandbox_providers` | `SandboxProvider` subclass | the registry key; it must equal the `.type` of the sandboxes the provider produces, checked on the first `create_*` call |
| `agent_env.state_providers` | `EnvStateProvider` subclass | must equal the class's `type`, checked at registration |
| `agent_env.explorer_plugins` | `ExplorerPlugin` subclass with its own `type` | must equal the class's `type` |

- Each registry takes the built-ins first, then plugins, then config. A plugin cannot replace a built-in: it is skipped with a warning, however many distributions claim that name.
- Two installed distributions registering any other name in one group raise `agent_env.plugins.PluginConflictError`, a `ConfigError` naming both distributions and their versions. Uninstall one of them.
- A plugin that fails to import, fails the checks above, or (for explorer plugins) fails to construct is skipped with a warning and the others still load. Resolving its name then reports the recorded error, for example `Unknown env type: browser (registered by 'browser' from agentenv-web 2.1.0 (…) but failed to load: ModuleNotFoundError(…))`.
- Config replaces a plugin's class with a warning: an `impls` entry of the same `type`, or a `[sandbox.providers.<name>]` / `[state.providers.<name>]` table carrying `impl`. Naming the plugin's own class is silent. A provider table without `impl` configures the plugin's provider the way it configures a built-in's, which is where per-deployment settings such as a secret name belong. A config-only provider table, or an `[artifacts] type_aliases` entry, that points at a plugin which failed to load is skipped with a warning rather than failing the whole registry.
- `agent_env.plugins.load_failures()` lists every plugin that did not take effect in the registries the current `Config` has built (failed to load, validate or construct, or clashed with a built-in), by group and name, with the reason. The record belongs to the `Config`, so `reset_config()` starts a new one. A deployment that ships its plugins can build its registries at startup and assert it is empty.
- `agent_env.plugins.inventory()` lists every installed distribution that declares entry points in these groups, with a status for each contribution: `active`; `replaced` (config names a different class, and `replaced_in` says where); `failed`; `skipped` (a built-in owns the name); `conflict`; `blocked` (the group's real build fails, for example because two distributions claim a name in it, so nothing in the group loads); or `unloaded`. It builds the registries on a throwaway `Config` that reads the same document, so the `Config` in use keeps its registries and its `load_failures()`, and it reports a conflict instead of raising it. `inventory(load=False)` reads installed metadata only and runs no plugin code, so a status that needs a build is `unloaded`. With the default `load=True` the plugins are imported and the explorer plugins constructed, with whatever process-wide effects that has. `discovery_errors` names each group whose installed entry points could not be read at all, so none of its plugins is listed.
- Discovery reads installed metadata and imports nothing. When entry points are loaded is unspecified: today each group is imported when its registry is first built, but a plugin must not depend on that.
- Loading a plugin never changes which config a `Config` reads: the `Config` resolves its document before it imports any plugin. A plugin package that sets `AGENT_ENV_CONFIG` on import affects only configs built afterwards, such as after `reset_config()`.

The distribution that declares the entry points is the plugin, so `pip uninstall` removes it completely. CLI commands are a separate contribution, described next.

### CLI plugins, root options, explorer routes

Installed packages add commands through two entry-point groups:

```toml
[project.entry-points."agent_env.cli_plugins"]
my-tools = "mycorp_demo.cli:my_tools"
[project.entry-points."agent_env.cli_root_options"]
tenant = "mycorp_demo.cli:tenant_option"
```

`agent_env.cli_plugins` entries are click commands or groups added next to the built-ins. `agent_env.cli_root_options` entries are optional `click.Option` instances with `expose_value=False`; their callback runs before the subcommand, so it can set `AGENT_ENV_CONFIG` and call `agent_env.config.reset_config()` to select the config for the whole process. Plugins load when `agent_env.cli` is imported, after the built-ins. Clash rules: a plugin command or flag that core already owns is skipped with a warning on stderr, and core wins. Two plugins claiming the same root flag abort the CLI with `RootOptionConflictError`. Nothing grafts onto existing groups.

Explorer routes: subclass `agent_env.explorer.plugin.ExplorerPlugin`, set `type`, and return a FastAPI `APIRouter` from the `router` property. List it under `[explorer.plugins] impls` or declare an `agent_env.explorer_plugins` entry point; `from_config()` takes no arguments. `agent-env up --no-bootstrap` mounts it before the core routers and needs no Docker.

### Manage plugins

`agent-env plugin` shows what the installed plugins contribute and whether each piece took effect.

```bash
agent-env plugin list            # every plugin package, what it provides, and its status
agent-env plugin show PACKAGE    # each contribution and why it is in that state
agent-env plugin check           # exit 1 if any contribution did not take effect
```

```
agent-env 0.9.1193 · uv tool at ~/.local/share/uv/tools/agentenv-framework
config: (none)

PACKAGE           VERSION  PROVIDES                                 STATUS
agentenv-browser  1.0.0    env browser, task step browser_navigate  ok
agentenv-grader   0.3.1    task step grade_essay                    1 failed
```

| Status | Meaning |
|---|---|
| `active` | registered and in use |
| `replaced` | config names a different class for the name; `show` says where. Not a failure |
| `failed` | failed to import, validate or construct |
| `skipped` | a built-in, a core command or root option, or (for CLI commands) a plugin loaded first owns the name |
| `conflict` | another installed package registers the same name |
| `blocked` | the group cannot load at all, for example because of a conflict in it, so this contribution does not either |
| `unloaded` | not loaded (`--no-load`). When loading, it means the status could not be determined, and `check` fails on it |

- The header says how agent-env is installed (uv tool, pipx, uv project, virtualenv or system Python) and where: a plugin has to be installed into that same environment. It warns when the retired `agent-env` distribution is installed next to `agentenv-framework`, since both write the same package.
- `list` and `show` take `--json` and `--no-load`. `--no-load` reads installed metadata only and imports no type plugin; CLI plugins are already loaded, because the CLI loads them when it starts. Without it, `list`, `show` and `check` import every type plugin and construct the explorer plugins, so that plugin code runs.
- `show` also reports whether importing the package sets `AGENT_ENV_CONFIG`, checked in a fresh interpreter.
- `check` builds every registry the way a process does, adds the CLI's own plugins, and fails on `failed`, `skipped`, `conflict`, `blocked` or `unloaded`, or when the config file or the installed entry points cannot be read (a malformed `entry_points.txt` hides every plugin, so it fails rather than passing empty). `check --json` prints the problems as JSON. A `replaced` contribution passes: config chose it.
- `plugin` is a core command: a CLI plugin that names a command `plugin` is skipped, like any clash with a core command.
- The Python equivalent is `agent_env.plugins.inventory()` (see [Register types from an installed package](#register-types-from-an-installed-package)).

### Building a platform plugin

One installable package can combine all of the above: bundled config files, a root option that selects one per invocation, store and provider classes, a `Runner`, custom steps, and explorer routers. The `--tenant` demo above is that pattern in miniature. Its callback points `AGENT_ENV_CONFIG` at a bundled `tenants/<name>.toml`; a name that does not exist fails loud with `ConfigError` on first use, while `--help` still works. A hosted control plane serves the explorer app from its own server process and lists its public hostnames under `[explorer] allowed_hosts` (see the comments in `.agentenv/config.example.toml`). Durable runners and hosted stores are the plugin's responsibility; this repository ships local implementations only.

## Contribute, release, license

### Development setup and test tiers

[CONTRIBUTING.md](CONTRIBUTING.md) is the contributor guide: open an issue before anything larger than a bug fix, one logical change per pull request with tests, pull request titles in the form `type(scope): summary`, an approving review from a code owner (`CODEOWNERS`) and green CI. [AGENTS.md](AGENTS.md) is the repository map and conventions file for contributors and coding agents; `CLAUDE.md` imports it.

Set up with `uv sync --extra dev` or `make install` (both in [Install the packages](#install-the-packages)). Tests are tiered by path:

| Command | Runs | Needs |
|---|---|---|
| `make unit-test`, or `python -m pytest tst/unit packages/agentenv-protocol/tests -n auto -q` | 3,086 unit tests (3,084 pass, 2 skip) in about 20 seconds; IP sockets are blocked by `pytest-socket` and AWS calls go to `moto` | nothing external |
| `make int-test-fast` | the 129 integration tests not marked `int_test_slow`, in parallel | Docker and a local OCI registry on `:5000` |
| `make int-test-slow` | the 285 `int_test_slow` tests (12 of the 27 integration modules; real image builds and sandboxes), serially | Docker and the registry; a model endpoint, a remote sandbox or a registered default agent for some |

Run the unit tier from the checkout root with `AGENT_ENV_CONFIG` unset; the Makefile targets hardcode `.venv/bin/python`. The integration tiers were not run for this guide. A test may skip only for a declared capability gap, with the reason `agentenv-capability-missing: <name>` where `<name>` is `model_endpoint_configured`, `remote_sandbox`, `default_a2a_agent` or `mcp_server_sources` (see `tst/util/capabilities.py`); CI rejects any other skip reason.

CI is GitHub Actions. `.github/workflows/local-backends.yml` runs the `unit`, `integration-local` and `integration-local-slow` jobs on Python 3.12, installed from public PyPI with no secrets and a `registry:2` service container for the integration jobs. The `unit` job also fails if `uv.lock` resolves anything from a registry other than PyPI, if `agentenv-protocol` is not the editable workspace member, or if `uv.lock` is out of date, before or after a trial run of the release bump (`scripts/bump_version.py`). The public jobs skip `tst/integration/env/gateway/gateway_test.py`, which needs an x86 Chromium build and, for its virtual-clock tests, MCP server sources named by `AGENT_ENV_TEST_MCP_SERVERS_DIR`; `make int-test-slow` runs it, so run that locally when a change touches the gateway and say so in the pull request. Dependabot (`.github/dependabot.yml`) opens weekly updates for the uv lock and the pinned actions.

### Conformance suites

`tst/store/` holds four backend-neutral suites: `conformance.py` (`DocumentStore`, 27 cases), `object_conformance.py` (`ObjectStore`, 14), `secret_conformance.py` (`SecretStore`, 3; seed the backend with `FIXTURE` first) and `image_conformance.py` (`ImageStore`, 4; push and pull need Docker). Each exposes a `CASES` list; a backend test builds its store fixture and parametrizes over `CASES`, exactly as `tst/unit/store/sqlite_document_store_test.py` does. Every built-in backend, local and hosted, is tested against the same cases.

The suites are not packaged in the wheel. Run them from a source checkout with `PYTHONPATH=.`:

```
cd <agent-env checkout> && PYTHONPATH=. python -m pytest -p no:cacheprovider -q <path>/tests/test_store_conformance.py
```

`pytest tst/store` alone collects nothing; the cases only run through a consumer test. Protocol package tests live in `packages/agentenv-protocol/tests/` and are part of the unit tier.

### Documentation map

| Document | Covers |
|---|---|
| [`packages/agentenv-protocol/README.md`](packages/agentenv-protocol/README.md) | wire contract, environment server SDK, A2A agent framework |
| [`docs/e2b-sandbox-provider.md`](docs/e2b-sandbox-provider.md) | `[sandbox]` configuration for the `e2b` provider, templates, networking |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | development setup, test tiers, CI jobs, pull request rules |
| [`AGENTS.md`](AGENTS.md) | repository map, configuration and extension-point summary, conventions for contributors and coding agents |
| [`SECURITY.md`](SECURITY.md) | private vulnerability reporting and supported versions |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Contributor Covenant |
| [`.agentenv/config.example.toml`](.agentenv/config.example.toml) | the all-local configuration to copy |
| [`.env.example`](.env.example) | a commented reference of the `AGENT_ENV_*` variables; agent-env never loads this file, export what you need yourself. Its `AGENT_ENV_ENVIRONMENT` line is read only by an installed plugin, never by agent-env |

Full configuration, CLI, step and extension references have not been split out of this README yet.

### Versioning and compatibility

The `agentenv-framework` distribution and `agentenv-protocol` are versioned separately (`0.9.x` and `0.1.x` today), both in their `pyproject.toml`; agent-env releases carry a `vX.Y.Z` tag, and agentenv-protocol is bumped in the same commit and has no separate tag today. There is no `agent_env.__version__` attribute and no `--version` flag (workaround in [Known gaps](#known-gaps)). Protocol extensions carry their version in the URI (`urn:agentenv:clock/v1`, `urn:agentenv:agent-config/v1`, `urn:agentenv:trajectory/v1`). Renamed CLI commands are removed outright; no deprecated aliases exist at this version (see [Deprecated aliases and legacy paths](#deprecated-aliases-and-legacy-paths)). A written compatibility and deprecation policy does not exist yet.

### Releases

A release is a version bump in both `pyproject.toml` files plus a `vX.Y.Z` tag. Maintainers cut releases: the bump is automated when a labelled pull request merges, so contributors do not edit `version` or push tags (see the Releases section of [CONTRIBUTING.md](CONTRIBUTING.md)). Neither package is published to a public index yet, and there is no `CHANGELOG.md`.

### Support, security, license

Report bugs and gaps as issues against this repository, with the installed `agentenv-framework` version and the sandbox backend in use. Report vulnerabilities privately through the contact in [SECURITY.md](SECURITY.md), not in public issues; only the latest release is supported, so reproduce against it first. Contributors follow [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md); every pull request needs a code-owner review. agent-env and agentenv-protocol are licensed under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
