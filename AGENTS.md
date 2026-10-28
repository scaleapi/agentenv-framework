# AGENTS.md

Guidance for coding agents and contributors working in this repository.

## What agent-env is

agent-env is a Python SDK for building environments that AI agents act in, and for running and
evaluating agents against them. It deploys environments (MCP servers, databases, websites, gateways)
into sandboxes, keeps immutable versioned artifacts, runs tasks as DAGs of steps (deploy an env,
deploy an agent, prompt it, verify the outcome) and records every run. Persistence, sandboxes, the
model endpoint and every extension are selected in one config file; a bare install runs fully local.
The in-repo `packages/agentenv-protocol` is the open data-plane protocol and server SDK for
environments; agent-env depends on it.

## Setup

Python 3.11.4 or newer. The repo convention is a `.venv` at the root, which the Makefile and CI use.

```bash
uv sync --extra dev && . .venv/bin/activate
```

Without uv:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ./packages/agentenv-protocol -e '.[dev]'
```

Both install the in-repo protocol package together with the dev extra: agent-env depends on it and
the workspace copy is the one to develop against (`uv sync` does this through the uv workspace in
`pyproject.toml`). `make install` is the pip route in one step. No cloud credentials are needed; the
default stores are local. The cloud store backends need their extra, both in `dev`: `aws` (boto3) for
S3, Secrets Manager, DynamoDB and ECR, `gcp` for the Google Cloud ones. Core must import without boto3: a
store package re-exports an `aws` backend on first use, and code outside the backends imports boto3
or one of them only where it is used.

## Commands

| Command | What it does |
|---|---|
| `make lint` | `ruff check` over the repository, with the rules in `pyproject.toml` (pyflakes) |
| `make unit-test` | `tst/unit` and `packages/agentenv-protocol/tests`, in parallel, no network |
| `make int-test-fast` | `tst/integration` minus `int_test_slow`, in parallel; needs Docker and a local OCI registry (`docker run -d -p 5000:5000 public.ecr.aws/docker/library/registry:2`) |
| `make int-test-slow` | the `int_test_slow` tests, serially (they build the same Docker tags from module fixtures and race under xdist) |
| `make installer-test` | `tst/installer`: `plugin add` / `remove` through the real pip, uv and pipx, offline against wheels built from the checkout, and the container user journey (`python:3.12-slim`, `--network none`) with the PEP 668 refusal; needs uv, pipx and Docker |
| `make clean-install-test` | the `clean-install` CI gate: both distributions built as the release builds them, installed into a fresh venv from public PyPI with an allowlisted environment, and `agent-env run hello` run twice by name and checked through the local store; needs Python 3.11, uv and git |
| `npm ci && npm run typecheck && npm run lint && npm run test:smoke` | in `src/agent_env/explorer/ui`: the explorer UI checks the `ui` job runs (it also runs `npm run build:static`); needs Node 22 |
| `pytest tst/unit/env/store_test.py -v` | one file; add `--log-cli-level=DEBUG` for debug logs |
| `uv build` | wheel `agentenv_framework-<version>-py3-none-any.whl` (`src/agent_env` only, without the UI source) and sdist `agentenv_framework-<version>.tar.gz` (`src`, `tst`, the protocol package, `.github/scripts`) |
| `agent-env`, `python -m agent_env.cli` | the CLI; `agent-env config show` prints which config file is in effect and where each section came from |

## Configuration

- One file: `.agentenv/config.toml`, found by walking up from the working directory, or the file
  `AGENT_ENV_CONFIG` names (it must exist). Exactly one file is read, whole; a section that is absent
  falls back to the code default, never to another file. `.agentenv/config.example.toml` is the
  committed template; `.agentenv/config.toml` is git-ignored.
- Precedence per section: built-in default < config file < `AGENT_ENV_*` environment variable
  (`AGENT_ENV_DOCUMENT_STORE`, `AGENT_ENV_OBJECT_STORE`, `AGENT_ENV_IMAGE_STORE`,
  `AGENT_ENV_SECRET_STORE`, `AGENT_ENV_RUNNER`) < `configure(...)` in code.
- Defaults are local and carry no external coordinates: SQLite document store, filesystem object
  store, an OCI registry at `localhost:5000`, env-var secret store, `LocalRunner`, `local` Docker
  sandboxes for envs and agents. MongoDB, DynamoDB, Firestore, S3, Cloud Storage, ECR, AWS Secrets
  Manager, Google Cloud Secret Manager, and the Modal, E2B, Sail, SmolVM, Tensorlake, Tenki, Vercel, boxd and
  Daytona sandboxes exist as implementations and are selected by config (the last seven through their extras:
  `sail`, `smol`, `tensorlake`, `tenki`, `vercel`, `boxd`, `daytona`; every extra but `smol` is in `dev`).
- The model endpoint is unset until `[model] base_url` / `api_key` (or `LITELLM_BASE_URL` /
  `LITELLM_API_KEY`) is configured. A loopback URL handed into a `local` container (the model
  endpoint, an agent, peer or gateway on this machine) becomes `host.docker.internal`. On Linux that
  name is the Docker bridge address (`docker0`, usually `172.17.0.1`), not loopback, so a service on
  this machine that local containers call, a model proxy say, has to listen there: bind it to the
  bridge address and use that address in its URL. Binding `0.0.0.0` also works, but exposes the
  service to the network.
- `[model.headers]` go with every call to the configured endpoint and nowhere else: `extra_headers` on
  agent-env's own calls, `LITELLM_EXTRA_HEADERS` (JSON, written by core, never read) for agents and the
  unit-test verifier. On `sail_vm` with `inject_model_key`, Sail sets them at egress like the key.
- A host-mode `install_agent` (no `container_name`) on a `local` sandbox runs the agent's
  `install_commands_host` on this machine, with your privileges. Every command a local sandbox runs
  carries `AGENTENV_SANDBOX=<sandbox id>`, and teardown stops the processes that carry it, then removes
  what steps staged in `/tmp` under that id. Those processes are found through `/proc`, so on macOS
  they keep running; files the install wrote elsewhere (`/opt`, packages) stay. Such an agent fetches the
  local store's grants only when `grant_advertise_host` names an address this machine reaches, the
  bridge address say: `host.docker.internal`, which grant URLs name by default, resolves only in containers.
- `[agents] default_a2a_agent_id`: the agent a `deploy_agent` step without an id deploys; built-in
  `@local/agentenv-framework/agents/claude-code`, the `agents` bundle's Claude Code
  (`src/agent_env/agents/`), overridable by `configure(default_a2a_agent_id=...)`. Any step's
  agent field left unset falls back to it (`entity.refs.default_agent_refs`), so a bundle run plans the
  default for it, pulling the built-in one from the installed `agents` bundle. Claude Code needs a model
  endpoint serving Anthropic's Messages API, such as a LiteLLM gateway. It declares `install/v1` too: its
  folder's `install.sh` installs it into a task container or onto a VM sandbox's host, with its own Claude
  Code, uv and Python, adding bash only to a target with neither bash nor zsh.
- Secrets never live in the file: use `secret:KEY` or `env:NAME` references, resolved through
  `[stores.secret]`.
- agent-env has no stage concept: "dev" versus "prod" is which file `AGENT_ENV_CONFIG` names. A
  platform plugin may pick that file; core must never read a stage variable, take a stage argument or
  carry a stage attribute (`tst/unit/config/test_no_stage_concept.py` enforces this).
- Re-pointing `AGENT_ENV_CONFIG` after the first store or registry is built is unsupported: set it
  before anything resolves, or call `reset_config()`, which drops the singleton and every registry.

## Architecture

Source lives in `src/agent_env/`. Every kind of object is identified on the wire by a `type` string
and deserialized through a registry.

| Package | Contents |
|---|---|
| `env/` | `Env` base class: a subclass declares `type`, implements `from_dict` and `async deploy(...) -> DeployedEnv`, and may override `reset` and `load_file_artifact_universe`. Built-ins in `env/envs/`: `MCPServerEnv` (`mcp_server`), `MultiEnv` (`multi`, MCP servers plus websites), `ServiceDBEnv` (`service_db`, PostgreSQL), `WebsiteEnv` (`website`), `GatewayEnv` (`gateway_server`), `ComputerDriverEnv` (`computer_driver`, a computer driver's files and what its class declares for each platform it supports, read at put in a process of its own, `agentenv_protocol.computer._declare`, where the driver's own dependencies needn't be installed; its deploy is refused until agent-env creates computers). `env/gateway/gateway.py` is the env-side gateway server, container-only code. While its clock is armed it serves agents `get_time` (`env/gateway/get_time.py`), which answers in UTC and, given an IANA `time_zone`, also in that zone; executors' working `hours` (`env/gateway/triggers.py`) take a fixed `utc_offset` or an IANA `time_zone`. |
| `artifact/` | Immutable versioned artifacts; every `put()` creates a new version. The document goes to the document store, binary payloads to the object store. Every artifact but an environment universe, whose `metadata` key holds its legacy metadata file ids, carries a `metadata` dict outside its versioned fields, changed by compare-and-swap (`merge_metadata`). Built-ins: `CliArtifact` (`cli`), `ComputerArtifact` (`computer`, a base computer: its architectures, OS, session user, settings and container form, whose image `put` reads from its registry into `metadata.validated_images`), `FileArtifact` (`file`), `FileArtifactUniverse` (`file_artifact_universe`), `DockerImageArtifact` (`docker_image`), `EnvironmentArtifact` (`environment`), `EnvironmentUniverseArtifact` (`environment_universe`), `SkillArtifact` (`skill`); `[artifacts] type_aliases` maps another stored spelling of a type onto one of them. A `DockerImageArtifact` holds a tar.gz of its image, or, written by `put_ref` (`--image-ref` on `env mcp-server put` and `a2a-agent put`), only a registry reference with its tag pinned to a digest, which sandboxes pull, or a published image's tag with its `image_digest` and its build context, which sandboxes pull by the digest and build from the context when they can't (the infra images, and a bundle agent built from a shipped agent's sources, `Plan.published`), or, written by `put_context`, only its build context, which a VM sandbox builds before running it (`VmSandbox.load_docker_images`) and Modal builds before creating its container (`ModalSandboxProvider.prepare_image`, outside a fallback chain's deadline); anything else that runs images by name (a container on the local provider, a lone server on any provider but Modal) refuses such an image. A build context (`utils/docker/context.py`) is what `docker build` would send, `.dockerignore` applied, written so equal files give equal bytes and an equal `source_digest`. |
| `task/`, `task_step/` | A `Task` holds its `TaskStep`s inline; `Task.run()` executes them as a DAG. `depends_on` (None means all prior steps) gates a step, independent steps run concurrently, `fail_task_on_error` makes a failure fatal or tolerated, `retry_config` rolls a failed span back through the step journal and re-dispatches it. Built-in steps live in `task_step/task_steps/` (`deploy_env`, `deploy_agent`, `prompt_agent`, the verifiers under `verifiers/`, `start_computer_recording` and `end_computer_recording`, whose open recordings `teardown_run` and `teardown_sandboxes` end first, and more). `task/episode/` runs a task with some of its agents played from outside, as a training loop does (`EpisodeSpec`). |
| `store/` | Four store ABCs with local and cloud implementations: `DocumentStore` (SQLite, MongoDB, DynamoDB, Firestore with MongoDB compatibility), `ObjectStore` (filesystem, S3, Cloud Storage), `ImageStore` (local OCI registry, ECR), `SecretStore` (env vars or file, AWS Secrets Manager, Google Cloud Secret Manager). `VersionedEntityStore` implements the shared versioned get/put logic, `QueryBuilder` is the immutable chained query API, `store/base.py` holds the error types. `store/routing.py` is the `@local` namespace: while the CLI or a `build_app` server runs, an `@local/…` id and what an `@local` task's run records go to local stores beside the configured ones, and that run's writes to the configured stores are refused. A new backend must pass the conformance kits in `tst/store/`. |
| `config/` | The `Config` singleton (`get_config`, `configure`, `reset_config`) in `config/runtime.py`, file discovery in `config/loader.py`, and `load_impl`, which resolves `module:Class` pointers. `agent_env.store` re-exports the config names for compatibility. |
| `providers/` | `providers/sandbox_providers/` holds the sandbox providers `local`, `modal`, `modal_vm`, `e2b`, `sail_vm` (the `sail` extra), `smol_vm` (the `smol` extra), `tensorlake` (the `tensorlake` extra), `tenki` (the `tenki` extra), `vercel` (the `vercel` extra), `boxd` (the `boxd` extra), `daytona` (the `daytona` extra); `[sandbox] default` and `agent_default` accept a comma-separated fallback chain, whose deploys skip a provider that can't run them. A sandbox provider declares what it runs (`ON_THIS_MACHINE`, and for VMs on this machine the address their requests to it arrive at, `REACHES_HOST_AT`; `CONTAINER_ACCEPTS`, `PRIVATE_NETWORK`; `SUPPORTED_SANDBOX_MODES`, derived from whether it implements `create_sandbox` and `create_vm`; and `SANDBOX_ACCEPTS`, the images an agent's sandbox runs, which defaults to those a VM loads when it implements `create_vm` and to `CONTAINER_ACCEPTS` otherwise, so only a provider that implements `create_vm` yet gives an agent a container, like the local one, declares it), each sandbox carries the `platform` of the images it runs (linux/amd64 unless its provider sets another when it creates it; None, any, unchecked, for a local sandbox), and code outside `providers/sandbox_providers/` reads those, never which provider it has. `providers/env_providers/` holds the environment providers: `EnvironmentProvider` (an env's containers and state store), `EnvironmentGatewayProvider` (`gateway`, an env's default), which renders a docker-compose for the gateway and its MCP servers inside the sandbox, and `EnvironmentServerProvider` (`server`), one MCP server in its own container with no gateway in front and no state store; `providers/env_state/` holds env-state providers (`local_postgres` built in). |
| `a2a_agent/` | The `A2AAgent` entity (`a2a_agent`), its stores and the validator steps. The protocol package provides the agent-side framework. |
| `openenv/` | `build_app(bundle)`: the [OpenEnv](https://github.com/openenv/OpenEnv) server for a bundle's tasks, each session playing one episode through the Episode API, with the envs' MCP tools, an added `end_turn` tool and the task's verifier score as the reward. It prepares bundles as `agent-env run` does (`prepare_bundle(played=True)`, without the agent the client plays) and keeps namespace routing on for the app's life. `openenv` comes with the `openenv` extra, a floor with no cap, and this package is the only one that imports it. |
| `runner/` | The `[runner]` seam: `Runner.submit()` returns a `RunHandle` (`run_id`, `instance_id`) before the run starts; `LocalRunner` is built in. `runner/store.py` is the run index every runner writes and the explorer reads. |
| `bundle/` | A bundle is a folder of entities (`envs/`, `agents/`, `artifacts/`, `skills/`, `tasks/<name>.json`, `evals/<name>.toml`) that runs without a registry: `agent-env run` plans it, writes it into the `@local` namespace (reusing each version whose inputs haven't changed) and runs its evals or tasks; `agent-env bundle init` writes a stored task or eval as one, with every env, agent and artifact it uses in its own folder (each image as its build context, pruned to what its Dockerfile reads and pinned to the Python and npm versions of the stored image) so it runs with no store. |
| `eval/` | `Eval` (`eval`), a versioned list of task references, and its store. |
| `plugins/` | Entry-point plugins (see Extension points): the group names, `inventory()`, which `agent-env plugin list/show/check` report, `load_failures` and `settings`. The top level is public; the underscored modules are internal. |
| `explorer/` | Optional local web UI: `agent-env up`, binds loopback `:8234`. |
| `cli/` | Click CLI with the groups `a2a-agent`, `artifact`, `bundle`, `config`, `env`, `eval`, `plugin`, `run`, `task`, `up`. `task run`, `task run-batch` and `eval run` hand their runs to the configured `[runner]` and follow them. |
| `agents/` | The `agents` bundle: the agents agent-env ships, one folder each under `agents/agents/`, with a smoke task each under `agents/tasks/`. Its Claude Code, `@local/agentenv-framework/agents/claude-code`, is the default agent. An agent's Python packages are its folder's `requirements.txt`, installed in its image, never in agent-env's own environment. A run that pulls one of its agents reads the whole bundle, so `tst/unit/bundle/shipped_bundles_test.py` parses and checks every shipped bundle. |
| `computers/` | The `computers` bundle: the base computers agent-env ships as `computer` artifacts, `ubuntu-24.04` (XFCE on Xvfb, supervisord as the session user, settings as `AGENTENV_SETTING_<NAME>`), aliased through `[artifacts]`. It has no tasks either. |
| `drivers/` | The `drivers` bundle: the computer drivers agent-env ships as `computer_driver` envs, the reference driver (X11 on Linux, recording through `x11grab`) aliased `reference`. It has no tasks: a bundle with neither tasks nor evals is refused unless its `aliases.toml` names something for other bundles to use. `tst/integration/computer/` serves it on the `ubuntu-24.04` base in a scaffold image and runs the conformance probe and an agent against it. |
| `examples/` | The example bundles agent-env ships (`hello`, and `echo-mcp`, which needs Docker). Tests run them, but they are examples for users, not fixtures. Every shipped bundle is registered under `agent_env.bundles` in `pyproject.toml`, `<name> = "<package>"` shipping `src/<package as a path>/<name>/`. |

## Extension points

Out-of-tree code plugs in either through `config.toml`, as a `module:Class` pointer constructed with
`from_config(**config)`, or as an entry point of an installed package:

- `[stores.document|object|image|secret]`, `[runner]`, `[sandbox.providers.<name>]`,
  `[state.providers.<name>]`, `[explorer.plugins]`: one implementation each.
- `[envs]`, `[artifacts]`, `[task_steps]` with `impls = ["module:Class", ...]`: extra classes for the
  registries. The identity is the class's `type` (a `ClassVar` on envs and steps, the Pydantic `type`
  default on artifacts); a class that keeps the base default, collides with a registered type, or
  leaves a method its base requires unimplemented fails at registry build with `ConfigError`.
- For sandbox and state providers the table name must equal the `type` the provider produces,
  because that string is persisted and used to reconnect.
- Entry points register the same kinds from an installed package, with the entry-point name as the
  registry key: `agent_env.envs`, `agent_env.artifacts`, `agent_env.task_steps`,
  `agent_env.sandbox_providers`, `agent_env.state_providers` and `agent_env.explorer_plugins`. A
  plugin whose name clashes with a built-in is skipped, a name two plugins claim is left out and
  config cannot take it, and config that registers a plugin's name takes it over.
  `agent-env plugin check` reports each plugin that did not take effect.
- Bundles install through `agent_env.bundles` entry points (`<name> = "<package>"`, the folder at
  `<package>/<name>/`), with ids rooted at `@local/<dist>/<name>`. A shared name is not left out:
  a bundle runs by its bare name unless another distribution installs that name too, and then by
  `<dist>/<name>`, except that agent-env's own bundles keep their bare names. Another bundle uses an
  installed bundle's env, agent or artifact by its id, or as the configured default agent: the planner
  pulls it into the run as one of the bundle's own (`bundle/plan.py`), and its ledger row is keyed by the
  installed bundle, so every bundle shares one write of it. Such an id takes no version pin.
- A bundle can give its agents aliases in an `aliases.toml` at its root (`[agents] <folder> = ["<alias>", ...]`, read
  by `plugins/_bundles.py`): the shipped `agents` bundle names Claude Code `claude-code`. A reference to an alias
  names the agent its bundle gives it unless the run's bundle has an entity of that name or the store an agent of that
  id, which wins. The planner pulls the agent in and writes the task naming it by its id (`Plan.aliases`); a step run
  outside a bundle run gets it from `A2AAgent.get`, which reads the store first and an alias after (the planner reads
  the store itself, as `STORE_GETTERS` does). agent-env's own bundles' aliases are theirs, and an alias two other
  bundles give is refused, naming both. `agent-env a2a-agent list` lists the installed agents with their aliases, then
  the store's; `a2a-agent get` prints the agent an alias names. `agent-env run --agent <name>` deploys that agent for
  the solver for one run (each step deploying or installing an agent a `prompt_agent` step prompts, not its user
  simulator: `plan.solver_steps`), pulled in and checked as a task's reference to it would be, through per-step
  `step_params` overrides, so the tasks stay as written.
- `aliases.toml` gives envs aliases the same way (`[envs] <folder> = ["<alias>", ...]`; the `mail-mcp` example names
  its env `mail`), resolved by the planner (`Plan.env_aliases`) and by `Env.get`, only in a task's steps: an env
  definition (a MultiEnv's servers) naming one is refused. The env `Env.get` gives for an alias answers to the alias's
  id, so a deployment records, and steps look it up by, the id the task names; its metadata writes go to the env the
  alias names (`Env.stored_id`). An alias names its env only where namespace routing is on. An env store refuses to
  create an env under an installed env alias's id (`EnvStore.put_document`), so a deployment recorded under an alias
  always reloads the alias's env; an env stored under such an id before keeps taking versions.
- `aliases.toml` gives artifacts aliases too (`[artifacts] "<folder>" = ["<alias>", ...]`; the `computers` bundle
  names its base `ubuntu-24.04`), resolved by the planner (`Plan.artifact_aliases`) in a task's steps alone, as env
  aliases are. `Artifact.get` reads no alias: the planner rewrites every alias a task names to its id. An alias may
  hold '.', so quote the folder in the toml.
  `A2AAgent.validate()` deploys `mail` (`A2AAgent.VALIDATION_ENV_ID`) with namespace routing on, first writing the
  example (its writes and infra, no run) when the store has no env of that id: on a remote sandbox its image is
  context-only and built in the VM, so a service like the hub's worker needs no Docker.
- Environment providers have no table: an installed package registers one as an `agent_env.env_providers`
  entry point named its `type`, which records carry as `env_provider_type`.
- `[plugins.<distribution name>]` is reserved for an installed plugin's own settings, read with
  `agent_env.plugins.settings`. Core reads nothing inside it and only reports it (`config show`,
  `config explain`); every other top-level table is core's, so do not add one for a plugin.
- Each section row in `config/describe.py` declares the keys its reader takes (`FileSection.keys`), and
  `config show` warns about any other. A reader that takes a new key declares it there in the same change.
- CLI: an installed package adds top-level groups through the `agent_env.cli_plugins` entry-point
  group and root options through `agent_env.cli_root_options`. Core names win, a plugin that fails to
  import is skipped with a warning, two different root options on one flag are both left off and
  reported as a conflict, and grafting subcommands onto built-in groups is unsupported. Core does not
  add a root option a known plugin uses.
  `plugin` is a core group (`agent-env plugin list/show/check/add/remove`), so a plugin command of that name is skipped.

Do not add a seam that nothing in this repository consumes or defaults: a new config section needs an
in-tree implementation or default, and new behaviour is selected in `config.toml`, not by a new
environment variable.

## Task.run() contract for embedders

`Task.run()` is the single execution entry point; the in-process `LocalRunner` and external
orchestrators call it the same way. What callers rely on:

- `register_task_instance()` creates the task-instance record at start; passing an existing
  `instance_id` makes it an idempotent upsert. After each step `record_step_complete()` writes only
  the fields that step changed (a path-level diff) and derives `current_step` / `status`; failures go
  through `record_task_failure()`, retries through `undo_steps()`.
- The runner writes its run identifier to `context.metadata["workflow_id"]` before calling `run()`;
  core stores it opaquely with the instance.
- A runner records every run it accepts in the run index (`agent_env.runner.store`): `insert_run` at
  submit, `set_running` when it starts, `mark_terminal` when it ends. The explorer lists, counts and
  groups runs from that collection and never calls the runner to ask, so a run missing there is
  invisible and one never marked terminal shows as in flight.
- Per-run overrides travel in `context.metadata["user_overrides"]` (`litellm_api_key`,
  `a2a_agent_id`, `agent_sandbox`, `env_sandbox`, `env_state_type`, and more), never in `os.environ`;
  steps fall back to the configured `[model]` key. `step_params` maps a step id to overrides of that
  step's params, which the step reads through `TaskStep.step_param_overrides()`.
- Every persisted or transmitted context snapshot passes through `TaskStepContext.to_safe_dict()`
  or the context-ops diff, which strip `_REDACTED_KEYS` (`task_step/context.py`);
  `regraft_redacted_keys` restores live values when a context is rebuilt from a stored document.
  Both write each prompt response through `PromptResponse.to_dict()`, which adds the S3-named key
  of every renamed trajectory field, since stored documents and raw-doc readers use them; serialize
  a context with `TaskStepContext.to_dict()`, not `dataclasses.asdict`.
- A per-run secret given as a bare `secret:KEY` or `env:NAME` reference names the secret rather
  than holding it, so redaction keeps it, and the step that uses it resolves it with
  `Config.resolve_reference()`: a runner whose steps run on another host passes per-run keys this
  way. A reference with a `?default` is stripped like a secret. The run index strips the same keys
  from a run's overrides. A reference resolves against the environment and `[stores.secret]` of the
  host running the step, so a runner that takes runs from callers who may not read those refuses
  references it did not create.
- A run is resumable: pass a previously persisted `context`, a `start_step` and the existing
  `instance_id`; completed steps are seeded as SUCCESS.
- A runner that must hear from a long step puts a zero-argument callable at
  `context.metadata[HEARTBEAT_KEY]` (`task_step/context.py`); a step that blocks for long calls
  `context.heartbeat()`, a no-op without one. The key is in `_REDACTED_KEYS`, so it is never persisted.
- An episode (`EpisodeSpec`, `task/episode/`) is one `Task.run()` of a copy of the task, with
  namespace routing on, in which each of its `trainer_agents` is played through a proxy agent
  (the `agents` bundle's `proxy`, `@local/agentenv-framework/agents/proxy`, which the episode writes
  from its folder as a bundle run would).
  The agent's `deploy_agent` or `install_agent`, wrapped as `episode_agent`, deploys the proxy in the
  agent's place when the task uses the agent's sandbox and serves it in this process when it doesn't;
  the agent's `prompt_agent` steps message it as they would the agent, and the caller plays each
  message as a `Turn` through its `urn:agentenv:proxy/v1`: every env tool call and `bash` command runs
  in the proxy, which records the turn and checks it against the model calls the caller records. Its
  record is tagged `context.metadata["episode"]` and names the original task, so it isn't resumable
  with `start_step`. The episode tears down what its run deployed, and an abandoned one is recorded
  `cancelled`.

## Tests

- Tiers are by path. `tst/unit/` runs offline: `tst/unit/conftest.py` blocks IP sockets with
  pytest-socket, so anything that needs the network is either missing a mock or belongs in
  `tst/integration/`. The AWS backends' tests live in `tst/unit/store/aws/` and run against moto or
  stubbed clients, and the directory is skipped without the `aws` extra. Opt out per test with `@pytest.mark.enable_socket`, or override the
  `_disable_network` fixture in a closer conftest when a loopback server is the point.
- `tst/integration/` needs Docker and the local registry on `:5000` and runs on the local default
  backends. Minutes-long tests (real image builds, sandbox VMs, gateways, agents) carry
  `@pytest.mark.int_test_slow` (or module-level `pytestmark`) and run serially. CI splits them across
  runners instead: `AGENTENV_TEST_SHARD=i/N` (`tst/util/shards.py`) runs the i-th of N groups, keeping a
  file's tests together and balancing the groups by `tst/integration/slow_durations.json`. When the
  shards' times drift apart, rewrite the table from a `main` run's `junit-integration-local-*`
  artifacts with `.github/scripts/slow_test_durations.py`.
- `tst/manual/` and `tst/benchmarks/` run only by hand, as each module's docstring says; no CI job
  collects them.
- Skips are declared capability gaps, decided at collection time: the gates in
  `tst/util/capabilities.py` produce `skipif` marks with the reason
  `agentenv-capability-missing: <name>` (`model_endpoint_configured`, `remote_sandbox`,
  `default_a2a_agent`, `mcp_server_sources`, `gnu_stat`). CI rejects any other skip reason; a broken configuration must fail loudly, not skip.
- The suite is backend-agnostic: a test about one particular backend lives with that backend, not
  here. A custom store must pass `tst/store/{conformance,object_conformance,image_conformance,secret_conformance}.py`,
  the same kits the built-ins pass, and a custom `[runner]` must pass `tst/runner/conformance.py`, which
  `LocalRunner` passes.
- Shared helpers live in `tst/util/` (`config.py` installs a config document, `a2a_test_agent.py` builds
  the echo A2A agent, `capabilities.py`, `image_cache.py`); fixtures and mock services in `tst/data/`.
  Nothing test-only ships under `src/`.
- `tst/conftest.py` resolves the config afresh per module and resets it per test; with no config
  file the suite runs on the local defaults.

## CI

A merge into `main` needs an approving review from a code owner (`CODEOWNERS`), every review thread
resolved, and the checks the `main` ruleset requires: `unit`, `integration-local` and
`integration-local-slow`. The other checks below run on every pull request too and are expected to
pass, but a failing one does not block the merge.
`.github/workflows/local-backends.yml` runs `unit`, `ui`, `integration-local`, `integration-local-slow`,
`integration-local-gateway` (the slow tier's `gateway_test.py`, run beside the rest of it) and
`installer`. The slow tier runs on four runners (`integration-local-slow (1/4)` to `(4/4)`) and the
gateway tests on two, and `integration-local-slow` and `integration-local-gateway` each pass when all
of their shards do. The Python jobs install from `uv.lock` (public PyPI only), with the uv that
`.github/ci-requirements.txt` pins, and use no secrets and no external services (a `registry:2`
container for the integration jobs). The `unit` job also fails if `uv.lock` resolves anything from a
registry other than PyPI, if `agentenv-framework-protocol` is not the editable workspace member, if
`uv lock --check` fails before or after a release's version bump (`scripts/bump_version.py`), or if
`ruff check` reports a problem. The `ui` job type-checks, lints, smoke-tests and statically exports
the explorer UI. The integration jobs run
`.github/scripts/check_skip_policy.py` over the JUnit report: only `agentenv-capability-missing`
skips for the allowed capabilities pass, and the fast job allows no skips under
`tst/integration/store/`. The `installer` job runs `tst/installer` when a pull request or push
touches the plugin CLI, `agent_env/plugins` or `agent_env/config`, the protocol package, the tier
itself, `pyproject.toml`, `uv.lock`, the `Makefile` or the CI that runs the tier, and allows no
skips; a `changes` job decides, and a skipped `installer` counts as passed.
`.github/workflows/plugin-api.yml` runs the `plugin-api` job on every pull
request, title edits included, and on `main`: `.github/scripts/check_plugin_api.py` compares the
plugin surface (its `BASES` and `USED`) between `HEAD^1` and `HEAD` with griffe, pinned in the `dev` extra, and fails on a break the
title does not mark with `!`. `.github/workflows/clean-install.yml` runs the `clean-install` job on
every pull request and on `main`: `.github/scripts/clean_install.py` builds both distributions as the
release does, fails if the wheel leaves out a file tracked in a shipped bundle's folder or an
`agent_env.bundles` entry point, installs the two wheels into a fresh venv from public PyPI with an
allowlisted environment (no AWS, no config, no plugin), and runs `agent-env run hello`
twice by name; `.github/scripts/check_clean_install.py`, run by that venv, checks the local store,
that the runs left no sandbox work folder, and that each published infra image's tag is the same from the
installed sources as from the checkout.
`.github/workflows/publish-pypi.yml` runs on the `v*` tag the release automation pushes: it waits
for `main`'s CI to pass on the tagged commit, publishes `agentenv-framework-protocol` and then
`agentenv-framework` to PyPI, pinning the protocol in `pyproject.toml` and in each shipped env's and artifact's
`requirements.txt` (`.github/scripts/shipped_bundles.py pin` finds them, so a new env is pinned with no workflow change,
and every job that reads those folders runs it first, so an image's tag, its build and the wheel see the same sources),
and creates the GitHub release, whose notes are the titles of the pull
requests merged since the previous release.
Once the repository variable `PUBLISH_INFRA_IMAGES` is `true`, it also publishes the infra images agent-env
ships and the shipped bundles', one per agent folder, one for the env `a2a-agent validate` deploys (the `mail-mcp`
example's) and one per shipped base computer (`agentenv-ubuntu-24.04`; `agent_env.env.published_images`), whose `src-<hex>` tags GHCR lacks: after `main`'s CI passes, each is
built on a runner of each platform, checked, and tagged as one multi-platform image, and a tag that exists is
never moved. The infra images are published before the protocol, and a failed infra image job sends neither package
to PyPI. An agent's `requirements.txt`, and that env's, keeps its protocol floor, unpinned, so its image is built again
only when its folder changes (raise the floor when it needs a newer protocol), once the protocol is on PyPI, and a
failed agent image job keeps `agentenv-framework` off it. The wheel and the release notes record each
image's digest. Run by hand on `main`, the workflow publishes the images alone.
Its jobs that run agent-env (`.github/scripts/infra_images.py`) hold no token that writes; the jobs that push
images or publish to PyPI install nothing and check what they're handed (`.github/scripts/infra_build.py`,
standard library and the docker CLI only).
Actions are pinned to commit SHAs; Dependabot updates them, the pinned uv, `uv.lock`, the UI's npm packages and
the infra images' base images, each pinned by digest so that a bump rebuilds and republishes the image; Dependabot
proposes only a base's digest and patch updates, and a major or minor upgrade is made by hand.

## Conventions

- PR titles (the squash commit) follow Conventional Commits: `type(scope)!?: imperative summary`,
  with the types in use `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `chore`, `perf`, `build`,
  `security`; `!` marks a breaking change, and `plugin-api` requires it for a plugin-surface break.
- Versions are cut by the maintainers' release automation. Never edit `version` in
  `pyproject.toml` or push a tag in a PR.
- Names: the distribution is `agentenv-framework`, the import package `agent_env`, the console
  script `agent-env`. Only the first is what `uv add`, `pip install` and `importlib.metadata.version()` take.
- Comments: default to none. The code says what; a docstring of one to three dense lines says why
  when that is not obvious. No comment that restates the code, no trailing assertion notes in tests.
- Imports at module top, never inside functions.
- No ticket ids, program codenames or review context in code, docstrings or docs; that belongs in
  the commit message and the PR.
- No defensive code for a case no caller produces; say so in the PR instead.
- Keep core vendor-neutral: no vendor hostnames, account ids or deployment facts in `src/`, `tst/`
  or this file. A deployment plugs in through the extension points above. Where the project itself
  publishes isn't a deployment fact: its PyPI names, its GitHub repository, and the GHCR namespace its
  release workflow publishes infra images to (`published_images.REGISTRY`).

## Docs

`README.md` is the front door: what AgentEnv Framework is, how to install it, and links. The user
guide (configuration, environments, steps, artifacts, providers, the CLI) is
www.agentenvframework.com/docs. A new feature, provider or option is documented there, not in the
README: say in the pull request what the docs should cover, and a maintainer updates the site.
`packages/agentenv-protocol/README.md` documents the protocol and the A2A agent framework.

Guides for coding agents sit next to the code they cover: `src/agent_env/agents/AGENTS.md` for writing an A2A agent and
`src/agent_env/providers/sandbox_providers/AGENTS.md` for writing a sandbox provider.
