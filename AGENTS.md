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

Python 3.11 or newer. The repo convention is a `.venv` at the root, which the Makefile and CI use.

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
| `make unit-test` | `tst/unit` and `packages/agentenv-protocol/tests`, in parallel, no network |
| `make int-test-fast` | `tst/integration` minus `int_test_slow`, in parallel; needs Docker and a local OCI registry (`docker run -d -p 5000:5000 public.ecr.aws/docker/library/registry:2`) |
| `make int-test-slow` | the `int_test_slow` tests, serially (they build the same Docker tags from module fixtures and race under xdist) |
| `make installer-test` | `tst/installer`: `plugin add` / `remove` through the real pip, uv and pipx, offline against wheels built from the checkout, and the container user journey (`python:3.12-slim`, `--network none`) with the PEP 668 refusal; needs uv, pipx and Docker |
| `make clean-install-test` | the `clean-install` CI gate: both distributions built as the release builds them, installed into a fresh venv from public PyPI with an allowlisted environment, and `agent-env run hello` run twice by name and checked through the local store; needs Python 3.11, uv and git |
| `pytest tst/unit/env/store_test.py -v` | one file; add `--log-cli-level=DEBUG` for debug logs |
| `uv build` | wheel `agentenv_framework-<version>-py3-none-any.whl` (`src/agent_env` only) and sdist `agentenv_framework-<version>.tar.gz` (`src`, `tst`, the protocol package) |
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
  sandboxes for envs and agents. MongoDB, S3, Cloud Storage, ECR, AWS Secrets Manager, Google Cloud
  Secret Manager, Modal and E2B exist as implementations and are selected by config.
- The model endpoint is unset until `[model] base_url` / `api_key` (or `LITELLM_BASE_URL` /
  `LITELLM_API_KEY`) is configured. A loopback URL handed into a `local` container (the model
  endpoint, an agent, peer or gateway on this machine) becomes `host.docker.internal`. On Linux that
  name is the Docker bridge address (`docker0`, usually `172.17.0.1`), not loopback, so a service on
  this machine that local containers call, a model proxy say, has to listen there: bind it to the
  bridge address and use that address in its URL. Binding `0.0.0.0` also works, but exposes the
  service to the network.
- `[agents] default_a2a_agent_id`: the agent a `deploy_agent` step without an id deploys; built-in
  `a2a-default`, overridable by `configure(default_a2a_agent_id=...)`.
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
| `env/` | `Env` base class: a subclass declares `type`, implements `from_dict` and `async deploy(...) -> DeployedEnv`, and may override `reset` and `load_file_artifact_universe`. Built-ins in `env/envs/`: `MCPServerEnv` (`mcp_server`), `MultiEnv` (`multi`, MCP servers plus websites), `ServiceDBEnv` (`service_db`, PostgreSQL), `WebsiteEnv` (`website`), `GatewayEnv` (`gateway_server`). `env/gateway/gateway.py` is the env-side gateway server, container-only code. |
| `artifact/` | Immutable versioned artifacts; every `put()` creates a new version. The document goes to the document store, binary payloads to the object store. Built-ins: `CliArtifact` (`cli`), `FileArtifact` (`file`), `FileArtifactUniverse` (`file_artifact_universe`), `DockerImageArtifact` (`docker_image`), `EnvironmentArtifact` (`environment`; legacy `service`), `EnvironmentUniverseArtifact` (`environment_universe`; legacy `service_universe`), `SkillArtifact` (`skill`). A `DockerImageArtifact` holds a tar.gz of its image, or, written by `put_ref` (`--image-ref` on `env mcp-server put` and `a2a-agent put`), only a registry reference with its tag pinned to a digest, which sandboxes pull. |
| `task/`, `task_step/` | A `Task` holds its `TaskStep`s inline; `Task.run()` executes them as a DAG. `depends_on` (None means all prior steps) gates a step, independent steps run concurrently, `fail_task_on_error` makes a failure fatal or tolerated, `retry_config` rolls a failed span back through the step journal and re-dispatches it. Built-in steps live in `task_step/task_steps/` (`deploy_env`, `deploy_agent`, `prompt_agent`, the verifiers under `verifiers/`, and more). |
| `store/` | Four store ABCs with local and cloud implementations: `DocumentStore` (SQLite, MongoDB), `ObjectStore` (filesystem, S3, Cloud Storage), `ImageStore` (local OCI registry, ECR), `SecretStore` (env vars or file, AWS Secrets Manager, Google Cloud Secret Manager). `VersionedEntityStore` implements the shared versioned get/put logic, `QueryBuilder` is the immutable chained query API, `store/base.py` holds the error types. A new backend must pass the conformance kits in `tst/store/`. |
| `config/` | The `Config` singleton (`get_config`, `configure`, `reset_config`) in `config/runtime.py`, file discovery in `config/loader.py`, and `load_impl`, which resolves `module:Class` pointers. `agent_env.store` re-exports the config names for compatibility. |
| `providers/` | `providers/sandbox_providers/` holds the sandbox providers `local`, `modal`, `modal_vm`, `e2b`, `sail_vm` (the `sail` extra); `[sandbox] default` and `agent_default` accept a comma-separated fallback chain. `providers/env_providers/` holds the environment providers: `EnvironmentProvider` (an env's containers and state store) and `EnvironmentGatewayProvider`, which renders a docker-compose for the gateway and its MCP servers inside the sandbox; `providers/env_state/` holds env-state providers (`local_postgres` built in). |
| `a2a_agent/` | The `A2AAgent` entity (`a2a_agent`), its stores and the validator steps. The protocol package provides the agent-side framework. |
| `runner/` | The `[runner]` seam: `Runner.submit()` returns `(run_id, instance_id)`; `LocalRunner` is built in. |
| `explorer/` | Optional local web UI: `agent-env up`, needs the `explorer` extra, binds loopback `:8234`. |
| `cli/` | Click CLI with the groups `a2a-agent`, `artifact`, `config`, `env`, `eval`, `plugin`, `run`, `task`, `up`. |
| `examples/` | The bundles agent-env ships (`hello`), registered under `agent_env.bundles` in `pyproject.toml`. Tests run them, but they are examples for users, not fixtures. |

## Extension points

Everything out-of-tree is declared in `config.toml` as a `module:Class` pointer and constructed with
`from_config(**config)`:

- `[stores.document|object|image|secret]`, `[runner]`, `[sandbox.providers.<name>]`,
  `[state.providers.<name>]`, `[explorer.plugins]`: one implementation each.
- `[envs]`, `[artifacts]`, `[task_steps]` with `impls = ["module:Class", ...]`: extra classes for the
  registries. The identity is the class's `type` (a `ClassVar` on envs and steps, the Pydantic `type`
  default on artifacts); a class that keeps the base default, collides with a registered type, or
  leaves a method its base requires unimplemented fails at registry build with `ConfigError`.
- For sandbox and state providers the table name must equal the `type` the provider produces,
  because that string is persisted and used to reconnect.
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
- Per-run overrides travel in `context.metadata["user_overrides"]` (`litellm_api_key`,
  `agent_sandbox`, `env_state_type`, `step_params`), never in `os.environ`; steps fall back to the
  configured `[model]` key.
- Every persisted or transmitted context snapshot passes through `TaskStepContext.to_safe_dict()`
  or the context-ops diff, which strip `_REDACTED_KEYS` (`task_step/context.py`);
  `regraft_redacted_keys` restores live values when a context is rebuilt from a stored document.
- A run is resumable: pass a previously persisted `context`, a `start_step` and the existing
  `instance_id`; completed steps are seeded as SUCCESS.

## Tests

- Tiers are by path. `tst/unit/` runs offline: `tst/unit/conftest.py` blocks IP sockets with
  pytest-socket, so anything that needs the network is either missing a mock or belongs in
  `tst/integration/`. The AWS backends' tests live in `tst/unit/store/aws/` and run against moto or
  stubbed clients, and the directory is skipped without the `aws` extra. Opt out per test with `@pytest.mark.enable_socket`, or override the
  `_disable_network` fixture in a closer conftest when a loopback server is the point.
- `tst/integration/` needs Docker and the local registry on `:5000` and runs on the local default
  backends. Minutes-long tests (real image builds, sandbox VMs, gateways, agents) carry
  `@pytest.mark.int_test_slow` (or module-level `pytestmark`) and run serially.
- Skips are declared capability gaps, decided at collection time: the gates in
  `tst/util/capabilities.py` produce `skipif` marks with the reason
  `agentenv-capability-missing: <name>` (`model_endpoint_configured`, `remote_sandbox`,
  `default_a2a_agent`, `mcp_server_sources`, `gnu_stat`). CI rejects any other skip reason; a broken configuration must fail loudly, not skip.
- The suite is backend-agnostic: a test about one particular backend lives with that backend, not
  here. A custom store must pass `tst/store/{conformance,object_conformance,image_conformance,secret_conformance}.py`,
  the same kits the built-ins pass.
- Shared helpers live in `tst/util/` (`config.py` installs a config document, `a2a_test_agent.py` builds
  the echo A2A agent, `capabilities.py`, `image_cache.py`); fixtures and mock services in `tst/data/`.
  Nothing test-only ships under `src/`.
- `tst/conftest.py` resolves the config afresh per module and resets it per test; with no config
  file the suite runs on the local defaults.

## CI

The required checks on pull requests and `main` are `unit`, `integration-local`,
`integration-local-slow`, `installer`, `plugin-api` and `clean-install`. `.github/workflows/local-backends.yml` runs
the first four, all installed from public PyPI with no secrets and no external services (a `registry:2` service container for the integration jobs). The
`unit` job also fails if `uv.lock` resolves anything from a registry other than PyPI or if
`agentenv-framework-protocol` is not the editable workspace member. The integration jobs run
`.github/scripts/check_skip_policy.py` over the JUnit report: only `agentenv-capability-missing`
skips for the allowed capabilities pass, and the fast job allows no skips under
`tst/integration/store/`. The `installer` job runs `tst/installer` when a pull request or push
touches the plugin CLI, `agent_env/plugins`, the tier itself or the lockfile, and allows no skips; a
`changes` job decides. `.github/workflows/plugin-api.yml` runs the `plugin-api` job on every pull
request, title edits included, and on `main`: `.github/scripts/check_plugin_api.py` compares the
plugin surface (its `BASES` and `USED`, which a unit test holds to the README "Plugin compatibility"
list) between `HEAD^1` and `HEAD` with griffe, pinned in the `dev` extra, and fails on a break the
title does not mark with `!`. `.github/workflows/clean-install.yml` runs the `clean-install` job on
every pull request and on `main`: `.github/scripts/clean_install.py` builds both distributions as the
release does, fails if the wheel leaves out a file tracked under `src/agent_env/examples` or a
`agent_env.bundles` entry point, installs the two wheels into a fresh venv from public PyPI with an
allowlisted environment (no AWS, no config, no plugin), and runs `agent-env run hello`
twice by name; `.github/scripts/check_clean_install.py`, run by that venv, checks the local store and
that the runs left no sandbox work folder.
Actions are pinned to commit SHAs.

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
  or this file. A deployment plugs in through the extension points above.

## Docs

`README.md` is the user-facing front door (configuration, custom envs, steps, artifacts and
providers, the CLI). `packages/agentenv-protocol/README.md` documents the
protocol and the A2A agent framework.
