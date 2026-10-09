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

`agent-env run hello` runs the built-in hello task, which needs no Docker, model or configuration. To try it without installing anything, run it with uvx:

```bash
uvx --from agentenv-framework agent-env run hello
```

Or install it with pip, into a virtualenv:

```bash
pip install agentenv-framework
```

To use the SDK in your own project, add it as a dependency with `uv add agentenv-framework`. The explorer, `agent-env up`, needs the `explorer` extra (`uv tool install 'agentenv-framework[explorer]'`, or the same extra with uvx or pip) and an `.agentenv/config.toml` in the current folder or above it; an empty one keeps every local default.

A plain install runs on local stores and needs no cloud SDK. The cloud store backends are extras: `aws` for S3, Secrets Manager, DynamoDB and ECR (`pip install 'agentenv-framework[aws]'`), and `gcp` for Cloud Storage, Secret Manager, Firestore and Artifact Registry (`pip install 'agentenv-framework[gcp]'`). A config that names a backend without its extra fails, naming the extra to install.

The distribution is named `agentenv-framework`, the import package is `agent_env` and the command is `agent-env`. It depends on `agentenv-framework-protocol`, whose import package is `agentenv_protocol`, and installs it too.

To work on agent-env itself, install from a clone with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/scaleapi/agentenv-framework && cd agentenv-framework
uv sync --extra dev
source .venv/bin/activate
```

## Freestyle sandboxes

Freestyle is available as the `freestyle` sandbox backend for environments and agents.
It boots Docker-capable VM snapshots, exposes requested ports over HTTPS, and supports
file transfer, reconnection, and teardown. See the [Freestyle provider guide](src/agent_env/providers/sandbox_providers/freestyle/README.md)
for configuration, resource sizing, and command/network limits.

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
