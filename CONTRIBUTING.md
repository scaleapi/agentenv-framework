# Contributing to AgentEnv Framework

Thank you for helping improve AgentEnv Framework. Bug fixes, new sandbox providers, environments,
task steps, tests and documentation are all welcome.

Security vulnerabilities are handled privately: see [SECURITY.md](SECURITY.md). Everyone
participating in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

Open an issue first for anything larger than a bug fix, so the design can be agreed before you
invest in it. Small fixes can go straight to a pull request.

agent-env is released under the Apache License 2.0 (see `LICENSE`). By submitting a contribution
you agree that it is licensed under the same terms.

## Development setup

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/) and, for the integration tiers,
Docker.

```bash
git clone https://github.com/scaleapi/agentenv-framework.git
cd agentenv-framework
uv venv .venv
uv pip install --python .venv/bin/python -e ./packages/agentenv-protocol -e '.[dev]'

The distribution is `agentenv-framework` (import `agent_env`, command `agent-env`). A virtualenv created
when the distribution was still called `agent-env` holds both after a reinstall, because the two
write the same files; run `uv pip uninstall --python .venv/bin/python agent-env` once before
reinstalling.
```

This is the same install CI performs. `make install` is equivalent.

## Running the tests

| Command | What it runs | Needs |
|---|---|---|
| `make unit-test` | unit suite for `agentenv-framework` and `agentenv-framework-protocol` | nothing external |
| `make int-test-fast` | integration tests against the local backends | Docker |
| `make int-test-slow` | tests that build images or deploy sandboxes, minutes each | Docker, and a sandbox backend for some |
| `make clean-install-test` | both distributions built as the release builds them, installed into a fresh venv from public PyPI, and `agent-env run hello` run twice by name | Python 3.11, uv and git |
| `make installer-test` | `plugin add` and `remove` through the real pip, uv and pipx, offline against wheels built from the checkout, and the new-user journey in containers with no network | uv, pipx and Docker; the first run downloads the dependencies once |

CI runs the same tiers as the `unit`, `integration-local` and `integration-local-slow` jobs;
all three must pass, and so must `plugin-api` (see "Making changes"). The `installer` job runs
`make installer-test` too, on pull requests that touch the plugin code, the installer tier or
the lockfile, and must pass when it runs. One exception: the public jobs exclude
`tst/integration/env/gateway/gateway_test.py`, which needs an x86 Chromium build that the
hosted runners do not have; its virtual-clock tests also build MCP servers from sources outside
this repository, found through `AGENT_ENV_TEST_MCP_SERVERS_DIR` (one `<server>/Dockerfile` per
server), and skip without them. `make int-test-slow` does run it, so run that locally when your
change touches the gateway, and say so in the pull request. A test may skip only when a capability is genuinely unavailable, and the skip reason
must name it (`agentenv-capability-missing: <name>`, see `tst/util/capabilities.py`); the
`check_skip_policy` step fails the run otherwise. The `clean-install` job runs the same script as
`make clean-install-test` on every pull request, and is one of the required checks.

## Making changes

- Keep a pull request to one logical change, with tests, and update the docs it affects.
- Configuration is a file: behaviour is switched through `config.toml`, never through new
  environment variables.
- Every HTTP client verifies TLS. A unit test rejects any `verify=False` in `src/`.
- Nothing test-only ships in `src/`; test helpers live under `tst/`.
- Imports go at the top of the module.
- Title the pull request `type(scope): summary`, for example `fix(gateway): forward the request
  target as received`. Types in use: `feat`, `fix`, `refactor`, `docs`, `test`, `build`, `ci`,
  `chore`, `perf`, `security`. The title becomes the commit title.
- Mark a breaking change with `!` after the scope: `feat(plugins)!: summary`. The `plugin-api` job
  fails a pull request that breaks the plugin surface without it,
  lists what broke, and runs again when you edit the title. To run it before you push:
  `.venv/bin/python .github/scripts/check_plugin_api.py --base origin/main --title "<title>"`.
- Commit with an email address you are comfortable publishing; the history of this repository is
  public. GitHub's `noreply` address is a good default.

Every pull request needs an approving review from a code owner (see `CODEOWNERS`) and green CI.

## Releases

Maintainers cut releases. Version bumps are automated when a labelled pull request merges, so
contributors should not edit `version` in `pyproject.toml`. Each release publishes
`agentenv-framework` and `agentenv-framework-protocol` to PyPI (`.github/workflows/publish-pypi.yml`, trusted publishing
on the release tag); the import package `agent_env` and the command `agent-env` keep their names. Once both are
on PyPI, the same workflow creates the GitHub release, whose notes are the titles of the pull requests merged since
the previous tag, so a clear pull request title is also the release note.

## Getting help

Ask questions in [Discussions Q&A](https://github.com/scaleapi/agentenv-framework/discussions/categories/q-a).
Report bugs with the [bug report form](https://github.com/scaleapi/agentenv-framework/issues/new?template=bug_report.yml):
it asks for the output of `agent-env plugin list`, which shows the installed version, how agent-env is
installed, the config file in effect and the plugins. Report vulnerabilities privately, as described in
[SECURITY.md](SECURITY.md).
