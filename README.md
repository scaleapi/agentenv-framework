<h1><img alt="AgentEnv Framework" src="https://raw.githubusercontent.com/scaleapi/agentenv-framework/main/assets/brand/readme-banner.png"></h1>

AgentEnv Framework is a Python SDK and CLI for building, deploying and running agentic environments and the tasks that grade agents inside them. Environments are containerized servers that speak the open `agentenv-framework-protocol`; it builds them into versioned images, deploys them behind a gateway, points an agent at them, and scores what the agent did.

**Documentation: [www.agentenvframework.com/docs](https://www.agentenvframework.com/docs)** covers environments, artifacts, agents, tasks, the registry and plugins.

## Install

Both packages are on PyPI. You need Python 3.11 or newer and, to run environments locally, a running Docker daemon. With [uv](https://docs.astral.sh/uv/), install the `agent-env` command as a tool:

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

The distribution is named `agentenv-framework`, the import package is `agent_env` and the command is `agent-env`. It depends on `agentenv-framework-protocol`, whose import package is `agentenv_protocol`, and installs it too.

To work on agent-env itself, install from a clone with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/scaleapi/agentenv-framework && cd agentenv-framework
uv sync --extra dev
source .venv/bin/activate
```

## Plugin contract

What a package that extends agent-env relies on. Unit tests and the plugin-API check in CI hold the code to these sections, so a pull request that changes the contract changes this README too.

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

A plugin that needs settings of its own reads them from `[plugins.<package>]`, where `<package>` is its distribution name: the name `uv add` and `pip install` take and `agent-env plugin list` prints. That table belongs to the plugin. agent-env reads nothing in it and checks none of its keys. Every other top-level table belongs to agent-env, which warns about one it does not read in `agent-env config show`, so a plugin keeps nothing of its own anywhere else.

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
- `agent-env config show` lists each table under its package: with the installed version, with `(declares no agent_env entry point)` when the distribution is installed but is not a plugin, which usually means a misspelled entry-point group, with `(agent-env itself)` for `agentenv-framework`, whose table agent-env does not read, or with `(not installed)`. `config explain plugins.<package>.<key>` says whose table holds the key; agent-env does not read or check it, so it cannot tell whether the plugin reads that key, and a misspelled key is still shown. A table for a plugin that is not installed is reported, never an error, because one config file is often shared by processes that install different plugins.
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
| `incompatible-core` | `failed` | The plugin's requirement on `agentenv-framework` or `agentenv-framework-protocol` excludes the installed version, so it was not imported. See [Plugin compatibility](#plugin-compatibility) | Upgrade agent-env, or install a version of the plugin that fits |
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

**What agent-env checks.** Before it imports a plugin, agent-env compares the plugin's requirements on `agentenv-framework` and `agentenv-framework-protocol` with the versions installed. A plugin they exclude is not loaded:

- `plugin list`, `show` and `check` report each of its contributions as `failed` with the code `incompatible-core`, with or without `--no-load`, and `check` exits 1;
- using one of its types names the requirement, as in `needs agentenv-framework>=0.9.1220 (installed: 0.9.1218)`;
- it claims none of its names, so a plugin that can load and registers the same name does not conflict with it.

An installer that resolves dependencies never gets you there; `pip install --no-deps` or a forced install can. There is no override: upgrade agent-env, or install a version of the plugin that fits. A requirement under an extra, or whose environment marker is false, does not count, and an agent-env with no installed metadata, such as a source tree on `sys.path`, is not checked. Requirements between plugins are the installer's to check; `pip check` lists any that are unmet.

## Contribute, release, license

[CONTRIBUTING.md](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md) is the contributor guide: open an issue before anything larger than a bug fix, one logical change per pull request with tests, pull request titles in the form `type(scope): summary`, an approving review from a code owner (`CODEOWNERS`) and green CI. [AGENTS.md](https://github.com/scaleapi/agentenv-framework/blob/main/AGENTS.md) is the repository map and conventions file for contributors and coding agents; `CLAUDE.md` imports it.

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

The `agentenv-framework` distribution and `agentenv-framework-protocol` are versioned separately (`0.9.x` and `0.1.x` today), both in their `pyproject.toml`; agent-env releases carry a `vX.Y.Z` tag, and agentenv-framework-protocol is bumped in the same commit and has no separate tag today. There is no `agent_env.__version__` attribute; `agent-env --version` prints the installed version. Protocol extensions carry their version in the URI (`urn:agentenv:clock/v1`, `urn:agentenv:agent-config/v1`, `urn:agentenv:trajectory/v1`). One version can take more than one request shape: under `v1` the skill, trajectory, snapshot and changelog extensions accept object-transfer requests next to their older shapes, and the request field lists on an agent's card say which ones that agent takes. Renamed CLI commands are removed outright; no deprecated aliases exist at this version. What plugins may rely on, and how changes to it are made, is in [Plugin compatibility](#plugin-compatibility); the plugin commands' `--json` output has its own format version and rules ([Plugin report format](#plugin-report-format)). Beyond those, a written compatibility and deprecation policy does not exist yet.

### Releases

A release is a version bump in both `pyproject.toml` files plus a `vX.Y.Z` tag. Maintainers cut releases: the bump is automated when a labelled pull request merges, so contributors do not edit `version` or push tags (see the Releases section of [CONTRIBUTING.md](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md)). Neither package is published to a public index yet, and there is no `CHANGELOG.md`.

### Support, security, license

Report bugs and gaps as issues against this repository, with the installed `agentenv-framework` version and the sandbox backend in use. Report vulnerabilities privately through the contact in [SECURITY.md](https://github.com/scaleapi/agentenv-framework/blob/main/SECURITY.md), not in public issues; only the latest release is supported, so reproduce against it first. Contributors follow [CONTRIBUTING.md](https://github.com/scaleapi/agentenv-framework/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/scaleapi/agentenv-framework/blob/main/CODE_OF_CONDUCT.md); every pull request needs a code-owner review. agent-env and agentenv-framework-protocol are licensed under the Apache License 2.0; see [`LICENSE`](https://github.com/scaleapi/agentenv-framework/blob/main/LICENSE) and [`NOTICE`](https://github.com/scaleapi/agentenv-framework/blob/main/NOTICE). Their third-party dependencies and those dependencies' licenses are listed in [`THIRD_PARTY_NOTICES.md`](https://github.com/scaleapi/agentenv-framework/blob/main/THIRD_PARTY_NOTICES.md).
