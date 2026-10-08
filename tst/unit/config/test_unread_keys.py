"""`config show` names the tables and keys nothing reads.

The file is checked against the keys each section declares on its row in `describe`, so the check
and the report share one list of sections. A free-form table is never descended into, a
`[plugins.<package>]` table is its plugin's, and a key another warning names is named once.
"""

import copy
import json
import pathlib
import re
import tomllib

import pytest
from click.testing import CliRunner

from agent_env.bundle.parse import CONFIG_FILES
from agent_env.cli.config import config as config_group
from agent_env.cli.config import render
from agent_env.config import describe, reset_config, runtime, snapshot
from agent_env.config.describe import describe_config, env_sources
from agent_env.config.runtime import Config
from agent_env.explorer import app as explorer_app
from agent_env.explorer import plugin as explorer_plugin
from agent_env.providers.sandbox_providers import sandbox_provider

_REPO = pathlib.Path(__file__).parents[3]


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for source in env_sources():
        monkeypatch.delenv(source.name, raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_config()
    yield
    reset_config()


def _use(tmp_path, monkeypatch, body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(body)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    reset_config()


def _warnings(tmp_path, monkeypatch, body: str) -> list[str]:
    _use(tmp_path, monkeypatch, body)
    return describe_config().warnings


# Every key a section declares, with something beneath every free-form table.
EVERY_KEY_TOML = """
runner = "local"

[stores]
document = "local"

[stores.object]
impl = "pkg.stores:Objects"
config = { bucket = "b", anything = { nested = 1 } }

[stores.image]
impl = "pkg.stores:Images"
[stores.image.config]
registry_host = "registry.invalid"
credentials = { impl = "pkg.creds:Creds", config = { region = "r" } }

[stores.secret]
impl = "pkg.stores:Secrets"

[model]
base_url = "https://example.invalid"
api_key = "env:KEY"
default = "a-model"
roles = { judge = "another-model" }
params = { temperature = 0.2 }

[conversations]
default_human_a2a_url = "https://example.invalid/a2a"

[agents]
default_a2a_agent_id = "an-agent"

[sandbox]
default = "modal"
agent_default = "local"
attribution = { product = "p", anything = "x" }

[sandbox.providers]
short = "pkg.providers:Short"

[sandbox.providers.modal.config]
gpu_type = "a100"

[sandbox.providers.custom]
impl = "pkg.providers:Custom"
config = { any_kwarg = 1 }

[state.providers]
short = "pkg.state:Short"

[state.providers.remote.config]
secret_name = "s"

[envs]
impls = []

[artifacts]
impls = []
type_aliases = { legacy = "canonical" }

[task_steps]
impls = []

[explorer]
port = 8234
cors_origins = []
allowed_hosts = []
static_dir = "ui"

[explorer.plugins]
impls = []

[plugins.acme-anything]
envz = 1
[plugins.acme-anything.sanbox]
implz = []
"""


def test_every_declared_key_and_every_free_form_table_warns_about_nothing(tmp_path, monkeypatch):
    assert _warnings(tmp_path, monkeypatch, EVERY_KEY_TOML) == []


@pytest.mark.parametrize("body, warning", [
    ('[sanbox]\ndefault = "modal"\n',
     "agent-env does not read [sanbox]: it has no such table. Did you mean [sandbox]?"),
    ('[task-steps]\nimpls = []\n',
     "agent-env does not read [task-steps]: it has no such table. Did you mean [task_steps]?"),
    ('[envs]\nimplz = ["pkg:Env"]\n',
     "[envs] has a key named 'implz', which agent-env does not read. Did you mean 'impls'?"),
    ('[sandbox.providers.custom]\nimpl = "pkg:Custom"\nconfg = {}\n',
     "[sandbox.providers.custom] has a key named 'confg', which agent-env does not read. "
     "Did you mean 'config'?"),
    # a seam named by a string, its backend or class, as the readers allow
    ('[stores]\ndocumnet = "local"\n',
     "[stores] has a key named 'documnet', which agent-env does not read. Did you mean 'document'?"),
    ('runer = "local"\n', "agent-env does not read the top-level key 'runer'. Did you mean 'runner'?"),
])
def test_a_misspelled_table_or_key_is_named_with_the_one_meant(tmp_path, monkeypatch, body, warning):
    assert _warnings(tmp_path, monkeypatch, body) == [warning]


@pytest.mark.parametrize("table", ["acme", "export"])
def test_a_table_with_no_near_name_says_where_a_plugins_settings_go(tmp_path, monkeypatch, table):
    """`export` is 0.71 from `explorer`: offering it would be wrong advice, so none is offered."""
    assert _warnings(tmp_path, monkeypatch, f"[{table}]\ntimeout = 30\n") == [
        f"agent-env does not read [{table}]: it has no such table, and a plugin's own settings go under "
        "[plugins.<package>]."
    ]


def test_a_key_beside_a_seams_impl_says_where_the_class_settings_go(tmp_path, monkeypatch):
    assert _warnings(tmp_path, monkeypatch, '[stores.secret]\nimpl = "pkg:Secrets"\nsecret_name = "s"\n') == [
        "[stores.secret] has a key named 'secret_name', which agent-env does not read: [stores.secret] "
        "takes 'impl' and 'config', and the class's own settings go under [stores.secret.config]."
    ]


def test_a_key_with_no_near_name_lists_what_its_table_takes(tmp_path, monkeypatch):
    assert _warnings(tmp_path, monkeypatch, '[conversations]\nurl = "https://x"\n') == [
        "[conversations] has a key named 'url', which agent-env does not read: [conversations] "
        "takes 'default_human_a2a_url'."
    ]


def test_a_table_an_env_var_replaces_is_still_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    warnings = _warnings(tmp_path, monkeypatch, '[stores.document]\nimpl = "pkg:Docs"\nconfg = {}\n')
    assert warnings == ["[stores.document] has a key named 'confg', which agent-env does not read. "
                        "Did you mean 'config'?"]


@pytest.mark.parametrize("body", [
    "sandbox = 5\n",
    "[sandbox]\nproviders = 5\n",
    "[sandbox.providers]\nx = 5\n",
    "[stores]\ndocument = 5\n",
    "[explorer]\nplugins = []\n",
    "plugins = 5\n",
    "[plugins]\nacme = 5\n",
])
def test_a_value_of_the_wrong_shape_is_left_to_its_reader(tmp_path, monkeypatch, body):
    assert _warnings(tmp_path, monkeypatch, body) == []


@pytest.mark.parametrize("body", [
    '[task_steps]\nimpls = []\nenvs = "pkg:VmEnv"\n',
    '[stores.document]\nimpl = "pkg:Docs"\ndatabase = "a"\n[stores.document.config]\ndatabase = "b"\n',
    "[plugin.acme]\nx = 1\n",
    '[sandbox]\nplugins = { acme = { x = 1 } }\n',
])
def test_a_key_another_warning_names_is_named_once(tmp_path, monkeypatch, body):
    assert len(_warnings(tmp_path, monkeypatch, body)) == 1


def test_a_config_key_in_a_free_form_table_shadows_nothing(tmp_path, monkeypatch):
    """[model.params] is passed through whole, `config` and all."""
    assert _warnings(tmp_path, monkeypatch, '[model.params.tool]\nmode = "outer"\nconfig = { mode = "inner" }\n') == []


def test_a_key_beside_config_in_a_table_that_takes_no_config_is_named(tmp_path, monkeypatch):
    takes = "[model] takes 'base_url', 'api_key', 'default', 'roles' and 'params'."
    assert _warnings(tmp_path, monkeypatch, '[model]\nmode = "x"\nconfig = { mode = "y" }\n') == [
        f"[model] has a key named 'mode', which agent-env does not read: {takes}",
        f"[model] has a key named 'config', which agent-env does not read: {takes}",
    ]


@pytest.mark.parametrize("body, warning", [
    ('[sandbox.providers."a.b"]\nimpl = "pkg:P"\nstray = 1\n',
     "[sandbox.providers.\"a.b\"] has a key named 'stray', which agent-env does not read: "
     "[sandbox.providers.\"a.b\"] takes 'impl' and 'config', and the class's own settings go under "
     "[sandbox.providers.\"a.b\".config]."),
    ('[sandbox.providers."a.b"]\nimpl = "pkg:P"\nregion = "r"\nconfig = { region = "s" }\n',
     "[sandbox.providers.\"a.b\"] sets 'region' both beside its config table and inside it; "
     "only [sandbox.providers.\"a.b\".config] region is read."),
    ('["my table"]\nx = 1\n',
     "agent-env does not read [\"my table\"]: it has no such table, and a plugin's own settings go under "
     "[plugins.<package>]."),
])
def test_a_table_is_named_as_the_file_writes_it(tmp_path, monkeypatch, body, warning):
    assert _warnings(tmp_path, monkeypatch, body) == [warning]


def _documented_configs() -> list[str]:
    docs = "".join((_REPO / name).read_text() for name in ("README.md", "PLUGINS.md"))
    blocks = re.findall(r"```toml\n(.*?)```", docs, re.S)
    # the pyproject.toml blocks declare entry points, and a block headed by a bundle path
    # (`# agents/solver/agent.toml`) is an entity's toml; neither is a config file
    entity_tomls = tuple(f"# {kind.value}/" for kind in CONFIG_FILES)
    configs = [b for b in blocks if "project" not in tomllib.loads(b) and not b.startswith(entity_tomls)]
    return configs + [(_REPO / ".agentenv" / "config.example.toml").read_text()]


@pytest.mark.parametrize("body", _documented_configs())
def test_every_documented_config_warns_about_nothing(tmp_path, monkeypatch, body):
    assert _warnings(tmp_path, monkeypatch, body) == []


def test_show_prints_it_with_the_other_warnings_and_still_exits_0(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[sanbox]\ndefault = "modal"\n')
    warning = "agent-env does not read [sanbox]: it has no such table. Did you mean [sandbox]?"

    assert f"(warning) {warning}" in render(describe_config())
    result = CliRunner().invoke(config_group, ["show", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["warnings"] == [warning]


@pytest.mark.parametrize("body, warning", [
    ('default = "modal"\n[sandbox]\nagent_default = "local"\n',
     "The top-level key 'default' is outside every table, and agent-env does not read it; "
     "[model] and [sandbox] take a key of that name."),
    ("version = 1\n", "The top-level key 'version' is outside every table, and agent-env does not read it."),
    # [plugins] takes only a table, so a value is not offered it
    ("plugs = 1\n", "The top-level key 'plugs' is outside every table, and agent-env does not read it."),
])
def test_a_key_above_every_table_is_named_as_a_key(tmp_path, monkeypatch, body, warning):
    assert _warnings(tmp_path, monkeypatch, body) == [warning]


class _Recorder(dict):
    """A config table that records each key a reader asks it for, through copies too."""

    def __init__(self, data, path, log):
        super().__init__({k: _Recorder(v, path + (k,), log) if isinstance(v, dict) else v
                          for k, v in dict.items(data)})
        self._path, self._log = path, log

    def get(self, key, default=None):
        self._log.add(self._path + (key,))
        return super().get(key, default)

    def __getitem__(self, key):
        self._log.add(self._path + (key,))
        return super().__getitem__(key)

    def __contains__(self, key):
        self._log.add(self._path + (key,))
        return super().__contains__(key)

    def __deepcopy__(self, memo):
        return _Recorder({k: copy.deepcopy(v, memo) for k, v in dict.items(self)}, self._path, self._log)


def _declared(path: tuple[str, ...]) -> bool:
    """Whether the key at ``path`` is declared, or lies in a table whose keys are free-form."""
    shape = describe._KNOWN
    for part in path:
        if isinstance(shape, describe.Named):
            shape = shape.shape
            continue
        if not isinstance(shape, dict):
            return True   # a value, or a free-form table
        if part not in shape:
            return False
        shape = shape[part]
    return True


def test_every_key_a_reader_asks_for_is_declared(monkeypatch):
    """The other direction from the declared-key test: a reader that starts taking a key nobody
    declared would make every valid config using it warn."""
    log: set = set()
    config = Config()
    config._snapshot = snapshot.Snapshot(path=None, _document=_Recorder(tomllib.loads(EVERY_KEY_TOML), (), log))
    monkeypatch.setattr(runtime, "get_config", lambda: config)
    readers = [config.env_registry, config.task_step_registry, config.artifact_registry, config.sandbox_registry,
               config.state_registry, sandbox_provider._default_sandbox_spec, sandbox_provider._agent_sandbox_spec,
               lambda: sandbox_provider.apply_default_attribution({}), explorer_app.explorer_settings,
               lambda: explorer_plugin.load_plugins(source=config)]
    for read in readers:
        try:
            read()
        except Exception:
            pass   # the fixture's classes do not exist; the keys asked before that are recorded

    assert log, "no reader asked the recording document for anything"
    assert sorted(path for path in log if path[0] in describe._TOP_LEVEL_SECTIONS and not _declared(path)) == []
