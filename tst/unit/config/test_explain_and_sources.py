"""`agent-env config explain` and `config sources` — the two provenance questions `show`
does not answer.

`show` reports every section at once; `explain` answers "where did *this* value come from"
for one path, and `sources` answers "which layers are in play at all" — including the ones
contributing nothing, which is what tells an operator the file they edited is not being
read. Both inherit `show`'s two properties: the report names the layer the resolver would
actually take, and inspecting creates nothing.
"""

import ast
import json
import pathlib

import pytest
from click.testing import CliRunner

import agent_env

from agent_env.cli.config import config as config_group
from agent_env.cli.config import render_explain
from agent_env.config import loader as config_loader
from agent_env.config import snapshot
from agent_env.config.errors import ConfigError
from agent_env.config.paths import state_root
from agent_env.config.describe import (
    env_sources,
    MASK,
    SOURCE_ENV,
    SOURCE_INSTALLED,
    SectionsReport,
    describe_config,
    explain_path,
    sources,
)
from agent_env.config.runtime import Config
from agent_env.config.provenance import KIND_DEFAULT, KIND_ENV, KIND_FILE, KIND_INSTALLED

CONFIG = """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { uri = "secret:mongodb_uri", database = "agent_env_dev" }

[model]
provider = "litellm"
api_key = "hunter2"

[plugins.agentenv-demo]
timeout = 30
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every declared source, unset.

    These tests assert what the report says *about the environment*, so inheriting the
    runner's would make them assert the machine instead of the behaviour — and CI does set
    some of these, which is exactly how this file first went red there and not locally.
    """
    for source in env_sources():
        monkeypatch.delenv(source.name, raising=False)
    monkeypatch.delenv(config_loader.ENV_CONFIG_PATH, raising=False)


@pytest.fixture
def config_file(tmp_path, monkeypatch, clean_env):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))
    return path


def test_a_section_is_explained_by_its_toml_path_or_its_name(config_file):
    # An operator reads `stores.document` off the file; `document` is the internal name.
    by_path, by_name = explain_path("stores.document"), explain_path("document")
    assert by_path.section == by_name.section == "document"
    assert by_path.value == by_name.value
    assert by_path.winner.kind == KIND_FILE


def test_explain_names_the_env_var_that_beat_the_file(config_file, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    report = explain_path("stores.document")
    assert report.winner.kind == KIND_ENV
    assert report.winner.where == "$AGENT_ENV_DOCUMENT_STORE"
    # and the file it beat is still named, or the operator cannot see what was displaced
    assert any(layer.kind == KIND_FILE for layer in report.shadowed)


def test_explain_masks_a_secret_shaped_leaf(config_file):
    assert explain_path("model.api_key").value == MASK


def test_a_path_no_layer_supplies_is_unset_not_attributed_to_a_default(config_file):
    # Claiming a built-in default supplied it would be the report inventing a layer.
    report = explain_path("nothing.here")
    assert report.winner is None and report.value is None and report.error is None
    assert "no layer supplies this path" in render_explain(report)


def test_explain_agrees_with_the_resolver_about_an_unresolvable_section(config_file, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "s3")
    report = explain_path("stores.object")
    assert report.error is not None                    # reported, not raised
    assert report.winner.where == "$AGENT_ENV_OBJECT_STORE"


def test_an_empty_path_is_refused(config_file):
    assert explain_path("").error == "empty path"


def test_sources_lists_every_layer_lowest_precedence_first(config_file):
    layers = sources()
    assert layers[0].kind == KIND_DEFAULT and layers[0].present
    assert layers[1].kind == KIND_FILE and layers[1].present
    assert str(config_file) == layers[1].where
    assert all(s.kind == KIND_ENV for s in layers[2:])


def test_sources_lists_unset_env_layers_too(config_file):
    # The checklist is the point: an absent layer is why an edit is not taking effect.
    env = [s for s in sources() if s.kind == KIND_ENV]
    assert env and not any(s.present for s in env)
    assert "$AGENT_ENV_DOCUMENT_STORE" in {s.where for s in env}


def test_sources_reports_an_unreadable_file_rather_than_raising(tmp_path, monkeypatch):
    bad = tmp_path / "bad.toml"
    bad.write_text("not toml [[[")
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(bad))
    layer = next(s for s in sources() if s.kind == KIND_FILE)
    assert not layer.present and "unreadable" in layer.detail


def test_sources_masks_an_env_override(config_file, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_SECRET_STORE", "super-secret-value")
    layer = next(s for s in sources() if s.where == "$AGENT_ENV_SECRET_STORE")
    assert layer.present and layer.detail == MASK


def test_neither_command_creates_the_agentenv_dir_or_the_state_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config_loader.ENV_CONFIG_PATH, raising=False)
    sources()
    explain_path("stores.document")
    assert not (tmp_path / ".agentenv").exists()
    assert not state_root().exists()


def test_sources_says_how_the_file_was_found(config_file, tmp_path, monkeypatch):
    # $AGENT_ENV_CONFIG is terminal over a discovered file, so conflating the two would
    # hide which layers could still apply.
    assert "via $AGENT_ENV_CONFIG" in next(s for s in sources() if s.kind == KIND_FILE).detail

    monkeypatch.delenv(config_loader.ENV_CONFIG_PATH)
    (tmp_path / ".agentenv").mkdir()
    (tmp_path / ".agentenv" / "config.toml").write_text(CONFIG)
    monkeypatch.chdir(tmp_path)
    assert "walking up" in next(s for s in sources() if s.kind == KIND_FILE).detail


# Read for identity (USER), or set from [sandbox.providers.sail.config] for the Sail SDK to read
# (SAIL_*), never configuration agent-env takes, so none is a layer.
_NOT_CONFIGURATION = {"USER", "SAIL_API_KEY", "SAIL_RUNTIME_THREADS"}


def _env_vars_read_by(root):
    """Every environment variable name agent-env reads, resolved through module constants.

    The whole package, not just `agent_env.config`: the Modal provider reads
    `AGENT_ENV_MODAL_APP_NAME` from `providers/`, and a checklist scoped to the config
    package would call itself complete while omitting it.
    """
    found = set()
    for module in root.rglob("*.py"):
        tree = ast.parse(module.read_text())
        # `os.getenv(_ENV_API_KEY)` is as much a read as the literal; resolving the
        # constant is what stops a new variable hiding behind a name.
        consts = {t.id: n.value.value
                  for n in ast.walk(tree)
                  if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                  and isinstance(n.value.value, str)
                  for t in n.targets if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            arg = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "getenv" and node.args):
                arg = node.args[0]
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                  and node.value.attr == "environ"):
                arg = node.slice
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in consts:
                found.add(consts[arg.id])
    return found


def test_every_env_var_agent_env_reads_is_declared():
    """The checklist is only worth having if it is complete, and completeness decays the
    moment someone adds a read anywhere. Asserting it makes that a test failure rather
    than a silent hole in a report that promises to be exhaustive."""
    declared = {e.name for e in env_sources()} | {config_loader.ENV_CONFIG_PATH}
    found = _env_vars_read_by(pathlib.Path(agent_env.__file__).parent)
    assert found <= declared | _NOT_CONFIGURATION, \
        f"undeclared env sources: {sorted(found - declared - _NOT_CONFIGURATION)}"


def test_the_guard_resolves_a_variable_hidden_behind_a_module_constant():
    # LITELLM_API_KEY is only ever read as `os.getenv(_ENV_API_KEY)`, so a guard that
    # matched literals alone would never have seen it.
    assert "LITELLM_API_KEY" in _env_vars_read_by(pathlib.Path(agent_env.__file__).parent)


def test_an_ancestor_path_reports_its_sections_not_the_file_table(config_file, monkeypatch):
    # `stores` has no single winner: each section under it resolves on its own, and
    # answering from the file would show a table a higher layer already replaced.
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    report = explain_path("stores")
    # its own type: an ancestor has no single winner, so there is no field to read as None
    assert isinstance(report, SectionsReport)
    assert {c.path for c in report.children} >= {"stores.document", "stores.object"}
    doc = next(c for c in report.children if c.path == "stores.document")
    assert doc.winner.where == "$AGENT_ENV_DOCUMENT_STORE"
    assert "MongoDocumentStore" not in render_explain(report)


def test_unset_is_not_printed_above_the_table_it_denies(config_file):
    # `show` substitutes `(unset)` only when there is nothing else to print; this command
    # copied that logic and dropped the condition.
    rendered = render_explain(explain_path("model"))
    assert "(unset)" not in rendered and "provider=litellm" in rendered


def test_a_shadowed_scalar_is_masked_not_printed(config_file, monkeypatch):
    # `summary()` on a table prints the class name, but on a scalar it prints the value —
    # so a shadowed secret-shaped key reaches the terminal unless it is masked first.
    monkeypatch.setenv("LITELLM_API_KEY", "from-the-env")
    rendered = render_explain(explain_path("model.api_key"))
    assert "hunter2" not in rendered and "shadowed" in rendered


def _run(args, **env):
    return CliRunner().invoke(config_group, args, env=env, catch_exceptions=False)


def test_explain_json_carries_the_provenance(config_file):
    payload = json.loads(_run(["explain", "stores.document", "--json"]).output)
    assert payload["section"] == "document"
    assert payload["winner"]["kind"] == KIND_FILE
    assert payload["path"] == "stores.document"


def test_sources_json_is_ordered_and_names_what_each_shadows(config_file):
    payload = json.loads(_run(["sources", "--json"]).output)["sources"]
    assert [s["kind"] for s in payload[:2]] == [KIND_DEFAULT, KIND_FILE]
    # array order *is* precedence order; no numeric layer id is published
    assert all("order" not in s for s in payload)
    doc = next(s for s in payload if s["where"] == "$AGENT_ENV_DOCUMENT_STORE")
    assert doc["shadows"] == "stores.document"


def test_json_survives_a_toml_date(tmp_path, monkeypatch):
    # TOML has first-class dates, so this is a valid config file, not a malformed one.
    path = tmp_path / "config.toml"
    path.write_text("[model]\nreleased = 2026-09-22\n")
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))
    assert json.loads(_run(["explain", "model", "--json"]).output)["value"]["released"] \
        == "2026-09-22"
    assert json.loads(_run(["show", "--json"]).output)["config_path"] == str(path)


def test_an_env_reference_in_the_file_is_listed_as_a_source(tmp_path, monkeypatch):
    # `env:NAME` is not a precedence layer — it resolves inside the document that wrote it
    # — but it is still a variable supplying a value, and an operator hunting a missing
    # value needs to see it named.
    path = tmp_path / "config.toml"
    path.write_text('[model]\nbase_url = "env:PRIVATE_LITELLM_URL"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))

    row = next(s for s in sources() if s.where == "$PRIVATE_LITELLM_URL")
    assert not row.present and "unset" in row.detail
    assert "[model.base_url]" in row.detail
    # `shadows` means "overrides"; a reference supplies the path instead, so it claims none
    assert row.shadows is None

    monkeypatch.setenv("PRIVATE_LITELLM_URL", "https://x")
    assert next(s for s in sources() if s.where == "$PRIVATE_LITELLM_URL").present


def test_a_child_that_cannot_resolve_is_not_shown_as_merely_empty(tmp_path, monkeypatch):
    # `headline(None)` renders "(unset)", which would report a broken section as an
    # absent one — the two need different fixes. (An unimportable `impl` is deliberately
    # *not* this case: the report is inert, so it names the class without loading it.)
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "s3")   # a backend with no coordinates

    report = explain_path("stores")
    broken = next(c for c in report.children if c.path == "stores.object")
    assert broken.error is not None
    rendered = render_explain(report)
    assert "(unresolved)" in rendered
    payload = json.loads(_run(["explain", "stores", "--json"]).output)
    assert next(c for c in payload["children"] if c["path"] == "stores.object")["error"]


def test_a_variable_that_is_both_declared_and_referenced_gets_one_row(tmp_path, monkeypatch):
    # Two rows for one name reads as two different sources.
    path = tmp_path / "config.toml"
    path.write_text('[model]\napi_key = "env:LITELLM_API_KEY"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))

    rows = [s for s in sources() if s.where == "$LITELLM_API_KEY"]
    assert len(rows) == 1
    assert rows[0].shadows == "model.api_key"          # still the declared override
    assert "also referenced by [model.api_key]" in rows[0].detail


def test_every_path_referencing_a_variable_is_named(tmp_path, monkeypatch):
    # One variable feeding two keys is the coupling worth seeing; naming one hides the other.
    path = tmp_path / "config.toml"
    path.write_text('[model]\nbase_url = "env:SHARED"\n'
                    '[conversations]\ndefault_human_a2a_url = "env:SHARED"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))

    row = next(s for s in sources() if s.where == "$SHARED")
    assert "[model.base_url]" in row.detail
    assert "[conversations.default_human_a2a_url]" in row.detail
    # a reference supplies a path rather than overriding it, so it claims no `shadows`
    assert row.shadows is None


def test_a_given_config_is_reported_instead_of_a_fresh_resolution(tmp_path, monkeypatch):
    """A service configures itself at startup, so the fresh resolution a CLI wants is the
    wrong answer there — it describes a process that does not exist."""
    path = tmp_path / "config.toml"
    path.write_text('[agents]\ndefault_a2a_agent_id = "from-the-file"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))

    configured = Config(default_a2a_agent_id="installed-by-configure")

    # The assertion that matters, and the one an earlier version of this test omitted:
    # the report must agree with the getter, not with the file the getter never reaches.
    assert configured.get_default_a2a_agent_id() == "installed-by-configure"
    report = explain_path("agents.default_a2a_agent_id", configured)
    assert report.value == "installed-by-configure"
    assert report.winner.kind == KIND_INSTALLED
    assert "configure(default_a2a_agent_id=...)" in report.winner.where
    # the file is still named as the layer it beat
    assert any(b.raw == "from-the-file" for b in report.shadowed)

    # and with no Config supplied the fresh resolution is unchanged
    assert explain_path("agents.default_a2a_agent_id").value == "from-the-file"
    assert describe_config().config_path == path


def test_a_config_with_its_own_document_does_not_credit_the_env_var(tmp_path, monkeypatch):
    """Claiming `via $AGENT_ENV_CONFIG` for a path that variable does not name is the
    report inventing provenance — worse than admitting it cannot prove one."""
    other = tmp_path / "other.toml"
    other.write_text('[model]\nprovider = "installed"\n')
    (tmp_path / "config.toml").write_text('[model]\nprovider = "ambient"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(tmp_path / "config.toml"))

    installed = Config()
    installed._snapshot = snapshot.Snapshot(path=other, _document={"model": {"provider": "installed"}})

    report = describe_config(installed)
    assert report.config_path == other
    assert report.config_source == SOURCE_INSTALLED
    # discovery's own answer is still reported honestly when it is the one in play
    assert describe_config().config_source == SOURCE_ENV


def test_a_live_store_that_disagrees_with_the_config_is_warned_about(tmp_path):
    # `get_document_store()` returns an installed object verbatim, so a report naming the
    # file's backend would describe a store the process is not using.
    from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore

    config = Config()
    config.set_document_store(LocalSqliteDocumentStore(path=str(tmp_path / "d.db")))
    config._snapshot = snapshot.Snapshot(
        path=None,
        _document={"stores": {"document": {"impl": "agent_env.store.document_store:MongoDocumentStore"}}},
    )
    warning = next(w for w in describe_config(config).warnings if "stores.document" in w)
    assert "MongoDocumentStore" in warning and "LocalSqliteDocumentStore" in warning


def test_a_given_config_keeps_its_own_installed_document(tmp_path, monkeypatch):
    # The point of passing one: it answers for that object, not for the environment.
    other = tmp_path / "other.toml"
    other.write_text('[model]\nprovider = "installed"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(tmp_path / "config.toml"))
    (tmp_path / "config.toml").write_text('[model]\nprovider = "ambient"\n')

    installed = Config()
    installed._snapshot = snapshot.Snapshot(path=other, _document={"model": {"provider": "installed"}})

    assert explain_path("model", installed).value == {"provider": "installed"}
    assert explain_path("model").value == {"provider": "ambient"}


def test_the_json_shape_is_importable_without_the_cli():
    # A service serving this must not have to import a command module.
    import agent_env.config.describe as core
    assert callable(core.as_dict)


def test_importing_the_config_package_stays_light():
    """The RDS group is declared by the state provider, which pulls psycopg2 and the store
    layer. Reaching for it at module scope would put that on every config import, which is
    exactly what an earlier version of this did."""
    import subprocess, sys
    probe = ("import sys, agent_env.config.describe;"
             "print([m for m in ('pymongo','psycopg2') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_a_broken_lower_layer_does_not_beat_a_higher_one(tmp_path, monkeypatch):
    """A file the winner was never going to read must not be able to fail the read.

    The shape check for a section has to run when the file is what gets taken, not before:
    hoisting it above the candidate loop let a malformed `[conversations]` raise while
    `AGENT_ENV_HUMAN_A2A_URL` was set, and a malformed `[agents]` raise while
    `configure(default_a2a_agent_id=...)` was set.
    """
    path = tmp_path / "config.toml"
    path.write_text('conversations = "not-a-table"\nagents = "also-not-a-table"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))

    monkeypatch.setenv("AGENT_ENV_HUMAN_A2A_URL", "http://valid")
    assert Config().get_default_human_a2a_url() == "http://valid"

    configured = Config(default_a2a_agent_id="from-configure")
    assert configured.get_default_a2a_agent_id() == "from-configure"


def test_a_broken_section_still_raises_when_it_is_the_layer_being_taken(tmp_path, monkeypatch):
    # The other half: with no higher layer, the shape error is still the right answer.
    path = tmp_path / "config.toml"
    path.write_text('conversations = "not-a-table"\nagents = "also-not-a-table"\n')
    monkeypatch.setenv(config_loader.ENV_CONFIG_PATH, str(path))

    with pytest.raises(ConfigError, match=r"\[conversations\] must be a table"):
        Config().get_default_human_a2a_url()
    with pytest.raises(ConfigError, match=r"\[agents\] must be a table"):
        Config().get_default_a2a_agent_id()


def test_every_explain_response_carries_the_same_json_keys(config_file):
    """Splitting the types was for this module's clarity; a caller that indexes the keys
    should not be able to tell. An ancestor answered with a different key set breaks it."""
    shapes = {p: set(json.loads(_run(["explain", p, "--json"]).output))
              for p in ("stores", "stores.document", "model.api_key", "nothing.here",
                        "plugins", "plugins.agentenv-demo.timeout")}
    assert len(set(map(frozenset, shapes.values()))) == 1, shapes
    assert "children" in next(iter(shapes.values()))
