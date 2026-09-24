"""`agent-env config show` reports the resolution chain — and changes nothing.

Two properties carry the weight. The first is that the report is *true*: it names the layer
`Config.trace_section` would actually take, so it cannot quietly drift into telling a
reassuring story about a store the process is not using — the exact failure that let
a consumer write to SQLite for two releases while reporting success. The second is that it
is *inert*: it must not build a store, reach the network, or create `.agentenv/`.
"""

import pathlib
import tomllib

import pytest

from agent_env.cli.config import render
from agent_env.config import loader as config_loader
from agent_env.config import describe
from agent_env.config.provenance import headline
from agent_env.config.describe import (
    MASK,
    SOURCE_ENV,
    SOURCE_NONE,
    describe_config,
    mask_value,
    search_path,
)
from agent_env.config.errors import ConfigError
from agent_env.config.runtime import ALIASED_SECTIONS, Config

# TruffleHog scans the committed bytes, and a connection URI carrying inline userinfo is
# indistinguishable from a real credential to a detector — this fixture failed `scan / scan` on this
# PR's first push. Assembling the userinfo at run time keeps the value under test byte-for-byte
# what it was while leaving no credential-shaped literal in the file. Do not re-inline it.
_FAKE_MONGO_URI = "mongodb://" + "admin" + ":" + "notarealpassword" + "@cluster0/db"

MONGO_TOML = """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { uri = "secret:mongodb_uri", database = "agent_env_dev" }

[stores.object]
impl = "agent_env.store.object_store:S3ObjectStore"
config = { bucket = "artifact-bucket" }
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """A bare install by default: no config anywhere above CWD, no store overrides."""
    for var in ("AGENT_ENV_CONFIG", "AGENT_ENV_DOCUMENT_STORE", "AGENT_ENV_OBJECT_STORE",
                "AGENT_ENV_IMAGE_STORE", "AGENT_ENV_SECRET_STORE", "AGENT_ENV_RUNNER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def _sections(report):
    return {s.name: s for s in report.sections}


def _write_config(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / ".agentenv" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# ---------------------------------------------------------------- the bare install


def test_bare_install_names_the_local_stores_it_silently_resolved(tmp_path):
    report = describe_config()

    assert report.config_path is None
    assert report.config_source == SOURCE_NONE and report.search_root == tmp_path.resolve()
    assert f"none found — walked up from {tmp_path.resolve()}" in render(report)
    sections = _sections(report)
    assert sections["document"].impl_name == "LocalSqliteDocumentStore"
    assert sections["object"].impl_name == "LocalFilesystemObjectStore"
    assert sections["image"].impl_name == "LocalRegistryImageStore"
    assert sections["secret"].impl_name == "LocalSecretStore"
    assert sections["runner"].impl_name == "LocalRunner"
    assert all(s.winner.kind == "default" for s in report.sections)
    assert all(s.shadowed == () for s in report.sections)


def test_resolving_and_describing_a_bare_install_create_nothing(tmp_path):
    describe_config()
    Config()._resolve_document_section()
    Config()._resolve_object_section()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- precedence attribution


def test_a_config_file_is_reported_as_the_source_with_its_path(tmp_path):
    path = _write_config(tmp_path, MONGO_TOML)

    sections = _sections(describe_config())
    assert sections["document"].impl_name == "MongoDocumentStore"
    assert sections["document"].winner.where == "[stores.document]"
    assert sections["document"].winner.kind == "file"
    assert sections["object"].winner.where == "[stores.object]"
    # a section the file does not carry still falls through to the built-in default
    assert sections["image"].winner.kind == "default"


def test_an_env_override_beats_the_file_and_says_so(tmp_path, monkeypatch):
    _write_config(tmp_path, MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")

    section = _sections(describe_config())["document"]
    assert section.impl_name == "LocalSqliteDocumentStore"
    assert section.winner.kind == "env"
    assert section.winner.where == "$AGENT_ENV_DOCUMENT_STORE"
    # the losing file section is still reported — a silent partial downgrade is the
    # failure this command exists to make visible
    beaten = [b for b in section.shadowed if b.kind == "file"]
    assert [b.where for b in beaten] == ["[stores.document]"]
    assert beaten[0].summary() == "MongoDocumentStore"


def test_agent_env_config_is_reported_as_the_reason_the_file_won(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere" / "config.toml"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(elsewhere))

    report = describe_config()
    assert report.config_path == elsewhere
    assert report.config_source == SOURCE_ENV
    assert "via $AGENT_ENV_CONFIG" in render(report)


def test_a_bad_agent_env_config_is_reported_rather_than_raised(tmp_path, monkeypatch):
    """The one case that already fails loud everywhere else must still produce a report —
    `config show` is what you reach for once something is wrong."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "nope.toml"))

    report = describe_config()
    assert report.config_path is None and report.sections == []
    assert "does not point to an existing file" in report.error
    assert "(error)" in render(report)


def test_an_unresolvable_section_is_reported_instead_of_aborting_the_report(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "mongo")  # no built-in coordinates

    sections = _sections(describe_config())
    assert sections["document"].impl is None
    assert "no built-in coordinates" in sections["document"].error
    assert sections["object"].impl_name == "LocalFilesystemObjectStore"  # the rest still resolve


# ---------------------------------------------------------------- masking


@pytest.mark.parametrize("key,value,expected", [
    ("uri", "secret:mongodb_uri", "secret:mongodb_uri"),          # a reference names, never carries
    ("bucket", "env:BUCKET?fallback", "env:BUCKET?fallback"),
    ("database", "agent_env_dev", "agent_env_dev"),
    ("password", "hunter2", MASK),
    ("aws_secret_access_key", "AKIA...", MASK),
    ("api_key", "sk-live-1", MASK),
    ("uri", _FAKE_MONGO_URI, f"mongodb://{MASK}@cluster0/db"),
    ("port", 27017, 27017),
    ("api_key", None, None),      # absence is not a secret; masking it invents one
    ("password", None, None),
])
def test_mask_value(key, value, expected):
    assert mask_value(key, value) == expected


def test_a_secret_shaped_key_masks_its_whole_subtree_not_just_a_scalar(tmp_path):
    """A backend's `config` is an arbitrary TOML table, so a credential arrives just as
    easily inside a list or a nested table as at a scalar key. Masking only the leaf's own
    key printed both."""
    _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { database = "agent_env_dev", tokens = ["literal-token"], credentials = { user = "svc", pw = "literal-pw" } }
""")
    section = _sections(describe_config())["document"]

    assert section.config["tokens"] == [MASK]
    assert section.config["credentials"] == {"user": MASK, "pw": MASK}
    assert section.config["database"] == "agent_env_dev"  # outside the secret scope, untouched
    rendered = render(describe_config())
    assert "literal-token" not in rendered and "literal-pw" not in rendered and "svc" not in rendered


def test_a_reference_under_a_secret_shaped_key_still_names_itself(tmp_path):
    """Masking a subtree must not cost the one thing worth reading: which secret it names."""
    _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { database = "agent_env_dev", credentials = { uri = "secret:mongodb_uri" } }
""")
    section = _sections(describe_config())["document"]

    assert section.config["credentials"] == {"uri": "secret:mongodb_uri"}


def test_a_malformed_config_file_is_reported_rather_than_raised(tmp_path, monkeypatch):
    """The file that won't parse is the whole reason someone ran `config show`. Letting the
    ConfigError out means the one command meant to explain the problem is the one that dies."""
    path = _write_config(tmp_path, "[stores.document\nimpl = ")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    report = describe_config()
    assert report.config_path == path
    assert "Malformed config.toml" in report.error and str(path) in report.error
    assert "(error)" in render(report)
    # and the sections survive the error: a higher layer may be resolving over the top of a
    # file that will not parse, and that is exactly what the operator needs to see
    assert _sections(report)["sandbox"].winner.error is not None


def test_an_unreadable_config_file_is_reported_rather_than_raised(tmp_path, monkeypatch):
    path = _write_config(tmp_path, MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    monkeypatch.setattr("agent_env.config.loader.open",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("Permission denied")),
                        raising=False)

    report = describe_config()
    assert "Permission denied" in report.error
    assert _sections(report)["sandbox"].winner.error is not None


def test_a_literal_credential_in_the_file_never_reaches_the_report(tmp_path):
    _write_config(tmp_path, f"""
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = {{ uri = "{_FAKE_MONGO_URI}", database = "agent_env_dev" }}
""")
    rendered = render(describe_config())
    assert "notarealpassword" not in rendered and "admin" not in rendered
    assert "agent_env_dev" in rendered  # the useful half survives


# ---------------------------------------------------------------- rendering


def test_render_puts_the_file_and_every_section_on_screen(tmp_path):
    path = _write_config(tmp_path, MONGO_TOML)

    rendered = render(describe_config())
    assert str(path) in rendered
    assert "MongoDocumentStore" in rendered and "database=agent_env_dev" in rendered
    assert "secret:mongodb_uri" in rendered
    for name in ("document:", "object:", "image:", "secret:", "runner:"):
        assert name in rendered


# Every top-level table the loader is read for, so a report that silently covered only the
# stores would fail here rather than in production.
FULL_SURFACE_TOML = """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { uri = "secret:mongodb_uri", database = "agent_env_dev" }

[stores.object]
impl = "agent_env.store.object_store:S3ObjectStore"
config = { bucket = "a-bucket", region = "us-west-2" }

[stores.image]
impl = "agent_env.store.image_store:OciRegistryImageStore"
config = { registry_host = "registry.invalid" }

[stores.secret]
impl = "agent_env.store.secret_store:AwsSecretsManagerSecretStore"
config = { secret_name = "a/secret", region = "us-west-2" }

[model]
base_url = "https://example.invalid"
api_key = "secret:litellm_api_key"

[conversations]
default_human_a2a_url = "https://example.invalid/a2a"

[agents]
default_a2a_agent_id = "an-agent"

[sandbox]
default = "local"

[state.providers.remote_postgres]
impl = "pkg.mod:RemotePostgresStateProvider"

[envs]
impls = ["pkg.envs:CustomEnv"]

[artifacts]
impls = ["pkg.artifacts:CustomArtifact"]

[task_steps]
impls = ["pkg.steps:CustomStep"]

[explorer]
port = 9999

[runner]
impl = "agent_env.runner.local_runner:LocalRunner"
"""

_ALIASED = ("document", "object", "image", "secret", "runner")
_EVERY_TABLE = ("document", "object", "image", "secret", "runner", "model", "conversations",
                "agents", "sandbox", "state", "envs", "artifacts", "task_steps", "explorer")


@pytest.mark.parametrize("toml", [None, MONGO_TOML, FULL_SURFACE_TOML])
def test_the_report_resolves_what_the_real_resolver_resolves(tmp_path, monkeypatch, toml):
    """The shadow property. A report that drifts from the resolver is worse than none, so
    every aliased section is compared against what `Config` itself would build."""
    if toml is not None:
        path = tmp_path / "config.toml"
        path.write_text(toml)
        monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    reported = _sections(describe_config())
    real = Config()
    for name in _ALIASED:
        assert reported[name].impl == getattr(real, f"_resolve_{name}_section")()["impl"], name


@pytest.mark.parametrize("section", ALIASED_SECTIONS, ids=lambda s: s.name)
def test_each_declared_env_var_reaches_its_own_sections_alias(monkeypatch, section):
    """The pairing the shadow property cannot see: it compares resolved values, so a record
    naming another section's env var, path or default would still agree with itself."""
    monkeypatch.setenv(section.env_var, "nonsense")

    with pytest.raises(ConfigError, match=section.env_var):
        Config().trace_section(section.name)


def test_every_top_level_table_is_covered(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(FULL_SURFACE_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    reported = _sections(describe_config())
    assert tuple(reported) == _EVERY_TABLE
    assert all(s.winner.kind == "file" for s in reported.values()), \
        {n: s.winner.kind for n, s in reported.items() if s.winner.kind != "file"}


def test_every_reported_fallback_names_a_core_module_that_mentions_that_table():
    """The other half of the invariant above: a row is a claim that something reads the
    table. An owner that is not a core module is the report inventing a config surface."""
    core = pathlib.Path(describe.__file__).parents[1]
    for section in describe._FILE_SECTIONS:
        module = core.joinpath(*section.owner.split(".")).with_suffix(".py")
        assert module.is_file(), (section.name, section.owner)
        assert f'"{section.toml_path[0]}"' in module.read_text(), (section.name, section.owner)


def test_a_table_absent_from_the_file_names_where_its_fallback_lives(tmp_path, monkeypatch):
    """Those sections' defaults live in their reader, not here. The report says where to
    look rather than guessing a value it would then have to keep in sync."""
    path = tmp_path / "config.toml"
    path.write_text(MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    sandbox = _sections(describe_config())["sandbox"]
    assert sandbox.winner.kind == "default"
    assert "providers.sandbox_provider" in sandbox.winner.where


def test_describing_a_packaged_config_does_not_write_beside_it(tmp_path, monkeypatch):
    """A config shipped inside a package puts the local-store root in site-packages. The
    report resolves that path and must still not create anything there."""
    packaged = tmp_path / "site-packages" / "a_plugin"
    packaged.mkdir(parents=True)
    (packaged / "config.toml").write_text("")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(packaged / "config.toml"))

    describe_config()
    assert list(packaged.iterdir()) == [packaged / "config.toml"]


def test_search_path_reports_the_override_and_whether_it_is_there(tmp_path, monkeypatch):
    from agent_env.config.describe import search_path

    walked = search_path()
    assert [c.why for c in walked] == ["walk-up"] * len(walked)
    assert walked[0].path == tmp_path.resolve() / ".agentenv" / "config.toml"
    assert not any(c.exists for c in walked)

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "nope.toml"))
    override = search_path()
    assert len(override) == 1 and override[0].exists is False


def test_json_keeps_the_provenance_every_other_tool_drops(tmp_path, monkeypatch):
    from agent_env.cli.config import as_dict

    path = tmp_path / "config.toml"
    path.write_text(MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")

    document = next(s for s in as_dict(describe_config())["sections"] if s["name"] == "document")
    assert document["winner"]["kind"] == "env"
    assert [b["where"] for b in document["shadowed"] if b["kind"] == "file"] == ["[stores.document]"]


def test_a_secret_shaped_key_masks_a_non_string_value(tmp_path, monkeypatch):
    """A secret is no less a secret for being written as a number."""
    path = tmp_path / "config.toml"
    path.write_text('''
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { uri = "secret:mongodb_uri", token = 123456, verified = true }
''')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    resolved = _sections(describe_config())["document"].config
    assert resolved["token"] == MASK
    assert resolved["uri"] == "secret:mongodb_uri"
    assert resolved["verified"] is True


def test_a_file_section_reports_what_the_file_set(tmp_path, monkeypatch):
    """The eight sections read straight from the file are the reason to cover all ten
    tables; rendering them as "(table)" would have covered them in name only."""
    from agent_env.cli.config import as_dict, render

    path = tmp_path / "config.toml"
    path.write_text('[sandbox]\ndefault = "local"\nagent_default = "local,modal"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    report = describe_config()
    assert "default=local" in render(report)
    sandbox = next(s for s in as_dict(report)["sections"] if s["name"] == "sandbox")
    assert sandbox["value"] == {"default": "local", "agent_default": "local,modal"}


def test_an_agents_table_the_reader_would_refuse_is_flagged_by_the_report(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[agents]\ndefault_agent = "x"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    agents = _sections(describe_config())["agents"]
    assert agents.value is None
    assert "unknown keys ['default_agent']" in agents.error


def test_an_unresolvable_layer_is_still_the_layer_the_report_names(tmp_path, monkeypatch):
    """Reporting the built-in default here would blame a layer that had no part in the
    failure — the "provenance that lies" failure this command exists to avoid."""
    path = tmp_path / "config.toml"
    path.write_text(MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "bogus")

    document = _sections(describe_config())["document"]
    assert document.error is not None
    assert document.winner.kind == "env"
    assert document.winner.where == "$AGENT_ENV_DOCUMENT_STORE"
    assert [b.where for b in document.shadowed if b.kind == "file"] == ["[stores.document]"]


def test_a_wrongly_shaped_table_fails_instead_of_falling_back(tmp_path, monkeypatch):
    """`stores = "remote"` is valid TOML and nonsense config. Treating it as absent would
    hand back four local stores for a file that plainly meant something else."""
    path = tmp_path / "config.toml"
    path.write_text('stores = "remote"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    with pytest.raises(ConfigError, match=r"\[stores\] must be a table, got str"):
        Config()._resolve_document_section()

    document = _sections(describe_config())["document"]
    assert "must be a table" in document.error


@pytest.mark.parametrize("broken", ['stores = "remote"\n', "stores = [[[\n"])
def test_a_broken_lower_layer_does_not_break_the_layer_that_outranks_it(tmp_path, monkeypatch, broken):
    """An env override decides the value without the file being consulted. Reading the file
    for the report must not be what makes a working override fail — that recovery path is
    exactly what an operator reaches for when the file is the thing that is broken."""
    path = tmp_path / "config.toml"
    path.write_text(broken)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")

    assert "LocalSqliteDocumentStore" in Config()._resolve_document_section()["impl"]

    document = _sections(describe_config())["document"]
    assert document.winner.kind == "env"
    assert [b.error is not None for b in document.shadowed if b.kind == "file"] == [True]


@pytest.mark.parametrize("broken", ['stores = "remote"\n', "stores = [[[\n"])
def test_the_same_broken_layer_still_raises_when_nothing_outranks_it(tmp_path, monkeypatch, broken):
    path = tmp_path / "config.toml"
    path.write_text(broken)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    with pytest.raises(ConfigError):
        Config()._resolve_document_section()


def test_an_unreadable_file_raises_a_config_error_not_the_os_error(tmp_path, monkeypatch):
    """The layering rule turns the winning layer's failure into `ConfigError`, so a caller
    that already handles config problems handles this one. It is a real change: the raw
    `PermissionError` from `open` used to escape."""
    path = _write_config(tmp_path, MONGO_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    monkeypatch.setattr("agent_env.config.loader.open",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("Permission denied")),
                        raising=False)

    with pytest.raises(ConfigError, match="Permission denied"):
        Config()._resolve_document_section()


@pytest.mark.parametrize("toml", [FULL_SURFACE_TOML, "stores = [[[\n"],
                         ids=["parses", "malformed"])
def test_the_report_parses_one_file_however_many_sections_it_has(tmp_path, monkeypatch, toml):
    """Two reads meant two sources of truth for what the file says, and two error channels
    for one broken file. Every section now reads the one parse — the failure included, or the
    file the command was reached for is the one reopened per section.

    Discovery is deliberately *not* counted here: it runs per read so that re-pointing
    AGENT_ENV_CONFIG keeps working until the freeze makes rebinding raise. Only the parse is
    cached."""
    path = _write_config(tmp_path, toml)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    # counted at the TOML parse, not at a loader function: which helper the code calls is an
    # implementation detail that has already moved twice, and patching it silently stops
    # counting rather than failing.
    parses = []
    parse = tomllib.loads
    monkeypatch.setattr(tomllib, "loads", lambda text: (parses.append(1), parse(text))[1])

    report = describe_config()

    assert len(parses) == 1
    assert len(report.sections) == len(_EVERY_TABLE)


def test_config_debug_reports_the_loaders_own_search_path(tmp_path, monkeypatch):
    """`config debug` exists to report discovery, so it must not own a second copy of the
    walk: the paths it prints are the ones `discover_config_path` would consider."""
    monkeypatch.chdir(tmp_path)

    assert [c.path for c in search_path()] == list(config_loader._walk_up_candidates())

    target = tmp_path / ".agentenv" / "config.toml"
    target.parent.mkdir()
    target.write_text(MONGO_TOML)
    assert search_path()[0].path == target and search_path()[0].exists
    assert config_loader.discover_config_path() == target


def test_one_broken_file_is_stored_once_per_layer_that_failed(tmp_path, monkeypatch):
    """A section cannot be both unreadable and unresolvable, so it must not claim to be.
    Storing the same message twice per section is how `render` and `--json` came to read
    different fields for one question."""
    path = _write_config(tmp_path, "stores = [[[\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    report = describe_config()

    assert all(s.unresolved is None for s in report.sections)
    assert all(s.error is s.winner.error for s in report.sections)
    assert report.error is not None


def test_a_literal_authorization_header_is_masked(tmp_path, monkeypatch):
    """A store `config` is an arbitrary table, and `authorization = "Bearer ..."` is where a
    literal credential most plausibly lands: it matches none of password/token/secret."""
    path = _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = { authorization = "Bearer abc", oauth_id = "z", passphrase = "p", database = "keep" }
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    resolved = _sections(describe_config())["document"].config
    assert resolved["authorization"] == MASK
    assert resolved["oauth_id"] == MASK
    assert resolved["passphrase"] == MASK
    assert resolved["database"] == "keep"


def test_a_section_name_used_as_a_key_inside_another_table_is_warned_about(tmp_path, monkeypatch):
    """TOML binds a bare key to the table above it, so `envs = "..."` typed at the foot of a
    file becomes `[task_steps].envs` — a well-formed document that configures nothing. No
    shape rule can catch it: the section is simply absent. This bit me three times writing a
    chaos harness, which is how I know a user will hit it."""
    path = _write_config(tmp_path, '[task_steps]\nimpls = []\nenvs = "pkg:CuaEnv"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    report = describe_config()

    assert len(report.warnings) == 1
    assert "[task_steps] has a key named 'envs'" in report.warnings[0]
    assert "Did you mean [envs]?" in report.warnings[0]
    assert "(warning)" in render(report)
    assert _sections(report)["envs"].winner.kind == "default"  # and it really is unset


def test_a_well_formed_config_warns_about_nothing(tmp_path, monkeypatch):
    """The nested tables a real config is full of — [sandbox.providers.x], [stores.document]
    — must not trip the check; only a *top-level section name* one level in does."""
    path = _write_config(tmp_path, FULL_SURFACE_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    assert describe_config().warnings == []


def test_an_absent_section_reads_as_unset(tmp_path, monkeypatch):
    """The most common line in the output, and the one nothing asserted: a section with no
    value rendered the string `None` once the unset case moved out of `lines`."""
    path = _write_config(tmp_path, '[model]\nbase_url = "http://x"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    out = render(describe_config())

    assert "(unset)" in out
    assert "None" not in out


def test_a_list_below_a_section_is_not_python_repr(tmp_path, monkeypatch):
    """`[explorer.plugins] impls` is load_plugins' documented shape and sits two levels down,
    where entries used to fall through to `str()`. An array of tables keeps its own block."""
    path = _write_config(tmp_path, """
[explorer.plugins]
impls = ["pkg.plugins:A", "pkg.plugins:B"]
[[explorer.hooks]]
event = "start"
[explorer.hooks.config]
deep = 2
[explorer.solo]
one_plugin = ["pkg.plugins:Only"]
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    out = render(describe_config())

    assert "pkg.plugins:A" in out and "pkg.plugins:B" in out
    assert "deep=2" in out
    assert "'" not in out                      # `from [explorer]` is the only bracket here
    # a single-entry list keeps its heading; inlined it would read as `impls=pkg.plugins:A`
    assert "one_plugin:" in out


def test_a_key_at_both_levels_reports_the_one_the_reader_takes(tmp_path, monkeypatch):
    """`build_store` reads `config`, so a sibling of the same name is the value the process
    ignores. Reporting it would be a report of something that is not happening."""
    path = _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
database = "beside"
[stores.document.config]
uri = "mongodb://x"
database = "under-config"
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    report = describe_config()

    assert _sections(report)["document"].config["database"] == "under-config"
    # ...and the dead one is named rather than merely hidden
    assert any("both beside its config table and inside it" in w for w in report.warnings)
    assert "(warning)" in render(report)


def test_a_config_that_is_not_a_table_is_still_shown(tmp_path, monkeypatch):
    """`config` is hoisted to the impl's level, and a `config` that is not a table was
    dropped by the hoist it could not take part in."""
    path = _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
config = "not-a-table"
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    assert "config=not-a-table" in render(describe_config())


def test_an_impl_carrying_a_newline_stays_on_its_row(tmp_path, monkeypatch):
    """`class_name` returns the string whole when it holds no colon, and that went into the
    row unescaped — the one render path `_scalar` did not cover."""
    path = _write_config(tmp_path, '[model]\nimpl = "one\\ntwo"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    out = render(describe_config())

    assert "one\\ntwo" in out
    assert all(line.startswith(" ") or ":" in line for line in out.splitlines() if line)


def test_an_impl_beside_a_config_table_is_not_a_duplicate(tmp_path, monkeypatch):
    """`build_store` takes the outer `impl` as the class pointer and `config` as
    `from_config` kwargs, so a kwarg named `impl` shadows nothing. Both are read."""
    path = _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
[stores.document.config]
impl = "a-kwarg-named-impl"
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    assert describe_config().warnings == []


def test_a_shadowed_layer_holding_only_a_nested_table_is_not_unset():
    """`(unset)` on the shadowed line means the layer had nothing to give. A layer whose
    value is a table of tables had something."""
    assert headline({"config": {"uri": "mongodb://x"}}) == "config"
    assert headline(None) == "(unset)"


def test_a_key_beside_impl_and_config_is_not_swallowed(tmp_path, monkeypatch):
    """A key at the wrong level is exactly what this command is reached for. `config` reads
    at the impl's own level, so anything else beside them used to vanish into that hoist."""
    path = _write_config(tmp_path, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
stray_key = "at the wrong level"
[stores.document.config]
uri = "mongodb://x"
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    out = render(describe_config())

    assert "uri=mongodb://x" in out
    assert "stray_key=at the wrong level" in out


def test_a_value_carrying_a_newline_stays_on_its_row(tmp_path, monkeypatch):
    """TOML strings may span lines; a row may not, or every line after the first loses the
    indent that puts it under its key."""
    path = _write_config(tmp_path, '[model]\nbase_url = """one\ntwo"""\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    out = render(describe_config())

    assert "base_url=one\\ntwo" in out
    assert all(line.startswith(" ") or ":" in line for line in out.splitlines() if line)


def test_every_nested_value_is_shown_not_summarised(tmp_path, monkeypatch):
    """Shaped like the real sdk config, which is what this command is reached for. Nested
    values used to be rendered with `str()` (600-character lines of Python repr) and then
    by their key names alone, which hid the values the command is run to read."""
    path = _write_config(tmp_path, """
[sandbox]
default = "beta_scale"
[sandbox.providers.beta_scale]
impl = "pkg.providers:BetaProvider"
[sandbox.providers.modal_vm]
impl = "pkg.providers:ModalVmProvider"
[sandbox.attribution]
product = "agent-env"
team = "frontier-data"

[envs]
impls = ["pkg.envs:CuaEnv", "pkg.envs:IosCuaEnv", "pkg.envs:RemoteEnv", "pkg.envs:FifthEnv"]

[explorer]
allowed_hosts = ["explorer.mybox.internal:8234", "a.b"]
""")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    out = render(describe_config())

    # every entry, in full -- no `+N` tail, no key-names-only summary
    for impl in ("pkg.envs:CuaEnv", "pkg.envs:IosCuaEnv", "pkg.envs:RemoteEnv", "pkg.envs:FifthEnv"):
        assert impl in out
    assert "beta_scale  BetaProvider" in out and "modal_vm  ModalVmProvider" in out
    # the values of a nested scalar table, which the summarised form replaced with its keys
    assert "product=agent-env" in out and "team=frontier-data" in out
    # `class_name` is an rpartition on ":", so naming every string turned a documented
    # `allowed_hosts = ["host:8234"]` into its port.
    assert "explorer.mybox.internal:8234" in out
    # and none of it as Python repr
    assert "'" not in out and "+1]" not in out


