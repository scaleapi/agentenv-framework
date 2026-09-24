"""One document per process: the Config resolves it once, and nothing reads it privately.

Eight readers each called ``load_config_file(discover_config_path())`` for themselves, so a
process that built every registry parsed the same file six times. The waste is the smaller
half: nothing made those reads agree, and the provider registries memoize without keying on
the path they were built from, so one re-point of ``AGENT_ENV_CONFIG`` left them stale.

Measured at the TOML parse rather than at ``load_config_file``: the readers bound that name
at import, so patching the loader module misses every one of them — which is how a first
attempt at this measurement reported 1 parse on both sides.
"""

import ast
import pathlib
import threading
import time
import tomllib

import pytest

import agent_env
from agent_env.config import get_config, reset_config, snapshot
from agent_env.config.errors import ConfigError
from agent_env.config.runtime import Config

SRC = pathlib.Path(agent_env.__file__).parent
_PRIVATE_READS = {"load_config_file", "discover_config_path"}


def _forbidden_references(source: str) -> set:
    """Every way a module can reach the document readers, not just one.

    Matching imported *names* alone missed `from agent_env.config import loader` followed by
    `loader.load_config_file(...)` — the more idiomatic import, so the form a future reader
    would reach for first. Attribute access is checked too, which catches the call however
    the module was imported."""
    hits = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            hits |= {alias.asname or alias.name for alias in node.names} & _PRIVATE_READS
        elif isinstance(node, ast.Attribute) and node.attr in _PRIVATE_READS:
            hits.add(node.attr)
    return hits


def test_no_module_outside_the_config_package_reads_the_document_itself():
    """The scan that keeps one resolution single. A reader that discovers and parses for
    itself cannot be made to agree with the others, and cannot be frozen."""
    offenders = {}
    for path in sorted(SRC.rglob("*.py")):
        if path.relative_to(SRC).parts[0] == "config":
            continue
        hits = _forbidden_references(path.read_text())
        if hits:
            offenders[str(path.relative_to(SRC))] = sorted(hits)

    assert offenders == {}


def test_the_scan_catches_a_module_that_imports_the_loader_itself():
    """The bypass the first version of the scan missed. Without this, the guard that keeps
    the resolution single passes on the idiomatic way of defeating it."""
    bypass = (
        "from agent_env.config import loader\n"
        "def sneaky():\n"
        "    return loader.load_config_file(loader.discover_config_path()).get('sandbox', {})\n"
    )

    assert _forbidden_references(bypass) == {"load_config_file", "discover_config_path"}


def _config_at(tmp_path, body):
    path = tmp_path / ".agentenv" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(body)
    return path


def test_the_document_is_parsed_once_however_many_readers_ask(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, "[envs]\nimpls = []\n")))
    parses = []
    real = tomllib.loads
    monkeypatch.setattr(tomllib, "loads", lambda t: (parses.append(1), real(t))[1])

    for _ in range(5):
        assert get_config()._document().section("envs") == {"impls": []}

    assert len(parses) == 1


def test_resolving_again_after_a_reset_re_reads_the_file(tmp_path, monkeypatch):
    path = _config_at(tmp_path, '[sandbox]\ndefault = "local"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    assert get_config()._document().section("sandbox") == {"default": "local"}

    path.write_text('[sandbox]\ndefault = "modal"\n')
    assert get_config()._document().section("sandbox") == {"default": "local"}  # still the resolved one
    reset_config()
    assert get_config()._document().section("sandbox") == {"default": "modal"}


def test_a_file_that_will_not_parse_still_reports_its_path(tmp_path, monkeypatch):
    """A parse failure travels *on* the snapshot rather than replacing it. `config show`
    exists to name the file you are pointed at, and needs that name most when the file is the
    broken thing — so the path stays available and only a reader wanting the document is
    refused."""
    path = _config_at(tmp_path, "stores = [[[\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    resolved = get_config()._document()
    assert resolved.path == path
    assert resolved.error is not None and "Malformed" in str(resolved.error)

    # ...but asking for the document itself is refused. The first version of this test
    # asserted `section(...) == {}` here, which is the silent downgrade, not the contract.
    with pytest.raises(ConfigError, match="Malformed"):
        resolved.section("stores")


def _read_envs():
    from agent_env.env.registry import _merge_config_toml_envs
    _merge_config_toml_envs({})


def _read_artifacts():
    from agent_env.artifact.registry import _merge_config_toml_artifacts
    _merge_config_toml_artifacts({})


def _read_task_steps():
    from agent_env.task_step.registry import _merge_config_toml_steps
    _merge_config_toml_steps({})


def _read_sandbox():
    from agent_env.providers.sandbox_provider import _merge_config_toml_sandbox_providers
    _merge_config_toml_sandbox_providers({})


def _read_state():
    from agent_env.providers.state.env_state_provider import _merge_config_toml_state_providers
    _merge_config_toml_state_providers({})


def _read_explorer_plugins():
    from agent_env.explorer.plugin import load_plugins
    load_plugins()


def _read_model():
    Config().get_default_model()


def _read_agents():
    Config().get_default_a2a_agent_id()


# Every file section, with a file a user would plausibly write and the reader that consumes
# it. The non-memoized `_merge_config_toml_*` entry points are used deliberately: the public
# registries cache, so a matrix through them would only ever test the first case; `model` and
# `agents` use a fresh `Config()` for the same reason.
# `explorer` itself is absent — `explorer.app` needs the optional fastapi extra, so its
# reader is not importable in this tier; `[explorer.plugins]` covers the same walk.
_SECTION_READERS = [
    ("envs", '[envs]\nimpls = []\n', _read_envs),
    ("artifacts", '[artifacts]\nimpls = []\n', _read_artifacts),
    ("task_steps", '[task_steps]\nimpls = []\n', _read_task_steps),
    ("sandbox", '[sandbox]\ndefault = "local"\n', _read_sandbox),
    ("state", "[state]\n", _read_state),
    ("explorer", '[explorer.plugins]\nimpls = []\n', _read_explorer_plugins),
    ("model", '[model]\nbase_url = "https://gw.example/v1"\n', _read_model),
    ("agents", '[agents]\ndefault_a2a_agent_id = "an-agent"\n', _read_agents),
]
_IDS = [name for name, _, _ in _SECTION_READERS]


@pytest.mark.parametrize("name, valid, reader", _SECTION_READERS, ids=_IDS)
def test_a_validly_shaped_section_reaches_its_reader(tmp_path, monkeypatch, name, valid, reader):
    """Half the shape rule: a file someone would plausibly write goes through untouched."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, valid)))

    reader()


@pytest.mark.parametrize("name, valid, reader", _SECTION_READERS, ids=_IDS)
def test_a_mis_shaped_section_fails_its_reader(tmp_path, monkeypatch, name, valid, reader):
    """The other half. `sandbox = 5` used to leave the reader on its built-in `local` while
    `config show` printed the 5 straight back — the report confirming a value the process
    never used, which is the failure that command exists to prevent."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, f"{name} = 5\n")))

    with pytest.raises(ConfigError, match=rf"\[{name}\] must be a table, got int"):
        reader()


def test_an_absent_section_still_falls_back(tmp_path, monkeypatch):
    """Absent is not mis-shaped: every one of these readers has a built-in default, and a
    config that simply says nothing about a section must keep getting it."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, '[model]\nbase_url = "x"\n')))

    for _, _, reader in _SECTION_READERS:
        reader()
    assert get_config()._document().section("envs") == {}
    assert get_config()._document().section("explorer", "plugins") == {}


def test_the_report_names_a_mis_shaped_section_instead_of_echoing_it(tmp_path, monkeypatch):
    """`config show` used to print `sandbox: 5` while the process ran on `local`."""
    from agent_env.cli.config import render
    from agent_env.config.describe import describe_config

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, "sandbox = 5\n")))

    rendered = render(describe_config())
    assert "must be a table, got int" in rendered
    assert "sandbox:        5" not in rendered


def test_a_nested_section_path_is_walked(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(
        tmp_path, '[explorer.plugins.a]\nimpl = "p:A"\n')))

    assert get_config()._document().section("explorer", "plugins") == {"a": {"impl": "p:A"}}
    assert get_config()._document().section("explorer", "plugins", "a") == {"impl": "p:A"}


def test_a_broken_document_fails_a_reader_instead_of_defaulting_it(tmp_path, monkeypatch):
    """The regression that matters. On a malformed config.toml the sandbox reader used to
    raise; routing it through a snapshot that answered `{}` made it return the built-in
    `local` instead — a broken file quietly choosing a different backend."""
    from agent_env.providers import sandbox_provider

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, "stores = [[[\n")))

    with pytest.raises(ConfigError, match="Malformed"):
        sandbox_provider._default_sandbox_spec()


def test_concurrent_first_readers_share_one_resolution(tmp_path, monkeypatch):
    """Two first callers must not both resolve. The cache is what introduces the race, so
    the invariant it exists for — one document, agreed by every reader — would otherwise
    hold only by luck."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(tmp_path, '[sandbox]\ndefault = "local"\n')))
    threads_n = 8
    parses, real = [], tomllib.loads

    def counted(text):
        parses.append(1)
        time.sleep(0.01)  # widen the window a second first-caller would slip through
        return real(text)

    monkeypatch.setattr(tomllib, "loads", counted)
    start, seen = threading.Barrier(threads_n), []
    threads = [threading.Thread(target=lambda: (start.wait(timeout=5), seen.append(get_config()._document())))
               for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(parses) == 1
    assert len({id(s) for s in seen}) == 1 and len(seen) == threads_n


def test_one_readers_mutation_cannot_reach_another(tmp_path, monkeypatch):
    """The hazard the consolidation would otherwise have created. While each reader parsed
    for itself, a reader that mutated what it got affected nobody; sharing one document made
    that reach every other reader, nested tables included."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(_config_at(
        tmp_path, '[sandbox]\ndefault = "local"\n[sandbox.providers.a]\nimpl = "p:A"\n')))

    first = get_config()._document().section("sandbox")
    first["default"] = "mutated"
    first["providers"]["a"]["impl"] = "p:Hijacked"
    del first["providers"]["a"]["impl"]

    second = get_config()._document().section("sandbox")
    assert second == {"default": "local", "providers": {"a": {"impl": "p:A"}}}
    assert second is not first

    # and the same holds for the whole document a Config hands back
    from agent_env.config import Config
    document = Config().config_file()
    document["sandbox"]["default"] = "mutated"
    assert Config().config_file()["sandbox"]["default"] == "local"


def test_re_pointing_the_config_env_var_needs_a_reset(tmp_path, monkeypatch):
    """The contract a Config makes: it resolves once and keeps it. Re-pointing mid-process
    reaches no reader until `reset_config()` drops the Config holding the old document.

    This used to take effect on its own, because the parse cache re-ran discovery every
    call. Half a process could move file while the other half had not.
    """
    a = tmp_path / "a" / "config.toml"
    b = tmp_path / "b" / "config.toml"
    for f, default in ((a, "local"), (b, "modal")):
        f.parent.mkdir(parents=True)
        f.write_text(f'[sandbox]\ndefault = "{default}"\n')

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(a))
    assert get_config()._document().section("sandbox") == {"default": "local"}

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(b))
    assert get_config()._document().section("sandbox") == {"default": "local"}  # not yet
    assert get_config()._document().path == a

    reset_config()
    assert get_config()._document().section("sandbox") == {"default": "modal"}
    assert get_config()._document().path == b


def test_a_reader_needs_a_reset_to_see_a_re_point(tmp_path, monkeypatch):
    """The same contract through a reader, which is where a consumer meets it: the sdk
    re-points per stage and pairs it with `reset_config()` for this reason."""
    from agent_env.providers import sandbox_provider

    good = _config_at(tmp_path / "good", '[envs]\nimpls = []\n')
    bad = _config_at(tmp_path / "bad", "stores = [[[\n")

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(good))
    assert sandbox_provider._default_sandbox_spec() == "local"

    monkeypatch.setenv("AGENT_ENV_CONFIG", str(bad))
    assert sandbox_provider._default_sandbox_spec() == "local"  # still the resolved document

    reset_config()
    with pytest.raises(ConfigError, match="Malformed"):
        sandbox_provider._default_sandbox_spec()


def test_a_file_that_vanishes_after_discovery_is_a_race_not_an_absent_config(tmp_path, monkeypatch):
    """Discovery returns a path it has just seen, so losing it before the parse is a race.
    Answering `{}` handed every reader its built-in default while `config show` named a file
    that was not there. The error rides on the snapshot, so the path stays reportable."""
    path = _config_at(tmp_path, '[sandbox]\ndefault = "modal"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    real = snapshot.loader.discover_config_path
    monkeypatch.setattr(snapshot.loader, "discover_config_path",
                        lambda *a, **k: (lambda p: (p.unlink(), p)[1])(real(*a, **k)))

    resolved = get_config()._document()
    assert resolved.path == path                      # still nameable
    assert "vanished between discovery and read" in str(resolved.error)
    with pytest.raises(ConfigError, match="vanished"):
        resolved.section("sandbox")                   # but no reader gets a default


def test_the_strict_read_and_the_lenient_one_have_different_contracts(tmp_path):
    """The two contracts, stated where a regression would show. `read_config_file` opens and
    lets a missing file raise — no existence check in front of it, because check-then-read
    only narrows a race rather than closing it. `load_config_file` stays lenient for the
    public callers, the sdk's exporter among them.

    These are contract tests, not a reproduction: the two shapes differ only in the genuine
    race — check passes, file goes, second check fails — which a static test cannot stage.
    """
    absent = tmp_path / "gone.toml"

    with pytest.raises(FileNotFoundError):
        snapshot.loader.read_config_file(absent)
    assert snapshot.loader.load_config_file(absent) == {}
    assert snapshot.loader.load_config_file(None) == {}


def test_a_snapshot_never_reports_an_empty_document_for_a_file_it_could_not_read(tmp_path, monkeypatch):
    """The invariant underneath both: a path the snapshot names but could not read must
    carry an error. An empty document with no error is the silent downgrade."""
    path = _config_at(tmp_path, '[sandbox]\ndefault = "modal"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    path.unlink()
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: True)  # discovery still yields it

    resolved = get_config()._document()

    assert resolved.path == path
    assert resolved.error is not None, "named a file, read nothing, reported no error"
    with pytest.raises(ConfigError):
        resolved.section("sandbox")
