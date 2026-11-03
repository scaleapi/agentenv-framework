"""Bundles run from a Hugging Face Hub dataset: ``hf://`` references, the bundle picked from a commit's file list, the
download kept per commit, the card's requirements, and the ids and record of ``agent-env run hf://``."""

import fnmatch
import json
import logging
import shutil
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from huggingface_hub.errors import RepositoryNotFoundError

from agent_env.bundle import BundleError
from agent_env.bundle import hub as hub_module
from agent_env.bundle.hub import HubBundle, HubReference, HubSource, _select, fetch_bundle, parse_reference
from agent_env.bundle.ledger import LEDGER_COLLECTION
from agent_env.bundle.requires import UnmetRequirements
from agent_env.cli import cli
from agent_env.config.paths import cache_root
from agent_env.config.runtime import Config
from agent_env.openenv import build_app
from agent_env.store import Filter
from agent_env.store.routing import namespace_routing
from agent_env.task.store import task_instances
from agent_env.task_step.task_step import TaskStep
from tst.unit.bundle._support import fake_plugins

SHA = "0123456789abcdef0123456789abcdef01234567"
TASK = json.dumps([{"id": "check", "type": "noop_hub_test"}])
SOURCE = HubSource("me/ds", "main", SHA, "bundles/hello")


class _Noop(TaskStep):
    type = "noop_hub_test"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Noop.type: _Noop}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)


@pytest.fixture
def quiet_logs():
    """pytest's live logging swaps its own stdout back in to print a record, so CliRunner loses whatever is
    echoed after the first one."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class FakeHub:
    """Serves the dataset ``me/ds`` from the folder ``repo``, every revision at SHA, recording each call."""

    def __init__(self, repo):
        self.repo = repo
        self.calls = []

    def repo_info(self, repo_id, *, repo_type, revision=None):
        self.calls.append("repo_info")
        assert repo_type == "dataset"
        if repo_id != "me/ds":
            response = httpx.Response(404, request=httpx.Request("GET", f"http://hub.test/api/datasets/{repo_id}"))
            raise RepositoryNotFoundError("404 Client Error. Repository Not Found", response=response)
        return SimpleNamespace(sha=SHA)

    def list_repo_files(self, repo_id, *, repo_type, revision):
        self.calls.append("list_repo_files")
        assert (repo_type, revision) == ("dataset", SHA)
        return self._files()

    def hf_hub_download(self, repo_id, filename, *, repo_type, revision, local_dir):
        self.calls.append(f"hf_hub_download {filename}")
        self._copy([filename], local_dir)

    def snapshot_download(self, repo_id, *, repo_type, revision, local_dir, allow_patterns):
        self.calls.append(f"snapshot_download {' '.join(allow_patterns)}")
        assert (repo_type, revision) == ("dataset", SHA)
        self._copy([file for file in self._files() if any(fnmatch.fnmatch(file, p) for p in allow_patterns)], local_dir)

    def layout(self, files):
        for rel, text in files.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text)

    def card(self, agentenv=None):
        meta = f"agentenv: {json.dumps(agentenv)}\n" if agentenv is not None else ""
        self.layout({"README.md": f"---\ntags:\n- agentenv\n{meta}---\n\n# A dataset\n"})

    def _files(self):
        return sorted(path.relative_to(self.repo).as_posix() for path in self.repo.rglob("*") if path.is_file())

    def _copy(self, files, local_dir):
        for file in files:
            (local_dir / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.repo / file, local_dir / file)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    fake = FakeHub(tmp_path / "repo")
    monkeypatch.setattr(hub_module, "HfApi", lambda: fake)
    return fake


def _commit(tmp_path):
    return tmp_path / "cache" / "agentenv" / "hf" / "me" / "ds" / SHA


def _downloaded(tmp_path):
    folder = _commit(tmp_path)
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*")
                  if path.is_file() and path.relative_to(folder).parts[0] != ".cache")


@pytest.mark.parametrize("text, reference", [
    ("hf://me/ds", HubReference("me/ds", None, None)),
    ("hf://me/ds@v1.0", HubReference("me/ds", "v1.0", None)),
    ("hf://me/ds/bundles/hello/", HubReference("me/ds", None, "bundles/hello")),
    (f"hf://Me-2/ds_v1.2@{SHA}/suite/x", HubReference("Me-2/ds_v1.2", SHA, "suite/x")),
    ("hf://me/ds@refs%2Fpr%2F3/bundles/hello", HubReference("me/ds", "refs/pr/3", "bundles/hello")),
])
def test_a_reference_names_a_repo_then_maybe_a_revision_and_a_path(text, reference):
    assert parse_reference(text) == reference


@pytest.mark.parametrize("text, problem", [
    ("hf://me", "name a Hub dataset's bundle as hf://<org>/<repo>[@<revision>][/<path>]"),
    ("hf:///ds", "name a Hub dataset's bundle as"),
    ("hf://me/ds@", "name a Hub dataset's bundle as"),
    ("hf://me/d s", "name a Hub dataset's bundle as"),
    ("hf://me/.ds", "name a Hub dataset's bundle as"),
    ("hf://me/ds/bundles/../../x", "the path in the repo can't hold '.' or '..'"),
])
def test_a_malformed_reference_is_refused(text, problem):
    with pytest.raises(BundleError) as caught:
        parse_reference(text)
    (refused,) = caught.value.problems
    assert refused.startswith(f"{text}: ") and problem in refused


def test_the_only_bundle_under_bundles_wins_over_a_root_of_data_files_and_only_it_and_the_card_are_downloaded(
        hub, tmp_path):
    hub.layout({"bundles/hello/tasks/t.json": TASK, "tasks/train.parquet": "rows", "runs/recording.mp4": "frames"})
    hub.card()
    said = []

    fetched = fetch_bundle("hf://me/ds", on_progress=said.append)

    assert said == ["me/ds at 0123456789ab, bundle bundles/hello"]
    assert fetched.root == _commit(tmp_path) / "bundles" / "hello"
    assert fetched.id_root == "@hf/me/ds/hello"
    assert fetched.source == SOURCE
    assert hub.calls == ["repo_info", "list_repo_files", "hf_hub_download README.md",
                         "snapshot_download bundles/hello/*"]
    assert _downloaded(tmp_path) == ["README.md", "bundles/hello/tasks/t.json"]


def test_several_bundles_need_a_path_or_the_cards_default(hub, tmp_path):
    hub.layout({"bundles/a/tasks/t.json": TASK, "bundles/b/tasks/t.json": TASK})
    hub.card()

    with pytest.raises(BundleError) as caught:
        fetch_bundle("hf://me/ds@v1")
    named = fetch_bundle("hf://me/ds/bundles/a")
    shutil.rmtree(tmp_path / "cache")
    hub.card({"default": "b"})
    by_default = fetch_bundle("hf://me/ds")

    assert caught.value.problems == (
        "me/ds at 0123456789ab holds several bundles under bundles/ and its card names no default (agentenv: "
        "default); run one of hf://me/ds@v1/bundles/a, hf://me/ds@v1/bundles/b",)
    assert (named.source.path, named.id_root) == ("bundles/a", "@hf/me/ds/a")
    assert (by_default.source.path, by_default.id_root) == ("bundles/b", "@hf/me/ds/b")


def test_a_path_names_any_bundle_folder_and_its_ids_follow_it(hub):
    hub.layout({"suites/hello/tasks/t.json": TASK, "suites/notes.md": "not a bundle"})

    fetched = fetch_bundle("hf://me/ds/suites/hello")
    with pytest.raises(BundleError) as caught:
        fetch_bundle("hf://me/ds/suites")

    assert (fetched.id_root, fetched.source.path) == ("@hf/me/ds/suites/hello", "suites/hello")
    assert caught.value.problems == ("me/ds at 0123456789ab: suites/ isn't a bundle: it has none of envs/, agents/, "
                                     "artifacts/, skills/, tasks/, evals/",)


def test_a_bundle_is_named_by_its_path_outside_bundles_so_two_folders_of_one_name_keep_apart(hub):
    hub.layout({"a/x/tasks/t.json": TASK, "b/x/tasks/t.json": TASK, "bundles/x/tasks/t.json": TASK})
    hub.card({"bundles": {"a/x": {"setup": "set up a/x"}, "x": {"setup": "set up bundles/x"}}})

    a, b, x = (fetch_bundle(f"hf://me/ds/{path}") for path in ("a/x", "b/x", "bundles/x"))

    assert [(bundle.id_root, bundle.setup) for bundle in (a, b, x)] == [
        ("@hf/me/ds/a/x", "set up a/x"), ("@hf/me/ds/b/x", None), ("@hf/me/ds/x", "set up bundles/x")]


def test_the_root_is_the_bundle_when_nothing_is_under_bundles_and_its_other_folders_stay_on_the_hub(hub, tmp_path):
    hub.layout({"tasks/t.json": TASK, "bundle.toml": "requires = []\n", "data/rows.parquet": "rows"})

    fetched = fetch_bundle("hf://me/ds")

    assert (fetched.root, fetched.id_root, fetched.source.path) == (_commit(tmp_path), "@hf/me/ds", "")
    assert _downloaded(tmp_path) == ["bundle.toml", "tasks/t.json"]


def test_a_repo_without_a_bundle_is_refused_before_any_download(hub):
    hub.layout({"data/rows.parquet": "rows", "bundles/README.md": "nothing here"})

    with pytest.raises(BundleError) as caught:
        fetch_bundle("hf://me/ds")

    assert caught.value.problems == ("me/ds at 0123456789ab holds no bundle: it has no bundles/<name>/ folder, and none "
                                     "of envs/, agents/, artifacts/, skills/, tasks/, evals/ at its root",)
    assert hub.calls == ["repo_info", "list_repo_files"]


def test_a_commit_already_downloaded_is_reused_and_a_pinned_one_needs_no_network(hub):
    hub.layout({"bundles/a/tasks/t.json": TASK, "bundles/b/tasks/t.json": TASK})

    first = fetch_bundle(f"hf://me/ds@{SHA}/bundles/a")
    hub.calls.clear()
    again = fetch_bundle(f"hf://me/ds@{SHA}/bundles/a")
    by_branch = fetch_bundle("hf://me/ds/bundles/a")
    reused, hub.calls[:] = list(hub.calls), []
    other = fetch_bundle(f"hf://me/ds@{SHA}/bundles/b")

    assert first.root == again.root == by_branch.root
    assert again.source.revision == SHA and by_branch.source.revision == "main"
    assert reused == ["repo_info"]
    assert hub.calls == ["snapshot_download bundles/b/*"]
    assert other.root.parent == first.root.parent


def test_a_repo_the_hub_wont_serve_names_how_to_log_in(hub):
    with pytest.raises(BundleError) as caught:
        fetch_bundle("hf://someone/private")

    assert caught.value.problems == ("hf://someone/private: 404 Client Error. Repository Not Found",)
    assert "log in with `hf auth login`, or set HF_TOKEN" in caught.value.__notes__[0]


def test_a_hub_bundle_runs_under_its_hf_id_root_and_each_run_records_where_it_came_from(hub, local_stores, quiet_logs):
    hub.layout({"bundles/hello/tasks/t.json": TASK})

    first = CliRunner().invoke(cli, ["run", "hf://me/ds"])
    second = CliRunner().invoke(cli, ["run", "hf://me/ds"])

    assert first.exit_code == 0, first.output
    assert first.output.startswith("me/ds at 0123456789ab, bundle bundles/hello\n")
    assert "From me/ds at 0123456789ab, bundle bundles/hello\nTasks:\n  tasks/t.json v1: unscored" in first.output
    assert "instance @hf/me/ds/hello/t-" in first.output
    assert "tasks/t.json: v1, unchanged" in second.output
    with namespace_routing():
        runs = task_instances("@hf/me/ds/hello/t")
    assert [(run.task_version, run.context["metadata"]["bundle_source"]) for run in runs] == [
        (1, {"hub": "hf", "repo": "me/ds", "revision": "main", "sha": SHA, "path": "bundles/hello"})] * 2
    assert HubSource.from_metadata(runs[0].context["metadata"]["bundle_source"]) == SOURCE
    rows = local_stores.local_namespace_document_store("hf").query(LEDGER_COLLECTION, Filter())
    assert {row["id"] for row in rows} == {"@hf/me/ds/hello/t"}
    assert not local_stores.local_namespace_document_store().path.exists()


def test_a_dry_run_downloads_and_checks_but_runs_and_writes_nothing(hub, local_stores, quiet_logs, tmp_path):
    hub.layout({"bundles/hello/tasks/t.json": TASK})

    result = CliRunner().invoke(cli, ["run", "hf://me/ds", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("me/ds at 0123456789ab, bundle bundles/hello\nDry run: nothing is written or run.\n")
    assert "Would run:\n  tasks/t.json v1\n" in result.output
    assert _downloaded(tmp_path) == ["bundles/hello/tasks/t.json"]
    assert not local_stores.local_namespace_document_store("hf").path.exists()


def test_a_bundle_without_a_bundle_toml_needs_the_plugins_its_card_names_before_it_is_downloaded(hub, local_stores,
                                                                                               quiet_logs):
    hub.layout({"bundles/hello/tasks/t.json": TASK})
    hub.card({"bundles": {"hello": {"plugins": ["agentenv-not-installed>=1"], "setup": "agent-env demo setup"}}})

    refused = CliRunner().invoke(cli, ["run", "hf://me/ds"])
    calls = list(hub.calls)
    hub.layout({"bundles/hello/bundle.toml": "requires = []\n"})
    shutil.rmtree(cache_root())
    declared = CliRunner().invoke(cli, ["run", "hf://me/ds"])

    assert refused.exit_code == 1
    assert ("me/ds's card: requires 'agentenv-not-installed>=1', which isn't installed; install it with: agent-env "
            "plugin add 'agentenv-not-installed>=1'\nme/ds's card sets this bundle up with: agent-env demo setup") in (
        refused.output)
    assert calls == ["repo_info", "list_repo_files", "hf_hub_download README.md"]
    assert declared.exit_code == 0, declared.output


@pytest.mark.parametrize("requires, problem, noted", [
    ('["agentenv-not-installed>=1"]', "bundle.toml: requires 'agentenv-not-installed>=1', which isn't installed; "
     "install it with: agent-env plugin add 'agentenv-not-installed>=1'", True),
    ('["agentenv-x @ git+https://example.invalid/x"]', "bundle.toml: requires: 'agentenv-x @ "
     "git+https://example.invalid/x' names a URL; name the package and the versions it needs, which install from "
     "wherever whoever runs the bundle installs from", False),
])
def test_a_bundle_toml_is_downloaded_alone_and_checked_as_a_folders_is_before_the_rest(hub, requires, problem, noted):
    hub.layout({"bundles/hello/tasks/t.json": TASK, "bundles/hello/bundle.toml": f"requires = {requires}\n"})
    hub.card({"bundles": {"hello": {"plugins": ["agentenv-the-card-names"], "setup": "agent-env demo setup"}}})

    with pytest.raises(BundleError) as caught:
        fetch_bundle("hf://me/ds")

    assert caught.value.problems == (problem,)
    assert getattr(caught.value, "__notes__", []) == (["me/ds's card sets this bundle up with: agent-env demo setup"]
                                                      if noted else [])
    assert hub.calls == ["repo_info", "list_repo_files", "hf_hub_download README.md",
                         "hf_hub_download bundles/hello/bundle.toml"]


@pytest.mark.parametrize("bundles, problem", [
    (["hello"], "me/ds's card: agentenv: bundles must map each bundle's name to what it needs"),
    ({"hello": "agentenv-x"}, "me/ds's card: agentenv: bundles: hello must be a table of plugins and setup"),
    ({"hello": {"plugins": "agentenv-x"}}, "me/ds's card: agentenv: bundles: hello: plugins must be a list of pip "
                                          "requirements"),
    ({"hello": {"setup": ["agent-env", "x"]}}, "me/ds's card: agentenv: bundles: hello: setup must be a command"),
])
def test_a_card_that_says_what_a_bundle_needs_in_the_wrong_shape_is_refused(hub, bundles, problem):
    hub.layout({"bundles/hello/tasks/t.json": TASK})
    hub.card({"bundles": bundles})

    with pytest.raises(BundleError) as caught:
        fetch_bundle("hf://me/ds")

    assert caught.value.problems == (problem,)


def test_an_openenv_server_of_a_hub_bundle_needs_the_plugins_its_card_names_before_it_is_downloaded_too(hub):
    hub.layout({"bundles/hello/tasks/t.json": TASK})
    hub.card({"bundles": {"hello": {"plugins": ["agentenv-not-installed>=1"]}}})

    with pytest.raises(UnmetRequirements, match="agent-env plugin add 'agentenv-not-installed>=1'"):
        build_app("hf://me/ds")

    assert hub.calls == ["repo_info", "list_repo_files", "hf_hub_download README.md"]


def test_a_cached_commit_is_refused_while_a_plugin_its_card_names_is_missing_by_the_cli_and_openenv_alike(
        hub, monkeypatch, local_stores, quiet_logs):
    hub.layout({"bundles/hello/tasks/t.json": TASK})
    hub.card({"bundles": {"hello": {"plugins": ["agentenv-gone>=1"], "setup": "agent-env demo setup"}}})
    fake_plugins(monkeypatch, versions={"agentenv-gone": "1.0"})
    fetch_bundle(f"hf://me/ds@{SHA}")
    fake_plugins(monkeypatch)
    hub.calls.clear()

    refused = CliRunner().invoke(cli, ["run", f"hf://me/ds@{SHA}"])
    with pytest.raises(UnmetRequirements) as served:
        build_app(f"hf://me/ds@{SHA}")

    assert refused.exit_code == 1
    assert ("plugin add 'agentenv-gone>=1'\nme/ds's card sets this bundle up with: agent-env demo setup\n"
            in refused.output)
    assert refused.output.count("sets this bundle up with") == 1
    assert served.value.__notes__ == ["me/ds's card sets this bundle up with: agent-env demo setup"]
    assert hub.calls == []


def test_selecting_a_bundle_from_a_commits_files_and_card_reads_nothing(tmp_path):
    files = ["README.md", "tasks/train.parquet", "bundles/a/bundle.toml", "bundles/a/tasks/t.json",
             "bundles/b/tasks/t.json"]
    card = {"default": "b", "bundles": {"a": {"plugins": ["agentenv-a"]}, "b": {"plugins": ["agentenv-b"],
                                                                                "setup": "set up b"}}}

    default = _select(parse_reference("hf://me/ds@v1"), SHA, files, card, tmp_path)
    declared = _select(parse_reference("hf://me/ds/bundles/a"), SHA, files, card, tmp_path)

    assert default == HubBundle(tmp_path / "bundles/b", "@hf/me/ds/b", HubSource("me/ds", "v1", SHA, "bundles/b"),
                                plugins=("agentenv-b",), setup="set up b")
    assert (declared.id_root, declared.plugins, declared.setup) == ("@hf/me/ds/a", (), None)


def test_the_cards_setup_command_is_shown_only_when_setting_up_could_fix_the_run(hub, local_stores, quiet_logs):
    note = "me/ds's card sets this bundle up with: agent-env demo setup"
    gap = {"id": "deploy", "type": "deploy_env", "env_id": "env-setup-writes"}
    other = {"id": "other", "type": "deploy_env", "env_id": "@hf/me/ds/hello/missing"}
    hub.card({"bundles": {"hello": {"setup": "agent-env demo setup"}}})
    hub.layout({"bundles/hello/tasks/t.json": json.dumps([gap, other])})

    mixed = CliRunner().invoke(cli, ["run", "hf://me/ds"])
    hub.layout({"bundles/hello/tasks/t.json": json.dumps([other])})
    shutil.rmtree(cache_root())
    neither = CliRunner().invoke(cli, ["run", "hf://me/ds"])

    assert mixed.exit_code == neither.exit_code == 1
    assert "step 'deploy': env_id: there is no env 'env-setup-writes' in the store" in mixed.output
    assert "step 'other': env_id: '@hf/me/ds/hello/missing' isn't in this bundle" in mixed.output
    assert note in mixed.output
    assert "step 'other': env_id: '@hf/me/ds/hello/missing' isn't in this bundle" in neither.output
    assert note not in neither.output
