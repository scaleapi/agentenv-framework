"""Reading a bundle folder into its entities: ids, kinds, artifact types and every folder rule."""

import os
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_env.bundle import BundleError, BundleKind, is_ignored, parse_bundle
from agent_env.bundle import _fs
from agent_env.store.ids import LOCAL_PREFIX, MAX_AUTHORED_LOCAL_ID_BYTES


def build(root: Path, spec: dict[str, str | bytes | None]) -> Path:
    """Create ``root`` with files (str or bytes content) and folders (a key ending in ``/``)."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in spec.items():
        path = root / rel
        if rel.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content or "")
    return root


def problems(root: Path, **kwargs) -> tuple[str, ...]:
    with pytest.raises(BundleError) as caught:
        parse_bundle(root, **kwargs)
    return caught.value.problems


def kinds(bundle) -> list[tuple[str, str, str]]:
    return [(entry.kind.value, entry.name, entry.type) for entry in bundle.entries]


@pytest.fixture(autouse=True)
def _tmp_is_home(tmp_path, monkeypatch):
    """Folders above home aren't part of an id, so the machine's tmp path can't break these tests."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.touch()
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


def test_golden_bundle(home):
    root = build(home / "work" / "triage", {
        "README.md": "\ufeff\n# Triage demo\nmore\n",
        "envs/tickets/Dockerfile": "FROM x",
        "envs/both/env.toml": 'type = "multi"\nmcp_server_envs = ["tickets"]\n',
        "agents/solver/Dockerfile": "FROM y",
        "artifacts/greeting/a.txt": "a",
        "artifacts/greeting/b.txt": "b",
        "artifacts/check/check.py": "c",
        "artifacts/base/Dockerfile": "FROM z",
        "artifacts/golden/artifact.toml": 'type = "vm_image"\n',
        "skills/pdf/SKILL.md": "---\nname: pdf\n---\n",
        "tasks/hello.json": '[{"id": "s", "type": "deploy_sandbox"}]',
        "evals/regression.toml": 'tasks = ["hello"]\n',
        "notes.txt": "",
        "docs/": None,
        ".git/HEAD": "",
    })
    bundle = parse_bundle(root)

    assert bundle.name == "triage"
    assert bundle.id_root == "@local/~/work/triage"
    assert bundle.description == "Triage demo"
    assert bundle.ignored == (Path("docs"), Path("notes.txt"))
    assert kinds(bundle) == [
        ("envs", "both", "multi"),
        ("envs", "tickets", "mcp_server"),
        ("agents", "solver", "a2a_agent"),
        ("artifacts", "base", "docker_image"),
        ("artifacts", "check", "file"),
        ("artifacts", "golden", "vm_image"),
        ("artifacts", "greeting", "file_artifact_universe"),
        ("skills", "pdf", "skill"),
        ("tasks", "hello", "task"),
        ("evals", "regression", "eval"),
    ]
    by_name = {entry.name: entry for entry in bundle.entries}
    assert by_name["tickets"].id == "@local/~/work/triage/tickets"
    assert by_name["tickets"].path == root / "envs" / "tickets"
    assert by_name["tickets"].config == {}
    assert by_name["both"].config["mcp_server_envs"] == ["tickets"]
    assert by_name["hello"].config == [{"id": "s", "type": "deploy_sandbox"}]
    assert by_name["hello"].path == root / "tasks" / "hello.json"
    assert by_name["pdf"].config is None


def test_kind_store_shares_the_artifact_store():
    assert BundleKind.SKILL.store == BundleKind.ARTIFACT.store == "artifact"
    assert [kind.store for kind in (BundleKind.ENV, BundleKind.AGENT, BundleKind.TASK, BundleKind.EVAL)] == [
        "env", "agent", "task", "eval"
    ]
    assert BundleKind.ENV != "envs"


def test_is_ignored():
    assert [is_ignored(name) for name in (".git", ".DS_Store", "__pycache__", "src", "a.pyc", "__init__.py")] == [
        True, True, True, False, False, False,
    ]


class TestIdRoot:
    def test_outside_home_is_the_absolute_path(self, tmp_path, home):
        root = build(tmp_path / "elsewhere" / "b", {"tasks/t.json": "[]"})
        assert parse_bundle(root).id_root == LOCAL_PREFIX + str(root.resolve())[1:]

    def test_home_itself(self, home):
        build(home, {"tasks/t.json": "[]"})
        bundle = parse_bundle(home)
        assert bundle.id_root == "@local/~"
        assert bundle.entries[0].id == "@local/~/t"

    @pytest.mark.parametrize("value", ["/", "", "relative/dir", "/no/such/home", None])
    def test_unusable_home_falls_back_to_the_passwd_entry(self, tmp_path, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("HOME")
        else:
            monkeypatch.setenv("HOME", value)
        entry = SimpleNamespace(pw_dir=str(tmp_path))
        monkeypatch.setattr(_fs, "pwd", SimpleNamespace(getpwuid=lambda uid: entry))
        root = build(tmp_path / "b", {"tasks/t.json": "[]"})
        assert parse_bundle(root).id_root == "@local/~/b"

    def test_no_home_at_all(self, tmp_path, monkeypatch):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        monkeypatch.setenv("HOME", str(fifo))
        monkeypatch.setattr(_fs, "pwd", None)
        root = build(tmp_path / "b", {"tasks/t.json": "[]"})
        assert parse_bundle(root).id_root == LOCAL_PREFIX + str(root.resolve())[1:]

    def test_symlinked_and_relative_roots_give_one_id(self, home, monkeypatch):
        root = build(home / "work" / "b", {"tasks/t.json": "[]"})
        (home / "link").symlink_to(root)
        monkeypatch.chdir(home / "work")
        ids = {parse_bundle(path).id_root for path in (root, home / "link", Path("b"), Path("../work/b"))}
        assert ids == {"@local/~/work/b"}

    def test_entry_point_root(self, tmp_path):
        root = build(tmp_path / "hello", {"artifacts/greeting/g.txt": "hi"})
        bundle = parse_bundle(root, id_root="@local/agentenv-framework/hello", name="hello")
        assert bundle.entries[0].id == "@local/agentenv-framework/hello/greeting"

    def test_entry_point_root_is_validated(self, tmp_path):
        root = build(tmp_path / "hello", {"tasks/t.json": "[]", "envs/e/Dockerfile": ""})
        (problem,) = problems(root, id_root="agentenv-framework/hello")
        assert "the bundle's id root 'agentenv-framework/hello'" in problem

    def test_nfd_folder_names_give_nfc_ids(self, home):
        nfd = unicodedata.normalize("NFD", "café")
        root = build(home / nfd, {f"envs/{nfd}/Dockerfile": "FROM x"})
        entry = parse_bundle(root).entries[0]
        nfc = unicodedata.normalize("NFC", "café")
        assert entry.name == nfc
        assert entry.id == f"@local/~/{nfc}/{nfc}"
        assert entry.path.name == nfd

    @pytest.mark.skipif(sys.platform != "darwin", reason="F_GETPATH is macOS-only")
    def test_mistyped_case_gives_the_on_disk_spelling(self, home):
        if not _case_insensitive(home):
            pytest.skip("case-sensitive filesystem")
        build(home / "Work" / "Triage", {"tasks/t.json": "[]"})
        assert parse_bundle(home / "work" / "triage").id_root == "@local/~/Work/Triage"

    def test_a_refused_folder_above_the_bundle(self, home):
        root = build(home / "notes#1" / "b", {"tasks/t.json": "[]"})
        (problem,) = problems(root)
        assert problem.endswith("notes#1: move the bundle, or rename the folder: 'notes#1' contains '#'; a local id "
                                "may contain only letters, digits, spaces and . _ - ~ / @ ( ) + , &")

    @pytest.mark.parametrize("folders", [
        "Library/CloudStorage/GoogleDrive-a.b@example.com/My Drive", "Dropbox (Personal)", "C++/R&D", "triage copy (2)",
    ])
    def test_synced_and_everyday_folder_names(self, home, folders):
        root = build(home / folders / "b", {"tasks/t.json": "[]"})
        assert parse_bundle(root).entries[0].id == f"@local/~/{folders}/b/t"

    @pytest.mark.parametrize("given, message", [("missing", "no such folder"), ("file.txt", "not a folder")])
    def test_bad_roots(self, tmp_path, given, message):
        (tmp_path / "file.txt").touch()
        assert problems(tmp_path / given) == (f"{tmp_path / given}: {message}",)

    def test_nul_in_the_root(self, tmp_path):
        assert "null" in problems(Path(f"{tmp_path}/a\0b"))[0]

    def test_a_lone_surrogate_in_the_root(self, tmp_path):
        assert "\\ud800" in problems(Path(f"{tmp_path}/\ud800"))[0]

    @pytest.mark.skipif(sys.platform != "darwin", reason="needs a case-insensitive filesystem")
    def test_the_listing_spells_the_root_off_macos(self, home, monkeypatch):
        build(home / "Work" / "Triage", {"tasks/t.json": "[]"})
        monkeypatch.setattr(sys, "platform", "linux")
        assert parse_bundle(home / "work" / "triage").id_root == "@local/~/Work/Triage"


class TestFolderRules:
    def test_not_a_bundle(self, tmp_path):
        root = build(tmp_path / "plain", {"README.md": "x", "src/": None})
        (problem,) = problems(root)
        assert "not a bundle" in problem

    @pytest.mark.parametrize("name", ["bundle.toml", "Bundle.TOML"])
    def test_bundle_toml_is_reserved(self, tmp_path, name):
        root = build(tmp_path / "b", {name: "", "tasks/t.json": "[]"})
        assert problems(root)[0].startswith(f"{name}: reserved")

    def test_kind_directory_near_miss(self, tmp_path):
        root = build(tmp_path / "b", {"Envs/tickets/Dockerfile": "FROM x"})
        assert problems(root) == ("Envs: rename to envs; names are case-sensitive",)

    def test_kind_directory_that_is_a_file(self, tmp_path):
        root = build(tmp_path / "b", {"tasks": "", "evals/e.toml": 'tasks = ["t"]\n'})
        assert problems(root) == ("tasks: a kind directory must be a folder",)

    @pytest.mark.parametrize("rel, expected", [
        ("envs/t/Env.toml", "envs/t/Env.toml: rename to env.toml"),
        ("envs/t/dockerfile", "envs/t/dockerfile: rename to Dockerfile"),
        ("agents/a/Agent.toml", "agents/a/Agent.toml: rename to agent.toml"),
        ("artifacts/x/Artifact.toml", "artifacts/x/Artifact.toml: rename to artifact.toml"),
        ("artifacts/x/dockerfile", "artifacts/x/dockerfile: rename to Dockerfile"),
        ("artifacts/x/skill.md", "artifacts/x: this looks like a skill"),
        ("skills/s/skill.md", "skills/s/skill.md: rename to SKILL.md"),
        ("tasks/t.JSON", "tasks/t.JSON: rename to t.json"),
    ])
    def test_reserved_name_near_misses(self, tmp_path, rel, expected):
        root = build(tmp_path / "b", {rel: "x"})
        assert problems(root)[0].startswith(expected)

    @pytest.mark.parametrize("spec, expected", [
        ({"artifacts/d/artifact.toml": 'type = "file_artifact_universe"\n', "artifacts/d/DockerFile": "",
          "artifacts/d/other": ""}, ("artifacts", "d", "file_artifact_universe")),
        ({"envs/e/env.toml": 'image = "base"\n', "envs/e/dockerfile": ""}, ("envs", "e", "mcp_server")),
        ({"agents/a/agent.toml": "", "agents/a/DOCKERFILE": ""}, ("agents", "a", "a2a_agent")),
    ])
    def test_a_dockerfile_spelled_another_way_is_content_where_it_decides_nothing(self, tmp_path, spec, expected):
        root = build(tmp_path / "b", spec)
        assert kinds(parse_bundle(root)) == [expected]

    @pytest.mark.parametrize("spec", [
        {"artifacts/img/Dockerfile": "FROM x", "artifacts/img/SKILL.md": "x"},
        {"artifacts/img/artifact.toml": 'type = "docker_image"\n', "artifacts/img/SKILL.md": "x"},
    ], ids=["built", "declared"])
    def test_an_image_folder_may_hold_a_skill_its_image_ships(self, tmp_path, spec):
        assert kinds(parse_bundle(build(tmp_path / "b", spec))) == [("artifacts", "img", "docker_image")]

    @pytest.mark.parametrize("spec, pair", [
        ({"envs/e/env.toml": "", "envs/e/ENV.toml": "app = 1\n"}, "envs/e/ENV.toml and envs/e/env.toml"),
        ({"tasks/t.json": "[]", "Tasks/notes.md": ""}, "Tasks and tasks"),
        ({"envs/e/Dockerfile": "FROM x", "envs/e/dockerfile": "notes"}, "envs/e/Dockerfile and envs/e/dockerfile"),
        ({"artifacts/d/artifact.toml": 'type = "file_artifact_universe"\n', "artifacts/d/Dockerfile": "",
          "artifacts/d/dockerfile": ""}, "artifacts/d/Dockerfile and artifacts/d/dockerfile"),
        ({"skills/s/SKILL.md": "", "skills/s/skill.md": ""}, "skills/s/SKILL.md and skills/s/skill.md"),
        ({"tasks/t.json": "[]", "tasks/t.JSON": "[]"}, "tasks/t.JSON and tasks/t.json"),
    ], ids=["toml", "kind-folder", "dockerfile", "declared-artifact", "skill-md", "suffix"])
    def test_both_spellings_of_a_reserved_name_collide(self, tmp_path, spec, pair):
        if _case_insensitive(tmp_path):
            pytest.skip("case-insensitive filesystem")
        message = "differ only in case or Unicode form; rename one so the bundle means the same on every filesystem"
        assert problems(build(tmp_path / "b", spec)) == (f"{pair} {message}",)

    def test_names_differing_only_in_case(self, tmp_path):
        if _case_insensitive(tmp_path):
            pytest.skip("case-insensitive filesystem")
        root = build(tmp_path / "b", {"envs/Tickets/Dockerfile": "", "envs/tickets/Dockerfile": ""})
        assert "differ only in case" in problems(root)[0]

    def test_names_differing_only_in_unicode_form(self, tmp_path):
        nfc, nfd = (unicodedata.normalize(form, "café") for form in ("NFC", "NFD"))
        root = build(tmp_path / "b", {f"envs/{nfc}/Dockerfile": ""})
        try:
            (root / "envs" / nfd).mkdir()
        except FileExistsError:
            pytest.skip("the filesystem treats both forms as one name")
        (root / "envs" / nfd / "Dockerfile").touch()
        assert "differ only in case or Unicode form" in problems(root)[0]

    def test_entities_need_their_marker_file(self, tmp_path):
        root = build(tmp_path / "b", {"envs/e/server.py": "", "agents/a/": None, "skills/s/notes.md": ""})
        assert problems(root) == (
            "agents/a: needs a Dockerfile or an agent.toml",
            "envs/e: needs a Dockerfile or an env.toml",
            "skills/s: a skill needs SKILL.md",
        )

    def test_agent_toml_alone_is_enough(self, tmp_path):
        root = build(tmp_path / "b", {"agents/usersim/agent.toml": 'image = "solver"\n'})
        assert kinds(parse_bundle(root)) == [("agents", "usersim", "a2a_agent")]

    def test_a_declared_agent_type_is_kept_for_row_2_2_to_check(self, tmp_path):
        root = build(tmp_path / "b", {"agents/a/agent.toml": 'type = "other"\n'})
        assert kinds(parse_bundle(root)) == [("agents", "a", "other")]

    @pytest.mark.parametrize("rel, message", [
        ("envs/e/Dockerfile/", "envs/e/Dockerfile: must be a file, not a folder"),
        ("skills/s/SKILL.md/", "skills/s/SKILL.md: must be a file, not a folder"),
        ("artifacts/x/Dockerfile/", "artifacts/x/Dockerfile: must be a file, not a folder"),
        ("envs/e/env.toml/", "envs/e/env.toml: not a regular file"),
    ])
    def test_marker_files_must_be_files(self, tmp_path, rel, message):
        root = build(tmp_path / "b", {rel: None, "artifacts/x/app.py": ""})
        assert message in problems(root)

    def test_type_must_be_a_non_empty_string(self, tmp_path):
        root = build(tmp_path / "b", {"envs/e/env.toml": 'type = ""\n'})
        assert problems(root) == ("envs/e/env.toml: type must be a non-empty string",)

    def test_ignored_and_reported_entries(self, tmp_path):
        root = build(tmp_path / "b", {
            ".hidden/": None, "__pycache__/x.pyc": "", "notes.md": "",
            "envs/README.md": "", "tasks/README.md": "", "tasks/t.json": "[]", "evals/notes.txt": "",
            "artifacts/__pycache__/x.pyc": "", "artifacts/.DS_Store": "",
        })
        bundle = parse_bundle(root)
        reported = ("envs/README.md", "evals/notes.txt", "notes.md", "tasks/README.md")
        assert bundle.ignored == tuple(map(Path, reported))
        assert kinds(bundle) == [("tasks", "t", "task")]

    @pytest.mark.parametrize("kind, spec", [
        ("tasks", {"tasks/group/t.json": "[]"}),
        ("evals", {"evals/group/e.toml": ""}),
    ])
    def test_task_and_eval_folders_are_not_supported_yet(self, tmp_path, kind, spec):
        root = build(tmp_path / "b", spec)
        assert problems(root) == (f"{kind}/group: {kind[:-1]} folders are not supported yet; "
                                  f"write {kind}/<name>.{'json' if kind == 'tasks' else 'toml'}",)

    def test_every_problem_in_one_error(self, tmp_path):
        root = build(tmp_path / "b", {"envs/e/": None, "tasks/t.json": "{}", "evals/e.toml": "x ="})
        error = pytest.raises(BundleError, parse_bundle, root).value
        assert len(error.problems) == 3
        assert list(error.problems) == sorted(error.problems)
        assert str(error) == "\n".join(error.problems)
        assert isinstance(error, ValueError)

    def test_readme_any_case(self, tmp_path):
        root = build(tmp_path / "b", {"Readme.md": "## Hello\n", "tasks/t.json": "[]"})
        assert parse_bundle(root).description == "Hello"

    def test_an_unreadable_readme_just_has_no_description(self, tmp_path):
        root = build(tmp_path / "b", {"README.md/": None, "tasks/t.json": "[]"})
        assert parse_bundle(root).description is None
        (root / "README.md").rmdir()
        (root / "README.md").symlink_to(root / "README.md")
        assert parse_bundle(root).description is None

    def test_a_second_readme_spelling_is_reported(self, tmp_path):
        if _case_insensitive(tmp_path):
            pytest.skip("case-insensitive filesystem")
        root = build(tmp_path / "b", {"README.md": "Main", "readme.md": "Other", "tasks/t.json": "[]"})
        bundle = parse_bundle(root)
        assert (bundle.description, bundle.ignored) == ("Main", (Path("readme.md"),))

    def test_control_characters_stay_on_one_line(self, tmp_path):
        root = build(tmp_path / "b", {"envs/a\nb/": None, "evals/c\td.Toml": ""})
        assert problems(root) == (
            "envs/a\\nb: needs a Dockerfile or an env.toml",
            "evals/c\\td.Toml: rename to c\\td.toml; names are case-sensitive",
        )

    @pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u202e"])
    def test_separators_and_format_characters_are_escaped(self, tmp_path, separator):
        root = build(tmp_path / "b", {f"envs/a{separator}b/": None})
        (problem,) = problems(root)
        assert len(problem.splitlines()) == 1 and separator not in problem

    @pytest.mark.parametrize("readme, expected", [
        ("---\nlicense: mit\n---\n# Real title\n", "Real title"),
        ("[![CI](https://x/b.svg)](https://x)\n<p align=center>x</p>\n####\nHello\n", "Hello"),
        ("---\nunterminated\n", "unterminated"),
    ])
    def test_description_skips_front_matter_badges_and_html(self, tmp_path, readme, expected):
        root = build(tmp_path / "b", {"README.md": readme, "tasks/t.json": "[]"})
        assert parse_bundle(root).description == expected


class TestArtifactTypes:
    @pytest.mark.parametrize("spec, expected", [
        ({"Dockerfile": "FROM x", "app.py": ""}, "docker_image"),
        ({"data.json": "{}"}, "file"),
        ({"check.py": "", ".DS_Store": "", "__pycache__/check.pyc": ""}, "file"),
        ({"hello.txt": "", ".env": ""}, "file_artifact_universe"),
        ({".env": ""}, "file"),
        ({"a.json": "", "b.json": ""}, "file_artifact_universe"),
        ({"sub/a.json": ""}, "file_artifact_universe"),
        ({"artifact.toml": 'type = "vm_image"\n'}, "vm_image"),
        ({"artifact.toml": 'type = "file_artifact_universe"\n', "Dockerfile": ""}, "file_artifact_universe"),
        ({"artifact.toml": 'id = "@local/acme/one"\n', "one.txt": ""}, "file"),
    ])
    def test_inference(self, tmp_path, spec, expected):
        root = build(tmp_path / "b", {f"artifacts/x/{rel}": content for rel, content in spec.items()})
        assert parse_bundle(root).entries[0].type == expected

    @pytest.mark.parametrize("spec, message", [
        ({".DS_Store": "", "Thumbs.db": "", "desktop.ini": ""}, "nothing in it"),
        ({"artifact.toml": ""}, "nothing in it"),
        ({"SKILL.md": ""}, "skills live in skills/<name>/"),
        ({"artifact.toml": 'type = "skill"\n', "x": ""}, "skills live in skills/<name>/"),
        ({"artifact.toml": "type = 3\n", "x": ""}, "type must be a non-empty string"),
    ])
    def test_refused(self, tmp_path, spec, message):
        root = build(tmp_path / "b", {f"artifacts/x/{rel}": content for rel, content in spec.items()})
        assert message in problems(root)[0]


class TestNames:
    @pytest.mark.parametrize("rel", ["artifacts/pdf__files/a", "skills/pdf__x/SKILL.md"])
    def test_double_underscore_is_reserved_for_artifacts_and_skills(self, tmp_path, rel):
        root = build(tmp_path / "b", {rel: ""})
        assert '"__" is reserved for derived ids' in problems(root)[0]

    def test_double_underscore_is_fine_elsewhere(self, tmp_path):
        root = build(tmp_path / "b", {
            "tasks/django__django-13741.json": "[]", "envs/django__django-13741/Dockerfile": "",
        })
        assert [entry.name for entry in parse_bundle(root).entries] == ["django__django-13741"] * 2

    def test_refused_characters_ask_for_a_rename(self, tmp_path):
        root = build(tmp_path / "b", {"envs/tick#ts/Dockerfile": ""})
        (problem,) = problems(root)
        assert problem.startswith("envs/tick#ts: rename it: 'tick#ts' contains '#'")

    @pytest.mark.parametrize("config", ["", 'id = "@local/acme/pdf"\n'])
    def test_artifact_and_skill_share_a_namespace(self, tmp_path, config):
        root = build(tmp_path / "b", {
            "artifacts/pdf/a": "", "artifacts/pdf/artifact.toml": config, "skills/pdf/SKILL.md": "",
        })
        assert problems(root) == (
            "artifacts/pdf and skills/pdf share a name; artifacts and skills share one store, so rename one",
        )

    def test_env_and_agent_may_share_a_name(self, tmp_path):
        root = build(tmp_path / "b", {"envs/solver/Dockerfile": "", "agents/solver/Dockerfile": ""})
        assert len(parse_bundle(root).entries) == 2

    def test_a_non_utf8_name(self, tmp_path):
        root = build(tmp_path / "b", {"envs/": None})
        bad = os.path.join(os.fsencode(root / "envs"), b"bad\xff")
        try:
            os.mkdir(bad)
        except OSError:
            pytest.skip("the filesystem refuses non-UTF-8 names")
        open(os.path.join(bad, b"Dockerfile"), "w").close()
        (problem,) = problems(root)
        assert problem == "envs/bad\\xff: rename it: the name is not valid UTF-8"


class TestDeclaredIds:
    def test_used_as_the_id(self, tmp_path):
        root = build(tmp_path / "b", {"envs/t/env.toml": 'id = "@local/acme/tickets"\n'})
        assert parse_bundle(root).entries[0].id == "@local/acme/tickets"

    def test_problems_name_the_config_file(self, tmp_path):
        root = build(tmp_path / "b", {"envs/t/env.toml": 'id = "bare"\n', "evals/e.toml": 'id = "bare"\n'})
        assert [problem.split(":")[0] for problem in problems(root)] == ["envs/t/env.toml", "evals/e.toml"]

    @pytest.mark.parametrize("declared, message", [
        ('"acme-tickets"', "must start with '@local/'"),
        ("3", "must start with '@local/'"),
        ('"@local/a/../b"', "'.' or '..'"),
        (f'"@local/{unicodedata.normalize("NFD", "café")}"', "NFC-normalized"),
    ])
    def test_refused(self, tmp_path, declared, message):
        root = build(tmp_path / "b", {"envs/t/env.toml": f"id = {declared}\n"})
        assert message in problems(root)[0]

    def test_artifact_leaf_may_not_contain_double_underscore(self, tmp_path):
        root = build(tmp_path / "b", {
            "artifacts/x/artifact.toml": 'id = "@local/b/pdf__files"\n', "artifacts/x/a": "",
        })
        assert '"__" is reserved for derived artifact ids' in problems(root)[0]
        env = build(tmp_path / "c", {"envs/x/env.toml": 'id = "@local/b/pdf__files"\n'})
        assert parse_bundle(env).entries[0].id == "@local/b/pdf__files"
        above = build(tmp_path / "d", {"artifacts/x/artifact.toml": 'id = "@local/a__b/pdf"\n', "artifacts/x/a": ""})
        assert parse_bundle(above).entries[0].id == "@local/a__b/pdf"

    def test_collisions(self, tmp_path):
        root = build(tmp_path / "b", {
            "envs/a/env.toml": 'id = "@local/acme/x"\n', "envs/b/env.toml": 'id = "@local/acme/x"\n',
        })
        assert problems(root) == ("envs/a and envs/b have the same id '@local/acme/x'",)

    def test_eval_collisions(self, tmp_path):
        same = 'id = "@local/acme/r"\ntasks = ["t"]\n'
        root = build(tmp_path / "b", {"evals/a.toml": same, "evals/b.toml": same})
        assert problems(root) == ("evals/a.toml and evals/b.toml have the same id '@local/acme/r'",)

    def test_eval_may_declare_an_id(self, tmp_path):
        root = build(tmp_path / "b", {"evals/e.toml": 'id = "@local/acme/regression"\ntasks = ["t"]\n'})
        assert parse_bundle(root).entries[0].id == "@local/acme/regression"

    def test_long_bare_ids_are_not_echoed_whole(self, tmp_path):
        root = build(tmp_path / "b", {"envs/e/env.toml": f'id = "{"x" * 100_000}"\n'})
        assert len(problems(root)[0]) < 300


class TestEvals:
    @pytest.mark.parametrize("toml", ["", "tasks = []\n", 'id = "@local/acme/r"\n'],
                             ids=["an-empty-file", "an-empty-list", "no-tasks"])
    def test_an_eval_needs_at_least_one_task(self, tmp_path, toml):
        root = build(tmp_path / "b", {"evals/e.toml": toml})
        assert problems(root) == ("evals/e.toml: an eval needs at least one task in tasks = [...]",)

    @pytest.mark.parametrize(("toml", "problem"), [
        ('tasks = "t"\n', "tasks must be a list, not 't'"),
        ("tasks = { t = 1 }\n", "tasks must be a list, not {'t': 1}"),
        ('tasks = ["t"]\ndescription = "x"\n', "unknown key 'description'; an eval takes tasks, type and id"),
        ('tasks = ["t"]\ntype = 3\n', "type must be a non-empty string"),
    ])
    def test_a_malformed_eval_is_refused(self, tmp_path, toml, problem):
        root = build(tmp_path / "b", {"evals/e.toml": toml})
        assert problems(root) == (f"evals/e.toml: {problem}",)

    def test_an_evals_shape_problems_are_reported_together(self, tmp_path):
        root = build(tmp_path / "b", {"evals/e.toml": 'weight = 1\nname = "x"\n'})
        assert problems(root) == (
            "evals/e.toml: an eval needs at least one task in tasks = [...]",
            "evals/e.toml: unknown keys 'name', 'weight'; an eval takes tasks, type and id",
        )


class TestLength:
    def test_declared_id_cap(self, tmp_path):
        at_cap = LOCAL_PREFIX + "a" * (MAX_AUTHORED_LOCAL_ID_BYTES - len(LOCAL_PREFIX))
        ok = build(tmp_path / "ok", {"envs/t/env.toml": f'id = "{at_cap}"\n'})
        assert parse_bundle(ok).entries[0].id == at_cap
        for extra in ("a", "a" * 100):
            over = build(tmp_path / f"over{len(extra)}", {"envs/t/env.toml": f'id = "{at_cap}{extra}"\n'})
            assert f"over the {MAX_AUTHORED_LOCAL_ID_BYTES}-byte limit; shorten it" in problems(over)[0]

    def test_deep_root_cap(self, tmp_path):
        root = build(tmp_path / "b", {"envs/tickets/Dockerfile": ""})
        deep = LOCAL_PREFIX + "/".join(["d" * 200] * 20) + "/" + "d" * 40
        assert problems(root, id_root=deep)[0].endswith("-byte limit; shorten the name")

    def test_an_entry_point_root_must_leave_room_for_a_name(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/t.json": "[]"})
        full = LOCAL_PREFIX + "r" * (MAX_AUTHORED_LOCAL_ID_BYTES - len(LOCAL_PREFIX))
        (problem,) = problems(root, id_root=full)
        assert problem.endswith("leaves no room for a name; shorten it")


class TestFiles:
    def test_task_must_be_an_array_of_objects(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/a.json": '{"steps": []}', "tasks/b.json": "[1]"})
        assert problems(root) == (
            "tasks/a.json: a task file must be a JSON array of step objects",
            "tasks/b.json: a task file must be a JSON array of step objects",
        )

    def test_null_is_not_a_task(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/t.json": "null"})
        assert problems(root) == ("tasks/t.json: a task file must be a JSON array of step objects",)

    @pytest.mark.parametrize("rel, content", [
        ("tasks/t.json", "[" * 100_000 + "]" * 100_000),
        ("evals/e.toml", "a = " + "[" * 5000 + "]" * 5000 + "\n"),
    ], ids=["json", "toml"])
    def test_deep_nesting(self, tmp_path, rel, content):
        root = build(tmp_path / "b", {rel: content})
        assert problems(root)[0].endswith("nested too deeply")

    def test_duplicate_step_keys(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/t.json": '[{"id": "a", "id": "b"}]'})
        assert problems(root) == ("tasks/t.json: not valid JSON: duplicate keys ['id']",)

    def test_byte_order_marks(self, tmp_path):
        root = build(tmp_path / "b", {
            "tasks/t.json": b"\xef\xbb\xbf[]", "evals/e.toml": b'\xef\xbb\xbftasks = ["t"]\n',
            "envs/x/env.toml": b'\xef\xbb\xbftype = "multi"\n',
        })
        assert kinds(parse_bundle(root)) == [
            ("envs", "x", "multi"), ("tasks", "t", "task"), ("evals", "e", "eval"),
        ]

    def test_toml_syntax_error_names_the_line(self, tmp_path):
        root = build(tmp_path / "b", {"evals/e.toml": "tasks = []\nname = \n"})
        assert "not valid TOML" in problems(root)[0]
        assert "line" in problems(root)[0]

    def test_config_size_cap(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/t.json": b"[" + b" " * (8 << 20) + b"]"})
        assert "larger than the 8 MiB limit" in problems(root)[0]

    def test_repr_survives_huge_config_values(self, tmp_path):
        root = build(tmp_path / "b", {"envs/e/env.toml": "a = 0x" + "f" * 5000 + "\n"})
        assert "config=" not in repr(parse_bundle(root))

    def test_nan_is_not_json(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/t.json": '[{"cpu": NaN}]'})
        assert problems(root) == ("tasks/t.json: not valid JSON: NaN is not valid JSON",)


class TestSpecialEntries:
    def test_fifo_does_not_hang(self, tmp_path):
        root = build(tmp_path / "b", {"tasks/": None, "envs/e/": None})
        os.mkfifo(root / "tasks" / "t.json")
        os.mkfifo(root / "envs" / "e" / "env.toml")
        assert problems(root) == (
            "envs/e/env.toml: not a regular file",
            "tasks/t.json: neither a regular file nor a folder",
        )

    def test_broken_symlink_and_loop(self, tmp_path):
        root = build(tmp_path / "b", {"envs/": None})
        (root / "envs" / "broken").symlink_to(root / "nowhere")
        (root / "envs" / "loop").symlink_to(root / "envs" / "loop")
        assert problems(root) == ("envs/broken: a broken symlink", "envs/loop: a symlink loop, or too many links")

    def test_symlinks_are_followed(self, tmp_path):
        shared = build(tmp_path / "shared", {"tickets/Dockerfile": "FROM x", "hello.json": "[]"})
        root = build(tmp_path / "b", {"envs/": None, "tasks/": None})
        (root / "envs" / "tickets").symlink_to(shared / "tickets")
        (root / "tasks" / "hello.json").symlink_to(shared / "hello.json")
        assert [(entry.name, entry.path) for entry in parse_bundle(root).entries] == [
            ("tickets", root / "envs" / "tickets"), ("hello", root / "tasks" / "hello.json"),
        ]

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads anything")
    def test_unreadable_entity(self, tmp_path):
        root = build(tmp_path / "b", {"envs/e/Dockerfile": ""})
        (root / "envs" / "e").chmod(0)
        try:
            assert problems(root) == ("envs/e: permission denied",)
        finally:
            (root / "envs" / "e").chmod(0o755)

    @pytest.mark.parametrize("spec", [
        {"artifacts/x/a.txt": "", "artifacts/x/b.txt": ""},
        {"artifacts/x/artifact.toml": 'type = "vm_image"\n'},
        {"skills/s/SKILL.md": ""},
    ])
    def test_every_entry_of_an_uploaded_entity_is_checked(self, tmp_path, spec):
        root = build(tmp_path / "b", spec)
        entity = root / next(iter(spec)).rsplit("/", 1)[0]
        (entity / "broken").symlink_to(entity / "nowhere")
        os.mkfifo(entity / "pipe")
        rel = entity.relative_to(root)
        assert problems(root) == (
            f"{rel}/broken: a broken symlink", f"{rel}/pipe: neither a regular file nor a folder",
        )

    @pytest.mark.parametrize("kind", ["envs", "agents"])
    def test_a_build_context_follows_docker_rules(self, tmp_path, kind):
        root = build(tmp_path / "b", {f"{kind}/x/Dockerfile": "FROM x", f"{kind}/x/app.py": ""})
        (root / kind / "x" / "config.local.env").symlink_to(tmp_path / "nowhere")
        assert [entry.name for entry in parse_bundle(root).entries] == ["x"]


def test_a_bundle_error_is_summarized_by_its_first_problem_and_how_many_more():
    assert BundleError(["b: two", "a: one"]).summary == "a: one (and 1 more)"
    assert BundleError(["a: one"]).summary == "a: one"
