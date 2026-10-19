"""``write_pruned``: each image's folder holding only what its build reads of a stored context."""

import io
import tarfile
import time
from pathlib import Path

import pytest

from agent_env.utils.docker.prune import write_pruned


def _archive(path: Path, files: dict[str, str], links: dict[str, str] | None = None,
             executable: tuple[str, ...] = ()) -> Path:
    """A context tar.gz holding ``files`` and ``links``, everything in it as another tool might have written it."""
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            data = text.encode()
            info.size, info.mode = len(data), 0o755 if name in executable else 0o644
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
    return path


def _files(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): path.read_text() for path in sorted(folder.rglob("*"))
            if path.is_file() and not path.is_symlink()}


SERVERS = {
    "__init__.py": "", "BaseService/base.py": "base", "wheels/x.whl": "wheel", "gsuite/g.py": "g",
    "gmail/Dockerfile": "FROM python\nCOPY __init__.py BaseService/ gsuite/ gmail/ wheels/ /app/\n",
    "gmail/main.py": "gmail", "slack/Dockerfile": "FROM python\nCOPY BaseService/ /app/BaseService/\nCOPY slack/ /s/\n",
    "slack/main.py": "slack", "websites/huge.bin": "x" * 4096,
}


def test_each_image_gets_only_what_its_dockerfile_reads_from_one_archive_they_share(tmp_path):
    archive = _archive(tmp_path / "ctx.tar.gz", SERVERS, executable=("gmail/main.py",))

    pruned = write_pruned(archive, {tmp_path / "gmail": "gmail/Dockerfile", tmp_path / "slack": "slack/Dockerfile"})

    assert _files(tmp_path / "gmail") == {name: SERVERS[name] for name in (
        "BaseService/base.py", "__init__.py", "gmail/Dockerfile", "gmail/main.py", "gsuite/g.py", "wheels/x.whl")}
    assert _files(tmp_path / "slack") == {name: SERVERS[name] for name in (
        "BaseService/base.py", "slack/Dockerfile", "slack/main.py")}
    assert (tmp_path / "gmail" / "gmail" / "main.py").stat().st_mode & 0o111
    assert pruned[tmp_path / "slack"].files == 3 and pruned[tmp_path / "slack"].whole is None
    assert pruned[tmp_path / "slack"].size == sum(len(SERVERS[name]) for name in (
        "BaseService/base.py", "slack/Dockerfile", "slack/main.py"))


def test_what_the_ignore_file_leaves_out_stays_out_and_a_dockerfiles_own_ignore_file_replaces_the_contexts(tmp_path):
    files = {
        "Dockerfile": "FROM python\nCOPY app/ /app/\n", ".dockerignore": "app/tests\n!app/tests/keep.py\n",
        "app/main.py": "m", "app/tests/t.py": "t", "app/tests/keep.py": "k",
        "sub/Dockerfile": "FROM python\nCOPY app/ /app/\n", "sub/Dockerfile.dockerignore": "# only notes left out\n**/*.md\n",
        "app/notes.md": "n",
    }
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    write_pruned(archive, {tmp_path / "root": "Dockerfile", tmp_path / "sub": "sub/Dockerfile"})

    assert set(_files(tmp_path / "root")) == {"Dockerfile", ".dockerignore", "app/main.py", "app/tests/keep.py",
                                              "app/notes.md"}
    assert set(_files(tmp_path / "sub")) == {"sub/Dockerfile", "sub/Dockerfile.dockerignore", ".dockerignore",
                                             "app/main.py", "app/tests/t.py", "app/tests/keep.py"}


def test_an_empty_dockerfiles_own_ignore_file_leaves_everything_in_and_both_ignore_files_stay(tmp_path):
    # docker build's own frontend reads an empty Dockerfile.dockerignore as leaving nothing out; the docker/dockerfile:1
    # frontend reads the context's .dockerignore instead. Both files stay, so either frontend reads what it would.
    files = {"Dockerfile": "FROM python\nCOPY app/ /app/\n", "Dockerfile.dockerignore": "", ".dockerignore": "**/*.md\n",
             "app/main.py": "m", "app/notes.md": "n"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "Dockerfile.dockerignore", ".dockerignore", "app/main.py",
                                             "app/notes.md"}


def test_a_dockerfile_that_reads_all_of_the_context_gets_what_docker_would_send_and_says_why(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY . /app\n", ".dockerignore": "*.log\n", "a.py": "a", "b.log": "b"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", ".dockerignore", "a.py"}
    assert pruned[tmp_path / "out"].whole == "COPY . copies all of it"


def test_a_kept_link_keeps_what_it_points_at_and_one_leading_out_of_the_context_is_written_as_it_is(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY current/ /app/\nCOPY lib/ /lib/\n", "v2/app.py": "2", "v1/app.py": "1",
             "lib/a.py": "a"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"current": "v2", "lib/out": "../../../etc/passwd",
                                                              "lib/abs": "/usr/share/zoneinfo/UTC"})

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "v2/app.py", "lib/a.py"}
    assert (tmp_path / "out" / "current").is_symlink()
    # docker build copies a link as a link, wherever it leads.
    assert [(tmp_path / "out" / "lib" / name).readlink().as_posix() for name in ("out", "abs")] == [
        "../../../etc/passwd", "/usr/share/zoneinfo/UTC"]


def test_a_link_is_followed_as_docker_build_follows_it_its_root_standing_for_slash_and_any_dotdot_above_it(tmp_path):
    # docker build copies a.txt, conf/f.cfg and a.txt again for this Dockerfile.
    files = {"Dockerfile": "FROM python\nCOPY sub/deep/up /x/up\nCOPY sub/abs/f.cfg /x/\nCOPY sub/top/a.txt /x/top\n",
             "a.txt": "a", "b.txt": "b", "conf/f.cfg": "f", "conf/g.cfg": "g"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"sub/deep/up": "../../../a.txt", "sub/abs": "/conf",
                                                              "sub/top": "../../../../.."})

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})[tmp_path / "out"]

    assert pruned.whole is None
    assert set(_files(tmp_path / "out")) == {"Dockerfile", "a.txt", "conf/f.cfg"}
    assert all((tmp_path / "out" / name).is_symlink() for name in ("sub/deep/up", "sub/abs", "sub/top"))


def test_nothing_is_written_through_a_link_that_leads_out_of_the_folder(tmp_path):
    archive = tmp_path / "ctx.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        dockerfile = b"FROM python\nCOPY . /app\n"
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        tar.addfile(info, io.BytesIO(dockerfile))
        link = tarfile.TarInfo("evil")
        link.type, link.linkname = tarfile.SYMTYPE, str(tmp_path / "elsewhere")
        tar.addfile(link)
        through = tarfile.TarInfo("evil/planted")
        through.size = 1
        tar.addfile(through, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="evil/planted"):
        write_pruned(archive, {tmp_path / "out": "Dockerfile"})
    assert not (tmp_path / "elsewhere" / "planted").exists()


def test_a_source_reached_through_a_link_keeps_the_link_and_what_it_leads_to(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY current/app.py /app.py\n", "releases/v2/app.py": "2",
             "releases/v1/app.py": "1"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"current": "stable", "stable": "releases/v2"})

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "releases/v2/app.py"}
    assert (tmp_path / "out" / "current").is_symlink() and (tmp_path / "out" / "stable").is_symlink()
    assert pruned[tmp_path / "out"].whole is None


def test_a_link_loop_or_one_out_of_the_context_in_a_source_keeps_all_of_it(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY loop/a.py /a.py\nCOPY out/b.py /b.py\n", "other.py": "o"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"loop": "loop2", "loop2": "loop", "out": "../elsewhere"})

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert pruned[tmp_path / "out"].whole == "loop/a.py matches nothing in the context"


@pytest.mark.parametrize(("glob", "kept"), [
    ("[!a].txt", {"!.txt", "a.txt"}), ("[^a].txt", {"!.txt", "b.txt"}), ("[a-b].txt", {"a.txt", "b.txt"}),
    ("[!-a].txt", {"!.txt", "a.txt"}), ("?.txt", {"!.txt", "a.txt", "b.txt"}),
], ids=["bang-is-a-character", "caret-negates", "range", "range-from-bang", "any-one"])
def test_a_wildcard_class_reads_as_gos_filepath_match_reads_it(tmp_path, glob, kept):
    files = {"Dockerfile": f"FROM python\nCOPY {glob} /x/\n", "!.txt": "!", "a.txt": "a", "b.txt": "b", "[x].txt": "x"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", *kept}


@pytest.mark.parametrize("glob", ["[a", "[z-a].txt", "[^z-a].txt", "[-a].txt", "[a-].txt", "[a-b-c].txt", "[]a].txt"],
                         ids=["unclosed", "reversed", "negated-reversed", "leading-dash", "trailing-dash", "two-ranges",
                              "leading-bracket"])
def test_a_wildcard_docker_build_refuses_keeps_all_of_the_context_rather_than_fail(tmp_path, glob):
    # docker build refuses each of these as a syntax error in the pattern.
    files = {"Dockerfile": f"FROM python\nCOPY {glob} /x/\n", "a.txt": "a", "b.txt": "b", "z.txt": "z"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    assert write_pruned(archive, {tmp_path / "out": "Dockerfile"})[tmp_path / "out"].whole == (
        f"{glob} matches nothing in the context")


def test_a_dockerfile_stored_as_a_link_is_read_where_it_leads(tmp_path):
    files = {"docker/real.Dockerfile": "FROM python\nCOPY app/ /app/\n", "app/main.py": "m", "other.py": "o"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"Dockerfile": "docker/real.Dockerfile"})

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"docker/real.Dockerfile", "app/main.py"}
    assert (tmp_path / "out" / "Dockerfile").is_symlink() and pruned[tmp_path / "out"].whole is None


def test_wildcards_match_one_part_of_a_path_as_dockers_do(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY *.txt [^b]*.cfg /x/\n", "a.txt": "a", "sub/c.txt": "c", "a.cfg": "a",
             "b.cfg": "b"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "a.txt", "a.cfg"}


def test_what_the_os_or_python_leaves_behind_in_a_copied_folder_is_left_out(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY lib/ /lib/\n", "lib/mod.py": "m", "lib/__pycache__/mod.cpython-312.pyc": "c",
             "lib/.DS_Store": "d"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "lib/mod.py"}


def test_a_source_that_matches_nothing_in_the_context_keeps_all_of_it_and_says_which(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY app/ /app/\nCOPY gone.txt /gone\n", "app/main.py": "m", "other.py": "o"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "app/main.py", "other.py"}
    assert pruned[tmp_path / "out"].whole == "gone.txt matches nothing in the context"


def test_a_chain_of_kept_links_keeps_what_its_last_link_points_at(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY current /app\n", "releases/v3/app.py": "3", "releases/v2/app.py": "2"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"current": "stable", "stable": "releases/v3"})

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "releases/v3/app.py"}
    assert (tmp_path / "out" / "stable").is_symlink()


def test_an_image_naming_no_dockerfile_builds_the_contexts_or_the_only_one_in_it(tmp_path):
    root = _archive(tmp_path / "root.tar.gz", {"Dockerfile": "FROM python\nCOPY a /a\n", "a": "a",
                                                "sub/Dockerfile": "FROM python\n"})
    nested = tmp_path / "nested.tar.gz"
    with tarfile.open(nested, "w:gz") as tar:  # the only Dockerfile, written three times as a stored context had it
        for text in ("FROM python\n", "FROM python\nCOPY b /b\n", "FROM python\nCOPY app/ /app/\n"):
            info = tarfile.TarInfo("claude_code/Dockerfile")
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text.encode()))
        info = tarfile.TarInfo("app/main.py")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"m"))

    from_root = write_pruned(root, {tmp_path / "root": None})[tmp_path / "root"]
    from_nested = write_pruned(nested, {tmp_path / "nested": None})[tmp_path / "nested"]

    assert from_root.dockerfile == "Dockerfile" and set(_files(tmp_path / "root")) == {"Dockerfile", "a"}
    assert from_nested.dockerfile == "claude_code/Dockerfile"
    assert _files(tmp_path / "nested") == {"claude_code/Dockerfile": "FROM python\nCOPY app/ /app/\n",
                                           "app/main.py": "m"}


@pytest.mark.parametrize(("files", "problem"), [
    ({"a/Dockerfile": "FROM x\n", "b/Dockerfile": "FROM x\n"}, "holds 2: a/Dockerfile, b/Dockerfile; name the one"),
    ({"a.py": "a"}, "names no Dockerfile and holds 0; name the one it builds"),
], ids=["two", "none"])
def test_an_image_naming_no_dockerfile_is_refused_when_the_context_doesnt_settle_which(tmp_path, files, problem):
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    with pytest.raises(ValueError, match=problem):
        write_pruned(archive, {tmp_path / "out": None})


def test_each_image_names_its_largest_files_for_the_report(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY data/ /data/\n",
             **{f"data/f{n}.bin": "x" * (100 * n) for n in range(1, 8)}, "big-unread.bin": "x" * 10_000}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})[tmp_path / "out"]

    assert pruned.largest == (("data/f7.bin", 700), ("data/f6.bin", 600), ("data/f5.bin", 500), ("data/f4.bin", 400),
                              ("data/f3.bin", 300))
    assert pruned.dockerfile == "Dockerfile"


def test_a_name_the_archive_holds_twice_counts_once_at_the_size_left_in_the_folder(tmp_path):
    archive = tmp_path / "ctx.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in (("Dockerfile", "FROM python\nCOPY data/ /data/\n"), ("data/a.bin", "x" * 5000),
                           ("data/b.bin", "x" * 100), ("data/a.bin", "x" * 10)):
            info = tarfile.TarInfo(name)
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text.encode()))

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})[tmp_path / "out"]

    assert (tmp_path / "out" / "data" / "a.bin").stat().st_size == 10
    assert (pruned.files, pruned.size) == (3, 30 + 100 + 10)
    assert pruned.largest == (("data/b.bin", 100), ("Dockerfile", 30), ("data/a.bin", 10))


def test_the_dockerfiles_link_target_is_kept_even_when_the_ignore_file_leaves_it_out(tmp_path):
    # docker build reads the Dockerfile and its ignore file where they lead, whatever the ignore file says.
    files = {"docker/real.Dockerfile": "FROM python\nCOPY app/ /app/\n", "docker/Dockerfile.dockerignore": "docker\n",
             "app/main.py": "m"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"docker/Dockerfile": "real.Dockerfile"})

    write_pruned(archive, {tmp_path / "out": "docker/Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"docker/real.Dockerfile", "docker/Dockerfile.dockerignore", "app/main.py"}


@pytest.mark.parametrize("dockerfile", ["FROM python\nCOPY . /app\n", "FROM python\nCOPY sub/ /app/\n"],
                         ids=["whole", "copied"])
def test_a_folder_the_archive_only_implies_is_written_when_the_ignore_file_leaves_out_all_it_holds(tmp_path, dockerfile):
    # docker build sends sub/ as an empty folder here, and COPY sub/ copies it.
    archive = _archive(tmp_path / "ctx.tar.gz", {"Dockerfile": dockerfile, ".dockerignore": "sub/*.md\n",
                                                 "sub/notes.md": "n"})

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert (tmp_path / "out" / "sub").is_dir() and not any((tmp_path / "out" / "sub").iterdir())


def test_a_kept_links_target_is_followed_through_the_links_on_its_way(tmp_path):
    # docker build copies releases/v2/app/main.py for COPY current here.
    files = {"Dockerfile": "FROM python\nCOPY current /app\n", "releases/v2/app/main.py": "2", "releases/v1/app/old.py": "1"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"current": "releases/latest/app", "releases/latest": "v2"})

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "releases/v2/app/main.py"}
    assert (tmp_path / "out" / "current").is_symlink() and (tmp_path / "out" / "releases" / "latest").is_symlink()


def test_no_folder_sharing_the_archive_is_written_through_a_link_that_leads_out_of_it(tmp_path):
    # The link leads into the first folder's own safe/, but out of the second one, into a folder beside it.
    (tmp_path / "b" / "one" / "safe").mkdir(parents=True)
    archive = tmp_path / "ctx.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        dockerfile = b"FROM python\nCOPY . /app\n"
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        tar.addfile(info, io.BytesIO(dockerfile))
        folder = tarfile.TarInfo("safe")
        folder.type, folder.mode = tarfile.DIRTYPE, 0o755
        tar.addfile(folder)
        link = tarfile.TarInfo("dir")
        link.type, link.linkname = tarfile.SYMTYPE, "../one/safe"
        tar.addfile(link)
        planted = tarfile.TarInfo("dir/file")
        planted.size = 1
        tar.addfile(planted, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="'dir/file' in the build context leads out of"):
        write_pruned(archive, {tmp_path / "a" / "one": "Dockerfile", tmp_path / "b" / "two": "Dockerfile"})
    assert not (tmp_path / "b" / "one" / "safe" / "file").exists()


def test_a_member_whose_name_leads_out_of_the_context_is_refused_before_anything_is_written(tmp_path):
    archive = _archive(tmp_path / "ctx.tar.gz", {"Dockerfile": "FROM python\nCOPY . /app\n",
                                                 "../outside/__pycache__/payload": "x"})

    with pytest.raises(ValueError, match="leads out of it"):
        write_pruned(archive, {tmp_path / "out" / "here": "Dockerfile"})
    assert not (tmp_path / "out" / "outside").exists() and not (tmp_path / "out" / "here").exists()


def test_each_file_is_written_at_the_time_of_writing_not_the_stored_contexts(tmp_path):
    # A stored context's files carry the time 0; docker build would read a rewritten one as unchanged.
    archive = _archive(tmp_path / "ctx.tar.gz", SERVERS)
    before = time.time()

    write_pruned(archive, {tmp_path / "gmail": "gmail/Dockerfile", tmp_path / "slack": "slack/Dockerfile"})

    for name in ("gmail/BaseService/base.py", "slack/BaseService/base.py", "slack/slack/main.py"):
        assert (tmp_path / name).stat().st_mtime >= before - 1


def test_a_dockerfile_missing_from_the_archive_is_refused(tmp_path):
    archive = _archive(tmp_path / "ctx.tar.gz", {"a.py": "a"})

    with pytest.raises(ValueError, match="the build context has no gmail/Dockerfile"):
        write_pruned(archive, {tmp_path / "out": "gmail/Dockerfile"})


def _with_members(path: Path, members: list[tuple[str, bytes, str | None]]) -> Path:
    """A context tar.gz of ``members``, each a name, its content and, for a hard link, the name it links to."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data, target in members:
            info = tarfile.TarInfo(name)
            if target is not None:
                info.type, info.linkname = tarfile.LNKTYPE, target
            else:
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data) if target is None else None)
    return path


def test_a_member_no_context_holds_is_refused(tmp_path):
    archive = tmp_path / "ctx.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        dockerfile = b"FROM python\nCOPY . /app\n"
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        tar.addfile(info, io.BytesIO(dockerfile))
        pipe = tarfile.TarInfo("pipe")
        pipe.type = tarfile.FIFOTYPE
        tar.addfile(pipe)

    with pytest.raises(ValueError, match="neither a file, a folder nor a link"):
        write_pruned(archive, {tmp_path / "out": "Dockerfile"})


def test_a_dockerfile_stored_as_a_hard_link_to_itself_is_read_and_written_as_the_file(tmp_path):
    dockerfile = b"FROM python\nCOPY contacts/main.py /app/\n"
    archive = _with_members(tmp_path / "ctx.tar.gz", [("contacts/Dockerfile", dockerfile, None),
                                                      ("contacts/main.py", b"main", None),
                                                      ("contacts/Dockerfile", b"", "contacts/Dockerfile")])

    pruned = write_pruned(archive, {tmp_path / "out": "contacts/Dockerfile"})

    assert _files(tmp_path / "out") == {"contacts/Dockerfile": dockerfile.decode(), "contacts/main.py": "main"}
    assert pruned[tmp_path / "out"].size == len(dockerfile) + 4


def test_a_hard_link_is_written_as_a_file_of_its_own_holding_what_it_links_to_kept_or_not(tmp_path):
    archive = _with_members(tmp_path / "ctx.tar.gz", [("Dockerfile", b"FROM python\nCOPY app/ /app/\n", None),
                                                      ("data/real.txt", b"real", None),
                                                      ("app/kept.txt", b"kept", None),
                                                      ("app/twin.txt", b"", "app/kept.txt"),
                                                      ("app/far.txt", b"", "data/real.txt")])

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    out = tmp_path / "out"
    assert _files(out) == {"Dockerfile": "FROM python\nCOPY app/ /app/\n", "app/kept.txt": "kept",
                           "app/twin.txt": "kept", "app/far.txt": "real"}
    assert not (out / "data").exists()
    assert (out / "app/kept.txt").stat().st_ino != (out / "app/twin.txt").stat().st_ino
