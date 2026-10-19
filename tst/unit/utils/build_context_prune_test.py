"""What a Dockerfile's build reads of its context, and writing each of several images only that part of a context
they share, from its tar.gz."""

import io
import tarfile
from pathlib import Path

import pytest

from agent_env.utils.build_context import ContextReads, context_reads, write_pruned


@pytest.mark.parametrize(("dockerfile", "paths"), [
    ("COPY gmail/requirements.txt .\nCOPY BaseService/ ./svc/BaseService/\n", ("gmail/requirements.txt", "BaseService")),
    ("FROM node AS cli\nFROM python\nCOPY --from=cli /usr/bin/node /usr/bin/node\nCOPY app /app\n", ("app",)),
    ("ADD https://example.com/f /f\nADD git@github.com:o/r.git /r\nADD local.tar.gz /x\n", ("local.tar.gz",)),
    ("COPY <<EOF /etc/x\nhi\nEOF\nCOPY a /a\n", ("a",)),
    ("COPY src/*.py /app/\n", ("src/*.py",)),
    ('COPY ["with space", "/app/"]\n', ("with space",)),
    ("COPY ./a/../b /x\n", ("b",)),
    ("RUN --mount=type=bind,source=tools,target=/t make\nRUN --mount=type=cache,target=/root/.cache pip install x\n"
     "RUN --mount=type=bind,from=build,source=/out,target=/o ls\n", ("tools",)),
], ids=["files-and-folders", "stage-copy", "add-remote-and-archive", "heredoc", "wildcard", "json", "normalized",
        "mounts"])
def test_a_build_reads_its_copy_and_add_sources_and_bind_mounts_from_the_context(dockerfile, paths):
    assert context_reads(f"FROM python\n{dockerfile}") == ContextReads(paths=paths)


@pytest.mark.parametrize(("dockerfile", "why"), [
    ("COPY . /app\n", "COPY . copies all of it"),
    ("RUN --mount=target=/src make\n", "RUN --mount binds ., all of it"),
    ("ARG SRC=a\nCOPY $SRC /x\n", "COPY $SRC names a variable"),
    ("COPY --parents a/b /x\n", "COPY --parents isn't followed"),
    ("COPY [broken\n", "COPY [broken can't be read"),
    ("COPY ../outside /x\n", "COPY ../outside copies all of it"),
], ids=["copy-dot", "bind-default-source", "variable", "unknown-flag", "unreadable", "outside"])
def test_a_build_that_reads_all_of_the_context_or_cant_be_told_says_why(dockerfile, why):
    assert context_reads(f"FROM python\n{dockerfile}") == ContextReads(whole=why)


def test_a_tab_a_continued_path_and_a_quoted_heredoc_are_read_as_docker_reads_them():
    dockerfile = ('FROM x\nCOPY\ta.txt /tab.txt\nCOPY pack\\\nage.json /joined.json\nRUN echo "<<EOF" > /q\n'
                  'RUN ["sh", "-c", "cat <<EOT"]\nCOPY b.txt /b.txt\n')

    assert context_reads(dockerfile) == ContextReads(paths=("a.txt", "package.json", "b.txt"))


def test_a_shell_here_string_leaves_the_copies_after_it_read():
    assert context_reads("FROM x\nRUN cat <<<word > /out\nCOPY a.txt /a.txt\n") == ContextReads(paths=("a.txt",))


def test_a_heredoc_left_open_reads_all_of_the_context_rather_than_miss_a_copy_it_swallowed():
    assert context_reads("FROM x\nRUN <<EOF\necho hi\nCOPY a /a\n") == ContextReads(
        whole="a heredoc in RUN never ends, so the rest can't be read")


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
    assert set(_files(tmp_path / "sub")) == {"sub/Dockerfile", "sub/Dockerfile.dockerignore", "app/main.py",
                                             "app/tests/t.py", "app/tests/keep.py"}


def test_a_dockerfile_that_reads_all_of_the_context_gets_what_docker_would_send_and_says_why(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY . /app\n", ".dockerignore": "*.log\n", "a.py": "a", "b.log": "b"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", ".dockerignore", "a.py"}
    assert pruned[tmp_path / "out"].whole == "COPY . copies all of it"


def test_a_kept_link_keeps_what_it_points_at_in_the_context_and_a_link_out_of_it_is_left_out(tmp_path):
    files = {"Dockerfile": "FROM python\nCOPY current/ /app/\nCOPY out /out\n", "v2/app.py": "2", "v1/app.py": "1"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"current": "v2", "out": "../../etc/passwd"})

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", "v2/app.py"}
    assert (tmp_path / "out" / "current").is_symlink()
    assert not (tmp_path / "out" / "out").exists()


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
    ("\\[x].txt", {"[x].txt"}), ("?.txt", {"!.txt", "a.txt", "b.txt"}),
], ids=["bang-is-a-character", "caret-negates", "range", "escaped-bracket", "any-one"])
def test_a_wildcard_class_reads_as_gos_filepath_match_reads_it(tmp_path, glob, kept):
    files = {"Dockerfile": f"FROM python\nCOPY {glob} /x/\n", "!.txt": "!", "a.txt": "a", "b.txt": "b", "[x].txt": "x"}
    archive = _archive(tmp_path / "ctx.tar.gz", files)

    write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"Dockerfile", *kept}


def test_a_malformed_wildcard_keeps_all_of_the_context(tmp_path):
    archive = _archive(tmp_path / "ctx.tar.gz", {"Dockerfile": "FROM python\nCOPY [a /x/\n", "a": "a", "b": "b"})

    assert write_pruned(archive, {tmp_path / "out": "Dockerfile"})[tmp_path / "out"].whole == (
        "[a matches nothing in the context")


def test_a_dockerfile_stored_as_a_link_is_read_where_it_leads(tmp_path):
    files = {"docker/real.Dockerfile": "FROM python\nCOPY app/ /app/\n", "app/main.py": "m", "other.py": "o"}
    archive = _archive(tmp_path / "ctx.tar.gz", files, links={"Dockerfile": "docker/real.Dockerfile"})

    pruned = write_pruned(archive, {tmp_path / "out": "Dockerfile"})

    assert set(_files(tmp_path / "out")) == {"docker/real.Dockerfile", "app/main.py"}
    assert (tmp_path / "out" / "Dockerfile").is_symlink() and pruned[tmp_path / "out"].whole is None


def test_a_bind_mount_after_a_tab_or_with_a_quoted_source_is_read():
    dockerfile = ('FROM x\nRUN --network=none\t--mount=type=bind,source=tools,target=/t ls\n'
                  'RUN --mount=type=bind,"source=my dir",target=/m ls\nRUN --mount=type=bind,source="my lib",target=/l ls\n')

    assert context_reads(dockerfile) == ContextReads(paths=("tools", "my dir", "my lib"))


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


def test_a_dockerfile_missing_from_the_archive_is_refused(tmp_path):
    archive = _archive(tmp_path / "ctx.tar.gz", {"a.py": "a"})

    with pytest.raises(ValueError, match="the build context has no gmail/Dockerfile"):
        write_pruned(archive, {tmp_path / "out": "gmail/Dockerfile"})


def test_a_member_no_context_holds_is_refused(tmp_path):
    archive = tmp_path / "ctx.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        dockerfile = b"FROM python\nCOPY . /app\n"
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        tar.addfile(info, io.BytesIO(dockerfile))
        hard = tarfile.TarInfo("hard")
        hard.type, hard.linkname = tarfile.LNKTYPE, "Dockerfile"
        tar.addfile(hard)

    with pytest.raises(ValueError, match="neither a file, a folder nor a link"):
        write_pruned(archive, {tmp_path / "out": "Dockerfile"})
