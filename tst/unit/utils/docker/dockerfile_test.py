"""A Dockerfile read into instructions as docker build reads it, a COPY or ADD's arguments, and what it reads."""

import pytest

from agent_env.utils.docker.dockerfile import (ContextReads, context_reads, copy_arguments, directives, instructions,
                                              leading_flags)

DOCKERFILE = """# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
# a comment
RUN apt-get update \\
    # a comment among the continued lines
    && apt-get install -y curl

COPY --chown=1:1 a.txt \\
     b/ /app/
RUN <<EOF
echo hi \\
EOF
COPY <<-EOT /etc/note
\tnote
\tEOT
"""


def test_the_instructions_joined_are_the_dockerfile_and_each_reads_as_docker_reads_it():
    found = instructions(DOCKERFILE)

    assert "".join(instruction.text for instruction in found) == DOCKERFILE
    assert [(i.keyword, i.arguments) for i in found if i.keyword] == [
        ("FROM", "python:3.12-slim AS base"),
        ("RUN", "apt-get update     && apt-get install -y curl"),
        ("COPY", "--chown=1:1 a.txt      b/ /app/"),
        ("RUN", "<<EOF"),
        ("COPY", "<<-EOT /etc/note"),
    ]
    assert [i.heredocs for i in found if i.heredocs] == [("echo hi \\\n",), ("\tnote\n",)]


def test_the_escape_directive_changes_what_continues_a_line():
    found = instructions("# escape=`\nFROM windows\nCOPY a.txt `\n  C:\\app\\\n")

    assert [(i.keyword, i.arguments) for i in found if i.keyword] == [("FROM", "windows"), ("COPY", "a.txt   C:\\app\\")]


def test_the_directives_a_dockerfile_opens_with_by_name():
    assert directives("# syntax=docker/dockerfile:1\n# Escape=`\nFROM x\n# check=skip=all\n") == {
        "syntax": "docker/dockerfile:1", "escape": "`"}


def test_a_directive_after_the_first_instruction_is_a_comment():
    found = instructions("FROM x\n# escape=`\nCOPY a `\n  b\n")

    assert [(i.keyword, i.arguments) for i in found if i.keyword] == [("FROM", "x"), ("COPY", "a `"), ("B", "")]


def test_words_split_on_any_whitespace_and_a_continued_line_joins_on_as_it_is():
    # docker build copies package.json for the second COPY: the escaped line break leaves no space behind.
    found = instructions("FROM x\nCOPY\ta.txt /tab.txt\nCOPY pack\\\nage.json /joined.json\nCOPY a \\  \n  b /c\n")

    assert [(i.keyword, i.arguments) for i in found if i.keyword] == [
        ("FROM", "x"), ("COPY", "a.txt /tab.txt"), ("COPY", "package.json /joined.json"), ("COPY", "a   b /c")]


def test_a_heredoc_is_only_ever_opened_outside_quotes_in_shell_form():
    # docker build writes the quoted <<EOF as text, opens no heredoc for an exec-form one, and runs the COPY after both.
    found = instructions('FROM x\nRUN echo "<<EOF" > /q\nRUN ["sh", "-c", "cat <<EOT"]\nRUN echo \'<<A\' > /a\n'
                         'COPY a.txt /a.txt\n')

    assert [(i.keyword, i.heredocs) for i in found if i.keyword] == [("FROM", ()), ("RUN", ()), ("RUN", ()), ("RUN", ()),
                                                                      ("COPY", ())]


@pytest.mark.parametrize(("line", "heredocs"), [
    ("RUN <<EOF.sh\necho hi\nEOF.sh\n", ("echo hi\n",)),
    ("RUN <<'EOF.x'\necho $HOME\nEOF.x\n", ("echo $HOME\n",)),
    ("RUN 3<<EOF cat <&3\nhello\nEOF\n", ("hello\n",)),
    ("RUN cat<<EOF > /x\n", ()),
], ids=["dotted-delimiter", "quoted-dotted", "file-descriptor", "no-space-before"])
def test_a_heredoc_is_a_word_opening_with_lt_lt_and_its_delimiter_is_the_rest_of_that_word(line, heredocs):
    # docker build closes <<EOF.sh at EOF.sh, opens one for 3<<EOF, and none for cat<<EOF.
    [instruction] = [i for i in instructions(f"FROM x\n{line}") if i.keyword == "RUN"]

    assert instruction.heredocs == heredocs


def test_a_copy_after_a_heredoc_whose_delimiter_holds_a_dot_is_read():
    dockerfile = "FROM x\nRUN <<EOF.sh\necho hi\nEOF.sh\nCOPY app /app\nRUN <<EOF\necho\nEOF\n"

    assert context_reads(dockerfile) == ContextReads(paths=("app",))


def test_flags_split_on_any_whitespace_outside_quotes():
    assert leading_flags('--network=none\t--mount=type=bind,"source=my dir",target=/m  cp /m/f /f') == (
        ["--network=none", '--mount=type=bind,"source=my dir",target=/m'], "cp /m/f /f")
    args = copy_arguments(next(i for i in instructions("COPY --chown=0:0\ta.txt /a\n") if i.keyword))
    assert (args.flags, args.sources) == (("--chown=0:0",), ("a.txt",))


def test_a_shell_here_string_is_no_heredoc_and_the_instructions_after_it_stand():
    # docker build runs ``RUN cat <<<word`` as written and reads the COPY after it as its own instruction.
    found = instructions("FROM x\nRUN cat <<<word > /out\nCOPY a.txt /a.txt\nword\n")

    assert [(i.keyword, i.heredocs) for i in found if i.keyword] == [("FROM", ()), ("RUN", ()), ("COPY", ()),
                                                                      ("WORD", ())]


def test_a_heredoc_left_open_runs_to_the_end_and_says_so():
    found = [i for i in instructions("FROM x\nRUN <<EOF\necho hi\nCOPY a /a\n") if i.keyword]

    assert [(i.keyword, i.complete) for i in found] == [("FROM", True), ("RUN", False)]
    assert found[1].heredocs == ("echo hi\nCOPY a /a\n",)


@pytest.mark.parametrize(("line", "flags", "sources", "destination", "json_form", "heredoc"), [
    ("COPY a b /dst/", (), ("a", "b"), "/dst/", False, False),
    ('COPY --chown=1:1 --link ["a b", "c", "/dst"]', ("--chown=1:1", "--link"), ("a b", "c"), "/dst", True, False),
    ("ADD --checksum=sha256:00 https://x/f /f", ("--checksum=sha256:00",), ("https://x/f",), "/f", False, False),
    ("COPY --from=build /out /out", ("--from=build",), ("/out",), "/out", False, False),
    ("COPY <<EOF /etc/x", (), (), "/etc/x", False, True),
], ids=["shell", "json", "add-url", "from", "heredoc"])
def test_a_copy_or_adds_arguments(line, flags, sources, destination, json_form, heredoc):
    [instruction] = [i for i in instructions(f"{line}\nEOF\n" if heredoc else f"{line}\n") if i.keyword]
    args = copy_arguments(instruction)

    assert (args.keyword, args.flags, args.sources, args.destination, args.json_form, args.heredoc) == (
        line.split()[0], flags, sources, destination, json_form, heredoc)


@pytest.mark.parametrize("line", ['COPY ["a", 1]', "COPY [broken", 'COPY "a b" /dst', "COPY onlyone", "RUN cp a b"],
                         ids=["json-not-strings", "bad-json", "quoted-shell", "no-destination", "not-a-copy"])
def test_arguments_it_cant_read_are_none(line):
    [instruction] = [i for i in instructions(f"{line}\n") if i.keyword]

    assert copy_arguments(instruction) is None


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


@pytest.mark.parametrize(("line", "why"), [
    ("COPY \\[x].txt /x/", "COPY \\[x].txt holds an escape"),
    ('COPY ["a\\\\b", "/x/"]', "COPY a\\b holds an escape"),
    ("RUN --mount=type=bind,source=a\\b,target=/m ls", "RUN --mount source a\\b holds an escape"),
], ids=["shell", "json", "mount"])
def test_a_source_holding_an_escape_keeps_all_of_the_context(line, why):
    # docker build unescapes a source before matching it: COPY \[x].txt copies x.txt, not [x].txt.
    assert context_reads(f"FROM x\n{line}\n") == ContextReads(whole=why)


def test_a_heredoc_left_open_reads_all_of_the_context_rather_than_miss_a_copy_it_swallowed():
    assert context_reads("FROM x\nRUN <<EOF\necho hi\nCOPY a /a\n") == ContextReads(
        whole="a heredoc in RUN never ends, so the rest can't be read")


@pytest.mark.parametrize("syntax", ["docker/dockerfile:1", "docker.io/docker/dockerfile:1.7-labs",
                                    "docker/dockerfile-upstream:master", f"docker/dockerfile:1@sha256:{'0' * 64}"])
def test_a_syntax_directive_naming_dockers_own_frontend_is_read(syntax):
    assert context_reads(f"# syntax={syntax}\nFROM x\nCOPY a /a\n") == ContextReads(paths=("a",))


def test_a_frontend_of_its_own_keeps_all_of_the_context():
    assert context_reads("# syntax=example.com/acme/frontend:1\nFROM x\nCOPY a /a\n") == ContextReads(
        whole="# syntax=example.com/acme/frontend:1 isn't followed")


def test_mounts_that_never_read_the_context_add_nothing_and_options_read_in_any_case():
    dockerfile = ("FROM x\nRUN --mount=type=cache,target=/c --mount=type=secret,id=k --mount=type=ssh "
                  "--mount=type=tmpfs,target=/t ls\nRUN --mount=TYPE=BIND,Source=tools,target=/m ls\n")

    assert context_reads(dockerfile) == ContextReads(paths=("tools",))


def test_a_mount_type_it_doesnt_know_keeps_all_of_the_context():
    assert context_reads("FROM x\nRUN --mount=type=future,source=a,target=/a ls\n") == ContextReads(
        whole="RUN --mount type=future isn't followed")


def test_a_bind_mount_after_a_tab_or_with_a_quoted_source_is_read():
    dockerfile = ('FROM x\nRUN --network=none\t--mount=type=bind,source=tools,target=/t ls\n'
                  'RUN --mount=type=bind,"source=my dir",target=/m ls\nRUN --mount=type=bind,source="my lib",target=/l ls\n')

    assert context_reads(dockerfile) == ContextReads(paths=("tools", "my dir", "my lib"))
