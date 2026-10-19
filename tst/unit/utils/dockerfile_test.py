"""A Dockerfile read into instructions as docker build reads it, and a COPY or ADD's arguments."""

import pytest

from agent_env.utils.dockerfile import copy_arguments, instructions

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
        ("RUN", "apt-get update && apt-get install -y curl"),
        ("COPY", "--chown=1:1 a.txt b/ /app/"),
        ("RUN", "<<EOF"),
        ("COPY", "<<-EOT /etc/note"),
    ]
    assert [i.heredocs for i in found if i.heredocs] == [("echo hi \\\n",), ("\tnote\n",)]


def test_the_escape_directive_changes_what_continues_a_line():
    found = instructions("# escape=`\nFROM windows\nCOPY a.txt `\n  C:\\app\\\n")

    assert [(i.keyword, i.arguments) for i in found if i.keyword] == [("FROM", "windows"), ("COPY", "a.txt C:\\app\\")]


def test_a_directive_after_the_first_instruction_is_a_comment():
    found = instructions("FROM x\n# escape=`\nCOPY a `\n  b\n")

    assert [(i.keyword, i.arguments) for i in found if i.keyword] == [("FROM", "x"), ("COPY", "a `"), ("B", "")]


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
