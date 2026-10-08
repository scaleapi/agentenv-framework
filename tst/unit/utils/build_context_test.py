"""A build context's tar.gz and digest depend on its files alone: what docker build sends, never the machine."""

import gzip
import io
import os
import tarfile
from pathlib import Path

import pytest

from agent_env.utils.build_context import BuildContext


def _tree(root: Path, files: dict[str, str], executable: tuple[str, ...] = ()) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    for name in executable:
        (root / name).chmod(0o755)
    return root


SERVER = {"Dockerfile": "FROM python:3.12\nCOPY . /app\n", "app/main.py": "print(1)\n", "app/run.sh": "#!/bin/sh\n",
          "data/empty/.keep": ""}


def _written(context: BuildContext, tmp_path: Path, name: str = "ctx.tar.gz") -> bytes:
    out = tmp_path / name
    context.write(out)
    return out.read_bytes()


def _members(tar_gz: bytes) -> list[tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(tar_gz), mode="r:gz") as tar:
        return tar.getmembers()


def test_the_same_files_give_the_same_bytes_and_digest_whatever_their_times_and_umask(tmp_path):
    first = _tree(tmp_path / "a", SERVER, executable=("app/run.sh",))
    second = _tree(tmp_path / "b", SERVER, executable=("app/run.sh",))
    (second / "app" / "main.py").chmod(0o664)
    os.utime(second / "Dockerfile", (1, 1))

    one, two = BuildContext.of(first, first / "Dockerfile"), BuildContext.of(second, second / "Dockerfile")

    assert _written(one, tmp_path, "1.tar.gz") == _written(two, tmp_path, "2.tar.gz")
    assert one.source_digest("linux/amd64") == two.source_digest("linux/amd64")


def test_the_tar_has_sorted_relative_names_git_modes_and_nothing_of_the_machine(tmp_path):
    root = _tree(tmp_path / "ctx", SERVER, executable=("app/run.sh",))

    data = _written(BuildContext.of(root, root / "Dockerfile"), tmp_path)

    members = _members(data)
    assert [m.name for m in members] == ["Dockerfile", "app", "app/main.py", "app/run.sh", "data", "data/empty",
                                         "data/empty/.keep"]
    assert {(m.name, m.mode) for m in members if m.name.startswith("app")} == {
        ("app", 0o755), ("app/main.py", 0o644), ("app/run.sh", 0o755)}
    assert {(m.mtime, m.uid, m.gid, m.uname, m.gname) for m in members} == {(0, 0, 0, "", "")}
    assert data[3] & 0x08 == 0 and data[4:8] == b"\0\0\0\0"  # gzip: no file name, no time


def test_what_the_dockerignore_excludes_is_left_out_but_the_dockerfile_and_ignore_file_stay(tmp_path):
    root = _tree(tmp_path / "ctx", {**SERVER, ".dockerignore": "node_modules\n*.log\n!keep.log\nDockerfile\n.dockerignore\n",
                                    "node_modules/x/index.js": "", "debug.log": "", "keep.log": "", ".env": "A=1"})

    names = [entry.name for entry in BuildContext.of(root, root / "Dockerfile").entries]

    assert names == [".dockerignore", ".env", "Dockerfile", "app", "app/main.py", "app/run.sh", "data", "data/empty",
                     "data/empty/.keep", "keep.log"]


def test_a_dockerfiles_own_ignore_file_replaces_the_contexts(tmp_path):
    root = _tree(tmp_path / "ctx", {**SERVER, ".dockerignore": "app\n", "Dockerfile.dockerignore": "data\n"})

    context = BuildContext.of(root, root / "Dockerfile")

    assert {entry.name for entry in context.entries} >= {"app/main.py", "Dockerfile.dockerignore"}
    assert not any(entry.name.startswith("data") for entry in context.entries)


def test_what_the_os_and_python_leave_behind_is_dropped(tmp_path):
    root = _tree(tmp_path / "ctx", {**SERVER, ".DS_Store": "", "app/__pycache__/main.cpython-312.pyc": ""})

    names = {entry.name for entry in BuildContext.of(root, root / "Dockerfile").entries}

    assert ".DS_Store" not in names and not any("__pycache__" in name for name in names)


def test_every_link_is_stored_as_a_link_and_never_followed(tmp_path):
    root = _tree(tmp_path / "ctx", SERVER)
    (root / "main-link.py").symlink_to("app/main.py")
    (root / "lib").symlink_to("app", target_is_directory=True)
    (root / "app" / "loop").symlink_to(root, target_is_directory=True)
    (root / ".venv-python").symlink_to("/usr/bin/python3")
    (root / "gone").symlink_to("nowhere")

    members = {m.name: m for m in _members(_written(BuildContext.of(root, root / "Dockerfile"), tmp_path))}

    assert {name: (m.linkname, m.mode) for name, m in members.items() if m.issym()} == {
        ".venv-python": ("/usr/bin/python3", 0o777), "app/loop": (str(root), 0o777), "gone": ("nowhere", 0o777),
        "lib": ("app", 0o777), "main-link.py": ("app/main.py", 0o777)}
    assert "lib/main.py" not in members


def test_a_link_the_dockerignore_excludes_is_left_out(tmp_path):
    root = _tree(tmp_path / "ctx", {**SERVER, ".dockerignore": ".venv\n"})
    (root / ".venv").mkdir()
    (root / ".venv" / "python").symlink_to("/usr/bin/python3")

    assert not any(entry.name.startswith(".venv") for entry in BuildContext.of(root, root / "Dockerfile").entries)


def test_a_special_file_is_named(tmp_path):
    root = _tree(tmp_path / "ctx", SERVER)
    os.mkfifo(root / "pipe")

    with pytest.raises(ValueError, match="pipe: neither a regular file, a folder nor a link"):
        BuildContext.of(root, root / "Dockerfile")


@pytest.mark.parametrize("change", [
    lambda root: (root / "app" / "main.py").write_text("print(2)\n"),
    lambda root: (root / "app" / "main.py").chmod(0o755),
    lambda root: (root / "app" / "new.py").write_text(""),
    lambda root: (root / "app" / "new.py").symlink_to("main.py"),
])
def test_the_digest_follows_content_modes_and_names(tmp_path, change):
    root = _tree(tmp_path / "ctx", SERVER)
    before = BuildContext.of(root, root / "Dockerfile").source_digest("linux/amd64")

    change(root)

    assert BuildContext.of(root, root / "Dockerfile").source_digest("linux/amd64") != before


def test_the_digest_follows_the_dockerfile_and_platform(tmp_path):
    root = _tree(tmp_path / "ctx", {**SERVER, "other.Dockerfile": "FROM python:3.12\n"})
    context = BuildContext.of(root, root / "Dockerfile")

    digests = {context.source_digest("linux/amd64"), context.source_digest("linux/arm64"),
               BuildContext.of(root, root / "other.Dockerfile").source_digest("linux/amd64")}

    assert len(digests) == 3


def test_a_dockerfile_outside_the_context_is_recorded_as_none(tmp_path):
    root = _tree(tmp_path / "ctx", SERVER)
    outside = tmp_path / "Dockerfile.ci"
    outside.write_text("FROM python:3.12\n")

    assert BuildContext.of(root, outside).dockerfile is None
    assert BuildContext.of(root, root / "Dockerfile").dockerfile == "Dockerfile"


def test_a_file_that_changes_before_the_tar_is_written_is_refused(tmp_path):
    root = _tree(tmp_path / "ctx", SERVER)
    context = BuildContext.of(root, root / "Dockerfile")
    (root / "app" / "main.py").write_text("print(3)\n")

    with pytest.raises(RuntimeError, match="app/main.py changed while its build context was being written"):
        context.write(tmp_path / "ctx.tar.gz")


def test_a_context_that_isnt_a_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not a folder"):
        BuildContext.of(tmp_path / "missing")


def test_the_written_tar_extracts_to_the_context(tmp_path):
    root = _tree(tmp_path / "ctx", SERVER, executable=("app/run.sh",))
    out = tmp_path / "out"

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(_written(BuildContext.of(root), tmp_path)))) as tar:
        tar.extractall(out, filter="data")

    assert (out / "app" / "main.py").read_text() == "print(1)\n" and os.access(out / "app" / "run.sh", os.X_OK)
    assert (out / "data" / "empty" / ".keep").exists()
