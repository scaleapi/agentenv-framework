"""``lock``: an image folder's pip and global npm installs pinned to the versions its stored image holds."""

from pathlib import Path

from agent_env.utils.docker.installed import Installed
from agent_env.utils.docker.lock import lock

INSTALLED = Installed(
    python={"pip": ("24.0",), "setuptools": ("69.0.0",), "requests": ("2.31.0",), "numpy": ("2.1.0",),
            "urllib3": ("2.2.1",), "six": ("1.16.0", "1.17.0"), "myapp": ("0.1.dev3",),
            "distro-info": ("1.1build1",)},
    npm={"typescript": "5.4.5", "@anthropic-ai/claude-code": "1.0.3", "npm": "10.5.0"},
    direct=frozenset({"myapp"}))
HEADER = "# Pinned by agent-env to the versions of the image the store held.\n"
ARGS = 'COPY ["{path}", "/tmp/agent-env-constraints.txt"]\nARG PIP_CONSTRAINT=/tmp/agent-env-constraints.txt\n' \
       "ARG UV_CONSTRAINT=/tmp/agent-env-constraints.txt\n"


def _folder(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


def test_each_stage_running_pip_reads_constraints_pinned_to_the_stored_versions(tmp_path):
    dockerfile = ("FROM python:3.12 AS build\nRUN pip wheel -r requirements.txt -w /wheels\n"
                  "FROM node:20 AS web\nRUN echo hi\n"
                  "FROM python:3.12\nCOPY --from=build /wheels /wheels\nRUN pip install /wheels/*\n")
    folder = _folder(tmp_path, {"Dockerfile": dockerfile, "requirements.txt": "requests\n"})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (folder / "constraints.txt").read_text() == f"{HEADER}numpy==2.1.0\nrequests==2.31.0\nurllib3==2.2.1\n"
    args = ARGS.format(path="constraints.txt")
    assert (folder / "Dockerfile").read_text() == (
        f"FROM python:3.12 AS build\n{args}RUN pip wheel -r requirements.txt -w /wheels\n"
        "FROM node:20 AS web\nRUN echo hi\n"
        f"FROM python:3.12\n{args}COPY --from=build /wheels /wheels\nRUN pip install /wheels/*\n")
    assert (locked.python, locked.constraints, locked.lock_file) == (3, "constraints.txt", None)
    assert set(locked.not_covered) == {
        "pip, setuptools, which isolated builds install their own versions of",
        "Python distro-info, whose version 1.1build1 pip can't read",
        "Python myapp, installed from a path or URL rather than an index",
        "Python six, which the image holds at 1.16.0, 1.17.0"}


def test_uv_pip_and_heredoc_runs_count_and_a_run_without_pip_does_not(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": "FROM a\nRUN uv pip install --system requests\n"
                                              "FROM b\nRUN <<EOF\nset -e\npython -m pip install numpy\nEOF\n"
                                              "FROM c\nRUN apt-get install -y curl\n"})
    lock(folder, "Dockerfile", INSTALLED)
    args = ARGS.format(path="constraints.txt")
    assert (folder / "Dockerfile").read_text() == (f"FROM a\n{args}RUN uv pip install --system requests\n"
                                                   f"FROM b\n{args}RUN <<EOF\nset -e\npython -m pip install numpy\nEOF\n"
                                                   "FROM c\nRUN apt-get install -y curl\n")


def test_a_version_the_build_asks_for_that_the_image_ended_past_is_left_unpinned(tmp_path):
    folder = _folder(tmp_path, {
        "Dockerfile": 'FROM python\nRUN pip install "numpy<2" && pip install -r reqs/requirements.txt\n',
        "reqs/requirements.txt": "# pinned\nurllib3==1.26.18 \\\n    --hash=sha256:abc\nrequests==2.31.0  # ok\n-e .\n"})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (folder / "constraints.txt").read_text() == f"{HEADER}requests==2.31.0\n"
    assert "Python numpy, which the Dockerfile asks for as numpy<2 though the image ends with 2.1.0" in locked.not_covered
    assert ("Python urllib3, which reqs/requirements.txt asks for as urllib3==1.26.18 though the image ends with 2.2.1"
            in locked.not_covered)


def test_a_lock_file_the_build_installs_from_is_reported_and_pip_installs_beside_it_are_still_pinned(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": "FROM python\nRUN pip install uv && uv sync --frozen\n",
                                "app/uv.lock": "version = 1\n"})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (locked.python, locked.constraints, locked.lock_file) == (3, "constraints.txt", "app/uv.lock")
    assert (folder / "Dockerfile").read_text() == f"FROM python\n{ARGS.format(path='constraints.txt')}" \
                                                  "RUN pip install uv && uv sync --frozen\n"


def test_a_lock_file_no_build_command_installs_from_neither_stops_pinning_nor_is_reported(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": "FROM python\nCOPY . /app\nRUN pip install requests\n",
                                "poetry.lock": "# unused\n"})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (locked.python, locked.constraints, locked.lock_file) == (3, "constraints.txt", None)


def test_a_stage_installing_only_from_its_lock_file_reads_no_constraints(tmp_path):
    dockerfile = "FROM python\nCOPY pyproject.toml poetry.lock ./\nRUN poetry install --no-root\n"
    folder = _folder(tmp_path, {"Dockerfile": dockerfile, "poetry.lock": "# used\n", "pyproject.toml": "\n"})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (locked.python, locked.constraints, locked.lock_file) == (0, None, "poetry.lock")
    assert (folder / "Dockerfile").read_text() == dockerfile


def test_constraints_go_beside_a_dockerfile_in_a_folder_under_a_name_of_their_own_when_taken(tmp_path):
    folder = _folder(tmp_path, {"svc/Dockerfile": "FROM python\nRUN pip install -c svc/constraints.txt requests\n",
                                "svc/constraints.txt": "requests==2.31.0\n"})
    locked = lock(folder, "svc/Dockerfile", INSTALLED)
    assert locked.constraints == "svc/agent-env-constraints.txt"
    assert (folder / "svc/constraints.txt").read_text() == "requests==2.31.0\n"
    assert (folder / "svc/Dockerfile").read_text().startswith(f"FROM python\n{ARGS.format(path=locked.constraints)}")


def test_an_ignore_file_leaving_the_constraints_out_writes_none(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": "FROM python\nRUN pip install requests\n",
                                "Dockerfile.dockerignore": "*\n!app\n", ".dockerignore": ""})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (locked.python, locked.constraints) == (0, None)
    assert "Python packages, since the folder's ignore file leaves out constraints.txt" in locked.not_covered
    assert (folder / "Dockerfile").read_text() == "FROM python\nRUN pip install requests\n"


def test_global_npm_installs_name_the_stored_versions(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": (
        "FROM node:20\n"
        "RUN npm install -g --prefix /usr/local typescript@^5 \\\n    @anthropic-ai/claude-code@latest leftpad"
        " && npm install express\n"
        "RUN npm i --global npm@latest; npm ci\n")})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (folder / "Dockerfile").read_text() == (
        "FROM node:20\n"
        "RUN npm install -g --prefix /usr/local typescript@5.4.5 \\\n    @anthropic-ai/claude-code@1.0.3 leftpad"
        " && npm install express\n"
        "RUN npm i --global npm@10.5.0; npm ci\n")
    assert locked.npm == ("typescript@5.4.5", "@anthropic-ai/claude-code@1.0.3", "npm@10.5.0")
    assert locked.not_covered == ("npm packages' own dependencies, which a global install resolves afresh",)
    assert locked.constraints is None


def test_nothing_to_pin_leaves_the_dockerfile_as_it_was(tmp_path):
    dockerfile = "# syntax=docker/dockerfile:1\nFROM alpine\nRUN apk add curl\n"
    folder = _folder(tmp_path, {"Dockerfile": dockerfile})
    assert lock(folder, "Dockerfile", INSTALLED).python == 0
    assert (folder / "Dockerfile").read_text() == dockerfile


def test_quoted_global_npm_packages_keep_their_quotes_and_one_written_in_a_form_not_rewritten_is_reported(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": (
        "FROM node:20\n"
        "RUN npm install -g \"typescript@^5\" '@anthropic-ai/claude-code@latest'\\\n    leftpad\n"
        "RUN npm i -g npm@\"10\"\n")})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert (folder / "Dockerfile").read_text() == (
        "FROM node:20\n"
        "RUN npm install -g \"typescript@5.4.5\" '@anthropic-ai/claude-code@1.0.3'\\\n    leftpad\n"
        "RUN npm i -g npm@\"10\"\n")
    assert locked.npm == ("typescript@5.4.5", "@anthropic-ai/claude-code@1.0.3")
    assert "npm npm, which an npm install -g names in a form agent-env doesn't rewrite" in locked.not_covered


def test_constraints_take_a_name_nothing_in_the_folder_has_and_leave_what_is_there_alone(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": "FROM python\nRUN pip install -c constraints.txt requests\n",
                                "constraints.txt": "requests==2.31.0\n", "agent-env-constraints.txt": "theirs\n"})
    locked = lock(folder, "Dockerfile", INSTALLED)
    assert locked.constraints == "agent-env-constraints-2.txt"
    assert ((folder / "constraints.txt").read_text(), (folder / "agent-env-constraints.txt").read_text()) == (
        "requests==2.31.0\n", "theirs\n")


def test_a_second_image_in_a_folder_isnt_held_to_the_constraints_the_first_ones_pinning_wrote(tmp_path):
    folder = _folder(tmp_path, {"Dockerfile": "FROM python\nRUN pip install requests\n",
                                "Dockerfile.worker": "FROM python\nRUN pip install requests\n"})
    lock(folder, "Dockerfile", INSTALLED)
    newer = Installed(python={"requests": ("2.32.3",)}, npm={})
    second = lock(folder, "Dockerfile.worker", newer)
    assert (second.python, second.constraints) == (1, "agent-env-constraints.txt")
    assert (folder / "agent-env-constraints.txt").read_text().splitlines()[-1] == "requests==2.32.3"
