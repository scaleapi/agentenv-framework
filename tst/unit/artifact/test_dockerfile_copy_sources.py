from pathlib import Path

from agent_env.artifact.artifacts.docker_image import _get_dockerfile_copy_sources


def _make_tree(root: Path) -> Path:
    ctx = root / "ctx"
    (ctx / "app").mkdir(parents=True)
    (ctx / "app" / "main.py").write_text("print('hi')\n")
    (ctx / "reset").mkdir()
    (ctx / "reset" / "run.py").write_text("x = 1\n")
    (ctx / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "COPY app/ ./app/\n"
        "COPY reset/ ./reset/\n"
    )
    return ctx


def test_cwd_relative_dockerfile_suffix_of_context_does_not_double(tmp_path, monkeypatch):
    _make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = _get_dockerfile_copy_sources(Path("ctx"), "ctx/Dockerfile")
    assert set(out) >= {"app", "reset"}


def test_absolute_dockerfile_and_context(tmp_path):
    ctx = _make_tree(tmp_path)
    out = _get_dockerfile_copy_sources(ctx, str(ctx / "Dockerfile"))
    assert set(out) >= {"app", "reset"}


def test_context_relative_dockerfile_still_works(tmp_path, monkeypatch):
    ctx = _make_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = _get_dockerfile_copy_sources(ctx, "Dockerfile")
    assert set(out) >= {"app", "reset"}


def test_no_dockerfile_returns_dot():
    assert _get_dockerfile_copy_sources(Path("/tmp"), None) == ["."]


def test_copying_the_whole_context_keeps_the_whole_context(tmp_path):
    ctx = _make_tree(tmp_path)
    (ctx / "Dockerfile").write_text("FROM python:3.11-slim\nCOPY app/ ./app/\nCOPY . /src\n")
    assert _get_dockerfile_copy_sources(ctx, str(ctx / "Dockerfile")) == ["."]
