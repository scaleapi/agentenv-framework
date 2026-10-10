"""Modal builds an image that is only a build context from its context, unpacked on this machine, with the context's own
ignore file: once per process for the same sources, before its container, and again on the next deploy after a failure.
Modal's build and sandbox calls are faked; the context is a real one on the local stores."""

from __future__ import annotations

import asyncio
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import modal_image_build, modal_sandbox
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider

SERVER = {"Dockerfile": "FROM python:3.12\nCOPY . /app\n", "src/app.py": "print(1)\n", ".dockerignore": "*.log\n",
          "debug.log": "noise"}


class _Builds:
    """Stands in for ``modal.Image.from_dockerfile``: each image it returns records its context when built."""

    def __init__(self, fail: Exception | None = None):
        self.fail, self.calls, self.contexts, self.dockerfiles = fail, [], [], []

    def __call__(self, path, *, context_dir, ignore):
        self.calls.append((Path(path), Path(context_dir), ignore))
        image = MagicMock()

        async def build(app):
            self.contexts.append(sorted(p.relative_to(context_dir).as_posix() for p in Path(context_dir).rglob("*")))
            self.dockerfiles.append(Path(path).read_text())
            await asyncio.sleep(0.01)  # long enough for a second deploy to wait on the first
            if self.fail:
                raise self.fail
            image.object_id = f"im-{len(self.contexts)}"

        image.build.aio = AsyncMock(side_effect=build)
        return image


@pytest.fixture(autouse=True)
def _no_built_images(monkeypatch):
    monkeypatch.setattr(modal_image_build, "_built_image_ids", {})


@pytest.fixture
def context(tmp_path):
    root = tmp_path / "ctx"
    for name, text in SERVER.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


@pytest.fixture
def builds(monkeypatch):
    builds = _Builds()
    monkeypatch.setattr(modal.Image, "from_dockerfile", builds)
    monkeypatch.setattr(modal.Image, "from_id", lambda image_id, client=None: ("from_id", image_id))
    return builds


def _provider() -> ModalSandboxProvider:
    provider = ModalSandboxProvider(app_name="agent-env-test")
    provider._get_client = AsyncMock(return_value=MagicMock())
    provider._get_app = AsyncMock(return_value="app")
    return provider


def _context_image(id: str, context: Path) -> DockerImageArtifact:
    return DockerImageArtifact.put_context(id, description="d", context_path=str(context), dockerfile_path="Dockerfile")


@pytest.mark.asyncio
async def test_an_image_is_built_from_its_unpacked_context_with_the_contexts_ignore_file(local_stores, context, builds):
    image = _context_image("solver-image", context)

    await _provider().prepare_image(image)

    [(dockerfile, context_dir, ignore)] = builds.calls
    assert dockerfile == context_dir.parent / "Dockerfile"  # beside the context, as Modal reads it
    assert builds.dockerfiles == [SERVER["Dockerfile"]]
    assert builds.contexts == [[".dockerignore", "Dockerfile", "src", "src/app.py"]]
    assert ignore == ["*.log"]
    assert not context_dir.exists()  # unpacked only for the build


@pytest.mark.asyncio
async def test_a_context_with_no_ignore_file_is_built_ignoring_nothing(local_stores, context, builds):
    (context / ".dockerignore").unlink()

    await _provider().prepare_image(_context_image("solver-image", context))

    [(_, _, ignore)] = builds.calls
    assert ignore == []


@pytest.mark.asyncio
async def test_deploys_of_the_same_sources_at_once_share_one_build_and_later_ones_reuse_it(local_stores, context, builds):
    first, second = _context_image("solver-image", context), _context_image("judge-image", context)
    assert first.source_digest == second.source_digest

    await asyncio.gather(_provider().prepare_image(first), _provider().prepare_image(second))
    await _provider().prepare_image(first)

    assert len(builds.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_build_names_the_image_and_the_next_deploy_builds_it_again(local_stores, context, builds):
    image = _context_image("solver-image", context)
    builds.fail = RuntimeError("step 2/2 COPY failed")

    with pytest.raises(RuntimeError, match=rf"Building {image.image_name} on Modal from its build context failed: "
                                           r"builtins.RuntimeError: step 2/2 COPY failed"):
        await _provider().prepare_image(image)
    builds.fail = None
    await _provider().prepare_image(image)

    assert len(builds.calls) == 2


@pytest.mark.asyncio
async def test_a_failed_modal_build_keeps_modals_pointer_to_its_build_log(local_stores, context, builds):
    builds.fail = modal.exception.ImageBuildError(
        "Image build for im-abc failed.\nView the build logs:\n  modal image logs im-abc", "im-abc")

    with pytest.raises(RuntimeError, match="(?s)Image build for im-abc failed.*modal image logs im-abc"):
        await _provider().prepare_image(_context_image("solver-image", context))


@pytest.mark.parametrize("line, built", [
    ('COPY ["a.txt", "/x/a.txt"]', "COPY a.txt /x/a.txt"),
    ('copy --chown=1:1 ["a.txt", "sub", "/y/"]', "COPY --chown=1:1 a.txt sub /y/"),
    ("ADD a.txt /added/", "COPY a.txt /added/"),
    ("ADD sub \\\n    /added/sub/", "COPY sub /added/sub/"),
    ("ADD *.txt /globbed/", "COPY *.txt /globbed/"),
    ('ADD ["a.txt", "/j/"]', "COPY a.txt /j/"),
    ("ADD --chmod=755 a.txt /c", "COPY --chmod=755 a.txt /c"),
], ids=["json-copy", "json-copy-flags", "add-file", "add-folder-continued", "add-glob", "json-add", "add-chmod"])
def test_what_modal_cant_take_is_written_as_its_docker_equivalent(tmp_path, line, built):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()

    assert modal_image_build.modal_dockerfile(f"FROM python\n{line}\nRUN true\n", tmp_path) == \
        f"FROM python\n{built}\nRUN true\n"


@pytest.mark.parametrize("line", [
    'COPY ["with space.txt", "/z/"]', "ADD archive.tar.gz /unpacked/", "ADD https://example.com/f /f",
    "ADD --checksum=sha256:abc a.txt /c", "ADD missing.txt /m", "COPY --from=build /out /out", "COPY a.txt /x/",
    'ADD "a.txt" /q/', "ADD <<EOF /h\nhi\nEOF",
], ids=["json-with-space", "add-archive", "add-url", "add-checksum", "add-missing", "copy-from", "shell-copy",
        "add-quoted", "add-heredoc"])
def test_everything_else_is_left_as_written(tmp_path, line):
    (tmp_path / "a.txt").write_text("a")
    with tarfile.open(tmp_path / "archive.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "a.txt", "a.txt")
    dockerfile = f"FROM python\n{line}\n"

    assert modal_image_build.modal_dockerfile(dockerfile, tmp_path) == dockerfile


@pytest.mark.asyncio
async def test_an_image_that_is_not_only_a_build_context_is_not_built(local_stores, builds):
    await _provider().prepare_image(DockerImageArtifact(id="solver-image", description="d",
                                                        image_name="registry.example/solver:v2"))

    assert builds.calls == []


def _fake_create():
    """A patched ``modal.Sandbox._experimental_create`` that stops right after recording its call."""
    create = MagicMock()
    create.aio = AsyncMock(side_effect=RuntimeError("stop"))
    return patch.object(modal_sandbox.modal.Sandbox, "_experimental_create", create)


@pytest.mark.asyncio
async def test_a_container_runs_the_built_image_and_builds_it_on_a_miss(local_stores, context, builds):
    image = _context_image("solver-image", context)
    provider = _provider()
    await provider.prepare_image(image)
    modal_image_build._built_image_ids.clear()

    with _fake_create() as create, pytest.raises(RuntimeError, match="Modal sandbox create failed"):
        await provider.create_container(image_name=image.image_name, port=8000, env={})

    assert create.aio.call_args.kwargs["image"] == ("from_id", "im-2")
    assert len(builds.calls) == 2


@pytest.mark.asyncio
async def test_a_built_image_modal_no_longer_holds_is_built_again_once_for_every_deploy_waiting_on_it(local_stores,
                                                                                                      context, builds):
    image = _context_image("solver-image", context)
    provider = _provider()
    await provider.prepare_image(image)
    created = []

    async def create(*args, image, **kwargs):
        created.append(image)
        if image == ("from_id", "im-1"):
            raise modal.exception.NotFoundError("No Image with ID 'im-1' found")
        raise RuntimeError("stop")

    with _fake_create() as fake:
        fake.aio.side_effect = create
        for result in await asyncio.gather(*(provider.create_container(image_name=image.image_name, port=8000, env={})
                                             for _ in range(2)), return_exceptions=True):
            assert isinstance(result, RuntimeError) and "stop" in str(result)

    assert len(builds.calls) == 2
    assert created.count(("from_id", "im-2")) == 2


@pytest.mark.asyncio
async def test_a_missing_image_it_was_not_built_from_a_context_is_not_rebuilt(local_stores, builds, monkeypatch):
    monkeypatch.setattr(modal.Image, "from_registry", lambda name, **kwargs: ("pulled", name))

    with _fake_create() as fake, pytest.raises(RuntimeError, match="No Image"):
        fake.aio.side_effect = modal.exception.NotFoundError("No Image with ID 'im-9' found")
        await _provider().create_container(image_name="registry.example/solver:v2", port=8000, env={})

    assert builds.calls == [] and fake.aio.call_count == 1


@pytest.mark.asyncio
async def test_a_container_of_an_image_it_wasnt_told_of_is_pulled(local_stores, builds, monkeypatch):
    pulled = MagicMock()
    monkeypatch.setattr(modal.Image, "from_registry", lambda name, **kwargs: (pulled, name))

    with _fake_create() as create, pytest.raises(RuntimeError, match="Modal sandbox create failed"):
        await _provider().create_container(image_name="registry.example/solver:v2", port=8000, env={})

    assert create.aio.call_args.kwargs["image"] == (pulled, "registry.example/solver:v2")
    assert builds.calls == []
