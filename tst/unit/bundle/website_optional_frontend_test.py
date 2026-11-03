"""A website's frontend image is optional in a bundle: a folder with no Dockerfile for it builds none."""

from pathlib import Path
from types import SimpleNamespace

from agent_env.bundle.resolve import _Resolver
from agent_env.entity.refs import EntityRef


def test_an_optional_image_key_with_no_dockerfile_builds_nothing_and_a_required_one_is_refused(tmp_path):
    (tmp_path / "Dockerfile.backend").write_text("FROM scratch\n")
    entry = SimpleNamespace(path=Path(tmp_path), id="site", kind=SimpleNamespace(store="env"))
    resolver = _Resolver.__new__(_Resolver)
    resolver.problems, resolver.images = [], []
    resolver._problem = lambda entry, problem: resolver.problems.append(problem)
    resolver._implied = lambda image, owner, key, ref, references: resolver.images.append(image.dockerfile)

    for ref in (EntityRef.artifact("backend_image", artifact_type="docker_image"),
                EntityRef.artifact("frontend_image", artifact_type="docker_image", optional=True)):
        assert resolver._image(entry, {}, ref, [])
    assert (resolver.images, resolver.problems) == (["Dockerfile.backend"], [])
    assert resolver._image(entry, {}, EntityRef.artifact("frontend_image", artifact_type="docker_image"), [])
    assert resolver.problems == ["frontend_image: there is no 'Dockerfile.frontend' in this folder to build"]
