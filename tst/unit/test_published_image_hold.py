"""The integration tier's hold on the registry: a test sees each image's published digest, or that it has none, as its
first lookup of it found it, though the registry changes later in the test."""

from agent_env.env import published_images
from tst.util.published_image import hold_registry

REF = "ghcr.io/example/agentenv-claude-code:src-" + "1" * 64
OTHER = "ghcr.io/example/agentenv-echo-mcp:src-" + "2" * 64


def test_an_image_published_after_the_tests_first_lookup_stays_unpublished_to_it(monkeypatch):
    registry: dict[str, str] = {}
    asked: list[str] = []

    def lookup(ref):
        asked.append(ref)
        return registry.get(ref)

    monkeypatch.setattr(published_images, "_registry_digest", lookup)
    hold_registry(monkeypatch)
    assert published_images._registry_digest(REF) is None
    registry[REF] = "sha256:" + "a" * 64
    registry[OTHER] = "sha256:" + "b" * 64
    assert published_images._registry_digest(REF) is None
    assert published_images._registry_digest(OTHER) == "sha256:" + "b" * 64
    registry[OTHER] = "sha256:" + "c" * 64
    assert published_images._registry_digest(OTHER) == "sha256:" + "b" * 64
    assert asked == [REF, OTHER]
