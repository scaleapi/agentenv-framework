"""Backend-neutral ImageStore conformance assertions.

Each case takes ``(store, repository)`` — a unique repository namespace. Both
``EcrImageStore`` and ``LocalRegistryImageStore`` must pass every case in
``CASES``; that shared pass is the "it generalizes" proof. The push/pull case
drives a real ``docker`` daemon, so callers run at the integration tier.
"""

import subprocess
import tempfile
import uuid
from pathlib import Path


def image_ref_contains_repository_and_tag(store, repository):
    ref = store.image_ref(repository, "v1")
    assert ref.endswith(":v1")
    assert repository in ref


def ensure_repository_is_idempotent(store, repository):
    store.ensure_repository(repository)
    store.ensure_repository(repository)


def auth_is_none_for_a_foreign_ref(store, repository):
    assert store.auth("foreign-registry.example/library/alpine:latest") is None


def push_pull_roundtrip(store, repository):
    ref = store.image_ref(repository, "roundtrip")
    local_tag = f"imagestore-conf-{uuid.uuid4().hex[:12]}"
    _build_scratch_image(local_tag)
    try:
        store.ensure_repository(repository)
        _docker("tag", local_tag, ref)
        _login(store, ref)
        _docker("push", ref)
        _docker("rmi", "-f", ref, local_tag)
        _docker("pull", ref)
        _docker("image", "inspect", ref)
    finally:
        _docker_quiet("rmi", "-f", ref, local_tag)


CASES = [
    image_ref_contains_repository_and_tag,
    ensure_repository_is_idempotent,
    auth_is_none_for_a_foreign_ref,
    push_pull_roundtrip,
]


def _docker(*args, input=None):
    result = subprocess.run(["docker", *args], input=input, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _docker_quiet(*args):
    subprocess.run(["docker", *args], capture_output=True, text=True)


def _login(store, ref):
    auth = store.auth(ref)
    if auth is not None:
        _docker("login", auth.registry, "--username", auth.username, "--password-stdin", input=auth.password)


def _build_scratch_image(tag):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "payload.txt").write_text(uuid.uuid4().hex)
        (root / "Dockerfile").write_text("FROM scratch\nCOPY payload.txt /payload.txt\n")
        _docker("build", "-t", tag, str(root))
