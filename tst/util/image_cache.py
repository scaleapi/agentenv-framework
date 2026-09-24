"""Content-hash reuse of test-fixture Docker image artifacts.

Test-only. The artifact model/store know nothing about this: the reuse policy
and the hash computation live here. The build inputs (Dockerfile bytes + every
file in the context + platform) are hashed and the hash is appended to the
caller's ``artifact_id`` - so the artifact is *content-addressed*. Identical
inputs deterministically map to the same artifact id (reuse); any input change
produces a new id (rebuild). Nothing in application code parses this.

Semantics:

- **Never stale for changed inputs.** The reuse key IS the build inputs. Any
  input change changes the id and forces a rebuild. Over-hashing is the safe
  direction: including extra context files can only cause a spurious rebuild,
  never a stale reuse. (The one exception is base-image drift under an
  unchanged FROM tag, which predates this cache and is mitigated by
  digest-pinned bases.)

- **Concurrency-safe.** Because the id encodes the content, two runs that
  first-build the same context are building byte-identical images. They race to
  ``put()`` the same content-addressed id; the artifact store's immutable-S3
  guard lets exactly one win. The loser catches ``ObjectAlreadyExistsError``
  and *adopts* the winner's artifact (see ``_adopt_concurrent_build``) - it
  never rebuilds. This matters because CI runs share these fixture ids across
  all PRs (``concurrent_build_limit`` > 1), so same-context first-builds do
  overlap in practice, e.g. after a base-image bump fans out to open PRs.

  Not handled: a *torn* write - if the winner dies between its S3 upload and
  its Mongo document write, that one content id is wedged (``get()`` misses
  forever, so every later run rebuilds and re-collides on the immutable S3
  object). Recovery is deleting that one prefix:
  ``s3://<dev bucket>/artifacts/docker_image/<artifact_id>-<hash>/1/``. Rare in
  practice; fixing it in general needs an application-side idempotent put,
  deliberately out of scope for this test helper.

- **Reuse skips ``docker build`` entirely**, so the local docker tag is NOT
  created on the reuse path. Callers must consume the returned artifact (ECR/
  S3-backed), not the local tag.
"""

import hashlib
import logging
import subprocess
import time
from pathlib import Path

from agent_env.artifact import DockerImageArtifact
from agent_env.store.base import NotFoundError, ObjectAlreadyExistsError

logger = logging.getLogger(__name__)

# Generated/volatile files excluded from hashing (nondeterministic content or
# irrelevant to the built image).
_VOLATILE_DIRS = {"__pycache__"}
_VOLATILE_SUFFIXES = (".pyc",)
_VOLATILE_NAMES = {".DS_Store"}


def _is_volatile(rel: Path) -> bool:
    if any(part in _VOLATILE_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return True
    return rel.name in _VOLATILE_NAMES or rel.suffix in _VOLATILE_SUFFIXES


def hash_build_context(dockerfile: Path, context: Path, platform: str) -> str:
    """Hash the Dockerfile + every file in the build context (path + bytes).

    Volatile generated files are excluded: .pyc headers embed the source
    mtime, which is the git-checkout time in CI, so any context containing an
    imported Python package would hash differently on every run and never
    cache-hit (this bit the two gateway images).

    When the Dockerfile lives inside the context (the usual case) its bytes
    are hashed twice - once here, once via the context walk. Deterministic,
    so harmless; kept because ANY change to this function invalidates every
    content-addressed id and costs one full rebuild cycle. Batch such changes.

    Deliberately ignores .dockerignore: see module docstring (over-hashing is
    safe). Files are visited in sorted order so the hash is deterministic.
    """
    h = hashlib.sha256()
    h.update(f"platform={platform}\n".encode())
    h.update(dockerfile.read_bytes())
    for f in sorted(p for p in context.rglob("*") if p.is_file()):
        rel = f.relative_to(context)
        if _is_volatile(rel):
            continue
        h.update(str(rel).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _try_get(artifact_id: str) -> DockerImageArtifact | None:
    """Latest version of ``artifact_id``, or None on a clean miss.

    Store errors (Mongo/S3/auth) are logged and treated as a miss so a flaky
    lookup can't abort the run - the subsequent put() will likely fail with the
    same root cause, and this is where the traceback is captured.
    """
    try:
        return DockerImageArtifact.get(artifact_id)
    except NotFoundError:
        return None
    except Exception:
        logger.warning(f"Artifact store lookup failed for {artifact_id}; rebuilding", exc_info=True)
        return None


def _adopt_concurrent_build(artifact_id: str, *, attempts: int = 6, delay: float = 0.5) -> DockerImageArtifact:
    """Fetch an artifact a concurrent run just published under the same id.

    Only reached on a same-content collision. ``put()`` writes the S3 object
    before the Mongo document, and the collision surfaces at the S3 step, so the
    winner's document may not be visible yet - tolerate that brief lag. This
    never rebuilds; it waits for an already-committed, byte-identical artifact.
    """
    for i in range(attempts):
        try:
            return DockerImageArtifact.get(artifact_id)
        except NotFoundError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")  # loop either returns or raises


def build_or_reuse(
    *,
    artifact_id: str,
    description: str,
    dockerfile: Path,
    context: Path,
    tag: str,
    platform: str = "linux/amd64",
) -> DockerImageArtifact:
    """Build + put the image, unless an artifact for a byte-identical context
    already exists - then return it without building/pushing.

    ``artifact_id`` is the caller-facing base name; the content hash is appended
    to it so the stored artifact is content-addressed (see module docstring).
    """
    dockerfile = Path(dockerfile)
    context = Path(context)
    ctx_hash = hash_build_context(dockerfile, context, platform)
    content_id = f"{artifact_id}-{ctx_hash}"

    # Hot path: this exact context already published -> reuse, no docker build.
    existing = _try_get(content_id)
    if existing is not None:
        logger.info(f"Reusing {content_id} v{existing.version}")
        return existing

    logger.info(f"Building {tag} for {content_id}...")
    result = subprocess.run(
        ["docker", "build", "--platform", platform, "-f", str(dockerfile), "-t", tag, str(context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{tag} build failed: {result.stderr}")

    # Publish. A concurrent run may have first-built the SAME content and raced
    # us to S3; since content_id encodes the bytes, whatever landed is identical
    # -> adopt it rather than fail on the immutable-S3 guard.
    try:
        return DockerImageArtifact.put(id=content_id, description=description, image_name=tag)
    except ObjectAlreadyExistsError:
        logger.info(f"{content_id}: concurrent build won the race; adopting it")
        return _adopt_concurrent_build(content_id)
