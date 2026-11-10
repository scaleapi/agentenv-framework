"""``.github/scripts/prune_registry.py``, which keeps only the computer images a slow-tier run used before the local
registry's storage is cached: their tags, the manifests those reach, and the layer links garbage collection leaves."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tst.util.scripts import SCRIPTS, load_script

SCRIPT = SCRIPTS / "prune_registry.py"
REPOSITORY = "agentenv-computers"
_USED, _STALE = "a" * 64, "b" * 64


@pytest.fixture
def script():
    return load_script(SCRIPT.name)


def test_it_imports_only_the_standard_library():
    """It runs under the runner's python3 with sudo, outside the project's environment."""
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_a_tag_is_used_when_the_log_has_a_request_for_its_manifest_by_it(script):
    log = "\n".join([
        f'time="t" level=info msg="response completed" http.request.method=HEAD '
        f'http.request.uri="/v2/{REPOSITORY}/manifests/{_USED}" http.response.status=200',
        f'172.17.0.1 - - [10/Nov/2026:05:00:00 +0000] "PUT /v2/{REPOSITORY}/manifests/{"c" * 64} HTTP/1.1" 201 0',
        f'172.17.0.1 - - [10/Nov/2026:05:00:01 +0000] "GET /v2/{REPOSITORY}/manifests/sha256:{"d" * 64} HTTP/1.1" 200',
        f'172.17.0.1 - - [10/Nov/2026:05:00:02 +0000] "HEAD /v2/other/manifests/{_STALE} HTTP/1.1" 200 0',
    ])

    assert script.used_tags(log, REPOSITORY) == {_USED, "c" * 64}


def test_pruning_keeps_the_used_tags_and_every_manifest_their_index_lists(script, tmp_path):
    platforms = [_blob(tmp_path, {"layers": [], "n": n}) for n in range(2)]
    index = _blob(tmp_path, {"manifests": [{"digest": digest} for digest in platforms]})
    stale = _blob(tmp_path, {"layers": [], "n": "stale"})
    _tag(tmp_path, _USED, index)
    _tag(tmp_path, _STALE, stale)
    for digest in (index, *platforms, stale):
        _revision(tmp_path, digest)

    kept, dropped = script.prune(tmp_path, REPOSITORY, {_USED})

    manifests = tmp_path / "docker/registry/v2/repositories" / REPOSITORY / "_manifests"
    assert (kept, dropped) == (1, 1)
    assert sorted(p.name for p in (manifests / "tags").iterdir()) == [_USED]
    assert {p.name for p in (manifests / "revisions/sha256").iterdir()} == {d.removeprefix("sha256:")
                                                                             for d in (index, *platforms)}


def test_links_to_layers_garbage_collection_freed_are_dropped(script, tmp_path):
    kept = _blob(tmp_path, {"layer": "kept"})
    layers = tmp_path / "docker/registry/v2/repositories" / REPOSITORY / "_layers/sha256"
    for digest in (kept, "sha256:" + "e" * 64):
        (layers / digest.removeprefix("sha256:")).mkdir(parents=True)
        (layers / digest.removeprefix("sha256:") / "link").write_text(digest)

    assert script.drop_freed_links(tmp_path, REPOSITORY) == 1
    assert [p.name for p in layers.iterdir()] == [kept.removeprefix("sha256:")]


def _blob(storage: Path, content: dict) -> str:
    data = json.dumps(content).encode()
    hex_digest = hashlib.sha256(data).hexdigest()
    path = storage / "docker/registry/v2/blobs/sha256" / hex_digest[:2] / hex_digest / "data"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return f"sha256:{hex_digest}"


def _tag(storage: Path, tag: str, digest: str) -> None:
    current = storage / "docker/registry/v2/repositories" / REPOSITORY / "_manifests/tags" / tag / "current"
    current.mkdir(parents=True)
    (current / "link").write_text(digest)


def _revision(storage: Path, digest: str) -> None:
    revision = (storage / "docker/registry/v2/repositories" / REPOSITORY / "_manifests/revisions/sha256"
                / digest.removeprefix("sha256:"))
    revision.mkdir(parents=True)
    (revision / "link").write_text(digest)
