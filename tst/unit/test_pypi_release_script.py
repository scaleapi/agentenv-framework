"""``.github/scripts/pypi_release.py``, which hands the CodeArtifact release the files the GitHub release put on PyPI."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from tst.util.scripts import load_script

REPO = Path(__file__).resolve().parents[2]
WHEEL = b"wheel bytes"
SDIST = b"sdist bytes"


def _file(name: str, kind: str, data: bytes) -> dict:
    return {"filename": name, "packagetype": kind, "url": f"https://files.example/{name}",
            "digests": {"sha256": hashlib.sha256(data).hexdigest()}}


class _PyPI:
    """PyPI answering 404 until ``listed`` polls in, then listing ``files``; a package in ``never`` is never listed."""

    def __init__(self, files: list[dict], listed: int = 0, blobs: dict[str, bytes] | None = None,
                 never: frozenset[str] = frozenset()):
        self.files, self.listed, self.never, self.polls = files, listed, never, 0
        self.blobs = blobs or {"https://files.example/x-1.whl": WHEEL, "https://files.example/x-1.tar.gz": SDIST}

    def get(self, url: str) -> bytes:
        if url.endswith("/json"):
            self.polls += 1
            if self.polls <= self.listed or url.split("/")[-3] in self.never:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO())
            return json.dumps({"urls": self.files}).encode()
        return self.blobs[url]


@pytest.fixture
def script(monkeypatch):
    script = load_script("pypi_release.py")
    clock = [0.0]
    monkeypatch.setattr(script.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(script.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return script


def _both() -> list[dict]:
    return [_file("x-1.whl", "bdist_wheel", WHEEL), _file("x-1.tar.gz", "sdist", SDIST)]


def test_it_waits_for_the_release_then_downloads_every_file_into_the_package_folder(script, monkeypatch, tmp_path):
    pypi = _PyPI(_both(), listed=3)
    monkeypatch.setattr(script, "_get", pypi.get)

    script.main(["download", str(tmp_path), "x==1"])

    assert pypi.polls == 4
    assert (tmp_path / "x" / "x-1.whl").read_bytes() == WHEEL
    assert (tmp_path / "x" / "x-1.tar.gz").read_bytes() == SDIST


def test_a_wheel_without_its_sdist_is_not_yet_the_release(script, monkeypatch, tmp_path):
    pypi = _PyPI([_file("x-1.whl", "bdist_wheel", WHEEL)])
    monkeypatch.setattr(script, "_get", pypi.get)

    with pytest.raises(SystemExit, match="x==1 is not on PyPI after 1 minutes"):
        script.main(["download", str(tmp_path), "x==1", "--wait-minutes", "1"])

    assert not (tmp_path / "x").exists()


def test_a_file_whose_sha256_is_not_the_one_pypi_lists_is_refused(script, monkeypatch, tmp_path):
    pypi = _PyPI(_both(), blobs={"https://files.example/x-1.whl": b"tampered", "https://files.example/x-1.tar.gz": SDIST})
    monkeypatch.setattr(script, "_get", pypi.get)

    with pytest.raises(SystemExit, match="x-1.whl has sha256"):
        script.main(["download", str(tmp_path), "x==1"])


def test_the_wait_covers_every_requirement_together(script, monkeypatch, tmp_path):
    pypi = _PyPI(_both(), listed=3, never=frozenset({"y"}))
    monkeypatch.setattr(script, "_get", pypi.get)

    with pytest.raises(SystemExit, match="y==2 is not on PyPI after 2 minutes"):
        script.main(["download", str(tmp_path), "x==1", "y==2", "--wait-minutes", "2"])

    assert (tmp_path / "x" / "x-1.whl").exists()


@pytest.mark.parametrize("requirement", ["x", "x>=1", "==1", "x==", "x==1==2"])
def test_a_requirement_must_name_one_exact_version(script, requirement, tmp_path):
    with pytest.raises(SystemExit):
        script.main(["download", str(tmp_path), requirement])


def test_the_codeartifact_release_uploads_the_files_the_pypi_release_put_there():
    config = yaml.safe_load((REPO / ".circleci" / "config.yml").read_text())
    wait, publish = (json.dumps(config["jobs"][job]) for job in ("wait-for-pypi", "publish"))
    jobs = {name: spec for entry in config["workflows"]["publish"]["jobs"] for name, spec in entry.items()}

    assert "pypi_release.py download dist" in wait and "agentenv-framework-protocol==" in wait
    assert "aws-cli/setup" not in wait
    assert "python -m build" not in publish and "uv build" not in publish
    assert -1 < publish.find("dist/agentenv-framework-protocol/*") < publish.find("dist/agentenv-framework/*")
    assert jobs["publish"]["requires"] == ["wait-for-pypi"]
