"""Upstream's source at OSWORLD_COMMIT: fetched once from GitHub's archive of the commit into the cache, checked by
sha256 and kept to the files the steps run; an archive of another sha256 is refused; a folder named by
AGENT_ENV_OSWORLD_SOURCE is used as it is, and one without upstream's evaluators is refused; easyocr is stood in for
when it isn't installed."""
from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

from agent_env.task_step.task_steps.computer import osworld
from agent_env.task_step.task_steps.computer.osworld import upstream

TOP = f"OSWorld-{osworld.OSWORLD_COMMIT}"


def _archive() -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name in ["LICENSE", "desktop_env/__init__.py", "desktop_env/evaluators/metrics/__init__.py",
                     "desktop_env/controllers/setup.py", "desktop_env/server/main.py", "mm_agents/agent.py"]:
            data = f"# {name}\n".encode()
            info = tarfile.TarInfo(f"{TOP}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


@pytest.fixture
def served(monkeypatch):
    archive = _archive()
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(archive)

    monkeypatch.setattr(upstream.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(upstream, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    return calls


def test_the_archive_is_upstreams_at_the_packages_commit():
    assert upstream.ARCHIVE_URL == f"https://github.com/xlang-ai/OSWorld/archive/{osworld.OSWORLD_COMMIT}.tar.gz"


def test_the_source_is_fetched_once_and_kept_to_what_the_steps_run(served, tmp_path):
    dest = upstream.fetch(tmp_path / "src")
    upstream.fetch(tmp_path / "src")
    assert served == [upstream.ARCHIVE_URL]
    kept = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and p.name != ".complete")
    assert kept == ["LICENSE", "desktop_env/__init__.py", "desktop_env/controllers/setup.py",
                    "desktop_env/evaluators/metrics/__init__.py"]


def test_an_archive_of_another_sha256_is_refused(served, monkeypatch, tmp_path):
    monkeypatch.setattr(upstream, "ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="has sha256"):
        upstream.fetch(tmp_path / "src")
    assert not (tmp_path / "src").exists()


def test_the_cache_is_under_the_cache_root_by_commit(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert upstream.cache_dir() == tmp_path / "agentenv" / "osworld" / osworld.OSWORLD_COMMIT


def test_a_named_folder_is_used_as_it_is(served, monkeypatch, tmp_path):
    (tmp_path / "baked" / "desktop_env" / "evaluators").mkdir(parents=True)
    monkeypatch.setenv(upstream.SOURCE_ENV, str(tmp_path / "baked"))
    assert upstream.source() == tmp_path / "baked" and served == []
    monkeypatch.setenv(upstream.SOURCE_ENV, str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="holds no desktop_env/evaluators"):
        upstream.source()


def test_easyocr_is_stood_in_for_when_it_isnt_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "easyocr", raising=False)
    monkeypatch.setattr(upstream.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name)))
    upstream._easyocr_stub()
    with pytest.raises(RuntimeError, match="compare_image_text needs easyocr"):
        sys.modules.pop("easyocr").Reader(["en"])



def test_dev_installs_what_the_requirements_name_for_the_e2e_test():
    import tomllib

    named = [line for line in upstream.REQUIREMENTS.read_text().splitlines() if line and not line.startswith("#")]
    dev = tomllib.loads((Path(osworld.__file__).parents[6] / "pyproject.toml").read_text())
    assert named and set(named) <= set(dev["project"]["optional-dependencies"]["dev"])


def test_a_worker_that_fetched_second_keeps_the_first_ones_copy(served, tmp_path, monkeypatch):
    """Two workers sharing an empty cache both fetch; the second's rename is refused and the first's files stay."""
    dest = tmp_path / "src"
    real = upstream.Path.rename

    def raced(self, target):
        if Path(target) == dest and not dest.exists():
            first = upstream.fetch(tmp_path / "first")
            real(first, dest)
            (dest / "desktop_env" / "by_first").touch()
        return real(self, target)

    monkeypatch.setattr(upstream.Path, "rename", raced)
    assert upstream.fetch(dest) == dest
    assert (dest / "desktop_env" / "by_first").exists() and (dest / ".complete").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["src"]
