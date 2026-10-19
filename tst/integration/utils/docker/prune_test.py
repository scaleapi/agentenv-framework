"""An image built from what ``write_pruned`` keeps holds what one built from the whole context does. Needs Docker."""

import hashlib
import io
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

import pytest

from agent_env.utils.docker.prune import write_pruned
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BASE = "mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf"
DOCKERFILE = f"""FROM {BASE} AS build
COPY tools/gen.sh /gen.sh
RUN sh /gen.sh > /built.txt

FROM {BASE}
COPY --from=build /built.txt /out/built.txt
COPY app/ /app/
COPY conf/*.cfg /etc/svc/
ADD vendor.tar.gz /vendor/
COPY current/ /release/
COPY current/app.txt /through-link.txt
COPY lit/[!a].txt /lit/
RUN --network=none	--mount=type=bind,"source=my dir",target=/m cp /m/f /out/spaced.txt
RUN --mount=type=cache,target=/root/.cache --mount=type=BIND,source=scripts,target=/scripts cp /scripts/run.sh /out/run.sh
"""
# What the build reads, beside a context ten times bigger it never does.
CONTEXT = {
    "server/real.Dockerfile": DOCKERFILE,
    "lit/!.txt": "bang\n", "lit/a.txt": "a\n", "lit/b.txt": "b\n", "my dir/f": "spaced\n",
    ".dockerignore": "app/tests\n**/*.pyc\n",
    "tools/gen.sh": "echo generated\n", "app/main.py": "print('main')\n", "app/util.py": "u = 1\n",
    "app/tests/test_main.py": "assert True\n", "app/cache.pyc": "compiled", "conf/a.cfg": "a=1\n",
    "conf/b.cfg": "b=2\n", "conf/readme.md": "not a cfg\n", "scripts/run.sh": "exec python /app/main.py\n",
    "releases/v2/app.txt": "release 2\n", "releases/v1/app.txt": "release 1\n",
    **{f"websites/site{n}/index.html": f"<p>{n}</p>" * 2000 for n in range(10)},
}
VENDOR = {"lib/vendored.py": "V = True\n"}


def _add(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    tar.addfile(info, io.BytesIO(data))


def _stored_context(path: Path) -> Path:
    """The context as a store might hold it, ignored files and all, with a vendored archive and a release link."""
    vendor = io.BytesIO()
    with tarfile.open(fileobj=vendor, mode="w:gz") as tar:
        for name, text in VENDOR.items():
            _add(tar, name, text.encode())
    with tarfile.open(path, "w:gz") as tar:
        for name, text in CONTEXT.items():
            _add(tar, name, text.encode(), 0o755 if name.endswith(".sh") else 0o644)
        _add(tar, "vendor.tar.gz", vendor.getvalue())
        for name, target in (("current", "releases/v2"), ("server/Dockerfile", "real.Dockerfile")):
            link = tarfile.TarInfo(name)
            link.type, link.linkname = tarfile.SYMTYPE, target
            tar.addfile(link)
    return path


def _build(context: Path, tag: str) -> None:
    result = subprocess.run(["docker", "build", "-q", "-f", str(context / "server" / "Dockerfile"), "-t", tag,
                             str(context)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _image_files(tag: str) -> dict[str, tuple[int, str]]:
    """Each file, folder and link the image holds under what the build wrote, by its path: its mode and content."""
    container = subprocess.run(["docker", "create", tag], capture_output=True, text=True, check=True).stdout.strip()
    try:
        exported = subprocess.run(["docker", "export", container], capture_output=True, check=True).stdout
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True)
    found = {}
    with tarfile.open(fileobj=io.BytesIO(exported)) as tar:
        for member in tar:
            if member.name.split("/", 1)[0] not in ("app", "out", "etc", "vendor", "release", "through-link.txt", "lit"):
                continue
            content = hashlib.sha256(tar.extractfile(member).read()).hexdigest() if member.isreg() else member.linkname
            found[member.name] = (member.mode, content)
    return found


@pytest.fixture
def docker():
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    tags = []
    yield tags
    for tag in tags:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def test_an_image_built_from_the_pruned_context_holds_what_one_from_the_whole_context_does(tmp_path, docker):
    archive = _stored_context(tmp_path / "ctx.tar.gz")
    full = tmp_path / "full"
    with tarfile.open(archive) as tar:
        tar.extractall(full, filter="tar")

    pruned = write_pruned(archive, {tmp_path / "pruned": "server/Dockerfile"})

    kept = {str(path.relative_to(tmp_path / "pruned")) for path in (tmp_path / "pruned").rglob("*")
            if path.is_file() or path.is_symlink()}
    assert kept == {"server/Dockerfile", "server/real.Dockerfile", ".dockerignore", "tools/gen.sh", "app/main.py",
                    "app/util.py", "conf/a.cfg", "conf/b.cfg", "vendor.tar.gz", "current", "releases/v2/app.txt",
                    "scripts/run.sh", "lit/!.txt", "lit/a.txt", "my dir/f"}
    assert pruned[tmp_path / "pruned"].whole is None
    suffix = uuid.uuid4().hex[:8]
    docker.extend([f"prune-full-{suffix}", f"prune-kept-{suffix}"])
    _build(full, docker[0])
    _build(tmp_path / "pruned", docker[1])

    built = _image_files(docker[1])
    assert built == _image_files(docker[0])
    assert {"app/main.py", "out/built.txt", "out/run.sh", "etc/svc/a.cfg", "vendor/lib/vendored.py",
            "release/app.txt", "through-link.txt", "lit/!.txt", "lit/a.txt", "out/spaced.txt"} <= set(built)
    assert "lit/b.txt" not in built
    assert not any(name.startswith("app/tests") or name.endswith(".pyc") for name in built)


def test_folders_of_one_name_written_from_two_stored_contexts_each_build_their_own_files(tmp_path, docker):
    # docker build skips sending a file whose path, size and time match one it was sent before, as a stored context's
    # fixed times would make them, and builds the old file.
    name = f"gmail-{uuid.uuid4().hex[:8]}"
    built = []
    for version in ("1", "2"):
        archive = tmp_path / f"v{version}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            _add(tar, "Dockerfile", f"FROM {BASE}\nCOPY version.txt /version.txt\n".encode())
            _add(tar, "version.txt", f"version {version}\n".encode())
        folder = tmp_path / f"bundle-v{version}" / "envs" / name
        write_pruned(archive, {folder: "Dockerfile"})
        docker.append(f"prune-{name}-v{version}")
        result = subprocess.run(["docker", "build", "-q", "--no-cache", "-t", docker[-1], str(folder)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        built.append(subprocess.run(["docker", "run", "--rm", docker[-1], "cat", "/version.txt"], capture_output=True,
                                    text=True, check=True).stdout)

    assert built == ["version 1\n", "version 2\n"]
