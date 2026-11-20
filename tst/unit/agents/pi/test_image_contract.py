"""The image and an install carry what the server runs: every file the Dockerfile copies is in the build context, every
module the server imports, its own and those the context takes from agent-env's sources, is beside it in the image and
on its path once installed, and install.sh, which both run, pins Node by its checksum and pi by the version the bridge
names."""

from __future__ import annotations

import ast
import posixpath
import re
import subprocess
import sys
import tarfile
import tomllib

from agent_env.utils.docker.context import BuildContext
from tst.unit.agents.pi.agent import AGENT_DIR, pi_agent

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
INSTALL_SH = (AGENT_DIR / "install.sh").read_text()
SOURCES = AGENT_DIR.parents[2]
SHARED = tomllib.loads((AGENT_DIR / "shared_sources.toml").read_text())


def _source(name: str):
    """Where the build context's ``name`` comes from: the folder, or a folder shared_sources.toml takes from agent-env's
    own sources at that name."""
    first, _, rest = name.partition("/")
    return SOURCES / SHARED[first] / rest if first in SHARED else AGENT_DIR / name


def _copied_into() -> dict[str, set[str]]:
    """The folders the image's last stage copies each file of the build context into."""
    workdir, folders = "/", {}
    stage = re.split(r"^FROM .*$", DOCKERFILE, flags=re.MULTILINE)[-1]
    for instruction, args in re.findall(r"^(WORKDIR|COPY) (.+)$", stage, re.MULTILINE):
        if instruction == "WORKDIR":
            workdir = posixpath.join(workdir, args)
        elif "--from=" not in args:
            *sources, dest = args.split()
            dest = posixpath.join(workdir, dest)
            folder = posixpath.normpath(dest if dest.endswith("/") or len(sources) > 1 else posixpath.dirname(dest))
            for source in sources:
                folders.setdefault(source, set()).add(folder)
    return folders


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _server_modules() -> dict[str, str]:
    """Every module the server imports, transitively, that the build context holds: by module name, its name there."""
    shared = {path.stem: f"{name}/{path.name}" for name, source in SHARED.items()
              for path in (SOURCES / source).glob("*.py")}
    modules = {**shared, **{path.stem: path.name for path in AGENT_DIR.glob("*.py")}}
    needed, pending = {}, ["a2a_server"]
    while pending:
        module = pending.pop()
        if module not in needed:
            needed[module] = modules[module]
            pending += _imported(_source(modules[module])) & set(modules)
    return needed


def test_every_file_the_image_copies_is_in_its_build_context():
    copied = _copied_into()

    assert copied and all(_source(name).is_file() or _source(name).is_dir() for name in copied), copied


def test_the_image_copies_every_module_the_server_imports_beside_it():
    needed, copied = _server_modules(), _copied_into()

    assert {"pi_agent", "agent_egress", "clockpage", "changelog", "peers"} <= set(needed)
    assert all("/app" in copied.get(name, set()) for name in needed.values()), (needed, copied)
    assert "WORKDIR /app" in DOCKERFILE and "a2a_server.py" in (AGENT_DIR / "root.conf").read_text()


def test_an_install_imports_every_module_the_server_imports_from_the_build_context_it_unpacks(tmp_path):
    """install/v1 unpacks the build context, which keeps the shared modules in the folders it takes them as; install.sh
    copies them beside the bridge, and start.sh runs the server from there, with no PYTHONPATH the model's commands
    would inherit."""
    archive, here = tmp_path / "ctx.tar.gz", tmp_path / "pi-a2a"
    BuildContext.of(AGENT_DIR, AGENT_DIR / "Dockerfile").write(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(here, filter="data")
    beside = INSTALL_SH[INSTALL_SH.index('if [ -d "$here/agent_clock" ]'):]
    imports = ", ".join(sorted(set(_server_modules()) - {"a2a_server"}))

    subprocess.run(["sh", "-c", f'here={here}; {beside}'], check=True, env={"PATH": "/usr/bin:/bin"})
    subprocess.run([sys.executable, "-c", f"import {imports}"], cwd=here, check=True, env={"PATH": "/usr/bin:/bin"})
    assert "PYTHONPATH" not in (AGENT_DIR / "start.sh").read_text()


def test_install_sh_pins_node_by_its_checksum_and_pi_by_the_version_the_bridge_names():
    version = re.search(r"^NODE_VERSION=(\S+)$", INSTALL_SH, re.MULTILINE).group(1)
    sums = dict(re.findall(r"^NODE_SHA256_(X64|ARM64)=([0-9a-f]{64})$", INSTALL_SH, re.MULTILINE))

    # Node honours NODE_USE_ENV_PROXY, which the egress rule sets for the programs the model runs, from 22.21.
    assert tuple(int(part) for part in version.split(".")) >= (22, 21)
    assert set(sums) == {"X64", "ARM64"}
    assert 'echo "$node_sha256  $node_archive" | sha256sum -c -' in INSTALL_SH
    assert re.search(r"^PI_VERSION=(\S+)$", INSTALL_SH, re.MULTILINE).group(1) == pi_agent.PI_VERSION


def test_the_bridge_starts_the_pi_and_the_node_install_sh_installs():
    """The bridge hands pi's bin to node itself, so node is the process the real-time marker reaches; the image and
    start.sh put install.sh's Node first among the nodes on the PATH."""
    node, script = pi_agent.PI_COMMAND

    assert node == "node" and script == "/opt/pi/bin/pi"
    assert '--prefix /opt/pi --ignore-scripts \\\n  "@earendil-works/pi-coding-agent@${PI_VERSION}"' in INSTALL_SH
    assert 'ln -sfn "/opt/node-v${NODE_VERSION}-linux-${arch}" /opt/node' in INSTALL_SH
    for path in (re.search(r'^ENV PATH="([^"]+)"$', DOCKERFILE, re.MULTILINE).group(1),
                 re.search(r'^export PATH="([^"]+)"$', (AGENT_DIR / "start.sh").read_text(), re.MULTILINE).group(1)):
        entries = path.split(":")
        assert "/opt/node/bin" in entries and not any("node" in entry for entry in entries[:entries.index("/opt/node/bin")])
