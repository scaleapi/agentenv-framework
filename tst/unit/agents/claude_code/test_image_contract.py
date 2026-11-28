"""The agent's folder is its build context: the Dockerfile copies only its own files and the folders its
shared_sources.toml takes from agent-env's sources, keeps TLS checks on, and installs the protocol at or above its
floor, which a release doesn't pin; the image and install.sh carry every module the server imports."""

from __future__ import annotations

import ast
import re
import tomllib

from tst.unit.agents.claude_code.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
INSTALL_SH = (AGENT_DIR / "install.sh").read_text()
SOURCES = AGENT_DIR.parents[2]
SHARED = tomllib.loads((AGENT_DIR / "shared_sources.toml").read_text())

# Harbor bundles pin the same versions (agentenvhub-sdk's _DOC_TOOL_PACKAGES), so a task's documents read alike in both.
DOC_LIBS = ["openpyxl==3.1.5", "pypdf==6.21.3", "python-docx==1.2.0", "python-pptx==1.0.2"]


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def _source(name: str):
    """Where the build context's ``name`` comes from: the folder, or a folder shared_sources.toml takes from agent-env's
    own sources at that name."""
    first, _, rest = name.partition("/")
    return SOURCES / SHARED[first] / rest if first in SHARED else AGENT_DIR / name


def _in_context(name: str) -> bool:
    path = _source(name)
    return path.is_dir() if name.endswith("/") else path.is_file()


def test_every_file_the_image_copies_is_in_the_folder():
    copied = _copied()

    assert copied and all(_in_context(name) for name in copied), copied
    # install.sh and run.sh are install/v1's; a build context pruned to what the Dockerfile reads keeps them so.
    assert {"a2a_server.py", "claude_bridge.py", "claude_code_atif.py", "kit.py", "tar_changelog.py", "entrypoint.sh", "install.sh", "run.sh"} <= set(copied)



def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_and_an_install_carry_every_module_the_server_imports():
    """The folder's own modules and those the build context takes from agent-env's sources, the egress rule, the clock
    page's format and the follower among them, each by its name in the context."""
    shared = {path.stem: f"{name}/{path.name}" for name, source in SHARED.items()
              for path in (SOURCES / source).glob("*.py")}
    modules = {**shared, **{path.stem: path.name for path in AGENT_DIR.glob("*.py")}}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(_source(name)) & set(modules)
    installed = set(re.search(r'^AGENT_FILES="([^"]+)"$', INSTALL_SH, re.MULTILINE).group(1).split())

    assert {"agent_clock/agent_egress.py", "clock/clockpage.py", "tail_follow/tail_follow.py"} <= needed
    assert needed <= set(_copied()) and needed <= installed, (needed, installed)


def test_the_image_installs_the_protocol_alone_from_its_requirements():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert len(requirements) == 1, requirements
    assert re.fullmatch(r"agentenv-framework-protocol\[agent\]>=\d+\.\d+\.\d+", requirements[0]), requirements


def test_the_image_gives_the_model_node_alone_from_a_pinned_image():
    assert re.search(r"^FROM public\.ecr\.aws/docker/library/node:\d+-slim@sha256:[0-9a-f]{64} AS node$", DOCKERFILE,
                     re.MULTILINE)
    copied = re.findall(r"^COPY --from=node (.+)$", DOCKERFILE, re.MULTILINE)
    assert copied == ["/usr/local/bin/node /usr/local/bin/node"]


def test_the_packages_the_image_gives_the_models_python_are_pinned():
    packages = [line for line in (AGENT_DIR / "model-requirements.txt").read_text().splitlines()
                if line and not line.startswith("#")]

    assert packages and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", line) for line in packages), packages
    assert any(line.startswith("boto3==") for line in packages)
    assert set(DOC_LIBS) <= set(packages), packages
    assert "-r requirements.txt -r model-requirements.txt" in DOCKERFILE


def test_the_image_pins_its_cli_checks_its_checksum_and_keeps_tls_on():
    assert re.search(r"^ARG CLAUDE_CODE_VERSION=\d+\.\d+\.\d+$", DOCKERFILE, re.MULTILINE)
    assert "sha256sum -c" in DOCKERFILE
    for path in (AGENT_DIR / "Dockerfile", AGENT_DIR / "entrypoint.sh"):
        assert "NODE_TLS_REJECT_UNAUTHORIZED" not in path.read_text()


def test_the_image_starts_through_cmd_which_modal_passes_its_env_to():
    assert 'CMD ["/usr/local/bin/start-agent"]' in DOCKERFILE
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE, re.MULTILINE)
