"""The agent's folder is its build context: the Dockerfile copies only its own files, keeps TLS checks on, and
installs the protocol at or above its floor, which a release doesn't pin."""

from __future__ import annotations

import re

from tst.unit.agents.claude_code.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()


def test_every_file_the_image_copies_is_in_the_folder():
    copied = [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
              for name in line.split()]

    assert copied and all((AGENT_DIR / name).is_file() for name in copied), copied
    # install.sh and run.sh are install/v1's; a build context pruned to what the Dockerfile reads keeps them so.
    assert {"a2a_server.py", "claude_bridge.py", "kit.py", "tar_changelog.py", "entrypoint.sh", "install.sh", "run.sh"} <= set(copied)


def test_the_image_installs_the_protocol_alone_from_its_requirements():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]>=0.1.351"]


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
    assert "-r requirements.txt -r model-requirements.txt" in DOCKERFILE


def test_the_image_pins_its_cli_checks_its_checksum_and_keeps_tls_on():
    assert re.search(r"^ARG CLAUDE_CODE_VERSION=\d+\.\d+\.\d+$", DOCKERFILE, re.MULTILINE)
    assert "sha256sum -c" in DOCKERFILE
    for path in (AGENT_DIR / "Dockerfile", AGENT_DIR / "entrypoint.sh"):
        assert "NODE_TLS_REJECT_UNAUTHORIZED" not in path.read_text()


def test_the_image_starts_through_cmd_which_modal_passes_its_env_to():
    assert 'CMD ["/opt/agent/entrypoint.sh"]' in DOCKERFILE
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE, re.MULTILINE)
