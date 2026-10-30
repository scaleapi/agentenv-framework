"""A builder VM on Modal, amd64, building for arm64 through the emulation it registers: the build context sent inline,
an apt package and a requirement resolved in a container of the base, a file it wrote stored off the VM, an image built
from them with the network closed, and the packages it installed read back, each in the VM. The stores stay local,
which a compose on Modal refuses, so the builder is driven directly."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_env.config import get_config
from agent_env.env.envs.computer.utils import builder
from agent_env.providers import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_remote_sandbox("modal_vm")]

BASE = "mirror.gcr.io/library/python:3.12-slim"
DOCKERFILE = f"""FROM {BASE}
COPY packages/ /tmp/packages/
RUN --network=none apt-get install -y --no-install-recommends /tmp/packages/apt/*.deb
COPY driver/ /opt/driver/
RUN --network=none python3 -m venv /opt/driver/venv && /opt/driver/venv/bin/pip install --no-index --no-deps \
/tmp/packages/wheels/*.whl && /opt/driver/venv/bin/pip list --format=json > /opt/driver/installed.json && \
/opt/driver/start && jq --version
"""


def test_a_builder_vm_resolves_stores_and_builds_an_arch_it_emulates(tmp_path):
    driver = tmp_path / builder.CONTEXT / "driver"
    driver.mkdir(parents=True)
    (driver / "requirements.txt").write_text("six==1.16.0\n")
    (driver / "start").write_text("#!/bin/sh\necho started\n")
    (driver / "start").chmod(0o755)
    vm = builder.VmBuilder(build_sandbox_provider("modal_vm"))
    try:
        vm.start(tmp_path, SimpleNamespace(image_name=BASE, pinned_name=BASE), "linux/arm64", {})
        result = vm.resolve({"apt": ["jq"], "sources": [], "python": [
            {"name": "driver", "folder": "/ctx/driver", "requirements": "requirements.txt", "add": []}]},
            BASE, "arm64", 1800)
        written = vm.files(f"{builder.RESOLVED}/out")
        (wheel,) = result["wheels"]["driver"]
        store = get_config().get_object_store()
        vm.store(f"{builder.RESOLVED}/out/wheels/driver/{wheel}", store, f"builder-test/{wheel}")
        stored = tmp_path / "stored.whl"
        store.download_to_file(store.object_url(f"builder-test/{wheel}"), str(stored))
        vm.place({f"{builder.CONTEXT}/packages/apt/{deb['file']}": f"{builder.RESOLVED}/out/apt/{deb['file']}"
                  for deb in result["apt"]} | {f"{builder.CONTEXT}/packages/wheels/{wheel}":
                                               f"{builder.RESOLVED}/out/wheels/driver/{wheel}"}, {})
        vm.build(DOCKERFILE, "agentenv-builder-test:arm64", "linux/arm64", {"dev.agentenv.test": "builder"})
        installed = vm.installed("agentenv-builder-test:arm64", "/opt/driver")
    finally:
        vm.close()

    assert wheel == "six-1.16.0-py2.py3-none-any.whl" and installed["six"] == "1.16.0"
    assert {deb["package"] for deb in result["apt"]} >= {"jq", "libjq1"}
    assert {deb["arch"] for deb in result["apt"]} == {"arm64"}
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == written[f"wheels/driver/{wheel}"].sha256
    assert vm.sandbox.platform == "linux/amd64" and Path(stored).stat().st_size == written[f"wheels/driver/{wheel}"].size
