"""The computer apps a release publishes composed (``APPS``, chromium), for the release workflow's jobs that only read:
this imports agent-env, so it runs where this release's wheels are installed, and no job that holds a token to push
images runs it (those run ``infra_build.py``).

``compose ARCH OUT`` composes each app for linux/ARCH on this machine's Docker, emulating nothing, as an install of this
release composes it: from the wheels installed, over the base computers the release published, its packages resolved
from Ubuntu's archive, PyPI and download URLs. Each is composed in stores of its own, so into the local registry's
``agentenv-computers``, and probed at full (gate 3). It writes, as OUT/ARCH.json, each app's installed env id, form and
arch, the key it took, its image there by digest and the lock it was composed from, and fails unless each one is
ready. The local registry is the one at localhost:5000, and the stores are the configured local ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.env import Env
from agent_env.env.envs.computer.utils import packages
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.store.routing import namespace_routing

#: Each app a release publishes composed, by the alias a task names it by and its installed env id.
APPS = {"chromium": "@local/agentenv-framework/browsers/chromium"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    compose = commands.add_parser("compose")
    compose.add_argument("arch", choices=["amd64", "arm64"])
    compose.add_argument("out", type=Path)
    args = parser.parse_args()
    return _compose(args.arch, args.out)


def _compose(arch: str, out: Path) -> int:
    if host_platform() != f"linux/{arch}":
        print(f"::error::this runner is {host_platform()}, and a release composes linux/{arch} natively")
        return 1
    with tempfile.TemporaryDirectory(prefix="agentenv-release-") as work, namespace_routing():
        tasks = Path(work) / "tasks"
        tasks.mkdir()
        for alias in APPS:
            (tasks / f"{alias}.json").write_text(json.dumps([{"id": "env", "type": "deploy_env", "env_id": alias}]))
        plan = plan_bundle(resolve_bundle(parse_bundle(Path(work))), installed=installed_bundles, default_agents=False)
        materialize(plan, composing=Composing(None, emulate=False, on_progress=print))
        entries = []
        for env_id in APPS.values():
            env = Env.get(env_id)
            built = env.metadata.get("composed", {}).get("container", {}).get(arch, {})
            if built.get("status") != "ready":
                print(f"::error::{env_id}'s container/{arch} is {built.get('status', 'not composed')}: "
                      f"{built.get('reason', 'no reason recorded')}")
                return 1
            lock = packages.stored(built["lock"], env.stored_id)
            entries.append({"env": env_id, "form": "container", "arch": arch, "key": built["key"],
                            "image": built["image"], "lock": json.loads(lock.model_dump_json())})
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{arch}.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"composed {', '.join(APPS)} for linux/{arch}: {out / f'{arch}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
