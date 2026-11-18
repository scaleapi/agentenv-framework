"""Copy every composed computer image a slow-tier runner's registry holds to the CI computer image mirror, so later runs
copy them back rather than build them."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from agent_env.env.envs.computer.utils.composed import REPOSITORY, mirror_tag
from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.store.image_store.registry_api import copy_image, pin_digest, pin_if_served

KEY = re.compile(r"^[0-9a-f]{64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mirror", help="the mirror repository, as ghcr.io/org/agentenv-ci-computers")
    parser.add_argument("--registry", default="localhost:5000", help="the runner's registry")
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(f"http://{args.registry}/v2/{REPOSITORY}/tags/list") as response:
            keys = sorted(tag for tag in json.load(response).get("tags") or [] if KEY.match(tag))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        keys = []
    auth = RegistryAuth(args.mirror.partition("/")[0], os.environ["GITHUB_ACTOR"], os.environ["GITHUB_TOKEN"])
    for key in keys:
        source = pin_digest(f"{args.registry}/{REPOSITORY}:{key}", None)
        copy_image(source, f"{args.mirror}:{mirror_tag(key)}", None, auth)
        print(f"shared {key}")
    print(f"share computer images: {len(keys)} images in {args.mirror}")
    if keys:
        try:
            pin_if_served(f"{args.mirror}:{mirror_tag(keys[0])}", None)
        except ValueError as e:
            print(f"::warning::the composer reads {args.mirror} anonymously and can't, so every run builds its images; "
                  f"make the package public: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
