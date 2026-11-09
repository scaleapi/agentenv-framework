"""Copy every composed computer image a slow-tier runner's registry holds to the CI computer image mirror, the one
``AGENT_ENV_COMPUTER_IMAGE_MIRROR`` names, so later runs copy them back rather than build them.

Each image goes under its mirror tag (``composed.mirror_tag``: the composer's digest, then the key), and again with
today's date, ``<tag>-seen-YYYYMMDD``, which is how ``prune_computer_images.py`` tells an image a recent run used from one
none has. An image already in the mirror is only tagged again: no blob it holds is sent twice. The mirror is written as
``GITHUB_ACTOR`` with ``GITHUB_TOKEN``; one the composer can't read anonymously, as GHCR makes a new package, warns.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
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

    with urllib.request.urlopen(f"http://{args.registry}/v2/{REPOSITORY}/tags/list") as response:
        keys = sorted(tag for tag in json.load(response).get("tags") or [] if KEY.match(tag))
    auth = RegistryAuth(args.mirror.partition("/")[0], os.environ["GITHUB_ACTOR"], os.environ["GITHUB_TOKEN"])
    seen = f"seen-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d}"
    for key in keys:
        source = pin_digest(f"{args.registry}/{REPOSITORY}:{key}", None)
        for tag in (mirror_tag(key), f"{mirror_tag(key)}-{seen}"):
            copy_image(source, f"{args.mirror}:{tag}", None, auth)
        print(f"shared {key}")
    print(f"share computer images: {len(keys)} images in {args.mirror}, each tagged {seen}")
    if keys:
        try:
            pin_if_served(f"{args.mirror}:{mirror_tag(keys[0])}", None)
        except ValueError as e:
            print(f"::warning::the composer reads {args.mirror} anonymously and can't, so every run builds its images; "
                  f"make the package public: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
