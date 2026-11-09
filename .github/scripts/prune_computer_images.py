"""Delete the CI computer image mirror's images no run has used for ``--days`` days.

``share_computer_images.py`` tags each image a passing main run used ``<tag>-seen-YYYYMMDD``. An image is kept while its
newest such tag, or its upload when it has none, is within ``--days`` of now, with every platform manifest a kept index
lists, since GHCR lists those as versions of their own. Every other version is deleted through GitHub's packages API,
read and written with ``GITHUB_TOKEN``, as GHCR is. A version GitHub won't delete warns. ``--dry-run`` prints what
would go and deletes nothing.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

SEEN = re.compile(r"^[0-9a-f]{16}-[0-9a-f]{64}-seen-(\d{8})$")
_INDEXES = {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
_ACCEPT = ", ".join([*_INDEXES, "application/vnd.oci.image.manifest.v1+json",
                     "application/vnd.docker.distribution.manifest.v2+json"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("org")
    parser.add_argument("package", help="the container package, as agentenv-ci-computers")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = f"https://api.github.com/orgs/{args.org}/packages/container/{args.package}/versions"
    try:
        versions = _versions(api)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        print(f"::warning::prune computer images: no package {args.org}/{args.package}, or this repository has no "
              "access to it, so nothing is pruned")
        return 0
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.days)
    stale = stale_versions(versions, cutoff, lambda digest: _children(args.org, args.package, digest))
    failed = 0
    for version in stale:
        print(f"{'would delete' if args.dry_run else 'deleting'} {version['name']} "
              f"{version['metadata']['container']['tags'] or '(untagged)'}")
        if not args.dry_run:
            try:
                _github("DELETE", f"{api}/{version['id']}")
            except urllib.error.HTTPError as e:
                failed += 1
                print(f"::warning::couldn't delete {version['name']}: HTTP {e.code} {e.read().decode()[:200]}")
    print(f"prune computer images: {len(stale) - failed} of {len(versions)} versions "
          f"{'would be ' if args.dry_run else ''}deleted, none used since {cutoff:%Y-%m-%d}")
    return 0


def stale_versions(versions: list[dict], cutoff: datetime.datetime, children) -> list[dict]:
    """The ``versions`` to delete: neither used nor uploaded since ``cutoff``, nor a platform manifest of one kept.
    ``children(digest)`` lists the digests of the manifests the index of that digest lists, or none for an image."""
    kept = {version["name"] for version in versions if _last_used(version) >= cutoff}
    for digest in list(kept):
        kept.update(children(digest))
    return [version for version in versions if version["name"] not in kept]


def _last_used(version: dict) -> datetime.datetime:
    created = datetime.datetime.fromisoformat(version["created_at"].replace("Z", "+00:00"))
    seen = [datetime.datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
            for tag in version["metadata"]["container"]["tags"] if (m := SEEN.match(tag))]
    return max([created, *(day + datetime.timedelta(days=1) for day in seen)])


def _versions(api: str) -> list[dict]:
    versions, page = [], 1
    while batch := _github("GET", f"{api}?per_page=100&page={page}"):
        versions += batch
        page += 1
    return versions


def _children(org: str, package: str, digest: str) -> list[str]:
    """The platform manifests the index ``digest`` lists, read from GHCR with ``GITHUB_TOKEN``; none for an image."""
    repository = f"{org}/{package}"
    signed_in = base64.b64encode(f"x:{os.environ['GITHUB_TOKEN']}".encode()).decode()
    with urllib.request.urlopen(urllib.request.Request(f"https://ghcr.io/token?scope=repository:{repository}:pull",
                                                       headers={"Authorization": f"Basic {signed_in}"})) as response:
        token = json.load(response)["token"]
    request = urllib.request.Request(f"https://ghcr.io/v2/{repository}/manifests/{digest}",
                                     headers={"Authorization": f"Bearer {token}", "Accept": _ACCEPT})
    with urllib.request.urlopen(request) as response:
        manifest = json.load(response)
        media = response.headers.get("Content-Type", "").split(";")[0].strip()
    return [child["digest"] for child in manifest.get("manifests", [])] if media in _INDEXES else []


def _github(method: str, url: str):
    request = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(request) as response:
        body = response.read()
    return json.loads(body) if body else None


if __name__ == "__main__":
    sys.exit(main())
