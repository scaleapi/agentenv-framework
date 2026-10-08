"""Read an image's digest from its registry over the OCI distribution API, to pin a tag to the image it names now."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import httpx

from agent_env.store.image_store.oci_registry_credentials import RegistryAuth, is_loopback_host, normalize_registry_host

# An index (multi-platform) or a single manifest, OCI or Docker: the digest is of whichever the registry serves.
_MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
_DIGEST = re.compile(r"^[a-z0-9]+(?:[.+_-][a-z0-9]+)*:[0-9a-fA-F]{32,}$")
_DOCKER_HUB_API = "registry-1.docker.io"
_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class _ImageRef:
    name: str  # the reference without its tag or digest, spelled as given
    host: str
    repository: str
    tag: str | None
    digest: str | None


def pin_digest(ref: str, auth: RegistryAuth | None) -> str:
    """``ref`` pinned to the digest its registry serves for it now, as ``name:tag@sha256:...``.

    A tag (``latest`` when none is given) is resolved; a digest is checked to exist; a ref giving both must agree,
    since a pull goes by the digest and ignores the tag. ``auth`` is used where the registry asks for credentials;
    without it, a registry's anonymous token is. Raises ValueError naming what the registry answered.
    """
    image = _parse(ref)
    if image.digest is not None:
        _check(image, image.digest, auth)
        if image.tag is None:
            return ref
        served = _check(image, image.tag, auth)
        if served != image.digest:
            raise ValueError(f"{ref}: the tag {image.tag!r} names {served} now, not {image.digest}; give the tag or the "
                             "digest alone")
        return ref
    tag = image.tag or "latest"
    return f"{image.name}:{tag}@{_check(image, tag, auth)}"


def _parse(ref: str) -> _ImageRef:
    name, _, digest = ref.partition("@")
    if digest and not _DIGEST.match(digest):
        raise ValueError(f"{ref}: {digest!r} isn't a digest (algorithm:hex)")
    head, slash, tail = name.rpartition("/")
    tag = None
    if ":" in tail:
        tail, tag = tail.split(":", 1)
        name = f"{head}{slash}{tail}"
    host, _, repository = name.partition("/")
    host = normalize_registry_host(host)
    if host == "docker.io":
        host = _DOCKER_HUB_API
        if "/" not in repository:
            repository = f"library/{repository}"
    return _ImageRef(name=name, host=host, repository=repository, tag=tag, digest=digest or None)


def _check(image: _ImageRef, reference: str, auth: RegistryAuth | None) -> str:
    """The digest the registry serves for ``reference`` (a tag or a digest) in ``image``'s repository."""
    # A registry on this machine serves plain HTTP, as Docker allows for a loopback one.
    scheme = "http" if is_loopback_host(image.host) else "https"
    url = f"{scheme}://{image.host}/v2/{image.repository}/manifests/{reference}"
    what = f"{image.name}{'@' if _DIGEST.match(reference) else ':'}{reference}"
    basic = (auth.username, auth.password) if auth is not None else None
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=True) as client:
            headers = {"Accept": _MANIFEST_TYPES}
            # Basic credentials go with every request until a Bearer token replaces them.
            credentials = basic
            response = client.head(url, headers=headers, auth=credentials)
            if response.status_code == 401 and (token := _bearer_token(client, response, basic)) is not None:
                headers["Authorization"] = f"Bearer {token}"
                credentials = None
                response = client.head(url, headers=headers)
            if response.status_code == 200 and not response.headers.get("docker-content-digest"):
                response = client.get(url, headers=headers, auth=credentials)
                if response.status_code == 200:
                    return f"sha256:{hashlib.sha256(response.content).hexdigest()}"
    except httpx.HTTPError as e:
        raise ValueError(f"{what}: couldn't read its manifest from {image.host}: {type(e).__name__}: {e}") from e
    if response.status_code == 200:
        served = response.headers["docker-content-digest"]
        if not _DIGEST.match(served):
            raise ValueError(f"{what}: {image.host} answered with {served!r}, which isn't a digest")
        return served
    if response.status_code == 404:
        raise ValueError(f"{what}: {image.host} has no such image")
    if response.status_code in (401, 403):
        raise ValueError(f"{what}: {image.host} refused to serve it (HTTP {response.status_code}); agent-env reaches a "
                         "private registry only with the configured image store's credentials, for its own registry")
    raise ValueError(f"{what}: {image.host} answered HTTP {response.status_code}")


def _bearer_token(client: httpx.Client, challenged: httpx.Response, basic: tuple[str, str] | None) -> str | None:
    """A token from the realm a ``Bearer`` challenge names, asked for with ``basic`` when given. None for any other
    challenge, or when the realm grants none."""
    challenge = challenged.headers.get("www-authenticate", "")
    if not challenge.lower().startswith("bearer "):
        return None
    params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = params.pop("realm", None)
    if not realm:
        return None
    response = client.get(realm, params=params, auth=basic)
    if response.status_code != 200:
        return None
    body = response.json()
    return body.get("token") or body.get("access_token")
