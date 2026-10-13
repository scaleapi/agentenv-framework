"""Read an image's digest from its registry over the OCI distribution API, to pin a tag to the image it names now, and
its config, to run it where only its name is known."""

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
# The digests Docker pulls by: the algorithm, then its full-length lowercase hex.
_DIGEST = re.compile(r"^(?:sha256:[a-f0-9]{64}|sha384:[a-f0-9]{96}|sha512:[a-f0-9]{128})$")
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
    if image.digest is None:
        tag = image.tag or "latest"
        return f"{image.name}:{tag}@{_check(image, tag, auth)}"
    served = _check(image, image.tag or image.digest, auth)
    if image.tag and served != image.digest:
        raise ValueError(f"{ref}: the tag {image.tag!r} names {served} now, not {image.digest}; give the tag or the "
                         "digest alone")
    return ref


def image_config(ref: str, auth: RegistryAuth | None, platform: str = "linux/amd64") -> dict:
    """The run config of the image ``ref`` names, as its config blob's ``config`` holds it (``Entrypoint``, ``Cmd``,
    ``Env``, ``WorkingDir``, ``User``), ``platform``'s from a multi-platform index. ``auth`` as for ``pin_digest``.
    Raises ValueError naming what the registry answered."""
    image = _parse(_qualified(ref))
    reference = image.digest or image.tag or "latest"
    what = f"{image.name}{'@' if image.digest else ':'}{reference}"
    scheme = "http" if is_loopback_host(image.host) else "https"
    base = f"{scheme}://{image.host}/v2/{image.repository}"
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=True) as client:
            fetch = _Fetch(client, auth, what)
            manifest = fetch.json(f"{base}/manifests/{reference}", _MANIFEST_TYPES)
            if (entries := manifest.get("manifests")) is not None:
                os_name, _, arch = platform.partition("/")
                digest = next((m.get("digest") for m in entries
                               if (m.get("platform") or {}).get("os") == os_name
                               and (m.get("platform") or {}).get("architecture") == arch), None)
                if digest is None:
                    raise ValueError(f"{what}: {image.host} serves no {platform} image for it")
                manifest = fetch.json(f"{base}/manifests/{digest}", _MANIFEST_TYPES)
            config_digest = (manifest.get("config") or {}).get("digest")
            if not config_digest:
                raise ValueError(f"{what}: {image.host} served a manifest with no config")
            blob = fetch.json(f"{base}/blobs/{config_digest}", "*/*")
    except httpx.HTTPError as e:
        raise ValueError(f"{what}: couldn't read its config from {image.host}: {type(e).__name__}: {e}") from e
    return blob.get("config") or {}


def _qualified(ref: str) -> str:
    """``ref`` with Docker Hub's host when it names none, as Docker reads it: a first part with no ``.`` or ``:`` that
    isn't ``localhost`` is a Docker Hub path."""
    first, slash, _ = ref.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        return ref
    return f"docker.io/{ref}"


class _Fetch:
    """GETs from one registry, with ``auth`` until its Bearer challenge gives a token, which then replaces it."""

    def __init__(self, client: httpx.Client, auth: RegistryAuth | None, what: str):
        self._client = client
        self._basic = (auth.username, auth.password) if auth is not None else None
        self._token: str | None = None
        self._what = what

    def json(self, url: str, accept: str) -> dict:
        response = self._get(url, accept)
        if response.status_code == 401 and self._token is None:
            self._token = _bearer_token(self._client, response, self._basic, self._what)
            if self._token is not None:
                response = self._get(url, accept)
        if response.status_code != 200:
            raise ValueError(f"{self._what}: {url} answered HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise ValueError(f"{self._what}: {url} answered with something other than a JSON object")
        return body

    def _get(self, url: str, accept: str) -> httpx.Response:
        if self._token is not None:
            return self._client.get(url, headers={"Accept": accept, "Authorization": f"Bearer {self._token}"})
        return self._client.get(url, headers={"Accept": accept}, auth=self._basic)


def _parse(ref: str) -> _ImageRef:
    name, _, digest = ref.partition("@")
    if digest and not _DIGEST.match(digest):
        raise ValueError(f"{ref}: {digest!r} isn't a sha256, sha384 or sha512 digest")
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
            if response.status_code == 401 and (token := _bearer_token(client, response, basic, what)) is not None:
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


def _bearer_token(client: httpx.Client, challenged: httpx.Response, basic: tuple[str, str] | None,
                  what: str) -> str | None:
    """A token from the realm a ``Bearer`` challenge names, asked for with ``basic`` when given. None for any other
    challenge, or when the realm refuses; a ValueError when it answers 200 without a token, as a sign-in page does."""
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
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict) or not (token := body.get("token") or body.get("access_token")):
        raise ValueError(f"{what}: the token service at {realm} answered without a token")
    return token
