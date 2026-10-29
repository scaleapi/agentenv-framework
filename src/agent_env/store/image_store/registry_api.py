"""Read an image from its registry over the OCI distribution API: its digest, to pin a tag to the image it names now,
and its manifests and blobs, to inspect what it holds without pulling it."""

from __future__ import annotations

import contextlib
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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


class NoSuchImageError(ValueError):
    """The registry has no image under the reference: it answered 404."""


class RegistryRefusedError(ValueError):
    """The registry refused to serve the reference (401 or 403), as GHCR does for a private package, or one that doesn't
    exist yet, asked without credentials that reach it."""


@dataclass(frozen=True)
class _ImageRef:
    name: str  # the reference without its tag or digest, spelled as given
    host: str
    repository: str
    tag: str | None
    digest: str | None


def pin_digest(ref: str, auth: RegistryAuth | None, *, timeout: float = _TIMEOUT_SECONDS) -> str:
    """``ref`` pinned to the digest its registry serves for it now, as ``name:tag@sha256:...``.

    A tag (``latest`` when none is given) is resolved; a digest is checked to exist; a ref giving both must agree,
    since a pull goes by the digest and ignores the tag. ``auth`` is used where the registry asks for credentials;
    without it, a registry's anonymous token is. Each request waits ``timeout`` seconds at most. Raises ValueError naming
    what the registry answered.
    """
    image = _parse(ref)
    if image.digest is None:
        tag = image.tag or "latest"
        return f"{image.name}:{tag}@{_check(image, tag, auth, timeout)}"
    served = _check(image, image.tag or image.digest, auth, timeout)
    if image.tag and served != image.digest:
        raise ValueError(f"{ref}: the tag {image.tag!r} names {served} now, not {image.digest}; give the tag or the "
                         "digest alone")
    return ref


def pinned_from(pinned: str, ref: str) -> bool:
    """Whether ``pinned``, a ref as ``pin_digest`` returns it, pins ``ref``: ``ref`` itself when it gives a digest, else
    the same name and tag (``latest`` when none is given) with a digest."""
    if "@" in ref:
        return pinned == ref
    image, given = _parse(pinned), _parse(ref)
    return image.digest is not None and (image.name, image.tag) == (given.name, given.tag or "latest")


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


def _check(image: _ImageRef, reference: str, auth: RegistryAuth | None, timeout: float) -> str:
    """The digest the registry serves for ``reference`` (a tag or a digest) in ``image``'s repository."""
    what = f"{image.name}{'@' if _DIGEST.match(reference) else ':'}{reference}"
    path = f"manifests/{reference}"
    with _Repository(image, auth, timeout, what) as repository:
        served = repository.head(path, accept=_MANIFEST_TYPES).headers.get("docker-content-digest")
        if not served:
            return f"sha256:{hashlib.sha256(repository.get(path, accept=_MANIFEST_TYPES).content).hexdigest()}"
    if not _DIGEST.match(served):
        raise ValueError(f"{what}: {image.host} answered with {served!r}, which isn't a digest")
    return served


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


def read_manifest(ref: str, auth: RegistryAuth | None, *, timeout: float = _TIMEOUT_SECONDS) -> dict:
    """The manifest the registry serves for ``ref``, by its digest when it gives one, else its tag: an index of one
    manifest per platform, or a single image's manifest, as its ``mediaType`` says. Raises as ``pin_digest`` does."""
    image = _parse(ref)
    with _Repository(image, auth, timeout, ref) as repository:
        response = repository.get(f"manifests/{image.digest or image.tag or 'latest'}", accept=_MANIFEST_TYPES)
        try:
            return response.json()
        except ValueError as e:
            raise ValueError(f"{ref}: {image.host} answered with a manifest that isn't JSON") from e


def read_blob(ref: str, digest: str, auth: RegistryAuth | None, path: Path, *,
              timeout: float = _TIMEOUT_SECONDS) -> None:
    """Download the blob ``digest`` from ``ref``'s repository to ``path``, refusing one whose bytes don't hash to it."""
    image = _parse(ref)
    if not digest.startswith("sha256:") or not _DIGEST.match(digest):
        raise ValueError(f"{ref}: {digest!r} isn't a sha256 digest")
    with _Repository(image, auth, timeout, ref) as repository:
        hashed = hashlib.sha256()
        try:
            with repository.stream(f"blobs/{digest}") as response, path.open("wb") as out:
                for chunk in response.iter_bytes():
                    hashed.update(chunk)
                    out.write(chunk)
        except httpx.HTTPError as e:
            raise ValueError(f"{ref}: reading the blob {digest} from {image.host} failed: {type(e).__name__}: {e}") from e
    if f"sha256:{hashed.hexdigest()}" != digest:
        path.unlink(missing_ok=True)
        raise ValueError(f"{ref}: the blob {image.host} served for {digest} hashes to sha256:{hashed.hexdigest()}")


class _Repository:
    """One image's repository on its registry, read with the Bearer token the registry's first challenge asks for."""

    def __init__(self, image: _ImageRef, auth: RegistryAuth | None, timeout: float, what: str) -> None:
        # A registry on this machine serves plain HTTP, as Docker allows for a loopback one.
        scheme = "http" if is_loopback_host(image.host) else "https"
        self._base = f"{scheme}://{image.host}/v2/{image.repository}"
        self._basic = (auth.username, auth.password) if auth is not None else None
        self._token: str | None = None
        self._host, self._what = image.host, what
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def __enter__(self) -> _Repository:
        return self

    def __exit__(self, *_exc) -> None:
        self._client.close()

    def head(self, path: str, *, accept: str | None = None) -> httpx.Response:
        response = self._send("HEAD", path, accept, stream=False)
        self._raise_for(response)
        return response

    def get(self, path: str, *, accept: str | None = None) -> httpx.Response:
        response = self._send("GET", path, accept, stream=False)
        self._raise_for(response)
        return response

    @contextlib.contextmanager
    def stream(self, path: str) -> Iterator[httpx.Response]:
        """``path`` as a streamed response, closed when the block ends."""
        response = self._send("GET", path, None, stream=True)
        try:
            if response.status_code != 200:
                response.read()
                self._raise_for(response)
            yield response
        finally:
            response.close()

    def _send(self, method: str, path: str, accept: str | None, *, stream: bool) -> httpx.Response:
        try:
            for _ in range(2):
                headers = {"Accept": accept} if accept else {}
                if self._token:
                    headers["Authorization"] = f"Bearer {self._token}"
                request = self._client.build_request(method, f"{self._base}/{path}", headers=headers)
                # Basic credentials go with every request until a Bearer token replaces them.
                response = self._client.send(request, auth=None if self._token else self._basic, stream=stream)
                if response.status_code != 401 or self._token:
                    return response
                if stream:
                    response.read()
                    response.close()
                if (token := _bearer_token(self._client, response, self._basic, self._what)) is None:
                    return response
                self._token = token
            return response
        except httpx.HTTPError as e:
            thing = "its manifest" if path.startswith("manifests/") else f"the blob {path.removeprefix('blobs/')}"
            raise ValueError(f"{self._what}: couldn't read {thing} from {self._host}: {type(e).__name__}: {e}") from e

    def _raise_for(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        if response.status_code == 404:
            raise NoSuchImageError(f"{self._what}: {self._host} has no such image")
        if response.status_code in (401, 403):
            raise RegistryRefusedError(f"{self._what}: {self._host} refused to serve it (HTTP {response.status_code}); "
                                       "agent-env reaches a private registry only with the configured image store's "
                                       "credentials, for its own registry")
        raise ValueError(f"{self._what}: {self._host} answered HTTP {response.status_code}")
