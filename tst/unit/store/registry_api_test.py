"""Pinning an image ref to the digest its registry serves, over a fake registry's OCI distribution API."""

import base64
import hashlib

import httpx
import pytest

from agent_env.store import RegistryAuth
from agent_env.store.image_store import registry_api
from agent_env.store.image_store.registry_api import pin_digest

DIGEST = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


@pytest.fixture
def requests(monkeypatch):
    """Route the resolver's HTTP client to ``handler`` and record each request it makes."""
    seen: list[httpx.Request] = []
    real = httpx.Client

    def route(handler):
        def recorded(request):
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(registry_api.httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(recorded),
                                                                               **kwargs))
        return seen

    return route


def _served(digest=DIGEST):
    return httpx.Response(200, headers={"Docker-Content-Digest": digest})


def test_a_tag_is_pinned_to_its_digest_through_the_registrys_anonymous_token(requests):
    def ghcr(request):
        if request.url.path == "/token":
            assert request.url.params["scope"] == "repository:org/tool:pull"
            return httpx.Response(200, json={"token": "anon"})
        if request.headers.get("authorization") == "Bearer anon":
            return _served()
        return httpx.Response(401, headers={"WWW-Authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io",'
                                                                  'scope="repository:org/tool:pull"'})

    seen = requests(ghcr)

    assert pin_digest("ghcr.io/org/tool:v1", None) == f"ghcr.io/org/tool:v1@{DIGEST}"
    assert [r.method for r in seen] == ["HEAD", "GET", "HEAD"]
    assert seen[0].url == "https://ghcr.io/v2/org/tool/manifests/v1"
    assert "application/vnd.oci.image.index.v1+json" in seen[0].headers["accept"]


def test_the_image_stores_credentials_go_to_a_registry_that_takes_them_directly(requests):
    def ecr(request):
        expected = "Basic " + base64.b64encode(b"AWS:token").decode()
        return _served() if request.headers.get("authorization") == expected else httpx.Response(401)

    requests(ecr)

    ref = "123456789012.dkr.ecr.us-west-2.amazonaws.com/team/app:v2"
    assert pin_digest(ref, RegistryAuth("123456789012.dkr.ecr.us-west-2.amazonaws.com", "AWS", "token")) == f"{ref}@{DIGEST}"


def test_no_tag_means_latest_and_docker_hub_names_its_library(requests):
    seen = requests(lambda request: _served())

    assert pin_digest("docker.io/alpine", None) == f"docker.io/alpine:latest@{DIGEST}"
    assert seen[0].url == "https://registry-1.docker.io/v2/library/alpine/manifests/latest"


def test_a_digest_is_kept_once_the_registry_serves_it(requests):
    seen = requests(lambda request: _served())

    assert pin_digest(f"ghcr.io/org/tool@{DIGEST}", None) == f"ghcr.io/org/tool@{DIGEST}"
    assert seen[0].url == f"https://ghcr.io/v2/org/tool/manifests/{DIGEST}"


def test_a_tag_and_digest_must_agree_since_a_pull_goes_by_the_digest(requests):
    requests(lambda request: _served(DIGEST if request.url.path.endswith(DIGEST) else OTHER))

    with pytest.raises(ValueError, match=f"the tag 'v1' names {OTHER} now, not {DIGEST}"):
        pin_digest(f"ghcr.io/org/tool:v1@{DIGEST}", None)


def test_a_registry_on_this_machine_is_read_over_plain_http(requests):
    seen = requests(lambda request: _served())

    assert pin_digest("localhost:5000/team/img:v1", None) == f"localhost:5000/team/img:v1@{DIGEST}"
    assert seen[0].url == "http://localhost:5000/v2/team/img/manifests/v1"


def test_a_registry_that_omits_the_digest_header_is_hashed_from_the_manifest(requests):
    body = b'{"schemaVersion": 2}'
    requests(lambda request: httpx.Response(200, content=body if request.method == "GET" else b""))

    assert pin_digest("ghcr.io/org/tool:v1", None) == f"ghcr.io/org/tool:v1@sha256:{hashlib.sha256(body).hexdigest()}"


def test_the_manifest_is_fetched_with_the_credentials_the_registry_took(requests):
    body = b'{"schemaVersion": 2}'
    expected = "Basic " + base64.b64encode(b"AWS:token").decode()

    def private(request):
        if request.headers.get("authorization") != expected:
            return httpx.Response(401)
        return httpx.Response(200, content=body if request.method == "GET" else b"")

    seen = requests(private)

    ref = "123456789012.dkr.ecr.us-west-2.amazonaws.com/team/app:v2"
    auth = RegistryAuth("123456789012.dkr.ecr.us-west-2.amazonaws.com", "AWS", "token")
    assert pin_digest(ref, auth) == f"{ref}@sha256:{hashlib.sha256(body).hexdigest()}"
    assert [r.method for r in seen] == ["HEAD", "GET"]


@pytest.mark.parametrize("header", ["sha256:xyz", "sha256:" + "a" * 32, "sha256:" + "A" * 64])
def test_a_digest_header_docker_couldnt_pull_by_is_refused(requests, header):
    requests(lambda request: _served(header))

    with pytest.raises(ValueError, match=f"ghcr.io answered with '{header}', which isn't a digest"):
        pin_digest("ghcr.io/org/tool:v1", None)


@pytest.mark.parametrize("status, message", [
    (404, "ghcr.io/org/tool:v9: ghcr.io has no such image"),
    (401, "ghcr.io/org/tool:v9: ghcr.io refused to serve it \\(HTTP 401\\); agent-env reaches a private registry only"),
    (500, "ghcr.io/org/tool:v9: ghcr.io answered HTTP 500"),
])
def test_what_the_registry_answers_is_named(requests, status, message):
    requests(lambda request: httpx.Response(status))

    with pytest.raises(ValueError, match=message):
        pin_digest("ghcr.io/org/tool:v9", None)


def test_an_unreachable_registry_is_named(requests):
    def down(request):
        raise httpx.ConnectError("connection refused")

    requests(down)

    with pytest.raises(ValueError, match="ghcr.io/org/tool:v1: couldn't read its manifest from ghcr.io: ConnectError"):
        pin_digest("ghcr.io/org/tool:v1", None)


@pytest.mark.parametrize("digest", ["sha256:xyz", "sha256:" + "a" * 32, "sha256:" + "A" * 64, "md5:" + "a" * 32])
def test_a_digest_docker_couldnt_pull_by_is_refused_before_any_request(requests, digest):
    seen = requests(lambda request: _served())

    with pytest.raises(ValueError, match=f"'{digest}' isn't a sha256, sha384 or sha512 digest"):
        pin_digest(f"ghcr.io/org/tool@{digest}", None)
    assert seen == []
