"""The CI computer image mirror's two scripts: what a runner shares into it, and which of its versions a prune deletes."""

from __future__ import annotations

import datetime
import io
import json
import sys

import pytest

from tst.util.scripts import load_script

NOW = datetime.datetime(2026, 11, 20, 12, tzinfo=datetime.timezone.utc)
CUTOFF = NOW - datetime.timedelta(days=7)
KEY, OTHER = "c" * 16 + "-" + "a" * 64, "c" * 16 + "-" + "b" * 64
RAW, RAW_OTHER = "a" * 64, "b" * 64


def _version(name: str, created: str, *tags: str) -> dict:
    return {"id": hash(name) % 10_000, "name": name, "created_at": created, "metadata": {"container": {"tags": list(tags)}}}


@pytest.fixture
def prune():
    return load_script("prune_computer_images.py")


def test_an_image_last_seen_within_the_window_is_kept_and_one_last_seen_before_it_is_deleted(prune):
    used = _version("sha256:1", "2026-10-01T00:00:00Z", KEY, f"{KEY}-seen-20261001", f"{KEY}-seen-20261115")
    unused = _version("sha256:2", "2026-10-01T00:00:00Z", OTHER, f"{OTHER}-seen-20261112")

    assert prune.stale_versions([used, unused], CUTOFF, lambda digest: []) == [unused]


def test_an_image_seen_on_the_cutoffs_day_is_kept_through_that_day(prune):
    edge = _version("sha256:1", "2026-10-01T00:00:00Z", KEY, f"{KEY}-seen-20261113")

    assert prune.stale_versions([edge], CUTOFF, lambda digest: []) == []


def test_a_version_uploaded_within_the_window_is_kept_though_no_run_has_tagged_it_seen(prune):
    fresh = _version("sha256:1", "2026-11-19T00:00:00Z")
    old = _version("sha256:2", "2026-11-01T00:00:00Z")

    assert prune.stale_versions([fresh, old], CUTOFF, lambda digest: []) == [old]


def test_the_platform_manifests_a_kept_index_lists_are_kept_and_a_deleted_ones_are_not(prune):
    kept = _version("sha256:k", "2026-10-01T00:00:00Z", KEY, f"{KEY}-seen-20261119")
    dropped = _version("sha256:d", "2026-10-01T00:00:00Z", OTHER, f"{OTHER}-seen-20261001")
    children = {"sha256:k": ["sha256:k-amd64", "sha256:k-attestation"], "sha256:d": ["sha256:d-amd64"]}
    untagged = [_version(digest, "2026-10-01T00:00:00Z") for digest in ("sha256:k-amd64", "sha256:k-attestation",
                                                                         "sha256:d-amd64")]

    stale = prune.stale_versions([kept, dropped, *untagged], CUTOFF, lambda digest: children.get(digest, []))

    assert [version["name"] for version in stale] == ["sha256:d", "sha256:d-amd64"]


def test_a_prune_without_the_package_deletes_nothing_passes_and_says_why(prune, monkeypatch, capsys):
    def missing(method, url):
        raise prune.urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(prune, "_github", missing)
    monkeypatch.setattr(sys, "argv", ["prune_computer_images.py", "org", "agentenv-ci-computers"])

    assert prune.main() == 0
    assert "no package org/agentenv-ci-computers, or this repository has no access to it" in capsys.readouterr().out


def test_a_runner_shares_each_image_under_its_mirror_tag_and_todays_seen_tag_and_leaves_its_indexes(monkeypatch,
                                                                                                    capsys):
    share = load_script("share_computer_images.py")
    tags = {"tags": [RAW, f"index-{RAW_OTHER}", RAW_OTHER]}
    copies = []
    monkeypatch.setattr(share.urllib.request, "urlopen", lambda url: io.BytesIO(json.dumps(tags).encode()))
    monkeypatch.setattr(share, "pin_digest", lambda ref, auth: f"{ref}@sha256:{ref[-64:]}")
    monkeypatch.setattr(share, "mirror_tag", lambda key: f"{'c' * 16}-{key}")
    monkeypatch.setattr(share, "copy_image", lambda source, target, source_auth, auth: copies.append(
        (source.rpartition("@")[2], target, auth.username)))
    monkeypatch.setattr(share, "pin_if_served", lambda ref, auth: f"{ref}@sha256:{RAW}")
    monkeypatch.setattr(share.datetime, "datetime", type("Fixed", (datetime.datetime,), {"now": staticmethod(lambda tz: NOW)}))
    monkeypatch.setenv("GITHUB_ACTOR", "bot")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(sys, "argv", ["share_computer_images.py", "ghcr.io/org/agentenv-ci-computers"])

    assert share.main() == 0
    assert copies == [(f"sha256:{raw}", f"ghcr.io/org/agentenv-ci-computers:{tag}", "bot")
                      for raw, key in ((RAW, KEY), (RAW_OTHER, OTHER)) for tag in (key, f"{key}-seen-20261120")]
    assert "::warning::" not in capsys.readouterr().out


def test_a_mirror_the_composer_cant_read_anonymously_warns_to_make_it_public(monkeypatch, capsys):
    share = load_script("share_computer_images.py")
    monkeypatch.setattr(share.urllib.request, "urlopen", lambda url: io.BytesIO(json.dumps({"tags": [RAW]}).encode()))
    monkeypatch.setattr(share, "pin_digest", lambda ref, auth: f"{ref}@sha256:{RAW}")
    monkeypatch.setattr(share, "copy_image", lambda *args: None)

    def private(ref, auth):
        raise ValueError(f"{ref}: ghcr.io refused to serve it (HTTP 403)")

    monkeypatch.setattr(share, "pin_if_served", private)
    monkeypatch.setenv("GITHUB_ACTOR", "bot")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(sys, "argv", ["share_computer_images.py", "ghcr.io/org/agentenv-ci-computers"])

    assert share.main() == 0
    assert "make the package public" in capsys.readouterr().out
