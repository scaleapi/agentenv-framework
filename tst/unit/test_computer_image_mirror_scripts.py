from __future__ import annotations

import io
import json
import sys

import pytest

from tst.util.scripts import load_script

KEY, OTHER = "c" * 16 + "-" + "a" * 64, "c" * 16 + "-" + "b" * 64
RAW, RAW_OTHER = "a" * 64, "b" * 64


def test_a_runner_shares_each_image_under_its_mirror_tag_and_leaves_its_indexes(monkeypatch, capsys):
    share = load_script("share_computer_images.py")
    tags = {"tags": [RAW, f"index-{RAW_OTHER}", RAW_OTHER]}
    copies = []
    monkeypatch.setattr(share.urllib.request, "urlopen", lambda url: io.BytesIO(json.dumps(tags).encode()))
    monkeypatch.setattr(share, "pin_digest", lambda ref, auth: f"{ref}@sha256:{ref[-64:]}")
    monkeypatch.setattr(share, "mirror_tag", lambda key: f"{'c' * 16}-{key}")
    monkeypatch.setattr(share, "copy_image", lambda source, target, source_auth, auth: copies.append(
        (source.rpartition("@")[2], target, auth.username)))
    monkeypatch.setattr(share, "pin_if_served", lambda ref, auth: f"{ref}@sha256:{RAW}")
    monkeypatch.setenv("GITHUB_ACTOR", "bot")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(sys, "argv", ["share_computer_images.py", "ghcr.io/org/agentenv-ci-computers"])

    assert share.main() == 0
    assert copies == [(f"sha256:{raw}", f"ghcr.io/org/agentenv-ci-computers:{key}", "bot")
                      for raw, key in ((RAW, KEY), (RAW_OTHER, OTHER))]
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


def test_a_runner_that_composed_no_image_shares_nothing_and_passes(monkeypatch, capsys):
    share = load_script("share_computer_images.py")

    def no_repository(url):
        raise share.urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(share.urllib.request, "urlopen", no_repository)
    monkeypatch.setattr(share, "copy_image", lambda *args: pytest.fail("copied with no image to share"))
    monkeypatch.setenv("GITHUB_ACTOR", "bot")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(sys, "argv", ["share_computer_images.py", "ghcr.io/org/agentenv-ci-computers"])

    assert share.main() == 0
    assert "0 images" in capsys.readouterr().out
