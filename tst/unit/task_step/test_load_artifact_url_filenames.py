"""LoadArtifactTaskStep `urls` entries: an object entry ``{"url", "filename"}`` is saved under its
filename verbatim, survives a round-trip, and a bad or colliding entry fails at construction."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep, _url_downloads

OPAQUE_URL = "https://files.example/objects/obj-4f9c2a"


def _step(urls) -> LoadArtifactTaskStep:
    return LoadArtifactTaskStep(id="load", version=None, sandbox_name="box", urls=urls)


def test_object_entries_survive_a_round_trip():
    urls = ["https://a.example/data.csv", {"url": OPAQUE_URL, "filename": "report.txt"}]
    data = _step(urls).to_dict()
    assert data["urls"] == urls
    assert LoadArtifactTaskStep.from_dict(data).urls == urls


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"url": OPAQUE_URL}, "non-empty string 'filename'"),
        ({"url": OPAQUE_URL, "name": "a.txt"}, r"unknown key\(s\) \['name'\]"),
        ({"url": OPAQUE_URL, "filename": "../a.txt"}, r"must not contain '\.\.' segments"),
        ({"url": OPAQUE_URL, "filename": "/etc/passwd"}, "must be relative"),
        ({"url": OPAQUE_URL, "filename": "sub/a.txt"}, "plain file name, not a path"),
    ],
)
def test_a_bad_object_entry_fails_at_construction(entry, match):
    with pytest.raises(ValueError, match=match):
        _step([entry])


def test_a_filename_given_twice_fails():
    with pytest.raises(ValueError, match="'a.txt' is given to more than one urls entry"):
        _step([{"url": OPAQUE_URL, "filename": "a.txt"}, {"url": OPAQUE_URL + "2", "filename": "a.txt"}])


def test_a_bare_url_named_like_an_explicit_filename_fails():
    with pytest.raises(ValueError, match=r"'https://a.example/data.csv' would be saved as 'data.csv'"):
        _step(["https://a.example/data.csv", {"url": OPAQUE_URL, "filename": "data.csv"}])


@pytest.mark.parametrize(
    "url, name",
    [
        ("https://a.example/x/%2e%2e%2fetc%2fpasswd", "passwd"),
        ("https://storage.example/b/bucket/o/path%2Fto%2Ffile.csv?alt=media", "file.csv"),
        ("https://a.example/x/dir%2F", "downloaded"),
    ],
)
def test_an_escaped_slash_in_a_bare_url_separates_like_a_slash(url, name):
    assert _url_downloads("load", [url]) == [(url, name)]


@pytest.mark.parametrize("url", ["https://a.example/x/%2e%2e", "https://a.example/x/..", "https://a.example/x/..%5Cevil"])
def test_a_bare_url_whose_name_is_no_plain_file_name_fails_at_construction(url):
    with pytest.raises(ValueError, match="give it a 'filename'"):
        _step([url])


def test_a_repeated_bare_url_name_skips_suffixes_already_taken():
    urls = ["https://a.example/x-1.txt", "https://b.example/x.txt", "https://c.example/x.txt"]
    assert [name for _, name in _url_downloads("load", urls)] == ["x-1.txt", "x.txt", "x-2.txt"]
