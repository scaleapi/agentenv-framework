"""``write_untagged``: a ``docker save`` archive copied naming no image, its layers as they are."""

import gzip
import io
import json
import tarfile

import pytest

from agent_env.utils.docker.untag import write_untagged
from tst.util.docker_save import saved_image, tar_bytes

LAYER = {"app/hello.txt": "hi\n"}
TAGS = ["acme/crm:1"]


def _untagged(data: bytes, **kwargs) -> bytes:
    out = io.BytesIO()
    write_untagged(io.BytesIO(data), out, **kwargs)
    return out.getvalue()


def _members(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isreg()}


@pytest.mark.parametrize("oci", [False, True], ids=["legacy", "oci"])
@pytest.mark.parametrize("gz", [False, True], ids=["tar", "gzipped"])
def test_an_archive_is_copied_naming_no_image_with_its_layers_as_they_are(oci, gz):
    saved = saved_image([LAYER], oci=oci, gz=gz, tags=TAGS)
    before = _members(gzip.decompress(saved) if gz else saved)

    after = _members(_untagged(saved))

    assert [entry["RepoTags"] for entry in json.loads(after.pop("manifest.json"))] == [None]
    if oci:
        (descriptor,) = json.loads(after.pop("index.json"))["manifests"]
        assert descriptor["annotations"] == {"org.example.kept": "yes"}
    assert "repositories" not in after
    assert after == {name: data for name, data in before.items()
                     if name not in ("manifest.json", "index.json", "repositories")}


def test_an_archive_naming_no_image_is_copied_naming_none_and_gzipped_when_asked():
    saved = saved_image([LAYER], oci=True)

    out = _untagged(saved, gzip_level=1)

    assert out[:2] == b"\x1f\x8b"
    assert json.loads(_members(gzip.decompress(out))["manifest.json"])[0]["RepoTags"] is None


@pytest.mark.parametrize("data, problem", [
    (tar_bytes({"hello.txt": "hi"}), "it holds no manifest.json or index.json"),
    (b"not a tar at all" * 64, "it isn't a tar docker save writes"),
    (tar_bytes({"manifest.json": "{"}), "its manifest.json isn't JSON"),
    (tar_bytes({"manifest.json": '{"Config": "c"}'}), "its manifest.json isn't a list of images"),
], ids=["no manifest", "not a tar", "bad json", "not a list"])
def test_what_docker_save_didnt_write_is_refused(data, problem):
    with pytest.raises(ValueError, match=problem):
        _untagged(data)
