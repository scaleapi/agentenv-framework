"""``installed_packages``: the Python and global npm packages a ``docker save`` tarball holds."""

import gzip
import hashlib
import io
import json
import tarfile

from agent_env.utils.docker.installed import Installed, installed_packages

SITE = "usr/local/lib/python3.12/site-packages"
NODE = "usr/local/lib/node_modules"


def _metadata(name: str, version: str) -> str:
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\nA long description.\n"


def _package_json(name: str, version: str) -> str:
    return json.dumps({"name": name, "version": version})


def _tar(files: dict[str, str | bytes], links: dict[str, str] | None = None, gz: bool = False) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz" if gz else "w") as tar:
        for name, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
    return out.getvalue()


def _saved(layers: list[dict[str, str]], oci: bool = False, gz_layers: bool = False, gz: bool = False,
           manifest_first: bool = False) -> io.BytesIO:
    """A ``docker save`` tarball of ``layers``, bottom first, in the legacy or the OCI layout."""
    blobs = [_tar(files, gz=gz_layers) for files in layers]
    names = ([f"blobs/sha256/{hashlib.sha256(blob).hexdigest()}" for blob in blobs] if oci
             else [f"{index:064x}/layer.tar" for index in range(len(blobs))])
    config = json.dumps({"rootfs": {"type": "layers"}}).encode()
    files = {name: blob for name, blob in zip(names, blobs)}
    files["blobs/sha256/config" if oci else "config.json"] = config
    manifest = {"manifest.json": json.dumps([{"Config": "config.json", "Layers": names}])}
    extra = {"oci-layout": '{"imageLayoutVersion": "1.0.0"}', "index.json": "{}"} if oci else {}
    data = _tar({**manifest, **files, **extra} if manifest_first else {**files, **extra, **manifest})
    return io.BytesIO(gzip.compress(data) if gz else data)


BASE = {f"{SITE}/pip-24.0.dist-info/METADATA": _metadata("pip", "24.0"),
        f"{SITE}/requests-2.31.0.dist-info/METADATA": _metadata("requests", "2.31.0"),
        f"{SITE}/Foo_Bar-1.0.dist-info/METADATA": _metadata("Foo_Bar", "1.0"),
        f"{SITE}/requests/__init__.py": "",
        f"{NODE}/typescript/package.json": _package_json("typescript", "5.4.5"),
        f"{NODE}/@anthropic-ai/claude-code/package.json": _package_json("@anthropic-ai/claude-code", "1.0.3"),
        f"{NODE}/@anthropic-ai/claude-code/node_modules/chalk/package.json": _package_json("chalk", "5.3.0"),
        f"{NODE}/npm/node_modules/semver/package.json": _package_json("semver", "7.6.0"),
        f"{NODE}/npm/package.json": _package_json("npm", "10.5.0")}


def test_legacy_save_names_each_python_distribution_and_global_npm_package():
    assert installed_packages(_saved([BASE])) == Installed(
        python={"foo-bar": ("1.0",), "pip": ("24.0",), "requests": ("2.31.0",)},
        npm={"@anthropic-ai/claude-code": "1.0.3", "npm": "10.5.0", "typescript": "5.4.5"})


def test_oci_save_with_gzipped_layers_and_gzipped_tarball_reads_the_same():
    expected = installed_packages(_saved([BASE]))
    assert installed_packages(_saved([BASE], oci=True, gz_layers=True, gz=True)) == expected
    assert installed_packages(_saved([BASE], oci=True, manifest_first=True)) == expected


def test_dist_packages_egg_info_and_leading_dot_slash_are_read():
    found = installed_packages(_saved([{
        "./usr/lib/python3/dist-packages/six-1.16.0.egg-info/PKG-INFO": _metadata("six", "1.16.0"),
        "/opt/venv/lib/python3.12/site-packages/attrs-23.2.0.dist-info/METADATA": _metadata("attrs", "23.2.0")}]))
    assert found.python == {"attrs": ("23.2.0",), "six": ("1.16.0",)}


def test_an_upgrade_in_a_later_layer_whites_out_the_lower_version():
    upgrade = {f"{SITE}/.wh.requests-2.31.0.dist-info": "",
               f"{SITE}/requests-2.32.3.dist-info/METADATA": _metadata("requests", "2.32.3"),
               f"{NODE}/.wh.typescript": ""}
    found = installed_packages(_saved([BASE, upgrade]))
    assert found.python["requests"] == ("2.32.3",)
    assert "typescript" not in found.npm


def test_an_opaque_folder_hides_what_lower_layers_held_in_it():
    found = installed_packages(_saved([BASE, {f"{SITE}/.wh..wh..opq": "",
                                              f"{SITE}/attrs-23.2.0.dist-info/METADATA": _metadata("attrs", "23.2.0")}]))
    assert found.python == {"attrs": ("23.2.0",)}
    assert found.npm == installed_packages(_saved([BASE])).npm


def test_a_distribution_in_two_environments_at_two_versions_names_both():
    found = installed_packages(_saved([BASE, {
        "opt/venv/lib/python3.12/site-packages/requests-2.28.0.dist-info/METADATA": _metadata("requests", "2.28.0")}]))
    assert found.python["requests"] == ("2.28.0", "2.31.0")


def test_a_distribution_installed_from_a_path_or_url_is_direct_until_reinstalled_from_an_index():
    local = {f"{SITE}/myapp-0.1.dev3.dist-info/METADATA": _metadata("myapp", "0.1.dev3"),
             f"{SITE}/myapp-0.1.dev3.dist-info/direct_url.json": '{"url": "file:///app", "dir_info": {}}'}
    assert installed_packages(_saved([BASE, local])).direct == frozenset({"myapp"})
    reinstalled = {f"{SITE}/myapp-0.1.dev3.dist-info/.wh..wh..opq": "",
                   f"{SITE}/myapp-0.1.dev3.dist-info/METADATA": _metadata("myapp", "0.1.dev3")}
    assert installed_packages(_saved([BASE, local, reinstalled])).direct == frozenset()


def test_a_layer_saved_twice_as_a_link_is_followed():
    blob = _tar(BASE)
    data = _tar({"a/layer.tar": blob, "manifest.json": json.dumps([{"Layers": ["a/layer.tar", "b/layer.tar"]}])},
                links={"b/layer.tar": "../a/layer.tar"})
    assert installed_packages(io.BytesIO(data)) == installed_packages(_saved([BASE]))


def test_no_manifest_or_an_unreadable_named_layer_is_none():
    assert installed_packages(io.BytesIO(_tar({"0/layer.tar": _tar(BASE)}))) is None
    assert installed_packages(io.BytesIO(_tar({"manifest.json": "[]", "0/layer.tar": _tar(BASE)}))) is None
    unreadable = _tar({"0/layer.tar": b"not a tar" * 100, "manifest.json": json.dumps([{"Layers": ["0/layer.tar"]}])})
    assert installed_packages(io.BytesIO(unreadable)) is None
    missing = _tar({"manifest.json": json.dumps([{"Layers": ["0/layer.tar"]}])})
    assert installed_packages(io.BytesIO(missing)) is None


def test_metadata_without_a_name_or_version_and_bad_package_json_are_skipped():
    found = installed_packages(_saved([{f"{SITE}/broken-1.dist-info/METADATA": "Metadata-Version: 2.1\n\n",
                                        f"{NODE}/bad/package.json": "{not json",
                                        f"{NODE}/odd/package.json": '{"name": "odd", "version": 3}'}]))
    assert found == Installed(python={}, npm={})
