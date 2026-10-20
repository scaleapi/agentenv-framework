"""``installed_packages``: the Python and global npm packages a ``docker save`` tarball holds."""

import io
import json

from agent_env.utils.docker.installed import Installed, installed_packages
from tst.util.docker_save import saved_image, tar_bytes

SITE = "usr/local/lib/python3.12/site-packages"
NODE = "usr/local/lib/node_modules"


def _metadata(name: str, version: str) -> str:
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\nA long description.\n"


def _package_json(name: str, version: str) -> str:
    return json.dumps({"name": name, "version": version})


def _saved(layers, **kwargs) -> io.BytesIO:
    return io.BytesIO(saved_image(layers, **kwargs))


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
    blob = tar_bytes(BASE)
    data = tar_bytes({"a/layer.tar": blob, "manifest.json": json.dumps([{"Layers": ["a/layer.tar", "b/layer.tar"]}])},
                links={"b/layer.tar": "../a/layer.tar"})
    assert installed_packages(io.BytesIO(data)) == installed_packages(_saved([BASE]))


def test_no_manifest_or_an_unreadable_named_layer_is_none():
    assert installed_packages(io.BytesIO(tar_bytes({"0/layer.tar": tar_bytes(BASE)}))) is None
    assert installed_packages(io.BytesIO(tar_bytes({"manifest.json": "[]", "0/layer.tar": tar_bytes(BASE)}))) is None
    unreadable = tar_bytes({"0/layer.tar": b"not a tar" * 100, "manifest.json": json.dumps([{"Layers": ["0/layer.tar"]}])})
    assert installed_packages(io.BytesIO(unreadable)) is None
    missing = tar_bytes({"manifest.json": json.dumps([{"Layers": ["0/layer.tar"]}])})
    assert installed_packages(io.BytesIO(missing)) is None


def test_metadata_without_a_name_or_version_and_bad_package_json_are_skipped():
    found = installed_packages(_saved([{f"{SITE}/broken-1.dist-info/METADATA": "Metadata-Version: 2.1\n\n",
                                        f"{NODE}/bad/package.json": "{not json",
                                        f"{NODE}/odd/package.json": '{"name": "odd", "version": 3}'}]))
    assert found == Installed(python={}, npm={})
