"""The wheel check in ``.github/scripts/check_explorer_ui.py``: the explorer's index.html and every ``/_next/`` asset
it references must be in the built wheel."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_explorer_ui.py"
_spec = importlib.util.spec_from_file_location("check_explorer_ui", SCRIPT)
check_explorer_ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_explorer_ui)

STATIC = "agent_env/explorer/static/"
INDEX = (
    '<html><head><link rel="stylesheet" href="/_next/static/css/app.css"/>'
    '<link rel="preload" href="/_next/static/media/font.woff2"/><link rel="icon" href="/favicon.ico"/></head>'
    '<body><a href="/docs">Docs</a><script src="/_next/static/chunks/main.js?dpl=1"></script></body></html>'
)
ASSETS = ["_next/static/css/app.css", "_next/static/media/font.woff2", "_next/static/chunks/main.js"]


def _wheel(tmp_path: Path, files: dict[str, str]) -> Path:
    wheel = tmp_path / "agentenv_framework-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return wheel


def test_a_wheel_with_index_html_and_every_asset_it_references_passes(tmp_path):
    wheel = _wheel(tmp_path, {STATIC + "index.html": INDEX, **{STATIC + asset: "x" for asset in ASSETS}})

    assert check_explorer_ui.ui_problems(wheel) == []


def test_each_referenced_asset_the_wheel_leaves_out_is_named(tmp_path):
    wheel = _wheel(tmp_path, {STATIC + "index.html": INDEX, STATIC + ASSETS[0]: "x"})

    assert check_explorer_ui.ui_problems(wheel) == [
        f"{STATIC}_next/static/chunks/main.js is missing",
        f"{STATIC}_next/static/media/font.woff2 is missing",
    ]


def test_a_percent_encoded_reference_matches_the_file_stored_under_its_literal_name(tmp_path):
    index = '<script src="/_next/static/chunks/pages/%5B%5B...slug%5D%5D-abc.js"></script>'
    wheel = _wheel(tmp_path, {STATIC + "index.html": index, STATIC + "_next/static/chunks/pages/[[...slug]]-abc.js": "x"})

    assert check_explorer_ui.ui_problems(wheel) == []


def test_a_wheel_without_index_html_fails(tmp_path):
    wheel = _wheel(tmp_path, {"agent_env/__init__.py": ""})

    assert check_explorer_ui.ui_problems(wheel) == [f"{STATIC}index.html is missing"]


def test_an_index_html_that_references_no_built_assets_fails(tmp_path):
    wheel = _wheel(tmp_path, {STATIC + "index.html": "<html><body>placeholder</body></html>"})

    assert check_explorer_ui.ui_problems(wheel) == [f"{STATIC}index.html references no /_next/ assets"]
