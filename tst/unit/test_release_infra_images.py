"""What ``.github/workflows/publish-pypi.yml`` relies on to publish the infra images and record them in the wheel."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from agent_env.env.published_images import PLATFORMS, RELEASE_FILE

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((REPO / ".github" / "workflows" / "publish-pypi.yml").read_text())
RECORD = str(RELEASE_FILE.relative_to(REPO))


def _steps(job: str) -> list[dict]:
    return WORKFLOW["jobs"][job]["steps"]


def test_the_wheel_records_the_digests_where_an_install_reads_them():
    (step,) = [step for step in _steps("framework") if "infra_images.py write" in step.get("run", "")]

    assert re.search(rf"infra_images\.py write {re.escape(RECORD)}$", step["run"], re.MULTILINE)


def test_the_record_is_written_before_the_protocol_pin_edits_the_locked_pyproject():
    runs = [step.get("run", "") for step in _steps("framework")]
    record = next(i for i, run in enumerate(runs) if "infra_images.py write" in run)
    pin = next(i for i, run in enumerate(runs) if "agentenv-framework-protocol==" in run)

    assert record < pin


def test_the_record_ships_in_the_sdist_and_the_wheel_and_git_ignores_it():
    targets = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]

    assert RECORD in targets["wheel"]["artifacts"] and RECORD in targets["sdist"]["artifacts"]
    assert RECORD in (REPO / ".gitignore").read_text().splitlines()


def test_each_image_is_built_for_every_platform_it_is_published_for():
    arches = WORKFLOW["jobs"]["infra-build"]["strategy"]["matrix"]["arch"]

    assert [f"linux/{arch}" for arch in arches] == list(PLATFORMS)


def test_nothing_reaches_pypi_from_a_run_by_hand():
    assert "github.event_name == 'push'" in WORKFLOW["jobs"]["protocol"]["if"]
    assert "needs.protocol.result == 'success'" in WORKFLOW["jobs"]["framework"]["if"]


@pytest.mark.parametrize("job", ["protocol", "framework"])
def test_a_failed_image_job_sends_neither_package_to_pypi(job):
    """Each waits for every image job, and runs only when each succeeded or, nothing to publish, was skipped."""
    spec = WORKFLOW["jobs"][job]
    images = ["infra-plan", "infra-build", "infra-publish"]

    assert set(images) <= set(spec["needs"])
    for image_job in images:
        assert f"""contains(fromJSON('["success", "skipped"]'), needs.{image_job}.result)""" in spec["if"]
    assert spec["if"].startswith("!cancelled()")


def test_the_release_notes_list_the_images_the_wheel_records():
    framework, release = WORKFLOW["jobs"]["framework"], WORKFLOW["jobs"]["release"]
    (record,) = [step for step in framework["steps"] if step.get("id") == "record"]
    (create,) = [step for step in release["steps"] if "gh release create" in step.get("run", "")]

    assert "infra_images.py notes" in record["run"] and framework["outputs"]["image_notes"] == "${{ steps.record.outputs.notes }}"
    assert create["env"]["IMAGE_NOTES"] == "${{ needs.framework.outputs.image_notes }}" and "IMAGE_NOTES" in create["run"]


def test_nothing_is_pushed_before_mains_ci_passes():
    jobs = WORKFLOW["jobs"]

    assert "main-ci" in jobs["infra-build"]["needs"] and "main-ci" in jobs["infra-publish"]["needs"]
