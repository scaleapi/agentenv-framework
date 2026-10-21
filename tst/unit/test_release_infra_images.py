"""What ``.github/workflows/publish-pypi.yml`` relies on to publish the infra images and record them in the wheel."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from agent_env.env.published_images import RELEASE_FILE

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((REPO / ".github" / "workflows" / "publish-pypi.yml").read_text())
JOBS = WORKFLOW["jobs"]
RECORD = str(RELEASE_FILE.relative_to(REPO))
# What installs or runs agent-env and its dependencies in a job.
_RUNS_DEPENDENCIES = re.compile(r"\buv (sync|run)\b|infra_images\.py")


def _runs(job: str) -> list[str]:
    return [step.get("run", "") for step in JOBS[job]["steps"]]


def _step(job: str, step_id: str) -> dict:
    (step,) = [step for step in JOBS[job]["steps"] if step.get("id") == step_id]
    return step


def _writes(job: str) -> bool:
    permissions = JOBS[job].get("permissions", {})
    return any(access == "write" for access in permissions.values())


@pytest.mark.parametrize("job", sorted(JOBS))
def test_a_job_that_installs_agent_env_holds_no_token_that_writes(job):
    """Then a dependency's code never runs where a token can push an image, publish to PyPI or edit the repository."""
    if any(_RUNS_DEPENDENCIES.search(run) for run in _runs(job)):
        assert not _writes(job), f"{job} installs agent-env and holds {JOBS[job]['permissions']}"


def test_the_jobs_that_push_or_publish_check_what_they_are_handed():
    build = "\n".join(_runs("infra-build"))
    publish = "\n".join(_runs("infra-publish"))

    assert 'infra_build.py check "$ENTRY"' in build and 'infra_build.py check "$entry"' in publish
    assert build.index("infra_build.py check") < build.index("docker build")
    assert publish.index("infra_build.py check") < publish.index("imagetools create")
    assert "infra_build.py verify-record " + RECORD in _step("framework", "record")["run"]


def test_a_published_tag_is_never_built_again_or_moved():
    wanted = _step("infra-build", "wanted")["run"]
    publish = "\n".join(_runs("infra-publish"))

    assert 'infra_build.py exists "$REF"' in wanted
    assert all(step.get("if") == "steps.wanted.outputs.build == 'true'"
               for step in JOBS["infra-build"]["steps"] if step.get("name") in ("Build", "Check that it runs",
                                                                               "Push it under its tag and platform"))
    assert publish.index('infra_build.py exists "$ref"') < publish.index("imagetools create")


def test_nothing_is_pushed_before_mains_ci_passes():
    assert "main-ci" in JOBS["infra-build"]["needs"] and "main-ci" in JOBS["infra-publish"]["needs"]


@pytest.mark.parametrize("job", ["protocol", "framework"])
def test_a_failed_image_job_sends_neither_package_to_pypi(job):
    """Each runs with publishing off, or once the record, which runs only after every image job did, is made."""
    spec = JOBS[job]

    assert {"infra-plan", "infra-record"} <= set(spec["needs"]) and spec["if"].startswith("!cancelled()")
    assert ("(needs.infra-plan.result == 'skipped' || (needs.infra-plan.result == 'success' "
            "&& needs.infra-record.result == 'success'))") in " ".join(spec["if"].split())
    for image_job in ("infra-build", "infra-publish"):
        assert f"""contains(fromJSON('["success", "skipped"]'), needs.{image_job}.result)""" in JOBS["infra-record"]["if"]


def test_the_wheel_carries_the_checked_record_where_an_install_reads_it():
    step = _step("framework", "record")

    assert step["env"]["RECORD"] == "${{ needs.infra-record.outputs.record }}"
    assert f"""printf '%s\\n' "$RECORD" > {RECORD}""" in step["run"]


def test_the_record_ships_in_the_sdist_and_the_wheel_and_git_ignores_it():
    targets = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]

    assert RECORD in targets["wheel"]["artifacts"] and RECORD in targets["sdist"]["artifacts"]
    assert RECORD in (REPO / ".gitignore").read_text().splitlines()


def test_the_release_notes_list_the_checked_record():
    (create,) = [step for step in JOBS["release"]["steps"] if "gh release create" in step.get("run", "")]

    assert JOBS["framework"]["outputs"]["record"] == "${{ steps.record.outputs.record }}"
    assert create["env"]["RECORD"] == "${{ needs.framework.outputs.record }}"
    assert "Infra images published for this release, by digest:" in create["run"]


def test_nothing_reaches_pypi_from_a_run_by_hand():
    assert "github.event_name == 'push'" in JOBS["protocol"]["if"]
    assert "needs.protocol.result == 'success'" in JOBS["framework"]["if"]


@pytest.mark.parametrize("job", ["main-ci", "infra-plan"])
def test_a_run_by_hand_goes_ahead_only_on_main_or_a_release_tag(job):
    """main-ci waits for CI runs on main, which a branch has none of: off main, every job is skipped instead."""
    assert "github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v')" in JOBS[job]["if"]


def test_a_build_pulls_its_base_images_from_the_mirror_first():
    """public.ecr.aws can rate-limit a hosted runner's anonymous pulls for hours; the build pulls from it only what the
    mirror doesn't give."""
    (step,) = [step for step in JOBS["infra-build"]["steps"] if "mirror_base_images.py" in step.get("run", "")]
    build = "\n".join(_runs("infra-build"))

    assert step["run"] == 'python3 .github/scripts/mirror_base_images.py "$DOCKERFILE"'
    assert step["if"] == "steps.wanted.outputs.build == 'true'"
    assert build.index("mirror_base_images.py") < build.index("docker build")


def test_what_a_registry_can_rate_limit_is_tried_again():
    """A 429 on a base image's pull would fail the build, and with it the release's PyPI jobs."""
    build = "\n".join(_runs("infra-build"))
    publish = "\n".join(_runs("infra-publish"))

    assert re.search(r"infra_build\.py retry \\\s+docker build", build)
    assert 'infra_build.py retry docker push "$BUILT"' in build
    assert "infra_build.py retry docker buildx imagetools create" in publish
    assert JOBS["infra-build"]["strategy"]["max-parallel"] <= 4
