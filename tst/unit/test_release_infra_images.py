"""What ``.github/workflows/publish-pypi.yml`` relies on to publish the images agent-env ships, its infra images and its
agents', and record them in the wheel."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from agent_env.env import bootstrap
from agent_env.env.published_images import AGENT_IMAGES, RELEASE_FILE
from tst.util.scripts import load_script

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((REPO / ".github" / "workflows" / "publish-pypi.yml").read_text())
JOBS = WORKFLOW["jobs"]
RECORD = str(RELEASE_FILE.relative_to(REPO))
# What installs or runs agent-env and its dependencies in a job.
_RUNS_DEPENDENCIES = re.compile(r"\buv (sync|run)\b|infra_images\.py")
# What reads the shipped bundles' folders: an image's tag, its build, and the wheel.
_READS_SHIPPED_SOURCES = re.compile(r"infra_images\.py (plan|write)|docker build\b|uv build --no-sources --out-dir dist")
_PIN = "python3 .github/scripts/shipped_bundles.py pin"


def _runs(job: str) -> list[str]:
    return [step.get("run", "") for step in JOBS[job]["steps"]]


def _step(job: str, step_id: str) -> dict:
    (step,) = [step for step in JOBS[job]["steps"] if step.get("id") == step_id]
    return step


def _writes(job: str) -> bool:
    permissions = JOBS[job].get("permissions", {})
    return any(access == "write" for access in permissions.values())


def _condition(job: str) -> str:
    return " ".join(JOBS[job]["if"].split())


def _finished(job: str) -> str:
    return f"""contains(fromJSON('["success", "skipped"]'), needs.{job}.result)"""


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


@pytest.mark.parametrize(("agent_job", "infra_job"), [("agent-build", "infra-build"), ("agent-publish", "infra-publish")])
def test_the_agent_image_jobs_are_the_infra_image_jobs_on_the_agents_list(agent_job, infra_job):
    """So each check here on how the infra images are built, pushed and tagged holds for the agents' images too."""
    def body(job: str) -> dict:
        return {key: value for key, value in JOBS[job].items() if key not in ("needs", "if")}

    assert yaml.safe_load(yaml.safe_dump(body(infra_job)).replace("missing_infra", "missing_agents")) == body(agent_job)


@pytest.mark.parametrize("job", ["infra-build", "infra-publish", "agent-build", "agent-publish"])
def test_nothing_is_pushed_before_mains_ci_passes(job):
    """A condition with ``!cancelled()`` runs whatever the jobs it needs ended as, so it checks main-ci's itself."""
    assert "main-ci" in JOBS[job]["needs"]
    assert "!cancelled()" not in JOBS[job]["if"] or "needs.main-ci.result == 'success'" in JOBS[job]["if"]


def test_a_failed_infra_image_job_sends_neither_package_to_pypi():
    """The protocol runs with publishing off, or once each infra image job succeeded or had nothing to do."""
    assert {"infra-plan", "infra-build", "infra-publish"} <= set(JOBS["protocol"]["needs"])
    assert _condition("protocol").startswith("!cancelled()")
    assert (f"(needs.infra-plan.result == 'skipped' || (needs.infra-plan.result == 'success' "
            f"&& {_finished('infra-build')} && {_finished('infra-publish')}))") in _condition("protocol")


def test_the_agents_images_are_built_once_the_protocol_they_install_is_on_pypi():
    """Each installs the version the release pins in its requirements.txt; run by hand, no protocol is published."""
    assert "protocol" in JOBS["agent-build"]["needs"]
    assert ("(needs.protocol.result == 'success' || (github.event_name == 'workflow_dispatch' "
            "&& needs.protocol.result == 'skipped'))") in _condition("agent-build")


def test_the_framework_waits_for_every_image_job():
    """It runs with publishing off, or once the record is made, which runs only after every image job did."""
    assert {"infra-plan", "infra-record"} <= set(JOBS["framework"]["needs"])
    assert _condition("framework").startswith("!cancelled()")
    assert ("(needs.infra-plan.result == 'skipped' || (needs.infra-plan.result == 'success' "
            "&& needs.infra-record.result == 'success'))") in _condition("framework")
    for image_job in ("infra-build", "infra-publish", "agent-build", "agent-publish"):
        assert image_job in JOBS["infra-record"]["needs"] and _finished(image_job) in _condition("infra-record")


def test_every_job_that_reads_the_shipped_bundles_pins_the_protocol_first():
    """The pin is a source of the agents' images: a job without it would tag, build, record or ship other sources."""
    readers = {job: "\n".join(_runs(job)) for job in JOBS if _READS_SHIPPED_SOURCES.search("\n".join(_runs(job)))}

    assert set(readers) == {"infra-plan", "infra-build", "agent-build", "infra-record", "framework"}
    for job, runs in readers.items():
        assert runs.count(_PIN) == 1 and runs.index(_PIN) < _READS_SHIPPED_SOURCES.search(runs).start(), job


def test_the_images_built_after_the_protocol_are_those_the_pin_changes():
    """An infra image is built before the protocol is on PyPI, so none of them can install the version a release pins."""
    shipped_bundles = load_script("shipped_bundles.py")
    pinned = [path for path in shipped_bundles.requirements_files()
              if any(line.startswith(shipped_bundles.PROTOCOL) for line in path.read_text().splitlines())]

    assert {image.name for image in bootstrap.PUBLISHED_IMAGES
            if any(path.is_relative_to(image.dockerfile.parent) for path in pinned)} == {i.name for i in AGENT_IMAGES}


def test_the_github_release_is_created_whenever_the_framework_is_published():
    """With no new image the image jobs are skipped, and a job's default condition skips it after any skipped job."""
    assert " ".join(JOBS["release"]["if"].split()) == "!cancelled() && needs.framework.result == 'success'"


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
    assert "Images published for this release, by digest:" in create["run"]


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


def test_every_infra_image_pins_its_bases_by_digest_and_dependabot_bumps_them():
    """A release rebuilds an image only when its sources change, so a base behind a bare tag never picks up its
    fixes; pinned, the pin is a source, and Dependabot bumping it is what rebuilds and republishes the image."""
    dependabot = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text())
    (docker,) = [update for update in dependabot["updates"] if update["package-ecosystem"] == "docker"]

    for image in bootstrap.PUBLISHED_IMAGES:
        text = image.dockerfile.read_text()
        stages = set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", text, re.IGNORECASE | re.MULTILINE))
        bases = [ref for ref in re.findall(r"^FROM\s+(?:--\S+\s+)*(\S+)", text, re.IGNORECASE | re.MULTILINE)
                 if ref not in stages]
        assert bases and all(re.search(r"@sha256:[0-9a-f]{64}$", ref) for ref in bases), (image.name, bases)
        assert f"/{image.dockerfile.parent.relative_to(REPO)}" in docker["directories"], image.name
