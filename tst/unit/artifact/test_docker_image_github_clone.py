"""put_from_github clones signed in from a credentials file when a token exists, anonymously otherwise."""

import os
import shlex
import subprocess

import pytest

from agent_env.artifact.artifacts.docker_image import _git_clone_command, _github_credentials


@pytest.mark.parametrize("folder", ["secrets", "secret files"])
def test_a_token_signs_the_clone_in_from_its_credentials_file_and_no_other_helper_keeps_it(tmp_path, folder):
    credentials = tmp_path / folder / "credentials"
    credentials.parent.mkdir()
    credentials.write_bytes(_github_credentials("ghs_s3cret"))
    kept = tmp_path / "kept"
    (tmp_path / ".gitconfig").write_text(f"[credential]\n\thelper = \"!f() {{ cat >> {kept}; }}; f\"\n")
    command = _git_clone_command("example-org", "example-repo", "main", str(credentials))

    words = shlex.split(command)
    assert words[:2] == ["GIT_TERMINAL_PROMPT=0", "git"] and words[words.index("clone"):] == [
        "clone", "--depth", "1", "--branch", "main", "https://github.com/example-org/example-repo.git", "/tmp/repo",
    ]
    options = words[words.index("git") + 1:words.index("clone")]
    env = {**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
    asked = "protocol=https\nhost=github.com\n\n"
    filled = subprocess.run(["git", *options, "credential", "fill"], input=asked, env=env, capture_output=True,
                            text=True, check=True)
    subprocess.run(["git", *options, "credential", "approve"], input=filled.stdout, env=env, check=True, text=True)
    assert "username=x-access-token\npassword=ghs_s3cret\n" in filled.stdout
    assert not kept.exists()


def test_without_a_token_the_clone_is_anonymous_and_never_prompts():
    assert _git_clone_command("docker-library", "hello-world", None, None) == (
        "GIT_TERMINAL_PROMPT=0 git clone --depth 1  https://github.com/docker-library/hello-world.git /tmp/repo"
    )
