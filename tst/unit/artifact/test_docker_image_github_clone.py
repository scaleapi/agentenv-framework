"""put_from_github clones with a throwaway askpass helper when a token exists, anonymously otherwise."""

from agent_env.artifact.artifacts.docker_image import _git_clone_commands


def test_token_rides_an_askpass_helper_removed_whether_or_not_the_clone_succeeds():
    cmds = _git_clone_commands("example-org", "example-repo", "main", "ghs_secret")
    assert len(cmds) == 3
    assert "ghs_secret" in cmds[0] and "x-access-token" in cmds[0]
    assert cmds[1] == "chmod +x /tmp/git-askpass.sh"
    assert cmds[2].startswith("trap 'rm -f /tmp/git-askpass.sh' EXIT; ")
    assert cmds[2].endswith("GIT_ASKPASS=/tmp/git-askpass.sh git clone --depth 1 --branch main https://github.com/example-org/example-repo.git /tmp/repo")
    assert "ghs_secret" not in cmds[2]


def test_without_a_token_the_clone_is_anonymous_and_never_prompts():
    cmds = _git_clone_commands("docker-library", "hello-world", None, None)
    assert len(cmds) == 1
    assert cmds[0].startswith("GIT_TERMINAL_PROMPT=0 git clone --depth 1")
    assert cmds[0].endswith("https://github.com/docker-library/hello-world.git /tmp/repo")
    assert "askpass" not in cmds[0] and "--branch" not in cmds[0]
