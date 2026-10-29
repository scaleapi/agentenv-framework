"""Aliases installed bundles give their agents, envs and artifacts: the aliases.toml format, and the table of every
installed bundle's."""

import os

import pytest

from agent_env.bundle import BundleError, parse_bundle
from agent_env.plugins._bundles import (
    ALIASES_FILE,
    ARTIFACT_ALIASES,
    ENV_ALIASES,
    InstalledBundle,
    installed_aliases,
    read_aliases,
)
from tst.unit.bundle._support import layout


def _bundle(tmp_path, name, package="agentenv-framework", aliases=None):
    root = tmp_path / package / name
    root.mkdir(parents=True)
    if aliases is not None:
        (root / ALIASES_FILE).write_text(aliases)
    return InstalledBundle(name, package, "1.0", package.replace("-", "_"), root, None, None)


def _read(tmp_path, text):
    (tmp_path / ALIASES_FILE).write_text(text)
    return read_aliases(tmp_path / ALIASES_FILE)


def test_an_aliases_file_gives_each_agent_env_and_artifact_its_aliases(tmp_path):
    text = ('[agents]\nclaude-code = ["claude-code", "claude"]\npi = ["pi"]\n[envs]\nmail = ["mail-mcp"]\n'
            '[artifacts]\n"ubuntu-24.04" = ["ubuntu-24.04", "ubuntu"]\n')

    assert _read(tmp_path, text) == (
        {"agents": {"claude-code": ("claude-code", "claude"), "pi": ("pi",)}, "envs": {"mail": ("mail-mcp",)},
         "artifacts": {"ubuntu-24.04": ("ubuntu-24.04", "ubuntu")}}, [])


def test_a_bundle_without_an_aliases_file_gives_none(tmp_path):
    assert read_aliases(tmp_path / ALIASES_FILE) == ({}, [])


@pytest.mark.parametrize("text, problem", [
    ('[agents]\nx = ["Claude"]\n', "agents.x: 'Claude' isn't an alias: lowercase letters"),
    ('[agents]\nx = ["@local/a/b"]\n', "agents.x: '@local/a/b' isn't an alias"),
    ('[agents]\nx = ["a/b"]\n', "agents.x: 'a/b' isn't an alias"),
    ('[agents]\nx = ["-x"]\n', "agents.x: '-x' isn't an alias"),
    ('[agents]\nx = ["' + "a" * 65 + '"]\n', "isn't an alias"),
    ('[agents]\nx = ["helper\\n"]\n', "agents.x: 'helper\\n' isn't an alias"),
    ('[agents]\nx = "x"\n', "agents.x must be a list of aliases, not 'x'"),
    ('[agents]\nx = []\n', "agents.x must be a list of aliases, not []"),
    ('[agents]\nx = ["a"]\ny = ["a"]\n', "agents.y: 'a' is listed for agents.x too; an alias names one agent"),
    ('[agents]\nx = ["a", "a"]\n', "agents.x: 'a' is listed twice"),
    ('[skills]\nx = ["a"]\n', "[skills] isn't a table it takes; it takes [agents], [envs] and [artifacts]"),
    ('[envs]\nx = ["a", "a"]\n', "envs.x: 'a' is listed twice; an alias names one env"),
    ('agents = 3\n', "[agents] must be a table"),
    ('[agents\n', "isn't valid TOML"),
], ids=["case", "namespaced", "slash", "edge", "long", "final-newline", "not-a-list", "empty", "two-agents", "twice", "other-table",
        "env-twice", "not-a-table", "toml"])
def test_an_aliases_file_with_a_problem_gives_no_aliases_and_says_why(tmp_path, text, problem):
    aliases, problems = _read(tmp_path, text)

    assert aliases == {} and len(problems) == 1 and problem in problems[0]


@pytest.mark.parametrize("make", [lambda path: path.mkdir(), lambda path: path.symlink_to(path.with_name("gone")),
                                  lambda path: os.mkfifo(path)], ids=["folder", "broken-symlink", "fifo"])
def test_an_aliases_file_that_isnt_a_readable_file_gives_no_aliases_and_says_so(tmp_path, make):
    make(tmp_path / ALIASES_FILE)

    aliases, [problem] = read_aliases(tmp_path / ALIASES_FILE)

    assert (aliases, problem) == ({}, "can't be read: it isn't a regular file")


def test_an_installed_bundles_alias_names_its_agent_under_the_bundles_id_root(tmp_path):
    agents = _bundle(tmp_path, "agents", aliases='[agents]\nclaude-code = ["claude-code"]\n')

    table = installed_aliases((agents,))

    named = table.target("claude-code")
    assert (named.target_id, named.bundle, named.declared_by) == (
        "@local/agentenv-framework/agents/claude-code", agents, "agentenv-framework's agents bundle")
    assert table.of(agents) == ("claude-code",) and table.target("pi") is None and table.refusal("claude-code") is None


def test_agent_envs_own_aliases_are_refused_to_any_other_bundle(tmp_path):
    ours = _bundle(tmp_path, "agents", aliases='[agents]\nclaude-code = ["claude-code"]\n')
    theirs = _bundle(tmp_path, "kit", package="acme-kit", aliases='[agents]\nmine = ["claude-code", "mine"]\n')

    table = installed_aliases((theirs, ours))

    assert table.target("claude-code").bundle == ours and table.target("mine").bundle == theirs
    assert table.refused_claims == {"acme-kit/kit": (
        "'claude-code' is agent-env's own alias, of @local/agentenv-framework/agents/claude-code, so this bundle "
        "can't give it",)}


def test_an_installed_bundle_gives_its_envs_and_artifacts_aliases_of_their_own(tmp_path):
    """An env, an agent and an artifact alias are kept apart, each where a reference to its kind looks."""
    mail = _bundle(tmp_path, "mail-mcp", aliases='[envs]\nmail = ["mail-mcp"]\n[agents]\nhelper = ["mail-mcp"]\n'
                                                 '[artifacts]\ninbox = ["mail-mcp"]\n')

    envs, agents = installed_aliases((mail,), ENV_ALIASES), installed_aliases((mail,))
    artifacts = installed_aliases((mail,), ARTIFACT_ALIASES)

    assert envs.target("mail-mcp").target_id == "@local/agentenv-framework/mail-mcp/mail"
    assert agents.target("mail-mcp").target_id == "@local/agentenv-framework/mail-mcp/helper"
    assert artifacts.target("mail-mcp").target_id == "@local/agentenv-framework/mail-mcp/inbox"
    assert envs.hint("mail_mcp") == "; did you mean the alias 'mail-mcp'?"


def test_a_name_close_to_an_alias_is_pointed_at_it(tmp_path):
    table = installed_aliases((_bundle(tmp_path, "agents", aliases='[agents]\nclaude-code = ["claude-code"]\n'),))

    assert table.hint("claude_code") == ("; did you mean the alias 'claude-code'? agent-env a2a-agent list lists the "
                                         "agents a task can name")
    assert table.hint("my-own-agent") == ""


def test_an_alias_two_other_bundles_give_names_neither_and_says_both(tmp_path):
    one = _bundle(tmp_path, "kit", package="acme-kit", aliases='[agents]\na = ["helper"]\n')
    two = _bundle(tmp_path, "tools", package="zeta-tools", aliases='[agents]\nb = ["helper"]\n')

    table = installed_aliases((one, two))

    assert table.target("helper") is None
    assert table.refusal("helper") == ("'helper' is an alias acme-kit's kit bundle and zeta-tools's tools bundle each "
                                       "give an agent, so it names none of them; name the agent by its id")
    assert set(table.refused_claims) == {"acme-kit/kit", "zeta-tools/tools"}


def test_a_bundle_that_cant_run_or_whose_aliases_file_has_a_problem_gives_no_aliases(tmp_path):
    broken = _bundle(tmp_path, "kit", package="acme-kit", aliases='[agents]\na = ["helper"]\n')
    broken = InstalledBundle(broken.name, broken.package, broken.version, broken.value, broken.root, "invalid-plugin",
                             "it can't be read")
    bad_file = _bundle(tmp_path, "tools", package="zeta-tools", aliases='[agents]\nb = ["Helper"]\n')

    assert installed_aliases((broken, bad_file)).aliases == {}


_AGENTS = {"agents/solver/Dockerfile": "FROM scratch\n", "agents/helper/Dockerfile": "FROM scratch\n",
           "tasks/t.json": "[]"}


def _problems(root):
    with pytest.raises(BundleError) as e:
        parse_bundle(root, id_root="@local/acme-kit/kit")
    return list(e.value.problems)


def test_a_bundle_gives_its_agents_the_aliases_its_aliases_file_lists(tmp_path):
    root = layout(tmp_path / "kit", {**_AGENTS, ALIASES_FILE: '[agents]\nsolver = ["solver", "fixer"]\n'})

    bundle = parse_bundle(root, id_root="@local/acme-kit/kit")

    assert bundle.aliases == {"agents": {"solver": ("solver", "fixer")}}
    assert ALIASES_FILE not in map(str, bundle.ignored)


@pytest.mark.parametrize("aliases, problem", [
    ('[agents]\nnobody = ["x"]\n', "aliases.toml: agents.nobody: this bundle has no agent 'nobody'"),
    ('[agents]\nsolver = ["helper"]\n', "aliases.toml: agents.solver: 'helper' is the agent 'helper''s name"),
    ('[agents]\nsolver = ["Solver"]\n', "aliases.toml: agents.solver: 'Solver' isn't an alias"),
    ('[envs]\nsolver = ["x"]\n', "aliases.toml: envs.solver: this bundle has no env 'solver'"),
    ('[artifacts]\nsolver = ["x"]\n', "aliases.toml: artifacts.solver: this bundle has no artifact 'solver'"),
], ids=["no-such-agent", "another-agents-name", "syntax", "an-agent-as-an-env", "an-agent-as-an-artifact"])
def test_an_aliases_file_that_doesnt_fit_the_bundles_agents_is_refused(tmp_path, aliases, problem):
    root = layout(tmp_path / "kit", {**_AGENTS, ALIASES_FILE: aliases})

    [found] = _problems(root)

    assert found.startswith(problem)


def test_a_bundle_gives_its_envs_the_aliases_its_aliases_file_lists(tmp_path):
    root = layout(tmp_path / "kit", {**_AGENTS, "envs/mail/Dockerfile": "FROM scratch\n",
                                     ALIASES_FILE: '[envs]\nmail = ["mail-mcp"]\n'})

    assert parse_bundle(root, id_root="@local/acme-kit/kit").aliases == {"envs": {"mail": ("mail-mcp",)}}


def test_an_alias_names_an_agent_by_its_folder_so_the_agent_keeps_its_folders_id(tmp_path):
    root = layout(tmp_path / "kit", {**_AGENTS, "agents/solver/agent.toml": 'id = "@local/acme-kit/elsewhere"\n',
                                     ALIASES_FILE: '[agents]\nsolver = ["solver"]\n'})

    [found] = _problems(root)

    assert found == ("aliases.toml: agents.solver: the agent declares the id '@local/acme-kit/elsewhere', and an alias "
                     "names an agent by its folder, so it must keep the id @local/acme-kit/kit/solver")


def test_a_misspelled_aliases_file_is_refused_rather_than_ignored(tmp_path):
    root = layout(tmp_path / "kit", {**_AGENTS, "Aliases.TOML": '[agents]\nsolver = ["solver"]\n'})

    [found] = _problems(root)

    assert found == "Aliases.TOML: spell it aliases.toml, the file a bundle gives its agents' aliases in"


@pytest.mark.parametrize("make, problem", [
    (lambda path: path.mkdir(), "aliases.toml: must be a file, not a folder"),
    (lambda path: path.symlink_to(path.with_name("gone")), "aliases.toml: a broken symlink"),
    (lambda path: os.mkfifo(path), "aliases.toml: neither a regular file nor a folder"),
], ids=["folder", "broken-symlink", "fifo"])
def test_an_aliases_file_that_isnt_a_file_is_refused_rather_than_ignored(tmp_path, make, problem):
    root = layout(tmp_path / "kit", _AGENTS)
    make(root / ALIASES_FILE)

    assert _problems(root) == [problem]
