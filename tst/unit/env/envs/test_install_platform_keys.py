from types import SimpleNamespace

from agent_env.env.envs.computer.utils.install import Install, for_os, installs


def test_an_install_is_for_an_os_type_or_one_of_its_distributions():
    serves = {"services": [{"name": "driver", "command": "venv/bin/python -m driver"}]}
    found, problems = installs({"linux": serves, "linux/omarchy": {**serves, "python": "requirements.txt"},
                                "plan9": serves, "linux/": serves, "linux/Omarchy": serves})

    assert set(found) == {"linux", "linux/omarchy"} and isinstance(found["linux/omarchy"], Install)
    assert [problem.partition(":")[0] for problem in problems] == ["install.plan9", "install.linux/",
                                                                   "install.linux/Omarchy"]


def test_a_distributions_entry_replaces_its_os_types_there_and_only_there():
    by_platform = {"linux": "any linux", "linux/omarchy": "omarchy"}

    assert for_os(by_platform, SimpleNamespace(type="linux", distribution="omarchy")) == "omarchy"
    assert for_os(by_platform, SimpleNamespace(type="linux", distribution="ubuntu")) == "any linux"
    assert for_os(by_platform, SimpleNamespace(type="linux")) == "any linux"
    assert for_os({"linux/omarchy": "omarchy"}, SimpleNamespace(type="linux", distribution="ubuntu")) is None
