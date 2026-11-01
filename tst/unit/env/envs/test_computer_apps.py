"""Computer envs of several apps: each app built into the computer after the driver, in the order the env lists them;
the apps checked against each other at put; and what their run scripts and install steps wrote compared before a push,
a part certainly undoing what an earlier one installed refused and any other file two parts wrote differently said
(``clashes``). Docker, the registry and the VMs are faked."""

import pytest

from agent_env.env import Env
from agent_env.env.envs.computer.utils import clashes
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.store.base import NotFoundError
from tst.unit.env.envs._computer import MAIL_SERVER, FakeVmProvider, _app, _put

MAIL, NOTES = "@local/~/kit/mail", "@local/~/kit/notes"
A, B, C = "@local/~/kit/a", "@local/~/kit/b", "@local/~/kit/c"


def test_a_computer_env_builds_its_apps_after_its_driver_in_the_order_it_lists_them(base, driver, docker, tmp_path):
    mail = _app(base, driver, tmp_path / "mail", compose=False)
    notes = _app(base, driver, tmp_path / "notes", compose=False, name="notes", id="@local/~/other/mail",
                 install={"ports": [8025], "services": [{"name": "sync", "command": "venv/bin/python sync.py"}]})

    env = _put(base, driver, apps=iter([notes, mail]))

    assert sorted(build["platform"] for build in docker.builds) == ["linux/amd64", "linux/arm64"]
    dockerfile, confs = docker.builds[0]["dockerfile"], docker.builds[0]["confs"]
    assert dockerfile.index("COPY driver/") < dockerfile.index("COPY apps/notes/") < dockerfile.index(
        "COPY apps/mail/") < dockerfile.index("USER user")
    assert [layer["name"] for layer in docker.resolved[0][1]["python"]][-2:] == ["apps-notes", "apps-mail"]
    assert confs["agentenv-app-mail.conf"].startswith(MAIL_SERVER) and "[program:app-notes-sync]" in confs[
        "agentenv-app-notes.conf"]
    assert sorted(confs) == ["agentenv-app-mail.conf", "agentenv-app-notes.conf", "agentenv-driver-desk.conf"]
    stored = Env.get(env.id)
    assert [(app.id, app.version) for app in stored.composed_apps()] == [(notes.id, 1), (mail.id, 1)]
    assert stored.to_dict()["apps"] == [{"id": notes.id, "version": 1}, {"id": mail.id, "version": 1}]
    assert stored.build_statuses() == {"container/amd64": "unverified", "container/arm64": "ready"}
    assert docker.saved == []
    assert _put(base, driver, apps=[mail, notes]).metadata["composed"]["container"]["arm64"]["key"] != (
        stored.metadata["composed"]["container"]["arm64"]["key"])


@pytest.mark.parametrize("apps, problem", [
    (lambda mail, other: [mail, mail], "apps: @local/~/kit/mail is listed twice"),
    (lambda mail, other: [mail, other(name="mail-2", environment_name="mail")],
     "apps: @local/~/kit/mail and @local/~/kit/mail-2 would share the environment name 'mail'"),
    (lambda mail, other: [mail, other(name="x", environment_name="Mail")],
     "apps: @local/~/kit/mail and @local/~/kit/x would share the folder '/opt/agentenv/apps/mail'"),
    (lambda mail, other: [other(name="a", install={"ports": [8025], "services": []}),
                          other(name="b", install={"ports": [8025, 9000], "services": []})],
     "apps: @local/~/kit/a and @local/~/kit/b would share the port 8025"),
    (lambda mail, other: [mail, other(name="desk-port", install={"ports": [18765], "services": []})],
     "apps[1]: @local/~/kit/desk-port listens on 18765, as @local/~/kit/desk does"),
    (lambda mail, other: [mail, other(name="arm", supports=[{"os": {"type": "linux", "display": ["x11"]},
                                                              "arch": ["riscv64"]}])],
     "apps[1]: no architecture @local/~/kit/ubuntu-24.04, @local/~/kit/desk and @local/~/kit/arm all run"),
    (lambda mail, other: [other(name=arch, supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": [arch]}])
                          for arch in ("amd64", "arm64")],
     "apps: no architecture @local/~/kit/ubuntu-24.04, @local/~/kit/desk and every app all run"),
    (lambda mail, other: [other(name="mail", install={"services": [{"name": "sync", "command": "venv/bin/sync"}]}),
                          other(name="mail-sync", install={"services": []})],
     "apps: @local/~/kit/mail and @local/~/kit/mail-sync would share the program 'app-mail-sync'"),
    (lambda mail, other: [mail, other(name="desk-app", environment_name="desktop")],
     "apps: @local/~/kit/desk-app is named 'desktop', as the computer env is, which names the computer's own files"),
    (lambda mail, other: [other(name="a", paths=["Mail", ".config/a"]), other(name="b", paths=["Mail/inbox"]),
                          other(name="c", paths=[".config/c"])],
     "apps: @local/~/kit/a keeps its data in 'Mail' and @local/~/kit/b in 'Mail/inbox', so they'd share it"),
], ids=["twice", "environment-name", "folder", "app-port", "driver-port", "arch", "no-shared-arch", "program",
        "computer-name", "paths"])
def test_apps_that_dont_fit_together_or_on_the_computer_are_refused_before_anything_is_stored(base, driver, docker,
                                                                                              tmp_path, apps, problem):
    def other(name="other", **changes):
        return _app(None, None, tmp_path / name, compose=False, name=name, **changes)

    listed = apps(_app(base, driver, tmp_path / "mail", compose=False), other)

    with pytest.raises(ValueError, match="computer env '@local/~/kit/desktop'") as refused:
        _put(base, driver, apps=listed)

    assert problem in str(refused.value) and str(refused.value).count(" would share ") <= 1
    with pytest.raises(NotFoundError):
        Env.get("@local/~/kit/desktop")
    assert docker.builds == []


def _setting_up(tmp_path, *names, scripts=("setup.sh",)):
    """An app of each of ``names`` whose install runs ``scripts`` as root, put without a computer."""
    return [_app(None, None, tmp_path / name, compose=False, name=name,
                 install={"run": [{"script": script} for script in scripts], "services": []},
                 extra={script: "#!/bin/sh\n" for script in scripts}) for name in names]


@pytest.mark.parametrize("layers, lost", [
    ([(A, {"/usr/local/bin/tool": "file 1"}), (B, {"/usr/local/bin/tool": "deleted"})],
     [f"/usr/local/bin/tool: {B} deletes {A}'s file"]),
    ([(A, {"/opt/shared/data": "file 1", "/opt/shared/sub": "dir"}),
      (B, {"/opt/shared": "opaque", "/opt/shared/other": "file 2"})],
     [f"/opt/shared/data: {B} deletes {A}'s file"]),
    ([(A, {"/usr/bin/tool": "link /opt/a/tool"}), (B, {"/usr/bin/tool": "link /opt/b/tool"})],
     [f"/usr/bin/tool: {B} points {A}'s link elsewhere"]),
    ([(A, {"/usr/bin/tool": "file 1"}), (B, {"/usr/bin/tool": "link /opt/b/tool"})],
     [f"/usr/bin/tool: {B} makes {A}'s file a link"]),
    ([(A, {"/opt/x": "dir"}), (B, {"/opt/x": "file 1"})], [f"/opt/x: {B} makes {A}'s folder a file"]),
    ([(A, {"/usr/local/bin/tool": "exec 1"}), (B, {"/usr/local/bin/tool": "exec 2"})],
     [f"/usr/local/bin/tool: {B} replaces {A}'s program"]),
    ([(A, {"/t": "exec 1"}), (B, {"/t": "exec 1"}), (B, {"/t": "deleted"})], [f"/t: {B} deletes {A}'s file"]),
    ([(A, {"/t": "link /a"}), (B, {"/t": "link /a"}), (B, {"/t": "link /b"})],
     [f"/t: {B} points {A}'s link elsewhere"]),
    ([(A, {"/o/data": "file 1"}), (B, {"/o": "opaque"}), (B, {"/o/data": "file 2"})],
     [f"/o/data: {B} deletes {A}'s file"]),
    ([(A, {"/t": "exec 1"}), (B, {"/t": "exec 2"}), (C, {"/t": "deleted"})],
     [f"/t: {B} replaces {A}'s program", f"/t: {C} deletes {B}'s file"]),
    ([(A, {"/opt/s": "dir", "/opt/s/t": "exec 1"}), (B, {"/opt/s": "deleted"}), (B, {"/opt/s/t": "exec 2"})],
     [f"/opt/s: {B} deletes {A}'s folder"]),
], ids=["deleted", "opaque", "link", "kind", "folder", "program", "kept-then-deleted", "kept-then-pointed",
        "opaque-then-written", "after-a-loss", "a-folder-deleted"])
def test_a_part_that_certainly_undoes_what_another_left_is_a_loss(layers, lost):
    assert clashes.compare(layers) == (lost, [])


@pytest.mark.parametrize("layers, differ", [
    ([(A, {"/etc/passwd": "file 1"}), (B, {"/etc/passwd": "file 2"})], [f"/etc/passwd ({A}, {B})"]),
    ([(A, {"/x": "file 1"}), (B, {"/x": "exec 2"})], [f"/x ({A}, {B})"]),
    ([(A, {"/x": "exec 1", "/l": "link /a", "/d": "dir"}), (B, {"/x": "exec 1", "/l": "link /a", "/d": "dir"})], []),
    ([(A, {path: "file 1" for path in ("/var/log/s.log", "/tmp/x", "/run/x", "/home/user/.cache/x",
                                        "/o/__pycache__/m.pyc")}),
      (B, {"/var/log/s.log": "deleted", "/tmp/x": "exec 2", "/run/x": "link /x", "/home/user/.cache/x": "dir",
           "/o/__pycache__/m.pyc": "file 2"})], []),
    ([(A, {"/x": "exec 1"}), (A, {"/x": "deleted"})], []),
    ([(A, {"/o/d": "file 1"}), (A, {"/o/d": "deleted"}), (B, {"/o": "opaque"})], []),
    ([(A, {"/t": "exec 1"}), (A, {"/t": "deleted"}), (B, {"/t": "exec 2"}), (B, {"/t": "deleted"})], []),
    ([(A, {"/o/t": "exec 1"}), (A, {"/o": "opaque"}), (B, {"/o/t": "exec 2"}), (B, {"/o/t": "deleted"})], []),
    ([(A, {"/a": "file 1"}), (B, {"/b": "file 2"})], []),
    ([(A, {"/o/t": "exec 1"}), (A, {"/o": "deleted"}), (B, {"/o/t": "exec 2"}), (B, {"/o/t": "deleted"})], []),
    ([(A, {"/usr/local/bin/tool": "exec => /usr/bin/tool"}), (B, {"/usr/local/bin/tool": "exec 9"})],
     [f"/usr/local/bin/tool ({A}, {B})"]),
], ids=["edited", "made-a-program", "same", "scratch", "own", "deleted-before-opaque", "recreated",
        "recreated-after-opaque", "apart", "recreated-after-its-folder-was-deleted", "hard-link-to-an-earlier-layer"])
def test_what_isnt_certainly_lost_isnt_a_loss_and_only_a_file_two_parts_wrote_differently_is_said(layers, differ):
    assert clashes.compare(layers) == ([], differ)


def test_a_part_that_certainly_undoes_what_an_earlier_one_installed_fails_its_pairs_at_build_before_the_push(
        base, driver, docker, tmp_path):
    docker.writes = {"apps/mail && ": {"usr/local/bin/tool": (b"mail's", 0o755)},
                     "apps/notes && ": {"usr/local/bin/tool": (b"notes'", 0o755)}}

    env = Env.get(_put(base, driver, apps=_setting_up(tmp_path, "mail", "notes")).id)

    assert sorted(docker.saved) == sorted(build["tag"] for build in docker.builds)
    composed = env.metadata["composed"]["container"]
    assert {arch: (r["status"], r["stage"], r["reason"]) for arch, r in composed.items()} == {
        arch: ("failed", "build", f"a part undoes what an earlier one installed: /usr/local/bin/tool: {NOTES} replaces "
               f"{MAIL}'s program") for arch in ("amd64", "arm64")}
    assert docker.probed == [] and "computer" not in env.metadata


@pytest.mark.parametrize("mail, notes, told", [
    ({"etc/passwd": b"root\\nmail\\n", "usr/share/icons/hicolor/icon-theme.cache": b"\\x00mail"},
     {"etc/passwd": b"root\\nmail\\nnotes\\n", "usr/share/icons/hicolor/icon-theme.cache": b"\\x00mail notes"},
     f"/etc/passwd ({MAIL}, {NOTES}), /usr/share/icons/hicolor/icon-theme.cache ({MAIL}, {NOTES})"),
    ({f"etc/conf.d/{i:02}": b"mail's" for i in range(12)}, {f"etc/conf.d/{i:02}": b"notes'" for i in range(12)},
     ", ".join(f"/etc/conf.d/{i:02} ({MAIL}, {NOTES})" for i in range(10)) + " and 2 more"),
    ({"etc/x": b"same"}, {"etc/x": b"same"}, None),
], ids=["edited", "many", "same"])
def test_files_two_parts_wrote_differently_are_said_on_a_progress_line_and_not_refused(base, driver, docker, tmp_path,
                                                                                       mail, notes, told):
    docker.writes, said = {"apps/mail && ": mail, "apps/notes && ": notes}, []

    env = Env.get(_put(base, driver, apps=_setting_up(tmp_path, "mail", "notes"),
                       composing=Composing(on_progress=said.append)).id)

    assert env.build_statuses() == {"container/amd64": "unverified", "container/arm64": "ready"}
    compared = [line.partition(": ")[2] for line in said if line.startswith("container/arm64: not refused")]
    assert compared == ([f"not refused, as each may keep the earlier part's change: two parts wrote {told}"]
                        if told else [])


def test_one_part_with_run_scripts_has_nothing_to_compare_and_its_image_isnt_saved(base, driver, docker, tmp_path):
    docker.writes = {"apps/mail && ": {"usr/local/bin/tool": b"mail's"}}
    mail = _setting_up(tmp_path, "mail")[0]

    env = Env.get(_put(base, driver, apps=[mail, _app(None, None, tmp_path / "notes", compose=False, name="notes")]).id)

    assert env.build_statuses() == {"container/amd64": "unverified", "container/arm64": "ready"} and docker.saved == []


def test_a_builder_vm_compares_what_each_part_wrote_in_a_container_of_the_image_before_pushing(base, driver, docker,
                                                                                              signing, tmp_path):
    docker.writes = {"apps/mail && ": {"usr/local/bin/tool": (b"mail's", 0o755)},
                     "apps/notes && ": {"usr/local/bin/tool": (b"notes'", 0o755)}}
    vms = FakeVmProvider(docker, tmp_path / "vms")

    env = Env.get(_put(base, driver, apps=_setting_up(tmp_path, "mail", "notes"), composing=Composing(vms)).id)

    composed = env.metadata["composed"]["container"]
    assert {arch: (r["status"], r["reason"].partition(": ")[2]) for arch, r in composed.items()} == {
        arch: ("failed", f"/usr/local/bin/tool: {NOTES} replaces {MAIL}'s program") for arch in ("amd64", "arm64")}
    compared = [line for line in vms.ran if "/layers.py" in line]
    assert len(compared) == 2 and all("docker run --rm --network none --user root" in line for line in compared)
    assert not [line for line in vms.ran if "docker push" in line] and all(vm.terminated for vm in vms.vms)
    assert not list((tmp_path / "vms").rglob("image.tar"))


