Base computers that come with agent-env, each a `computer` artifact a computer env composes a driver and apps onto.

`artifacts/ubuntu-24.04/` is Ubuntu 24.04 (`amd64` and `arm64`) with XFCE on Xvfb, D-Bus and AT-SPI. tini starts
supervisord, which runs every program as the session user, `user` (uid 1000); nothing in it runs as root. Its home
has a desktop's standard folders (`Desktop`, `Documents`, `Downloads`, `Pictures` and the rest, from `xdg-user-dirs`). Its
container form's image is built from the folder's `Dockerfile`, and a release publishes it as
`ghcr.io/scaleapi/agentenv-ubuntu-24.04:src-<hex>`, tagged by its sources, so a run pulls it by digest instead of
building it. Name it from another bundle as `ubuntu-24.04`, the alias `aliases.toml` gives it, or by its id,
`@local/agentenv-framework/computers/ubuntu-24.04`.

It also has a disk form, a disk a VM boots from UEFI, which is the container form's root filesystem on the Linux boot
recipe in `disk/linux/`: a GPT disk of an EFI system partition holding one UKI, Ubuntu's kernel with a busybox initramfs,
and an ext4 root filesystem that is the image's own, with `/etc/agentenv/vm/image.env`, its entrypoint, command,
user, working folder and environment, from its OCI config. The init mounts it, brings up the network, reads the deploy's
settings from a NoCloud seed (`CIDATA`) and becomes the image's entrypoint, as its user. A release publishes it as
`ghcr.io/scaleapi/agentenv-ubuntu-24.04-disk:src-<hex>`, tagged by both its sources; without it, composing makes it.
Composing a computer env makes each arch's disk from its composed container, the root filesystem rebuilt with
`mkfs.ext4 -d` by the disk tools (`disk/linux/tools/`) on the builder's own platform and the base disk's EFI system
partition kept, so nothing is emulated or privileged. A disk boots where `/dev/kvm` is, in the QEMU launcher
(`disk/linux/launcher/`): gate 3 probes it there, and elsewhere it stays `unverified` until a deploy's first boot with
KVM probes it. A deploy boots it with `env_params = { form = "disk" }`. The launcher serves each of the computer's ports
through socat, whose listen backlog takes a burst of connections, in front of QEMU's own forward, whose backlog is 1, and
a reset boots the disk afresh on a new overlay in the same launcher, at the same URLs. It boots on local Docker where
Docker can pass it `/dev/kvm`, and on `modal_vm`, amd64 only, whose sandbox runs the launcher as its image under nested
virtualization: the provider exports the disk from the image store with this machine's Docker, compressed, once for each
composed disk, into the per-user cache, and mounts it on the sandbox at create, which Modal uploads once for each
content and the launcher expands onto the sandbox's disk. Such a sandbox is ended once it has been idle, with no
connection open, for the provider's `idle_timeout`, 30 minutes unless its config sets another, as a running QEMU isn't
activity. A gate 3 probe that ends with the connection to the driver dropped, refused or timed out runs again on a fresh
boot, up to three in all, and the record keeps each attempt; one with a failed check is recorded as it is.

A put, `env build` and the release compose every form; a run composes only the forms its sandbox's deploys boot: on each
link of a chain, the first form it boots natively, as a deploy picks, so local Docker composes containers, and a sandbox
that boots only disks composes disks. A run composes them on the env's default provider, this machine unless the config
names another, never on the sandbox it deploys on. A compose of a disk also composes the container it's made from,
unless that's built. A deploy that boots a form no compose has started, a disk named by `env_params` included, composes
it first on that same provider, saying which, reusing what's built, as every build is reused by its key; deploys of one
env at once compose it once, the others
waiting for it, a deploy that finds it building included. In `agent-env run` it says why and each stage as it goes, and
a deploy waiting for another's compose says what it waits for and since when. A local run doesn't compose a registered
env's form at deploy where a configured store isn't the per-user one, since the run's fence refuses its record: the
deploy refuses for want of a ready build and names `agent-env env build`.

Its settings reach the computer as environment variables, `AGENTENV_SETTING_<NAME>`, read as it starts:

| Setting | Default | What it does |
| --- | --- | --- |
| `resolution` | `1280x800` | The size of the Xvfb screen, as `<width>x<height>`, each 1 to 32767 pixels |
| `timezone` | `UTC` | The time zone every program runs in, a name under `/usr/share/zoneinfo` |

A value it can't use stops the computer at start, with the reason on stderr. Composing a driver or an app onto it adds
its services as supervisord programs in `/etc/supervisor/conf.d/`; they inherit the display (`DISPLAY=:99`), the
session bus and the time zone. A program that needs X runs as `start-desktop with-x <command>`, which waits for X, and
with `startsecs=0`, so X dying however often never leaves it `FATAL`, as the XFCE session does.

A computer env composes a driver onto it at put: an `envs/<name>/env.toml` of `type = "computer"` names
`computer = "ubuntu-24.04"` and `driver = "reference"`, and its put builds an image of the two for each arch this
machine's Docker can build, then boots the one this machine runs natively and probes it. The shipped `desktop` example
is that env, and `agent-env run desktop` deploys it.

A computer app env (`type = "computer_app"`) deployed alone is composed onto it the same way, with the app's files
under `/opt/agentenv/apps/<name>/`, a venv there with the protocol's `app` extra, and the app's own server and its
services as programs beside the driver's, run as `user` like everything else. A deploy waits for the computer to be
ready up to 180 s, or 30 s more than the slowest `boot` app's own `timeout_s`.

Everything in a computer runs as that one user, so any code in it, an app's, a driver's, a `run_code` script or a
command run through `computer-exec`, can read the environment of any other process there, settings included. A secret
setting, the base's or an app's, is kept out of the stored env, the task and the run's record, which hold only its
`secret:` or `env:` reference, but not from code in the computer: give a computer only secrets its own code may see.

This bundle has no tasks of its own: other bundles pull its computers by alias, and `agent-env run computers` has
nothing to run.
