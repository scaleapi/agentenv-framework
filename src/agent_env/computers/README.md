Base computers that come with agent-env, each a `computer` artifact a computer env composes a driver and apps onto.

`artifacts/ubuntu-24.04/` is Ubuntu 24.04 (`amd64` and `arm64`) with XFCE on Xvfb, D-Bus and AT-SPI. tini starts
supervisord, which runs every program as the session user, `user` (uid 1000); nothing in it runs as root. Its home
has a desktop's standard folders (`Desktop`, `Documents`, `Downloads`, `Pictures` and the rest, from `xdg-user-dirs`). Its
container form's image is built from the folder's `Dockerfile`, and a release publishes it as
`ghcr.io/scaleapi/agentenv-ubuntu-24.04:src-<hex>`, tagged by its sources, so a run pulls it by digest instead of
building it. Name it from another bundle as `ubuntu-24.04`, the alias `aliases.toml` gives it, or by its id,
`@local/agentenv-framework/computers/ubuntu-24.04`.

`artifacts/ubuntu-24.04-headless/` is Ubuntu 24.04 (`amd64` and `arm64`) with no display server or desktop
(`display = "none"`), for one website browser, which runs headless there, and the `page` driver, whose screen is the
browser's page: a browser computer, which agentenv's `chromium` deploys alone as. supervisord runs every program as the
session user, among them a stand-in site proxy on `127.0.0.1:3128`, the browser's only route out, which forwards every
request to wherever it names, refusing nothing: the sandbox's network policy is all that limits it until the shared
site proxy replaces it. It has a container form only. Name it from another bundle as `ubuntu-24.04-headless`, or by
its id, `@local/agentenv-framework/computers/ubuntu-24.04-headless`.

It also has a disk form, a disk a VM boots from UEFI, which is the container form's root filesystem on the Linux boot
recipe in `disk/linux/`: a GPT disk of an EFI system partition holding one UKI, Ubuntu's kernel with a busybox initramfs,
and an ext4 root filesystem that is the image's own, with `/etc/agentenv/vm/image.env`, its entrypoint, command,
user, working folder and environment, from its OCI config. The init mounts it, brings up the network, reads the deploy's
settings from a NoCloud seed (`CIDATA`) and becomes the image's entrypoint, as its user. The seed is
`agentenv_protocol.computer.seed`'s: it carries every version its launcher writes, and the init (`disk/linux/boot/`,
its reader in `seed-env`) reads the newest it knows, so a disk keeps booting under a later agent-env's launcher, and
stops the boot, saying to recompose the disk, when the seed carries none. A release publishes it as
`ghcr.io/scaleapi/agentenv-ubuntu-24.04-disk:src-<hex>`, tagged by both its sources; without it, composing makes it.
Composing a computer env makes each arch's disk from its composed container, the root filesystem rebuilt with
`mkfs.ext4 -d` by the disk tools (`disk/linux/tools/`) on the builder's own platform and the base disk's EFI system
partition kept, so nothing is emulated or privileged. A disk boots where `/dev/kvm` is, in the QEMU launcher
(`disk/linux/launcher/`): gate 3 probes it there, and elsewhere it stays `unverified` until a deploy's first boot with
KVM probes it. A deploy boots it with `env_params = { form = "disk" }`. The launcher serves each of the computer's ports
through socat, whose listen backlog takes a burst of connections, in front of QEMU's own forward, whose backlog is 1, and
a reset boots the disk afresh on a new overlay in the same launcher, at the same URLs; a container an agent-env from
before reset in place started is refused a reset, to be redeployed. It boots on local Docker where
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

`artifacts/macos-26/` is macOS 26 on Apple silicon (`arm64`), a VM bundle: a whole saved virtual machine, its disk,
NVRAM and hardware identity, which Virtualization.framework restores as it was saved, in Lume's format. Its image is
Cua's public base, `ghcr.io/trycua/macos:26-20261021-fca2193`, pinned by digest beside the tag Lume pulls it by: its
user `lume` logs in at boot, System Integrity Protection is off and ssh is on. Name it as `macos-26`, or by its id,
`@local/agentenv-framework/computers/macos-26`. A VM bundle's image is a registry image pinned so, or `local:<name>`,
a VM bundle already on the Mac that boots it, as composing leaves one.

A macOS computer env is composed on a Mac, by a sandbox provider that boots macOS VM bundles there, `lume`: the base's
VM bundle is cloned under a name of the compose's key, booted with a build folder shared read-write, and readied for
agent-env (`vm_bundle/macos/prepare.sh`): `/opt/agentenv` for the layers; a pinned Python, python-build-standalone's
CPython 3.12 with Tk, checked by its sha256 (`vm_bundle/macos/lib.sh`), `python3` first on the `PATH` of what services
run, which their venvs and the probe's fixture run on; `agentenv-service`, which starts each
service with the deploy's environment; `set-display`, which puts the display in the virtual display's size at login,
since a guest boots in the mode its disk was last in, with the sRGB colour profile, so a screenshot holds the pixels as
drawn; and Cua Spacesd and the Login Items banner off. Then each layer, the driver's and each app's, installs as its
`[install.macos]` says, by a script composing writes (`env/envs/computer/utils/macos_vm_bundle.py`): its files under its
prefix, its downloads checked by sha256, its venv, installed offline from the wheels of its lock, its
run scripts and install steps, as root or the session user, its `grants`, the TCC permissions of an app bundle it
installs, bound to the bundle's code signature, and its services as LaunchAgents of the login session. A service runs
through `agentenv-service`, or, when `direct`, as its own command, with the variables it names, as an app bundle whose
TCC permissions macOS gives only to the executable launchd starts itself must. macOS has no apt: a layer's
`[install.macos]` names no `packages` or `sources`. The venvs' wheels are locked as a Linux layer's packages are: the
first compose of given inputs (each venv's requirements, the protocol's source, the base, the pinned Python and the
resolve script) resolves them once in a VM of the base, which builds every venv's wheels, its requirements and the
protocol in one resolve, into the folder it shares and is deleted; they're stored in the object store with a lock
naming them, whose digest joins the VM bundle's key. Every compose of those inputs, on any Mac, stages them and
installs offline, so one name is one set of wheels. Then the VM shuts down, is kept on that Mac as the VM bundle of its
name, and gate 3 probes it at full, a check its driver declares it fails on macOS (`known_failures`) recorded and not
failing the build; a compose of the same inputs on that Mac reuses it. A VM bundle is kept on the Mac that composed it
alone, so a deploy on a Mac that doesn't hold the one its env's record names composes it there first, from the same
inputs and so under the same name. A deploy that would compose another name, as a different agent-env does, refuses
before composing; `agent-env run` and `agent-env env build` on a Mac that doesn't hold the VM bundle compose it there
and record what they composed, so a deploy after them boots it. A VM bundle's compose and probe hold a lock of its name
on its Mac, which a deploy takes to look for it, so a deploy waits for one under way and boots none before its probe
passed; one that fails its probe is deleted.

A deploy clones the env's VM bundle, writes the deploy's environment into a folder the VM shares as
`agentenv_protocol.computer.seed`'s seed, which `agentenv-service` reads (waiting for the share, which mounts at login),
boots it, and fronts the driver's port with a relay on this Mac's loopback, run by Apple's `/usr/bin/python3`, since
macOS's Local Network privacy lets only Apple's own binaries reach a VM's address; agent-env reaches the computer at
`127.0.0.1:<port>`. The relay listens on loopback alone unless `relay_host` names another address, since a computer's
MCP endpoint takes no token. A reset clones it afresh under its MAC address, so it keeps its address and its URLs.
Virtualization.framework runs a VM only in a logged-in session, so on a Mac with no one at it (an EC2 Mac), the user
agent-env runs as logs in at boot and agent-env runs in that session (`launchctl asuser`). A Mac runs two macOS VMs at
once, as Apple's licence allows and Virtualization.framework enforces; `lume` refuses a third. Its `resolution` setting
sizes the virtual display.

This bundle has no tasks of its own: other bundles pull its computers by alias, and `agent-env run computers` has
nothing to run.
