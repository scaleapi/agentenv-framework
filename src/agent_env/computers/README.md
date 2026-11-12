Base computers that come with agent-env, each a `computer` artifact a computer env composes a driver and apps onto.
`agent-env run <bundle> --set envs/desktop.computer=ubuntu-24.04-browser` composes a bundle's `desktop` env on
another of them for one run, the bundle left as written.

`artifacts/ubuntu-24.04/` is Ubuntu 24.04 (`amd64` and `arm64`) with XFCE on Xvfb, D-Bus and AT-SPI. tini starts
supervisord, which runs every program as the session user, `user` (uid 1000); nothing in it runs as root. Its home
has a desktop's standard folders (`Desktop`, `Documents`, `Downloads`, `Pictures` and the rest, from `xdg-user-dirs`). Its
container form's image is built from the folder's `Dockerfile`, and a release publishes it as
`ghcr.io/scaleapi/agentenv-ubuntu-24.04:src-<hex>`, tagged by its sources, so a run pulls it by digest instead of
building it. Name it from another bundle as `ubuntu-24.04`, the alias `aliases.toml` gives it, or by its id,
`@local/agentenv-framework/computers/ubuntu-24.04`.

`artifacts/ubuntu-24.04-browser/` is Ubuntu 24.04 (`amd64` and `arm64`) with no desktop (`display = "none"`), for one
website browser and the `page` driver, whose screen is the browser's page: a browser computer, which agentenv's
`chromium` deploys alone as. The browser runs headful on an X server of its own, Xvfb on `:99` at 1920x1080, which no
agent sees, records or drives, so the sites it visits see a browser with a window, not a headless one; the page's size
is the browser's `window` setting. supervisord runs every program as the session user, among them that X server and a
stand-in site proxy on `127.0.0.1:3128`, the browser's only route out when the computer
routes no sites. It forwards a request or a `CONNECT` tunnel to any host, never decrypting a tunnel and refusing
nothing: the sandbox's network policy is all that limits it. Every line it writes to `/var/log/agentenv/site-proxy.log`
is JSON, one per connection and one per connection that fails, in the same shapes as the shared site proxy's. It
keeps the browser's network log too, in `/var/log/agentenv/network.jsonl`, which `AGENTENV_NETWORK_LOG` names: an
`open` and an `end` line for each connection, numbered, in the shared site proxy's shapes. A computer that routes sites names the shared site proxy's forward port,
`127.0.0.1:3129`, to the browser instead: it serves the run's sites, refuses every other host and runs out of the agent's reach. It has a
container form only. Name it from another bundle as `ubuntu-24.04-browser`, or by its id,
`@local/agentenv-framework/computers/ubuntu-24.04-browser`. It was `ubuntu-24.04-headless`, a name nothing resolves
any more, so what still names it fails rather than gets a headful browser; its image, `agentenv-ubuntu-24.04-headless`,
stays published for what pins it.

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
before reset in place started is refused a reset, to be redeployed. It boots on local Docker where Docker can pass it
`/dev/kvm`, the disk image's files mounted read-only into the launcher where Docker mounts images (28 and later, on the
containerd image store), so no copy of a disk, nor of a Windows disk's base, is made for each launcher, else copied into
an image of the launcher it keeps for the next boot. There its computer uses a local object store's grants as a container
does, at `host.docker.internal`: Linux's init and Windows' setup map that name to 10.0.2.2, QEMU's address for the
launcher, which forwards the grant server's port to this machine (`VM_HOST_PORTS`), and the local transfer CA's trust
files ride in the seed (`VM_TRUST`), where the init and agentenv-session point TLS clients at them; and on `modal_vm`, amd64 only, whose sandbox runs the launcher as
its image under nested virtualization: the provider exports the disk from the image store with this machine's Docker,
compressed, once for each composed disk, into the per-user cache, and mounts it on the sandbox at create, which Modal
uploads once for each content and the launcher expands onto the sandbox's disk; a thin disk, a Windows one, carries its
base too, exported once for each base and expanded beside it there, so another computer on that base adds its layer
alone. Such a sandbox is ended once it has been idle, with no connection open, for the provider's `idle_timeout`, 30
minutes unless its config sets another, as a running QEMU isn't activity. A gate 3 probe that ends with the connection
to the driver dropped, refused or timed out runs again on a fresh boot, up to three in all, and the record keeps each
attempt; one with a failed check is recorded as it is.

A put, `env build` and the release compose every form; a run composes only the forms its sandbox's deploys boot: on each
link of a chain, the first form it boots natively, as a deploy picks, so local Docker composes containers, and a sandbox
that boots only disks composes disks. A run composes them on the env's default provider, this machine unless the config
names another, never on the sandbox it deploys on. A compose of a disk also composes the container it's made from,
unless that's built. A deploy that boots a form no compose has started, a disk named by `env_params` included, composes
it first on that same provider, saying which, reusing what's built, as every build is reused by its key; deploys of one
env at once compose it once, the others
waiting for it, a deploy that finds it building included. A container built but left unverified though the deploy's
sandbox runs it natively, as one is when a setting its probe boots with doesn't resolve, is composed again the same way:
its image reused and probed, and booted only once ready; a deploy that still has no ready build names why each such
container wasn't probed. In `agent-env run` it says why and each stage as it goes, and
a deploy waiting for another's compose says what it waits for and since when. A local run doesn't compose a registered
env's form at deploy where a configured store isn't the per-user one, since the run's fence refuses its record: the
deploy refuses for want of a ready build and names `agent-env env build`.

Each build's record names the digest of the code that composed it: the composer, these build files, the Lume provider
and the protocol's `computer` package. A build an agent-env with other composing code recorded, or one from before
records named it, isn't deployed: a deploy that boots its form composes it again first, saying so, as `env build` and
a put do, and its record says `by another composer` until then. So upgrading agent-env never boots a computer composed
the old way.

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
services as programs beside the driver's, run as `user` like everything else. The computer's own card is named
`<name>-driver`, and the app's, which it lists, keeps the app's name, so a universe's environment of that name loads
into the app and `snapshot_env` keeps both. A deploy waits for the computer to be ready up to 180 s, or 30 s more than
the slowest `boot` app's own `timeout_s`.

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
prefix, its downloads checked by sha256, its venv, on a copy of the pinned Python so its interpreter is its own,
installed offline from the wheels of its lock, its run scripts and install steps, as root or the session user, its
`grants`, the TCC permissions of an app bundle it installs (`app`, granted to the bundle's identifier) or of an
executable under its prefix (`executable`, granted to that path), bound to the code signature it has then, and its
services as LaunchAgents of the login session. A service runs through `agentenv-service` on the layer's own Python, or,
when `direct`, as its own command, with the variables it names. macOS gives a process the TCC permissions of the program
launchd started, and what that program starts has them too, so an app bundle's service is direct, and a grant to
`venv/bin/python` covers the layer's other services and what they run. macOS has no apt: a layer's
`[install.macos]` names no `packages` or `sources`. The venvs' wheels are locked as a Linux layer's packages are: the
first compose of given inputs (each venv's requirements, the protocol's source, the base, the pinned Python and the
resolve script) resolves them once in a VM of the base, which builds every venv's wheels, its requirements and the
protocol in one resolve, into the folder it shares and is deleted; they're stored in the object store with a lock
naming them, whose digest joins the VM bundle's key. Every compose of those inputs, on any Mac, stages them and
installs offline, so one name is one set of wheels. Then the VM shuts down, is kept on that Mac as the VM bundle of its
name (one that hasn't shut down within two minutes is synced, stopped from outside and kept all the same, what its guest
logged of the shutdown saved under the state root and warned of), and gate 3 probes it at full, a check its driver declares it fails on macOS (`known_failures`) recorded and not
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

`artifacts/windows-11/` is Windows 11 Enterprise (`amd64`), Microsoft's evaluation, as a disk. Windows isn't
agent-env's to publish, so no image of it is: the first compose on a Linux machine with KVM fetches the evaluation ISO
from Microsoft's link into the per-user cache, once, printing one line that says so, and installs it unattended in the
QEMU launcher (`disk/windows/`): `autounattend.xml` answers setup, with the licence accepted, a UEFI disk and two local
accounts, `agentadmin`, whom agent-env composes as over ssh with a key of the base's own, and `user`, the session user
the desktop signs in as at every boot; `setup.ps1` runs once at `agentadmin`'s first sign-in, OpenSSH first, so a
setup that fails is read back over ssh at once (`setup-failed`), then turns off what would get in an agent's way
(updates, the lock screen, notifications, Widgets, Copilot, first-run pages, the page file and the firewall, behind
QEMU's network, which reaches the guest at the ports the launcher forwards alone), installs the pinned Python with Tk
and `agentenv-session`, and switches the sign-in to `user`. Setup takes about 13 minutes. The installed disk, trimmed,
becomes the base's image, `disk/base.img` in a layer of its own under an empty overlay of it, with the date its 90-day
evaluation expires and the key to `agentadmin`; it's pushed to the env's image store, as Linux disks
are, under a tag of its recipe, so another machine composing on it pulls it rather than installing Windows, and fetches
no ISO. A compose installs a base afresh once fewer than 7 days of its evaluation are left, from the ISO Microsoft's
link serves then. Name it as `windows-11`, or by its id, `@local/agentenv-framework/computers/windows-11`.

A Windows computer env is composed on the same machine as a thin disk (`env/envs/computer/utils/thin_disk.py`, with
what's Windows' own in `windows_disk.py`): the base booted on an overlay kept in a folder of its own, each layer
installed over ssh as `agentadmin` by a PowerShell script composing writes, as its `[install.windows]` says (its files
under `C:\agentenv\drivers\<name>`, or an app's under `C:\agentenv\apps\<name>`, its downloads checked by sha256, its
venv on the pinned Python, its PowerShell run scripts, elevated, an app's install steps, elevated too, and its
services), then shut down and packed onto the base's image, so every computer composed on a base shares one copy of it,
and pushed to the env's image store under its key, pinned, as a Linux disk is; a compose of the same inputs on any
machine reuses it, and a deploy pulls it. Gate 3 probes it at full. At the session user's sign-in, `agentenv-session`,
run windowless by `pythonw`, reads the deploy's environment from the NoCloud seed, which the launcher attaches as a
removable USB drive, since Windows mounts a FAT filesystem with no partition table on removable media alone, and starts
each layer's services in that session, again whenever one exits. A Windows computer boots with 4 vCPUs and 8 GB, on
local Docker with KVM and on `modal_vm`; it has computer-exec and data/add, through the protocol's Windows defaults, so
the probe runs its fixture there as on Linux, with Python one-liners for the commands it would give a shell. On
`modal_vm` it runs in your own Modal workspace, under the evaluation's terms or a licence of your own; whether they
allow it there is yours to check. Composing leaves `C:\agentenv` the admins', as `/opt/agentenv` is root's on Linux:
every user, and every sandboxed process (Chrome's, as Program Files allows it), reads and runs what's there, and the
session user writes only the services' logs and each app's `.agentenv` folder. An app's own server runs in the session
as its services do, on a loopback port it writes with a token to that folder, which only the session user, SYSTEM and
Administrators reach; it launches the app in a Job object, which ends whatever the app started. Composing signs in as
`agentadmin`, so an app's install step for the session user can't run on Windows, and gate 1 refuses an app that
declares one there.

`artifacts/omarchy-4.5/` is Omarchy 4.5 (`amd64`), an Arch Linux desktop on Hyprland, as a disk alone: Hyprland needs a
GPU, which a container doesn't get. Its base is built from Omarchy's own stable VM image, which its `[forms.disk]` names
by `source`, a URL and a sha256, fetched once into this machine's cache and checked. It's booted once by the QEMU
launcher as a `linux-desktop` guest (a virtio GPU at the computer's resolution, 4 vCPUs and 8 GB), with root's ssh key
passed as a systemd credential, Omarchy signing its user `omarchy` in as it does at every boot, and made a base by
`disk/omarchy/recipe.sh`, run as root over ssh. The recipe refuses an image whose os-release isn't the artifact's
`omarchy` `4.5.0`; adds `agentenv.lua` to Hyprland's config (cua-driver's Hyprland plugin loaded at start, a US keymap,
the idle cursor hidden, the display's mode); installs the seed unit; has SDDM log back in after a logout; sets Omarchy's
toggles for an unattended desktop (staying awake, no animations, no screensaver or suspend, notifications silenced);
turns off crash toasts, dictation and its keys, housekeeping timers that would run at boot, and the firewall, which lets
in only ssh and LocalSend while the launcher forwards only the computer's own ports; builds the plugin; and locks
`omarchy`'s password and leaves the machine id empty. The plugin is the one cua-driver's release at the driver's
Hyprland pin ships, built from its source by that release's kit, unmodified, as `omarchy` with `makepkg`, each archive
pinned by sha256. The kit builds only on what it pins, Arch's Hyprland 0.56.2-1 and the GCC 16.1.1 it was built with,
where Omarchy 4.5 ships newer builds of both, so first the recipe installs those from that day of the Arch Linux
Archive (`disk/omarchy/arch-day.conf`, packages signature-checked by pacman), with the Hyprland libraries of that day,
and cmake and ninja, which it removes after the build. It ends the builder's desktop session first, as the kit says to
install it, and only then adds the drop-in: Hyprland reloads its config whenever a file of it changes, and the one
running there can neither load the plugin nor show, without hanging, the error a config naming a missing one makes. Trimmed, the image becomes the base's, with the key to root, and is pushed to the env's image store
under a tag of its recipe and source. Name it as `omarchy-4.5`, or by its id,
`@local/agentenv-framework/computers/omarchy-4.5`.

An Omarchy computer env is composed on the same machine as a thin disk too, with what's Omarchy's own in
`omarchy_disk.py`: each layer is installed over ssh as root by a bash script composing writes, as its
`[install."linux/omarchy"]` says (its files under `/opt/agentenv/drivers/<name>`, its downloads checked by sha256, its
venv on the system Python, its run scripts), and its services become systemd user units of the graphical session, which
start with the session, and again with it, through `with-seed`. The composed disk has no ssh: only a compose logs
in, and it boots the base. At every boot, before SDDM starts the session,
`agentenv-seed.service` reads the deploy's environment from the NoCloud seed with the Linux disk's own `seed-env` and
writes it for `with-seed`, root's and the session user's alone, the seed's trust files first; it makes the folder an
app's server marks its launches in (`/run/agentenv/apps`) the session user's; it sets the hostname and
the computer's `timezone` (UTC by default; a deploy is refused one that isn't an IANA zone, and one the disk doesn't
have is written as the refusal every service then names), names `host.docker.internal` at QEMU's address for the launcher, and writes the display's mode for Hyprland
from the computer's `resolution`. Gate 3 probes it at full. Each app is a layer of its own beside the driver, under
`/opt/agentenv/apps/<name>`, installed as its `[install."linux/omarchy"]` says: its venv with the protocol's `app`
extra, its own `.agentenv/` folder the session user's alone, its run scripts and install steps as their scopes say,
and its server and services as user units beside the driver's. Gate 1 refuses an app asking for `shm_mb`, and a
driver or app whose Linux install names apt packages, which Omarchy has no apt for. When
Hyprland crashes, its watchdog brings it back in safe mode, without cua-driver's plugin, and cua-driver reports the
computer failed for good.

This bundle has no tasks of its own: other bundles pull its computers by alias, and `agent-env run computers` has
nothing to run.
