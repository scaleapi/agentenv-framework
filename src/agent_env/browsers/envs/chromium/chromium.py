"""agentenv's chromium app: Chrome for Testing at the build Playwright 1.64.1 pins, for amd64 and arm64 (Google ships
no Linux arm64 Chrome), a website browser whose flags, policies and settings agentenv's base class gives it, with the
flags and policies of its own that stop the requests this build makes of its own. Its data is
its profile, which the image has none of, so every deploy and every reset starts with a clean one."""
import subprocess
from pathlib import Path

from agentenv_protocol.computer import Download, Launch, Packages, Ready, computer_app, install_step, linux
from agentenv_protocol.website_browser import WebsiteBrowserEnvironment

VERSION = "156.0.8078.4"
_ARCHIVES = {"amd64": ("linux64/chrome-linux64.zip",
                       "43fd31766c7533a02dda2ded5ad50c2f861cd5400be7509c0abad71129f79d79"),
             "arm64": ("linux-arm64/chrome-linux-arm64.zip",
                       "910a12fff6170202195f72ef3d353801c7063d301ab80b1f51b77d791fda60ce")}
DOWNLOADS = [Download(url=f"https://storage.googleapis.com/chrome-for-testing-public/{VERSION}/{path}", sha256=sha256,
                      to=f"chromium/chrome-{arch}.zip", arch=arch)
             for arch, (path, sha256) in _ARCHIVES.items()]
#: What it links against that Ubuntu 24.04 lacks, a font for the pages it shows, and unzip for its install step.
PACKAGES = ["fonts-liberation", "libasound2t64", "libatk-bridge2.0-0t64", "libatk1.0-0t64", "libatspi2.0-0t64",
            "libcairo2", "libcups2t64", "libdbus-1-3", "libdrm2", "libexpat1", "libgbm1", "libglib2.0-0t64",
            "libgtk-3-0t64", "libnspr4", "libnss3", "libpango-1.0-0", "libvulkan1", "libx11-6", "libxcb1",
            "libxcomposite1", "libxdamage1", "libxext6", "libxfixes3", "libxkbcommon0", "libxrandr2", "libxshmfence1",
            "unzip"]
WINDOW_CLASS = "Chromium"
PROFILE = ".config/chromium"
#: The shared memory, in MiB, Chrome's renderers get in /dev/shm, so a heavy page doesn't fill Docker's 64 MiB.
SHM_MB = 1024
# What stops the requests this build makes of its own, found against VERSION behind a browser alone's proxy: check them
# again when VERSION moves. One is left, a Google account check at launch, which none of them stops.
#: Flags: Safe Browsing's list updates, among its other background services; and the component updater's checks, at
#: launch and a minute later, pointed at a file it can't fetch, since ComponentUpdatesEnabled leaves the components
#: Chrome deems critical, such as its CRLSets, checking in.
QUIET_FLAGS = ["--disable-background-networking", "--component-updater=url-source=file:///dev/null"]
#: Features off: the AI Mode eligibility check at launch, which no policy stops; and the preconnect to its search engine
#: at launch, which NetworkPredictionOptions would stop only by turning off the preconnects and prefetches it makes for
#: pages too.
DISABLED_FEATURES = ["AimEnabled", "PreconnectToSearch"]
#: Every language Chromium has a spellcheck dictionary for, from its spellcheck_common.cc: blocklisted, so it fetches
#: none for the deploy's locale at launch. SpellcheckEnabled false would void the blocklist and still fetch it.
SPELLCHECK_LANGUAGES = ["af", "bg", "ca", "cs", "cy", "da", "de", "de-DE", "el", "en", "en-AU", "en-CA", "en-GB",
                        "en-GB-oxendict", "en-US", "es", "es-419", "es-AR", "es-ES", "es-MX", "es-US", "et", "fa", "fo",
                        "fr", "fr-FR", "gl", "he", "hi", "hr", "hu", "hy", "id", "it", "it-IT", "ko", "lt", "lv", "nb",
                        "nl", "pl", "pt", "pt-BR", "pt-PT", "ro", "ru", "sh", "sk", "sl", "sq", "sr", "sv", "ta", "tg",
                        "tr", "uk", "vi"]
#: Policies, over agentenv's: the network time query, spellcheck dictionaries, and the autofill server's field
#: predictions for a page's forms, which agentenv's AutofillSettings leaves on in this build.
QUIET_POLICIES = {"BrowserNetworkTimeQueriesEnabled": False, "SpellcheckLanguageBlocklist": SPELLCHECK_LANGUAGES,
                  "AutofillAddressEnabled": False, "AutofillCreditCardEnabled": False}
COMMAND = ["chromium/chrome/chrome", "--no-sandbox", f"--class={WINDOW_CLASS}", *QUIET_FLAGS,
           f"--disable-features={','.join(DISABLED_FEATURES)}"]


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11", "none"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=PACKAGES), downloads=DOWNLOADS, paths=[PROFILE], shm_mb=SHM_MB,
       launch=Launch(at="boot", command=COMMAND, window={"class": WINDOW_CLASS}, ready=Ready(window=True)))
class Chromium(WebsiteBrowserEnvironment):
    """Chrome for Testing without its sandbox, which a container can't give it, and with none of the requests it makes
    of its own but one: a subclass that declares its own launch keeps ``QUIET_FLAGS`` and ``DISABLED_FEATURES`` in its
    command."""

    version = VERSION
    #: Where Chrome for Testing reads managed policies; it reads neither Chromium's folder nor Chrome's.
    policy_dir = Path("/etc/opt/chrome_for_testing/policies/managed")
    policies = QUIET_POLICIES
    profile = PROFILE

    @install_step(scope="system", os="linux")
    def chromium(self) -> None:
        unpack(Path.cwd())


def unpack(folder: Path) -> None:
    """Unpack the arch's archive under ``folder``'s ``chromium/`` to ``chromium/chrome/``, with unzip, which keeps its
    execute bits, in its place."""
    [archive] = (folder / "chromium").glob("chrome-*.zip")
    unpacked = folder / "chromium" / "unpacked"
    subprocess.run(["unzip", "-q", str(archive), "-d", str(unpacked)], check=True)
    [top] = unpacked.iterdir()
    top.rename(folder / "chromium" / "chrome")
    unpacked.rmdir()
    archive.unlink()
