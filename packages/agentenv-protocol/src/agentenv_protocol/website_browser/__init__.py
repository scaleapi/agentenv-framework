"""WebsiteBrowserEnvironment: a website browser as a computer app, run inside a computer or alone. A subclass says how
its build installs and starts; this class adds what agentenv owns, the same in both: its flags after the subclass's
command, the unattended-use policies, a CDP connection over the browser's debugging pipe, the settings every browser
takes, a fresh cache and the deploy's prompt answers at every launch, another window for a launch while it runs, its
skill and ``browser/v1`` on its card. It runs alone on a computer with no display, a browser computer's, whose screen
is its page, headful on an X server that computer gives it alone, else headless; which site proxy it goes through comes
from a variable agentenv sets, never from its settings."""
from .control import (ACTIVE_CHECK_TIMEOUT_S, AGENT, AGENTENV, CAPTURE_TIMEOUT_S, DOWNLOADS, EXIT, FAILED,
                      HASH_CHUNK_BYTES, LIVE_START_TAB_TIMEOUT_S, LOADED, PAGE_INPUT, PAGES, PIN_TIMEOUT_S, STOP,
                      TIMED_OUT, UNKNOWN, UNNAVIGATED_CAPTURE_TIMEOUT_S, UPLOADS, BrowserControl)
from .devtools import MAX_MESSAGE_BYTES, PIPE_IN, PIPE_OUT, DevToolsError, DevToolsPipe
from .environment import (ALONE_BYPASS, BLOCK, BROWSER, COMPUTER_BYPASS, OWNED_FLAGS, POLICIES, READY_POLL_S,
                          SHM_MIN_BYTES, SITE_PROXY_ENV, SITES, SKILL, BrowserSettings, WebsiteBrowserEnvironment,
                          Window, start_tab_why)
from .shapes import (BrowserRecord, ClosedTab, Cookie, CookieToSet, Dialog, Download, LaunchEntry, OpenTab,
                     OriginStorage, PartitionKey, SiteNames, StartTab, StartTabsLaunch, StorageItem, origin_of)

__all__ = ["ACTIVE_CHECK_TIMEOUT_S", "AGENT", "AGENTENV", "ALONE_BYPASS", "BLOCK", "BROWSER", "CAPTURE_TIMEOUT_S",
           "COMPUTER_BYPASS", "DOWNLOADS", "EXIT", "FAILED", "HASH_CHUNK_BYTES", "LIVE_START_TAB_TIMEOUT_S", "LOADED",
           "MAX_MESSAGE_BYTES", "OWNED_FLAGS", "PAGE_INPUT", "PAGES", "PIN_TIMEOUT_S", "PIPE_IN", "PIPE_OUT",
           "POLICIES", "READY_POLL_S", "SHM_MIN_BYTES", "SITE_PROXY_ENV", "SITES", "SKILL", "STOP", "TIMED_OUT",
           "UNKNOWN", "UNNAVIGATED_CAPTURE_TIMEOUT_S", "UPLOADS", "BrowserControl", "BrowserSettings", "DevToolsError",
           "DevToolsPipe", "WebsiteBrowserEnvironment", "Window", "BrowserRecord", "ClosedTab", "Cookie", "CookieToSet",
           "Dialog", "Download", "LaunchEntry", "OpenTab", "OriginStorage", "PartitionKey", "SiteNames", "StartTab",
           "StartTabsLaunch", "StorageItem", "origin_of", "start_tab_why"]
