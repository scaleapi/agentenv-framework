-- agent-env's Hyprland drop-in, required last from hyprland.lua: cua-driver's plugin, loaded as Hyprland starts, as it
-- must be; a US keymap, the only one the plugin's keys take; the idle cursor hidden, so a capture leaves it out; and the
-- display's mode, which agentenv-seed writes each boot from the computer's resolution.
hl.env("GDK_SCALE", "1")
hl.plugin.load("/usr/lib/cua/hyprland/cua-hyprland-plugin.so")
hl.config({
  plugin = { cua = { enabled = true } },
  cursor = { inactive_timeout = 1 },
  input = { kb_rules = "evdev", kb_model = "pc105", kb_layout = "us", kb_variant = "", kb_options = "",
            numlock_by_default = false },
})
pcall(require, "hypr.agentenv-display")
