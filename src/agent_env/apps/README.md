The computer apps that come with agent-env, today `libreoffice`, a `computer_app` env a computer env lists in its
`apps`. Name it from another bundle as `libreoffice`, the alias `aliases.toml` gives it, or by its id,
`@local/agentenv-framework/apps/libreoffice`.

`envs/libreoffice/` is LibreOffice for an X11 desktop on Linux (`amd64` and `arm64`): Writer, Impress and Calc from
Ubuntu's apt, with the GTK3 interface, at the versions a put's package lock pins, and fonts with the metrics of
Office's, Carlito for Calibri, Caladea for Cambria and Liberation for Arial, Times New Roman and Courier New, so Office
files keep their layout.

- **Unattended use:** its install step writes `agentenv.xcd` into `/etc/libreoffice/registry/`, a layer of LibreOffice's
  configuration under the session user's profile: no tip of the day, no first-run, new-version, "Get involved" or
  donation bar, no question whether to keep an Office format on save, no lock files beside documents, and no document
  recovery after the app was stopped around a load or a reset.
- **Launch:** on demand, through its server, the Start Center first (`soffice --nologo`), ready once a LibreOffice
  window shows. Every LibreOffice launcher on the computer runs `open.py`: the install step writes each of LibreOffice's
  desktop entries into `/usr/local/share/applications/`, which takes precedence, the same but for that. With no copy
  running, `open.py` has the server launch one and waits until it's ready, then hands the files to it, as LibreOffice
  runs one copy per user, so the server starts every copy and a load or a reset can stop it. A file the file manager or
  `exo-open` opens goes there by its type: a .docx, .doc, .odt or .rtf to Writer, a .pptx, .ppt or .odp to Impress, an
  .xlsx, .xls, .ods or .csv to Calc, a CSV comma-separated in UTF-8 with no import dialog. A launch while it runs shows a
  Start Center in the copy that runs.
- **Data:** its profile, `~/.config/libreoffice`, which the image has none of, so each deploy and each reset starts with
  a clean one, under the same defaults. Documents are the computer's own, in Documents and Downloads.
- **Skill:** `SKILL.md`, for an agent working through its windows: opening each app, saving in Office formats, slides,
  animations and formulas.

The `office` example's desktop composes it with the `ubuntu-24.04` base and the `reference` driver, and
`tst/integration/computer/libreoffice_app_test.py` deploys it.
