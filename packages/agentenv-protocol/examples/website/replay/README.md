# Widget Docs: a captured site, replayed

Widget Docs is a small docs site captured to a WARC file and served by
`agentenv_protocol.website.replay.ReplayWebsite`. The captured pages are served as they were captured, for any host,
with only their same-site links rewritten to stay on the replay. The newsletter sign-up, which a capture can't answer, is written as `@api` routes on top.

| File | What it is |
| --- | --- |
| `source/` | The site that was captured: a home page, two guide pages, a stylesheet and a script. |
| `capture.warc.gz` | The capture of `source/` at `--depth 1`: `/`, `/guide/install.html`, `/guide/usage.html?utm_source=home`, `/style.css` and `/app.js`. `/guide/advanced.html` is two links from home, so it isn't in the capture. |
| `Dockerfile`, `env.toml` | The server image, built from the repo root as the Notes example's is, and the website env as a bundle entry. |
| `site.py` | The server: `WidgetDocs(ReplayWebsite)` with `capture = "capture.warc.gz"`, plus `POST /api/subscribe` and `GET /api/subscriptions`. |

## Capture a site

```bash
python -m agentenv_protocol.website.replay capture https://docs.example.com/ --out capture.warc.gz --depth 2
```

The tool needs only this package and `httpx`. It fetches each start URL. It then fetches the same-site pages those URLs
link to, up to `--depth` links away, and the assets (stylesheets, scripts, images) each captured page loads, with the
stylesheets, images and fonts a captured stylesheet's `@import` and `url()` name. Every
answer goes into a WARC 1.1 file, gzipped one record at a time, which pywb and warcio also read. Notes:

- Only GETs are captured.
- Redirects are kept as they were answered, and their targets are captured too.
- Bodies are stored decoded, so the stored headers leave out `Content-Encoding` and `Transfer-Encoding`.
- A host is the same site as a start URL when it is that URL's host (without `www.`) or a subdomain of it.
- Request records name only the method, the target and `Host`, so no credential the capturing client sent ends up in
  the file.
- `--max-requests` (500 by default) caps a capture, and `--no-assets` leaves out what pages load.

To capture `source/` again:

```bash
python -m http.server -d source 8765 &
python -m agentenv_protocol.website.replay capture http://127.0.0.1:8765/ --out capture.warc.gz --depth 1
```

## How a request is matched

A request is matched by its method (HEAD as GET), its path and its query, never by its host or headers. The query is
normalised before matching:

- Its parameters are sorted.
- Blank parameters are kept.
- The `utm_*`, `fbclid`, `gclid`, `msclkid` and `_` parameters are dropped. Set `ignore_query` on the class to change
  this list.

Each path segment is percent-decoded and then encoded again, so `%7E` and `~` match. A request the capture doesn't
hold gets a 404. If a capture holds one request twice, the first answer is served.

When the capture's answers are served:

- `Set-Cookie` loses its `Domain` attribute, so the cookie works on the replay's own host.
- A `Location` that points at a captured host is made relative.
- In HTML pages (`href`, `src`, `srcset`, `action`, `formaction`, `poster`, `style` and `<style>`) and stylesheets
  (`url()`, `@import`), a URL on a captured host is made root-relative, and a root-relative URL is led by the prefix
  the replay is served under (`X-Forwarded-Prefix`, such as `/website/<name>`). Script bodies aren't changed.
- `Strict-Transport-Security` and `Alt-Svc` are dropped.

## Synthetic routes on top

`@api` routes win over the capture. Here `POST /api/subscribe` keeps sign-ups in a list of the site's own. Its
`@get_data` exports them through `data/get`, which is how a run's sign-ups are read; `data/reset` empties them through
`reset_site_data`, and a `data/add` data part's `subscriptions` are added through `add_site_data`. The capture itself
isn't task state. A `data/add` file part named `*.warc` or `*.warc.gz` (or typed `application/warc`)
replaces the declared capture until the next `data/reset`. This lets a large capture be loaded at deploy time instead
of being built into the image.

## Run and test it

```bash
PYTHONPATH=<repo>/packages/agentenv-protocol/src python site.py      # serves on PORT, 8000 by default
python -m agentenv_protocol.website.replay serve capture.warc.gz --domain docs.widgets.test   # a bare replay
python -m pytest packages/agentenv-protocol/tests/test_website_replay.py
docker build -f packages/agentenv-protocol/examples/website/replay/Dockerfile -t widget-docs .   # from the repo root
```
