# The egress proxy

`egress` is the `ubuntu-24.04` base computer's explicit proxy, the way every session program goes out. The session's
programs read the universe's time (the clock library in `../clock/`), so a client that checks a site's certificate with
its own TLS, Chromium, Java or an official Node build, refuses a live site once the clock is months back or ahead. The
proxy runs on real time and hands such a client a certificate that holds at the universe's date when it must.

It is one static Go program. The base's Dockerfile builds it in a stage of its own, after `go vet` and its tests, so no
Go toolchain is left in the image. `go.sum` pins its modules, and Dependabot's `gomod` entry keeps them current.

## What runs where

- `egress boot` runs once, as root, in the base's init (`../start-desktop`), before root's supervisord starts:
  - it mints the computer's CA once, valid from 1950 to 9999 as routed sites' certificates are, its key in
    `/etc/agentenv/egress/` and readable by `agentenv-egress` alone;
  - it trusts the CA in the system store, agent-env's trust files (`/etc/agentenv/ca.pem`, `ca-bundle.pem`) and the
    session user's NSS database, and fails the boot naming any of these it can't;
  - it trusts the CA in every certifi bundle and every JDK's store an app brings, naming any that can't take it among
    its warnings, and the computer boots;
  - its report is `/var/log/agentenv-root/egress-boot.log`.
- `egress serve` runs as `agentenv-egress` under root's supervisord (`../root.conf`), on `127.0.0.1:3128`. The clock
  library leaves that user on real time.
- The session's supervisord (`../supervisord.conf`), and a login shell through `agentenv-egress.sh`, point
  `HTTP(S)_PROXY`, `ALL_PROXY`, Java's proxy properties (`JAVA_TOOL_OPTIONS`) and Node's `fetch`
  (`NODE_USE_ENV_PROXY`) at it.
- agent-env's own servers in the computer, the driver and its apps' servers, run on real time and send their own
  transfers to the object store directly (`agentenv_protocol.transfers.direct`); the programs they start keep the
  proxy.

## What it does with a connection

- **Plain HTTP** (an absolute URL): the client's own head goes on with its request line and the proxy-only headers
  rewritten, then bytes are relayed both ways. An upgrade keeps its `Connection` and `Upgrade` headers; any other
  request closes once answered.
- **`CONNECT`**: the destination's name is resolved, and the connection is refused unless every address is global
  (`notGlobal` lists the blocks that aren't). A routed site's name, in the hosts file's `# agentenv sites` block, goes
  to the site proxy there. Bytes that aren't TLS, and a routed site's TLS, are relayed untouched.
- **TLS, with the clock unarmed**: relayed untouched.
- **TLS, with the clock armed**: each of the name's addresses is decided by its own probe, a handshake the proxy makes
  with the client's own ClientHello, cached per address and SNI. An address already found working is tried first; one
  that doesn't answer is passed over.
  - The probe's site verifies, and every universe a client may check against is more than `Margin` (8 days) inside the
    dates its verified chain shares: **passed through** untouched.
  - The probe's handshake doesn't finish (a site asking for the client's certificate, say): **passed through**, for
    the client to answer.
  - Otherwise: **intercepted**. The site is verified against real time and the image's public roots, and the client is
    handed a leaf for its names from the computer's CA, or from a CA trusted nowhere when the site doesn't verify. The
    proxy's own handshake with the site uses uTLS, built from the client's ClientHello, so the site sees the client's
    fingerprint and ALPN offer, and h2 runs end to end.

The universes a client may check against are now, the universe at the clock's anchor, and, for `Hold` (30 s) after a
change, the universe the clock left. The universe is read from the clock page the follower writes (`clock.go`);
`testdata/` holds pages the follower's writer wrote, which both sides' tests read.

## Its log

One JSON line per connection, on stdout, which root's supervisord keeps in `/var/log/agentenv-root/egress.log`, out of
the session user's reach. Its fields, each there when it applies:

| Field | What it is |
|---|---|
| `t` | When the line was written, in seconds |
| `client`, `method`, `host`, `port` | Who asked, how, and for what |
| `allowed`, `dst` | `global` or `routed`, and the address dialled |
| `mode` | `pass`, `splice`, `intercept`, `http`, `upgrade` or `refused` |
| `decision` | `real`, `inside`, `outside`, `unverified`, `probe_failed`, `routed` or `not TLS` |
| `sni`, `alpn_offered`, `alpn` | The client's server name and protocols, and the one the site picked |
| `universes`, `upstream_valid` | The universes checked, and the verified chain's shared dates, in seconds |
| `probed`, `probe_error` | Whether this connection probed, and why a probe failed |
| `verified`, `upstream_verify`, `upstream_tls`, `minted` | An interception's upstream: verified or why not, its version and cipher, a new leaf |
| `upstream_ms`, `handshake_ms` | Milliseconds to the upstream, and to the client's handshake |
| `error` | Why the connection ended early |

## Testing it

```bash
make egress-test   # gofmt, go vet and go test -race, in the Go image the Dockerfile builds with
```

It needs Docker, not Go. CI runs it as the `egress` job. `tst/integration/computer/egress_test.py` runs the proxy in
a computer: the session user's reach, the refused destinations, Java and Node 182 days back, and handshakes through week
jumps.
