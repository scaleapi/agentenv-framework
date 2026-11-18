// egress: the computer's explicit egress proxy, on real time, for the session's programs, which run on the universe's.
//
// It runs as its own user, agentenv-egress, a uid the clock library leaves on real time, under root's supervisord, and
// serves CONNECT and absolute-form HTTP on 127.0.0.1:3128, or where AGENTENV_EGRESS_PROXY names. Session programs
// reach it through HTTP(S)_PROXY and ALL_PROXY, or a proxy of the base's own that hands it their connections; nothing
// redirects or filters, so it needs no NET_ADMIN and works on any provider.
//
// A TLS connection is spliced (the client's bytes relayed untouched) when the clock is unarmed, or when every universe
// a client may check against lies inside the site's real certificate by more than Margin either side. Otherwise it is
// intercepted: the site is verified against real time and the public roots, and the client is handed a leaf from this
// computer's CA, valid over every date the clock can show. The upstream handshake is uTLS's, built from the client's
// own ClientHello, so the site sees the client's fingerprint and ALPN offer either way.
//
//	egress boot    as root at the computer's start: the CA, minted once, its key this user's alone, trusted everywhere
//	egress serve   as agentenv-egress: serve, one JSON line per connection on stdout
package main

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"
)

const (
	Listen         = "127.0.0.1:3128"
	ProxyEnv       = "AGENTENV_EGRESS_PROXY"
	EgressUser     = "agentenv-egress"
	SessionUser    = "user"
	Folder         = "/etc/agentenv/egress"
	PublicCA       = "/etc/agentenv/egress-ca.pem"
	PublicRoots    = "/usr/local/lib/agentenv-egress/roots.pem"
	ClockPage      = "/dev/shm/agentenv-clock"
	Hosts          = "/etc/hosts"
	HostsBegin     = "# agentenv sites"
	HostsEnd       = "# end agentenv sites"
	CAName         = "agentenv egress CA"
	NSSNickname    = "agentenv egress"
	Hold           = 30 * time.Second
	Margin         = 8 * 24 * time.Hour
	ProbeTTL       = time.Hour
	ProbeFailedTTL = time.Minute
	FirstBytes     = 300 * time.Millisecond
	HelloTimeout   = 10 * time.Second
	DialTimeout    = 10 * time.Second

	// How long a client has to send its request's head, and how long either handshake of an interception, the
	// client's with the proxy and the proxy's with the site, has to finish.
	HeadTimeout      = 15 * time.Second
	HandshakeTimeout = 15 * time.Second

	// The longest request head a client may send, and the most of an error a log row keeps, in bytes.
	HeadLimit  = 64 << 10
	ErrorLimit = 300

	// How far inside the CA's dates each leaf it signs is, so no leaf outlasts its CA.
	LeafInset = 2 * time.Hour

	// How often the clock page is read, and how many reads may find it mid-write before the last universe stands.
	PagePoll  = 20 * time.Millisecond
	PageReads = 1000
)

// The dates the CA and every leaf it signs are valid between: the widest a certificate carries, as routed sites'
// (sites.UNIVERSE_FROM and UNIVERSE_UNTIL in agent-env's site routing), so they hold at any date the clock can show.
var (
	CAFrom  = time.Date(1950, 1, 1, 0, 0, 0, 0, time.UTC)
	CAUntil = time.Date(9999, 12, 31, 23, 59, 59, 0, time.UTC)
)

func main() {
	command := ""
	if len(os.Args) > 1 {
		command = os.Args[1]
	}
	switch command {
	case "boot":
		report, err := boot(defaultBootPaths())
		out, _ := json.Marshal(report)
		fmt.Println(string(out))
		if err != nil {
			fmt.Fprintln(os.Stderr, "egress boot:", err)
			os.Exit(1)
		}
	case "serve":
		listen, err := serveListen(os.Args[2:], os.Getenv(ProxyEnv))
		if err != nil {
			fmt.Fprintln(os.Stderr, "egress serve:", err)
			os.Exit(2)
		}
		if err := serveForever(listen); err != nil {
			fmt.Fprintln(os.Stderr, "egress serve:", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "usage: egress boot|serve, not %q\n", command)
		os.Exit(2)
	}
}

// serveListen is the address serve listens on: the host and port of named, the URL a base whose own proxy holds Listen
// gives its proxies for this one in ProxyEnv, else Listen.
func serveListen(args []string, named string) (string, error) {
	if len(args) != 0 {
		return "", fmt.Errorf("usage: egress serve, not %q", args)
	}
	if named == "" {
		return Listen, nil
	}
	u, err := url.Parse(named)
	if err != nil || u.Scheme != "http" || u.Hostname() == "" || u.Port() == "" || strings.Trim(u.Path, "/") != "" {
		return "", fmt.Errorf("%s is http://HOST:PORT, not %q", ProxyEnv, named)
	}
	return u.Host, nil
}
