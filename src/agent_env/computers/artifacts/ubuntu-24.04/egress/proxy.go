package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Proxy struct {
	Leaves   *Leaves
	Universe interface {
		View(time.Time) (bool, []float64)
	}
	// Roots are the public roots a site is verified against, as the image had them before any agentenv CA joined.
	Roots *x509.CertPool
	// Resolve, Allowed, Routed and Dial default to the system's resolver, a global address, the hosts file's agentenv
	// sites block and a TCP dial; tests replace them.
	Resolve func(ctx context.Context, host string) ([]netip.Addr, error)
	Allowed func(netip.Addr) bool
	Routed  func() map[string]string
	Dial    func(dst string) (net.Conn, error)
	Log     func(row map[string]any)
	probes  probes
}

func serveForever() error {
	leaves, err := LoadLeaves(Folder)
	if err != nil {
		return fmt.Errorf("the CA (has boot run?): %w", err)
	}
	roots, err := os.ReadFile(PublicRoots)
	if err != nil {
		return err
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(roots) {
		return fmt.Errorf("no certificates in %s", PublicRoots)
	}
	universe := NewUniverse(ClockPage)
	go universe.Poll()
	ln, err := net.Listen("tcp", Listen)
	if err != nil {
		return err
	}
	p := &Proxy{Leaves: leaves, Universe: universe, Roots: pool, Log: jsonLines(os.Stdout)}
	p.Log(map[string]any{"event": "serving", "listen": Listen, "uid": os.Getuid(), "pid": os.Getpid()})
	return p.Serve(ln)
}

func jsonLines(w io.Writer) func(map[string]any) {
	var mu sync.Mutex
	encoder := json.NewEncoder(w)
	return func(row map[string]any) {
		row["t"] = float64(time.Now().UnixMilli()) / 1e3
		mu.Lock()
		defer mu.Unlock()
		encoder.Encode(row)
	}
}

func (p *Proxy) Serve(ln net.Listener) error {
	for {
		conn, err := ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		go p.handle(conn)
	}
}

func (p *Proxy) log(row map[string]any) {
	if p.Log != nil {
		p.Log(row)
	}
}

func (p *Proxy) handle(conn net.Conn) {
	start := time.Now()
	row := map[string]any{"client": conn.RemoteAddr().String()}
	defer func() {
		if r := recover(); r != nil {
			row["error"] = fmt.Sprint("panic: ", r)
			p.log(row)
		}
		conn.Close()
	}()
	br := bufio.NewReaderSize(conn, HeadLimit)
	conn.SetReadDeadline(start.Add(HeadTimeout))
	method, target, version, headers, err := readHead(br)
	if err != nil {
		row["error"] = truncate("request: "+err.Error(), ErrorLimit)
		p.log(row)
		return
	}
	row["method"] = method
	if method == http.MethodConnect {
		host, port := splitAuthority(target, 443)
		row["host"], row["port"] = host, port
		dsts, why, err := p.destination(host, port)
		if err != nil {
			row["mode"], row["error"] = "refused", err.Error()
			reply(conn, "403 Forbidden", "egress: "+err.Error()+"\n")
			p.log(row)
			return
		}
		row["allowed"] = why
		if _, err := io.WriteString(conn, "HTTP/1.1 200 Connection established\r\n\r\n"); err != nil {
			return
		}
		p.tunnel(conn, br, host, port, dsts, why, row, start)
		return
	}
	u, err := url.Parse(target)
	if err != nil || u.Scheme != "http" || u.Host == "" {
		row["mode"], row["error"] = "refused", "not an absolute http URL"
		reply(conn, "400 Bad Request", "egress: CONNECT, or an absolute http:// URL\n")
		p.log(row)
		return
	}
	host, port := splitAuthority(u.Host, 80)
	row["host"], row["port"], row["mode"] = host, port, "http"
	dsts, why, err := p.destination(host, port)
	if err != nil {
		row["mode"], row["error"] = "refused", err.Error()
		reply(conn, "403 Forbidden", "egress: "+err.Error()+"\n")
		p.log(row)
		return
	}
	row["allowed"] = why
	up, dst, err := p.dialAny(dsts)
	if err != nil {
		row["error"] = truncate(err.Error(), ErrorLimit)
		reply(conn, "502 Bad Gateway", "egress: "+err.Error()+"\n")
		p.log(row)
		return
	}
	defer up.Close()
	row["dst"] = dst
	upgrade := false
	for _, h := range headers {
		if headerName(h) == "upgrade" {
			upgrade = true
		}
	}
	// The client's own head, but for its request line, which names the path alone, and the headers meant for this
	// proxy; a request that isn't an upgrade closes once answered, so its connection can't carry one for another site.
	var head strings.Builder
	head.WriteString(method + " " + u.RequestURI() + " " + version + "\r\n")
	for _, h := range headers {
		switch headerName(h) {
		case "proxy-connection", "proxy-authorization":
			continue
		case "connection", "keep-alive":
			if !upgrade {
				continue
			}
		}
		head.WriteString(h + "\r\n")
	}
	if upgrade {
		row["mode"] = "upgrade"
	} else {
		head.WriteString("Connection: close\r\n")
	}
	head.WriteString("\r\n")
	conn.SetReadDeadline(time.Time{})
	if _, err := io.WriteString(up, head.String()); err != nil {
		row["error"] = truncate("upstream: "+err.Error(), ErrorLimit)
		p.log(row)
		return
	}
	p.log(row)
	// Both ways at once: a body the client sends once told to continue, or an upgraded connection's frames.
	go func() {
		io.Copy(up, br)
		closeWrite(up)
	}()
	io.Copy(conn, up)
}

// readHead is a request's head, read through its blank line: its request line's parts and its header lines.
func readHead(r *bufio.Reader) (string, string, string, []string, error) {
	var lines []string
	size := 0
	for {
		line, err := r.ReadString('\n')
		size += len(line)
		if size > HeadLimit {
			return "", "", "", nil, errors.New("request head too long")
		}
		if err != nil {
			return "", "", "", nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if len(lines) == 0 {
				continue
			}
			break
		}
		lines = append(lines, line)
	}
	parts := strings.SplitN(lines[0], " ", 3)
	if len(parts) != 3 {
		return "", "", "", nil, fmt.Errorf("malformed request line %q", truncate(lines[0], 100))
	}
	return parts[0], parts[1], parts[2], lines[1:], nil
}

func headerName(line string) string {
	name, _, _ := strings.Cut(line, ":")
	return strings.ToLower(strings.TrimSpace(name))
}

// destination is the addresses to dial for host:port, in the order to try them, and why they're allowed: "routed" for
// a name the hosts file's agentenv sites block maps (the site proxy, as a client that bypasses this proxy reaches it),
// "global" for a name whose every address is global, IPv4 first. Anything else is refused.
func (p *Proxy) destination(host string, port int) ([]string, string, error) {
	host = strings.TrimSuffix(strings.ToLower(strings.Trim(host, "[]")), ".")
	routed := p.Routed
	if routed == nil {
		routed = routedSites
	}
	if address, ok := routed()[host]; ok {
		return []string{net.JoinHostPort(address, strconv.Itoa(port))}, "routed", nil
	}
	resolve := p.Resolve
	if resolve == nil {
		resolve = func(ctx context.Context, host string) ([]netip.Addr, error) {
			return net.DefaultResolver.LookupNetIP(ctx, "ip", host)
		}
	}
	allowed := p.Allowed
	if allowed == nil {
		allowed = isGlobal
	}
	ctx, cancel := context.WithTimeout(context.Background(), DialTimeout)
	defer cancel()
	addresses, err := resolve(ctx, host)
	if err != nil {
		return nil, "", fmt.Errorf("%s doesn't resolve: %w", host, err)
	}
	if len(addresses) == 0 {
		return nil, "", fmt.Errorf("%s has no address", host)
	}
	var private []string
	var v4, v6 []string
	for _, a := range addresses {
		a = a.Unmap()
		dst := netip.AddrPortFrom(a, uint16(port)).String()
		switch {
		case !allowed(a):
			private = append(private, a.String())
		case a.Is4():
			v4 = append(v4, dst)
		default:
			v6 = append(v6, dst)
		}
	}
	if len(private) > 0 {
		return nil, "", fmt.Errorf("%s:%d isn't public (%s)", host, port, strings.Join(private, ", "))
	}
	return append(v4, v6...), "global", nil
}

// notGlobal are the special-purpose blocks (IANA's registries) no global address is in.
var notGlobal = func() []netip.Prefix {
	var out []netip.Prefix
	for _, s := range []string{
		"0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
		"192.0.0.0/24", "192.0.2.0/24", "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
		"224.0.0.0/4", "240.0.0.0/4",
		"::/128", "::1/128", "64:ff9b:1::/48", "100::/64", "2001::/23", "2001:db8::/32", "fc00::/7", "fe80::/10",
		"ff00::/8",
	} {
		out = append(out, netip.MustParsePrefix(s))
	}
	return out
}()

func isGlobal(a netip.Addr) bool {
	a = a.Unmap()
	if !a.IsValid() {
		return false
	}
	for _, block := range notGlobal {
		if block.Contains(a) {
			return false
		}
	}
	return true
}

// routedSites is the hosts file's agentenv sites block, name to address: root's, so the session user can't add to it.
func routedSites() map[string]string {
	out := map[string]string{}
	data, err := os.ReadFile(Hosts)
	if err != nil {
		return out
	}
	inside := false
	for _, line := range strings.Split(string(data), "\n") {
		text := strings.TrimSpace(line)
		switch {
		case text == HostsBegin:
			inside = true
		case text == HostsEnd:
			inside = false
		case inside && text != "" && !strings.HasPrefix(text, "#"):
			fields := strings.Fields(text)
			for _, name := range fields[1:] {
				out[strings.ToLower(name)] = fields[0]
			}
		}
	}
	return out
}

func (p *Proxy) tunnel(conn net.Conn, br *bufio.Reader, host string, port int, dsts []string, why string,
	row map[string]any, start time.Time) {
	wait := FirstBytes
	if port == 443 {
		wait = HelloTimeout
	}
	conn.SetReadDeadline(time.Now().Add(wait))
	first, err := br.Peek(1)
	if err != nil && !isTimeout(err) {
		row["error"] = "closed before sending"
		p.log(row)
		return
	}
	if len(first) == 0 || first[0] != recordHandshake || why == "routed" {
		row["mode"], row["decision"] = "pass", "not TLS"
		if len(first) > 0 && first[0] == recordHandshake {
			row["mode"], row["decision"] = "splice", "routed"
		}
		if !p.splice(conn, br, dsts, row, start) {
			p.log(row)
		}
		return
	}
	conn.SetReadDeadline(time.Now().Add(HelloTimeout))
	hello, err := clientHello(br, time.Now().Add(HelloTimeout))
	if err != nil {
		row["error"] = truncate("ClientHello: "+err.Error(), ErrorLimit)
		p.log(row)
		return
	}
	sni, offered, err := parseHello(hello)
	if err != nil {
		row["error"] = truncate("ClientHello: "+err.Error(), ErrorLimit)
		p.log(row)
		return
	}
	row["sni"], row["alpn_offered"] = sni, offered
	verifyName := sni
	if verifyName == "" {
		verifyName = strings.Trim(host, "[]")
	}
	armed, universes := p.Universe.View(time.Now())
	if !armed {
		row["decision"], row["mode"] = "real", "splice"
		if !p.splice(conn, br, dsts, row, start) {
			p.log(row)
		}
		return
	}
	row["universes"] = rounded(universes)
	// Each address in turn, by its own probe: one that doesn't answer is passed over, so the connection goes to a
	// server whose own certificate decided it.
	var unreachable []string
	for _, dst := range p.preferChecked(dsts, sni) {
		delete(row, "probe_error")
		delete(row, "upstream_valid")
		seen, probed := p.probeAt(dst, sni, verifyName, hello)
		if seen.unreachable {
			unreachable = append(unreachable, seen.problem)
			continue
		}
		row["probed"] = probed
		decision := "outside"
		switch {
		case seen.failed:
			// a handshake only the client can finish, as one asking for its certificate: the client decides
			decision, row["probe_error"] = "probe_failed", seen.problem
		case !seen.verified:
			decision, row["probe_error"] = "unverified", seen.problem
		case inside(seen.from, seen.until, universes):
			decision = "inside"
		}
		if seen.verified {
			row["upstream_valid"] = []int64{seen.from.Unix(), seen.until.Unix()}
		}
		row["decision"] = decision
		var dialled bool
		if decision == "inside" || decision == "probe_failed" {
			row["mode"] = "splice"
			dialled = p.splice(conn, br, []string{dst}, row, start)
		} else {
			row["mode"] = "intercept"
			dialled = p.intercept(conn, br, dst, sni, verifyName, hello, row, start)
		}
		if dialled {
			return
		}
		// it stopped answering since its probe: forget it, and go on to the next
		p.forget(dst, sni)
		unreachable = append(unreachable, fmt.Sprint(row["error"]))
		delete(row, "error")
	}
	row["error"] = truncate("upstream: no address answered: "+strings.Join(unreachable, "; "), ErrorLimit)
	p.log(row)
}

// preferChecked is dsts with those whose fresh probe answered first, each group in its own order, so an address
// already found working isn't kept waiting behind one whose probe has to be made again.
func (p *Proxy) preferChecked(dsts []string, sni string) []string {
	p.probes.mu.Lock()
	defer p.probes.mu.Unlock()
	var checked, rest []string
	for _, dst := range dsts {
		if hit, ok := p.probes.cache[dst+"|"+sni]; ok && hit.fresh() && !hit.unreachable {
			checked = append(checked, dst)
		} else {
			rest = append(rest, dst)
		}
	}
	return append(checked, rest...)
}

// splice relays the connection to the first of dsts that answers, untouched; false, with the error in row, when none
// does.
func (p *Proxy) splice(conn net.Conn, br *bufio.Reader, dsts []string, row map[string]any, start time.Time) bool {
	up, dst, err := p.dialAny(dsts)
	if err != nil {
		row["error"] = truncate("upstream: "+err.Error(), ErrorLimit)
		return false
	}
	defer up.Close()
	row["dst"] = dst
	conn.SetReadDeadline(time.Time{})
	row["upstream_ms"] = time.Since(start).Milliseconds()
	p.log(row)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(up, br)
		closeWrite(up)
	}()
	io.Copy(conn, up)
	closeWrite(conn)
	wg.Wait()
	return true
}

// intercept hands the client a leaf of this computer's CA and relays the plaintext to dst over a handshake of the
// proxy's own; false, with the error in row, when dst doesn't answer.
func (p *Proxy) intercept(conn net.Conn, br *bufio.Reader, dst, sni, verifyName string, hello []byte,
	row map[string]any, start time.Time) bool {
	uc, found, err := p.handshake([]string{dst}, sni, verifyName, hello)
	var unanswered dialError
	if errors.As(err, &unanswered) {
		row["error"] = truncate("upstream: "+err.Error(), ErrorLimit)
		return false
	}
	if err != nil {
		row["error"] = truncate("upstream: "+err.Error(), ErrorLimit)
		p.log(row)
		return true
	}
	defer uc.NetConn().Close()
	row["dst"] = dst
	row["alpn"], row["verified"], row["upstream_tls"] = found.ALPN, found.Verified, []string{found.Version, found.Cipher}
	row["upstream_ms"] = time.Since(start).Milliseconds()
	if len(found.Dropped) > 0 {
		row["dropped_extensions"] = found.Dropped
	}
	var sans []string
	if found.Verified {
		// the site's own names, as its leaf lists them, so the client pools connections as it would with the site's
		// certificate
		for _, name := range found.SANs {
			if sni == "" || strings.HasPrefix(name, "DNS:") {
				sans = append(sans, name)
			}
		}
		if sni == "" {
			sans = append(sans, "IP:"+strings.Trim(hostOf(dst), "[]"))
		}
	} else {
		row["upstream_verify"] = truncate(found.VerifyErr, 200)
	}
	cert, minted, err := p.Leaves.For(verifyName, sans, found.Verified)
	if err != nil {
		row["error"] = truncate("leaf: "+err.Error(), ErrorLimit)
		p.log(row)
		return true
	}
	row["minted"] = minted
	config := &tls.Config{Certificates: []tls.Certificate{*cert}, SessionTicketsDisabled: true}
	if found.ALPN != "" {
		config.NextProtos = []string{found.ALPN}
	}
	client := tls.Server(&peeked{Conn: conn, r: br}, config)
	conn.SetDeadline(time.Now().Add(HandshakeTimeout))
	if err := client.Handshake(); err != nil {
		row["error"] = truncate("client handshake: "+err.Error(), ErrorLimit)
		p.log(row)
		return true
	}
	conn.SetDeadline(time.Time{})
	row["handshake_ms"] = time.Since(start).Milliseconds()
	p.log(row)
	done := make(chan struct{}, 2)
	go func() {
		io.Copy(uc, client)
		done <- struct{}{}
	}()
	go func() {
		io.Copy(client, uc)
		done <- struct{}{}
	}()
	<-done
	client.Close()
	uc.NetConn().Close()
	<-done
	return true
}

// peeked is a connection whose reads come through the reader that peeked its first bytes.
type peeked struct {
	net.Conn
	r *bufio.Reader
}

func (c *peeked) Read(b []byte) (int, error) { return c.r.Read(b) }

func closeWrite(c net.Conn) {
	if cw, ok := c.(interface{ CloseWrite() error }); ok {
		cw.CloseWrite()
	}
}

func isTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

func reply(conn net.Conn, status, body string) {
	fmt.Fprintf(conn, "HTTP/1.1 %s\r\nContent-Type: text/plain\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s",
		status, len(body), body)
}

func splitAuthority(authority string, port int) (string, int) {
	host, p, err := net.SplitHostPort(authority)
	if err != nil {
		return strings.Trim(authority, "[]"), port
	}
	n, err := strconv.Atoi(p)
	if err != nil {
		return host, port
	}
	return host, n
}

func hostOf(hostport string) string {
	host, _, err := net.SplitHostPort(hostport)
	if err != nil {
		return hostport
	}
	return host
}

func rounded(values []float64) []int64 {
	out := make([]int64, len(values))
	for i, v := range values {
		out[i] = int64(v)
	}
	return out
}
