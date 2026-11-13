package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

type fixedUniverse struct {
	armed     bool
	universes []float64
}

func (f fixedUniverse) View(time.Time) (bool, []float64) { return f.armed, f.universes }

func TestOnlyDestinationsWhoseEveryAddressIsGlobalAreDialled(t *testing.T) {
	refused := []string{"10.0.0.1", "10.255.255.254", "172.16.0.1", "172.31.255.1", "192.168.1.1", "127.0.0.1",
		"127.1.2.3", "0.0.0.0", "169.254.169.254", "169.254.1.1", "100.64.0.1", "100.127.255.254", "192.0.2.1",
		"198.18.0.1", "224.0.0.1", "255.255.255.255", "::1", "::", "fd00::1", "fe80::1", "::ffff:10.0.0.1",
		"2001:db8::1"}
	if len(refused) != 22 {
		t.Fatal(len(refused))
	}
	p := &Proxy{Routed: func() map[string]string { return nil }}
	for _, address := range refused {
		p.Resolve = func(context.Context, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr(address)}, nil
		}
		if _, _, err := p.destination("somewhere.example", 443); err == nil {
			t.Errorf("%s dialled", address)
		}
	}
	p.Resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("2606:4700::1111"), netip.MustParseAddr("1.1.1.1")}, nil
	}
	dst, why, err := p.destination("one.one.one.one", 443)
	if err != nil || strings.Join(dst, " ") != "1.1.1.1:443 [2606:4700::1111]:443" || why != "global" {
		t.Fatalf("a public name, IPv4 first: %q %q %v", dst, why, err)
	}
	p.Resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("1.1.1.1"), netip.MustParseAddr("10.0.0.1")}, nil
	}
	if _, _, err := p.destination("split.example", 443); err == nil || !strings.Contains(err.Error(), "10.0.0.1") {
		t.Fatalf("a name with one private address: %v", err)
	}
	p.Routed = func() map[string]string { return map[string]string{"notes.test": "127.0.0.1"} }
	if dst, why, err := p.destination("Notes.Test.", 443); err != nil || strings.Join(dst, " ") != "127.0.0.1:443" ||
		why != "routed" {
		t.Fatalf("a routed site: %q %q %v", dst, why, err)
	}
}

// harness is a proxy in front of the site server, which it reaches as example.com, and a client whose clock is at
// clientTime.
type harness struct {
	t          *testing.T
	site       *httptest.Server
	proxy      *Proxy
	url        *url.URL
	ca         signer
	clientCert *tls.Certificate
	rows       []map[string]any
	rowsMu     sync.Mutex
}

func newHarness(t *testing.T, site *httptest.Server, universe fixedUniverse, trustSite bool) *harness {
	t.Helper()
	ca, err := mintCA(CAName)
	if err != nil {
		t.Fatal(err)
	}
	leaves, err := NewLeaves(ca)
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if trustSite && site.Certificate() != nil {
		roots.AddCert(site.Certificate())
	}
	h := &harness{t: t, site: site, ca: ca}
	h.proxy = &Proxy{
		Leaves: leaves, Universe: universe, Roots: roots,
		Resolve: func(context.Context, string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
		},
		Allowed: func(netip.Addr) bool { return true },
		Routed:  func() map[string]string { return nil },
		Log: func(row map[string]any) {
			h.rowsMu.Lock()
			h.rows = append(h.rows, row)
			h.rowsMu.Unlock()
		},
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go h.proxy.Serve(ln)
	h.url, _ = url.Parse("http://" + ln.Addr().String())
	return h
}

func (h *harness) get(scheme string, clientTime time.Time, http2 bool) (*http.Response, string, error) {
	roots := x509.NewCertPool()
	if h.site.Certificate() != nil {
		roots.AddCert(h.site.Certificate())
	}
	roots.AddCert(h.ca.cert)
	config := &tls.Config{RootCAs: roots, Time: func() time.Time { return clientTime }}
	if h.clientCert != nil {
		config.Certificates = []tls.Certificate{*h.clientCert}
	}
	transport := &http.Transport{Proxy: http.ProxyURL(h.url), ForceAttemptHTTP2: http2, TLSClientConfig: config}
	defer transport.CloseIdleConnections()
	_, port, _ := net.SplitHostPort(h.site.Listener.Addr().String())
	resp, err := (&http.Client{Transport: transport, Timeout: 10 * time.Second}).Get(
		fmt.Sprintf("%s://example.com:%s/hello", scheme, port))
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return resp, string(body), err
}

func (h *harness) lastRow() map[string]any {
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		h.rowsMu.Lock()
		n := len(h.rows)
		h.rowsMu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	h.rowsMu.Lock()
	defer h.rowsMu.Unlock()
	if len(h.rows) == 0 {
		h.t.Fatal("no row logged")
	}
	return h.rows[len(h.rows)-1]
}

func site(t *testing.T, http2 bool) *httptest.Server {
	s := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "hello over %s", r.Proto)
	}))
	s.EnableHTTP2 = http2
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func TestOnRealTimeTheClientsBytesReachTheSiteUntouched(t *testing.T) {
	s := site(t, false)
	h := newHarness(t, s, fixedUniverse{}, true)
	resp, body, err := h.get("https", time.Now(), false)
	if err != nil || body != "hello over HTTP/1.1" {
		t.Fatalf("%q %v", body, err)
	}
	if !resp.TLS.PeerCertificates[0].Equal(s.Certificate()) {
		t.Fatal("the client didn't see the site's own certificate")
	}
	if row := h.lastRow(); row["mode"] != "splice" || row["decision"] != "real" {
		t.Fatalf("%v", row)
	}
}

func TestAUniverseInsideTheSitesCertificateIsSplicedAndOneOutsideIntercepted(t *testing.T) {
	s := site(t, true)
	now := float64(time.Now().Unix())
	h := newHarness(t, s, fixedUniverse{armed: true, universes: []float64{now, now}}, true)
	resp, _, err := h.get("https", time.Now(), true)
	if err != nil || !resp.TLS.PeerCertificates[0].Equal(s.Certificate()) {
		t.Fatalf("inside: %v", err)
	}
	if row := h.lastRow(); row["mode"] != "splice" || row["decision"] != "inside" {
		t.Fatalf("%v", row)
	}

	// httptest's certificate ends in 2084: a universe in 2090 is outside it.
	later := time.Date(2090, 6, 1, 0, 0, 0, 0, time.UTC)
	u := float64(later.Unix())
	h = newHarness(t, s, fixedUniverse{armed: true, universes: []float64{u, u}}, true)
	resp, body, err := h.get("https", later, true)
	if err != nil || body != "hello over HTTP/2.0" {
		t.Fatalf("outside: %q %v", body, err)
	}
	leaf := resp.TLS.PeerCertificates[0]
	if leaf.CheckSignatureFrom(h.ca.cert) != nil || !leaf.NotAfter.After(later) || leaf.NotBefore.Year() != 1950 {
		t.Fatalf("the client's leaf: issuer %s, %s to %s", leaf.Issuer, leaf.NotBefore, leaf.NotAfter)
	}
	row := h.lastRow()
	if row["mode"] != "intercept" || row["decision"] != "outside" || row["alpn"] != "h2" || row["verified"] != true {
		t.Fatalf("%v", row)
	}
}

func TestASiteThatDoesntVerifyIsHandedOnUnderACATrustedNowhere(t *testing.T) {
	s := site(t, false)
	u := float64(time.Date(2090, 6, 1, 0, 0, 0, 0, time.UTC).Unix())
	h := newHarness(t, s, fixedUniverse{armed: true, universes: []float64{u}}, false)
	_, _, err := h.get("https", time.Now(), false)
	if err == nil || !strings.Contains(err.Error(), "certificate") {
		t.Fatalf("the client took it: %v", err)
	}
	if row := h.lastRow(); row["decision"] != "unverified" {
		t.Fatalf("%v", row)
	}
}

func TestPlainHTTPInAbsoluteFormIsForwarded(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "plain %s %s ua=%q", r.Method, r.URL.Path, r.UserAgent())
	}))
	t.Cleanup(s.Close)
	h := newHarness(t, s, fixedUniverse{}, false)
	_, body, err := h.get("http", time.Now(), false)
	if err != nil || !strings.HasPrefix(body, "plain GET /hello") {
		t.Fatalf("%q %v", body, err)
	}
}

func TestARefusedDestinationIsAnswered403(t *testing.T) {
	s := site(t, false)
	h := newHarness(t, s, fixedUniverse{}, true)
	h.proxy.Allowed = isGlobal
	_, _, err := h.get("https", time.Now(), false)
	if err == nil || !strings.Contains(err.Error(), "Forbidden") {
		t.Fatalf("%v", err)
	}
	if row := h.lastRow(); row["mode"] != "refused" {
		t.Fatalf("%v", row)
	}
}

func TestEveryCheckedAddressIsTriedInTurn(t *testing.T) {
	s := site(t, false)
	h := newHarness(t, s, fixedUniverse{}, true)
	h.proxy.Resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("127.0.0.2"), netip.MustParseAddr("127.0.0.1")}, nil
	}
	_, body, err := h.get("https", time.Now(), false)
	if err != nil || body != "hello over HTTP/1.1" {
		t.Fatalf("%q %v", body, err)
	}
	if row := h.lastRow(); !strings.HasPrefix(fmt.Sprint(row["dst"]), "127.0.0.1:") {
		t.Fatalf("%v", row)
	}
}

func TestAnUpgradeOverPlainHTTPRelaysBothWays(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Upgrade") != "echo" || !strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade") {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		conn, rw, _ := w.(http.Hijacker).Hijack()
		defer conn.Close()
		rw.WriteString("HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: echo\r\n\r\n")
		rw.Flush()
		line, _ := rw.ReadString('\n')
		rw.WriteString(line)
		rw.Flush()
	}))
	t.Cleanup(s.Close)
	h := newHarness(t, s, fixedUniverse{}, false)
	_, port, _ := net.SplitHostPort(s.Listener.Addr().String())
	conn, err := net.Dial("tcp", h.url.Host)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(5 * time.Second))
	fmt.Fprintf(conn, "GET http://example.com:%s/ws HTTP/1.1\r\nHost: example.com:%s\r\nConnection: Upgrade\r\n"+
		"Upgrade: echo\r\n\r\n", port, port)
	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, nil)
	if err != nil || resp.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("%v %v", resp, err)
	}
	fmt.Fprint(conn, "ping\n")
	if line, err := br.ReadString('\n'); err != nil || line != "ping\n" {
		t.Fatalf("%q %v", line, err)
	}
}

func TestAnUploadThatWaitsFor100ContinueGoesThroughAtOnce(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		fmt.Fprint(w, len(body))
	}))
	t.Cleanup(s.Close)
	h := newHarness(t, s, fixedUniverse{}, false)
	_, port, _ := net.SplitHostPort(s.Listener.Addr().String())
	req, _ := http.NewRequest("POST", "http://example.com:"+port+"/up", bytes.NewReader(make([]byte, 100000)))
	req.Header.Set("Expect", "100-continue")
	transport := &http.Transport{Proxy: http.ProxyURL(h.url), ExpectContinueTimeout: 10 * time.Second}
	defer transport.CloseIdleConnections()
	started := time.Now()
	resp, err := (&http.Client{Transport: transport, Timeout: 20 * time.Second}).Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(body) != "100000" || time.Since(started) > 5*time.Second {
		t.Fatalf("%q after %s", body, time.Since(started))
	}
}

func TestEveryCertificateOfTheVerifiedChainBoundsPassingThrough(t *testing.T) {
	year := func(y int) time.Time { return time.Date(y, 1, 1, 0, 0, 0, 0, time.UTC) }
	leaf := &x509.Certificate{NotBefore: year(2020), NotAfter: year(2030)}
	intermediate := &x509.Certificate{NotBefore: year(2019), NotAfter: year(2025)}
	root := &x509.Certificate{NotBefore: year(2000), NotAfter: year(2040)}
	from, until := chainWindow([][]*x509.Certificate{{leaf, intermediate, root}})
	if !from.Equal(year(2020)) || !until.Equal(year(2025)) {
		t.Fatalf("%s to %s", from, until)
	}
	if inside(from, until, []float64{float64(year(2027).Unix())}) {
		t.Fatal("a universe past the intermediate's end passed through")
	}
}

func TestASiteAskingForTheClientsCertificateIsPassedThroughForTheClientToAnswer(t *testing.T) {
	s := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "hello %d certificate", len(r.TLS.PeerCertificates))
	}))
	s.TLS = &tls.Config{ClientAuth: tls.RequireAnyClientCert, MaxVersion: tls.VersionTLS12}
	s.StartTLS()
	t.Cleanup(s.Close)
	u := float64(time.Date(2090, 6, 1, 0, 0, 0, 0, time.UTC).Unix())
	h := newHarness(t, s, fixedUniverse{armed: true, universes: []float64{u}}, true)
	client, err := mintCA("client")
	if err != nil {
		t.Fatal(err)
	}
	h.clientCert = &tls.Certificate{Certificate: [][]byte{client.cert.Raw}, PrivateKey: client.key}
	_, body, err := h.get("https", time.Now(), false)
	if err != nil || body != "hello 1 certificate" {
		t.Fatalf("%q %v", body, err)
	}
	if row := h.lastRow(); row["mode"] != "splice" || row["decision"] != "probe_failed" {
		t.Fatalf("%v", row)
	}
}

func TestTheProbeAndLeafCachesStayWithinTheirLimit(t *testing.T) {
	limit := cacheLimit
	cacheLimit = 3
	defer func() { cacheLimit = limit }()
	ca, _ := mintCA(CAName)
	leaves, _ := NewLeaves(ca)
	for i := 0; i < 10; i++ {
		if _, _, err := leaves.For(fmt.Sprintf("site%d.test", i), nil, true); err != nil {
			t.Fatal(err)
		}
	}
	if len(leaves.cache) != 3 || leaves.cache["site9.test|"] == nil {
		t.Fatalf("%d leaves kept", len(leaves.cache))
	}
	p := &Proxy{Dial: func(string) (net.Conn, error) { return nil, fmt.Errorf("unreachable") }}
	for i := 0; i < 10; i++ {
		p.probeAt(fmt.Sprintf("192.0.2.%d:443", i), "", "x", goHello(t, "x", nil))
	}
	if len(p.probes.cache) != 3 {
		t.Fatalf("%d probes kept", len(p.probes.cache))
	}
}

func TestAConnectionTheProbeLetsThroughGoesToTheServerItProbed(t *testing.T) {
	s := site(t, false)
	now := float64(time.Now().Unix())
	h := newHarness(t, s, fixedUniverse{armed: true, universes: []float64{now}}, true)
	h.proxy.Resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("127.0.0.2"), netip.MustParseAddr("127.0.0.1")}, nil
	}
	var mu sync.Mutex
	var dialled []string
	h.proxy.Dial = func(dst string) (net.Conn, error) {
		mu.Lock()
		dialled = append(dialled, dst)
		mu.Unlock()
		return net.DialTimeout("tcp", dst, DialTimeout)
	}
	if _, _, err := h.get("https", time.Now(), false); err != nil {
		t.Fatal(err)
	}
	row := h.lastRow()
	mu.Lock()
	defer mu.Unlock()
	// the probe tried 127.0.0.2, then reached 127.0.0.1; the connection went straight to 127.0.0.1
	if row["decision"] != "inside" || len(dialled) != 3 || !strings.HasPrefix(dialled[2], "127.0.0.1:") {
		t.Fatalf("%v %v", row, dialled)
	}
	if hit := h.proxy.probes.cache[dialled[0]+"|example.com"]; !hit.unreachable || hit.verified {
		t.Fatalf("the address that didn't answer is cached as %+v", hit)
	}
}

func TestASlowProbeHoldsUpNoOtherSite(t *testing.T) {
	release := make(chan struct{})
	p := &Proxy{Dial: func(dst string) (net.Conn, error) {
		if strings.HasPrefix(dst, "192.0.2.1:") {
			<-release
		}
		return nil, fmt.Errorf("unreachable")
	}}
	hello := goHello(t, "x", nil)
	p.probeAt("192.0.2.2:443", "", "x", hello)
	go p.probeAt("192.0.2.1:443", "", "x", hello)
	time.Sleep(50 * time.Millisecond)
	started := time.Now()
	if _, probed := p.probeAt("192.0.2.2:443", "", "x", hello); probed || time.Since(started) > 100*time.Millisecond {
		t.Fatalf("a cached site waited %s behind a slow one (probed again: %v)", time.Since(started), probed)
	}
	close(release)
}

// standIns resolves example.com to its addresses and dials each to the site unless it's down; both may change
// between requests.
type standIns struct {
	mu        sync.Mutex
	addresses []string
	down      map[string]bool
}

func (s *standIns) set(down []string, addresses ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.addresses, s.down = addresses, map[string]bool{}
	for _, a := range down {
		s.down[a] = true
	}
}

func newStandIns(h *harness, addresses ...string) *standIns {
	s := &standIns{addresses: addresses, down: map[string]bool{}}
	h.proxy.Resolve = func(context.Context, string) ([]netip.Addr, error) {
		s.mu.Lock()
		defer s.mu.Unlock()
		var out []netip.Addr
		for _, a := range s.addresses {
			out = append(out, netip.MustParseAddr(a))
		}
		return out, nil
	}
	h.proxy.Dial = func(dst string) (net.Conn, error) {
		host, _, _ := net.SplitHostPort(dst)
		s.mu.Lock()
		down := s.down[host]
		s.mu.Unlock()
		if down {
			return nil, fmt.Errorf("%s is down", host)
		}
		return net.DialTimeout("tcp", h.site.Listener.Addr().String(), DialTimeout)
	}
	return s
}

func TestAnAddressThatStopsAnsweringAfterItsProbeGivesWayToTheNext(t *testing.T) {
	s := site(t, false)
	now := float64(time.Now().Unix())
	h := newHarness(t, s, fixedUniverse{armed: true, universes: []float64{now}}, true)
	addresses := newStandIns(h, "198.51.100.1", "198.51.100.2")
	if _, _, err := h.get("https", time.Now(), false); err != nil {
		t.Fatal(err)
	}
	if row := h.lastRow(); !strings.HasPrefix(fmt.Sprint(row["dst"]), "198.51.100.1:") {
		t.Fatalf("%v", row)
	}
	addresses.set([]string{"198.51.100.1"}, "198.51.100.1", "198.51.100.2")
	if _, _, err := h.get("https", time.Now(), false); err != nil {
		t.Fatal(err)
	}
	if row := h.lastRow(); !strings.HasPrefix(fmt.Sprint(row["dst"]), "198.51.100.2:") || row["decision"] != "inside" {
		t.Fatalf("%v", row)
	}
}

func TestAnAddressThatDidntAnswerLeavesANewAddressToItsOwnProbe(t *testing.T) {
	s := site(t, false)
	u := float64(time.Date(2090, 6, 1, 0, 0, 0, 0, time.UTC).Unix())
	h := newHarness(t, s, fixedUniverse{armed: true, universes: []float64{u}}, true)
	addresses := newStandIns(h)
	addresses.set([]string{"198.51.100.1"}, "198.51.100.1")
	if _, _, err := h.get("https", time.Date(2090, 6, 1, 0, 0, 0, 0, time.UTC), false); err == nil {
		t.Fatal("reached a site none of whose addresses answer")
	}
	addresses.set([]string{"198.51.100.1"}, "198.51.100.1", "198.51.100.2")
	if _, _, err := h.get("https", time.Date(2090, 6, 1, 0, 0, 0, 0, time.UTC), false); err != nil {
		t.Fatal(err)
	}
	row := h.lastRow()
	if !strings.HasPrefix(fmt.Sprint(row["dst"]), "198.51.100.2:") || row["mode"] != "intercept" || row["decision"] != "outside" {
		t.Fatalf("the new address wasn't judged by its own certificate: %v", row)
	}
}

func TestAClientHelloWhoseFieldsOverrunItIsRefusedNotPanickedOn(t *testing.T) {
	hello := goHello(t, "example.com", nil)
	p := 4 + 2 + 32
	p += 1 + int(hello[p])
	p += 2 + int(hello[p])<<8 | int(hello[p+1])
	bad := append([]byte{}, hello...)
	bad[p] = 0xff // compression methods' length, past the message's end
	bad = bad[:p+3]
	if _, _, err := parseHello(bad); err == nil {
		t.Fatal("parseHello took it")
	}
	if _, _, err := helloRecord(bad); err == nil {
		t.Fatal("helloRecord took it")
	}
}

func TestAProbeThatPanicsLeavesNoOneWaitingOnIt(t *testing.T) {
	p := &Proxy{Dial: func(string) (net.Conn, error) { panic("dial") }}
	hello := goHello(t, "x", nil)
	probe := func() (panicked bool) {
		defer func() { panicked = recover() != nil }()
		p.probeAt("192.0.2.9:443", "", "x", hello)
		return false
	}
	if !probe() {
		t.Fatal("the probe didn't panic")
	}
	done := make(chan bool)
	go func() { done <- probe() }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("a second probe waited on the one that panicked")
	}
}

func TestAnAddressAlreadyFoundWorkingIsTriedBeforeOnesNotYetProbed(t *testing.T) {
	p := &Proxy{}
	p.probes.cache = map[string]probe{
		"[2001:db8::1]:443|x": {verified: true, at: time.Now()},
		"192.0.2.1:443|x":     {unreachable: true, at: time.Now().Add(-2 * ProbeFailedTTL)},
		"192.0.2.3:443|x":     {unreachable: true, at: time.Now()},
	}
	got := p.preferChecked([]string{"192.0.2.1:443", "192.0.2.2:443", "192.0.2.3:443", "[2001:db8::1]:443"}, "x")
	want := []string{"[2001:db8::1]:443", "192.0.2.1:443", "192.0.2.2:443", "192.0.2.3:443"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("%v", got)
	}
}
