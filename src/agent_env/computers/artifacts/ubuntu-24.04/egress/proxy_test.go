package main

import (
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
	if dst, why, err := p.destination("one.one.one.one", 443); err != nil || dst != "1.1.1.1:443" || why != "global" {
		t.Fatalf("a public name: %q %q %v", dst, why, err)
	}
	p.Resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("1.1.1.1"), netip.MustParseAddr("10.0.0.1")}, nil
	}
	if _, _, err := p.destination("split.example", 443); err == nil || !strings.Contains(err.Error(), "10.0.0.1") {
		t.Fatalf("a name with one private address: %v", err)
	}
	p.Routed = func() map[string]string { return map[string]string{"notes.test": "127.0.0.1"} }
	if dst, why, err := p.destination("Notes.Test.", 443); err != nil || dst != "127.0.0.1:443" || why != "routed" {
		t.Fatalf("a routed site: %q %q %v", dst, why, err)
	}
}

// harness is a proxy in front of the site server, which it reaches as example.com, and a client whose clock is at
// clientTime.
type harness struct {
	t      *testing.T
	site   *httptest.Server
	proxy  *Proxy
	url    *url.URL
	ca     signer
	rows   []map[string]any
	rowsMu sync.Mutex
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
	transport := &http.Transport{Proxy: http.ProxyURL(h.url), ForceAttemptHTTP2: http2,
		TLSClientConfig: &tls.Config{RootCAs: roots, Time: func() time.Time { return clientTime }}}
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
