package main

import (
	"crypto/x509"
	"fmt"
	"net"
	"sync"
	"time"

	utls "github.com/refraction-networking/utls"
)

// upstream is what a handshake with a site found: the protocol it picked, its leaf's validity and names, and whether
// that leaf verifies against real time and the public roots.
type upstream struct {
	ALPN      string
	Verified  bool
	VerifyErr string
	NotBefore time.Time
	NotAfter  time.Time
	SANs      []string
	Version   string
	Cipher    string
	Dropped   []uint16
}

// handshake is uTLS's handshake with the site at dst, built from the client's own ClientHello, so the site sees the
// client's fingerprint and ALPN offer. Records go full-size from the start and the connection closes without a
// close_notify, as Chromium's BoringSSL does.
func (p *Proxy) handshake(dst, sni, verifyName string, hello []byte) (*utls.UConn, upstream, error) {
	var found upstream
	record, gone, err := helloRecord(hello)
	if err != nil {
		return nil, found, err
	}
	found.Dropped = gone
	spec, err := (&utls.Fingerprinter{AllowBluntMimicry: true}).FingerprintClientHello(record)
	if err != nil {
		return nil, found, fmt.Errorf("fingerprint: %w", err)
	}
	raw, err := p.dial(dst)
	if err != nil {
		return nil, found, err
	}
	uc := utls.UClient(raw, &utls.Config{ServerName: sni, InsecureSkipVerify: true, OmitEmptyPsk: true,
		DynamicRecordSizingDisabled: true}, utls.HelloCustom)
	if err := uc.ApplyPreset(spec); err != nil {
		raw.Close()
		return nil, found, fmt.Errorf("apply: %w", err)
	}
	raw.SetDeadline(time.Now().Add(15 * time.Second))
	if err := uc.Handshake(); err != nil {
		raw.Close()
		return nil, found, fmt.Errorf("handshake: %w", err)
	}
	raw.SetDeadline(time.Time{})
	st := uc.ConnectionState()
	found.ALPN, found.Version, found.Cipher = st.NegotiatedProtocol, fmt.Sprintf("%#04x", st.Version),
		utls.CipherSuiteName(st.CipherSuite)
	if len(st.PeerCertificates) == 0 {
		found.VerifyErr = "no certificate"
		return uc, found, nil
	}
	leaf := st.PeerCertificates[0]
	found.NotBefore, found.NotAfter = leaf.NotBefore, leaf.NotAfter
	for _, name := range leaf.DNSNames {
		found.SANs = append(found.SANs, "DNS:"+name)
	}
	for _, ip := range leaf.IPAddresses {
		found.SANs = append(found.SANs, "IP:"+ip.String())
	}
	intermediates := x509.NewCertPool()
	for _, c := range st.PeerCertificates[1:] {
		intermediates.AddCert(c)
	}
	if _, err := leaf.Verify(x509.VerifyOptions{Roots: p.Roots, Intermediates: intermediates, DNSName: verifyName,
		CurrentTime: time.Now()}); err != nil {
		found.VerifyErr = err.Error()
	} else {
		found.Verified = true
	}
	return uc, found, nil
}

// probe is a site's leaf validity, from a handshake of its own, cached per (address, SNI) for ProbeTTL.
type probe struct {
	from, until time.Time
	verified    bool
	at          time.Time
	problem     string
}

type probes struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
	cache map[string]probe
}

func (p *Proxy) validity(dst, sni, verifyName string, hello []byte) (probe, bool) {
	key := dst + "|" + sni
	p.probes.mu.Lock()
	if p.probes.locks == nil {
		p.probes.locks, p.probes.cache = map[string]*sync.Mutex{}, map[string]probe{}
	}
	lock, ok := p.probes.locks[key]
	if !ok {
		lock = &sync.Mutex{}
		p.probes.locks[key] = lock
	}
	p.probes.mu.Unlock()
	lock.Lock()
	defer lock.Unlock()
	p.probes.mu.Lock()
	hit, ok := p.probes.cache[key]
	p.probes.mu.Unlock()
	if ok && time.Since(hit.at) < ProbeTTL {
		return hit, false
	}
	hit = probe{at: time.Now()}
	uc, found, err := p.handshake(dst, sni, verifyName, hello)
	switch {
	case err != nil:
		hit.problem = truncate(err.Error(), 200)
	case !found.Verified:
		hit.problem = truncate(found.VerifyErr, 200)
	default:
		hit.from, hit.until, hit.verified = found.NotBefore, found.NotAfter, true
	}
	if uc != nil {
		uc.NetConn().Close()
	}
	p.probes.mu.Lock()
	p.probes.cache[key] = hit
	p.probes.mu.Unlock()
	return hit, true
}

// inside is whether every universe lies inside [from + Margin, until - Margin].
func inside(from, until time.Time, universes []float64) bool {
	lo := float64(from.Add(Margin).UnixNano()) / 1e9
	hi := float64(until.Add(-Margin).UnixNano()) / 1e9
	for _, u := range universes {
		if u < lo || u > hi {
			return false
		}
	}
	return true
}

func (p *Proxy) dial(dst string) (net.Conn, error) {
	if p.Dial != nil {
		return p.Dial(dst)
	}
	return net.DialTimeout("tcp", dst, DialTimeout)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
