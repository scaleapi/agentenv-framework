package main

import (
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	utls "github.com/refraction-networking/utls"
)

// upstream is what a handshake with a site found: the protocol it picked, its leaf's validity and names, and whether
// that leaf verifies against real time and the public roots.
type upstream struct {
	// Dst is the address the handshake reached.
	Dst       string
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
func (p *Proxy) handshake(dsts []string, sni, verifyName string, hello []byte) (*utls.UConn, upstream, error) {
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
	raw, dst, err := p.dialAny(dsts)
	if err != nil {
		return nil, found, dialError{err}
	}
	found.Dst = dst
	uc := utls.UClient(raw, &utls.Config{ServerName: sni, InsecureSkipVerify: true, OmitEmptyPsk: true,
		DynamicRecordSizingDisabled: true}, utls.HelloCustom)
	if err := uc.ApplyPreset(spec); err != nil {
		raw.Close()
		return nil, found, fmt.Errorf("apply: %w", err)
	}
	raw.SetDeadline(time.Now().Add(HandshakeTimeout))
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
	chains, err := leaf.Verify(x509.VerifyOptions{Roots: p.Roots, Intermediates: intermediates, DNSName: verifyName,
		CurrentTime: time.Now()})
	if err != nil {
		found.VerifyErr = err.Error()
	} else {
		found.Verified = true
		found.NotBefore, found.NotAfter = chainWindow(chains)
	}
	return uc, found, nil
}

// chainWindow is the dates every certificate of every verified chain is valid between: a client building any of
// those chains takes the site only then, so passing it through is safe only inside them.
func chainWindow(chains [][]*x509.Certificate) (time.Time, time.Time) {
	var from, until time.Time
	for _, chain := range chains {
		for _, cert := range chain {
			if from.IsZero() || cert.NotBefore.After(from) {
				from = cert.NotBefore
			}
			if until.IsZero() || cert.NotAfter.Before(until) {
				until = cert.NotAfter
			}
		}
	}
	return from, until
}

// probe is what a handshake of the proxy's own found of one address of a site: the dates its verified chain shares,
// or that it didn't answer, or that its handshake didn't finish. It's cached per (address, SNI): a finished one for
// ProbeTTL, the others for ProbeFailedTTL, since they may be a moment's trouble.
type probe struct {
	from, until time.Time
	verified    bool
	// failed is a handshake that didn't finish; unreachable, an address that didn't answer.
	failed, unreachable bool
	at                  time.Time
	problem             string
}

func (hit probe) fresh() bool {
	if hit.failed || hit.unreachable {
		return time.Since(hit.at) < ProbeFailedTTL
	}
	return time.Since(hit.at) < ProbeTTL
}

// probes caches each address's probe, up to cacheLimit entries, the stale ones dropped first once it's full. An
// address's concurrent probes wait on the one under way; no lock is held while one is.
type probes struct {
	mu       sync.Mutex
	cache    map[string]probe
	underway map[string]*underway
}

type underway struct {
	done chan struct{}
	hit  probe
	ok   bool
}

// cacheLimit bounds the probe and leaf caches, so a computer that visits many sites keeps a bounded memory.
var cacheLimit = 4096

// dialError is a handshake's that never reached the site.
type dialError struct{ error }

// probeAt is the probe of dst for sni, cached or new, and whether it probed now.
func (p *Proxy) probeAt(dst, sni, verifyName string, hello []byte) (probe, bool) {
	key := dst + "|" + sni
	p.probes.mu.Lock()
	if p.probes.cache == nil {
		p.probes.cache, p.probes.underway = map[string]probe{}, map[string]*underway{}
	}
	if hit, ok := p.probes.cache[key]; ok && hit.fresh() {
		p.probes.mu.Unlock()
		return hit, false
	}
	if running, ok := p.probes.underway[key]; ok {
		p.probes.mu.Unlock()
		<-running.done
		if !running.ok {
			return probe{at: time.Now(), failed: true, problem: "the probe it waited on failed"}, false
		}
		return running.hit, false
	}
	running := &underway{done: make(chan struct{})}
	p.probes.underway[key] = running
	p.probes.mu.Unlock()
	defer func() {
		p.probes.mu.Lock()
		if running.ok {
			p.store(key, running.hit)
		}
		delete(p.probes.underway, key)
		p.probes.mu.Unlock()
		close(running.done)
	}()

	hit := probe{at: time.Now()}
	uc, found, err := p.handshake([]string{dst}, sni, verifyName, hello)
	var unanswered dialError
	switch {
	case errors.As(err, &unanswered):
		hit.unreachable, hit.problem = true, truncate(err.Error(), 200)
	case err != nil:
		hit.failed, hit.problem = true, truncate(err.Error(), 200)
	case !found.Verified:
		hit.problem = truncate(found.VerifyErr, 200)
	default:
		hit.from, hit.until, hit.verified = found.NotBefore, found.NotAfter, true
	}
	if uc != nil {
		uc.NetConn().Close()
	}
	running.hit, running.ok = hit, true
	return hit, true
}

// store keeps hit for key, dropping the stale entries, then any others, once the cache is past cacheLimit.
func (p *Proxy) store(key string, hit probe) {
	p.probes.cache[key] = hit
	if len(p.probes.cache) <= cacheLimit {
		return
	}
	for k, v := range p.probes.cache {
		if !v.fresh() {
			delete(p.probes.cache, k)
		}
	}
	for k := range p.probes.cache {
		if len(p.probes.cache) <= cacheLimit {
			break
		}
		if k != key {
			delete(p.probes.cache, k)
		}
	}
}

// forget drops dst's probe for sni, for an address that stopped answering since.
func (p *Proxy) forget(dst, sni string) {
	p.probes.mu.Lock()
	delete(p.probes.cache, dst+"|"+sni)
	p.probes.mu.Unlock()
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

// dialAny dials each address in turn until one answers: which, or every one's error.
func (p *Proxy) dialAny(dsts []string) (net.Conn, string, error) {
	var problems []string
	for _, dst := range dsts {
		conn, err := p.dial(dst)
		if err == nil {
			return conn, dst, nil
		}
		problems = append(problems, err.Error())
	}
	if len(problems) == 0 {
		return nil, "", errors.New("no address to dial")
	}
	return nil, "", errors.New(strings.Join(problems, "; "))
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
