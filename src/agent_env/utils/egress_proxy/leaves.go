package main

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// The longest common name X.509 allows (ub-common-name).
const commonNameLimit = 64

// Leaves signs the leaves an intercepted client is handed: by this computer's CA when the site verified, else by a CA
// trusted nowhere, so the client refuses it as it would the site. Each is valid over all of its CA's window, so it
// holds wherever the universe goes, and is minted once per name and its alternative names.
type Leaves struct {
	ca, untrusted signer
	key           *ecdsa.PrivateKey
	mu            sync.Mutex
	cache         map[string]*tls.Certificate
	// order is the cache's keys, oldest first, so it drops the oldest past cacheLimit; a connection keeps its own.
	order []string
}

type signer struct {
	cert *x509.Certificate
	key  crypto.Signer
}

func LoadLeaves(folder string) (*Leaves, error) {
	certPEM, err := os.ReadFile(filepath.Join(folder, "ca.pem"))
	if err != nil {
		return nil, err
	}
	keyPEM, err := os.ReadFile(filepath.Join(folder, "ca.key"))
	if err != nil {
		return nil, err
	}
	ca, err := parseSigner(certPEM, keyPEM)
	if err != nil {
		return nil, err
	}
	return NewLeaves(ca)
}

func NewLeaves(ca signer) (*Leaves, error) {
	untrusted, err := mintCA("Untrusted upstream")
	if err != nil {
		return nil, err
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	return &Leaves{ca: ca, untrusted: untrusted, key: key, cache: map[string]*tls.Certificate{}}, nil
}

// For is the leaf for host and the names sans ("DNS:..." or "IP:..."), by this computer's CA when trusted, and
// whether it was minted now.
func (l *Leaves) For(host string, sans []string, trusted bool) (*tls.Certificate, bool, error) {
	sorted := append([]string{}, sans...)
	sort.Strings(sorted)
	key := host + "|" + strings.Join(sorted, ",")
	by := l.ca
	if !trusted {
		key, by = "untrusted|"+key, l.untrusted
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if cert, ok := l.cache[key]; ok {
		return cert, false, nil
	}
	template := &x509.Certificate{
		SerialNumber:          serial(),
		Subject:               pkix.Name{CommonName: truncate(host, commonNameLimit)},
		NotBefore:             by.cert.NotBefore.Add(LeafInset),
		NotAfter:              by.cert.NotAfter.Add(-LeafInset),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}
	for _, name := range append([]string{host}, sans...) {
		name = strings.TrimPrefix(strings.TrimPrefix(name, "DNS:"), "IP:")
		if ip := net.ParseIP(strings.Trim(name, "[]")); ip != nil {
			template.IPAddresses = appendIP(template.IPAddresses, ip)
		} else if name != "" && !contains(template.DNSNames, name) {
			template.DNSNames = append(template.DNSNames, name)
		}
	}
	der, err := x509.CreateCertificate(rand.Reader, template, by.cert, &l.key.PublicKey, by.key)
	if err != nil {
		return nil, false, err
	}
	cert := &tls.Certificate{Certificate: [][]byte{der, by.cert.Raw}, PrivateKey: l.key}
	l.cache[key] = cert
	l.order = append(l.order, key)
	for len(l.order) > cacheLimit {
		delete(l.cache, l.order[0])
		l.order = l.order[1:]
	}
	return cert, true, nil
}

// mintCA is a self-signed CA over the window, its key in memory.
func mintCA(name string) (signer, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return signer{}, err
	}
	template := &x509.Certificate{
		SerialNumber:          serial(),
		Subject:               pkix.Name{CommonName: name},
		NotBefore:             CAFrom,
		NotAfter:              CAUntil,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLenZero:        true,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return signer{}, err
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		return signer{}, err
	}
	return signer{cert: cert, key: key}, nil
}

func parseSigner(certPEM, keyPEM []byte) (signer, error) {
	block, _ := pem.Decode(certPEM)
	if block == nil {
		return signer{}, errors.New("no certificate in the CA's PEM")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return signer{}, err
	}
	block, _ = pem.Decode(keyPEM)
	if block == nil {
		return signer{}, errors.New("no key in the CA key's PEM")
	}
	key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return signer{}, err
	}
	s, ok := key.(crypto.Signer)
	if !ok {
		return signer{}, errors.New("the CA key can't sign")
	}
	return signer{cert: cert, key: s}, nil
}

func serial() *big.Int {
	n, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 127))
	return n
}

func appendIP(ips []net.IP, ip net.IP) []net.IP {
	for _, have := range ips {
		if have.Equal(ip) {
			return ips
		}
	}
	return append(ips, ip)
}

func contains(list []string, s string) bool {
	for _, have := range list {
		if strings.EqualFold(have, s) {
			return true
		}
	}
	return false
}
