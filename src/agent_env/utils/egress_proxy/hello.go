package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"strings"
	"time"
)

// A TLS record's content type for handshake messages, a ClientHello's handshake type, and the most plaintext one
// record carries.
const (
	recordHandshake = 22
	clientHelloType = 1
	maxRecord       = 16384
)

// clientHello is the ClientHello handshake message the client has sent, peeked from r so the handshake stays
// untouched for whoever reads it next; it may span several records.
func clientHello(r *bufio.Reader, deadline time.Time) ([]byte, error) {
	need := 5
	for {
		if time.Now().After(deadline) {
			return nil, errors.New("ClientHello incomplete")
		}
		data, err := r.Peek(need)
		if err != nil {
			if errors.Is(err, bufio.ErrBufferFull) {
				return nil, errors.New("ClientHello larger than the read buffer")
			}
			return nil, err
		}
		if data[0] != recordHandshake {
			return nil, errors.New("not TLS")
		}
		var body []byte
		for p := 0; ; {
			if p+5 > len(data) {
				need = p + 5
				break
			}
			if data[p] != recordHandshake {
				return nil, errors.New("ClientHello interrupted by another record")
			}
			size := int(binary.BigEndian.Uint16(data[p+3:]))
			if p+5+size > len(data) {
				need = p + 5 + size
				break
			}
			body = append(body, data[p+5:p+5+size]...)
			p += 5 + size
			if len(body) >= 4 {
				if whole := 4 + (int(body[1])<<16 | int(body[2])<<8 | int(body[3])); len(body) >= whole {
					return body[:whole], nil
				}
			}
		}
	}
}

// parseHello is the server name and the ALPN protocols a ClientHello message offers.
func parseHello(hello []byte) (string, []string, error) {
	p, err := helloExtensions(hello)
	if err != nil {
		return "", nil, err
	}
	if len(hello) < p+2 {
		return "", nil, nil
	}
	end := p + 2 + int(binary.BigEndian.Uint16(hello[p:]))
	if end > len(hello) {
		return "", nil, errors.New("extensions overrun the ClientHello")
	}
	var sni string
	var alpn []string
	for q := p + 2; q+4 <= end; {
		kind := binary.BigEndian.Uint16(hello[q:])
		size := int(binary.BigEndian.Uint16(hello[q+2:]))
		body := hello[q+4 : min(q+4+size, end)]
		q += 4 + size
		switch {
		case kind == 0 && len(body) > 5 && body[2] == 0:
			n := int(binary.BigEndian.Uint16(body[3:]))
			if 5+n <= len(body) {
				sni = strings.ToLower(string(body[5 : 5+n]))
			}
		case kind == 16 && len(body) > 2:
			for r := 2; r < len(body); {
				n := int(body[r])
				if r+1+n > len(body) {
					break
				}
				alpn = append(alpn, string(body[r+1:r+1+n]))
				r += 1 + n
			}
		}
	}
	return sni, alpn, nil
}

// resumption: what only the proxy's own session with the client could mean, never the site's.
var dropped = map[uint16]bool{41: true, 42: true}

// helloRecord is the ClientHello as one handshake record, without the extensions only a resumption with the proxy put
// there, and the extensions it left out.
func helloRecord(hello []byte) ([]byte, []uint16, error) {
	p, err := helloExtensions(hello)
	if err != nil {
		return nil, nil, err
	}
	body := append([]byte{}, hello[4:p]...)
	var gone []uint16
	if len(hello) >= p+2 {
		end := p + 2 + int(binary.BigEndian.Uint16(hello[p:]))
		if end > len(hello) {
			return nil, nil, errors.New("extensions overrun the ClientHello")
		}
		var exts []byte
		for q := p + 2; q+4 <= end; {
			kind := binary.BigEndian.Uint16(hello[q:])
			size := int(binary.BigEndian.Uint16(hello[q+2:]))
			if q+4+size > end {
				return nil, nil, errors.New("an extension overruns the ClientHello")
			}
			if dropped[kind] {
				gone = append(gone, kind)
			} else {
				exts = append(exts, hello[q:q+4+size]...)
			}
			q += 4 + size
		}
		body = binary.BigEndian.AppendUint16(body, uint16(len(exts)))
		body = append(body, exts...)
	}
	if len(body)+4 > maxRecord {
		return nil, nil, errors.New("ClientHello longer than one record")
	}
	msg := append([]byte{clientHelloType, byte(len(body) >> 16), byte(len(body) >> 8), byte(len(body))}, body...)
	return append([]byte{recordHandshake, 3, 1, byte(len(msg) >> 8), byte(len(msg))}, msg...), gone, nil
}

// helloExtensions is where a ClientHello message's extensions start, past its version, random, session id, cipher
// suites and compression methods, each length checked against the message.
func helloExtensions(hello []byte) (int, error) {
	if len(hello) < 4 || hello[0] != clientHelloType {
		return 0, errors.New("not a ClientHello")
	}
	p := 4 + 2 + 32
	for _, width := range []int{1, 2, 1} { // session id, cipher suites, compression methods
		if len(hello) < p+width {
			return 0, errors.New("short ClientHello")
		}
		n := int(hello[p])
		if width == 2 {
			n = int(binary.BigEndian.Uint16(hello[p:]))
		}
		p += width + n
		if p > len(hello) {
			return 0, errors.New("a ClientHello field overruns the message")
		}
	}
	return p, nil
}
