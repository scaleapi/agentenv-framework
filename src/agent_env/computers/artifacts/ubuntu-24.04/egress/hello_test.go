package main

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"encoding/binary"
	"net"
	"reflect"
	"testing"
	"time"
)

// goHello is the ClientHello message Go's TLS client sends for name, offering alpn.
func goHello(t *testing.T, name string, alpn []string) []byte {
	t.Helper()
	client, server := net.Pipe()
	go tls.Client(client, &tls.Config{ServerName: name, NextProtos: alpn, InsecureSkipVerify: true}).Handshake()
	defer client.Close()
	defer server.Close()
	hello, err := clientHello(bufio.NewReader(server), time.Now().Add(5*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	return hello
}

func records(message []byte, size int) []byte {
	var out []byte
	for len(message) > 0 {
		n := min(size, len(message))
		out = append(out, 22, 3, 1, byte(n>>8), byte(n))
		out = append(out, message[:n]...)
		message = message[n:]
	}
	return out
}

func TestAClientHelloIsReadWholeAcrossRecordsAndNamesItsSiteAndProtocols(t *testing.T) {
	hello := goHello(t, "Example.COM", []string{"h2", "http/1.1"})
	sni, alpn, err := parseHello(hello)
	if err != nil || sni != "example.com" || !reflect.DeepEqual(alpn, []string{"h2", "http/1.1"}) {
		t.Fatalf("parsed %q %v %v", sni, alpn, err)
	}
	split := records(hello, 100)
	again, err := clientHello(bufio.NewReader(bytes.NewReader(split)), time.Now().Add(time.Second))
	if err != nil || !bytes.Equal(again, hello) {
		t.Fatalf("across records: %v", err)
	}
	if _, err := clientHello(bufio.NewReader(bytes.NewReader([]byte("GET / HTTP/1.1\r\n\r\n"))),
		time.Now().Add(time.Second)); err == nil {
		t.Fatal("plain HTTP read as TLS")
	}
}

func withExtension(hello []byte, kind uint16, body []byte) []byte {
	p := 4 + 2 + 32
	p += 1 + int(hello[p])
	p += 2 + int(binary.BigEndian.Uint16(hello[p:]))
	p += 1 + int(hello[p])
	exts := append(append([]byte{}, hello[p+2:]...), byte(kind>>8), byte(kind), byte(len(body)>>8), byte(len(body)))
	exts = append(exts, body...)
	out := append(append([]byte{}, hello[:p]...), byte(len(exts)>>8), byte(len(exts)))
	out = append(out, exts...)
	n := len(out) - 4
	out[1], out[2], out[3] = byte(n>>16), byte(n>>8), byte(n)
	return out
}

func TestTheUpstreamRecordLeavesOutWhatOnlyAResumptionWithTheProxyMeans(t *testing.T) {
	hello := withExtension(goHello(t, "example.com", nil), 41, []byte{0, 1, 2})
	record, gone, err := helloRecord(hello)
	if err != nil || !reflect.DeepEqual(gone, []uint16{41}) {
		t.Fatalf("dropped %v %v", gone, err)
	}
	if record[0] != 22 || int(binary.BigEndian.Uint16(record[3:])) != len(record)-5 {
		t.Fatal("not one handshake record")
	}
	if _, again, _ := helloRecord(record[5:]); len(again) != 0 {
		t.Fatalf("the record still has %v", again)
	}
}
