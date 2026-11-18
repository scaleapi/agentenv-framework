package main

import "testing"

func TestServeListensWhereTheBaseNamesItElseOnListen(t *testing.T) {
	for _, tc := range []struct {
		args  []string
		named string
		want  string
		ok    bool
	}{
		{nil, "", Listen, true},
		{nil, "http://127.0.0.1:3130", "127.0.0.1:3130", true},
		{nil, "http://127.0.0.1:3130/", "127.0.0.1:3130", true},
		{nil, "http://[::1]:3130", "[::1]:3130", true},
		{nil, "127.0.0.1:3130", "", false},
		{nil, "http://127.0.0.1", "", false},
		{nil, "https://127.0.0.1:3130", "", false},
		{nil, "http://127.0.0.1:3130/proxy", "", false},
		{[]string{"--listen", "127.0.0.1:3130"}, "", "", false},
	} {
		got, err := serveListen(tc.args, tc.named)
		if (err == nil) != tc.ok || got != tc.want {
			t.Errorf("serveListen(%q, %q) = %q, %v; want %q, ok %v", tc.args, tc.named, got, err, tc.want, tc.ok)
		}
	}
}
