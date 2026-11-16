package main

import "testing"

func TestServeListensOnListenUnlessTold(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
		ok   bool
	}{
		{nil, Listen, true},
		{[]string{"--listen", "127.0.0.1:3130"}, "127.0.0.1:3130", true},
		{[]string{"--listen"}, "", false},
		{[]string{"--listen", ""}, "", false},
		{[]string{"--port", "3130"}, "", false},
	} {
		got, err := serveListen(tc.args)
		if (err == nil) != tc.ok || got != tc.want {
			t.Errorf("serveListen(%q) = %q, %v; want %q, ok %v", tc.args, got, err, tc.want, tc.ok)
		}
	}
}
