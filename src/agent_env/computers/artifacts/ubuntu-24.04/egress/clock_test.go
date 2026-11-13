package main

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func page(seq uint64, universe, anchor time.Time, rate float64, armed bool) []byte {
	b := make([]byte, pageSize)
	u, a := universe.UnixNano(), anchor.UnixNano()
	var on uint32
	if armed {
		on = 1
	}
	binary.LittleEndian.PutUint32(b[0:], pageMagic)
	binary.LittleEndian.PutUint32(b[4:], 2)
	binary.LittleEndian.PutUint64(b[8:], seq)
	binary.LittleEndian.PutUint64(b[16:], uint64(u))
	binary.LittleEndian.PutUint64(b[24:], uint64(a))
	binary.LittleEndian.PutUint64(b[32:], math.Float64bits(rate))
	binary.LittleEndian.PutUint32(b[40:], on)
	binary.LittleEndian.PutUint64(b[48:], 7)
	binary.LittleEndian.PutUint64(b[56:], pageCheck(seq, u, a, rate, on, 7, a, a))
	binary.LittleEndian.PutUint64(b[64:], uint64(a))
	binary.LittleEndian.PutUint64(b[72:], uint64(a))
	return b
}

func TestAPageReadsAsTheUniverseItHoldsAndATornOneNot(t *testing.T) {
	universe, anchor := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC), time.Unix(time.Now().Unix(), 0)
	s, ok := readPage(page(4, universe, anchor, 2, true))
	if !ok || !s.Armed || s.Rate != 2 || int64(s.V) != universe.Unix() {
		t.Fatalf("read %+v %v", s, ok)
	}
	if got := s.at(float64(anchor.Unix()) + 10); int64(got) != universe.Unix()+20 {
		t.Fatalf("10 s on at rate 2: %v", got)
	}
	if _, ok := readPage(page(5, universe, anchor, 2, true)); ok {
		t.Fatal("a page being written (odd seq) read as whole")
	}
	torn := page(4, universe, anchor, 2, true)
	torn[20] ^= 1
	if _, ok := readPage(torn); ok {
		t.Fatal("a torn page read as whole")
	}
	if s, ok := readPage(page(2, time.Time{}, time.Time{}, 1, false)); !ok || s.Armed {
		t.Fatalf("an unarmed page: %+v %v", s, ok)
	}
}

func TestTheUniverseFollowsThePageFileAndHoldsTheOneItLeft(t *testing.T) {
	path := filepath.Join(t.TempDir(), "page")
	if err := os.WriteFile(path, page(2, time.Time{}, time.Time{}, 1, false), 0o644); err != nil {
		t.Fatal(err)
	}
	u := NewUniverse(path)
	if armed, _ := u.View(time.Now()); armed {
		t.Fatal("unarmed page viewed as armed")
	}
	past := time.Now().Add(-180 * 24 * time.Hour)
	f, _ := os.OpenFile(path, os.O_WRONLY, 0)
	f.WriteAt(page(4, past, time.Now(), 1, true), 0)
	f.Close()
	armed, universes := u.View(time.Now())
	if !armed || len(universes) != 2 || int64(universes[1]) != past.Unix() {
		t.Fatalf("armed page: %v %v", armed, universes)
	}
}

func TestAClientMayCheckAgainstTheUniverseItLeftForHoldAfterAChange(t *testing.T) {
	now := time.Now()
	past := state{Armed: true, V: float64(now.Add(-100 * 24 * time.Hour).Unix()), Anchor: float64(now.Unix()), Rate: 1}
	if armed, _ := view(unarmed, state{}, time.Time{}, now); armed {
		t.Fatal("real time viewed as armed")
	}
	armed, universes := view(unarmed, past, now.Add(-Hold/2), now)
	if !armed || len(universes) != 2 {
		t.Fatalf("within Hold of leaving an armed universe: %v %v", armed, universes)
	}
	if armed, _ := view(unarmed, past, now.Add(-2*Hold), now); armed {
		t.Fatal("past Hold, the universe left still counts")
	}
}

// golden is a page the follower's writer wrote (clockpage.encode), as testdata keeps it.
func golden(t *testing.T, name string) []byte {
	text, err := os.ReadFile(filepath.Join("testdata", "page-"+name+".hex"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := hex.DecodeString(strings.TrimSpace(string(text)))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestThePagesTheFollowersWriterWritesReadAsTheUniverseTheyHold(t *testing.T) {
	s, ok := readPage(golden(t, "armed"))
	if !ok || !s.Armed || s.V != 1577836800 || s.Rate != 2 || math.Abs(s.Anchor-1794000000.123456789) > 1e-6 {
		t.Fatalf("armed: %+v %v", s, ok)
	}
	if s, ok := readPage(golden(t, "unarmed")); !ok || s != unarmed {
		t.Fatalf("unarmed: %+v %v", s, ok)
	}
}
