package main

import (
	"encoding/binary"
	"math"
	"os"
	"sync"
	"syscall"
	"time"
)

// The clock page the follower writes (clockpage.py, version 2): magic version seq universe_ns anchor_ns rate armed
// reserved generation check mono_universe_ns mono_anchor_ns, little-endian, 80 bytes, seq odd while it's written.
const (
	pageSize  = 80
	pageMagic = 0x4B434541
	checkSalt = 0x9E3779B97F4A7C15
)

// state is the universe as the session reads it: armed, the universe at the anchor (seconds), the anchor (real
// seconds) and the rate.
type state struct {
	Armed  bool
	V      float64
	Anchor float64
	Rate   float64
}

var unarmed = state{Rate: 1}

func (s state) at(t float64) float64 {
	if !s.Armed {
		return t
	}
	if s.Rate == 0 || t <= s.Anchor {
		return s.V
	}
	return s.V + (t-s.Anchor)*s.Rate
}

func pageCheck(seq uint64, u, a int64, rate float64, armed uint32, generation uint64, mu, ma int64) uint64 {
	return seq ^ uint64(u) ^ uint64(a) ^ math.Float64bits(rate) ^ uint64(armed) ^ generation ^ uint64(mu) ^
		uint64(ma) ^ checkSalt
}

// readPage reads a consistent state from the page's bytes, or false when it's being written, torn or not a page.
func readPage(b []byte) (state, bool) {
	if len(b) < pageSize || binary.LittleEndian.Uint32(b[0:]) != pageMagic {
		return state{}, false
	}
	seq := binary.LittleEndian.Uint64(b[8:])
	if seq&1 == 1 {
		return state{}, false
	}
	u := int64(binary.LittleEndian.Uint64(b[16:]))
	a := int64(binary.LittleEndian.Uint64(b[24:]))
	rate := math.Float64frombits(binary.LittleEndian.Uint64(b[32:]))
	armed := binary.LittleEndian.Uint32(b[40:])
	generation := binary.LittleEndian.Uint64(b[48:])
	check := binary.LittleEndian.Uint64(b[56:])
	mu := int64(binary.LittleEndian.Uint64(b[64:]))
	ma := int64(binary.LittleEndian.Uint64(b[72:]))
	if check != pageCheck(seq, u, a, rate, armed, generation, mu, ma) || binary.LittleEndian.Uint64(b[8:]) != seq {
		return state{}, false
	}
	if armed == 0 {
		return unarmed, true
	}
	return state{Armed: true, V: float64(u) / 1e9, Anchor: float64(a) / 1e9, Rate: rate}, true
}

// Universe follows the clock page, read every PagePoll, so a change is seen and the universe it left is held for Hold.
type Universe struct {
	path     string
	mu       sync.Mutex
	page     []byte
	current  state
	known    bool
	previous state
	leftAt   time.Time
}

func NewUniverse(path string) *Universe {
	u := &Universe{path: path, current: unarmed}
	u.refresh(time.Now())
	return u
}

func (u *Universe) Poll() {
	for {
		time.Sleep(PagePoll)
		u.refresh(time.Now())
	}
}

func (u *Universe) read() state {
	if u.page == nil {
		f, err := os.Open(u.path)
		if err != nil {
			return unarmed
		}
		mapped, err := syscall.Mmap(int(f.Fd()), 0, pageSize, syscall.PROT_READ, syscall.MAP_SHARED)
		f.Close()
		if err != nil {
			return unarmed
		}
		u.page = mapped
	}
	for i := 0; i < PageReads; i++ {
		if s, ok := readPage(u.page); ok {
			return s
		}
	}
	return u.current
}

func (u *Universe) refresh(now time.Time) state {
	u.mu.Lock()
	defer u.mu.Unlock()
	s := u.read()
	if u.known && s != u.current {
		u.previous, u.leftAt = u.current, now
	}
	u.current, u.known = s, true
	return s
}

// View is false when the session is on real time and was armed nowhere within Hold; else true, with the universes a
// client may check against: now, the universe at the anchor, and the one left within Hold when it was armed.
func (u *Universe) View(now time.Time) (bool, []float64) {
	u.refresh(now)
	u.mu.Lock()
	defer u.mu.Unlock()
	return view(u.current, u.previous, u.leftAt, now)
}

func view(current, previous state, leftAt time.Time, now time.Time) (bool, []float64) {
	held := !leftAt.IsZero() && now.Sub(leftAt) <= Hold && previous.Armed
	if !current.Armed && !held {
		return false, nil
	}
	t := float64(now.UnixNano()) / 1e9
	out := []float64{current.at(t)}
	if current.Armed {
		out = append(out, current.V)
	}
	if held {
		out = append(out, previous.at(t))
	}
	return true, out
}
