/** The virtual clock an env runs on (clock/v1) and the arithmetic for a trigger's due mark against it.
 *  Separate from event parsing: this reads the axis events' `virtual_time` is measured on (a run can be at
 *  86400x, one real second = one virtual day), so nothing here is wall time. Pure. */
import { asRecord, num, str } from './coerce';

export interface EnvClock {
  envId: string;
  armed: boolean;
  t0?: string;
  rate?: number;
  virtualTime?: string;
  readAtUtc?: string;
}

/** Time left before a mark arrives, on both axes. */
export interface Countdown {
  virtualMs: number;
  realMs: number;
  due: boolean;
}

/** The reading the clock vended, not extrapolated toward now: it goes stale between polls, but a reading the gateway actually reported beats an inferred one. */
function vendedReading(clock: EnvClock): number | null {
  if (!clock.armed || !clock.virtualTime) return null;
  const base = Date.parse(clock.virtualTime);
  return Number.isNaN(base) ? null : base;
}

/** How far the mark is from the clock's last reported reading. Refreshes with
 *  the poll rather than ticking, so it trails by up to one interval. */
export function countdownTo(mark: string, clock: EnvClock): Countdown | null {
  const target = Date.parse(mark);
  const now = vendedReading(clock);
  if (now === null || Number.isNaN(target)) return null;
  const virtualMs = target - now;
  // Guarded: a rate of 0 is a stopped clock, and the mark never arrives.
  const rate = clock.rate && clock.rate > 0 ? clock.rate : null;
  return {
    virtualMs,
    realMs: rate === null ? Infinity : virtualMs / rate,
    due: virtualMs <= 0,
  };
}

/** `3720000` → `1h 2m`. Coarse on purpose: virtual spans run to years. */
export function humanizeMs(ms: number): string {
  if (!Number.isFinite(ms)) return '∞';
  const s = Math.max(0, Math.round(Math.abs(ms) / 1000));
  if (s < 60) return `${s}s`;
  const units: [number, string][] = [
    [86400 * 365, 'y'],
    [86400, 'd'],
    [3600, 'h'],
    [60, 'm'],
  ];
  const parts: string[] = [];
  let rest = s;
  for (const [size, label] of units) {
    const n = Math.floor(rest / size);
    if (n > 0) {
      parts.push(`${n}${label}`);
      rest -= n * size;
    }
    if (parts.length === 2) break;
  }
  return parts.join(' ') || `${s}s`;
}

/** Per-env clock readings from the endpoint's `state_meta`. An env with no clock/v1 reports `clock: null` and is skipped, so an absent entry means "no clock". */
export function parseEnvClocks(stateMeta: unknown): EnvClock[] {
  const clocks: EnvClock[] = [];
  for (const [envId, rawMeta] of Object.entries(asRecord(stateMeta) ?? {})) {
    const meta = asRecord(rawMeta);
    const clock = asRecord(meta?.clock);
    if (!clock) continue;
    clocks.push({
      envId,
      armed: clock.armed === true,
      t0: str(clock.t0),
      rate: num(clock.virtual_seconds_per_real_second),
      virtualTime: str(clock.virtual_time),
      readAtUtc: str(meta?.clock_read_at_utc),
    });
  }
  return clocks;
}
