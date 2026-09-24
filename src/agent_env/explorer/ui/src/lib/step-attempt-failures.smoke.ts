/**
 * Smoke test for the step-attempt failure ledger parsing + copy.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   yarn test:smoke        (or: npx tsx src/lib/step-attempt-failures.smoke.ts)
 *
 * `.smoke.ts`, not `.test.ts`, matching the five sibling smoke files in this directory.
 * The package has no Jest config and `test` is a no-op, so there is no runner for a
 * `.test.ts` file to be picked up by — the `test:smoke` script is the entry point.
 */
import {
  ledgerSummary,
  ledgerTone,
  parseStepAttemptFailures,
} from './step-attempt-failures';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

function main(): void {
  // The real prod entry from aeh-0000000000000-000000000000000a (completed run, one retry).
  const real = {
    attempt: 1,
    step_id: 'verify-response',
    step_type: 'rubrics_verifier',
    error_class: 'ValueError',
    error_message: 'Judge returned 0 results but expected 31 criteria',
    duration_s: 196.596,
    // The worker's real format: `strftime('%Y-%m-%d %H:%M UTC')`. Minute precision, a
    // literal " UTC" suffix, no ISO 'T'. Do not "fix" this to an ISO string — an earlier
    // version of this fixture was ISO, and it hid a bug where reformatting clobbered
    // the T inside "UTC" and rendered "00:14 U C UTC".
    occurred_at_utc: '2026-07-28 18:53 UTC',
  };

  {
    const [entry] = parseStepAttemptFailures([real]);
    assert(entry?.stepId === 'verify-response', 'maps step_id');
    assert(entry?.stepType === 'rubrics_verifier', 'maps step_type');
    assert(entry?.errorClass === 'ValueError', 'maps error_class');
    assert(entry?.durationS === 196.596, 'maps duration_s');
    assert(
      entry?.message === 'Judge returned 0 results but expected 31 criteria',
      'maps error_message',
    );
  }

  {
    // Oldest failure first. $addToSet doesn't guarantee stored order, so parsing has to sort.
    const parsed = parseStepAttemptFailures([
      { ...real, attempt: 3, occurred_at_utc: '2026-07-29 00:14 UTC' },
      { ...real, attempt: 1, occurred_at_utc: '2026-07-29 00:10 UTC' },
      { ...real, attempt: 2, occurred_at_utc: '2026-07-29 00:12 UTC' },
    ]);
    assert(
      parsed.map(e => e.attempt).join(',') === '1,2,3',
      'sorts oldest failure first',
    );
  }

  {
    // The stored format is fixed-width and zero-padded, so a string compare is chronological.
    // Hour and day rollovers are where a naive compare would break if it weren't.
    const parsed = parseStepAttemptFailures([
      { ...real, attempt: 2, occurred_at_utc: '2026-07-29 09:05 UTC' },
      { ...real, attempt: 3, occurred_at_utc: '2026-07-30 00:00 UTC' },
      { ...real, attempt: 1, occurred_at_utc: '2026-07-29 08:59 UTC' },
    ]);
    assert(
      parsed.map(e => e.attempt).join(',') === '1,2,3',
      'string compare is chronological across hour and day rollovers',
    );
  }

  {
    // Timestamps are minute-precision, so a step failing in seconds produces ties.
    // Shape from aeh-0000000000000-000000000000000b, whose verify-response attempts took
    // 83s / 11s / 22s.
    const parsed = parseStepAttemptFailures([
      { ...real, attempt: 3, occurred_at_utc: '2026-07-29 00:49 UTC' },
      { ...real, attempt: 2, occurred_at_utc: '2026-07-29 00:49 UTC' },
      { ...real, attempt: 1, occurred_at_utc: '2026-07-29 00:49 UTC' },
    ]);
    assert(
      parsed.map(e => e.attempt).join(',') === '1,2,3',
      'same-minute entries fall back to attempt order',
    );
  }

  {
    // A partial entry with no timestamp must not be treated as the oldest and jump the queue.
    const parsed = parseStepAttemptFailures([
      { attempt: 9 },
      { ...real, attempt: 1, occurred_at_utc: '2026-07-29 00:10 UTC' },
    ]);
    assert(
      parsed.map(e => e.attempt).join(',') === '1,9',
      'entries with no timestamp sort last',
    );
  }

  {
    // Steps run sequentially, so chronological order groups each step's attempts together
    // without sorting on step at all. Real shape: `ask` failed once and recovered, then
    // `verify-response` exhausted all three attempts.
    const parsed = parseStepAttemptFailures([
      {
        ...real,
        step_id: 'verify-response',
        attempt: 2,
        occurred_at_utc: '2026-07-29 00:48 UTC',
      },
      {
        ...real,
        step_id: 'ask',
        attempt: 1,
        occurred_at_utc: '2026-07-29 00:19 UTC',
      },
      {
        ...real,
        step_id: 'verify-response',
        attempt: 3,
        occurred_at_utc: '2026-07-29 00:49 UTC',
      },
      {
        ...real,
        step_id: 'verify-response',
        attempt: 1,
        occurred_at_utc: '2026-07-29 00:47 UTC',
      },
    ]);
    assert(
      parsed.map(e => `${e.stepId}#${e.attempt}`).join(' ') ===
        'ask#1 verify-response#1 verify-response#2 verify-response#3',
      "chronological order groups a step's attempts together",
    );
  }

  {
    // Absent / slim-projection / malformed inputs must degrade, never throw.
    assert(parseStepAttemptFailures(undefined).length === 0, 'undefined -> []');
    assert(parseStepAttemptFailures(null).length === 0, 'null -> []');
    assert(
      parseStepAttemptFailures('not an array').length === 0,
      'non-array -> []',
    );
    assert(
      parseStepAttemptFailures([null, 'nope', 42]).length === 0,
      'drops non-object entries',
    );
  }

  {
    // A partial entry (only `attempt`) is what a pin-lagged writer could produce.
    const [entry] = parseStepAttemptFailures([{ attempt: 2 }]);
    assert(entry?.attempt === 2, 'partial entry keeps attempt');
    assert(entry?.stepId === null, 'partial entry nulls step_id');
    assert(entry?.durationS === null, 'partial entry nulls duration_s');
    assert(entry?.message === '', 'partial entry empties message');
  }

  {
    // Blank strings are as good as absent — they'd render as an empty ' · ' segment.
    const [entry] = parseStepAttemptFailures([
      { attempt: 1, step_id: '   ', error_class: '', error_message: ' x ' },
    ]);
    assert(entry?.stepId === null, 'whitespace-only step_id -> null');
    assert(entry?.errorClass === null, 'empty error_class -> null');
    assert(entry?.message === 'x', 'trims error_message');
  }

  {
    // Passed through untouched — it's already a display string. Anything that rewrote it
    // would have to cope with the literal "UTC" suffix.
    const [entry] = parseStepAttemptFailures([real]);
    assert(
      entry?.occurredAtUtc === '2026-07-28 18:53 UTC',
      'occurred_at_utc is passed through verbatim',
    );
  }

  {
    assert(ledgerTone('completed') === 'recovered', 'completed -> recovered');
    assert(ledgerTone('failed') === 'failed', 'failed -> failed');
    assert(ledgerTone('timed_out') === 'failed', 'timed_out -> failed');
    assert(ledgerTone('cancelled') === 'failed', 'cancelled -> failed');
    assert(ledgerTone('running') === 'in-progress', 'running -> in-progress');
    assert(ledgerTone('') === 'in-progress', 'unknown status -> in-progress');
  }

  {
    assert(
      ledgerSummary('recovered', 1) ===
        'Recovered after retry · 1 failed task step attempt',
      'recovered copy is singular for one entry',
    );
    assert(
      ledgerSummary('failed', 3) === 'Failed Task Step Details',
      'failed copy is a fixed heading, not a count',
    );
    assert(
      ledgerSummary('in-progress', 2) ===
        '2 failed task step attempts so far · may still be retrying',
      'in-progress copy does not claim recovery',
    );
  }

  if (failures > 0) {
    console.error(`\n${failures} assertion(s) failed`);
    process.exit(1);
  }
  console.log('\nall assertions passed');
}

main();
