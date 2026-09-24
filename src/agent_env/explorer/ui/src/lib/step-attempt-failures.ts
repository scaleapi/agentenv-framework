/** Per-attempt failure ledger (TaskInstance.step_attempt_failures): one entry per failed activity attempt.
 *  Needed because an instance's terminal status/error only describes the LAST attempt — a step that failed on
 *  attempt 1 and passed on 2 leaves the instance `completed` with error null. Pure logic, testable without React. */

export interface StepAttemptFailure {
  attempt: number | null;
  stepId: string | null;
  stepType: string | null;
  /** The Python exception class (`ValueError`). Not the same notion as `PromptResponse.error_type`, which is the agent's semantic category (`provider_error`). */
  errorClass: string | null;
  message: string;
  durationS: number | null;
  /** Already a display string, not a machine timestamp: the worker writes strftime('%Y-%m-%d %H:%M UTC'), e.g.
   *  `2026-07-29 00:14 UTC`. Render as-is — there's no offset to interpret and reformatting mangles the "UTC" suffix. */
  occurredAtUtc: string | null;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

/** Parse instance.step_attempt_failures. Every field is optional, so anything unreadable degrades to null;
 *  a slim projection omitting the field yields []. Ordered oldest-first (chronological): steps run sequentially, so a step's retries group together. */
export function parseStepAttemptFailures(raw: unknown): StepAttemptFailure[] {
  if (!Array.isArray(raw)) return [];
  const entries = raw
    .map((entry): StepAttemptFailure | null => {
      if (!entry || typeof entry !== 'object') return null;
      const e = entry as Record<string, unknown>;
      return {
        attempt: typeof e.attempt === 'number' ? e.attempt : null,
        stepId: nonEmptyString(e.step_id),
        stepType: nonEmptyString(e.step_type),
        errorClass: nonEmptyString(e.error_class),
        message: nonEmptyString(e.error_message) ?? '',
        durationS: typeof e.duration_s === 'number' ? e.duration_s : null,
        occurredAtUtc: nonEmptyString(e.occurred_at_utc),
      };
    })
    .filter((e): e is StepAttemptFailure => e !== null);

  return entries.sort((a, b) => {
    // A plain string compare is chronological: the stored format is fixed-width, zero-padded, so lexicographic
    // order matches time order (no Date parsing needed). Timestamp-less entries sort last.
    const aTime = a.occurredAtUtc ?? '￿';
    const bTime = b.occurredAtUtc ?? '￿';
    if (aTime !== bTime) return aTime < bTime ? -1 : 1;
    // Stored precision is only to the minute, so attempts of a fast-failing step can share a
    // timestamp — attempt number breaks the tie.
    return (a.attempt ?? 0) - (b.attempt ?? 0);
  });
}

/** What the entries *mean* depends on where the run ended up, so the copy has to follow. */
export type LedgerTone = 'failed' | 'recovered' | 'in-progress';

export function ledgerTone(status: string): LedgerTone {
  if (status === 'completed') return 'recovered';
  if (status === 'failed' || status === 'cancelled' || status === 'timed_out') {
    return 'failed';
  }
  // Still running: entries exist for a step that may yet fail, so it isn't a recovery.
  return 'in-progress';
}

export function ledgerSummary(tone: LedgerTone, count: number): string {
  const attempts = `${count} failed task step attempt${count === 1 ? '' : 's'}`;
  if (tone === 'recovered') return `Recovered after retry · ${attempts}`;
  if (tone === 'in-progress')
    return `${attempts} so far · may still be retrying`;
  // Fixed heading rather than a count: this sits directly under the banner's "Failed",
  // which already establishes the outcome.
  return 'Failed Task Step Details';
}
