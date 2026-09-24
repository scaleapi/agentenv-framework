'use client';

import { AlertTriangle, ChevronDown, ChevronUp, XCircle } from 'lucide-react';
import {
  ledgerSummary,
  ledgerTone,
  parseStepAttemptFailures,
} from '../lib/step-attempt-failures';
import { cn } from '../lib/utils';

/** The per-attempt failure ledger for one instance. Renders nothing when empty, so callers drop it in
 *  unconditionally. Always collapsed (diagnostic detail). `nested` drops the panel chrome so it can sit inside the failure banner. */
export function StepAttemptFailures({
  failures,
  status,
  nested = false,
}: {
  /** Raw `instance.step_attempt_failures`; unparsed so the caller needs no knowledge of the shape. */
  failures: unknown;
  status: string;
  nested?: boolean;
}) {
  const entries = parseStepAttemptFailures(failures);
  if (entries.length === 0) return null;

  const tone = ledgerTone(status);
  const isFailed = tone === 'failed';
  const Icon = isFailed ? XCircle : AlertTriangle;

  return (
    // Native <details> for cheap, accessible collapsibility — same pattern as the
    // Task Prompt / Trajectory panels in task-instance-viewer.
    <details
      className={cn(
        'group',
        nested && 'mt-2 ml-6',
        !nested && 'border-b px-4 py-3 text-sm',
        !nested &&
          (isFailed
            ? 'border-red-500/30 bg-red-500/5'
            : 'border-[var(--border)] bg-[var(--secondary)]'),
      )}
    >
      <summary
        className={cn(
          'flex items-center gap-2 cursor-pointer select-none font-medium',
          nested && 'text-xs',
          isFailed ? 'text-red-500' : 'text-[var(--muted-foreground)]',
        )}
      >
        <ChevronDown size={12} className="flex-shrink-0 group-open:hidden" />
        <ChevronUp
          size={12}
          className="flex-shrink-0 hidden group-open:block"
        />
        {/* When nested, the banner's own "Failed" heading already carries the tone icon —
            a second one directly beneath it is just noise. */}
        {!nested && <Icon size={16} className="flex-shrink-0" aria-hidden />}
        {ledgerSummary(tone, entries.length)}
      </summary>
      <div className={cn('mt-2 space-y-2', !nested && 'ml-6')}>
        {entries.map((entry, i) => {
          const step = entry.stepId ?? 'unknown';
          // Step first, then attempt: which step failed is what you scan for, and rows are
          // in chronological order, which puts a step's attempts next to each other.
          const meta = [
            entry.stepType ? `${step} (${entry.stepType})` : step,
            `attempt ${entry.attempt ?? '?'}`,
            entry.errorClass,
            entry.durationS === null ? null : `${entry.durationS.toFixed(1)}s`,
            // Rendered verbatim: the stored value is already `... UTC`, not an ISO timestamp.
            entry.occurredAtUtc,
          ].filter((part): part is string => part !== null);
          return (
            <div key={i} className="text-xs">
              <div className="font-mono text-[var(--muted-foreground)]">
                {meta.join(' · ')}
              </div>
              <pre className="mt-0.5 whitespace-pre-wrap break-words font-mono text-[var(--foreground)]">
                {entry.message || '(no detail recorded)'}
              </pre>
            </div>
          );
        })}
      </div>
    </details>
  );
}
