import { XCircle } from 'lucide-react';
import { stepLabel, STEP_STATE_TEXT } from './step-progress';

interface StepRef {
  id: string;
  type?: string;
}

interface StepCount {
  done?: number;
  failed?: number;
}

export function RunGroupStepFunnel({
  steps,
  stepCounts,
  total,
}: {
  steps: StepRef[];
  stepCounts: Record<string, StepCount>;
  total: number;
}) {
  const known = new Set(steps.map(s => s.id));
  // Displayed-version steps first; tally-only ids (other task versions) appended.
  const rows = [
    ...steps.map(s => ({ id: s.id, label: stepLabel(s) })),
    ...Object.keys(stepCounts)
      .filter(id => !known.has(id))
      .map(id => ({ id, label: id })),
  ];
  if (rows.length === 0 || total === 0) return null;

  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
        Batch progress
      </div>
      <ol className="flex flex-wrap gap-x-4 gap-y-2">
        {rows.map(r => {
          const c = stepCounts[r.id] ?? {};
          const done = c.done ?? 0;
          const failed = c.failed ?? 0;
          return (
            <li key={r.id} className="min-w-[7rem] flex-1">
              <div className="mb-1 flex items-center justify-between gap-2 text-xs">
                <span
                  className="truncate font-mono text-[var(--foreground)]"
                  title={r.id}
                >
                  {r.label}
                </span>
                <span className="flex items-center gap-1 whitespace-nowrap">
                  <span
                    className={STEP_STATE_TEXT[done > 0 ? 'done' : 'pending']}
                  >
                    {done}/{total}
                  </span>
                  {failed > 0 && (
                    <span
                      className={`flex items-center gap-0.5 ${STEP_STATE_TEXT.failed}`}
                    >
                      <XCircle size={11} aria-hidden />
                      {failed}
                    </span>
                  )}
                </span>
              </div>
              <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-[var(--border)]">
                <div
                  className="bg-emerald-500"
                  style={{ width: `${(done / total) * 100}%` }}
                />
                <div
                  className="bg-red-500"
                  style={{ width: `${(failed / total) * 100}%` }}
                />
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
