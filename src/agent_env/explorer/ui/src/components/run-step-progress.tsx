import { useEffect, useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Circle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { BACKEND_URL, apiFetch } from './shared';
import {
  type StepState,
  stepLabel,
  isStepFailure,
  STEP_STATE_CHIP,
} from './step-progress';

interface StepRef {
  id: string;
  type?: string;
}

interface CompletedStep {
  step_id: string;
  status?: string;
}

const POLL_MS = 3000;
const MAX_BACKOFF_MS = 30000;
const TERMINAL_FAIL = ['failed', 'cancelled', 'timed_out'];

/** Per-run step progress. Polls `/progress` for truth (status + completed_steps + live step index) rather than the `instance` prop, which is the slim rollouts projection. */
export function RunStepProgress({
  steps,
  taskId,
  instanceId,
  status: initialStatus,
  completedSteps: initialCompleted,
  totalSteps,
}: {
  steps: StepRef[];
  taskId?: string;
  instanceId?: string;
  status?: string;
  completedSteps?: CompletedStep[];
  totalSteps?: number;
}) {
  const [status, setStatus] = useState(initialStatus ?? '');
  const [completed, setCompleted] = useState<CompletedStep[]>(
    initialCompleted ?? [],
  );
  const [liveStep, setLiveStep] = useState<number | null>(null);
  const [userToggled, setUserToggled] = useState<boolean | null>(null);

  useEffect(() => {
    if (!taskId || !instanceId) return;
    let active = true;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let delay = POLL_MS;
    const schedule = () => {
      timer = setTimeout(poll, delay);
    };
    const poll = () => {
      apiFetch(
        `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(taskId)}/instances/${encodeURIComponent(instanceId)}/progress`,
        { signal: controller.signal },
      )
        .then(r =>
          r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)),
        )
        .then(
          (d: {
            status?: string;
            completed_steps?: CompletedStep[];
            live_step_index?: number | null;
            done?: boolean;
          }) => {
            if (!active) return;
            delay = POLL_MS;
            setStatus(d.status ?? '');
            setCompleted(d.completed_steps ?? []);
            setLiveStep(
              typeof d.live_step_index === 'number' ? d.live_step_index : null,
            );
            if (!d.done) schedule();
          },
        )
        .catch(() => {
          // Backoff-retry on transient errors so one hiccup doesn't freeze a live run.
          if (active) {
            delay = Math.min(delay * 2, MAX_BACKOFF_MS);
            schedule();
          }
        });
    };
    poll();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [taskId, instanceId]);

  const running = status === 'running';
  const byId = new Map(completed.map(c => [c.step_id, c.status ?? 'success']));
  const named = steps.length > 0;
  const total = named ? steps.length : totalSteps ?? completed.length;

  const rows = named
    ? steps.map(s => ({
        key: s.id,
        label: stepLabel(s),
        title: s.id,
        done: byId.get(s.id),
      }))
    : Array.from({ length: total }, (_, i) => ({
        key: `step-${i}`,
        label: `Step ${i + 1}`,
        title: `Step ${i + 1}`,
        done:
          i < completed.length ? completed[i]?.status ?? 'success' : undefined,
      }));

  const firstPending = rows.findIndex(r => r.done === undefined);
  const runningIdx = liveStep ?? firstPending;

  if (total === 0) return null;

  const stateOf = (r: (typeof rows)[number], idx: number): StepState => {
    if (isStepFailure(r.done)) return 'failed';
    if (r.done !== undefined) return 'done';
    // Live query can lead completed_steps: steps before the running one are done.
    if (running && runningIdx > idx) return 'done';
    if (running && idx === runningIdx) return 'running';
    return 'pending';
  };
  const states = rows.map((r, i) => stateOf(r, i));
  const doneCount = states.filter(s => s === 'done').length;

  const failedIdx = states.indexOf('failed');
  const failedLabel = failedIdx >= 0 ? rows[failedIdx]?.label : undefined;
  const hasFail = failedIdx >= 0 || TERMINAL_FAIL.includes(status);
  const active = running || status === 'provisioning';
  // Auto-expand while in-progress/failed; a manual toggle overrides.
  const open = userToggled ?? (active || hasFail);

  return (
    <div className="border-b border-[var(--border)] px-4 py-3">
      <button
        type="button"
        onClick={() => setUserToggled(!open)}
        className="flex w-full items-center gap-1.5 text-left"
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
          Run progress
        </span>
        <span className="text-xs text-[var(--muted-foreground)]">
          · {doneCount}/{total} steps
        </span>
        {hasFail ? (
          <span className="ml-1 flex items-center gap-1 text-xs text-red-600">
            <XCircle size={12} aria-hidden />
            {failedLabel ? `failed at ${failedLabel}` : status}
          </span>
        ) : doneCount === total ? (
          <CheckCircle2
            size={12}
            className="ml-1 text-emerald-600"
            aria-hidden
          />
        ) : active ? (
          <Loader2
            size={12}
            className="ml-1 animate-spin text-amber-600"
            aria-hidden
          />
        ) : null}
        <span className="ml-auto text-[var(--muted-foreground)]">
          {open ? (
            <ChevronDown size={14} aria-hidden />
          ) : (
            <ChevronRight size={14} aria-hidden />
          )}
        </span>
      </button>
      {open && (
        <ol className="mt-2 flex flex-wrap gap-1.5">
          {rows.map((r, idx) => {
            const state = states[idx] ?? 'pending';
            return (
              <li
                key={r.key}
                title={r.title}
                className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${STEP_STATE_CHIP[state]}`}
              >
                {state === 'done' ? (
                  <CheckCircle2 size={12} aria-hidden />
                ) : state === 'failed' ? (
                  <XCircle size={12} aria-hidden />
                ) : state === 'running' ? (
                  <Loader2 size={12} className="animate-spin" aria-hidden />
                ) : (
                  <Circle size={12} aria-hidden />
                )}
                <span className="font-mono">{r.label}</span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
