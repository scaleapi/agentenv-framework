export type StepState = 'done' | 'failed' | 'running' | 'pending';

export function stepLabel(step: { id: string; type?: string }): string {
  return step.type || step.id;
}

export function isStepFailure(status?: string): boolean {
  return status === 'failure' || status === 'failed';
}

export const STEP_STATE_CHIP: Record<StepState, string> = {
  done: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-600',
  failed: 'border-red-500/30 bg-red-500/5 text-red-600',
  running: 'border-amber-500/40 bg-amber-500/10 text-amber-700',
  pending: 'border-[var(--border)] text-[var(--muted-foreground)]',
};

export const STEP_STATE_TEXT: Record<StepState, string> = {
  done: 'text-emerald-600',
  failed: 'text-red-600',
  running: 'text-amber-700',
  pending: 'text-[var(--muted-foreground)]',
};
