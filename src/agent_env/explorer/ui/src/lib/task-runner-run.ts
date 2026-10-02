/** The task runner's run lifecycle without React: the request a run sends, how a failed launch
 *  surfaces, how a poll of the task's instances is reconciled, and when polling may stop. */

export type Instance = Record<string, unknown>;
export type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export interface RunOptions {
  version?: number;
  start_step?: number;
  context_json?: Record<string, unknown> | null;
  // Prefer over context_json when re-running one of this task's instances:
  // `instances` comes from GET /instances, which strips a2a_card.
  context_from_instance_id?: string;
  // From-start re-run: inherit only the prior run's user_overrides.
  overrides_from_instance_id?: string;
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

export function isTerminal(inst: Instance): boolean {
  return TERMINAL_STATUSES.has(String(inst.status ?? ''));
}

export function allTerminal(instances: Instance[]): boolean {
  return instances.length > 0 && instances.every(isTerminal);
}

function taskUrl(base: string, taskId: string): string {
  return `${base}/api/v1/tasks/${encodeURIComponent(taskId)}`;
}

export function buildRunBody(
  opts: RunOptions | undefined,
  taskVersion: number | null,
  projectId: string | null,
): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  body.version = opts?.version ?? taskVersion ?? undefined;
  if (opts?.start_step != null) body.start_step = opts.start_step;
  // Mutually exclusive server-side.
  if (opts?.context_from_instance_id)
    body.context_from_instance_id = opts.context_from_instance_id;
  else if (opts?.context_json) body.context_json = opts.context_json;
  if (opts?.overrides_from_instance_id)
    body.overrides_from_instance_id = opts.overrides_from_instance_id;
  // Only-if-truthy so "" can't defeat the backend fallback.
  if (projectId) body.project_id = projectId;
  return body;
}

/** POST /run. Resolves to the run's workflow id; a refused launch throws the backend's
 *  `detail` (e.g. "Budget has been exceeded!") rather than a bare status code. */
export async function startRun(
  fetchFn: Fetch,
  base: string,
  taskId: string,
  body: Record<string, unknown>,
): Promise<string | null> {
  const res = await fetchFn(`${taskUrl(base, taskId)}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Failed to start run (${res.status})`);
  }
  const data = await res.json();
  return (data.workflow_id as string | null) ?? null;
}

/** Cancel by the run's workflow id, the only cancel route there is. */
export async function cancelRun(
  fetchFn: Fetch,
  base: string,
  taskId: string,
  workflowId: string,
): Promise<void> {
  const res = await fetchFn(
    `${taskUrl(base, taskId)}/cancel-run?workflow_id=${encodeURIComponent(workflowId)}`,
    { method: 'POST' },
  );
  if (!res.ok) throw new Error(`Cancel failed (${res.status})`);
}

export function workflowIdOf(inst: Instance | undefined): string | undefined {
  const context = inst?.context as Instance | undefined;
  const metadata = context?.metadata as Instance | undefined;
  return metadata?.workflow_id as string | undefined;
}

export interface InstancePoll {
  snapshot: string;
  ids: Set<string>;
  /** The run the page was waiting on, once it appears; clears "Starting…". */
  startedInstanceId: string | null;
  /** The newest completed instance, whose context seeds a re-run. */
  latestCompleted: Instance | null;
}

/** Reconcile one GET /instances page against the last one; null when nothing shown changed. */
export function reconcileInstances(
  items: Instance[],
  previousSnapshot: string,
  knownIds: Set<string>,
  pendingRun: boolean,
): InstancePoll | null {
  const snapshot = JSON.stringify(
    items.map(i => `${i.instance_id}:${i.status}:${i.current_step}`),
  );
  if (snapshot === previousSnapshot) return null;
  const started = pendingRun
    ? items.find(i => !knownIds.has(String(i.instance_id)))
    : undefined;
  return {
    snapshot,
    ids: new Set(items.map(i => String(i.instance_id))),
    startedInstanceId: started ? String(started.instance_id) : null,
    latestCompleted: items.find(i => i.status === 'completed') ?? null,
  };
}

/** Poll while a run is live or a started run has not appeared yet. 'error'/'failed' are
 *  terminal: a failed init can leave a task id with no instances, so the instance-derived
 *  stop would never trip. */
export function shouldPoll(
  taskId: string | null,
  phase: string,
  instancesAllTerminal: boolean,
  pendingRun: boolean,
): boolean {
  if (!taskId || phase === 'initializing' || phase === 'error' || phase === 'failed')
    return false;
  return !(instancesAllTerminal && !pendingRun);
}
