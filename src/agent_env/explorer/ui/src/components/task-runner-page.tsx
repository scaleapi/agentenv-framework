import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Play, Loader2, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { selectFinalScore } from '../lib/verifier-classification';
import { BACKEND_URL, apiFetch } from './shared';
import { StepsPipeline } from './steps-pipeline';
import { TaskInstanceViewer, type TaskStepRef } from './task-instance-viewer';
import { type StepType, EVALUATOR_STEP_TYPES } from './task-steps-shared';

type Phase =
  | 'initializing'
  | 'idle'
  | 'running'
  | 'completed'
  | 'failed'
  | 'error';

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600)
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function getInstanceScore(inst: Record<string, unknown>): number | null {
  if (inst.status !== 'completed') return null;
  const context = inst.context as Record<string, unknown> | null;
  if (!context) return null;
  const metadata = context.metadata as Record<string, unknown> | undefined;
  return selectFinalScore(
    metadata?.verifications as Record<string, unknown> | undefined,
  );
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

function isTerminal(inst: Record<string, unknown>) {
  return TERMINAL_STATUSES.has(String(inst.status ?? ''));
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function TaskRunnerPage({
  taskId: urlTaskId,
  onEditEvaluator,
  onRerunFromStep,
  savedContextJson: externalContextJson,
  onContextCaptured,
}: {
  taskId?: string | null;
  onEditEvaluator?: (taskId: string) => void;
  onRerunFromStep?: (taskId: string, stepIndex: number) => void;
  savedContextJson?: Record<string, unknown> | null;
  onContextCaptured?: (ctx: Record<string, unknown>) => void;
}) {
  /* --- core state --- */
  const [phase, setPhase] = useState<Phase>('initializing');
  const [resolvedTaskId, setResolvedTaskId] = useState<string | null>(null);
  const [taskVersion, setTaskVersion] = useState<number | null>(null);
  const [task, setTask] = useState<Record<string, unknown> | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [lastContextJson, setLastContextJsonRaw] = useState<Record<
    string,
    unknown
  > | null>(externalContextJson ?? null);
  // Instance the captured context came from (when from this task's runs). Tagged with its task id since loading is task-scoped.
  const [lastContextSource, setLastContextSource] = useState<{
    taskId: string;
    instanceId: string;
  } | null>(null);

  // Wrap setter to also report to parent
  const setLastContextJson = useCallback(
    (ctx: Record<string, unknown> | null) => {
      setLastContextJsonRaw(ctx);
      if (ctx) onContextCaptured?.(ctx);
    },
    [onContextCaptured],
  );

  /* --- instances --- */
  const [instances, setInstances] = useState<Record<string, unknown>[]>([]);
  const [selectedInstanceId, setSelectedInstanceId] = useState<string | null>(
    null,
  );
  const [pendingRun, setPendingRun] = useState(false);
  const lastWorkflowIdRef = useRef<string | null>(null);
  const snapshotRef = useRef('');

  // Render-synced refs so callbacks can read the latest values without re-subscribing.
  const instancesRef = useRef(instances);
  instancesRef.current = instances;
  const resolvedTaskIdRef = useRef(resolvedTaskId);
  resolvedTaskIdRef.current = resolvedTaskId;
  // The page is not remounted per task, so it initialises once per mount.
  const initializedRef = useRef(false);

  const setLastContextJsonRef = useRef(setLastContextJson);
  setLastContextJsonRef.current = setLastContextJson;

  /* --- fetch task metadata --- */
  const fetchTask = useCallback(async (id: string) => {
    const res = await apiFetch(
      `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(id)}`,
    );
    if (!res.ok) throw new Error(`Failed to fetch task (${res.status})`);
    return res.json();
  }, []);

  /* --- poll instances from API --- */
  const instanceIdsRef = useRef<Set<string>>(new Set());
  const pendingRunRef = useRef(false);
  pendingRunRef.current = pendingRun;

  const pollInstances = useCallback(async () => {
    if (!resolvedTaskId) return;
    try {
      const params = new URLSearchParams({
        limit: '100',
        sort: 'created_at_utc',
        order: 'desc',
      });
      const res = await apiFetch(
        `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
          resolvedTaskId,
        )}/instances?${params}`,
      );
      if (!res.ok) return;
      const data = await res.json();
      // Everything below writes "current task's runs", so a response for a task this page has since
      // moved off must be dropped, not applied.
      if (resolvedTaskIdRef.current !== resolvedTaskId) return;
      const items = (data.items ?? []) as Record<string, unknown>[];
      const snap = JSON.stringify(
        items.map(i => `${i.instance_id}:${i.status}:${i.current_step}`),
      );
      if (snap === snapshotRef.current) return;

      // Detect new instance for pending run
      if (pendingRunRef.current) {
        const newInst = items.find(
          i => !instanceIdsRef.current.has(String(i.instance_id)),
        );
        if (newInst) {
          setPendingRun(false);
          lastWorkflowIdRef.current = null;
          setSelectedInstanceId(String(newInst.instance_id));
        }
      }

      // Capture context from most recent completed instance
      const completed = items.find(
        (inst: Record<string, unknown>) => inst.status === 'completed',
      );
      if (completed) {
        const ctx = completed.context as Record<string, unknown> | undefined;
        // `ctx` is from the list endpoint (a2a_card stripped) — keep the id instead.
        if (ctx) setLastContextJsonRef.current(ctx);
        setLastContextSource({
          taskId: resolvedTaskId,
          instanceId: String(completed.instance_id),
        });
      }

      snapshotRef.current = snap;
      instanceIdsRef.current = new Set(items.map(i => String(i.instance_id)));
      setInstances(items);
    } catch {
      /* ignore */
    }
  }, [resolvedTaskId]);

  // Poll while a run is active; stop once every instance is terminal. Deriving stop from instance status
  // (not phase, which never goes terminal here) lets a fresh run flip polling back on.
  const allInstancesTerminal =
    instances.length > 0 && instances.every(isTerminal);
  useEffect(() => {
    if (
      !resolvedTaskId ||
      phase === 'initializing' ||
      // 'error'/'failed' are terminal: a failed init can leave resolvedTaskId
      // set with no instances, so the instance-derived stop below never trips.
      phase === 'error' ||
      phase === 'failed'
    )
      return;
    if (allInstancesTerminal && !pendingRun) return;
    pollInstances();
    const interval = setInterval(pollInstances, 5000);
    return () => clearInterval(interval);
  }, [resolvedTaskId, phase, pollInstances, allInstancesTerminal, pendingRun]);

  /* --- run task --- */
  const buildRunBody = useCallback(
    (opts?: {
      version?: number;
      start_step?: number;
      context_json?: Record<string, unknown> | null;
      // Prefer over context_json when re-running one of this task's instances:
      // `instances` comes from GET /instances, which strips a2a_card.
      context_from_instance_id?: string;
      // From-start re-run: inherit only the prior run's user_overrides.
      overrides_from_instance_id?: string;
    }) => {
      const body: Record<string, unknown> = {};
      body.version = opts?.version ?? taskVersion ?? undefined;
      // priority=0 (interactive): a human is waiting. Sent explicitly so intent survives backend default changes.
      body.priority = 0;
      if (opts?.start_step != null) body.start_step = opts.start_step;
      // Mutually exclusive server-side.
      if (opts?.context_from_instance_id)
        body.context_from_instance_id = opts.context_from_instance_id;
      else if (opts?.context_json) body.context_json = opts.context_json;
      if (opts?.overrides_from_instance_id)
        body.overrides_from_instance_id = opts.overrides_from_instance_id;
      // Only-if-truthy so "" can't defeat the backend fallback.
      const projectId =
        new URLSearchParams(window.location.search).get('projectId') ||
        undefined;
      if (projectId) body.project_id = projectId;
      return body;
    },
    [taskVersion],
  );

  const handleRun = useCallback(
    async (opts?: {
      version?: number;
      start_step?: number;
      context_json?: Record<string, unknown> | null;
      context_from_instance_id?: string;
      overrides_from_instance_id?: string;
    }) => {
      if (!resolvedTaskId || pendingRun) return;
      setPendingRun(true);
      setErrorMsg(null);
      try {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            resolvedTaskId,
          )}/run`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(buildRunBody(opts)),
          },
        );
        if (!res.ok) {
          // Surface the backend's `detail` (e.g. "Budget has been exceeded!") instead of a bare status code.
          const errData = await res.json().catch(() => ({}));
          throw new Error(
            errData.detail || `Failed to start run (${res.status})`,
          );
        }
        const data = await res.json();
        lastWorkflowIdRef.current = (data.workflow_id as string | null) ?? null;
      } catch (e) {
        lastWorkflowIdRef.current = null;
        setPendingRun(false);
        setErrorMsg(e instanceof Error ? e.message : 'Failed to start run');
      }
    },
    [resolvedTaskId, pendingRun, buildRunBody],
  );

  /* --- cancel instance --- */
  const handleCancel = useCallback(
    async (instanceId: string) => {
      if (!resolvedTaskId) return;
      // Cancel by the run's workflow_id (the route that exists) — there is no
      // per-instance cancel route. Read it off the instance's context.metadata.
      const inst = instancesRef.current.find(
        i => String(i.instance_id) === instanceId,
      );
      const context = inst?.context as Record<string, unknown> | undefined;
      const metadata = context?.metadata as Record<string, unknown> | undefined;
      const workflowId = metadata?.workflow_id as string | undefined;
      if (!workflowId) return;
      try {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            resolvedTaskId,
          )}/cancel-run?workflow_id=${encodeURIComponent(workflowId)}`,
          { method: 'POST' },
        );
        if (!res.ok) throw new Error(`Cancel failed (${res.status})`);
        // Mark cancelled only after the backend confirms — no false success.
        setInstances(prev =>
          prev.map(i =>
            String(i.instance_id) === instanceId
              ? { ...i, status: 'cancelled' }
              : i,
          ),
        );
      } catch (e) {
        console.warn('Cancel instance failed', e);
      }
    },
    [resolvedTaskId],
  );

  /* --- cancel pending run (before instance appears) --- */
  const handleCancelPending = useCallback(async () => {
    if (!resolvedTaskId || !lastWorkflowIdRef.current) return;
    setPendingRun(false);
    try {
      await apiFetch(
        `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
          resolvedTaskId,
        )}/cancel-run?workflow_id=${encodeURIComponent(
          lastWorkflowIdRef.current,
        )}`,
        { method: 'POST' },
      );
    } catch {
      /* best-effort */
    }
    lastWorkflowIdRef.current = null;
  }, [resolvedTaskId]);

  /* --- initialization effect --- */
  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;
    if (!urlTaskId) {
      setPhase('error');
      setErrorMsg(
        'No task ID provided. Pass a task ID in the URL (/task-runner/<task-id>).',
      );
      return;
    }
    setResolvedTaskId(urlTaskId);
    fetchTask(urlTaskId)
      .then(data => {
        setTask(data);
        setTaskVersion(data.version ?? null);
        setPhase('idle');
      })
      .catch(e => {
        setPhase('error');
        setErrorMsg(e instanceof Error ? e.message : 'Failed to load task');
      });
  }, [urlTaskId, fetchTask]);

  // Ref to always access the latest handleEvaluatorSaved without stale closures
  const handleEvaluatorSavedRef = useRef<
    | ((taskId: string, version: number, startStep?: number) => Promise<void>)
    | null
  >(null);

  // One run at a time: Run is blocked while one is active; per-run Cancel escapes.
  const activeRunExists = pendingRun || instances.some(i => !isTerminal(i));

  /* --- handle return from evaluator editor --- */
  // Store pending eval rerun params so the effect below can fire startRun
  // after resolvedTaskId state has updated.
  const [pendingEvalRun, setPendingEvalRun] = useState<{
    version: number;
    startStep: number;
    // Unset for an externally supplied context, which belongs to no instance.
    contextInstanceId?: string;
    contextJson: Record<string, unknown>;
  } | null>(null);

  const handleEvaluatorSaved = useCallback(
    async (
      savedTaskId: string,
      newVersion: number,
      overrideStartStep?: number,
    ) => {
      setResolvedTaskId(savedTaskId);
      setTaskVersion(newVersion);
      setInstances([]);
      snapshotRef.current = '';

      // Refresh task data
      try {
        const data = await fetchTask(savedTaskId);
        setTask(data);

        // Use override start step, or find first evaluator step
        const stepDicts = (data.steps ?? []) as Record<string, unknown>[];
        const evalStartStep =
          overrideStartStep ??
          stepDicts.findIndex(s =>
            EVALUATOR_STEP_TYPES.has(s.type as StepType),
          );

        // Only usable if it belongs to the task being saved.
        const contextInstanceId =
          lastContextSource?.taskId === savedTaskId
            ? lastContextSource.instanceId
            : undefined;
        // A captured context belongs to lastContextSource's task; only the externalContextJson prop is task-agnostic.
        const fallbackContext = lastContextSource ? null : lastContextJson;

        if (evalStartStep >= 0 && (contextInstanceId || fallbackContext)) {
          // Defer the run to next render when resolvedTaskId is updated
          setPhase('running');

          setErrorMsg(null);
          setPendingEvalRun({
            version: newVersion,
            startStep: evalStartStep,
            contextInstanceId,
            contextJson: fallbackContext ?? {},
          });
        } else {
          // No context or no evaluator step — will do full re-run after state updates
          setPhase('running');

          setErrorMsg(null);
          setPendingEvalRun({
            version: newVersion,
            startStep: 0,
            contextJson: {},
          });
        }
      } catch (e) {
        setPhase('error');
        setErrorMsg(
          e instanceof Error ? e.message : 'Failed to run after save',
        );
      }
    },
    [fetchTask, lastContextJson, lastContextSource],
  );
  handleEvaluatorSavedRef.current = handleEvaluatorSaved;

  // Fire handleRun after resolvedTaskId state has updated
  useEffect(() => {
    if (!pendingEvalRun || !resolvedTaskId) return;
    const { version, startStep, contextInstanceId, contextJson } =
      pendingEvalRun;
    setPendingEvalRun(null);
    if (startStep > 0 && contextInstanceId) {
      handleRun({
        version,
        start_step: startStep,
        context_from_instance_id: contextInstanceId,
      });
    } else if (startStep > 0 && Object.keys(contextJson).length > 0) {
      handleRun({ version, start_step: startStep, context_json: contextJson });
    } else {
      handleRun({ version });
    }
  }, [pendingEvalRun, resolvedTaskId, handleRun]);

  /* --- derived state --- */
  const taskSteps = (task?.steps ?? []) as Record<string, unknown>[];

  // Slim {id, type} list for TaskInstanceViewer (groups collected artifacts by collect step). Memoized off
  // `task` so identity is stable across polls, else the viewer's trajectory effect re-fires and collapses trajectories.
  const viewerTaskSteps: TaskStepRef[] = useMemo(() => {
    const rawSteps = (task?.steps ?? []) as Record<string, unknown>[];
    return rawSteps.map(s => ({
      id: String(s.id ?? ''),
      prompt_id: (s.prompt_id ?? null) as string | null,
      type: (s.type ?? undefined) as string | undefined,
      base_path: (s.base_path ?? undefined) as string | undefined,
      artifact_paths: (s.artifact_paths ?? undefined) as string[] | undefined,
      env_id: (s.env_id ?? null) as string | null,
      agent_name: (s.agent_name ?? null) as string | null,
      triggers: s.triggers,
    }));
  }, [task]);

  const rubricsStep = taskSteps.find(s => s.type === 'rubrics_verifier') as
    | Record<string, unknown>
    | undefined;
  const rubricsCriteria = rubricsStep?.criteria as
    | Record<string, unknown>[]
    | undefined;
  // Forward to RubricGradingResults so the Score badge can be hidden for all_pass (redundant with the pass count). Undefined → all_pass.
  const rubricsAggregator = rubricsStep?.score_aggregator as string | undefined;

  /* ------------------------------------------------------------------ */
  /*  Render                                                             */
  /* ------------------------------------------------------------------ */

  return (
    <div className="p-8 flex flex-col h-full overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold">Task Runner</h1>
        </div>
        <div className="flex items-center gap-2">
          {resolvedTaskId && (
            <button
              onClick={() => {
                snapshotRef.current = '';
                pollInstances();
              }}
              title="Refresh instances"
              className="flex items-center gap-1.5 px-2 py-1.5 rounded-md border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
            >
              <RefreshCw size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Task ID */}
      {resolvedTaskId && (
        <div className="mb-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
            Task ID
          </span>
          <p className="font-mono text-sm mt-0.5">
            {resolvedTaskId}
            {taskVersion != null && (
              <span className="ml-2 text-xs font-mono text-[var(--muted-foreground)] bg-[var(--secondary)] px-2 py-0.5 rounded">
                v{taskVersion}
              </span>
            )}
          </p>
        </div>
      )}

      {/* Error message */}
      {errorMsg && (
        <div className="mb-4 p-3 rounded-lg border border-red-300 bg-red-50 text-sm text-red-700">
          {errorMsg}
        </div>
      )}

      {/* Pipeline visualization */}
      {taskSteps.length > 0 && (
        <div className="mb-6">
          <StepsPipeline steps={taskSteps} />
        </div>
      )}

      {/* Initializing */}
      {phase === 'initializing' && (
        <div className="mb-6 flex items-center gap-3 p-4 rounded-lg border border-[var(--border)] bg-[var(--secondary)]">
          <Loader2 size={20} className="animate-spin text-gray-500" />
          <p className="text-sm font-medium">Initializing...</p>
        </div>
      )}

      {/* Runs */}
      {resolvedTaskId && phase !== 'initializing' && (
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Runs ({instances.length})
            </h3>
          </div>

          {/* Run / Starting button */}
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {pendingRun ? (
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--border)] bg-[var(--secondary)]">
                <Loader2 size={14} className="animate-spin text-amber-500" />
                <span className="text-sm text-[var(--muted-foreground)]">
                  Starting...
                </span>
                <button
                  onClick={handleCancelPending}
                  className="ml-2 px-2 py-0.5 rounded text-[11px] font-medium border border-red-300 text-red-600 hover:bg-red-50 transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => handleRun()}
                disabled={activeRunExists}
                className="flex items-center gap-1.5 px-4 py-2 rounded-md bg-violet-600 text-white text-sm font-medium hover:bg-violet-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play size={14} />
                {activeRunExists ? 'Run in progress' : 'Run'}
              </button>
            )}
          </div>

          {/* Instance list — horizontal scroll */}
          <div className="flex gap-2 mb-4 overflow-x-auto pb-3">
            {instances.map(inst => {
              const id = String(inst.instance_id);
              const score = getInstanceScore(inst);
              const status = String(inst.status ?? 'unknown');
              // `provisioning` is the pre-worker window (env still deploying); treat it as in-progress like `running`.
              const isActive =
                status === 'running' || status === 'provisioning';
              const isSelected = selectedInstanceId === id;
              return (
                <button
                  key={id}
                  onClick={() => setSelectedInstanceId(id)}
                  className={`flex flex-col gap-1 p-3 rounded-lg border text-left transition-colors flex-shrink-0 w-48 ${
                    isSelected
                      ? 'border-violet-600 bg-violet-50'
                      : 'border-[var(--border)] hover:bg-[var(--accent)]'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    {status === 'completed' && score != null && score >= 1 ? (
                      <CheckCircle2 size={14} className="text-green-500" />
                    ) : status === 'completed' && score != null ? (
                      <XCircle size={14} className="text-red-500" />
                    ) : status === 'completed' ? (
                      <CheckCircle2 size={14} className="text-gray-400" />
                    ) : status === 'failed' || status === 'cancelled' ? (
                      <XCircle size={14} className="text-red-500" />
                    ) : isActive ? (
                      <Loader2
                        size={14}
                        className="animate-spin text-amber-500"
                      />
                    ) : (
                      <XCircle size={14} className="text-gray-400" />
                    )}
                    <span className="text-xs font-mono font-medium truncate">
                      {id}
                    </span>
                    {inst.task_version != null && (
                      <span className="text-[10px] font-mono text-[var(--muted-foreground)] bg-[var(--secondary)] px-1.5 py-0.5 rounded flex-shrink-0">
                        v{String(inst.task_version)}
                      </span>
                    )}
                  </div>
                  {isActive &&
                    inst.current_step != null &&
                    inst.total_steps != null && (
                      <span className="text-[11px] text-amber-600">
                        {/* current_step is a COUNT of completed steps
                            (agent-env sets it to len(completed_steps)), not a
                            0-based index. Render completed/total to match the
                            RunStepProgress panel; the old `current_step + 1`
                            showed a still-running instance as a full "2/2"
                            that read as done and contradicted that panel. */}
                        {Number(inst.current_step)}/{Number(inst.total_steps)}{' '}
                        steps
                      </span>
                    )}
                  {score != null && (
                    <span
                      className={`text-[11px] font-semibold px-1.5 py-0.5 rounded w-fit ${
                        score >= 1
                          ? 'bg-green-500/10 text-green-500'
                          : 'bg-red-500/10 text-red-500'
                      }`}
                    >
                      {Math.round(score * 100)}%
                    </span>
                  )}
                  <span className="text-[11px] text-[var(--muted-foreground)]">
                    {formatDuration(inst.duration_seconds as number | null)}
                  </span>
                  {isActive && (
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        handleCancel(id);
                      }}
                      className="mt-1 px-2 py-0.5 rounded text-[11px] font-medium border border-red-300 text-red-600 hover:bg-red-50 transition-colors"
                    >
                      Cancel
                    </button>
                  )}
                </button>
              );
            })}
          </div>

          {/* Selected instance viewer */}
          {selectedInstanceId &&
            (() => {
              const inst = instances.find(
                i => String(i.instance_id) === selectedInstanceId,
              );
              if (!inst) return null;
              return (
                <TaskInstanceViewer
                  key={selectedInstanceId}
                  instance={inst}
                  taskId={resolvedTaskId ?? undefined}
                  rubricsCriteria={rubricsCriteria}
                  rubricsAggregator={rubricsAggregator}
                  taskSteps={viewerTaskSteps}
                />
              );
            })()}
        </div>
      )}
    </div>
  );
}
