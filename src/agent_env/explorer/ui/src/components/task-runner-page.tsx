import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  Plus,
  Save,
  RefreshCw,
  RotateCcw,
  AlertTriangle,
} from 'lucide-react';
import { AlertDialog, Button, DropdownMenu, Flex } from '@radix-ui/themes';
import {
  useExternalApp,
  type SubmissionItem,
} from '../lib/external-app';
import { selectFinalScore } from '../lib/verifier-classification';
import {
  materializeInstance,
  type MaterializeOptions,
} from '../lib/materialize';
import { BACKEND_URL, apiFetch, objectContentUrl } from './shared';
import { StepsPipeline } from './steps-pipeline';
import { TaskInstanceViewer, type TaskStepRef } from './task-instance-viewer';
import {
  type StepState,
  type StepType,
  ADDABLE_STEP_TYPES,
  STEP_TYPE_LABELS,
  STEP_COLORS,
  EVALUATOR_STEP_TYPES,
  makeDefaultStep,
  getRecommendedStep,
  getJsonError,
  stepToDict,
  stepFromDict,
  StepEditor,
} from './task-steps-shared';

/* ------------------------------------------------------------------ */
/*  Config drift fingerprinting                                        */
/* ------------------------------------------------------------------ */

// Key-sorted JSON so equal content hashes identically; drops undefined keys (they don't survive JSON round-trips).
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value) ?? 'null';
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(',')}]`;
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj)
    .filter(k => obj[k] !== undefined)
    .sort();
  return `{${keys
    .map(k => `${JSON.stringify(k)}:${stableStringify(obj[k])}`)
    .join(',')}}`;
}

// Fingerprint of a config's run-affecting content (steps only; id/project_id excluded).
// Call only on the parent INIT_STATE config — a stored task is backend-normalized and would always differ.
function configFingerprint(
  config: { steps?: unknown } | null | undefined,
): string | null {
  if (!config || config.steps == null) return null;
  return stableStringify(config.steps);
}

// Short stable digest (cyrb53) of a config fingerprint — a compact opaque drift key.
function shortHash(str: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 =
    Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^
    Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 =
    Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^
    Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

// Drift key for a parent config: a short hash of its steps, compared input-to-input.
function configHash(
  config: { steps?: unknown } | null | undefined,
): string | null {
  const fp = configFingerprint(config);
  return fp === null ? null : shortHash(fp);
}

const MIN_RUNS_PER_CLICK = 1;
const MAX_RUNS_PER_CLICK = 5;

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface TaskRunnerConfig {
  task_id?: string;
  task_config?: {
    id?: string;
    steps: Record<string, unknown>[];
    project_id?: string;
  };
  auto_run?: boolean;
  allow_edit_evaluator?: boolean;
  max_runs?: number;
  preferred_run_count?: number;
  runs_per_click?: number;
  /** Drop context/materialized from instances, leaving bodies only on preferred_instances.
   *  Opt-in; no-op without preferred_run_count (nothing is starred). */
  omit_instance_bodies?: boolean;
  /** Models offered per run, applied as agent_model on POST /run. prompt_agent prefers it over the
   *  step's model, so switching models keeps every attempt on one version (comparable pass@k). */
  agent_models?: string[];
  /** Reveals the Task ID + pipeline sections when embedded, as does is_runnable_interactively. */
  is_editable?: boolean;
  /** Show the manual "Run" button when embedded (auto_run is independent). This or is_editable also
   *  reveals the Task ID + pipeline sections. */
  is_runnable_interactively?: boolean;
  /** Per-instance derived fields to materialize onto each instance; see lib/materialize.ts. */
  materialize?: MaterializeOptions;
  /** Show the Task Run Context tab while embedded (hidden in the iframe chrome). Explicit so a config
   *  can enable it on one grading step without turning it on for every step. */
  show_run_context?: boolean;
  /** Attempts to grade (a runner step's starred instances), one run each resumed
   * against that attempt's context. `agent_model` must match the attempt. */
  grade_attempts?: Array<{
    instance_id?: string;
    agent_model?: string;
    context?: Record<string, unknown> | null;
    /** Per-step param overrides, keyed by task step id. Cannot ride
     * `context_json`: the backend shallow-merges `user_overrides` over it. */
    step_overrides?: Record<string, Record<string, unknown>>;
  }>;
  /** Index a graded run resumes at. Must be the judge's own `deploy_agent`, not the
   * verifier, or `rubrics_verifier` fails with "Agent 'judge' not found". */
  grade_start_step?: number;
}

type Phase =
  | 'initializing'
  | 'creating'
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

/** Steps that stand up the judge before it grades, so a resume must include them. */
function isJudgeSetupStep(type: unknown) {
  return (
    type === 'deploy_agent' || type === 'load_artifact' || type === 'run_code'
  );
}

/** Steps a graded run resumes at; grading replays these against a prior context. */
function isGradingStep(type: unknown) {
  return (
    type === 'cua_evaluate' ||
    type === 'rubrics_verifier' ||
    type === 'run_openclaw_unit_test'
  );
}

function isTerminal(inst: Record<string, unknown>) {
  return TERMINAL_STATUSES.has(String(inst.status ?? ''));
}

/** Order-independent signature of a preferred-run selection. */
function preferredKey(ids: Iterable<string>) {
  return [...ids].sort().join(',');
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

// A transient failure backs off and retries on a later submission; only a terminal status caches a miss.
const WARM_BACKOFF_MS = [15_000, 60_000, 300_000];

/** Over the cap or gone — retrying cannot help, unlike a 5xx/429. */
function isTerminalWarmFailure(status: number): boolean {
  return status === 413 || status === 404 || status === 410;
}

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
  const externalApp = useExternalApp();
  const isEmbedded = typeof window !== 'undefined' && window.parent !== window;

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

  /* --- steps (for editing) --- */
  const [steps, setSteps] = useState<StepState[]>([]);
  /** Never set: the editor and its save path are kept but unreachable, because the control plane
   *  serves tasks read-only (`POST /api/v1/tasks` is not routed). Restoring the entry point means
   *  adding that route first, so the button cannot appear before the endpoint it calls. */
  const [editing, setEditing] = useState(false);
  const [selectedStepIndex, setSelectedStepIndex] = useState<number | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  const nextStepKeyRef = useRef(0);

  /* --- run limit --- */
  const [maxRuns, setMaxRuns] = useState<number | null>(null);

  const [runsPerClick, setRunsPerClick] = useState<number>(1);

  /* --- preferred runs --- */
  const [preferredRunCount, setPreferredRunCount] = useState<number | null>(
    null,
  );
  // Both read through refs — sendResults is memoised and gates omission on them.
  const preferredRunCountRef = useRef<number | null>(null);
  preferredRunCountRef.current = preferredRunCount;
  const [omitInstanceBodies, setOmitInstanceBodies] = useState(false);
  const omitInstanceBodiesRef = useRef(false);
  omitInstanceBodiesRef.current = omitInstanceBodies;
  const [agentModels, setAgentModels] = useState<string[]>([]);
  // Empty only while no list has arrived; a provided list always resolves to one of its
  // entries, so a run never silently falls back to the config's own model.
  const [agentModel, setAgentModel] = useState('');
  // handleRun is memoised, so read the selection through a ref to avoid a stale value.
  const agentModelRef = useRef('');
  agentModelRef.current = agentModel;
  const [preferredInstanceIds, setPreferredInstanceIds] = useState<Set<string>>(
    new Set(),
  );
  const preferredInstanceIdsRef = useRef<Set<string>>(new Set());
  preferredInstanceIdsRef.current = preferredInstanceIds;
  // The task whose selection we report. Set once task identity is settled; null = nothing to say about stars yet.
  const selectionOwnerRef = useRef<string | null>(null);
  // The task WE have expressed a selection for (star toggle / version reset). Once set, INIT_STATE echoes
  // for this task are stale and ignored; until then a late seed is the real saved selection, not an echo.
  const selectionIntentRef = useRef<string | null>(null);
  // What the parent last heard from us, per task. Only a divergence is worth posting; equality avoids an INIT_STATE↔SUBMISSION loop.
  const lastSentPreferredRef = useRef<{ taskId: string; key: string } | null>(
    null,
  );
  // Adopt `ids` as taskId's selection, baselined as what the parent already holds — so the next divergence reads as a real user change.
  const adoptSelection = useCallback((taskId: string, ids: string[]) => {
    selectionOwnerRef.current = taskId;
    lastSentPreferredRef.current = { taskId, key: preferredKey(ids) };
    setPreferredInstanceIds(prev => {
      // Preserve the reference when the content is unchanged — the re-send
      // effect depends on this state by identity.
      if (ids.length === prev.size && ids.every(id => prev.has(id)))
        return prev;
      return new Set(ids);
    });
  }, []);
  /** Adopt a selection of OUR making, locking out any later parent seed. */
  const claimSelection = useCallback(
    (taskId: string, ids: string[]) => {
      selectionIntentRef.current = taskId;
      adoptSelection(taskId, ids);
    },
    [adoptSelection],
  );

  /* --- which derived fields to materialize onto each instance --- */
  const [materialize, setMaterialize] = useState<
    MaterializeOptions | undefined
  >(undefined);
  // sendResults is memoised on [isEmbedded, externalApp], so it would close over
  // a stale value — read through a ref, same as preferredInstanceIds.
  const materializeRef = useRef<MaterializeOptions | undefined>(undefined);
  materializeRef.current = materialize;
  // s3_uri -> records. Keyed by uri (immutable per run) so a re-send does not
  // re-download; a failed fetch stores [] so it is not retried on every send.
  const trajectoryCacheRef = useRef<Map<string, unknown[]>>(new Map());
  // uri/key -> { failures, retryAfter epoch ms }. Bounds the RATE of retries, not
  // their number.
  const warmAttemptsRef = useRef<
    Map<string, { failures: number; retryAfter: number }>
  >(new Map());
  const warmBlocked = useCallback(
    (key: string) =>
      Date.now() < (warmAttemptsRef.current.get(key)?.retryAfter ?? 0),
    [],
  );
  // One pending timer, aimed at the soonest cooldown and re-armed after every wake-up, so a later
  // deadline still gets scheduled when an earlier key already succeeded.
  const warmRetryTimerRef = useRef<{ at: number; id: number } | null>(null);
  const armWarmRetryRef = useRef<() => void>(() => undefined);
  const armWarmRetry = useCallback(() => {
    const now = Date.now();
    let soonest = Infinity;
    for (const { retryAfter } of warmAttemptsRef.current.values()) {
      if (retryAfter > now && retryAfter < soonest) soonest = retryAfter;
    }
    if (soonest === Infinity) return; // nothing cooling down
    const pending = warmRetryTimerRef.current;
    if (pending && pending.at <= soonest) return; // an earlier wake-up covers it
    if (pending) window.clearTimeout(pending.id);
    const id = window.setTimeout(
      () => {
        warmRetryTimerRef.current = null;
        const b = latestBatchRef.current;
        if (b.taskId)
          sendResultsRef.current(b.taskId, b.version, b.allInstances);
        // Anything still cooling down gets the next wake-up.
        armWarmRetryRef.current();
      },
      // +250ms so the deadline the skip-filters compare against has passed.
      soonest - now + 250,
    );
    warmRetryTimerRef.current = { at: soonest, id };
  }, []);
  armWarmRetryRef.current = armWarmRetry;

  const noteWarmFailure = useCallback(
    (key: string) => {
      const failures = (warmAttemptsRef.current.get(key)?.failures ?? 0) + 1;
      const backoff =
        WARM_BACKOFF_MS[Math.min(failures - 1, WARM_BACKOFF_MS.length - 1)] ??
        0;
      warmAttemptsRef.current.set(key, {
        failures,
        retryAfter: Date.now() + backoff,
      });
      armWarmRetry();
    },
    [armWarmRetry],
  );

  // A terminal failure is never retried, so its entry must not keep a wake-up alive.
  const forgetWarmKey = useCallback((key: string) => {
    warmAttemptsRef.current.delete(key);
  }, []);

  useEffect(
    () => () => {
      if (warmRetryTimerRef.current) {
        window.clearTimeout(warmRetryTimerRef.current.id);
      }
    },
    [],
  );

  // task_version -> that version's steps. Immutable, so one GET per version.
  const taskStepsCacheRef = useRef<Map<number, Record<string, unknown>[]>>(
    new Map(),
  );

  /* --- embed interactivity flags (opt-in from parent app) --- */
  const [isEditable, setIsEditable] = useState(false);
  const [isRunnableInteractively, setIsRunnableInteractively] = useState(false);

  /* --- latest task_config from parent (used by Apply / Start Fresh) --- */
  const [parentTaskConfig, setParentTaskConfig] = useState<{
    id?: string;
    steps: Record<string, unknown>[];
    project_id?: string;
  } | null>(null);
  // Fingerprint of the parent config the resolved version reflects. null = unconfirmed (fresh resume:
  // treat as drifted until the first Run). Always derived from the parent config, never the stored task.
  const [appliedConfigHash, setAppliedConfigHash] = useState<string | null>(
    null,
  );
  const appliedConfigHashRef = useRef<string | null>(null);
  appliedConfigHashRef.current = appliedConfigHash;

  /* --- opt-in from the config: keep the Task Run Context tab in an embed --- */
  const [showRunContext, setShowRunContext] = useState(false);
  /* --- attempts handed down by an upstream runner step, graded one run each --- */
  const [gradeAttempts, setGradeAttempts] = useState<
    NonNullable<TaskRunnerConfig['grade_attempts']>
  >([]);
  const [gradeStartStep, setGradeStartStep] = useState<number | null>(null);

  /* --- instances --- */
  const [instances, setInstances] = useState<Record<string, unknown>[]>([]);
  const [selectedInstanceId, setSelectedInstanceId] = useState<string | null>(
    null,
  );
  const [pendingRun, setPendingRun] = useState(false);
  const lastWorkflowIdRef = useRef<string | null>(null);
  const snapshotRef = useRef('');

  // Render-synced refs so effects/callbacks can read the latest values without
  // re-subscribing (adding them as deps would re-fire the auto-run effects).
  const instancesRef = useRef(instances);
  instancesRef.current = instances;
  const resolvedTaskIdRef = useRef(resolvedTaskId);
  resolvedTaskIdRef.current = resolvedTaskId;
  // Ref-mirror so buildBody reads the latest project without a handleRun dep
  // (adding one would re-fire the auto-run effects).
  const parentTaskConfigRef = useRef(parentTaskConfig);
  parentTaskConfigRef.current = parentTaskConfig;


  /* --- track last processed inputs to detect new configs --- */
  const lastInputsRef = useRef<Record<string, unknown> | null | undefined>(
    undefined,
  );
  // Content signature of the last processed init. The identity guard above only catches the same object;
  // the host re-emits a fresh INIT_STATE after every SUBMISSION, so dedupe on content to avoid a
  // fetch→setSteps→sendResult loop that re-runs dagre.layout until the tab OOMs.
  const lastInitKeyRef = useRef<string | null>(null);

  /* --- lifecycle signal to parent (created / resumed / updated) --- */
  const sendResult = useCallback(
    (
      taskId: string,
      version: number | null,
      status: string,
      appliedConfigHash?: string | null,
    ) => {
      if (!isEmbedded) return;
      const metadata: Record<string, unknown> = {
        task_id: taskId,
        version: version ?? undefined,
      };
      // Persist the input-side hash of the applied config so a later resume detects drift input-to-input.
      // Every caller re-stamps it so it survives a host that replaces (rather than merges) the output item.
      if (appliedConfigHash != null) {
        metadata.applied_config_hash = appliedConfigHash;
      }
      const item: SubmissionItem = {
        content: {
          id: 'task-runner-result',
          type: 'json',
          data: { task_id: taskId, version, status },
        },
        metadata,
      };
      externalApp.sendSubmission([item]);
    },
    [isEmbedded, externalApp],
  );

  /* --- batch all terminal instances with full context --- */
  /** Warm the trajectory cache, then re-send so the records land. */
  const warmTrajectories = useCallback(
    async (instances: Record<string, unknown>[], onDone: () => void) => {
      // Every turn, not just the newest — records are embedded per turn.
      const uris = instances
        .flatMap(inst => {
          const ctx = (inst.context ?? {}) as Record<string, unknown>;
          const rs = Array.isArray(ctx.prompt_responses)
            ? (ctx.prompt_responses as Record<string, unknown>[])
            : [];
          return rs.map(r => r.agent_trajectory_s3_uri);
        })
        .filter(
          (u): u is string =>
            typeof u === 'string' &&
            u !== '' &&
            !trajectoryCacheRef.current.has(u) &&
            !warmBlocked(u),
        );
      if (uris.length === 0) return;
      await Promise.all(
        uris.map(async uri => {
          try {
            const res = await apiFetch(objectContentUrl(uri));
            if (res.ok) {
              const data: unknown = await res.json();
              trajectoryCacheRef.current.set(
                uri,
                Array.isArray(data) ? data : [],
              );
              warmAttemptsRef.current.delete(uri);
            } else if (isTerminalWarmFailure(res.status)) {
              // Over the size cap or gone: retrying cannot help, so cache the miss.
              trajectoryCacheRef.current.set(uri, []);
              forgetWarmKey(uri);
            } else {
              noteWarmFailure(uri);
            }
          } catch {
            // Network-level failure — transient until proven otherwise.
            noteWarmFailure(uri);
          }
        }),
      );
      onDone();
    },
    [noteWarmFailure, warmBlocked, forgetWarmKey],
  );

  /** Fetch step dicts per task_version in this send (for materialized.systemPrompt). One GET per version;
   *  versions are immutable, so each is fetched at most once. */
  const warmTaskSteps = useCallback(
    async (
      taskId: string,
      instances: Record<string, unknown>[],
      onDone: () => void,
    ) => {
      const versions = [
        ...new Set(
          instances
            .map(inst => inst.task_version)
            .filter(
              (v): v is number =>
                typeof v === 'number' && !taskStepsCacheRef.current.has(v),
            ),
        ),
      ];
      if (versions.length === 0) return;
      await Promise.all(
        versions.map(async v => {
          try {
            const res = await apiFetch(
              `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
                taskId,
              )}?version=${v}`,
            );
            const key = `steps:${v}`;
            if (res.ok) {
              const data = (await res.json()) as Record<string, unknown>;
              taskStepsCacheRef.current.set(
                v,
                Array.isArray(data.steps)
                  ? (data.steps as Record<string, unknown>[])
                  : [],
              );
              warmAttemptsRef.current.delete(key);
            } else if (isTerminalWarmFailure(res.status)) {
              // A deleted version is never coming back.
              taskStepsCacheRef.current.set(v, []);
              forgetWarmKey(key);
            } else {
              noteWarmFailure(key);
            }
          } catch {
            noteWarmFailure(`steps:${v}`);
          }
        }),
      );
      onDone();
    },
    [noteWarmFailure, warmBlocked, forgetWarmKey],
  );

  const sendResults = useCallback(
    (
      taskId: string,
      version: number | null,
      allInstances: Record<string, unknown>[],
    ) => {
      if (!isEmbedded) return;
      // A warm re-send after a rotation would post the OLD task_id and roll the run set back; the filter below misses it.
      if (resolvedTaskIdRef.current !== taskId) return;
      // Newest batch wins: the warm passes re-send after an await, by which point a
      // later poll may have submitted a bigger one. They read this instead.
      latestBatchRef.current = { taskId, version, allInstances };
      // Each instance doc carries its task, so the payload (not clear-timing) decides what's filed under
      // taskId. A batch that outlived its task contributes nothing; an all-foreign batch drops out below.
      const terminalInstances = allInstances
        .filter(isTerminal)
        .filter(
          inst => inst.task_id == null || String(inst.task_id) === taskId,
        );
      if (terminalInstances.length === 0) return;
      // Only completed runs may be preferred — never failed/cancelled/errored
      // ones, even if their id somehow ended up in the set.
      const isStarred = (inst: Record<string, unknown>) =>
        inst.status === 'completed' &&
        preferredInstanceIdsRef.current.has(String(inst.instance_id));

      // Wire contract: with omit_instance_bodies on, instances carries ids only — bodies ride
      // preferred_instances, context_omitted = "not sent". A starred run is bodyless here but full there, so it still needs materializing.
      const toShape = (inst: Record<string, unknown>, full: boolean) => {
        const base = {
          instance_id: String(inst.instance_id),
          status: String(inst.status),
          task_version:
            typeof inst.task_version === 'number'
              ? inst.task_version
              : undefined,
          created_at_utc:
            inst.created_at_utc != null
              ? String(inst.created_at_utc)
              : undefined,
          completed_at_utc:
            inst.completed_at_utc != null
              ? String(inst.completed_at_utc)
              : undefined,
          duration_seconds:
            typeof inst.duration_seconds === 'number'
              ? inst.duration_seconds
              : undefined,
        };
        if (!full) return { ...base, context_omitted: true };
        return {
          ...base,
          context: (inst.context as Record<string, unknown>) ?? {},
          materialized: materializeInstance(
            inst,
            materializeRef.current,
            trajectoryCacheRef.current,
            taskStepsCacheRef.current,
          ),
        };
      };

      // Interlocked on `preferred_run_count` even when opted in: without stars
      // there is no `preferred_instances`, so this would drop every body.
      const omitBodies =
        omitInstanceBodiesRef.current && preferredRunCountRef.current != null;
      const carriesBody = (inst: Record<string, unknown>) =>
        !omitBodies || isStarred(inst);

      const preferredList = terminalInstances
        .filter(isStarred)
        .map(inst => toShape(inst, true));
      // Every post routes through here, incl. the poll path (no INIT_STATE wait). Until we own this task's
      // selection, an empty list means "not told yet", not "no stars"; emit [] only once owned, else
      // "cleared last star" is indistinguishable from silence and the parent re-echoes the stale list.
      const ownsPreferred = selectionOwnerRef.current === taskId;
      const metadata: Record<string, unknown> = {
        task_id: taskId,
        version: version ?? undefined,
      };
      // Shares the 'task-runner-result' id with sendResult, so re-stamp the drift baseline here too (input-side, via ref).
      if (appliedConfigHashRef.current != null) {
        metadata.applied_config_hash = appliedConfigHashRef.current;
      }
      const item: SubmissionItem = {
        content: {
          id: 'task-runner-result',
          type: 'json',
          data: {
            task_id: taskId,
            version,
            instances: terminalInstances.map(inst =>
              toShape(inst, !omitBodies),
            ),
            ...((ownsPreferred || preferredList.length > 0) && {
              preferred_instances: preferredList,
            }),
          },
        },
        metadata,
      };
      externalApp.sendSubmission([item]);
      if (ownsPreferred) {
        // Single writer for the baseline, keyed on the local selection (our intent). Leaving it to the
        // re-send effect would let it go stale so a toggle-back reads as a no-op and is dropped.
        lastSentPreferredRef.current = {
          taskId,
          key: preferredKey(preferredInstanceIdsRef.current),
        };
      }
      const fullInstances = terminalInstances.filter(carriesBody);
      // systemPrompt needs the ran version's steps — same fire-and-forget shape.
      void warmTaskSteps(taskId, fullInstances, () => {
        const b = latestBatchRef.current;
        sendResultsRef.current(b.taskId, b.version, b.allInstances);
      });
      const resend = () => {
        const b = latestBatchRef.current;
        sendResultsRef.current(b.taskId, b.version, b.allInstances);
      };
      // Records download after the first submission, then a re-send embeds them. Self-limiting: a second pass finds all cached.
      if (materializeRef.current?.trajectory) {
        void warmTrajectories(fullInstances, resend);
      }
    },
    [isEmbedded, externalApp, warmTaskSteps, warmTrajectories],
  );

  // The most recent batch handed to sendResults — see the note in its body.
  const latestBatchRef = useRef<{
    taskId: string;
    version: number | null;
    allInstances: Record<string, unknown>[];
  }>({ taskId: '', version: null, allInstances: [] });

  const sendResultsRef = useRef(sendResults);
  sendResultsRef.current = sendResults;
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
  const sentResultIdsRef = useRef<Set<string>>(new Set());
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
      // The resolved task can change mid-flight (parent switch, save/upsert). Everything below writes
      // "current task's runs", so a late response must be dropped, not applied.
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

      // Re-send full batch whenever a new terminal instance appears
      const terminalIds = new Set(
        items
          .filter(isTerminal)
          .map((i: Record<string, unknown>) => String(i.instance_id)),
      );
      const hasNew = [...terminalIds].some(
        id => !sentResultIdsRef.current.has(id),
      );
      if (hasNew) {
        sentResultIdsRef.current = new Set([
          ...sentResultIdsRef.current,
          ...terminalIds,
        ]);
        sendResultsRef.current(
          resolvedTaskId,
          (items[0]?.task_version as number) ?? null,
          items,
        );
      }

      snapshotRef.current = snap;
      instanceIdsRef.current = new Set(items.map(i => String(i.instance_id)));
      setInstances(items);
    } catch {
      /* ignore */
    }
  }, [resolvedTaskId]);

  // A parent-driven task switch changes resolvedTaskId without clearing instances, so the on-screen rows
  // are still the previous task's. Clear here so a stale star can't file them under the new task id;
  // this is also what makes claiming the selection before the fetch lands safe.
  const prevResolvedTaskIdRef = useRef<string | null>(null);
  useEffect(() => {
    const prev = prevResolvedTaskIdRef.current;
    prevResolvedTaskIdRef.current = resolvedTaskId;
    // Only a switch between two real tasks — not the first resolve, and not the
    // same-id version bumps, which clear up after themselves.
    if (!resolvedTaskId || prev === null || prev === resolvedTaskId) return;
    setInstances([]);
    setSelectedInstanceId(null);
    sentResultIdsRef.current = new Set();
    instanceIdsRef.current = new Set();
    snapshotRef.current = '';
    // The intent lapses with the task we left, else A→B→A refuses the seed and leaves B's stars on A.
    // Safe because anything expressed was already posted at the time.
    selectionIntentRef.current = null;
  }, [resolvedTaskId]);

  // Poll while a run is active; stop once every instance is terminal. Deriving stop from instance status
  // (not phase, which never goes terminal here) lets a fresh run flip polling back on.
  const allInstancesTerminal =
    instances.length > 0 && instances.every(isTerminal);
  useEffect(() => {
    if (
      !resolvedTaskId ||
      phase === 'initializing' ||
      phase === 'creating' ||
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

  // Re-send results when the starred selection changes after all runs are terminal (pollInstances only
  // fires for NEW terminal instances). Post only a divergence from what the parent last heard; a missing
  // baseline or one from another task means we don't own this task's selection yet (a switch changes
  // resolvedTaskId without clearing instances), so posting then would file the old runs under the new task.
  useEffect(() => {
    const terminal = instancesRef.current.filter(isTerminal);
    if (terminal.length === 0 || !resolvedTaskId) return;
    if (
      terminal.some(
        inst => inst.task_id != null && String(inst.task_id) !== resolvedTaskId,
      )
    )
      return;
    const prev = lastSentPreferredRef.current;
    if (!prev || prev.taskId !== resolvedTaskId) return;
    if (prev.key === preferredKey(preferredInstanceIds)) return;
    sendResultsRef.current(
      resolvedTaskId,
      (terminal[0]?.task_version as number) ?? null,
      instancesRef.current,
    );
  }, [preferredInstanceIds, resolvedTaskId]);

  /* --- run task --- */
  // Lifted out of `handleRun` so batch grading can send per-attempt bodies;
  // `handleRun`'s `count` only repeats one identical body.
  const buildRunBody = useCallback(
    (opts?: {
      version?: number;
      start_step?: number;
      agent_model?: string;
      context_json?: Record<string, unknown> | null;
      // Prefer over context_json when re-running one of this task's instances:
      // `instances` comes from GET /instances, which strips a2a_card.
      context_from_instance_id?: string;
      // From-start re-run: inherit only the prior run's user_overrides.
      overrides_from_instance_id?: string;
      step_overrides?: Record<string, Record<string, unknown>>;
    }) => {
      const body: Record<string, unknown> = {};
      body.version = opts?.version ?? taskVersion ?? undefined;
      // priority=0 (interactive): a human is waiting. Sent explicitly so intent survives backend default changes.
      body.priority = 0;
      if (opts?.start_step != null) body.start_step = opts.start_step;
      // Per-run model override — no upsert, so attempts stay on one task version.
      const model = opts?.agent_model ?? agentModelRef.current;
      if (model) body.agent_model = model;
      // Mutually exclusive server-side.
      if (opts?.context_from_instance_id)
        body.context_from_instance_id = opts.context_from_instance_id;
      else if (opts?.context_json) body.context_json = opts.context_json;
      if (opts?.overrides_from_instance_id)
        body.overrides_from_instance_id = opts.overrides_from_instance_id;
      if (opts?.step_overrides && Object.keys(opts.step_overrides).length > 0)
        body.step_overrides = opts.step_overrides;
      // Forward the host's project (config or ?projectId=), NOT ?taskId=. Only-if-truthy so "" can't defeat the backend fallback.
      const projectId =
        parentTaskConfigRef.current?.project_id ||
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
      agent_model?: string;
      context_json?: Record<string, unknown> | null;
      context_from_instance_id?: string;
      overrides_from_instance_id?: string;
      count?: number;
    }) => {
      if (!resolvedTaskId || pendingRun) return;
      const requested = Math.max(
        MIN_RUNS_PER_CLICK,
        Math.floor(opts?.count ?? 1),
      );
      const count =
        maxRuns != null
          ? Math.max(
              0,
              Math.min(requested, maxRuns - instancesRef.current.length),
            )
          : requested;
      if (count === 0) return;
      setPendingRun(true);
      setErrorMsg(null);

      const buildBody = () => buildRunBody(opts);

      const startOne = async (): Promise<string | null> => {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            resolvedTaskId,
          )}/run`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(buildBody()),
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
        return (data.workflow_id as string | null) ?? null;
      };

      const results = await Promise.allSettled(
        Array.from({ length: count }, () => startOne()),
      );
      const firstOk = results.find(
        (r): r is PromiseFulfilledResult<string | null> =>
          r.status === 'fulfilled',
      );
      lastWorkflowIdRef.current = firstOk?.value ?? null;
      const failed = results.filter(r => r.status === 'rejected').length;
      if (!firstOk) {
        const firstErr = results.find(
          (r): r is PromiseRejectedResult => r.status === 'rejected',
        );
        setPendingRun(false);
        setErrorMsg(
          firstErr?.reason instanceof Error
            ? firstErr.reason.message
            : 'Failed to start run',
        );
      } else if (failed > 0) {
        // Some launches failed but at least one started: keep pendingRun and warn (fewer than runs_per_click).
        setErrorMsg(
          `Started ${
            count - failed
          } of ${count} runs; ${failed} failed to launch.`,
        );
      }
    },
    [
      resolvedTaskId,
      taskVersion,
      pendingRun,
      maxRuns,
      buildRunBody,
    ],
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

  /* --- create task --- */
  const createTask = useCallback(
    async (config: {
      id: string;
      steps: Record<string, unknown>[];
      project_id?: string;
    }) => {
      setPhase('creating');

      // No URL rewriting: config values are already fetchable URLs in the
      // standalone hub, so they are used verbatim.
      const translatedConfig = config;

      try {
        const res = await apiFetch(`${BACKEND_URL}/api/v1/tasks`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(translatedConfig),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(
            data.detail || `Failed to create task (${res.status})`,
          );
        }
        const data = await res.json();
        return data as { id: string; version: number };
      } catch (e) {
        setPhase('error');
        setErrorMsg(e instanceof Error ? e.message : 'Failed to create task');
        return null;
      }
    },
    [],
  );

  /* --- initialization effect --- */
  useEffect(() => {
    // In embedded mode, wait for useExternalApp to be ready
    if (isEmbedded && !externalApp.isReady) return;

    // Skip if we already processed these exact inputs
    if (lastInputsRef.current === externalApp.receivedInputs) return;
    lastInputsRef.current = externalApp.receivedInputs;

    // The host wraps inputs as {state, value, error} before posting INIT_STATE. Unwrap .value to reach the TaskRunnerConfig.
    const inputs =
      (externalApp.receivedInputs as { value?: TaskRunnerConfig } | null)
        ?.value ?? null;
    const output = externalApp.receivedOutput;

    // Capture config from parent
    setShowRunContext(inputs?.show_run_context === true);
    // Unconditional, like `materialize`: a config that stops asking for this must
    // not keep doing it.
    setOmitInstanceBodies(inputs?.omit_instance_bodies === true);
    // Unconditional, like `materialize`: a stale attempt list would keep offering
    // to grade instances the upstream step no longer stars.
    setGradeAttempts(
      Array.isArray(inputs?.grade_attempts)
        ? inputs.grade_attempts.filter(
            a => a && typeof a === 'object' && !!a.context,
          )
        : [],
    );
    setGradeStartStep(
      typeof inputs?.grade_start_step === 'number' &&
        Number.isInteger(inputs.grade_start_step) &&
        inputs.grade_start_step >= 0
        ? inputs.grade_start_step
        : null,
    );
    if (inputs?.max_runs != null) setMaxRuns(inputs.max_runs);
    if (inputs?.preferred_run_count != null)
      setPreferredRunCount(inputs.preferred_run_count);
    // Unconditional, unlike the fields above: leaving stale options active would keep
    // downloading and uploading for a config that no longer asks for it.
    setMaterialize(inputs?.materialize);
    if (Array.isArray(inputs?.agent_models)) {
      // Trimmed and de-duplicated: a hand-written config list picks up stray whitespace
      // and repeats, and either would show up as a bogus dropdown entry.
      const models = [
        ...new Set(
          inputs.agent_models
            .filter((m): m is string => typeof m === 'string')
            .map(m => m.trim())
            .filter(Boolean),
        ),
      ];
      setAgentModels(models);
      // A provided list means the run's model is explicit — default to the first entry
      // rather than falling back to whatever the task config happens to pin.
      setAgentModel(prev => (models.includes(prev) ? prev : models[0] ?? ''));
    } else {
      // A config that stops offering models must not leave a stale selection behind.
      setAgentModels([]);
      setAgentModel('');
    }
    if (inputs?.runs_per_click != null)
      setRunsPerClick(
        Math.max(
          MIN_RUNS_PER_CLICK,
          Math.min(MAX_RUNS_PER_CLICK, Math.floor(inputs.runs_per_click)),
        ),
      );
    if (inputs?.is_editable != null) setIsEditable(inputs.is_editable);
    if (inputs?.is_runnable_interactively != null)
      setIsRunnableInteractively(inputs.is_runnable_interactively);

    // Stash the parent's task_config so Run can upsert it on demand (drift
    // detection + Run-gated apply) without re-mounting the iframe.
    if (inputs?.task_config) setParentTaskConfig(inputs.task_config);

    // Resumption priority:
    // 1. Previous submission output (reload case)
    const savedTaskId = output?.items?.[0]?.metadata?.task_id as
      | string
      | undefined;
    // Input-side config hash persisted on the last create/apply, round-tripped via submission metadata. Seeds the resume baseline.
    const savedConfigHash = output?.items?.[0]?.metadata
      ?.applied_config_hash as string | undefined;
    // 2. task_id from parent inputs
    const inputTaskId = inputs?.task_id;
    // 3. task_id from URL
    const existingTaskId = savedTaskId ?? inputTaskId ?? urlTaskId;

    // Seed the starred selection from the saved submission on reload — but only while we have no selection
    // of our own for this task, so an echo of the pre-un-star list can't overwrite a newer local one.
    const savedPreferredInstances = (
      output?.items?.[0]?.content as
        | {
            data?: {
              preferred_instances?: { instance_id?: string; status?: string }[];
            };
          }
        | undefined
    )?.data?.preferred_instances;
    if (existingTaskId && selectionIntentRef.current !== existingTaskId) {
      adoptSelection(
        existingTaskId,
        (Array.isArray(savedPreferredInstances) ? savedPreferredInstances : [])
          // Only completed runs are preferable — never seed a stale
          // failed/cancelled selection back into the star state.
          .filter(inst => inst.status === 'completed')
          .map(inst => String(inst.instance_id))
          .filter(id => id && id !== 'undefined'),
      );
    }

    // Break the INIT_STATE↔SUBMISSION loop: if this init resolves to the task we already handled, skip the
    // heavy path (fetchTask→setSteps→sendResult, which re-runs dagre and posts another SUBMISSION).
    const initKey = JSON.stringify({
      task: existingTaskId ?? null,
      cfgId: inputs?.task_config?.id ?? null,
    });
    if (initKey === lastInitKeyRef.current) return;
    lastInitKeyRef.current = initKey;

    if (existingTaskId) {
      // Resume: fetch the stored task only — Run applies parent edits. Seed the drift baseline from the
      // persisted hash (input-to-input): same config → no warning; newer parent → warning + apply; absent → safe fallback.
      setAppliedConfigHash(savedConfigHash ?? null);
      appliedConfigHashRef.current = savedConfigHash ?? null;
      setResolvedTaskId(existingTaskId);
      fetchTask(existingTaskId)
        .then(data => {
          setTask(data);
          setTaskVersion(data.version ?? null);
          const loadedSteps = (
            (data.steps ?? []) as Record<string, unknown>[]
          ).map((s: Record<string, unknown>) =>
            stepFromDict(s, nextStepKeyRef.current++),
          );
          setSteps(loadedSteps);
          setPhase('idle');
          // Send submission so the parent has the task_id; re-stamp the hash it already holds (idempotent) so the baseline survives a replace.
          sendResult(
            existingTaskId,
            data.version ?? null,
            'resumed',
            savedConfigHash,
          );
        })
        .catch(e => {
          setPhase('error');
          setErrorMsg(e instanceof Error ? e.message : 'Failed to load task');
        });
    } else if (inputs?.task_config) {
      // Create new task — fill in id/project_id from pipeline context if missing
      const urlParams = new URLSearchParams(window.location.search);
      const pipelineTaskId = urlParams.get('taskId');
      const pipelineProjectId = urlParams.get('projectId');
      const rawConfig = inputs.task_config;
      const config = {
        ...rawConfig,
        id:
          rawConfig.id ||
          (pipelineTaskId ? `aeh-${pipelineTaskId}` : `aeh-${Date.now()}`),
        project_id:
          rawConfig.project_id ||
          pipelineProjectId ||
          pipelineTaskId ||
          undefined,
      };
      const autoRun = inputs.auto_run !== false;
      const loadedSteps = config.steps.map((s: Record<string, unknown>) =>
        stepFromDict(s, nextStepKeyRef.current++),
      );
      setSteps(loadedSteps);

      createTask(config).then(result => {
        if (!result) return;
        setResolvedTaskId(result.id);
        setTaskVersion(result.version);
        // A just-upserted task has no selection of ours yet: adopt an empty one so a later "cleared last
        // star" is postable, while still letting the parent's output seed us when it lands.
        adoptSelection(result.id, []);
        // Created this task FROM the parent config, so the version reflects it → no drift. Persist the hash for a later resume.
        const baseline = configHash(inputs.task_config);
        setAppliedConfigHash(baseline);
        appliedConfigHashRef.current = baseline;
        sendResult(result.id, result.version, 'created', baseline);

        // Fetch full task for metadata
        fetchTask(result.id)
          .then(data => setTask(data))
          .catch(() => {});

        // Auto-run only on first creation (version 1). On reload the upsert bumps the version, so >1 means it already existed.
        if (autoRun && result.version <= 1) {
          autoRunTriggeredRef.current = false;
          setPhase('running');
          // handleRun needs resolvedTaskId — triggered from auto-run effect.
        } else {
          setPhase('idle');
        }
      });
    } else if (!isEmbedded && !urlTaskId) {
      // Standalone with no task ID
      setPhase('error');
      setErrorMsg(
        'No task configuration provided. Pass a task ID in the URL (/task-runner/<task-id>) or embed this page with task config.',
      );
    } else if (isEmbedded) {
      setPhase('error');
      setErrorMsg('No task configuration or task ID received from parent app.');
    }
  }, [
    isEmbedded,
    externalApp.isReady,
    externalApp.receivedInputs,
    externalApp.receivedOutput,
    urlTaskId,
    fetchTask,
    createTask,
    sendResult,
    adoptSelection,
  ]);

  // Ref to always access the latest handleEvaluatorSaved without stale closures
  const handleEvaluatorSavedRef = useRef<
    | ((taskId: string, version: number, startStep?: number) => Promise<void>)
    | null
  >(null);

  /* --- step editing callbacks --- */
  const addStep = useCallback((type: StepType) => {
    setSteps(prev => {
      const newStep = makeDefaultStep(type, nextStepKeyRef.current++);
      if (type === 'deploy_agent') {
        const allEnvIds = prev
          .filter(s => s.type === 'deploy_env' && s.fields.env_id)
          .map(s => String(s.fields.env_id));
        newStep.fields.env_ids = allEnvIds.join(', ');
      }
      return [...prev, newStep];
    });
  }, []);

  const moveStep = useCallback((index: number, direction: -1 | 1) => {
    setSteps(prev => {
      const target = index + direction;
      if (target < 0 || target >= prev.length) return prev;
      const next = [...prev];
      const a = next[index]!,
        b = next[target]!;
      next[index] = b;
      next[target] = a;
      return next;
    });
    setSelectedStepIndex(prev => (prev === index ? index + direction : prev));
  }, []);

  const removeStep = useCallback((index: number) => {
    setSteps(prev => prev.filter((_, i) => i !== index));
  }, []);

  const updateStep = useCallback(
    (index: number, updates: Partial<StepState>) => {
      setSteps(prev =>
        prev.map((s, i) => (i === index ? { ...s, ...updates } : s)),
      );
    },
    [],
  );

  const updateField = useCallback(
    (index: number, key: string, value: unknown) => {
      setSteps(prev =>
        prev.map((s, i) =>
          i === index ? { ...s, fields: { ...s.fields, [key]: value } } : s,
        ),
      );
    },
    [],
  );

  /* --- save & re-run --- */
  const handleSave = useCallback(async () => {
    if (steps.length === 0) return;
    if (steps.some(s => !s.id.trim())) return;
    if (steps.some(s => getJsonError(s.advancedJson) !== null)) return;

    setSaving(true);
    try {
      const body = {
        id: resolvedTaskId ?? `task-${Date.now()}`,
        steps: steps.map(stepToDict),
      };
      const res = await apiFetch(`${BACKEND_URL}/api/v1/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Failed (${res.status})`);
      }
      const data = await res.json();
      setResolvedTaskId(data.id);
      setTaskVersion(data.version);
      setTask(prev =>
        prev
          ? { ...prev, version: data.version, steps: steps.map(stepToDict) }
          : prev,
      );
      setEditing(false);
      sendResult(data.id, data.version, 'updated');
      // Reset instances for the new version. Claiming (not a bare set) keeps the clear ours so an INIT_STATE echo can't re-apply old stars.
      setInstances([]);
      claimSelection(data.id, []);
      sentResultIdsRef.current = new Set();
      setSelectedInstanceId(null);
      snapshotRef.current = '';
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : 'Failed to save task');
    } finally {
      setSaving(false);
    }
  }, [steps, resolvedTaskId, sendResult, claimSelection]);

  /* --- upsert the parent's latest task_config onto the current task --- */
  // Returns the upserted { id, version } (or null). Does NOT run — that's the caller's job, so apply-then-run is one explicit gesture.
  const applyLatestConfig = useCallback(async (): Promise<{
    id: string;
    version: number;
  } | null> => {
    if (!parentTaskConfig) return null;
    const urlParams = new URLSearchParams(window.location.search);
    const pipelineTaskId = urlParams.get('taskId');
    const pipelineProjectId = urlParams.get('projectId');

    const writeId =
      resolvedTaskId ??
      parentTaskConfig.id ??
      (pipelineTaskId ? `aeh-${pipelineTaskId}` : `aeh-${Date.now()}`);

    const config = {
      ...parentTaskConfig,
      id: writeId,
      project_id:
        parentTaskConfig.project_id ||
        pipelineProjectId ||
        pipelineTaskId ||
        undefined,
    };

    // Reset instance UI — the version bump invalidates the polled list until the next poll. Claimed on writeId (the id we POST).
    setInstances([]);
    claimSelection(writeId, []);
    sentResultIdsRef.current = new Set();
    setSelectedInstanceId(null);
    snapshotRef.current = '';

    const result = await createTask(config);
    if (!result) return null;
    setResolvedTaskId(result.id);
    setTaskVersion(result.version);
    // Baseline = hash of the parent config just applied (input-side). Clears the drift warning until the parent differs; persisted for resume.
    const baseline = configHash(parentTaskConfig);
    setAppliedConfigHash(baseline);
    appliedConfigHashRef.current = baseline;
    sendResult(result.id, result.version, 'updated', baseline);

    fetchTask(result.id)
      .then(data => {
        setTask(data);
        const loadedSteps = (
          (data.steps ?? []) as Record<string, unknown>[]
        ).map((s: Record<string, unknown>) =>
          stepFromDict(s, nextStepKeyRef.current++),
        );
        setSteps(loadedSteps);
      })
      .catch(() => {});

    setPhase('idle');
    return result;
  }, [
    parentTaskConfig,
    resolvedTaskId,
    createTask,
    fetchTask,
    sendResult,
    claimSelection,
  ]);

  /* --- Run button: apply the parent's latest config first if it drifted --- */
  // "Apply Latest Config" is absorbed into Run. Reconcile only here, on an explicit click — never in an
  // effect — so the INIT_STATE↔SUBMISSION loop can't form. Drift is computed input-to-input from the ref.
  const handleRunLatest = useCallback(async () => {
    const parentHash = configHash(parentTaskConfig);
    const needsApply =
      parentTaskConfig != null &&
      parentHash !== null &&
      (appliedConfigHashRef.current === null ||
        parentHash !== appliedConfigHashRef.current);
    if (needsApply) {
      const result = await applyLatestConfig();
      if (!result) return;
      // resolvedTaskId is unchanged by apply (same id, new version); pass the
      // new version explicitly so we run what we just applied, not stale state.
      await handleRun({ version: result.version, count: runsPerClick });
      return;
    }
    await handleRun({ count: runsPerClick });
  }, [parentTaskConfig, applyLatestConfig, handleRun, runsPerClick]);

  // Drift signal for the warning banner: the parent's current config differs from what was last applied (seeded on resume from the persisted hash).
  const parentConfigHash = useMemo(
    () => configHash(parentTaskConfig),
    [parentTaskConfig],
  );
  const configDrifted =
    !!resolvedTaskId &&
    parentConfigHash !== null &&
    parentConfigHash !== appliedConfigHash;

  // Whether to surface the Run control. Standalone always; embedded once a task exists this session.
  // Run absorbs "Apply Latest Config", so without it the CUA interactive-hack flow has no button.
  const canRun = !isEmbedded || isRunnableInteractively || !!resolvedTaskId;

  // Cap concurrent runs at 1 unless the parent opted into max_runs. Blocks a new run while one is active so
  // a CB can't spawn parallel expensive interactive envs; re-runnable once terminal (per-run Cancel escapes).
  const activeRunExists = pendingRun || instances.some(i => !isTerminal(i));
  const blockedByActiveCap = maxRuns == null && activeRunExists;

  // Auto-run effect: triggers when resolvedTaskId is set after creation with auto_run
  const autoRunTriggeredRef = useRef(false);
  useEffect(() => {
    if (phase === 'running' && resolvedTaskId && !autoRunTriggeredRef.current) {
      autoRunTriggeredRef.current = true;
      handleRun();
    }
  }, [phase, resolvedTaskId, handleRun]);

  /* --- start a new run set, abandoning the current one --- */
  // The host drops the previous id's entries, keyed on data.task_id. Below, in order: derive the id from
  // parentTaskConfig.id (so suffixes don't stack); disarm auto-run (a rotated id upserts at v1); CLAIM the selection.
  const [resettingRunSet, setResettingRunSet] = useState(false);
  const handleResetRunSet = useCallback(async () => {
    const parent = parentTaskConfigRef.current;
    if (!resolvedTaskId || resettingRunSet || !parent) return;
    const rotated = `${parent.id || resolvedTaskId}-r${Date.now().toString(
      36,
    )}`;

    setResettingRunSet(true);
    setErrorMsg(null);
    const urlParams = new URLSearchParams(window.location.search);
    const pipelineTaskId = urlParams.get('taskId');
    const result = await createTask({
      ...parent,
      id: rotated,
      project_id:
        parent.project_id ||
        urlParams.get('projectId') ||
        pipelineTaskId ||
        undefined,
    });
    if (!result) {
      setResettingRunSet(false);
      return;
    }

    autoRunTriggeredRef.current = true;

    setInstances([]);
    setSelectedInstanceId(null);
    sentResultIdsRef.current = new Set();
    instanceIdsRef.current = new Set();
    snapshotRef.current = '';
    setLastContextJson(null);
    setResolvedTaskId(result.id);
    setTaskVersion(result.version);
    claimSelection(result.id, []);
    const baseline = configHash(parent);
    setAppliedConfigHash(baseline);
    appliedConfigHashRef.current = baseline;
    setPhase('idle');
    // Trips the host reset: new `task_id`, no `instances` key.
    sendResult(result.id, result.version, 'created', baseline);
    setResettingRunSet(false);
  }, [
    resolvedTaskId,
    resettingRunSet,
    createTask,
    claimSelection,
    sendResult,
    setLastContextJson,
  ]);

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
      claimSelection(savedTaskId, []);
      sentResultIdsRef.current = new Set();
      snapshotRef.current = '';

      // Refresh task data
      try {
        const data = await fetchTask(savedTaskId);
        setTask(data);
        const loadedSteps = (
          (data.steps ?? []) as Record<string, unknown>[]
        ).map((s: Record<string, unknown>) =>
          stepFromDict(s, nextStepKeyRef.current++),
        );
        setSteps(loadedSteps);

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
    [fetchTask, lastContextJson, lastContextSource, claimSelection],
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
  const taskSteps = (task?.steps ?? steps.map(stepToDict)) as Record<
    string,
    unknown
  >[];
  const displaySteps = editing ? steps.map(stepToDict) : taskSteps;

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
      init_config: s.init_config,
      evaluator: s.evaluator,
      env_id: (s.env_id ?? null) as string | null,
      agent_name: (s.agent_name ?? null) as string | null,
      triggers: s.triggers,
      osworld_v2_task_url: s.osworld_v2_task_url as string | undefined,
    }));
  }, [task]);

  // First grading step; re-running from here replays grading against the prior
  // run's context. -1 when the task grades nothing.
  const verifierStepIndex = displaySteps.findIndex(s => isGradingStep(s.type));

  // Config-supplied resume point, else derived. Indexed off the PARENT config (applyLatestConfig refreshes `task` async, so a task-derived index lags a version).
  const parentGradingStepIndex = useMemo(() => {
    const steps = parentTaskConfig?.steps;
    if (!steps) return -1;
    const first = steps.findIndex(st => isGradingStep(st.type));
    if (first < 0) return -1;
    // A rubrics_verifier naming an agent needs it DEPLOYED, so resuming at the verifier would skip its
    // deploy_agent/load_artifact. Walk back over the contiguous setup steps for that agent.
    const judge = steps[first]?.agent_name;
    if (
      steps[first]?.type !== 'rubrics_verifier' ||
      typeof judge !== 'string' ||
      judge === ''
    )
      return first;
    let start = first;
    while (start > 0) {
      const prev = steps[start - 1];
      if (prev?.agent_name !== judge || !isJudgeSetupStep(prev?.type)) break;
      start -= 1;
    }
    return start;
  }, [parentTaskConfig]);
  const resumeStep =
    gradeStartStep ??
    (parentGradingStepIndex >= 0 ? parentGradingStepIndex : verifierStepIndex);

  // One run per attempt, each resumed at the grading step against that attempt's
  // context.
  const gradeAllAttempts = useCallback(async () => {
    if (!resolvedTaskId || pendingRun || resumeStep < 0) return;
    const attempts = gradeAttempts.filter(a => !!a.context);
    if (attempts.length === 0) return;

    // Grade what the parent currently says, like handleRunLatest: `grade_start_step`
    // indexes the new step list, and the steps carry the verifier source.
    let version = taskVersion ?? undefined;
    const parentHash = configHash(parentTaskConfig);
    if (
      parentTaskConfig != null &&
      parentHash !== null &&
      (appliedConfigHashRef.current === null ||
        parentHash !== appliedConfigHashRef.current)
    ) {
      const applied = await applyLatestConfig();
      if (!applied) return;
      version = applied.version;
    }

    const room =
      maxRuns != null
        ? Math.max(0, maxRuns - instancesRef.current.length)
        : attempts.length;
    const planned = attempts.slice(0, room);
    if (planned.length === 0) {
      setErrorMsg(
        `Run cap reached (${maxRuns}) — nothing graded. Clear runs and retry.`,
      );
      return;
    }

    setPendingRun(true);
    setErrorMsg(null);
    const failures: string[] = [];

    // Launched together, like handleRun's `count` fan-out: every attempt differs
    // only in `context_json` / `agent_model`, so there is nothing to serialize.
    const results = await Promise.allSettled(
      planned.map(async attempt => {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            resolvedTaskId,
          )}/run`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(
              buildRunBody({
                version,
                start_step: resumeStep,
                agent_model: attempt.agent_model,
                context_json: attempt.context ?? null,
                step_overrides: attempt.step_overrides,
              }),
            ),
          },
        );
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(
            `${attempt.instance_id ?? 'attempt'}: ${
              errData.detail || `HTTP ${res.status}`
            }`,
          );
        }
        const data = await res.json();
        return (data.workflow_id as string | null) ?? null;
      }),
    );

    const started = results.filter(r => r.status === 'fulfilled').length;
    for (const r of results)
      if (r.status === 'rejected')
        failures.push(
          r.reason instanceof Error ? r.reason.message : 'failed to launch',
        );
    const lastWorkflowId =
      results.find(
        (r): r is PromiseFulfilledResult<string | null> =>
          r.status === 'fulfilled' && !!r.value,
      )?.value ?? null;

    lastWorkflowIdRef.current = lastWorkflowId;
    if (started === 0) {
      setPendingRun(false);
      setErrorMsg(`Failed to start grading. ${failures[0] ?? ''}`.trim());
    } else if (failures.length > 0) {
      // Keep pendingRun: the started runs are coming. Name the shortfall rather
      // than letting a partial grade read as a complete one.
      setErrorMsg(
        `Grading ${started} of ${attempts.length} attempt(s); ${failures.length} failed to launch. ${failures[0]}`,
      );
    } else if (planned.length < attempts.length) {
      setErrorMsg(
        `Grading ${planned.length} of ${attempts.length} attempt(s) — run cap (${maxRuns}) reached.`,
      );
    }
  }, [
    resolvedTaskId,
    pendingRun,
    resumeStep,
    gradeAttempts,
    maxRuns,
    taskVersion,
    buildRunBody,
    parentTaskConfig,
    applyLatestConfig,
  ]);

  const isCuaTask = taskSteps.some(
    s =>
      s.type === 'cua_initialize' ||
      s.type === 'cua_evaluate' ||
      (s.type === 'deploy_env' &&
        typeof s.env_id === 'string' &&
        s.env_id.includes('cua')),
  );
  const cuaEvalStep = taskSteps.find(s => s.type === 'cua_evaluate') as
    | Record<string, unknown>
    | undefined;
  const evaluatorConfig = cuaEvalStep?.evaluator as
    | Record<string, unknown>
    | undefined;
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
      {resolvedTaskId &&
        (!isEmbedded || isEditable || isRunnableInteractively) && (
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
      {(!isEmbedded || isEditable || isRunnableInteractively) &&
        displaySteps.length > 0 && (
          <div className="mb-6">
            <StepsPipeline
              steps={displaySteps}
              selectedIndex={selectedStepIndex}
              onStepSelect={editing ? setSelectedStepIndex : undefined}
            />
          </div>
        )}

      {/* Step editor (when editing) */}
      {editing && (
        <div className="mb-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
            Steps ({steps.length})
          </h3>
          {steps.length === 0 ? (
            <p className="text-sm text-[var(--muted-foreground)] mb-3">
              No steps. Add a step to get started.
            </p>
          ) : (
            <div className="flex flex-col gap-3 mb-3">
              {steps.map((step, i) => (
                <div key={step.key} onClick={() => setSelectedStepIndex(i)}>
                  <StepEditor
                    index={i}
                    step={step}
                    allSteps={steps}
                    isSelected={selectedStepIndex === i}
                    onUpdate={updateStep}
                    onUpdateField={updateField}
                    onRemove={removeStep}
                    onMove={moveStep}
                    totalSteps={steps.length}
                  />
                </div>
              ))}
            </div>
          )}
          {(() => {
            const recommended = getRecommendedStep(steps);
            return (
              <div className="flex items-center gap-2">
                {recommended && (
                  <button
                    onClick={() => addStep(recommended)}
                    className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-white transition-colors"
                    style={{
                      backgroundColor: STEP_COLORS[recommended] || '#6b7280',
                    }}
                  >
                    <Plus size={14} />
                    Add {STEP_TYPE_LABELS[recommended]}
                  </button>
                )}
                <DropdownMenu.Root>
                  <DropdownMenu.Trigger>
                    <button className="flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors">
                      <Plus size={14} />
                      {recommended ? 'Other Steps' : 'Add Step'}
                    </button>
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Content sideOffset={4} align="start">
                    {ADDABLE_STEP_TYPES.map(type => (
                      <DropdownMenu.Item
                        key={type}
                        onSelect={() => addStep(type)}
                      >
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{
                            backgroundColor: STEP_COLORS[type] || '#6b7280',
                          }}
                        />
                        {STEP_TYPE_LABELS[type]}
                      </DropdownMenu.Item>
                    ))}
                  </DropdownMenu.Content>
                </DropdownMenu.Root>
              </div>
            );
          })()}
          <div className="flex items-center gap-2 mt-4">
            <button
              onClick={handleSave}
              disabled={saving || steps.length === 0}
              className="flex items-center gap-1.5 px-4 py-2 rounded-md bg-violet-600 text-white text-sm font-medium hover:bg-violet-700 transition-colors disabled:opacity-50"
            >
              {saving ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Save size={14} />
              )}
              Save
            </button>
            <button
              onClick={() => {
                if (task) {
                  const restored = (
                    (task.steps ?? []) as Record<string, unknown>[]
                  ).map(s => stepFromDict(s, nextStepKeyRef.current++));
                  setSteps(restored);
                }
                setEditing(false);
              }}
              className="px-4 py-2 rounded-md border border-[var(--border)] text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Creating progress */}
      {phase === 'creating' && (
        <div className="mb-6 flex items-center gap-3 p-4 rounded-lg border border-[var(--border)] bg-[var(--secondary)]">
          <Loader2 size={20} className="animate-spin text-blue-600" />
          <p className="text-sm font-medium">Creating task...</p>
        </div>
      )}

      {/* Initializing */}
      {phase === 'initializing' && (
        <div className="mb-6 flex items-center gap-3 p-4 rounded-lg border border-[var(--border)] bg-[var(--secondary)]">
          <Loader2 size={20} className="animate-spin text-gray-500" />
          <p className="text-sm font-medium">
            {isEmbedded
              ? 'Waiting for configuration from parent...'
              : 'Initializing...'}
          </p>
        </div>
      )}

      {/* Runs */}
      {resolvedTaskId && phase !== 'initializing' && phase !== 'creating' && (
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Runs ({instances.length})
            </h3>
            {preferredRunCount != null && (
              <span className="text-xs font-semibold text-violet-600">
                Preferred: {preferredInstanceIds.size}/{preferredRunCount}{' '}
                selected
              </span>
            )}
          </div>

          {/* Drift warning — the parent's latest config isn't applied yet.
              "Apply Latest Config" is folded into Run, so clicking Run picks
              up these changes. */}
          {configDrifted && canRun && (
            <div className="mb-3 flex items-center gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              <AlertTriangle size={14} className="shrink-0" />
              <span>
                Your last run is saved and shown below. This task has newer
                changes that aren’t part of it — click Run only if you want to
                start a new run with those changes.
              </span>
            </div>
          )}

          {/* Run / Starting button, plus the run-set reset */}
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {canRun &&
              (pendingRun ? (
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
                <>
                  {/* Per-run model, only when the config offers a list. `agent_model` is a
                      RUN parameter, so switching models does not upsert a new version and
                      every attempt stays comparable (prompt_agent prefers
                      context.agent_model over the step's own `model`). */}
                  {agentModels.length > 0 && (
                    <select
                      value={agentModel}
                      onChange={e => setAgentModel(e.target.value)}
                      className="rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1.5 text-xs"
                      title="Model for the next run(s) — sent as agent_model, so no new task version"
                    >
                      {agentModels.map(m => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  )}
                  {/* Grading: one run per attempt handed down by the upstream runner
                    step, each resumed at the grading step against that attempt's
                    context. Replaces "Run" rather than sitting beside it — a fresh
                    run of a grading task would score a new attempt nobody asked
                    for, which is the confusing part worth designing out. */}
                  {gradeAttempts.length > 0 && resumeStep >= 0 ? (
                    <button
                      onClick={() => void gradeAllAttempts()}
                      disabled={
                        editing ||
                        blockedByActiveCap ||
                        (maxRuns != null && instances.length >= maxRuns)
                      }
                      className="flex items-center gap-1.5 px-4 py-2 rounded-md bg-violet-600 text-white text-sm font-medium hover:bg-violet-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Runs the rubric judge and unit tests once per starred attempt, against that attempt's own snapshot"
                    >
                      <Play size={14} />
                      {maxRuns != null && instances.length >= maxRuns
                        ? `Limit reached (${maxRuns})`
                        : blockedByActiveCap
                        ? 'Grading in progress'
                        : `Grade ${gradeAttempts.length} attempt${
                            gradeAttempts.length === 1 ? '' : 's'
                          }`}
                    </button>
                  ) : (
                    <button
                      onClick={() => handleRunLatest()}
                      disabled={
                        editing ||
                        blockedByActiveCap ||
                        (maxRuns != null && instances.length >= maxRuns)
                      }
                      className="flex items-center gap-1.5 px-4 py-2 rounded-md bg-violet-600 text-white text-sm font-medium hover:bg-violet-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Play size={14} />
                      {maxRuns != null && instances.length >= maxRuns
                        ? `Limit reached (${maxRuns})`
                        : blockedByActiveCap
                        ? 'Run in progress'
                        : 'Run'}
                    </button>
                  )}
                </>
              ))}

            {/* Disabled while a run is live: rotating orphans the in-flight
                instance — filtered out by task_id, never posted — mid-sandbox.
                Said inline too: `title` is suppressed on a disabled button. */}
            {parentTaskConfig && instances.length > 0 && (
              <div className="ml-auto flex items-center gap-2">
                {activeRunExists && !resettingRunSet && (
                  <span className="text-xs text-[var(--muted-foreground)]">
                    Available once the current run finishes
                  </span>
                )}
                <AlertDialog.Root>
                  <AlertDialog.Trigger>
                    <button
                      disabled={editing || resettingRunSet || activeRunExists}
                      title="Discard these runs and begin a fresh set"
                      className="flex items-center gap-1.5 px-3 py-2 rounded-md border border-[var(--border)] text-sm font-medium text-[var(--muted-foreground)] transition-colors hover:border-red-300 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:border-[var(--border)] disabled:hover:bg-transparent disabled:hover:text-[var(--muted-foreground)]"
                    >
                      {resettingRunSet ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <RotateCcw size={14} />
                      )}
                      {resettingRunSet ? 'Starting over…' : 'Start over'}
                    </button>
                  </AlertDialog.Trigger>
                  <AlertDialog.Content maxWidth="480px">
                    <AlertDialog.Title>
                      Discard these runs and start over?
                    </AlertDialog.Title>
                    <AlertDialog.Description size="2">
                      All {instances.length} run
                      {instances.length === 1 ? '' : 's'}
                      {preferredInstanceIds.size > 0 &&
                        ` and ${preferredInstanceIds.size} starred selection${
                          preferredInstanceIds.size === 1 ? '' : 's'
                        }`}{' '}
                      will be removed from this step, and a fresh run set takes
                      their place.
                    </AlertDialog.Description>
                    <div className="mt-3 flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                      <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                      <span>
                        This cannot be undone, and any attempt that has not been
                        graded becomes ungradable.
                      </span>
                    </div>
                    <Flex gap="3" mt="4" justify="end">
                      <AlertDialog.Cancel>
                        <Button variant="soft" color="gray">
                          Keep these runs
                        </Button>
                      </AlertDialog.Cancel>
                      <AlertDialog.Action>
                        <Button
                          color="red"
                          onClick={() => void handleResetRunSet()}
                        >
                          <RotateCcw size={14} />
                          Start over
                        </Button>
                      </AlertDialog.Action>
                    </Flex>
                  </AlertDialog.Content>
                </AlertDialog.Root>
              </div>
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
              const isPreferred = preferredInstanceIds.has(id);
              const atCap =
                preferredRunCount != null &&
                preferredInstanceIds.size >= preferredRunCount;
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
                  {preferredRunCount != null && status === 'completed' && (
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        // Ours from here on: no later INIT_STATE echo may seed
                        // this task's stars back over the user's choice.
                        selectionIntentRef.current = resolvedTaskId;
                        setPreferredInstanceIds(prev => {
                          const next = new Set(prev);
                          if (next.has(id)) next.delete(id);
                          else if (next.size < preferredRunCount) next.add(id);
                          return next;
                        });
                      }}
                      disabled={!isPreferred && atCap}
                      className={`mt-1 px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
                        isPreferred
                          ? 'border-amber-400 text-amber-600 bg-amber-50 hover:bg-amber-100'
                          : 'border-[var(--border)] text-[var(--muted-foreground)] hover:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed'
                      }`}
                    >
                      {isPreferred ? '★ Preferred' : '☆ Prefer'}
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
                  envType={isCuaTask ? 'cua' : undefined}
                  evaluatorConfig={evaluatorConfig}
                  rubricsCriteria={rubricsCriteria}
                  rubricsAggregator={rubricsAggregator}
                  taskSteps={viewerTaskSteps}
                  showRunContext={showRunContext}
                />
              );
            })()}
        </div>
      )}
    </div>
  );
}
