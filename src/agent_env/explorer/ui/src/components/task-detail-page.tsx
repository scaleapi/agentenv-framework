import React, {
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
} from 'react';
import {
  ArrowLeft,
  Ban,
  ChevronDown,
  CheckCircle2,
  XCircle,
  Download,
  Loader2,
  RefreshCw,
  X,
} from 'lucide-react';
import { Button } from '@radix-ui/themes';
import {
  BACKEND_URL,
  apiFetch,
  formatCellValue,
  SECTION_HEADER_CLASS,
} from './shared';
import { RubricMatrix, type MatrixRun } from './rubric-matrix';
import {
  type RubricCriterion,
  type VerificationResults,
} from './rubric-grading-results';
import { selectFinalScore } from '../lib/verifier-classification';
import { TaskInstanceViewer, type TaskStepRef } from './task-instance-viewer';
import { RunGroupStepFunnel } from './run-group-step-funnel';
import { StepsPipeline } from './steps-pipeline';
import { TriggersGraph } from './triggers-graph';
import { StartRunsPanel } from './start-runs-panel';
import {
  clearRolloutHash,
  CopyRolloutLinkButton,
  parseRolloutHash,
  useRolloutDeepLinkForTask,
  writeRolloutHash,
} from './rollout-deep-link';

interface RunGroup {
  run_group_id: string;
  task_version: number | null;
  earliest_created_at_utc: string | null;
  instances: Record<string, unknown>[];
  counts: Record<string, number>;
  step_counts?: Record<string, { done?: number; failed?: number }>;
}

function activeCount(counts: Record<string, number>): number {
  return (
    (counts.running ?? 0) + (counts.waiting ?? 0) + (counts.provisioning ?? 0)
  );
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600)
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

// First verification only — multi-rubric detail belongs in RubricMatrix.
function getInstanceScore(inst: Record<string, unknown>): number | null {
  if (inst.status !== 'completed') return null;
  const context = inst.context as Record<string, unknown> | null;
  if (!context) return null;
  const metadata = context.metadata as Record<string, unknown> | undefined;
  return selectFinalScore(
    metadata?.verifications as Record<string, unknown> | undefined,
  );
}

function getInstanceVerificationCount(inst: Record<string, unknown>): number {
  const context = inst.context as Record<string, unknown> | null;
  if (!context) return 0;
  const metadata = context.metadata as Record<string, unknown> | undefined;
  const verifications = metadata?.verifications as
    | Record<string, unknown>
    | undefined;
  return verifications ? Object.keys(verifications).length : 0;
}

// binary: pass iff score === 1 (ALL_PASS/ANY_PASS). gradient: partial credit
// (WEIGHTED_AVERAGE per-instance + group-header means).
type ScoreColorMode = 'binary' | 'gradient';

function aggregatorColorMode(
  aggregator: string | null | undefined,
): ScoreColorMode {
  return aggregator === 'weighted_average' ? 'gradient' : 'binary';
}

function getScoreColorClass(score: number, mode: ScoreColorMode): string {
  if (mode === 'binary') {
    return score >= 1 ? 'text-green-500' : 'text-red-500';
  }
  if (score >= 0.75) return 'text-green-500';
  if (score >= 0.25) return 'text-yellow-500';
  return 'text-red-500';
}

function getGroupDurationLabel(group: RunGroup, active: number): string {
  if (active > 0) {
    if (!group.earliest_created_at_utc) return '--';
    const startMs = new Date(group.earliest_created_at_utc).getTime();
    if (Number.isNaN(startMs)) return '--';
    const elapsed = Math.max(0, (Date.now() - startMs) / 1000);
    return formatDuration(elapsed);
  }
  let longest = -1;
  for (const inst of group.instances) {
    const d = inst.duration_seconds as number | null | undefined;
    if (typeof d === 'number' && d > longest) longest = d;
  }
  if (longest < 0) return '--';
  return formatDuration(longest);
}

function onActivateKey(
  e: React.KeyboardEvent<HTMLElement>,
  action: () => void,
) {
  if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
    e.preventDefault();
    action();
  }
}

/** The LLM that ran an instance, earliest authoritative source first so failed/running rows aren't blank:
 *  context.agent_model → prompt_responses[0].model → default_agent_model → null ("--"). */
function getInstanceModel(inst: Record<string, unknown>): string | null {
  const context = inst.context as Record<string, unknown> | null;
  if (!context) return null;

  const agentModel = context.agent_model;
  if (typeof agentModel === 'string' && agentModel) {
    return agentModel;
  }

  const promptResponses = context.prompt_responses as
    | Array<{ model?: string | null }>
    | undefined;
  const fromPrompt = promptResponses?.[0]?.model;
  if (typeof fromPrompt === 'string' && fromPrompt) {
    return fromPrompt;
  }

  const defaultAgentModel = context.default_agent_model;
  return typeof defaultAgentModel === 'string' && defaultAgentModel
    ? defaultAgentModel
    : null;
}

const POLL_INTERVAL_RUNNING_MS = 5000; // refresh the Rollouts table while anything is still running
const CANCEL_FALLBACK_TIMEOUT_MS = 60_000;
const CANCEL_TOAST_DURATION_MS = 8_000;
// Heuristic for the re-collect expiry hint only — the backend liveness probe is the real gate.

const PAGE_SIZE_OPTIONS = [10, 25, 50] as const;

const ROLLOUTS_PAGE_SIZE_KEY = 'agent-env-explorer:rollouts-page-size';
const EXPANDED_INSTANCE_KEY_PREFIX = 'agent-env-explorer:expanded-instance:';

/** useState mirrored to localStorage. Reads on mount and when `key` changes (so taskId-scoped keys swap
 *  per task); writes on change. SSR-safe. */
function usePersistentState<T>(
  key: string,
  initial: T,
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const read = useCallback((): T => {
    if (typeof window === 'undefined') return initial;
    try {
      const raw = window.localStorage.getItem(key);
      return raw == null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  }, [key, initial]);

  const [state, setState] = useState<T>(read);

  // Re-read when key changes (e.g. taskId changed and the key is taskId-scoped).
  const lastKey = useRef(key);
  useEffect(() => {
    if (lastKey.current === key) return;
    lastKey.current = key;
    setState(read());
  }, [key, read]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // Quota / private mode — non-fatal, drop the persistence.
    }
  }, [key, state]);

  return [state, setState];
}

/** Stable run-group comparator: earliest_created_at_utc DESC, run_group_id ASC tiebreaker (the backend's
 *  $sort after $group isn't stable on ties, so groups can otherwise swap on each poll). */
function compareRunGroup(a: RunGroup, b: RunGroup): number {
  const ta = a.earliest_created_at_utc ?? '';
  const tb = b.earliest_created_at_utc ?? '';
  if (ta !== tb) return tb.localeCompare(ta);
  return a.run_group_id.localeCompare(b.run_group_id);
}

/** Stable instance comparator within a group: created_at_utc ASC, instance_id ASC tiebreaker. */
function compareInstance(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
): number {
  const ta = String(a.created_at_utc ?? '');
  const tb = String(b.created_at_utc ?? '');
  if (ta !== tb) return ta.localeCompare(tb);
  return String(a.instance_id ?? '').localeCompare(String(b.instance_id ?? ''));
}

interface RunGroupSummary {
  run_group_id: string;
  task_id: string;
  total: number;
  completed: number;
  failed: number;
  running: number;
  instances: Array<{
    instance_id?: string | null;
    workflow_id?: string | null;
    status: string;
  }>;
}

/** Per-row button: fetches a run group's trajectory manifest (JSON of presigned URLs, 24h TTL) and
 *  downloads it. Accepts a runGroupId or an instance_id. */
function ManifestDownloadButton({
  taskId,
  runGroupId,
  className,
}: {
  taskId: string;
  runGroupId: string;
  className?: string;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const onClick = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(
        `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
          taskId,
        )}/runs/${encodeURIComponent(runGroupId)}/trajectory-manifest`,
      );
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(
          `manifest request failed: ${res.status} ${res.statusText}${
            detail ? ` — ${detail.slice(0, 200)}` : ''
          }`,
        );
      }
      const manifest = await res.json();
      const blob = new Blob([JSON.stringify(manifest, null, 2)], {
        type: 'application/json',
      });
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = `${runGroupId}.trajectory-manifest.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Defer revoke so Safari has a chance to fire the download.
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [taskId, runGroupId]);

  return (
    <button
      type="button"
      onClick={e => {
        e.stopPropagation();
        onClick();
      }}
      onKeyDown={e =>
        onActivateKey(e, () => {
          onClick();
        })
      }
      disabled={loading}
      aria-label={`Download trajectory manifest for run group ${runGroupId}`}
      title="Download trajectory manifest (JSON of presigned S3 URLs, valid 24h)"
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded px-2 py-1 text-xs hover:bg-[var(--accent)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)] disabled:opacity-50 ${
        className ?? ''
      }`}
    >
      {loading ? (
        <Loader2 size={12} className="animate-spin" aria-hidden />
      ) : (
        <Download size={12} aria-hidden />
      )}
      <span>Trajectory Manifest</span>
      {error && (
        <span role="alert" className="text-red-500" title={error}>
          failed
        </span>
      )}
    </button>
  );
}

interface InstanceRowProps {
  taskId: string;
  inst: Record<string, unknown>;
  nested: boolean;
  expandedInstance: string | null;
  setExpandedInstance: React.Dispatch<React.SetStateAction<string | null>>;
  cancellingInstances: Set<string>;
  handleCancelInstance: (instanceId: string, workflowId?: string) => void;
  rubricsCriteria: Record<string, unknown>[] | undefined;
  // Score aggregator from the rubrics_verifier step. Forwarded to RubricGradingResults to hide the Score badge for all_pass (redundant with N/M passed).
  rubricsAggregator: string | undefined;
  fullInstance: Record<string, unknown> | undefined;
  isFullInstanceLoading: boolean;
  fullInstanceError: string | null;
  onRetryFullInstance: () => void;
  scoreColorMode: ScoreColorMode;
  // Ordered {id, prompt_id} from the task definition. Forwarded to TaskInstanceViewer to label trajectories by step.id and sort in pipeline order.
  taskSteps: TaskStepRef[];
  // Deep-linked instance_id: the matching row briefly highlights and scrolls in.
  highlightInstanceId: string | null;
}

function InstanceRow({
  taskId,
  inst,
  nested,
  expandedInstance,
  setExpandedInstance,
  cancellingInstances,
  handleCancelInstance,
  rubricsCriteria,
  rubricsAggregator,
  fullInstance,
  scoreColorMode,
  isFullInstanceLoading,
  fullInstanceError,
  onRetryFullInstance,
  taskSteps,
  highlightInstanceId,
}: InstanceRowProps) {
  const isPending = inst.__pending === true;
  const instId = String(inst.instance_id);
  // /run-groups rows carry workflow_id at the top level and no context at all; the
  // context path only resolves once the full instance doc has been lazily fetched, so
  // reading it alone left Cancel with undefined on every freshly-listed row.
  const instWorkflowId = (inst.workflow_id ??
    (
      (inst.context as Record<string, unknown> | undefined)?.metadata as
        | Record<string, unknown>
        | undefined
    )?.workflow_id) as string | undefined;
  const isExpanded = !isPending && expandedInstance === instId;
  const instScore = getInstanceScore(inst);
  const verificationCount = getInstanceVerificationCount(inst);
  const instModel = getInstanceModel(inst);
  const panelId = `instance-panel-${instId}`;
  const panelRef = useRef<HTMLTableRowElement | null>(null);
  const rowRef = useRef<HTMLTableRowElement | null>(null);
  const [hover, setHover] = useState(false);

  const isHighlighted = !isPending && highlightInstanceId === instId;

  useEffect(() => {
    if (isExpanded && panelRef.current) {
      panelRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [isExpanded]);

  useEffect(() => {
    if (isHighlighted && rowRef.current) {
      rowRef.current.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [isHighlighted]);

  const toggle = useCallback(() => {
    if (isPending) return;
    // Mirror the open/closed panel in the URL hash so it stays shareable.
    if (isExpanded) {
      clearRolloutHash();
    } else {
      writeRolloutHash({
        instanceId: instId,
        version: (inst.task_version as number | null) ?? null,
      });
    }
    setExpandedInstance(prev => (prev === instId ? null : instId));
  }, [isPending, isExpanded, instId, inst.task_version, setExpandedInstance]);

  const boxShadow = isHighlighted
    ? 'inset 0 0 0 2px var(--ring)'
    : isExpanded
    ? 'inset 3px 0 0 0 var(--ring)'
    : hover && !isPending
    ? 'inset 3px 0 0 0 var(--ring)'
    : undefined;

  return (
    <React.Fragment>
      <tr
        ref={rowRef}
        onClick={toggle}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        className={`border-t border-[var(--border)] align-top transition-colors ${
          isPending
            ? 'bg-[var(--secondary)]/40'
            : 'hover:bg-[var(--secondary)] cursor-pointer'
        } ${isExpanded ? 'bg-[var(--secondary)]' : ''} group`}
        style={{
          boxShadow,
          opacity: isPending ? 0.85 : undefined,
          borderLeft: isPending
            ? '2px dashed var(--muted-foreground)'
            : undefined,
        }}
        title={
          isPending
            ? 'Awaiting worker — the run was submitted but the worker has not yet registered it.'
            : undefined
        }
      >
        <td
          className={`px-3 py-2 font-mono text-xs ${nested ? 'pl-8' : ''}`}
          title={isPending ? 'pending — workflow id not yet observed' : instId}
        >
          {isPending ? (
            <span className="text-[var(--muted-foreground)] italic">
              awaiting worker…
            </span>
          ) : (
            <span className="flex items-start gap-1">
              <span className="min-w-0 break-words">{instId}</span>
              <CopyRolloutLinkButton
                link={{
                  instanceId: instId,
                  version: (inst.task_version as number | null) ?? null,
                }}
              />
            </span>
          )}
        </td>
        <td
          className="px-3 py-2 text-xs text-[var(--foreground)]"
          title={instModel ?? undefined}
        >
          {instModel ? (
            <span className="block truncate">{instModel}</span>
          ) : (
            <span className="text-[var(--muted-foreground)]">--</span>
          )}
        </td>
        <td
          className="px-3 py-2 font-mono text-xs text-[var(--muted-foreground)]"
          title={String(inst.run_group_id ?? '')}
        >
          <span className="block truncate">
            {String(inst.run_group_id ?? '--')}
          </span>
        </td>
        <td className="px-3 py-2 text-xs text-[var(--muted-foreground)]">
          {inst.task_version != null ? `v${inst.task_version}` : '--'}
        </td>
        <td className="px-3 py-2 text-xs text-[var(--muted-foreground)]">
          {inst.created_at_utc
            ? formatCellValue('created_at_utc', inst.created_at_utc)
            : '--'}
        </td>
        <td className="px-3 py-2 text-xs text-[var(--muted-foreground)]">
          {formatDuration(inst.duration_seconds as number | null)}
        </td>
        <td className="px-3 py-2">
          {cancellingInstances.has(instId) ? (
            <span
              className="flex items-center gap-1.5 text-xs text-orange-500"
              title="Cancellation requested — waiting for terminal status"
            >
              <Loader2 size={14} className="animate-spin" aria-hidden />
              Cancelling…
            </span>
          ) : isPending ? (
            <span
              className="flex items-center gap-1.5 text-xs text-amber-500"
              title="Worker has not yet registered this run"
            >
              <Loader2 size={14} className="animate-spin" aria-hidden />
              Awaiting worker
            </span>
          ) : inst.status === 'waiting' ? (
            <span className="flex items-center gap-1.5 text-xs text-blue-500">
              <Loader2 size={14} className="animate-spin" aria-hidden />
              Waiting to start
            </span>
          ) : inst.status === 'completed' ? (
            <span className="flex items-center gap-1.5 text-xs text-green-500">
              <CheckCircle2 size={14} aria-hidden />
              Completed
            </span>
          ) : inst.status === 'failed' ? (
            <span
              className="flex items-center gap-1.5 text-xs text-red-500"
              title={typeof inst.error === 'string' ? inst.error : undefined}
            >
              <XCircle size={14} aria-hidden />
              Failed
            </span>
          ) : inst.status === 'running' ? (
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1.5 text-xs text-yellow-500">
                <Loader2 size={14} className="animate-spin" aria-hidden />
                In Progress
              </span>
              <button
                onClick={e => {
                  e.stopPropagation();
                  handleCancelInstance(instId, instWorkflowId);
                }}
                disabled={cancellingInstances.has(instId)}
                aria-label={`Cancel run ${instId}`}
                className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-red-500 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500/50"
                title="Cancel this run"
              >
                <XCircle size={12} aria-hidden />
                Cancel
              </button>
            </div>
          ) : inst.status === 'provisioning' ? (
            // Distinct from `running`: queued by the runner but no worker has picked it up yet (not executing a step). Still cancellable.
            <div className="flex items-center gap-2">
              <span
                className="flex items-center gap-1.5 text-xs text-amber-500"
                title="Queued — no worker has started it yet"
              >
                <Loader2 size={14} className="animate-spin" aria-hidden />
                Provisioning
              </span>
              <button
                onClick={e => {
                  e.stopPropagation();
                  handleCancelInstance(instId, instWorkflowId);
                }}
                disabled={cancellingInstances.has(instId)}
                aria-label={`Cancel run ${instId}`}
                className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-red-500 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500/50"
                title="Cancel this run"
              >
                <XCircle size={12} aria-hidden />
                Cancel
              </button>
            </div>
          ) : inst.status === 'cancelled' ? (
            <span className="flex items-center gap-1.5 text-xs text-[var(--muted-foreground)]">
              <Ban size={14} aria-hidden />
              Cancelled
            </span>
          ) : (
            <span className="text-xs text-[var(--muted-foreground)]">
              {String(inst.status ?? '--')}
            </span>
          )}
        </td>
        <td className="px-3 py-2">
          {instScore != null ? (
            <span
              className={`inline-flex items-center gap-1 text-xs font-medium ${getScoreColorClass(
                instScore,
                scoreColorMode,
              )}`}
              title={
                verificationCount > 1
                  ? `Score from first of ${verificationCount} rubrics — see Rubric Verifier Matrix below for full detail`
                  : undefined
              }
            >
              {scoreColorMode === 'binary' ? (
                instScore >= 1 ? (
                  <CheckCircle2 size={12} aria-hidden />
                ) : (
                  <XCircle size={12} aria-hidden />
                )
              ) : instScore >= 0.75 ? (
                <CheckCircle2 size={12} aria-hidden />
              ) : instScore < 0.25 ? (
                <XCircle size={12} aria-hidden />
              ) : null}
              {(instScore * 100).toFixed(0)}%
              {verificationCount > 1 ? (
                <span className="text-[var(--muted-foreground)] ml-0.5">*</span>
              ) : null}
            </span>
          ) : (
            <span className="text-xs text-[var(--muted-foreground)]">--</span>
          )}
        </td>
        <td className="px-3 py-2">
          {!isPending && !nested ? (
            <div className="flex flex-col items-start gap-1">
              <ManifestDownloadButton
                taskId={taskId}
                runGroupId={String(inst.run_group_id ?? inst.instance_id ?? '')}
              />
            </div>
          ) : null}
        </td>
        <td className="px-2 py-2 text-[var(--muted-foreground)]">
          {!isPending && (
            <button
              type="button"
              onClick={e => {
                e.stopPropagation();
                toggle();
              }}
              onKeyDown={e => onActivateKey(e, toggle)}
              aria-expanded={isExpanded}
              aria-controls={panelId}
              aria-label={
                isExpanded
                  ? `Collapse details for ${instId}`
                  : `Expand details for ${instId}`
              }
              className="inline-flex items-center justify-center rounded p-1 hover:bg-[var(--accent)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
            >
              <ChevronDown
                size={14}
                aria-hidden
                className={`transition-transform ${
                  isExpanded ? '' : '-rotate-90'
                }`}
              />
            </button>
          )}
        </td>
      </tr>
      {isExpanded && (
        <tr
          ref={panelRef}
          id={panelId}
          className="border-t border-[var(--border)]"
          style={{ boxShadow: 'inset 3px 0 0 0 var(--ring)' }}
        >
          <td colSpan={10} className="p-4">
            {isFullInstanceLoading && !fullInstance && (
              <div
                role="status"
                aria-live="polite"
                className="flex items-center gap-2 text-xs text-[var(--muted-foreground)] mb-3"
              >
                <Loader2 size={14} className="animate-spin" aria-hidden />
                Loading full trajectory…
              </div>
            )}
            {fullInstanceError && !fullInstance && (
              <div
                role="alert"
                className="flex items-center justify-between gap-3 mb-3 px-3 py-2 rounded border border-red-500/30 bg-red-500/5 text-xs text-red-500"
              >
                <span>
                  Couldn't load full trajectory: {fullInstanceError}. Showing
                  partial data.
                </span>
                <button
                  type="button"
                  onClick={onRetryFullInstance}
                  className="px-2 py-0.5 rounded border border-red-500/30 hover:bg-red-500/10"
                >
                  Retry
                </button>
              </div>
            )}
            <TaskInstanceViewer
              instance={fullInstance ?? inst}
              taskId={taskId}
              rubricsCriteria={rubricsCriteria}
              rubricsAggregator={rubricsAggregator}
              taskSteps={taskSteps}
            />
          </td>
        </tr>
      )}
    </React.Fragment>
  );
}

interface GroupHeaderRowProps {
  taskId: string;
  group: RunGroup;
  expanded: boolean;
  childOpen: boolean;
  setGroupOverrides: React.Dispatch<React.SetStateAction<Map<string, boolean>>>;
  setExpandedInstance: React.Dispatch<React.SetStateAction<string | null>>;
}

function GroupHeaderRow({
  taskId,
  group,
  expanded,
  childOpen,
  setGroupOverrides,
  setExpandedInstance,
}: GroupHeaderRowProps) {
  const total = group.instances.length;
  let scoreSum = 0;
  let scoreN = 0;
  for (const inst of group.instances) {
    const s = getInstanceScore(inst);
    if (s != null) {
      scoreSum += s;
      scoreN += 1;
    }
  }
  const aggScore = scoreN > 0 ? scoreSum / scoreN : null;
  const active = activeCount(group.counts);
  const completed = group.counts.completed ?? 0;
  const failed = group.counts.failed ?? 0;
  const cancelled = group.counts.cancelled ?? 0;
  const durationLabel = getGroupDurationLabel(group, active);
  const durationTooltip =
    active > 0
      ? `Time since the earliest run started; ${active} still active`
      : durationLabel === '--'
      ? undefined
      : `Longest among ${total} runs`;

  // Closing the group also closes any open viewer; otherwise an explicit
  // collapse is a no-op while childOpen keeps expanded forced to true.
  const runGroupId = group.run_group_id;
  const onToggle = useCallback(() => {
    const next = !expanded;
    setGroupOverrides(prev => {
      const m = new Map(prev);
      m.set(runGroupId, next);
      return m;
    });
    if (!next && childOpen) setExpandedInstance(null);
  }, [expanded, childOpen, runGroupId, setGroupOverrides, setExpandedInstance]);

  return (
    <tr
      onClick={onToggle}
      className="border-t border-[var(--border)] align-top bg-[var(--secondary)] hover:bg-[var(--accent)] cursor-pointer transition-colors"
    >
      <td className="px-3 py-2 font-mono text-xs">
        <span className="inline-flex items-center gap-1.5">
          <ChevronDown
            size={14}
            aria-hidden
            className={`transition-transform ${expanded ? '' : '-rotate-90'}`}
          />
          <span className="font-semibold">
            {total} {total === 1 ? 'run' : 'runs'}
          </span>
        </span>
      </td>
      <td className="px-3 py-2 text-xs text-[var(--muted-foreground)]">--</td>
      <td
        className="px-3 py-2 font-mono text-xs text-[var(--muted-foreground)]"
        title={group.run_group_id}
      >
        <span className="block truncate">{group.run_group_id}</span>
      </td>
      <td className="px-3 py-2 text-xs text-[var(--muted-foreground)]">
        {group.task_version != null ? `v${group.task_version}` : '--'}
      </td>
      <td className="px-3 py-2 text-xs text-[var(--muted-foreground)]">
        {group.earliest_created_at_utc
          ? formatCellValue('created_at_utc', group.earliest_created_at_utc)
          : '--'}
      </td>
      <td
        className="px-3 py-2 text-xs text-[var(--muted-foreground)]"
        title={durationTooltip}
      >
        {durationLabel}
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2 text-xs">
          {active > 0 && (
            <span
              className="flex items-center gap-1 text-yellow-500"
              aria-label={`${active} active`}
              title={`${active} active (running, waiting, or provisioning)`}
            >
              <Loader2 size={12} className="animate-spin" aria-hidden />
              {active}
            </span>
          )}
          {completed > 0 && (
            <span
              className="flex items-center gap-1 text-green-500"
              aria-label={`${completed} completed`}
              title={`${completed} completed`}
            >
              <CheckCircle2 size={12} aria-hidden />
              {completed}
            </span>
          )}
          {failed > 0 && (
            <span
              className="flex items-center gap-1 text-red-500"
              aria-label={`${failed} failed`}
              title={`${failed} failed`}
            >
              <XCircle size={12} aria-hidden />
              {failed}
            </span>
          )}
          {cancelled > 0 && (
            <span
              className="flex items-center gap-1 text-[var(--muted-foreground)]"
              aria-label={`${cancelled} cancelled`}
              title={`${cancelled} cancelled`}
            >
              <Ban size={12} aria-hidden />
              {cancelled}
            </span>
          )}
        </div>
      </td>
      <td className="px-3 py-2">
        {aggScore != null ? (
          <span
            className={`inline-flex items-center gap-1 text-xs font-medium ${getScoreColorClass(
              aggScore,
              // Group score is the mean — gradient even under a binary aggregator.
              'gradient',
            )}`}
            title={`Mean score across ${scoreN} of ${total} runs`}
          >
            {(aggScore * 100).toFixed(0)}%
          </span>
        ) : (
          <span className="text-xs text-[var(--muted-foreground)]">--</span>
        )}
      </td>
      <td className="px-3 py-2">
        <div className="flex flex-col items-start gap-1">
          <ManifestDownloadButton taskId={taskId} runGroupId={runGroupId} />
        </div>
      </td>
      <td className="px-2 py-2 text-[var(--muted-foreground)]">
        <button
          type="button"
          onClick={e => {
            e.stopPropagation();
            onToggle();
          }}
          onKeyDown={e => onActivateKey(e, onToggle)}
          aria-expanded={expanded}
          aria-label={
            expanded
              ? `Collapse run group ${group.run_group_id}`
              : `Expand run group ${group.run_group_id}`
          }
          className="inline-flex items-center justify-center rounded p-1 hover:bg-[var(--accent)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
        >
          <ChevronDown
            size={14}
            aria-hidden
            className={`transition-transform ${expanded ? '' : '-rotate-90'}`}
          />
        </button>
      </td>
    </tr>
  );
}

export function TaskDetailPage({
  taskId,
  onBack,
}: {
  taskId: string;
  onBack: () => void;
}) {
  const [task, setTask] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const [runGroups, setRunGroups] = useState<RunGroup[]>([]);
  const [instancesHasMore, setInstancesHasMore] = useState(false);
  const [instancesPage, setInstancesPage] = useState(0);
  const [instancesLoading, setInstancesLoading] = useState(true);
  const [instancesError, setInstancesError] = useState<string | null>(null);
  // Keyed by instance_id (not index) to survive group reordering. Persisted per-task so the open panel survives refreshes and reloads.
  const [expandedInstance, setExpandedInstance] = usePersistentState<
    string | null
  >(`${EXPANDED_INSTANCE_KEY_PREFIX}${taskId}`, null);
  // tristate: true = expanded, false = collapsed, missing = default
  // (expanded iff group has active runs).
  const [groupOverrides, setGroupOverrides] = useState<Map<string, boolean>>(
    () => new Map(),
  );
  // Lazy-loaded full instance docs; the slim /run-groups list omits
  // context.prompt_responses etc.
  const [fullInstances, setFullInstances] = useState<
    Map<string, Record<string, unknown>>
  >(() => new Map());
  const [fullInstanceLoading, setFullInstanceLoading] = useState<Set<string>>(
    () => new Set(),
  );
  const [fullInstanceErrors, setFullInstanceErrors] = useState<
    Map<string, string>
  >(() => new Map());
  const [fullInstanceRetryTick, setFullInstanceRetryTick] = useState(0);
  const [versionFilter, setVersionFilter] = useState<string>('all');

  // The #<instance_id>__<version> deep link, if any: selects the version filter, opens the trajectory, highlights the row.
  const rolloutDeepLink = useRolloutDeepLinkForTask(taskId);
  const [highlightInstanceId, setHighlightInstanceId] = useState<string | null>(
    null,
  );
  // Open+highlight applies once per task. (Version is applied in the task fetch
  // below, so it costs no extra /run-groups call.)
  const deepLinkFocusAppliedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch(`${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(taskId)}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
        return res.json();
      })
      .then(data => {
        if (cancelled) return;
        setTask(data);
        if (data?.version != null) {
          // Prefer the deep link's version (read live from the hash) so the
          // linked run group is in the first /run-groups fetch — no extra call.
          const linkedVersion = parseRolloutHash(window.location.hash)?.version;
          setVersionFilter(
            String(linkedVersion != null ? linkedVersion : data.version),
          );
        }
        setLoading(false);
      })
      .catch(e => {
        if (!cancelled) {
          setFetchError(e instanceof Error ? e.message : 'Failed to load');
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const [runGroupsPageSize, setRunGroupsPageSize] = usePersistentState<number>(
    ROLLOUTS_PAGE_SIZE_KEY,
    10,
  );

  const initialLoadDone = useRef(false);
  const fetchGeneration = useRef(0);
  // Stampede guard: short-circuit if another fetch started <250ms ago.
  const lastFetchAtRef = useRef(0);
  useEffect(() => {
    initialLoadDone.current = false;
    setRunGroups([]);
    setGroupOverrides(new Map());
    // Re-arm the deep-link focus for the new task.
    deepLinkFocusAppliedRef.current = false;
    setHighlightInstanceId(null);
    // expandedInstance is intentionally NOT reset — usePersistentState re-reads its taskId-scoped key, so each task's open panel survives navigation.
    setFullInstances(new Map());
    setFullInstanceLoading(new Set());
    setFullInstanceErrors(new Map());
  }, [taskId]);
  // groupOverrides intentionally NOT reset on versionFilter change.

  const fetchRunGroups = useCallback(
    (showSpinner = false) => {
      // versionFilter isn't authoritative until `task` has loaded.
      if (task == null) return Promise.resolve();
      if (!showSpinner && Date.now() - lastFetchAtRef.current < 250) {
        return Promise.resolve();
      }
      lastFetchAtRef.current = Date.now();
      if (!initialLoadDone.current || showSpinner) setInstancesLoading(true);
      setInstancesError(null);
      const gen = ++fetchGeneration.current;
      const params = new URLSearchParams({
        limit: String(runGroupsPageSize),
        offset: String(instancesPage * runGroupsPageSize),
      });
      if (versionFilter !== 'all') params.set('task_version', versionFilter);
      return apiFetch(
        `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
          taskId,
        )}/run-groups?${params}`,
      )
        .then(res => {
          if (!res.ok)
            throw new Error(`Failed to fetch run groups (${res.status})`);
          return res.json();
        })
        .then((data: { items: RunGroup[]; has_more?: boolean }) => {
          if (gen !== fetchGeneration.current) return;
          setRunGroups(data.items);
          setInstancesHasMore(Boolean(data.has_more));
          setInstancesLoading(false);
          initialLoadDone.current = true;
        })
        .catch(e => {
          if (gen !== fetchGeneration.current) return;
          setInstancesError(
            e instanceof Error ? e.message : 'Failed to load run groups',
          );
          setInstancesLoading(false);
          initialLoadDone.current = true;
        });
    },
    [task, taskId, versionFilter, instancesPage, runGroupsPageSize],
  );

  useEffect(() => {
    fetchRunGroups();
  }, [fetchRunGroups]);

  useEffect(() => {
    setInstancesPage(0);
  }, [versionFilter, runGroupsPageSize]);

  const steps = (task?.steps ?? []) as Record<string, unknown>[];
  const rubricsStep = steps.find(s => s.type === 'rubrics_verifier') as
    | Record<string, unknown>
    | undefined;
  const rubricsCriteria = rubricsStep?.criteria as
    | Record<string, unknown>[]
    | undefined;
  // Default to binary so a missing aggregator never paints partial green.
  const verifierStep =
    rubricsStep ??
    (steps.find(s => s.type === 'env_outcome_verifier') as
      | Record<string, unknown>
      | undefined);
  const scoreColorMode = aggregatorColorMode(
    verifierStep?.score_aggregator as string | undefined,
  );

  // Slim {id, prompt_id} list for TaskInstanceViewer (labels trajectories by step.id, sorts in pipeline order).
  // Memoized off `task` so identity is stable across polls, else the trajectory-setup effect resets loaded trajectories.
  const taskSteps: TaskStepRef[] = useMemo(() => {
    const rawSteps = (task?.steps ?? []) as Record<string, unknown>[];
    return rawSteps.map(s => ({
      id: String(s.id ?? ''),
      prompt_id: (s.prompt_id ?? null) as string | null,
      type: (s.type ?? undefined) as string | undefined,
      target: (s.target ?? undefined) as string | undefined,
      base_path: (s.base_path ?? undefined) as string | undefined,
      artifact_paths: (s.artifact_paths ?? undefined) as string[] | undefined,
      env_id: (s.env_id ?? null) as string | null,
      agent_name: (s.agent_name ?? null) as string | null,
      triggers: s.triggers,
    }));
  }, [task]);

  const handleRefreshInstances = useCallback(
    () => fetchRunGroups(true),
    [fetchRunGroups],
  );

  const displayRunGroups = React.useMemo<RunGroup[]>(() => {
    // Deterministic tiebreaker (run_group_id) — the backend's $sort after $group isn't stable on ties. Instances
    // also get an explicit (created_at_utc, instance_id) sort so child rows don't shuffle between polls.
    return runGroups
      .map(g => ({ ...g, instances: [...g.instances].sort(compareInstance) }))
      .sort(compareRunGroup);
  }, [runGroups]);

  const flatInstances = React.useMemo(
    () => runGroups.flatMap(g => g.instances),
    [runGroups],
  );

  // Once the linked instance loads, open its trajectory so a shared link lands on the instance view (auto-expands the group, persists expandedInstance).
  useEffect(() => {
    if (deepLinkFocusAppliedRef.current) return;
    if (!rolloutDeepLink || instancesLoading) return;
    const group = displayRunGroups.find(g =>
      g.instances.some(
        i => String(i.instance_id) === rolloutDeepLink.instanceId,
      ),
    );
    if (!group) return; // not on this page — see "no pagination" note
    deepLinkFocusAppliedRef.current = true;
    setExpandedInstance(rolloutDeepLink.instanceId);
    setHighlightInstanceId(rolloutDeepLink.instanceId);
  }, [
    rolloutDeepLink,
    instancesLoading,
    displayRunGroups,
    setExpandedInstance,
  ]);

  // Fade the highlight after a beat (scroll has landed by then).
  useEffect(() => {
    if (!highlightInstanceId) return;
    const t = setTimeout(() => setHighlightInstanceId(null), 2500);
    return () => clearTimeout(t);
  }, [highlightInstanceId]);

  const [cancellingInstances, setCancellingInstances] = useState<Set<string>>(
    () => new Set(),
  );
  const [cancelErrorToast, setCancelErrorToast] = useState<string | null>(null);
  const cancelToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  // Fallback timers so a stuck "Cancelling…" overlay eventually clears.
  const cancelFallbackTimersRef = useRef<
    Map<string, ReturnType<typeof setTimeout>>
  >(new Map());
  useEffect(
    () => () => {
      if (cancelToastTimerRef.current)
        clearTimeout(cancelToastTimerRef.current);
      for (const t of cancelFallbackTimersRef.current.values()) clearTimeout(t);
      cancelFallbackTimersRef.current.clear();
    },
    [],
  );

  const showCancelToast = useCallback((msg: string) => {
    if (cancelToastTimerRef.current) clearTimeout(cancelToastTimerRef.current);
    setCancelErrorToast(msg);
    cancelToastTimerRef.current = setTimeout(
      () => setCancelErrorToast(null),
      CANCEL_TOAST_DURATION_MS,
    );
  }, []);

  const handleCancelInstance = useCallback(
    async (instanceId: string, workflowId?: string) => {
      if (!workflowId) {
        showCancelToast('Cannot cancel: run has no workflow id yet.');
        return;
      }
      setCancellingInstances(prev => new Set(prev).add(instanceId));
      const existingTimer = cancelFallbackTimersRef.current.get(instanceId);
      if (existingTimer) clearTimeout(existingTimer);
      const timer = setTimeout(() => {
        cancelFallbackTimersRef.current.delete(instanceId);
        setCancellingInstances(prev => {
          if (!prev.has(instanceId)) return prev;
          const next = new Set(prev);
          next.delete(instanceId);
          return next;
        });
        showCancelToast(
          `Cancel did not take effect within ${
            CANCEL_FALLBACK_TIMEOUT_MS / 1000
          }s — please retry.`,
        );
      }, CANCEL_FALLBACK_TIMEOUT_MS);
      cancelFallbackTimersRef.current.set(instanceId, timer);

      try {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            taskId,
          )}/cancel-run?workflow_id=${encodeURIComponent(workflowId)}`,
          { method: 'POST' },
        );
        if (!res.ok) throw new Error(`Cancel failed (${res.status})`);
        // Cancel endpoint flips Mongo to `cancelled` synchronously; refresh
        // so the row transitions without waiting for the next background poll.
        fetchRunGroups();
      } catch (e) {
        console.warn('Cancel instance failed', e);
        const msg = e instanceof Error ? e.message : 'Cancel failed';
        showCancelToast(msg);
        // Only clear the "Cancelling…" overlay on error. Success case clears
        // it in the effect below once Mongo reports a terminal state.
        setCancellingInstances(prev => {
          if (!prev.has(instanceId)) return prev;
          const next = new Set(prev);
          next.delete(instanceId);
          return next;
        });
        const t = cancelFallbackTimersRef.current.get(instanceId);
        if (t) {
          clearTimeout(t);
          cancelFallbackTimersRef.current.delete(instanceId);
        }
      }
    },
    [taskId, fetchRunGroups, showCancelToast],
  );

  // Drop `cancelling` markers once the row leaves `running` in Mongo — avoids the status column flickering back to "In Progress".
  useEffect(() => {
    setCancellingInstances(prev => {
      if (prev.size === 0) return prev;
      let changed = false;
      const next = new Set(prev);
      for (const inst of flatInstances) {
        const id = String(inst.instance_id);
        if (prev.has(id) && inst.status !== 'running') {
          next.delete(id);
          changed = true;
          const t = cancelFallbackTimersRef.current.get(id);
          if (t) {
            clearTimeout(t);
            cancelFallbackTimersRef.current.delete(id);
          }
        }
      }
      return changed ? next : prev;
    });
  }, [flatInstances]);

  // Lazy-fetch the full instance doc on expand; refetched per expand so
  // a still-running instance refreshes its trajectory.
  useEffect(() => {
    if (!expandedInstance) return;
    const id = expandedInstance;
    let cancelled = false;
    setFullInstanceLoading(prev => {
      if (prev.has(id)) return prev;
      const next = new Set(prev);
      next.add(id);
      return next;
    });
    setFullInstanceErrors(prev => {
      if (!prev.has(id)) return prev;
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
    apiFetch(
      `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
        taskId,
      )}/instances/${encodeURIComponent(id)}`,
    )
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(full => {
        if (cancelled || !full) return;
        setFullInstances(prev => {
          const next = new Map(prev);
          next.set(id, full as Record<string, unknown>);
          return next;
        });
        setFullInstanceLoading(prev => {
          if (!prev.has(id)) return prev;
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      })
      .catch(e => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : 'Failed to load';
        setFullInstanceLoading(prev => {
          if (!prev.has(id)) return prev;
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
        setFullInstanceErrors(prev => {
          const next = new Map(prev);
          next.set(id, msg);
          return next;
        });
      });
    return () => {
      cancelled = true;
      // Clear loading for the cancelled fetch — neither .then nor .catch
      // runs after cancel, so the entry would otherwise leak.
      setFullInstanceLoading(prev => {
        if (!prev.has(id)) return prev;
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    };
  }, [expandedInstance, taskId, fullInstanceRetryTick]);

  const retryFullInstance = useCallback(() => {
    setFullInstanceRetryTick(t => t + 1);
  }, []);

  const anyActive = runGroups.some(g => activeCount(g.counts) > 0);
  const shouldPoll = anyActive;

  const handleStartedRuns = useCallback(() => {
    void fetchRunGroups(true);
  }, [fetchRunGroups]);

  // Pass@K summary fetched separately so it survives pagination.
  const latestRunGroupId = runGroups[0]?.run_group_id ?? null;
  const [runGroupSummary, setRunGroupSummary] =
    useState<RunGroupSummary | null>(null);
  useEffect(() => {
    if (!latestRunGroupId) {
      setRunGroupSummary(null);
      return;
    }
    let cancelled = false;
    const url = `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
      taskId,
    )}/run-groups/${encodeURIComponent(latestRunGroupId)}`;

    const fetchOnce = async () => {
      try {
        const resp = await apiFetch(url);
        if (!resp.ok) return false;
        const body = (await resp.json()) as RunGroupSummary;
        if (!cancelled) setRunGroupSummary(body);
        return body.running > 0;
      } catch {
        return false;
      }
    };

    let timer: ReturnType<typeof setTimeout> | null = null;
    const loop = async () => {
      const stillRunning = await fetchOnce();
      if (cancelled) return;
      if (stillRunning) {
        timer = setTimeout(loop, POLL_INTERVAL_RUNNING_MS);
      }
    };
    loop();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [latestRunGroupId, taskId, flatInstances.length]);

  useEffect(() => {
    if (!shouldPoll) return;
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout>;
    const poll = async () => {
      if (typeof document === 'undefined' || !document.hidden) {
        await fetchRunGroups();
      }
      if (!cancelled) timeoutId = setTimeout(poll, POLL_INTERVAL_RUNNING_MS);
    };
    timeoutId = setTimeout(poll, POLL_INTERVAL_RUNNING_MS);
    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [shouldPoll, fetchRunGroups]);

  return (
    <div className="p-8 pb-16 flex flex-col h-full overflow-y-auto">
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors mb-6"
      >
        <ArrowLeft size={14} />
        Back
      </button>

      {loading && (
        <p className="text-sm text-[var(--muted-foreground)]">Loading...</p>
      )}
      {fetchError && <p className="text-sm text-red-500">{fetchError}</p>}

      {task && !loading && (
        <>
          <div className="mb-6">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-semibold font-mono">{taskId}</h1>
              <span className="text-sm text-[var(--muted-foreground)]">
                v{String(task.version)}
              </span>
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-[var(--secondary)] text-[var(--foreground)]">
                task
              </span>
            </div>
            {!!task.created_at_utc && (
              <p className="text-sm text-[var(--muted-foreground)] mt-1">
                Created:{' '}
                {formatCellValue('created_at_utc', task.created_at_utc)}
              </p>
            )}
          </div>


          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <h3 className={SECTION_HEADER_CLASS}>
                Task Workflow ({steps.length})
              </h3>
            </div>
            <StepsPipeline steps={steps} />
          </div>

          <TriggersGraph steps={steps} />

          {/* Start Runs — `key={taskId}` so seed/config state doesn't bleed
              across task navigation (sessionStorage config is global but the
              transient upload + active-group state is per-task). */}
          <div className="mb-6">
            <StartRunsPanel
              key={String(task?.id ?? '')}
              taskId={String(task?.id ?? '')}
              taskVersion={Number(task?.version ?? 1)}
              taskSteps={steps}
              taskProjectId={
                typeof task?.project_id === 'string'
                  ? (task.project_id as string)
                  : undefined
              }
              onStarted={handleStartedRuns}
              onCompleted={fetchRunGroups}
            />
          </div>

          {/* Rollouts — historical instances across all run groups */}
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <h3 className={SECTION_HEADER_CLASS}>Rollouts</h3>
              <label className="sr-only" htmlFor="rollouts-version-filter">
                Filter rollouts by task version
              </label>
              <select
                id="rollouts-version-filter"
                value={versionFilter}
                onChange={e => setVersionFilter(e.target.value)}
                aria-label="Filter rollouts by task version"
                className="rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-0.5 text-xs text-[var(--muted-foreground)] focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
              >
                <option value="all">All versions</option>
                {task &&
                  Number(task.version) > 0 &&
                  Array.from({ length: Number(task.version) }, (_, i) => i + 1)
                    .reverse()
                    .map(v => (
                      <option key={v} value={String(v)}>
                        v{v}
                        {v === Number(task.version) ? ' (latest)' : ''}
                      </option>
                    ))}
              </select>
              <button
                onClick={handleRefreshInstances}
                disabled={instancesLoading}
                aria-label="Refresh rollouts"
                className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors disabled:opacity-50"
                title="Refresh rollouts"
              >
                <RefreshCw
                  size={12}
                  aria-hidden
                  className={instancesLoading ? 'animate-spin' : ''}
                />
              </button>
            </div>
            {instancesLoading ? (
              <p className="text-sm text-[var(--muted-foreground)]">
                Loading rollouts…
              </p>
            ) : instancesError ? (
              <p className="text-sm text-red-500">{instancesError}</p>
            ) : initialLoadDone.current && displayRunGroups.length === 0 ? (
              <div className="rounded-lg border border-dashed border-[var(--border)] px-4 py-8 text-center">
                <p className="text-sm font-medium text-[var(--foreground)]">
                  No runs yet
                </p>
                <p className="text-xs text-[var(--muted-foreground)] mt-1">
                  Use the <span className="font-medium">Start Runs</span> panel
                  above to launch a batch — runs will appear here as they
                  progress.
                </p>
              </div>
            ) : (
              <div className="rounded-lg border border-[var(--border)] overflow-hidden">
                <table className="w-full text-sm table-fixed">
                  {/* Columns sum to 100. Trajectory Manifest Download
                   * column trims a few % from Status + Score; identifier
                   * columns kept wide enough to avoid truncating IDs. */}
                  <colgroup>
                    <col className="w-[18%]" />
                    <col className="w-[13%]" />
                    <col className="w-[10%]" />
                    <col className="w-[5%]" />
                    <col className="w-[14%]" />
                    <col className="w-[7%]" />
                    <col className="w-[10%]" />
                    <col className="w-[9%]" />
                    <col className="w-[10%]" />
                    <col className="w-[4%]" />
                  </colgroup>
                  <thead>
                    <tr className="bg-[var(--secondary)] text-left">
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Instance ID
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Model
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Run Group
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Version
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Start Time
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Duration
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Status
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Score
                      </th>
                      <th className="px-3 py-2 text-xs font-semibold text-[var(--muted-foreground)]">
                        Downloads
                      </th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {displayRunGroups.flatMap(group => {
                      const first = group.instances[0];
                      if (group.instances.length === 1 && first) {
                        const id = String(first.instance_id);
                        return [
                          <InstanceRow
                            key={id}
                            taskId={taskId}
                            inst={first}
                            nested={false}
                            expandedInstance={expandedInstance}
                            setExpandedInstance={setExpandedInstance}
                            cancellingInstances={cancellingInstances}
                            handleCancelInstance={handleCancelInstance}
                            rubricsCriteria={rubricsCriteria}
                            rubricsAggregator={
                              verifierStep?.score_aggregator as
                                | string
                                | undefined
                            }
                            fullInstance={fullInstances.get(id)}
                            isFullInstanceLoading={fullInstanceLoading.has(id)}
                            fullInstanceError={
                              fullInstanceErrors.get(id) ?? null
                            }
                            onRetryFullInstance={retryFullInstance}
                            scoreColorMode={scoreColorMode}
                            taskSteps={taskSteps}
                            highlightInstanceId={highlightInstanceId}
                          />,
                        ];
                      }
                      const isActive = activeCount(group.counts) > 0;
                      const override = groupOverrides.get(group.run_group_id);
                      // childOpen wins over a stale `false` override.
                      const childOpen =
                        expandedInstance != null &&
                        group.instances.some(
                          i => String(i.instance_id) === expandedInstance,
                        );
                      const expanded = childOpen || (override ?? isActive);
                      const headerRow = (
                        <GroupHeaderRow
                          key={`group-${group.run_group_id}`}
                          taskId={taskId}
                          group={group}
                          expanded={expanded}
                          childOpen={childOpen}
                          setGroupOverrides={setGroupOverrides}
                          setExpandedInstance={setExpandedInstance}
                        />
                      );
                      if (!expanded) return [headerRow];
                      return [
                        headerRow,
                        <tr
                          key={`funnel-${group.run_group_id}`}
                          className="border-t border-[var(--border)] bg-[var(--secondary)]"
                        >
                          <td colSpan={10} className="px-4 pb-3 pt-1">
                            <RunGroupStepFunnel
                              steps={taskSteps}
                              stepCounts={group.step_counts ?? {}}
                              total={group.instances.length}
                            />
                          </td>
                        </tr>,
                        ...group.instances.map(inst => {
                          const id = String(inst.instance_id);
                          return (
                            <InstanceRow
                              key={id}
                              taskId={taskId}
                              inst={inst}
                              nested
                              expandedInstance={expandedInstance}
                              setExpandedInstance={setExpandedInstance}
                              cancellingInstances={cancellingInstances}
                              handleCancelInstance={handleCancelInstance}
                              rubricsCriteria={rubricsCriteria}
                              fullInstance={fullInstances.get(id)}
                              isFullInstanceLoading={fullInstanceLoading.has(
                                id,
                              )}
                              fullInstanceError={
                                fullInstanceErrors.get(id) ?? null
                              }
                              onRetryFullInstance={retryFullInstance}
                              scoreColorMode={scoreColorMode}
                              taskSteps={taskSteps}
                              rubricsAggregator={
                                verifierStep?.score_aggregator as
                                  | string
                                  | undefined
                              }
                              highlightInstanceId={highlightInstanceId}
                            />
                          );
                        }),
                      ];
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {!instancesLoading && (instancesHasMore || instancesPage > 0) && (
              <div className="flex items-center justify-between mt-2 text-xs text-[var(--muted-foreground)]">
                <div className="flex items-center gap-3">
                  <span>
                    Showing {instancesPage * runGroupsPageSize + 1}–
                    {instancesPage * runGroupsPageSize + runGroups.length} run
                    group{runGroups.length === 1 ? '' : 's'}
                    {instancesHasMore ? ' (more available)' : ''}
                  </span>
                  <label className="flex items-center gap-1">
                    <span className="sr-only">Run groups per page</span>
                    <select
                      value={runGroupsPageSize}
                      onChange={e =>
                        setRunGroupsPageSize(Number(e.target.value))
                      }
                      aria-label="Run groups per page"
                      className="rounded border border-[var(--border)] bg-[var(--background)] px-1.5 py-0.5 text-xs focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
                    >
                      {PAGE_SIZE_OPTIONS.map(n => (
                        <option key={n} value={n}>
                          {n} / page
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setInstancesPage(p => p - 1)}
                    disabled={instancesPage === 0}
                    aria-label="Previous page"
                    className="px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--accent)] disabled:opacity-40 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setInstancesPage(p => p + 1)}
                    disabled={!instancesHasMore}
                    aria-label="Next page"
                    className="px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--accent)] disabled:opacity-40 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
            {rubricsCriteria &&
              !instancesLoading &&
              flatInstances.length > 0 && (
                <div className="mt-4">
                  <h3 className={`${SECTION_HEADER_CLASS} mb-2`}>
                    Rubric Verifier Matrix
                  </h3>
                  <RubricMatrix
                    runs={
                      flatInstances.map(inst => ({
                        id: String(inst.instance_id),
                        name: String(inst.instance_id).split('-').slice(-1)[0],
                        status: String(inst.status ?? ''),
                        verificationResults: (
                          (inst.context as Record<string, unknown> | null)
                            ?.metadata as Record<string, unknown> | undefined
                        )?.verifications as VerificationResults | undefined,
                      })) as MatrixRun[]
                    }
                    rubrics={rubricsCriteria as unknown as RubricCriterion[]}
                  />
                </div>
              )}
          </div>

        </>
      )}

      {cancelErrorToast && (
        <div
          role="alert"
          aria-live="polite"
          className="fixed bottom-6 right-6 flex items-start gap-3 px-4 py-2 rounded-lg bg-red-500 text-white text-sm shadow-lg max-w-md"
        >
          <span className="flex-1">{cancelErrorToast}</span>
          <button
            type="button"
            onClick={() => {
              if (cancelToastTimerRef.current)
                clearTimeout(cancelToastTimerRef.current);
              setCancelErrorToast(null);
            }}
            aria-label="Dismiss notification"
            className="text-white/80 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-white/50 rounded"
          >
            <X size={14} aria-hidden />
          </button>
        </div>
      )}
    </div>
  );
}

/** Compact live progress banner for the latest run group, above the instances table so a long pass@k
 *  is visible at a glance. Color encodes terminal state; a pulsing dot signals auto-update. */
function RunGroupProgressBanner({ summary }: { summary: RunGroupSummary }) {
  const { run_group_id, total, completed, failed, running } = summary;
  const done = running === 0 && total > 0;
  const allPassed = done && failed === 0;

  const palette = allPassed
    ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-600'
    : failed > 0 && done
    ? 'border-red-500/30 bg-red-500/5 text-red-600'
    : running > 0
    ? 'border-blue-400/30 bg-blue-400/5 text-blue-500'
    : 'border-[var(--border)] bg-[var(--secondary)] text-[var(--muted-foreground)]';

  const shortId = run_group_id.slice(0, 12);
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <div className={`mb-3 rounded-lg border ${palette}`}>
      <div className="flex items-center gap-3 px-3 py-2 text-xs">
        {running > 0 && (
          <span
            className="inline-block w-2 h-2 rounded-full bg-current animate-pulse flex-shrink-0"
            aria-hidden
          />
        )}
        <span className="font-semibold">Latest run group</span>
        <span className="font-mono opacity-75">
          {shortId}
          {'\u2026'}
        </span>
        <span className="opacity-75">
          {completed}/{total} completed
          {failed > 0 && ` \u2022 ${failed} failed`}
          {running > 0
            ? ` \u2022 ${running} running`
            : done
            ? ' \u2022 done'
            : ''}
        </span>
        <span className="ml-auto opacity-60 font-mono">{pct}%</span>
      </div>
      {/* Progress bar */}
      <div className="h-1 w-full overflow-hidden bg-current/10">
        <div
          className="h-full bg-current transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
