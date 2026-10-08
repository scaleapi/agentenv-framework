import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Button, DropdownMenu, ScrollArea, Tabs } from '@radix-ui/themes';
import JSZip from 'jszip';
import { MCPEnvValidationEntry } from './mcp-env-validation-entry';
import { RunStepProgress } from './run-step-progress';
import { type AgentCard } from './advertised-agent-card';
import { StepAttemptFailures } from './step-attempt-failures';
import {
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  FileText,
  Image as ImageIcon,
  Loader2,
  XCircle,
} from 'lucide-react';
import { BACKEND_URL, apiFetch, objectContentUrl } from './shared';
import { ConversationsPanel } from './conversations-panel';
import { PeerQnAPanel } from './peer-qna-panel';
import {
  ServerConfigPanel,
  type ServerConfigChange,
  type ServerConfigSkip,
} from './server-config-panel';
import { TriggersPanel, TriggerTurnStrip } from './triggers-panel';
import { parseTriggerRuntime } from '../lib/parse-trigger-runtime';
import { indexAuthoredTriggers } from '../lib/parse-triggers';
import {
  RubricGradingResults,
  type RubricCriterion,
  type VerificationResults,
} from './rubric-grading-results';
import {
  classifyVerifier,
  renderJudgeOutputFormatVerifier,
} from '../lib/verifier-visualizers';
import {
  buildBarDownloads,
  shouldCollapseBarDownloads,
  type BarDownload,
} from '../lib/bar-downloads';
import { collectedZipFiles, isAbsoluteArtifactPath } from '../lib/collected-zip-files';

function renderRubricVerifierPanel(
  verifierId: string,
  verifier: Record<string, unknown>,
  rubricsCriteria: Record<string, unknown>[],
  rubricsAggregator?: string,
): React.ReactNode {
  const fromRegistry = renderJudgeOutputFormatVerifier({
    verifierId,
    verifier,
    rubricsCriteria,
    rubricsAggregator,
  });
  if (fromRegistry != null) return fromRegistry;
  return (
    <RubricGradingResults
      verifierId={verifierId}
      verificationResults={
        { [verifierId]: verifier } as unknown as VerificationResults
      }
      rubrics={rubricsCriteria as unknown as RubricCriterion[]}
      aggregator={rubricsAggregator}
    />
  );
}

import { TrajectoryViewer } from './trajectory-viewer';
import {
  type ParsedTrajectory,
  type OtelSpan,
  parseOtelTrajectory,
} from '../lib/parse-trajectory';
import {
  type TrajectoryUrlKeys,
  perTurnTrajectoryUrls,
  trajectoryUrl,
} from '../lib/trajectory-url';

interface PromptResponseData extends TrajectoryUrlKeys {
  prompt_id: string;
  response: string;
  prompt_text?: string;
  // A null entry means "identical to prompt_text" (the initial prompt is stored once).
  source_agent_per_turn_prompt_parts?: Array<Array<{
    kind: string;
    text?: string;
    file?: { name?: string; uri?: string };
    data?: unknown;
  }> | null>;
  model?: string;
  // step_id is the human-readable step name (e.g. "run-solver"). Older instances may have only prompt_id.
  step_id?: string;
}

interface TrajectoryState {
  // Lazy-load: trajectories start 'idle' so a multi-step task doesn't fire N parallel fetches on mount.
  status: 'idle' | 'loading' | 'loaded' | 'error';
  objectUrl: string;
  // The prompt the agent received for this step. Multi-step tasks each have a distinct prompt, rendered above their trajectory section.
  promptText?: string;
  trajectory?: ParsedTrajectory;
  error?: string;
}

export interface TaskStepRef {
  id: string;
  prompt_id?: string | null;
  type?: string;
  target?: string;
  base_path?: string;
  artifact_paths?: string[];
  // Trigger-registration step fields, threaded through for the badge
  // popovers' authored-config index.
  env_id?: string | null;
  agent_name?: string | null;
  triggers?: unknown;
}

export function TaskInstanceViewer({
  instance,
  taskId: taskIdProp,
  rubricsCriteria,
  rubricsAggregator,
  taskSteps,
}: {
  instance: Record<string, unknown>;
  // List-derived instances may omit task_id; callers that know it (runner/detail
  // pages) pass it so downstream calls can't send task_id=undefined.
  taskId?: string;
  rubricsCriteria?: Record<string, unknown>[];
  /** `score_aggregator` from the rubrics_verifier step ('all_pass' | 'any_pass' | 'weighted_average').
   *  Suppresses the redundant Score badge when it's implied by the pass count. Undefined = all_pass (default). */
  rubricsAggregator?: string;
  // Ordered steps from the task definition: label trajectories by step.id (not "Prompt N") and sort them
  // in pipeline order rather than persisted order (which isn't guaranteed chronological).
  taskSteps?: TaskStepRef[];
}) {
  const instanceId = instance.instance_id as string;
  const taskId = taskIdProp ?? (instance.task_id as string);
  const [trajectories, setTrajectories] = useState<
    {
      label: string;
      state: TrajectoryState;
      // Join keys for the per-turn trigger strip. stepId is the RUN-TIME step id (the ledger's key), not
      // the resolved label id, which can diverge. Undefined for combined-only entries (no strip).
      stepId?: string;
      turnIndex?: number;
      // The user message authored after this turn (next turn's prompt) —
      // the strip quotes it as the user-sim / trigger-injected reply.
      nextTurnPromptText?: string;
    }[]
  >([]);
  const [copiedContext, setCopiedContext] = useState(false);
  const [selectedOverviewFile, setSelectedOverviewFile] = useState<
    string | null
  >(null);
  // Per-deployment A2A cards for the Agent Card tab. The registered agent doc carries no card; it lives in
  // context.deployed_agents[].a2a_card, which the list endpoint strips — fall back to the single-instance GET.
  const deployedAgentsForCard = ((instance.context as Record<string, unknown> | null)
    ?.deployed_agents ?? []) as Record<string, unknown>[];
  const agentCardSig = deployedAgentsForCard
    .map(a => `${String(a.agent_name ?? '')}:${a.a2a_card ? 1 : 0}`)
    .join('|');
  const [agentCards, setAgentCards] = useState<
    { name: string; card: AgentCard }[]
  >([]);
  useEffect(() => {
    const pick = (agents: Record<string, unknown>[]) =>
      agents
        .filter(a => a.a2a_card)
        .map(a => ({
          name: String(a.agent_name ?? 'agent'),
          card: a.a2a_card as AgentCard,
        }));
    const inline = pick(deployedAgentsForCard);
    if (inline.length > 0) {
      setAgentCards(inline);
      return;
    }
    if (deployedAgentsForCard.length === 0 || !taskId || !instanceId) {
      setAgentCards([]);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            taskId,
          )}/instances/${encodeURIComponent(instanceId)}`,
        );
        if (!res.ok) return;
        const body = await res.json();
        const agents = (body?.context?.deployed_agents ??
          []) as Record<string, unknown>[];
        if (!cancelled) setAgentCards(pick(agents));
      } catch {
        /* no card panel on fetch failure */
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId, instanceId, agentCardSig]);


  // Build trajectory entries (one per prompt_response with a trajectory URL), all 'idle' — no eager fetch (a
  // multi-step task can have 5+). Auto-load only when there's exactly one. Re-run when the prompt_responses
  // array reference changes (task-detail swaps a thin row for the full doc), keyed on a content signature
  // since the parent doesn't memoize.
  const promptResponsesArr = (
    instance.context as Record<string, unknown> | null
  )?.prompt_responses as TrajectoryUrlKeys[] | undefined;
  const promptResponsesSignature = useMemo(
    () =>
      (promptResponsesArr ?? [])
        .map(pr => {
          // Include per-turn URIs in the signature so the effect re-fires when new turns land between polls.
          const perTurn = (perTurnTrajectoryUrls(pr) ?? []).join(',');
          return `${trajectoryUrl(pr) ?? ''}#${perTurn}`;
        })
        .join('|'),
    [promptResponsesArr],
  );
  useEffect(() => {
    const context = instance.context as Record<string, unknown> | null;
    if (!context) {
      setTrajectories([]);
      return;
    }

    const promptResponses = (context.prompt_responses ??
      []) as PromptResponseData[];

    // Fan-out: each PromptResponse becomes 1+ entries — one per non-null turn URI for multi-turn, else one.
    type FlatPR = PromptResponseData & {
      _entryObjectUrl: string;
      _turnIndex?: number;
      _totalTurns?: number;
    };
    const flat: FlatPR[] = [];
    for (const pr of promptResponses) {
      const perTurn = perTurnTrajectoryUrls(pr);
      const url = trajectoryUrl(pr);
      if (perTurn && perTurn.length > 0) {
        perTurn.forEach((uri, turnIdx) => {
          if (!uri) return; // skip turns whose trajectory upload failed
          flat.push({
            ...pr,
            _entryObjectUrl: uri,
            _turnIndex: turnIdx,
            _totalTurns: perTurn.length,
          });
        });
      } else if (url) {
        flat.push({ ...pr, _entryObjectUrl: url });
      }
    }

    // Build prompt_id → step.id / step-index lookups so anonymous "Prompt N" labels become step names and
    // trajectories sort in pipeline order rather than persisted order.
    const stepIdByPrompt: Record<string, string> = {};
    const stepIdxByPrompt: Record<string, number> = {};
    (taskSteps ?? []).forEach((s, idx) => {
      if (s.prompt_id && s.id) {
        stepIdByPrompt[s.prompt_id] = s.id;
        stepIdxByPrompt[s.prompt_id] = idx;
      }
    });

    const ordered = [...flat].sort((a, b) => {
      const ai = stepIdxByPrompt[a.prompt_id] ?? Number.MAX_SAFE_INTEGER;
      const bi = stepIdxByPrompt[b.prompt_id] ?? Number.MAX_SAFE_INTEGER;
      if (ai !== bi) return ai - bi;
      // Same step → preserve turn order
      return (a._turnIndex ?? 0) - (b._turnIndex ?? 0);
    });

    setTrajectories(prev => {
      // Index prior loaded/loading state by objectUrl so a re-fire (e.g. taskSteps arriving late) doesn't undo
      // in-progress work — only label/order is recomputed, payloads preserved.
      const prevByUri = new Map<string, TrajectoryState>();
      for (const t of prev) prevByUri.set(t.state.objectUrl, t.state);

      return ordered.map((pr, i) => {
        const resolvedStepId = stepIdByPrompt[pr.prompt_id] ?? pr.step_id;
        const carryover = prevByUri.get(pr._entryObjectUrl);
        const initialStatus: TrajectoryState['status'] =
          carryover?.status ?? (ordered.length === 1 ? 'loading' : 'idle');
        const turnSuffix =
          pr._turnIndex !== undefined && (pr._totalTurns ?? 0) > 1
            ? ` · Turn ${pr._turnIndex + 1}/${pr._totalTurns}`
            : '';
        const stepLabel = resolvedStepId ?? `Prompt ${i + 1}`;
        const modelSuffix = pr.model ? ` (${pr.model})` : '';
        // Multi-turn: the per-turn prompt is the text sent that turn (initial on turn 1, reply on turn 2+);
        // fall back to prompt_text (single-turn) — a null turn-0 entry also means prompt_text. Text parts only.
        const perTurnParts =
          pr._turnIndex !== undefined
            ? pr.source_agent_per_turn_prompt_parts?.[pr._turnIndex]
            : undefined;
        const perTurnPromptText = perTurnParts
          ? perTurnParts
              .filter(p => p.kind === 'text' && typeof p.text === 'string')
              .map(p => p.text!)
              .join('\n')
          : undefined;
        const promptText =
          perTurnPromptText !== undefined
            ? perTurnPromptText
            : pr._turnIndex === undefined || pr._turnIndex === 0
            ? pr.prompt_text
            : undefined;
        // The reply authored AFTER this turn (user-sim or trigger-injected)
        // is the NEXT turn's prompt — surfaced by the trigger strip.
        const nextTurnParts =
          pr._turnIndex !== undefined
            ? pr.source_agent_per_turn_prompt_parts?.[pr._turnIndex + 1]
            : undefined;
        const nextTurnPromptText = nextTurnParts
          ? nextTurnParts
              .filter(p => p.kind === 'text' && typeof p.text === 'string')
              .map(p => p.text!)
              .join('\n')
          : undefined;
        return {
          label:
            ordered.length > 1
              ? `${stepLabel}${turnSuffix}${modelSuffix}`
              : `Trajectory${turnSuffix}${modelSuffix}`,
          stepId: pr.step_id ?? resolvedStepId ?? undefined,
          turnIndex: pr._turnIndex,
          nextTurnPromptText,
          state: {
            status: initialStatus,
            objectUrl: pr._entryObjectUrl,
            promptText,
            trajectory: carryover?.trajectory,
            error: carryover?.error,
          },
        };
      });
    });

    // Auto-fetch only when there's exactly one (single-step UX). `cancelled` drops a parse that lands
    // after the effect re-fired, so it can't overwrite the newer fetch.
    let cancelled = false;
    if (ordered.length === 1) {
      const pr = ordered[0]!;
      fetchTrajectory(pr._entryObjectUrl, pr.model)
        .then(parsed => {
          if (cancelled) return;
          setTrajectories(prev =>
            prev.map(t => ({
              ...t,
              state: { ...t.state, status: 'loaded', trajectory: parsed },
            })),
          );
        })
        .catch(err => {
          if (cancelled) return;
          setTrajectories(prev =>
            prev.map(t => ({
              ...t,
              state: {
                ...t.state,
                status: 'error',
                error: err instanceof Error ? err.message : 'Failed to load',
              },
            })),
          );
        });
    }
    return () => {
      cancelled = true;
    };
  }, [instanceId, taskSteps, promptResponsesSignature]);

  const loadTrajectoryAt = useCallback(
    (index: number) => {
      const target = trajectories[index];
      if (!target) return;
      // Commit by objectUrl, not array index: `trajectories` can be rebuilt/reordered while a load is in flight.
      const objectUrl = target.state.objectUrl;
      setTrajectories(prev =>
        prev.map(t =>
          t.state.objectUrl === objectUrl
            ? { ...t, state: { ...t.state, status: 'loading' } }
            : t,
        ),
      );
      fetchTrajectory(objectUrl)
        .then(parsed => {
          setTrajectories(prev =>
            prev.map(t =>
              t.state.objectUrl === objectUrl
                ? {
                    ...t,
                    state: { ...t.state, status: 'loaded', trajectory: parsed },
                  }
                : t,
            ),
          );
        })
        .catch(err => {
          setTrajectories(prev =>
            prev.map(t =>
              t.state.objectUrl === objectUrl
                ? {
                    ...t,
                    state: {
                      ...t.state,
                      status: 'error',
                      error:
                        err instanceof Error ? err.message : 'Failed to load',
                    },
                  }
                : t,
            ),
          );
        });
    },
    [trajectories],
  );

  const context = instance.context as Record<string, unknown> | null;
  const promptResponses = context
    ? ((context.prompt_responses ?? []) as PromptResponseData[])
    : [];
  const hasTrajectories = promptResponses.some(pr => trajectoryUrl(pr));
  const deployedEnvs = context
    ? ((context.deployed_envs ?? []) as Record<string, unknown>[])
    : [];
  const metadata = context?.metadata as Record<string, unknown> | undefined;
  const serverConfigChanges = (metadata?.server_config_changes ??
    []) as ServerConfigChange[];
  const serverConfigFailures = (
    (metadata?.failed_steps ?? []) as Array<Record<string, unknown>>
  )
    .filter(f => f.step_type === 'apply_server_config')
    .map(f => ({
      error: String(f.error ?? ''),
      error_type: f.error_type as string | undefined,
      step_id: f.step_id as string | undefined,
    }));
  const serverConfigSkipped = (metadata?.server_config_skipped ??
    []) as ServerConfigSkip[];
  const hasServerConfig =
    serverConfigChanges.length > 0 ||
    serverConfigFailures.length > 0 ||
    serverConfigSkipped.length > 0;
  // Runtime trigger ledger: null when the run recorded no trigger
  // metadata, which also hides the tab.
  const triggerRuntime = useMemo(
    () => parseTriggerRuntime(metadata),
    [metadata],
  );
  // Authored trigger configs for the badge popovers. Empty when the surface passes slim step refs — popovers then show runtime history only.
  const authoredTriggers = useMemo(
    () => indexAuthoredTriggers(taskSteps),
    [taskSteps],
  );
  // A PeerAgentsTaskStep records the routing table on context.metadata.agent_peerings.
  // Its presence gates the Peer Q&A tab (peer_send_message exchanges pulled from the trajectory).
  const hasPeerAgents =
    Array.isArray(metadata?.agent_peerings) &&
    metadata.agent_peerings.length > 0;
  // The user-sim / HITL flows record an A2A transcript under context.metadata.a2a_conversations; its presence gates the Conversation tab.
  const hasConversations =
    !!metadata?.a2a_conversations &&
    Object.keys(metadata.a2a_conversations as Record<string, unknown>).length >
      0;
  const verifications = metadata?.verifications as
    | Record<
        string,
        {
          results: {
            id: string;
            score: number;
            result: boolean;
            message: string;
            title?: string;
            justification?: string;
          }[];
          score: number;
          format?: string;
        }
      >
    | undefined;
  const hasVerifications =
    verifications && Object.keys(verifications).length > 0;

  // `collect_artifacts` writes a structured map context.metadata.collected_artifacts[step_id] = { artifacts:
  // { filename: object_url }, file_artifact_universe }, plus a legacy flat context.metadata.artifacts mirror.
  // Prefer the structured map; fall back to the flat mirror for older instances.
  const collectedArtifactsByStep = (metadata?.collected_artifacts ??
    null) as Record<string, { artifacts?: Record<string, string> }> | null;

  // The on-sandbox source path is implicit: <base_path>/<filename> (base_path from the collect step's config, default /app/artifact).
  const collectArtifactsSteps = (taskSteps ?? []).filter(
    s => s.type === 'collect_artifacts',
  );
  const basePathForStep = (stepId: string | null): string => {
    const step =
      (stepId ? collectArtifactsSteps.find(s => s.id === stepId) : undefined) ??
      collectArtifactsSteps[0];
    return step?.base_path ?? '/app/artifact';
  };

  // One render section per collect step that produced files (structured map),
  // else a single legacy section from the flat mirror.
  const collectedArtifactSections: {
    stepId: string | null;
    artifacts: Record<string, string>;
    basePath: string;
  }[] = collectedArtifactsByStep
    ? Object.entries(collectedArtifactsByStep)
        .map(([stepId, v]) => ({
          stepId,
          artifacts: v?.artifacts ?? {},
          basePath: basePathForStep(stepId),
        }))
        .filter(s => Object.keys(s.artifacts).length > 0)
    : (() => {
        const flat =
          (metadata?.artifacts as Record<string, string> | undefined) ?? {};
        return Object.keys(flat).length > 0
          ? [{ stepId: null, artifacts: flat, basePath: basePathForStep(null) }]
          : [];
      })();

  // Flat merged map (filename -> object_url) for inline content links elsewhere
  // (verifier-card path chips, reviewer-overview lookup).
  const collectedArtifacts: Record<string, string> | null =
    collectedArtifactSections.length > 0
      ? collectedArtifactSections.reduce<Record<string, string>>(
          (acc, s) => Object.assign(acc, s.artifacts),
          {},
        )
      : null;
  const hasCollectedArtifacts = collectedArtifactSections.length > 0;
  const collectedArtifactsCount = collectedArtifactSections.reduce(
    (n, s) => n + Object.keys(s.artifacts).length,
    0,
  );

  // A "reviewer overview" HTML page (plain-English functionality, API detail collapsed). Rendered in its
  // own tab so a non-technical reviewer reads it in-app instead of downloading a file.
  const reviewerOverviewKey = collectedArtifacts
    ? Object.keys(collectedArtifacts).find(k =>
        k.endsWith('reviewer_overview.html'),
      )
    : undefined;
  const reviewerOverviewUrl =
    reviewerOverviewKey && collectedArtifacts
      ? collectedArtifacts[reviewerOverviewKey]
      : null;

  // Bucket each verifier output by stamped `format` or legacy shape
  // so the verifier tab can show aggregate/sandbox/rubric sections together.
  const verifierEntries: Array<[string, Record<string, unknown>]> =
    verifications
      ? (Object.entries(verifications) as Array<
          [string, Record<string, unknown>]
        >)
      : [];
  const aggregateVerifiers = verifierEntries.filter(
    ([, v]) => classifyVerifier(v) === 'aggregate',
  );
  const sandboxVerifiers = verifierEntries.filter(
    ([, v]) => classifyVerifier(v) === 'sandbox',
  );
  const rubricVerifiers = verifierEntries.filter(
    ([, v]) => classifyVerifier(v) === 'rubric',
  );
  const hasAggregateOrSandbox =
    aggregateVerifiers.length > 0 || sandboxVerifiers.length > 0;

  const handleDownloadTrajectory = useCallback((objectUrl: string) => {
    window.open(objectContentUrl(objectUrl), '_blank');
  }, []);


  // One list for both renderings, so the inline row and the collapsed dropdown
  // always show the same entries in the same order.
  const barDownloads = useMemo(
    () => buildBarDownloads(promptResponses),
    [promptResponses],
  );
  const collapseBarDownloads = shouldCollapseBarDownloads(barDownloads.length);
  const runBarDownload = useCallback(
    (download: BarDownload) => {
      void handleDownloadTrajectory(download.objectUrl);
    },
    [handleDownloadTrajectory],
  );

  // Initial tab by status: for terminal runs lead to the most diagnostic view — verifier verdict
  // (completed) or run context (failed). Running keeps the default.
  const instanceStatus = (instance.status as string | undefined) ?? '';
  const isTerminalCompleted = instanceStatus === 'completed';
  const isTerminalFailed =
    instanceStatus === 'failed' ||
    instanceStatus === 'cancelled' ||
    instanceStatus === 'timed_out';
  const isRunning = instanceStatus === 'running';
  // Reviewer overview: the collected copy in the object store via /objects/content. Multi-file viewer
  // scoped to /app/reviewer_overview/ (legacy top-level reviewer_overview.html as fallback); appears once
  // the run's artifacts are collected (no live-from-sandbox route in the standalone explorer).
  const collectedFiles = collectedArtifacts
    ? Object.keys(collectedArtifacts)
        .filter(
          k =>
            k.startsWith('reviewer_overview/') ||
            k.endsWith('reviewer_overview.html'),
        )
        .sort()
    : [];
  const overviewFileList = collectedFiles;
  const activeOverviewFile =
    selectedOverviewFile && overviewFileList.includes(selectedOverviewFile)
      ? selectedOverviewFile
      : overviewFileList.find(f => f.endsWith('index.html')) ??
        overviewFileList.find(f => f.endsWith('reviewer_overview.html')) ??
        overviewFileList.find(f => f.endsWith('.html')) ??
        overviewFileList[0] ??
        null;
  const activeCollectedUri =
    activeOverviewFile && collectedArtifacts
      ? collectedArtifacts[activeOverviewFile]
      : undefined;
  const reviewerOverviewSrc = activeCollectedUri
    ? objectContentUrl(activeCollectedUri)
    : reviewerOverviewUrl
    ? objectContentUrl(reviewerOverviewUrl)
    : null;
  const showReviewerOverview =
    collectedFiles.length > 0 || !!reviewerOverviewUrl;
  const liveDefaultValue = hasTrajectories
    ? 'trajectory'
    : hasVerifications
    ? 'verifier'
    : 'context';
  // A failed run defaults to the context tab.
  const defaultTabValue = isTerminalCompleted
    ? hasVerifications
      ? 'verifier'
      : liveDefaultValue
    : isTerminalFailed
    ? 'context'
    : liveDefaultValue;
  const instanceError =
    typeof instance.error === 'string' && instance.error.trim()
      ? instance.error
      : null;
  // Agent-reported errors from context.prompt_responses — only for prompt_agent steps, and only the one
  // persisted attempt (distinct from the step-attempt ledger below, which covers every step/attempt).
  const promptAgentPerStepErrors: {
    stepId: string | null;
    model: string | null;
    message: string;
  }[] = (() => {
    const ctx = instance.context as Record<string, unknown> | null | undefined;
    const prs = ctx?.prompt_responses;
    if (!Array.isArray(prs)) return [];
    return prs
      .map(pr => {
        const p = pr as Record<string, unknown>;
        const message =
          typeof p.error_message === 'string' ? p.error_message.trim() : '';
        if (!message) return null;
        return {
          stepId: typeof p.prompt_id === 'string' ? p.prompt_id : null,
          model: typeof p.model === 'string' ? p.model : null,
          message,
        };
      })
      .filter(
        (
          x,
        ): x is {
          stepId: string | null;
          model: string | null;
          message: string;
        } => x !== null,
      );
  })();
  const showFailureBanner =
    isTerminalFailed && (instanceError || promptAgentPerStepErrors.length > 0);
  return (
    <div className="border border-[var(--border)] rounded-lg bg-[var(--background)] overflow-hidden">
      <RunStepProgress
        key={instanceId}
        steps={taskSteps ?? []}
        completedSteps={
          (instance.completed_steps as
            | { step_id: string; status?: string }[]
            | undefined) ?? []
        }
        totalSteps={instance.total_steps as number | undefined}
        status={instanceStatus}
        taskId={taskId}
        instanceId={instanceId}
      />
      {showFailureBanner ? (
        <div
          role="alert"
          className="border-b border-red-500/30 bg-red-500/5 px-4 py-3 text-sm"
        >
          <div className="flex items-start gap-2 text-red-500">
            <XCircle size={16} className="mt-0.5 flex-shrink-0" aria-hidden />
            <div className="font-medium">
              {instanceStatus === 'cancelled'
                ? 'Cancelled'
                : instanceStatus === 'timed_out'
                ? 'Timed out'
                : 'Failed'}
            </div>
          </div>
          {instanceError && (
            <pre className="mt-2 ml-6 whitespace-pre-wrap break-words font-mono text-xs text-[var(--foreground)]">
              {instanceError}
            </pre>
          )}
          {promptAgentPerStepErrors.length > 0 && (
            <div className="mt-2 ml-6 space-y-2">
              {promptAgentPerStepErrors.map((err, i) => (
                <div key={i} className="text-xs">
                  <div className="text-[var(--muted-foreground)]">
                    Step{' '}
                    <span className="font-mono">{err.stepId ?? 'unknown'}</span>
                    {err.model && (
                      <>
                        {' · '}
                        <span className="font-mono">{err.model}</span>
                      </>
                    )}
                  </div>
                  <pre className="mt-0.5 whitespace-pre-wrap break-words font-mono text-[var(--foreground)]">
                    {err.message}
                  </pre>
                </div>
              ))}
            </div>
          )}
          <StepAttemptFailures
            failures={instance.step_attempt_failures}
            status={instanceStatus}
            nested
          />
        </div>
      ) : (
        // No banner to fold into (a completed or running run, or a failure whose only
        // record is the ledger itself) — stand on its own.
        <StepAttemptFailures
          failures={instance.step_attempt_failures}
          status={instanceStatus}
        />
      )}
      <Tabs.Root defaultValue={defaultTabValue}>
        <div className="flex items-center justify-between">
          <Tabs.List>
            {hasTrajectories && (
              <Tabs.Trigger value="trajectory" className="my-1">
                Trajectory Viewer
              </Tabs.Trigger>
            )}
            {hasVerifications && (
              <Tabs.Trigger value="verifier" className="my-1">
                {hasAggregateOrSandbox
                  ? 'Verifier Results'
                  : rubricsCriteria
                  ? 'Rubric Verifier'
                  : 'Verifier Results'}
              </Tabs.Trigger>
            )}
            {showReviewerOverview && (
              <Tabs.Trigger value="overview" className="my-1">
                Overview
              </Tabs.Trigger>
            )}
            {hasCollectedArtifacts && (
              <Tabs.Trigger value="collected-artifacts" className="my-1">
                Collected Artifacts ({collectedArtifactsCount})
              </Tabs.Trigger>
            )}
            {hasServerConfig && (
              <Tabs.Trigger value="server-config" className="my-1">
                Server Config
              </Tabs.Trigger>
            )}
            {triggerRuntime && (
              <Tabs.Trigger value="triggers" className="my-1">
                Triggers
              </Tabs.Trigger>
            )}
            {hasPeerAgents && (
              <Tabs.Trigger value="peer-qna" className="my-1">
                Peer Q&amp;A
              </Tabs.Trigger>
            )}
            {hasConversations && (
              <Tabs.Trigger value="conversations" className="my-1">
                Conversation
              </Tabs.Trigger>
            )}
            <Tabs.Trigger value="context" className="my-1">
              Task Run Context
            </Tabs.Trigger>
          </Tabs.List>
          <div className="flex flex-shrink-0 items-center gap-2 px-2">
            {collapseBarDownloads ? (
              <DropdownMenu.Root>
                <DropdownMenu.Trigger>
                  <button
                    title={`Download any of ${barDownloads.length} files`}
                    className="flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
                  >
                    <Download size={12} />
                    Download Trajectories ({barDownloads.length})
                    <ChevronDown size={12} />
                  </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content sideOffset={4} align="end">
                  {/* Scroll on an inner element, not on Content: Radix Themes'
                      menu content sets its own overflow (it clips to the border
                      radius), so a max-height + overflow-y there is not
                      reliably honoured. role="none" keeps the wrapper out of
                      the menu's a11y tree; Radix tracks items by context, not
                      DOM parentage, so keyboard navigation is unaffected. */}
                  <div role="none" className="max-h-80 overflow-y-auto">
                    {barDownloads.map(item => (
                      <DropdownMenu.Item
                        key={item.key}
                        onSelect={() => runBarDownload(item)}
                        title={item.title}
                      >
                        <Download size={12} />
                        {item.label}
                      </DropdownMenu.Item>
                    ))}
                  </div>
                </DropdownMenu.Content>
              </DropdownMenu.Root>
            ) : (
              barDownloads.map(item => (
                <button
                  key={item.key}
                  onClick={() => runBarDownload(item)}
                  title={item.title}
                  className="flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
                >
                  <Download size={12} />
                  {item.label}
                </button>
              ))
            )}
          </div>
        </div>

        {hasTrajectories && (
          <Tabs.Content value="trajectory" className="p-4">
            {trajectories.length === 0 ? (
              <p className="text-sm text-[var(--muted-foreground)]">
                No trajectories were uploaded for this run yet.
              </p>
            ) : (
              <div className="flex flex-col gap-6">
                {trajectories.map((t, i) => (
                  <div key={i}>
                    {trajectories.length > 1 && (
                      // Per-section header + co-located download so users needn't scroll to the top toolbar for a step's trajectory.
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                          {t.label}
                        </h4>
                        <button
                          onClick={() =>
                            handleDownloadTrajectory(t.state.objectUrl)
                          }
                          title={`Download ${t.label}`}
                          className="flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
                        >
                          <Download size={12} />
                          Download
                        </button>
                      </div>
                    )}
                    {t.state.promptText && (
                      // Native <details>, open by default so the prompt shows without a click; collapsible when a long prompt pushes the trajectory below the fold.
                      <details
                        open
                        className="mb-3 rounded-md border border-[var(--border)] bg-[var(--secondary)] group"
                      >
                        <summary className="flex items-center justify-between cursor-pointer select-none p-3 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
                          <span>Task Prompt</span>
                          <span className="text-[var(--muted-foreground)] text-[10px] normal-case font-normal">
                            ({t.state.promptText.length.toLocaleString()} chars
                            · click to toggle)
                          </span>
                        </summary>
                        <p className="text-sm whitespace-pre-wrap px-4 pb-4 pt-1 border-t border-[var(--border)]">
                          {t.state.promptText}
                        </p>
                      </details>
                    )}
                    {t.state.status === 'idle' && (
                      <button
                        onClick={() => loadTrajectoryAt(i)}
                        className="flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors border border-[var(--border)]"
                      >
                        Load trajectory
                      </button>
                    )}
                    {t.state.status === 'loading' && (
                      <div className="flex items-center gap-2 text-sm text-[var(--muted-foreground)]">
                        <Loader2 size={14} className="animate-spin" />
                        Loading trajectory...
                      </div>
                    )}
                    {t.state.status === 'error' && (
                      <p className="text-sm text-red-500">{t.state.error}</p>
                    )}
                    {t.state.status === 'loaded' && t.state.trajectory && (
                      // <details open> so the trajectory shows right after Load; collapsible to scroll past it in a multi-step task.
                      <details
                        open
                        className="rounded-md border border-[var(--border)] group"
                      >
                        <summary className="flex items-center justify-between cursor-pointer select-none p-3 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
                          <span>Trajectory</span>
                          <span className="text-[var(--muted-foreground)] text-[10px] normal-case font-normal">
                            {t.state.trajectory.numTurns} turn
                            {t.state.trajectory.numTurns === 1
                              ? ''
                              : 's'} · {t.state.trajectory.toolCallCount} tool
                            call
                            {t.state.trajectory.toolCallCount === 1 ? '' : 's'}
                            {' · click to collapse'}
                          </span>
                        </summary>
                        <div className="border-t border-[var(--border)] p-3">
                          <TrajectoryViewer
                            trajectory={t.state.trajectory}
                            screenshotBaseUri={t.state.objectUrl}
                          />
                        </div>
                      </details>
                    )}
                    <TriggerTurnStrip
                      runtime={triggerRuntime}
                      stepId={t.stepId}
                      turnIndex={t.turnIndex}
                      authored={authoredTriggers}
                      nextPromptText={t.nextTurnPromptText}
                    />
                  </div>
                ))}
              </div>
            )}
          </Tabs.Content>
        )}

        {hasVerifications && (
          <Tabs.Content
            value="verifier"
            className={hasAggregateOrSandbox || !rubricsCriteria ? 'p-4' : ''}
          >
            {hasAggregateOrSandbox ? (
              <div className="flex flex-col gap-6">
                {/* Aggregate score(s) — the headline value: shown at top
                    so users see the final merged outcome before scrolling
                    through the individual verifiers that fed into it. */}
                {aggregateVerifiers.map(([id, v]) => (
                  <AggregateScoreCard key={id} verifierId={id} verifier={v} />
                ))}
                {/* Rubric panels — one per rubric-shaped verifier. We
                    previously passed the full set into a single panel,
                    but `getFirstVerifierOutput` returns only the first
                    key, silently dropping outputs from any additional
                    rubric verifiers (e.g. an AgentPromptResponseVerifier
                    + a separate rubrics_verifier in the same task). Rendering
                    one panel per id surfaces all of them. */}
                {rubricsCriteria &&
                  rubricVerifiers.map(([id, v]) => (
                    <React.Fragment key={id}>
                      {renderRubricVerifierPanel(
                        id,
                        v,
                        rubricsCriteria,
                        rubricsAggregator,
                      )}
                    </React.Fragment>
                  ))}
                {/* Sandbox verifier(s) — filesystem/probe outcomes. */}
                {sandboxVerifiers.map(([id, v]) => (
                  <SandboxVerifierCard
                    key={id}
                    verifierId={id}
                    verifier={v}
                    collectedArtifacts={collectedArtifacts}
                  />
                ))}
              </div>
            ) : rubricsCriteria ? (
              rubricVerifiers.length > 1 ? (
                <div className="flex flex-col gap-6">
                  {rubricVerifiers.map(([id, v]) => (
                    <React.Fragment key={id}>
                      {renderRubricVerifierPanel(
                        id,
                        v,
                        rubricsCriteria,
                        rubricsAggregator,
                      )}
                    </React.Fragment>
                  ))}
                </div>
              ) : rubricVerifiers[0] ? (
                renderRubricVerifierPanel(
                  rubricVerifiers[0][0],
                  rubricVerifiers[0][1],
                  rubricsCriteria,
                  rubricsAggregator,
                )
              ) : (
                <RubricGradingResults
                  verificationResults={
                    verifications as unknown as VerificationResults
                  }
                  rubrics={rubricsCriteria as unknown as RubricCriterion[]}
                  aggregator={rubricsAggregator}
                />
              )
            ) : (
              <div className="flex flex-col gap-4">
                {Object.entries(verifications!).map(([verifierId, v]) => {
                  const results = Array.isArray(v.results) ? v.results : [];
                  const isValidationStyle = 'passed' in v;

                  if (isValidationStyle) {
                    return (
                      <MCPEnvValidationEntry
                        key={verifierId}
                        validationKey={verifierId}
                        data={v as Record<string, unknown>}
                      />
                    );
                  }

                  const passed = results.filter(r => r.result).length;
                  const total = results.length;
                  const verifierPassed = v.score >= 1;
                  return (
                    <div key={verifierId} className="flex flex-col gap-4">
                      {/* Summary */}
                      <div className="flex items-center gap-3 flex-wrap">
                        {verifierPassed ? (
                          <CheckCircle2 size={18} className="text-green-500" />
                        ) : (
                          <XCircle size={18} className="text-red-500" />
                        )}
                        <span className="text-sm font-semibold">
                          {passed}/{total} checks passed
                        </span>
                        <span
                          className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                            verifierPassed
                              ? 'bg-green-500/10 text-green-500'
                              : 'bg-red-500/10 text-red-500'
                          }`}
                        >
                          Score: {(v.score * 100).toFixed(0)}%
                        </span>
                      </div>

                      {results.map((r, i) => (
                        <div
                          key={i}
                          className="rounded-lg border border-[var(--border)] p-4"
                        >
                          {/* Check header */}
                          <div className="flex items-center gap-2 mb-2">
                            {r.result ? (
                              <CheckCircle2
                                size={16}
                                className="text-green-500 flex-shrink-0"
                              />
                            ) : (
                              <XCircle
                                size={16}
                                className="text-red-500 flex-shrink-0"
                              />
                            )}
                            <span className="text-sm font-semibold">
                              {(r.id ?? `check ${i + 1}`).replace(/_/g, ' ')}
                            </span>
                            {typeof r.score === 'number' && r.score !== 0 && r.score !== 1 && (
                              <span className="text-xs text-[var(--muted-foreground)]">
                                score:{' '}
                                {typeof r.score === 'number'
                                  ? r.score.toFixed(2)
                                  : r.score}
                              </span>
                            )}
                          </div>
                          {r.message ? (
                            <p className="text-xs text-[var(--muted-foreground)] whitespace-pre-wrap">
                              {r.message}
                            </p>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            )}
          </Tabs.Content>
        )}

        {showReviewerOverview && reviewerOverviewSrc && (
          <Tabs.Content value="overview" className="p-0">
            {overviewFileList.length > 1 && (
              <div className="flex items-center gap-2 px-3 py-2 text-sm text-[var(--gray-11)]">
                <select
                  value={activeOverviewFile ?? ''}
                  onChange={e => setSelectedOverviewFile(e.target.value)}
                  className="rounded border border-[var(--gray-6)] bg-transparent px-2 py-1 text-sm"
                >
                  {overviewFileList.map(f => (
                    <option key={f} value={f}>
                      {f.replace(/^\/app\//, '')}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <iframe
              key={reviewerOverviewSrc}
              title="Sandbox file"
              src={reviewerOverviewSrc}
              sandbox=""
              className="w-full"
              style={{ height: '80vh', border: 'none' }}
            />
          </Tabs.Content>
        )}

        {hasCollectedArtifacts && (
          <Tabs.Content value="collected-artifacts" className="p-4">
            <div className="flex flex-col gap-5">
              {collectedArtifactSections.map((section, i) => (
                <div
                  key={section.stepId ?? `legacy-${i}`}
                  className="flex flex-col gap-3"
                >
                  {collectedArtifactSections.length > 1 && section.stepId && (
                    <div className="text-xs font-medium text-[var(--muted-foreground)]">
                      Step{' '}
                      <code className="px-1.5 py-0.5 rounded bg-[var(--background)] border border-[var(--border)] font-mono">
                        {section.stepId}
                      </code>
                    </div>
                  )}
                  <CollectedArtifactsList
                    key={`${instanceId}-${section.stepId ?? i}`}
                    artifacts={section.artifacts}
                    basePath={section.basePath}
                    taskId={taskId}
                    instanceId={instanceId}
                    stepId={section.stepId ?? ''}
                  />
                </div>
              ))}
            </div>
          </Tabs.Content>
        )}

        {hasServerConfig && (
          <Tabs.Content value="server-config" className="p-4">
            <ServerConfigPanel
              changes={serverConfigChanges}
              failures={serverConfigFailures}
              skipped={serverConfigSkipped}
            />
          </Tabs.Content>
        )}

        {triggerRuntime && (
          <Tabs.Content value="triggers" className="p-4">
            <TriggersPanel
              runtime={triggerRuntime}
              authored={authoredTriggers}
              taskId={taskId}
              instanceId={instanceId}
            />
          </Tabs.Content>
        )}

        {hasPeerAgents && (
          <Tabs.Content value="peer-qna" className="p-0">
            <PeerQnAPanel
              trajectories={trajectories.map(t => ({
                label: t.label,
                trajectory: t.state.trajectory,
              }))}
            />
          </Tabs.Content>
        )}

        {hasConversations && (
          <Tabs.Content value="conversations" className="p-0">
            <ConversationsPanel instanceId={instanceId} />
          </Tabs.Content>
        )}


        <Tabs.Content value="context" className="p-4">
          {context ? (
            <div>
              <div className="mb-2">
                <Button
                  variant="outline"
                  size="1"
                  onClick={() => {
                    navigator.clipboard.writeText(
                      JSON.stringify(context, null, 2),
                    );
                    setCopiedContext(true);
                    setTimeout(() => setCopiedContext(false), 1500);
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  {copiedContext ? (
                    <>
                      <Check size={12} className="text-green-500" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy size={12} />
                      Copy
                    </>
                  )}
                </Button>
              </div>
              <ScrollArea
                scrollbars="both"
                style={{ maxHeight: 500 }}
                className="rounded-md border border-[var(--border)] bg-[var(--secondary)]"
              >
                <pre className="p-4 text-xs font-mono">
                  {JSON.stringify(context, null, 2)}
                </pre>
              </ScrollArea>
            </div>
          ) : (
            <p className="text-sm text-[var(--muted-foreground)]">
              No context available for this task run.
            </p>
          )}
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}

// Surface the backend's structured `detail` (e.g. the 413 "too large" message)
// instead of a bare status code.
async function trajectoryFetchError(
  res: Response,
  fallback: string,
): Promise<Error> {
  const detail = await res
    .json()
    .then(body => (body as { detail?: string })?.detail)
    .catch(() => undefined);
  return new Error(detail || `${fallback} (${res.status})`);
}

async function fetchTrajectory(
  objectUrl: string,
  modelHint?: string,
): Promise<ParsedTrajectory> {
  const base = objectContentUrl(objectUrl);
  let res = await apiFetch(base);
  // Too large to inline? Retry the screenshot-trimmed stream (frames become lazy-loaded placeholders).
  // Only .json is trimmable, so a non-JSON 413 keeps its message rather than retrying into an identical 413.
  if (res.status === 413 && objectUrl.endsWith('.json')) {
    res = await apiFetch(`${base}&trim=screenshots`);
  }
  if (!res.ok) {
    throw await trajectoryFetchError(res, 'Failed to fetch trajectory');
  }
  const spans = (await res.json()) as OtelSpan[];

  // modelHint feeds formats whose trajectory carries no model (e.g. OpenClaw),
  // sourced from the prompt-response's `model`.
  return parseOtelTrajectory(spans, { modelHint });
}

/** Top-of-tab card for the merged score from an `aggregate_verifiers` step. Reads `source_verifier_ids`
 *  so users see which verifiers fed the aggregate. */
function AggregateScoreCard({
  verifierId,
  verifier,
}: {
  verifierId: string;
  verifier: Record<string, unknown>;
}) {
  const score = typeof verifier.score === 'number' ? verifier.score : 0;
  const sourceIds = Array.isArray(verifier.source_verifier_ids)
    ? (verifier.source_verifier_ids as string[])
    : [];
  const aggregator =
    typeof verifier.score_aggregator === 'string'
      ? (verifier.score_aggregator as string)
      : null;
  const passed = score >= 1;
  return (
    <div
      className={`rounded-lg border-2 p-5 ${
        passed
          ? 'border-emerald-300 bg-emerald-50/40'
          : score >= 0.5
          ? 'border-amber-300 bg-amber-50/40'
          : 'border-red-300 bg-red-50/40'
      }`}
    >
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          {passed ? (
            <CheckCircle2 size={28} className="text-emerald-600" />
          ) : (
            <XCircle size={28} className="text-red-500" />
          )}
          <div className="flex flex-col">
            <span className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Final Aggregate Score
            </span>
            <span className="text-2xl font-bold">
              {(score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        {aggregator && (
          <span
            className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded border border-[var(--border)] text-[var(--muted-foreground)] bg-[var(--background)]"
            title={`Aggregator: ${aggregator}`}
          >
            {aggregator}
          </span>
        )}
        <code className="text-[10px] font-mono text-[var(--muted-foreground)] ml-auto">
          {verifierId}
        </code>
      </div>
      {sourceIds.length > 0 && (
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          <span className="text-xs text-[var(--muted-foreground)]">
            Merged from:
          </span>
          {sourceIds.map(sid => (
            <code
              key={sid}
              className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--background)] border border-[var(--border)]"
            >
              {sid}
            </code>
          ))}
        </div>
      )}
    </div>
  );
}

/** Sandbox-side verifier outcomes from a `verify_sandbox` step. Per-criterion rows show the criterion
 *  `type` (probe_file_exists, bash_cmd_succeeds, …) and its `paths`/`cmd`. */
// Sandbox verifier — justification-string parsers. Each probe type emits its own `justification` format:
//   probe_file_exists / probe_dir_exists: "All paths exist" | "Missing: a, b, c"
//   probe_file_contains: "contains 'needle'" | "does not contain 'needle'"  (Python !r repr)
//   bash_cmd_succeeds: "exit=N; stderr=<first 200 chars>"
// Parsing structures these into per-path markers, an exit/stderr panel, and a found/not-found badge.

function parseMissingPaths(justification: string): Set<string> | null {
  const m = justification.match(/^Missing:\s*(.+)$/);
  if (!m) return null;
  return new Set(
    (m[1] ?? '')
      .split(',')
      .map(p => p.trim())
      .filter(Boolean),
  );
}

function parseBashOutcome(
  justification: string,
): { exitCode: number | null; stderr: string } | null {
  // [\s\S]* (not .*) catches multiline stderr — the emitter caps at 200 chars but doesn't strip newlines.
  const m = justification.match(/^exit=(-?\d+);\s*stderr=([\s\S]*)$/);
  if (!m) return null;
  const code = Number.parseInt(m[1] ?? '', 10);
  return {
    exitCode: Number.isFinite(code) ? code : null,
    stderr: m[2] ?? '',
  };
}

function parseContainsOutcome(
  justification: string,
): { found: boolean; needle: string } | null {
  // Python's !r is single-quoted by default, double when the string contains a single quote. Strip matching outer quotes.
  const stripQuotes = (s: string) => s.replace(/^(['"`])([\s\S]*)\1$/, '$2');
  if (justification.startsWith('does not contain ')) {
    return {
      found: false,
      needle: stripQuotes(
        justification.slice('does not contain '.length).trim(),
      ),
    };
  }
  if (justification.startsWith('contains ')) {
    return {
      found: true,
      needle: stripQuotes(justification.slice('contains '.length).trim()),
    };
  }
  return null;
}

/** Open a collected-artifact path chip via the /objects/content seam in a new tab, mirroring the Collected Artifacts download. */
function openCollectedArtifact(objectUrl: string): void {
  window.open(objectContentUrl(objectUrl), '_blank');
}

/** Path chip in the Sandbox Verifier card: a link when `objectUrl` is set (a collect step uploaded the file),
 *  green/red when a "Missing: …" set was parsed (`hasMissingSet`), else neutral. */
function SandboxPathChip({
  path,
  isMissing,
  hasMissingSet,
  objectUrl,
}: {
  path: string;
  isMissing: boolean;
  hasMissingSet: boolean;
  objectUrl: string | undefined;
}) {
  const stateClasses = hasMissingSet
    ? isMissing
      ? 'bg-red-50 text-red-700 border-red-300'
      : 'bg-emerald-50 text-emerald-700 border-emerald-300'
    : 'bg-[var(--background)] text-[var(--foreground)] border-[var(--border)]';
  const baseClasses = `inline-flex items-center gap-1 px-1.5 py-0.5 rounded font-mono border ${stateClasses}`;
  const icon = hasMissingSet ? (
    isMissing ? (
      <XCircle size={10} className="text-red-500" />
    ) : (
      <CheckCircle2 size={10} className="text-emerald-500" />
    )
  ) : null;

  if (objectUrl) {
    return (
      <button
        type="button"
        onClick={() => openCollectedArtifact(objectUrl)}
        className={`${baseClasses} cursor-pointer hover:underline hover:bg-[var(--accent)]`}
        title={
          hasMissingSet && isMissing
            ? 'Missing — open the collected version from a sibling step'
            : 'Open file from collected artifacts'
        }
      >
        {icon}
        {path}
      </button>
    );
  }
  return (
    <code
      className={baseClasses}
      title={hasMissingSet ? (isMissing ? 'Missing' : 'Present') : undefined}
    >
      {icon}
      {path}
    </code>
  );
}

function SandboxVerifierCard({
  verifierId,
  verifier,
  collectedArtifacts,
}: {
  verifierId: string;
  verifier: Record<string, unknown>;
  /** Map of filename → object_url from a sibling `collect_artifacts` step; when set, matching path chips link
   *  to the object. Null when nothing was collected. */
  collectedArtifacts: Record<string, string> | null;
}) {
  const score = typeof verifier.score === 'number' ? verifier.score : 0;
  const rawResults = Array.isArray(verifier.results)
    ? (verifier.results as Record<string, unknown>[])
    : [];
  const nonSkipped = rawResults.filter(r => !r?.skipped);
  const passedCount = nonSkipped.filter(r => r.result === true).length;
  const totalCount = nonSkipped.length;
  // Sandbox probes are deterministic 0/1 checks, not the agent judge — keep the score-based gate.
  const allPassed = score >= 1;

  return (
    <div className="rounded-lg border border-[var(--border)]">
      <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--secondary)]/40 flex items-center gap-3 flex-wrap">
        {allPassed ? (
          <CheckCircle2 size={18} className="text-emerald-500" />
        ) : (
          <XCircle size={18} className="text-red-500" />
        )}
        <span className="text-sm font-semibold">Sandbox Verifier</span>
        <span className="text-xs text-[var(--muted-foreground)]">
          {passedCount}/{totalCount} probes passed
        </span>
        <span
          className={`text-xs font-semibold px-1.5 py-0.5 rounded ml-auto ${
            allPassed
              ? 'bg-green-500/10 text-green-500'
              : 'bg-red-500/10 text-red-500'
          }`}
        >
          {(score * 100).toFixed(0)}%
        </span>
        <code className="text-[10px] font-mono text-[var(--muted-foreground)]">
          {verifierId}
        </code>
      </div>
      <div className="flex flex-col divide-y divide-[var(--border)]">
        {rawResults.map((r, i) => {
          const type =
            typeof r.type === 'string' ? (r.type as string) : 'criterion';
          const passed = r.result === true;
          const skipped = r.skipped === true;
          const justification =
            typeof r.justification === 'string'
              ? (r.justification as string)
              : '';
          const paths = Array.isArray(r.paths) ? (r.paths as string[]) : [];
          // The bash_cmd_succeeds field is `bash_cmd`; older configs use `cmd` — accept either.
          const cmd =
            (typeof r.bash_cmd === 'string' ? (r.bash_cmd as string) : null) ??
            (typeof r.cmd === 'string' ? (r.cmd as string) : null);
          // Stdout isn't currently emitted (justification has exit + stderr only), but read defensively for future versions.
          const stdout =
            (typeof r.stdout === 'string' ? (r.stdout as string) : null) ??
            (typeof r.bash_stdout === 'string'
              ? (r.bash_stdout as string)
              : null);
          // The substring field is `expected`; older configs / test data use `substring` — fall back to either.
          const expectedSubstr =
            (typeof r.expected === 'string' ? (r.expected as string) : null) ??
            (typeof r.substring === 'string' ? (r.substring as string) : null);
          // Partial scores are agent-env's escape hatch for "you got some
          // credit"; surface them so a 0.5 isn't visually identical to a 0.
          const rawScore =
            typeof r.score === 'number' ? (r.score as number) : null;
          const showPartialScore =
            rawScore !== null && rawScore !== 0 && rawScore !== 1;

          // Structured parses (null when format doesn't apply / failed to match)
          const missing =
            type === 'probe_file_exists' || type === 'probe_dir_exists'
              ? parseMissingPaths(justification)
              : null;
          const bashOutcome =
            type === 'bash_cmd_succeeds'
              ? parseBashOutcome(justification)
              : null;
          const containsOutcome =
            type === 'probe_file_contains'
              ? parseContainsOutcome(justification)
              : null;

          // Show raw justification only when we didn't structure it — otherwise it duplicates info already shown.
          const showRawJustification =
            justification &&
            !missing &&
            !bashOutcome &&
            !containsOutcome &&
            // Suppress the always-emitted "All paths exist" line on exists probes — the path chips already say it.
            !(
              (type === 'probe_file_exists' || type === 'probe_dir_exists') &&
              passed
            );

          return (
            <div key={i} className="px-4 py-3">
              <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                {skipped ? (
                  <span className="text-gray-400">—</span>
                ) : passed ? (
                  <CheckCircle2
                    size={16}
                    className="text-emerald-500 flex-shrink-0"
                  />
                ) : (
                  <XCircle size={16} className="text-red-500 flex-shrink-0" />
                )}
                <code className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[var(--foreground)]">
                  {type}
                </code>
                {skipped && (
                  <span className="text-[10px] font-semibold uppercase text-[var(--muted-foreground)]">
                    skipped
                  </span>
                )}
                {showPartialScore && (
                  <span
                    className="text-[10px] font-mono text-[var(--muted-foreground)]"
                    title={`Raw score: ${rawScore}`}
                  >
                    score {rawScore!.toFixed(2)}
                  </span>
                )}
              </div>

              {/* Paths — when this is an exists/dir probe and we parsed a
                  missing-set, mark each chip individually so the failing
                  path stands out instead of just the parent row flipping
                  red. */}
              {paths.length > 0 && (
                <div className="text-xs text-[var(--muted-foreground)] flex items-center gap-2 flex-wrap mb-1">
                  <span>Paths:</span>
                  {paths.map(p => (
                    <SandboxPathChip
                      key={p}
                      path={p}
                      isMissing={missing?.has(p) ?? false}
                      hasMissingSet={!!missing}
                      objectUrl={collectedArtifacts?.[p]}
                    />
                  ))}
                </div>
              )}

              {cmd && (
                <div className="text-xs text-[var(--muted-foreground)] flex items-center gap-2 mb-1">
                  <span>$</span>
                  <code className="px-1.5 py-0.5 rounded bg-[var(--background)] border border-[var(--border)] font-mono">
                    {cmd}
                  </code>
                </div>
              )}

              {/* Expected substring + found/not-found badge for
                  probe_file_contains. Pulls the needle from the criterion
                  field when present; falls back to the parsed-from-
                  justification needle. */}
              {(expectedSubstr || containsOutcome) && (
                <div className="text-xs text-[var(--muted-foreground)] flex items-center gap-2 mb-1 flex-wrap">
                  <span>{containsOutcome ? 'Result:' : 'Contains:'}</span>
                  {containsOutcome && (
                    <span
                      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                        containsOutcome.found
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-300'
                          : 'bg-red-50 text-red-700 border border-red-300'
                      }`}
                    >
                      {containsOutcome.found ? (
                        <CheckCircle2 size={10} className="text-emerald-500" />
                      ) : (
                        <XCircle size={10} className="text-red-500" />
                      )}
                      {containsOutcome.found ? 'Found' : 'Not found'}
                    </span>
                  )}
                  {(expectedSubstr ?? containsOutcome?.needle) && (
                    <code className="px-1.5 py-0.5 rounded bg-[var(--background)] border border-[var(--border)] font-mono">
                      {expectedSubstr ?? containsOutcome?.needle}
                    </code>
                  )}
                </div>
              )}

              {/* Structured exit + stderr panel for bash_cmd_succeeds.
                  Always shown when the probe ran (even on exit=0), since
                  exit code on its own is useful confirmation; stderr
                  block is suppressed when empty. */}
              {bashOutcome && (
                <div className="mt-1 flex flex-col gap-1">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-[var(--muted-foreground)]">
                      Exit code:
                    </span>
                    <span
                      className={`font-mono px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                        bashOutcome.exitCode === 0
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-300'
                          : 'bg-red-50 text-red-700 border border-red-300'
                      }`}
                    >
                      {bashOutcome.exitCode ?? '?'}
                    </span>
                  </div>
                  {stdout && stdout.trim() && (
                    <div className="text-xs">
                      <div className="text-[var(--muted-foreground)] mb-0.5">
                        stdout:
                      </div>
                      <pre className="text-[11px] font-mono bg-[var(--secondary)] rounded p-2 whitespace-pre-wrap break-all max-h-64 overflow-auto">
                        {stdout}
                      </pre>
                    </div>
                  )}
                  {bashOutcome.stderr.trim() && (
                    <div className="text-xs">
                      <div className="text-[var(--muted-foreground)] mb-0.5">
                        stderr:
                      </div>
                      <pre className="text-[11px] font-mono bg-[var(--secondary)] rounded p-2 whitespace-pre-wrap break-all max-h-64 overflow-auto">
                        {bashOutcome.stderr}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {showRawJustification && (
                <p className="text-xs text-[var(--muted-foreground)] mt-1 whitespace-pre-wrap">
                  {justification}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Collected Artifacts — files pulled off the sandbox by a `collect_artifacts` step. The instance carries
// metadata.artifacts = { filename: object_url }; the source path derives from the step's base_path config.

type ArtifactKind =
  | 'image'
  | 'text'
  | 'pdf'
  | 'docx'
  | 'xlsx'
  | 'pptx'
  | 'other';

const _IMAGE_EXTS = new Set([
  'png',
  'jpg',
  'jpeg',
  'gif',
  'webp',
  'svg',
  'bmp',
  'ico',
  'avif',
]);

const _TEXT_EXTS = new Set([
  'txt',
  'log',
  'md',
  'markdown',
  'json',
  'jsonl',
  'yaml',
  'yml',
  'py',
  'js',
  'jsx',
  'ts',
  'tsx',
  'sh',
  'bash',
  'zsh',
  'sql',
  'html',
  'css',
  'scss',
  'xml',
  'csv',
  'patch',
  'diff',
  'c',
  'cpp',
  'h',
  'hpp',
  'java',
  'rb',
  'go',
  'rs',
  'conf',
  'ini',
  'toml',
  'env',
  'gitignore',
]);

// Filenames with no extension that we still treat as text. Compared case-
// insensitively against the basename.
const _TEXT_NAMES = new Set([
  'dockerfile',
  'makefile',
  'readme',
  'license',
  'changelog',
]);

function classifyArtifact(filename: string): ArtifactKind {
  const base = (filename.split('/').pop() ?? filename).toLowerCase();
  const dot = base.lastIndexOf('.');
  if (dot < 0) {
    return _TEXT_NAMES.has(base) ? 'text' : 'other';
  }
  const ext = base.slice(dot + 1);
  if (_IMAGE_EXTS.has(ext)) return 'image';
  if (ext === 'pdf') return 'pdf';
  if (ext === 'docx') return 'docx';
  if (ext === 'xlsx') return 'xlsx';
  if (ext === 'pptx') return 'pptx';
  if (_TEXT_EXTS.has(ext)) return 'text';
  return 'other';
}

// Client-side zip of the files shown in the tab, each at its precomputed base_path-relative path.
function CollectedArtifactsZipButton({
  instanceId,
  files,
}: {
  instanceId: string;
  files: { path: string; objectUrl: string }[];
}) {
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{
    done: number;
    total: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleZip = useCallback(async () => {
    // Dedupe paths so same-named files don't silently overwrite.
    const seen = new Set<string>();
    const uniquePath = (p: string): string => {
      if (!seen.has(p)) return seen.add(p), p;
      const slash = p.lastIndexOf('/');
      const dot = p.lastIndexOf('.');
      const [stem, ext] =
        dot > slash ? [p.slice(0, dot), p.slice(dot)] : [p, ''];
      let i = 2;
      while (seen.has(`${stem} (${i})${ext}`)) i++;
      const c = `${stem} (${i})${ext}`;
      return seen.add(c), c;
    };
    const jobs = files.map(f => ({
      objectUrl: f.objectUrl,
      zipPath: uniquePath(f.path),
    }));
    if (jobs.length === 0) return;

    setBusy(true);
    setError(null);
    setProgress({ done: 0, total: jobs.length });
    const zip = new JSZip();
    const problems: string[] = [];
    let done = 0;
    let next = 0;
    const worker = async () => {
      while (next < jobs.length) {
        const job = jobs[next++]!;
        for (let attempt = 0; attempt <= 2; attempt++) {
          try {
            const r = await apiFetch(objectContentUrl(job.objectUrl));
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            zip.file(job.zipPath, await r.blob());
            break;
          } catch (e) {
            if (attempt === 2)
              problems.push(
                `FETCH FAILED: ${job.zipPath} (${
                  e instanceof Error ? e.message : String(e)
                })`,
              );
          }
        }
        done += 1;
        setProgress({ done, total: jobs.length });
      }
    };
    try {
      await Promise.all(
        Array.from({ length: Math.min(6, jobs.length) }, worker),
      );
      if (problems.length) zip.file('_failed.txt', problems.join('\n') + '\n');
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${instanceId}-artifacts.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Defer revoke so Safari has a chance to start the download.
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'zip failed');
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }, [instanceId, files]);

  const total = files.length;
  if (total === 0) return null;

  return (
    <button
      type="button"
      onClick={handleZip}
      disabled={busy}
      title={
        error
          ? `Zip failed: ${error} — click to retry`
          : `Download all ${total} file${
              total === 1 ? '' : 's'
            } as a zip (folder structure preserved)`
      }
      className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors disabled:opacity-50 ${
        error
          ? 'text-red-500 hover:bg-[var(--accent)]'
          : 'text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)]'
      }`}
    >
      {busy ? (
        <Loader2 size={12} className="animate-spin" />
      ) : (
        <Download size={12} />
      )}
      {busy && progress
        ? `Zipping ${progress.done}/${progress.total}`
        : 'Download All'}
    </button>
  );
}

function CollectedArtifactsList({
  artifacts,
  basePath,
  taskId,
  instanceId,
  stepId,
}: {
  artifacts: Record<string, string>;
  basePath: string;
  taskId: string;
  instanceId: string;
  stepId: string;
}) {
  const basename = (p: string) => p.split(/[\\/]/).pop() || p;

  const cards = Object.entries(artifacts).map(([path, uri]) => ({
    key: path,
    filename: basename(path),
    objectUrl: uri,
    sourcePath: isAbsoluteArtifactPath(path)
      ? path
      : `${basePath.replace(/\/+$/, '')}/${path}`,
  }));

  const zipFiles = collectedZipFiles(artifacts, basePath);

  // Render every card — no pagination. Each lazy-loads its URL/bytes/preview only when expanded, so unexpanded rows are cheap.
  const total = cards.length;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-[var(--muted-foreground)]">
          {total} file{total === 1 ? '' : 's'}.
        </p>
        {zipFiles.length > 0 && (
          <CollectedArtifactsZipButton
            instanceId={instanceId}
            files={zipFiles}
          />
        )}
      </div>
      <div className="flex flex-col gap-3">
        {cards.map(c => (
          <ArtifactCard
            key={c.key}
            filename={c.filename}
            objectUrl={c.objectUrl}
            sourcePath={c.sourcePath}
          />
        ))}
      </div>
    </div>
  );
}

const LARGE_FILE_BYTES = 10 * 1024 * 1024; // 10 MB

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function ArtifactCard({
  filename,
  objectUrl,
  sourcePath,
}: {
  filename: string;
  objectUrl: string;
  sourcePath: string;
}) {
  const kind = classifyArtifact(filename);
  const [contentUrl, setContentUrl] = useState<string | null>(null);
  const [sizeBytes, setSizeBytes] = useState<number | null>(null);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [textPreview, setTextPreview] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [loadingText, setLoadingText] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // Lazy-resolve the content URL on first need. Returns url + size so
  // callers can gate on size without waiting for a second state flush.
  const ensureContentUrl = useCallback(async (): Promise<{
    url: string;
    sizeBytes: number;
  } | null> => {
    if (contentUrl !== null && sizeBytes !== null)
      return { url: contentUrl, sizeBytes };
    try {
      const res = await apiFetch(
        `${BACKEND_URL}/api/v1/objects/metadata?object_url=${encodeURIComponent(objectUrl)}`,
      );
      if (!res.ok) throw new Error(`metadata HTTP ${res.status}`);
      const data = (await res.json()) as { size_bytes: number | null };
      const url = objectContentUrl(objectUrl);
      const size = data.size_bytes ?? 0;
      setContentUrl(url);
      setSizeBytes(size);
      return { url, sizeBytes: size };
    } catch (e) {
      setResolveError(e instanceof Error ? e.message : 'could not resolve the file');
      return null;
    }
  }, [contentUrl, sizeBytes, objectUrl]);

  const handleDownload = useCallback(async () => {
    const result = await ensureContentUrl();
    if (result) window.open(result.url, '_blank');
  }, [ensureContentUrl]);

  const handleToggle = useCallback(async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    const result = await ensureContentUrl();
    if (!result) return;
    // Files over the threshold show a download-only message — skip the
    // content fetch to avoid loading large payloads into the browser.
    if (result.sizeBytes > LARGE_FILE_BYTES) return;
    // image/pdf/docx render directly from the content URL; only text is
    // fetched-and-inlined here.
    if (kind !== 'text') return;
    if (!textPreview && !previewError) {
      setLoadingText(true);
      try {
        const res = await fetch(result.url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.text();
        // Cap inline render at ~64KB so a giant log file doesn't lock the
        // tab; the full file is still one click away via Download.
        const MAX = 64 * 1024;
        setTextPreview(
          body.length > MAX
            ? `${body.slice(0, MAX)}\n\n…[truncated — ${
                body.length - MAX
              } more bytes; use Download for the full file]`
            : body,
        );
      } catch (e) {
        setPreviewError(e instanceof Error ? e.message : 'preview failed');
      } finally {
        setLoadingText(false);
      }
    }
  }, [expanded, kind, ensureContentUrl, textPreview, previewError]);

  const Icon =
    kind === 'image'
      ? ImageIcon
      : kind === 'text' ||
        kind === 'pdf' ||
        kind === 'docx' ||
        kind === 'xlsx' ||
        kind === 'pptx'
      ? FileText
      : Download;

  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden">
      {/* role="button" wrapper (not a real <button>) so the nested Download
          <button> is valid HTML — nested buttons get auto-flattened by the
          parser, breaking event isolation. */}
      <div
        role="button"
        tabIndex={0}
        onClick={handleToggle}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            void handleToggle();
          }
        }}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-[var(--secondary)] transition-colors cursor-pointer"
      >
        <ChevronRight
          size={14}
          className={`flex-shrink-0 text-[var(--muted-foreground)] transition-transform ${
            expanded ? 'rotate-90' : ''
          }`}
        />
        <Icon
          size={14}
          className="flex-shrink-0 text-[var(--muted-foreground)]"
        />
        <span className="font-mono text-sm font-semibold truncate">
          {filename}
        </span>
        <span className="text-xs text-[var(--muted-foreground)] truncate ml-2 hidden sm:inline">
          {sourcePath}
        </span>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={e => {
              e.stopPropagation();
              handleDownload();
            }}
            className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
            title={`Download ${filename}`}
          >
            <Download size={12} />
            Download
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-[var(--border)] p-3">
          {/* Always show source path on expand, including on small screens
              where it's hidden in the row header. */}
          <div className="text-xs text-[var(--muted-foreground)] mb-2 flex items-center gap-2 flex-wrap">
            <span>Sandbox path:</span>
            <code className="px-1.5 py-0.5 rounded bg-[var(--background)] border border-[var(--border)] font-mono">
              {sourcePath}
            </code>
          </div>
          {resolveError && (
            <p className="text-xs text-red-500">
              Could not resolve download URL: {resolveError}
            </p>
          )}
          {!resolveError &&
          sizeBytes !== null &&
          sizeBytes > LARGE_FILE_BYTES ? (
            <p className="text-xs text-[var(--muted-foreground)]">
              File is {formatBytes(sizeBytes)} — too large to preview in the
              browser. Use the Download button to save it locally.
            </p>
          ) : (
            <>
              {kind === 'image' && (
                <ArtifactImagePreview contentUrl={contentUrl} />
              )}
              {kind === 'pdf' && (
                <ArtifactPdfPreview contentUrl={contentUrl} />
              )}
              {/* docx/xlsx fetch the file bytes on mount, so gate them on a
                  known size — otherwise they'd fetch before the large-file
                  guard (which needs sizeBytes) can apply. Once sizeBytes is
                  known and within the limit, this branch renders; oversized
                  files take the "too large" branch above. */}
              {kind === 'docx' &&
                (sizeBytes !== null ? (
                  <ArtifactDocxPreview objectUrl={objectUrl} />
                ) : (
                  <ArtifactResolving />
                ))}
              {kind === 'xlsx' &&
                (sizeBytes !== null ? (
                  <ArtifactXlsxPreview objectUrl={objectUrl} />
                ) : (
                  <ArtifactResolving />
                ))}
              {kind === 'pptx' &&
                (sizeBytes !== null ? (
                  <ArtifactPptxPreview objectUrl={objectUrl} />
                ) : (
                  <ArtifactResolving />
                ))}
              {kind === 'text' && (
                <ArtifactTextPreview
                  loading={loadingText}
                  text={textPreview}
                  error={previewError}
                />
              )}
              {kind === 'other' && !resolveError && (
                <p className="text-xs text-[var(--muted-foreground)]">
                  Inline preview not supported for this file type — use Download
                  to grab it.
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ArtifactImagePreview({
  contentUrl,
}: {
  contentUrl: string | null;
}) {
  if (!contentUrl) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
        <Loader2 size={12} className="animate-spin" />
        Resolving URL…
      </div>
    );
  }
  return (
    <img
      src={contentUrl}
      alt=""
      className="max-w-full max-h-[600px] rounded border border-[var(--border)] bg-white object-contain"
    />
  );
}

function ArtifactResolving() {
  return (
    <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
      <Loader2 size={12} className="animate-spin" />
      Resolving URL…
    </div>
  );
}

function ArtifactPdfPreview({ contentUrl }: { contentUrl: string | null }) {
  if (!contentUrl) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
        <Loader2 size={12} className="animate-spin" />
        Resolving URL…
      </div>
    );
  }
  // Browsers render PDFs natively in an <iframe>. The url is same-origin: the
  // `/api/v1/objects/content` proxy.
  //
  // Sandboxed like the other previews: a PDF is agent-produced bytes rendered in an
  // unauthenticated origin, and while Chrome and Firefox both isolate PDF scripting
  // from the embedder, that is the viewer's choice rather than ours. `allow-popups`
  // keeps the built-in viewer's download and print affordances working.
  return (
    <iframe
      src={contentUrl}
      title="PDF preview"
      sandbox="allow-popups"
      className="w-full h-[600px] rounded border border-[var(--border)] bg-white"
    />
  );
}

/**
 * Prepare a sandboxed iframe document for rendering an untrusted file preview.
 *
 * These previews render agent-produced files inside an unauthenticated control-plane
 * origin, and the renderers build DOM straight from the file bytes — docx-preview
 * assigns a raw `w:sym w:char` into `innerHTML`, which needs no click to fire, and the
 * SheetJS HTML writer emits `<a href>` without filtering the `javascript:` scheme.
 *
 * `sandbox="allow-same-origin"` WITHOUT `allow-scripts` means nothing inside the frame
 * executes — no scripts, no `onerror`/`onload`, no `javascript:` navigation. Do not add
 * `allow-scripts`; together with `allow-same-origin` it lets framed content escape the
 * sandbox entirely.
 *
 * READ THIS BEFORE BUMPING docx-preview OR pptx-preview. Neither library renders *into*
 * this frame. `docx-preview` hardcodes `new HtmlRenderer(window.document)` and
 * `pptx-preview` does `document.createElement("span"); s.innerHTML = ...` — both parse
 * attacker-controlled markup in the HOST document and hand us a finished tree. A
 * detached host-document element still fires `onerror`, so what actually saves us is
 * that both append into the frame synchronously, in the same job as the parse, before
 * the host can run the handler. Measured: adopting in the same job does not fire;
 * adopting 300ms later does.
 *
 * So the containment is one dependency bump wide, not one attribute wide. A version
 * that awaits between building and appending re-opens this with no change here and
 * `preview-sandbox.smoke.ts` still green. The structural fix is to render into a fully
 * opaque `sandbox=""` + `srcdoc` frame the way the spreadsheet preview does; until
 * then the versions in package.json are load-bearing.
 *
 * Same-origin is also what lets `fitSandboxFrame` size the frame to its content from
 * out here, so a short document does not sit in a tall empty box — the usual
 * postMessage-from-inside trick would need a script in the frame, which is the one
 * thing we are preventing.
 */
function prepareSandboxDoc(frame: HTMLIFrameElement): HTMLElement | null {
  const doc = frame.contentDocument;
  if (!doc) return null;
  doc.open();
  doc.write(
    '<!doctype html><html><head><meta charset="utf-8">' +
      '<meta http-equiv="Content-Security-Policy" ' +
      "content=\"default-src 'none'; img-src data: blob:; " +
      "style-src 'unsafe-inline'; font-src data:\">" +
      '<style>html,body{margin:0;padding:8px;background:#fff;color:#000;' +
      'font:12px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}' +
      '</style></head><body></body></html>',
  );
  doc.close();
  return doc.body;
}

const SANDBOX_FRAME_CLASS =
  'w-full rounded border border-[var(--border)] bg-white';

/** Tallest a preview frame grows before it scrolls internally. */
const SANDBOX_FRAME_MAX_HEIGHT = 600;

/**
 * The height a sandboxed preview frame needs for its content, capped at
 * `SANDBOX_FRAME_MAX_HEIGHT` — restoring the `maxHeight` behaviour the previews had
 * before they moved into iframes. Safe because the frame is same-origin; it reads
 * scrollHeight from out here rather than asking the (deliberately scriptless) frame.
 *
 * Returns rather than assigning `style.height`, so the value can live in React state.
 * Assigning it directly loses the race with the `setStatus('ready')` re-render, which
 * rewrites the style prop straight back to its default.
 */
function measureSandboxFrame(frame: HTMLIFrameElement): number {
  const doc = frame.contentDocument;
  if (!doc?.documentElement) return SANDBOX_FRAME_MAX_HEIGHT;
  const content = Math.max(
    doc.documentElement.scrollHeight,
    doc.body?.scrollHeight ?? 0,
  );
  return Math.min(content, SANDBOX_FRAME_MAX_HEIGHT);
}

function ArtifactDocxPreview({ objectUrl }: { objectUrl: string }) {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>(
    'loading',
  );
  const [frameHeight, setFrameHeight] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    setError(null);
    void (async () => {
      try {
        // docx isn't browser-native: fetch the bytes (through the backend, same-origin, so the store's CORS doesn't block) and render with docx-preview.
        const res = await fetch(objectContentUrl(objectUrl));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        if (cancelled || !frameRef.current) return;
        const { renderAsync } = await import('docx-preview');
        if (cancelled || !frameRef.current) return;
        const body = prepareSandboxDoc(frameRef.current);
        if (!body) throw new Error('preview frame unavailable');
        await renderAsync(blob, body, undefined, {
          className: 'docx-preview',
          inWrapper: true,
        });
        const fitted = measureSandboxFrame(frameRef.current);
        if (!cancelled) {
          setFrameHeight(fitted);
          setStatus('ready');
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'render failed');
          setStatus('error');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [objectUrl]);

  return (
    <div>
      {status === 'loading' && (
        <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
          <Loader2 size={12} className="animate-spin" />
          Rendering document…
        </div>
      )}
      {status === 'error' && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
          Inline preview unavailable ({error}). Use Download to fetch the file
          directly.
        </p>
      )}
      {/* Always mounted so the ref exists when renderAsync targets it; hidden
          until the render completes. Sandboxed — see prepareSandboxDoc. */}
      <iframe
        ref={frameRef}
        title="Document preview"
        sandbox="allow-same-origin"
        // Height, never `display: none`. Hiding an iframe with display:none and
        // revealing it later makes the browser re-navigate it to about:blank, which
        // throws away everything renderAsync wrote into contentDocument — the preview
        // then renders blank. A div survives being unhidden; an iframe does not.
        style={{ height: status === 'ready' ? frameHeight : 0 }}
        className={`${SANDBOX_FRAME_CLASS} ${
          status === 'ready' ? 'border' : 'border-0'
        }`}
      />
    </div>
  );
}

function ArtifactXlsxPreview({ objectUrl }: { objectUrl: string }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>(
    'loading',
  );
  const [error, setError] = useState<string | null>(null);
  const [sheets, setSheets] = useState<{ name: string; html: string }[]>([]);
  const [active, setActive] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    setError(null);
    void (async () => {
      try {
        // Fetch via the backend (same-origin) so the store's CORS doesn't block the
        // read, then parse + render to an HTML table with SheetJS client-side.
        const res = await fetch(objectContentUrl(objectUrl));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();
        const XLSX = await import('@e965/xlsx');
        const wb = XLSX.read(buf, { type: 'array' });
        // sheet_to_html HTML-escapes cell values, so the output is a static
        // table with no executable content.
        const parsed = wb.SheetNames.flatMap(name => {
          const ws = wb.Sheets[name];
          // header/footer '' omits the full-document <html>/<title> wrapper, so
          // we inject just the <table> (no leaking of the page <title>).
          return ws
            ? [
                {
                  name,
                  html: XLSX.utils.sheet_to_html(ws, {
                    header: '',
                    footer: '',
                  }),
                },
              ]
            : [];
        });
        if (!cancelled) {
          setSheets(parsed);
          setActive(0);
          setStatus('ready');
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'render failed');
          setStatus('error');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [objectUrl]);

  if (status === 'loading') {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
        <Loader2 size={12} className="animate-spin" />
        Rendering spreadsheet…
      </div>
    );
  }
  if (status === 'error') {
    return (
      <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
        Inline preview unavailable ({error}). Use Download to fetch the file
        directly.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      {sheets.length > 1 && (
        <div className="flex flex-wrap gap-1">
          {sheets.map((s, i) => (
            <button
              key={s.name}
              type="button"
              onClick={() => setActive(i)}
              className={`px-2 py-0.5 rounded text-xs font-mono ${
                active === i
                  ? 'bg-[var(--accent)] text-[var(--foreground)]'
                  : 'text-[var(--muted-foreground)] hover:bg-[var(--secondary)]'
              }`}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}
      {/* SheetJS escapes cell *values*, but its HTML writer emits `<a href>` from
          `cell.l.Target` without filtering the scheme, so a crafted sheet can ship a
          clickable `javascript:` link. srcdoc + a scriptless sandbox neutralises it:
          no allow-scripts means no execution and no javascript: navigation. */}
      <iframe
        title="Spreadsheet preview"
        sandbox=""
        style={{ height: SANDBOX_FRAME_MAX_HEIGHT }}
        className={`${SANDBOX_FRAME_CLASS} overflow-auto`}
        // Unlike the docx/pptx frames this one is fully opaque (sandbox="", not
        // allow-same-origin) because srcdoc needs no parent access to render. That
        // also puts contentDocument out of reach, so it cannot be fitted to content
        // and stays at the cap. Keeping the stronger sandbox is worth a tall box for
        // a small sheet; do not relax it to allow-same-origin just to shrink this.
        srcDoc={
          '<!doctype html><html><head><meta charset="utf-8">' +
          '<meta http-equiv="Content-Security-Policy" ' +
          "content=\"default-src 'none'; style-src 'unsafe-inline'\">" +
          '<style>body{margin:0;padding:8px;background:#fff;color:#000;' +
          'font:12px/1.5 ui-sans-serif,system-ui,sans-serif}' +
          'table{border-collapse:collapse}' +
          'td{border:1px solid #d1d5db;padding:2px 6px;white-space:nowrap}' +
          '</style></head><body>' +
          (sheets[active]?.html ?? '') +
          '</body></html>'
        }
      />
    </div>
  );
}

function ArtifactPptxPreview({ objectUrl }: { objectUrl: string }) {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>(
    'loading',
  );
  const [frameHeight, setFrameHeight] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    setError(null);
    void (async () => {
      try {
        // pptx isn't browser-native: fetch the bytes through the backend (same-origin) and render with pptx-preview.
        const res = await fetch(objectContentUrl(objectUrl));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();
        if (cancelled || !frameRef.current) return;
        const { init } = await import('pptx-preview');
        if (cancelled || !frameRef.current) return;
        const body = prepareSandboxDoc(frameRef.current);
        if (!body) throw new Error('preview frame unavailable');
        const width = frameRef.current.clientWidth || 960;
        const previewer = init(body, {
          width,
          height: Math.round(width * 0.5625), // 16:9
        });
        await previewer.preview(buf);
        const fitted = measureSandboxFrame(frameRef.current);
        if (!cancelled) {
          setFrameHeight(fitted);
          setStatus('ready');
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'render failed');
          setStatus('error');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [objectUrl]);

  return (
    <div>
      {status === 'loading' && (
        <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
          <Loader2 size={12} className="animate-spin" />
          Rendering slides…
        </div>
      )}
      {status === 'error' && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
          Inline preview unavailable ({error}). Use Download to fetch the file
          directly.
        </p>
      )}
      {/* Always mounted so the ref exists when preview() targets it; hidden
          until the render completes. Sandboxed — see prepareSandboxDoc. */}
      <iframe
        ref={frameRef}
        title="Slides preview"
        sandbox="allow-same-origin"
        // Height, never `display: none`. Hiding an iframe with display:none and
        // revealing it later makes the browser re-navigate it to about:blank, which
        // throws away everything preview() wrote into contentDocument — the preview
        // then renders blank. A div survives being unhidden; an iframe does not.
        style={{ height: status === 'ready' ? frameHeight : 0 }}
        className={`${SANDBOX_FRAME_CLASS} ${
          status === 'ready' ? 'border' : 'border-0'
        }`}
      />
    </div>
  );
}

function ArtifactTextPreview({
  loading,
  text,
  error,
}: {
  loading: boolean;
  text: string | null;
  error: string | null;
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
        <Loader2 size={12} className="animate-spin" />
        Loading preview…
      </div>
    );
  }
  if (error) {
    // Common cause: CORS blocks a cross-origin read. The file still downloads (CORS isn't enforced on navigation / <a download>).
    return (
      <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
        Inline preview unavailable ({error}). Use Download to fetch the file
        directly.
      </p>
    );
  }
  if (text === null) return null;
  return (
    <ScrollArea
      scrollbars="both"
      style={{ maxHeight: 500 }}
      className="rounded border border-[var(--border)] bg-[var(--secondary)]"
    >
      <pre className="p-3 text-xs font-mono whitespace-pre">{text}</pre>
    </ScrollArea>
  );
}
