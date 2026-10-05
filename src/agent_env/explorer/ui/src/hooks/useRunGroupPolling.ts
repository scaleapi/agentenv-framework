import { useState, useCallback, useRef, useEffect } from 'react';
import { BACKEND_URL, apiFetch } from '../components/shared';
import { readSSEStream } from '../lib/sse-parser';

// Types mirror the backend `/tasks/{id}/runs` + `/run-groups/{gid}` schemas.

export interface RunOverrides {
  agent_model?: string;
  agent_artifact_id?: string;
  a2a_agent_id?: string;
  [k: string]: string | number | boolean | null | undefined;
}

interface StartRunsConfig {
  version: number;
  overrides?: RunOverrides;
  count?: number;
  seeds?: Record<string, string>[];
  concurrency?: number;
}

export interface RunInfo {
  index: number;
  workflow_id?: string | null;
  seed?: Record<string, string> | null;
  error?: string | null;
}

interface StartRunsResponse {
  run_group_id: string;
  task_id: string;
  task_version: number;
  total: number;
  runs: RunInfo[];
}

interface RunGroupInstance {
  instance_id?: string | null;
  workflow_id?: string | null;
  seed?: Record<string, string> | null;
  status: string;
  created_at_utc?: string | null;
  completed_at_utc?: string | null;
}

interface RunGroupStatus {
  run_group_id: string;
  task_id: string;
  total: number;
  completed: number;
  failed: number;
  running: number;
  // Runs the backend accepted but whose worker hasn't reported progress yet (usually still provisioning). Optional for forward-compat.
  provisioning?: number;
  instances: RunGroupInstance[];
}

type PollingStatus = 'idle' | 'starting' | 'polling' | 'done' | 'error';

// Stream via fetch+ReadableStream (not EventSource) so auth propagates like every other fetch. Reconnect on drops; surface an error only after repeated failures.
const MAX_CONSECUTIVE_STREAM_ERRORS = 5;
const STREAM_RETRY_BASE_MS = 2000;
const STREAM_RETRY_MAX_MS = 60000;

export function useRunGroupPolling(taskId: string) {
  const [status, setStatus] = useState<PollingStatus>('idle');
  const [runGroup, setRunGroup] = useState<RunGroupStatus | null>(null);
  const [startResponse, setStartResponse] = useState<StartRunsResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const generationRef = useRef(0);

  const closeStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const startRuns = useCallback(
    async (config: StartRunsConfig) => {
      closeStream();
      const gen = ++generationRef.current;
      setStatus('starting');
      setRunGroup(null);
      setStartResponse(null);
      setError(null);

      const body: Record<string, unknown> = {
        version: config.version,
      };
      if (config.overrides && Object.keys(config.overrides).length > 0) {
        body.overrides = config.overrides;
      }
      if (config.seeds !== undefined) body.seeds = config.seeds;
      if (config.count !== undefined) body.count = config.count;
      if (config.concurrency !== undefined)
        body.concurrency = config.concurrency;

      let startBody: StartRunsResponse;
      try {
        const res = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(taskId)}/runs`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          },
        );
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`Failed to start runs (${res.status}): ${errText}`);
        }
        startBody = (await res.json()) as StartRunsResponse;
      } catch (e) {
        if (gen !== generationRef.current) return;
        setError(e instanceof Error ? e.message : 'Failed to start runs');
        setStatus('error');
        return;
      }

      if (gen !== generationRef.current) return;
      setStartResponse(startBody);
      setStatus('polling');

      if (startBody.runs.every(r => !r.workflow_id)) {
        setError('No runs were started — worker did not accept any requests');
        setStatus('error');
        return;
      }

      const url = `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
        taskId,
      )}/run-groups/${encodeURIComponent(startBody.run_group_id)}/stream`;
      const abort = new AbortController();
      abortRef.current = abort;

      // Set by the event handler on `complete`/`not_found`/`timeout` so the
      // reconnect loop knows a clean server-side end isn't a retry.
      let terminated = false;
      let consecutiveErrors = 0;

      const handleEvent = (
        eventName: string,
        data: Record<string, unknown>,
      ) => {
        if (gen !== generationRef.current) return;
        if (eventName === 'snapshot') {
          setRunGroup(data as unknown as RunGroupStatus);
          consecutiveErrors = 0;
        } else if (eventName === 'complete') {
          terminated = true;
          setStatus('done');
          abort.abort();
        } else if (eventName === 'not_found' || eventName === 'timeout') {
          terminated = true;
          setError(
            eventName === 'timeout'
              ? 'Run group stream timed out'
              : 'Run group not found',
          );
          setStatus('error');
          abort.abort();
        }
      };

      // Fire-and-forget reconnect loop. abortRef + generation counter are the kill-switches; a stream ending without a terminal event falls through to reconnect.
      (async () => {
        while (
          gen === generationRef.current &&
          !terminated &&
          !abort.signal.aborted
        ) {
          try {
            const res = await apiFetch(url, { signal: abort.signal });
            if (!res.ok || !res.body) {
              throw new Error(`Stream HTTP ${res.status}`);
            }
            await readSSEStream<Record<string, unknown>>(res.body, handleEvent);
            if (terminated || abort.signal.aborted) return;
          } catch (e) {
            if (abort.signal.aborted || gen !== generationRef.current) return;
            consecutiveErrors++;
            if (consecutiveErrors >= MAX_CONSECUTIVE_STREAM_ERRORS) {
              setError(
                e instanceof Error
                  ? `Lost connection to run group stream: ${e.message}`
                  : 'Lost connection to run group stream',
              );
              setStatus('error');
              return;
            }
            const delayMs = Math.min(
              STREAM_RETRY_MAX_MS,
              STREAM_RETRY_BASE_MS * 2 ** (consecutiveErrors - 1),
            );
            await new Promise(r => setTimeout(r, delayMs));
          }
        }
      })();
    },
    [taskId, closeStream],
  );

  // Close on unmount + taskId change (defensive against callers that don't
  // `key={taskId}` remount the panel).
  useEffect(() => () => closeStream(), [taskId, closeStream]);

  return { status, runGroup, startResponse, error, startRuns };
}
