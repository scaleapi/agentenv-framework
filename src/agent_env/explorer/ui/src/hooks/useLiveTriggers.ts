/** The live half of the Triggers tab: polls the aggregation endpoint while a run is in flight and returns
 *  what the panel renders. Extracted from the panel so the machinery (visibility gating, backoff, fingerprint gating, transcript diffing) has a name. */
import { useEffect, useRef, useState } from 'react';
import { BACKEND_URL, apiFetch } from '../components/shared';
import { timelineFingerprint } from '../lib/parse-trigger-events';
import { parseEnvClocks, type EnvClock } from '../lib/trigger-clock';

const POLL_MS = 3000;
const MAX_BACKOFF_MS = 30000;
// Turn markers are context, not the signal, and the conversations payload is the
// whole transcript — refresh it a few rounds apart from the events.
const CONVERSATION_POLL_EVERY = 4;

type Payload = Record<string, unknown>;

/** loading/failed have nothing to show; the other three do and differ only in whether it's still current.
 *  A union so "an error the render has no branch for" can't be expressed. `failed` means the panel is empty, not that the loop gave up — a first-poll 5xx keeps retrying. */
export type TriggerFeed =
  | { kind: 'loading' }
  | { kind: 'failed'; error: string }
  | { kind: 'ok'; payload: Payload }
  | { kind: 'retrying'; payload: Payload; error: string }
  | { kind: 'stopped'; payload: Payload; error: string };

export interface LiveTriggers {
  feed: TriggerFeed;
  /** Re-read every poll: the reading is the one thing that moves continuously. */
  clocks: EnvClock[];
  conversations: unknown[];
}

export function useLiveTriggers(
  taskId: string,
  instanceId: string,
): LiveTriggers {
  const [feed, setFeed] = useState<TriggerFeed>({ kind: 'loading' });
  const [clocks, setClocks] = useState<EnvClock[]>([]);
  const [conversations, setConversations] = useState<unknown[]>([]);
  // `null` rather than '' so a payload that fingerprints empty still lands once.
  const fingerprint = useRef<string | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let delay = POLL_MS;
    let round = 0;
    let paused = false;

    /** Keep whatever is on screen and say it is no longer current; blank only
     *  when there was never anything to keep. */
    const degrade = (kind: 'retrying' | 'stopped', error: string) =>
      setFeed(prev =>
        'payload' in prev
          ? { kind, payload: prev.payload, error }
          : { kind: 'failed', error },
      );

    // A backgrounded tab still costs two round trips every 3s for a view nobody's watching, so the loop parks and resumes on return.
    const schedule = () => {
      if (document.hidden) {
        paused = true;
        return;
      }
      timer = setTimeout(() => void poll(), delay);
    };

    const onVisibility = () => {
      if (!active || document.hidden || !paused) return;
      paused = false;
      void poll();
    };
    document.addEventListener('visibilitychange', onVisibility);

    // Turn markers are best-effort context: a conversations failure must not
    // cost the caller the event log.
    let lastTranscript = '';
    const loadConversations = async () => {
      try {
        const resp = await apiFetch(
          `${BACKEND_URL}/api/v1/task-instances/${encodeURIComponent(
            instanceId,
          )}/conversations`,
          { signal: controller.signal },
        );
        if (!resp.ok || !active) return;
        // Compared before parsing: an unchanged transcript as a new array would invalidate the timeline memo, whose parse walks every row.
        const text = await resp.text();
        if (!active || text === lastTranscript) return;
        lastTranscript = text;
        const data = JSON.parse(text) as { conversations?: unknown[] };
        setConversations(data.conversations ?? []);
      } catch {
        /* keep the timeline without turn markers */
      }
    };

    const poll = async () => {
      try {
        const resp = await apiFetch(
          `${BACKEND_URL}/api/v1/tasks/${encodeURIComponent(
            taskId,
          )}/instances/${encodeURIComponent(instanceId)}/triggers`,
          { signal: controller.signal },
        );
        if (!active) return;
        if (!resp.ok) {
          // 4xx is a verdict, not a hiccup — retrying it only makes noise. It
          // still has to say so, or stale data passes for current.
          if (resp.status < 500) {
            degrade(
              'stopped',
              `Failed to load trigger events (${resp.status})`,
            );
            return;
          }
          throw new Error(`HTTP ${resp.status}`);
        }
        const body = (await resp.json()) as Payload;
        if (!active) return;
        setClocks(parseEnvClocks(body.state_meta));
        // Payload identity is held steady while nothing the timeline draws has
        // changed, so a quiet run neither re-parses nor re-reconciles its log.
        const print = timelineFingerprint(body);
        const changed = print !== fingerprint.current;
        fingerprint.current = print;
        setFeed(prev =>
          !changed && 'payload' in prev
            ? prev.kind === 'ok'
              ? prev
              : { kind: 'ok', payload: prev.payload }
            : { kind: 'ok', payload: body },
        );
        delay = POLL_MS;
        // The payload's own status is authority — it normalizes a stale `running` to `cancelled`, so we never poll a
        // reaped gateway, and the poll that sees the terminal status IS the handoff (source flips live → artifact).
        const running = body.status === 'running';
        // The terminal poll is the last one, so it always refreshes the turn markers — otherwise a finished run's closing turns depend on where the round counter landed.
        if (round % CONVERSATION_POLL_EVERY === 0 || !running) {
          void loadConversations();
        }
        round += 1;
        if (running) schedule();
      } catch (e) {
        if (!active || controller.signal.aborted) return;
        // Backoff-retry keeping the last good payload on screen: one hiccup must
        // not blank a run in flight.
        degrade('retrying', (e as Error).message);
        delay = Math.min(delay * 2, MAX_BACKOFF_MS);
        schedule();
      }
    };

    void poll();
    return () => {
      active = false;
      document.removeEventListener('visibilitychange', onVisibility);
      controller.abort();
      clearTimeout(timer);
    };
  }, [taskId, instanceId]);

  return { feed, clocks, conversations };
}
