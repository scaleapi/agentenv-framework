/** Runtime trigger evidence model for the instance viewer's Triggers tab. Parses the trigger ledger the
 *  worker writes into context.metadata: env/agent registrations, env_trigger_state (end-of-run per-env summary
 *  or {error}), agent_trigger_firings, env_trigger_snapshots/usersim_turn_outputs (per-turn, absent on older
 *  runs), agent_trigger_state (best-effort readback). Turns are 1-based reaction turns. Null when metadata
 *  carries no trigger keys. */

export interface TriggerRegistrationRow {
  kind: 'env' | 'agent';
  groupKey: string;
  stepId: string;
  added: string[];
  executorAgentName?: string;
}

/** Per-trigger capture projection (env triggers only). A recurring time trigger re-arms after every arrival,
 *  so `fireCount` is its only progress signal; `nextMark` is the next due instant. Absent on older ledgers. */
export interface TriggerCapture {
  type?: string;
  status?: string;
  fireCount?: number;
  nextMark?: string;
}

export interface EnvStateSummary {
  envId: string;
  statuses: Record<string, string>;
  triggers?: Record<string, TriggerCapture>;
  /** False = the record could still change (a trigger mid-firing or an
   *  armed time trigger); None/absent = older gateway, unknowable. */
  captureIsFinal?: boolean;
  /** Events evicted from the gateway's head+tail log; `eventCount`
   *  included on newer gateways. */
  eventsDropped?: number;
  eventCount?: number;
  objectUrl?: string;
  capturedAtUtc?: string;
  error?: string;
}

export interface TurnDelta {
  envId: string;
  triggerId: string;
  /** null on the first snapshot — renders as the initial status. */
  from: string | null;
  to: string;
}

export interface TurnRow {
  turn: number;
  fired: string[];
  deltas: TurnDelta[];
  hasSnapshot: boolean;
  usersim?: Record<string, unknown>;
}

export interface StepTriggerRuntime {
  stepId: string;
  turns: TurnRow[];
  firingLogAvailable: boolean;
  hasPerTurnData: boolean;
}

export interface TriggerRuntime {
  registrations: TriggerRegistrationRow[];
  envState: EnvStateSummary[];
  steps: StepTriggerRuntime[];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asStatusMap(value: unknown): Record<string, string> {
  const rec = asRecord(value);
  if (!rec) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(rec)) out[k] = String(v);
  return out;
}

function parseRegistrations(
  metadata: Record<string, unknown>,
): TriggerRegistrationRow[] {
  const rows: TriggerRegistrationRow[] = [];
  const env = metadata.env_trigger_registrations;
  if (Array.isArray(env)) {
    for (const r of env) {
      const rec = asRecord(r);
      if (!rec) continue;
      rows.push({
        kind: 'env',
        groupKey: String(rec.env_id ?? ''),
        stepId: String(rec.step_id ?? ''),
        added: Array.isArray(rec.added) ? rec.added.map(String) : [],
        executorAgentName:
          rec.executor_agent_name != null
            ? String(rec.executor_agent_name)
            : undefined,
      });
    }
  }
  const agent = metadata.agent_trigger_registrations;
  if (Array.isArray(agent)) {
    for (const r of agent) {
      const rec = asRecord(r);
      if (!rec) continue;
      rows.push({
        kind: 'agent',
        groupKey: String(rec.agent_name ?? ''),
        stepId: String(rec.step_id ?? ''),
        added: Array.isArray(rec.added) ? rec.added.map(String) : [],
      });
    }
  }
  return rows;
}

function parseTriggerCaptures(
  value: unknown,
): Record<string, TriggerCapture> | undefined {
  const rec = asRecord(value);
  if (!rec) return undefined;
  const out: Record<string, TriggerCapture> = {};
  for (const [id, entry] of Object.entries(rec)) {
    const t = asRecord(entry);
    if (!t) continue;
    out[id] = {
      type: t.type != null ? String(t.type) : undefined,
      status: t.status != null ? String(t.status) : undefined,
      fireCount: typeof t.fire_count === 'number' ? t.fire_count : undefined,
      nextMark: t.next_mark != null ? String(t.next_mark) : undefined,
    };
  }
  return out;
}

function parseEnvState(metadata: Record<string, unknown>): EnvStateSummary[] {
  const state = asRecord(metadata.env_trigger_state);
  if (!state) return [];
  return Object.entries(state).map(([envId, entry]) => {
    const rec = asRecord(entry) ?? {};
    return {
      envId,
      statuses: asStatusMap(rec.statuses),
      triggers: parseTriggerCaptures(rec.triggers),
      captureIsFinal:
        typeof rec.capture_is_final === 'boolean'
          ? rec.capture_is_final
          : undefined,
      eventsDropped:
        typeof rec.events_dropped === 'number' && rec.events_dropped > 0
          ? rec.events_dropped
          : undefined,
      eventCount:
        typeof rec.event_count === 'number' ? rec.event_count : undefined,
      objectUrl: rec.object_url != null ? String(rec.object_url) : undefined,
      capturedAtUtc:
        rec.captured_at_utc != null ? String(rec.captured_at_utc) : undefined,
      error: rec.error != null ? String(rec.error) : undefined,
    };
  });
}

function snapshotDeltas(
  prev: Record<string, Record<string, string>> | null,
  curr: Record<string, Record<string, string>>,
): TurnDelta[] {
  const deltas: TurnDelta[] = [];
  for (const [envId, statuses] of Object.entries(curr)) {
    for (const [triggerId, to] of Object.entries(statuses)) {
      const from = prev?.[envId]?.[triggerId] ?? null;
      if (from !== to) deltas.push({ envId, triggerId, from, to });
    }
  }
  return deltas;
}

function parseSteps(metadata: Record<string, unknown>): StepTriggerRuntime[] {
  const firings = asRecord(metadata.agent_trigger_firings) ?? {};
  const snapshots = asRecord(metadata.env_trigger_snapshots) ?? {};
  const usersim = asRecord(metadata.usersim_turn_outputs) ?? {};
  const firingLog = asRecord(metadata.agent_trigger_state) ?? {};

  const stepIds = [
    ...new Set([
      ...Object.keys(firings),
      ...Object.keys(snapshots),
      ...Object.keys(usersim),
      ...Object.keys(firingLog),
    ]),
  ].sort();

  return stepIds.map(stepId => {
    const firedByTurn = new Map<number, string[]>();
    const stepFirings = firings[stepId];
    if (Array.isArray(stepFirings)) {
      for (const f of stepFirings) {
        const rec = asRecord(f);
        if (!rec || typeof rec.turn !== 'number') continue;
        firedByTurn.set(
          rec.turn,
          Array.isArray(rec.fired) ? rec.fired.map(String) : [],
        );
      }
    }

    const snapshotByTurn = new Map<
      number,
      Record<string, Record<string, string>>
    >();
    const stepSnapshots = snapshots[stepId];
    if (Array.isArray(stepSnapshots)) {
      for (const s of stepSnapshots) {
        const rec = asRecord(s);
        if (!rec || typeof rec.turn !== 'number') continue;
        const envs = asRecord(rec.envs) ?? {};
        const parsed: Record<string, Record<string, string>> = {};
        for (const [envId, statuses] of Object.entries(envs))
          parsed[envId] = asStatusMap(statuses);
        snapshotByTurn.set(rec.turn, parsed);
      }
    }

    const usersimByTurn = new Map<number, Record<string, unknown>>();
    const stepUsersim = usersim[stepId];
    if (Array.isArray(stepUsersim)) {
      for (const u of stepUsersim) {
        const rec = asRecord(u);
        if (!rec || typeof rec.turn !== 'number') continue;
        const fields = asRecord(rec.fields);
        if (fields) usersimByTurn.set(rec.turn, fields);
      }
    }

    const turnNumbers = [
      ...new Set([
        ...firedByTurn.keys(),
        ...snapshotByTurn.keys(),
        ...usersimByTurn.keys(),
      ]),
    ].sort((a, b) => a - b);

    let prevSnapshot: Record<string, Record<string, string>> | null = null;
    const turns: TurnRow[] = turnNumbers.map(turn => {
      const snapshot = snapshotByTurn.get(turn);
      const deltas = snapshot ? snapshotDeltas(prevSnapshot, snapshot) : [];
      if (snapshot) prevSnapshot = snapshot;
      return {
        turn,
        fired: firedByTurn.get(turn) ?? [],
        deltas,
        hasSnapshot: snapshot !== undefined,
        usersim: usersimByTurn.get(turn),
      };
    });

    return {
      stepId,
      turns,
      firingLogAvailable: stepId in firingLog,
      hasPerTurnData: snapshotByTurn.size > 0 || usersimByTurn.size > 0,
    };
  });
}

export function parseTriggerRuntime(
  metadata: Record<string, unknown> | null | undefined,
): TriggerRuntime | null {
  const md = asRecord(metadata);
  if (!md) return null;
  const registrations = parseRegistrations(md);
  const envState = parseEnvState(md);
  const steps = parseSteps(md);
  if (registrations.length === 0 && envState.length === 0 && steps.length === 0)
    return null;
  return { registrations, envState, steps };
}

/** Join for the per-turn trigger strips: the ledger row for one step's 1-based turn, or null when there's
 *  nothing to render. A first-snapshot `armed` (from=null→'armed') is registration state, not activity, and is
 *  dropped; a first-snapshot non-armed status is a genuine turn-1 transition and kept. */
export function turnRowFor(
  runtime: TriggerRuntime | null,
  stepId: string | undefined,
  reactionTurn: number | undefined,
): TurnRow | null {
  if (!runtime || !stepId || reactionTurn === undefined) return null;
  const step = runtime.steps.find(s => s.stepId === stepId);
  const row = step?.turns.find(t => t.turn === reactionTurn);
  if (!row) return null;
  const deltas = row.deltas.filter(d => d.from !== null || d.to !== 'armed');
  if (deltas.length === 0 && row.fired.length === 0 && !row.usersim)
    return null;
  return { ...row, deltas };
}

export interface TriggerHistory {
  /** Every recorded status change, unfiltered — includes the initial
   *  from=null entry, across all prompt steps in turn order. */
  trail: { turn: number; from: string | null; to: string }[];
  firedTurns: number[];
  /** End-of-run capture (env triggers only). */
  finalStatus?: string;
  /** Capture projection (env triggers only): cumulative fires and the next due instant — the only progress signals a re-arming recurrence has. */
  fireCount?: number;
  nextMark?: string;
  registeredBy?: string;
}

/** One trigger's runtime story for the badge popovers: status trail from snapshots, firing turns from the decide log, final status, registering step. */
export function triggerHistoryFor(
  runtime: TriggerRuntime,
  kind: 'env' | 'agent',
  triggerId: string,
  envId?: string,
): TriggerHistory {
  const trail: TriggerHistory['trail'] = [];
  const firedTurns: number[] = [];
  for (const step of runtime.steps) {
    for (const t of step.turns) {
      if (kind === 'env') {
        for (const d of t.deltas) {
          if (d.triggerId === triggerId && (!envId || d.envId === envId))
            trail.push({ turn: t.turn, from: d.from, to: d.to });
        }
      } else if (t.fired.includes(triggerId)) {
        firedTurns.push(t.turn);
      }
    }
  }
  const envSummary =
    kind === 'env'
      ? runtime.envState.find(s => !envId || s.envId === envId)
      : undefined;
  const capture = envSummary?.triggers?.[triggerId];
  const registration = runtime.registrations.find(
    r =>
      r.kind === kind &&
      r.added.includes(triggerId) &&
      (kind === 'agent' || !envId || r.groupKey === envId),
  );
  return {
    trail,
    firedTurns,
    finalStatus: envSummary?.statuses[triggerId],
    fireCount: capture?.fireCount,
    nextMark: capture?.nextMark,
    registeredBy: registration?.stepId,
  };
}
