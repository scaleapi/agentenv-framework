// Pure derivation of `materialized` for a task-runner submission. Split out of
// task-runner-page.tsx, which was ~90 KB.

import { perTurnTrajectoryUrls, trajectoryUrl } from './trajectory-url';

/** Opt-in per-instance fields. Everything else (conversation, prompt, systemPrompt, defaultModel,
 *  environmentId, snapshot id, summary) is derived from the instance record, so it's free. */
export type MaterializeOptions = {
  /** Embed each turn's raw records; costs one download per turn. */
  trajectory?: boolean;
  /** snapshotS3Urls from the run's snapshot_json_url. No-op unless the task has a snapshot_agent_state step
   *  with both env_id and universe_artifact_id. */
  snapshot_urls?: boolean;
};

/** A trajectory record in the harness's own format — deliberately opaque here. */
type RawRecord = unknown;

type Str = (v: unknown) => string | null;
type StepLookup = (
  resp: Record<string, unknown>,
) => Record<string, unknown> | undefined;

/** A2A parts -> text. Parts are `{kind:'text',text}`; cf. prompt_agent.py:439. */
function partsText(parts: unknown): string | null {
  if (!Array.isArray(parts)) return null;
  const texts = parts
    .filter(
      (p): p is Record<string, unknown> =>
        !!p && typeof p === 'object' && (p as Record<string, unknown>).kind === 'text',
    )
    .map(p => p.text)
    .filter((t): t is string => typeof t === 'string' && t !== '');
  return texts.length ? texts.join('\n\n') : null;
}

/** The capture with prompts prepended: per turn system?, user, then that turn's records. Prompts go in
 *  FRONT (every harness parser detects on records[0]); `turn` counts exchanges flat. */
function buildConversation(
  responses: Record<string, unknown>[],
  stepFor: StepLookup,
  harnessFor: (resp: Record<string, unknown>) => unknown,
  str: Str,
  recordsFor?: (uri: string | null) => RawRecord[] | undefined,
): RawRecord[] {
  const convo: RawRecord[] = [];
  let turn = 0;
  let prevSys: string | null = null;

  for (const resp of responses) {
    const sys = str(stepFor(resp)?.system_prompt);
    const base = {
      promptId: str(resp.prompt_id),
      stepId: str(resp.step_id),
      agentName: str(resp.agent_name),
      model: str(resp.model),
      // What wrote the records that follow ('claude-code-cli' | 'gemini-cli' | …),
      // from the deploy_agent step's `a2a_agent_id` — config, not a sniff.
      harness: str(harnessFor(resp)),
    };

    const pushTurn = (
      ask: string | null,
      answer: string | null,
      uri: string | null,
      toolCalls: number,
    ) => {
      // Only when it changes, so a multi-KB prompt is not repeated per turn.
      if (sys && sys !== prevSys) {
        convo.push({ turn, role: 'system', content: sys });
        prevSys = sys;
      }
      convo.push({
        turn,
        role: 'user',
        content: ask,
        ...base,
        toolCallCount: toolCalls,
        trajectoryS3Uri: uri,
      });
      const records = recordsFor?.(uri);
      if (records?.length) {
        convo.push(...records);
      } else if (answer) {
        convo.push({ turn, role: 'assistant', content: answer });
      }
      turn += 1;
    };

    const innerUris = perTurnTrajectoryUrls(resp) ?? null;
    if (innerUris?.length) {
      const innerParts = Array.isArray(resp.source_agent_per_turn_prompt_parts)
        ? (resp.source_agent_per_turn_prompt_parts as unknown[])
        : null;
      const last = innerUris.length - 1;
      innerUris.forEach((uri, k) => {
        pushTurn(
          partsText(innerParts?.[k]) ?? (k === 0 ? str(resp.prompt_text) : null),
          // Only the final inner turn's reply text is on the record; the earlier
          // ones exist only inside their own capture.
          k === last ? str(resp.response) : null,
          str(uri),
          k === last && typeof resp.tool_call_count === 'number'
            ? resp.tool_call_count
            : 0,
        );
      });
    } else {
      pushTurn(
        str(resp.prompt_text),
        str(resp.response),
        str(trajectoryUrl(resp)),
        typeof resp.tool_call_count === 'number' ? resp.tool_call_count : 0,
      );
    }
  }
  return convo;
}

/** Derive an OpenClaw-`trajectories`-shaped entry from a finished instance, so a consumer of that map works against a runner result. */
export function materializeInstance(
  inst: Record<string, unknown>,
  opts: MaterializeOptions | undefined,
  trajectoryCache?: Map<string, unknown[]>,
  taskStepsCache?: Map<number, Record<string, unknown>[]>,
): Record<string, unknown> {
  const ctx = (inst.context ?? {}) as Record<string, unknown>;
  const meta = (ctx.metadata ?? {}) as Record<string, unknown>;
  const responses = Array.isArray(ctx.prompt_responses)
    ? (ctx.prompt_responses as Record<string, unknown>[])
    : [];
  // Last response: within one instance these are conversation turns, so the
  // newest is the one a consumer means.
  const r = responses[responses.length - 1] ?? {};
  const envs = Array.isArray(ctx.deployed_envs)
    ? (ctx.deployed_envs as Record<string, unknown>[])
    : [];
  const snaps = Array.isArray(meta.agent_snapshots)
    ? (meta.agent_snapshots as Record<string, unknown>[])
    : [];
  const snap = snaps[snaps.length - 1] ?? {};
  const str = (v: unknown): string | null =>
    typeof v === 'string' && v !== '' ? v : null;

  // `prompt_text` is on the response; `system_prompt` is not, so it comes off the
  // prompt_agent step of the version THIS INSTANCE RAN (else it drifts on edit).
  const ver = typeof inst.task_version === 'number' ? inst.task_version : null;
  const versionSteps = ver !== null ? taskStepsCache?.get(ver) : undefined;
  // Resolve a response back to the prompt_agent step that drove it. `step_id` is
  // the exact link when present; `prompt_id` is the fallback for older records.
  const stepFor = (resp: Record<string, unknown>) =>
    (versionSteps ?? []).find(
      st =>
        st.type === 'prompt_agent' &&
        (resp.step_id
          ? st.id === resp.step_id
          : !resp.prompt_id || st.id === resp.prompt_id),
    );
  // Harness comes from deploy_agent's `a2a_agent_id` — config, not a record sniff.
  const harnessFor = (resp: Record<string, unknown>) =>
    (versionSteps ?? []).find(
      st =>
        st.type === 'deploy_agent' &&
        (!resp.agent_name || st.agent_name === resp.agent_name),
    )?.a2a_agent_id;
  const promptStep = stepFor(r);

  const out: Record<string, unknown> = {
    // Our own schema, not a harness passthrough — bump when `turns[]` changes shape
    // so a consumer can tell a pre-`turns` payload from this one.
    schema: 'agent-env-explorer.materialized/1',
    prompt: str(r.prompt_text),
    systemPrompt: str(promptStep?.system_prompt),
    defaultModel: str(r.model),
    environmentId: str((envs[0] ?? {}).env_id),
    agentEnvSnapshotArtifactId: str(snap.id),
    agentEnvSnapshotArtifactVersion:
      typeof snap.version === 'number' ? snap.version : null,
    // `trajectory` is our own turn-labelled, role-shaped schema, NOT a passthrough of the harness capture —
    // that's what makes prompts recoverable (prepending a record to raw capture reroutes it to the wrong
    // parser, since each harness detects on record shape). Every message carries `turn`; records are embedded
    // only when asked for and cached, else the `response` text.
    trajectory: buildConversation(
      responses,
      stepFor,
      harnessFor,
      str,
      opts?.trajectory
        ? uri => (uri ? trajectoryCache?.get(uri) : undefined)
        : undefined,
    ),
    summary: {
      turnCount: responses.length,
      toolCallCount:
        typeof r.tool_call_count === 'number' ? r.tool_call_count : 0,
      instanceId: str(inst.instance_id),
    },
  };

  if (opts?.trajectory) {
    // Pure per-turn captures for anything that runs a parser. A turn whose records haven't arrived is OMITTED
    // (not []), so "not fetched" is distinguishable from "empty"; a later submission fills it in.
    const records = responses.flatMap((resp, i) => {
      const uri = str(trajectoryUrl(resp));
      const recs = uri ? trajectoryCache?.get(uri) : undefined;
      return recs ? [[i, recs] as const] : [];
    });
    if (records.length > 0) out.trajectoryRecords = Object.fromEntries(records);
  }


  if (opts?.snapshot_urls) {
    // snapshot_json_url is a JSON *string* of { service: "s3://…" }, not an object — a malformed value must not take the submission down.
    const raw = meta.snapshot_json_url;
    if (typeof raw === 'string' && raw !== '') {
      try {
        const parsed: unknown = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          out.snapshotS3Urls = parsed;
        }
      } catch {
        // Leave the key absent rather than emitting a quoted blob a consumer
        // would have to re-parse.
      }
    }
  }

  return out;
}