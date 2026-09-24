/**
 * Smoke test for the authored-trigger graph parser + layout.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   yarn test:smoke        (or: npx tsx src/lib/parse-triggers.smoke.ts)
 *
 * The fixture is a synthetic trigger surface shaped like a real task
 * (`task-demo-agent-triggers-a1b2c3`, via its dev clone
 * `task-demo-triggers-e2e`): 6 env triggers (permission grant,
 * nl-with-verify, two actionless state sensors, plus the two time
 * triggers — the `v6-clock-correction` after/offset form from
 * `task-demo-clock-d4e5f6` and a recurring actionless `tick`)
 * and 4 agent triggers (env_trigger-conditioned, step-conditioned, an
 * all() composite, and an end guard).
 */
import {
  humanizeDuration,
  indexAuthoredTriggers,
  parseTriggerGraph,
  summarizeWhen,
} from './parse-triggers';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

const ENV_STEP: Record<string, unknown> = {
  id: 'demo-register-triggers',
  type: 'register_env_triggers',
  env_id: 'demo-triggers-env',
  watch_roles: ['default'],
  executor_agent_name: 'executor-a1b2c3',
  executor_timeout_seconds: 180,
  triggers: [
    {
      id: 'v6-grant',
      when: { type: 'action', tool: 'gdocs_create_document' },
      actions: [
        {
          type: 'permission',
          action: 'enable',
          role: 'default',
          tools: [
            'snowflake_submit_query',
            'snowflake_get_job_status',
            'snowflake_get_job_result',
            'snowflake_query_history',
            'snowflake_list_tables',
            'snowflake_describe_table',
          ],
        },
      ],
      notify: 'warehouse access granted on first draft creation',
    },
    {
      id: 'v6-rate',
      when: { type: 'action', tool: 'gdocs_create_document' },
      actions: [
        {
          type: 'nl',
          instruction:
            "Post exactly this message in the #demo Slack channel: 'Heads up team — Acme just told us their contract art rate goes from $50/hr to $99/hr effective next month.'",
          verify: {
            tool: 'slack_conversations_history',
            args: { channel_id: 'C0000000000', limit: 10 },
            predicate: { regex: '(?i)\\$99/hr' },
          },
        },
      ],
      notify: 'vendor rate-change event realized in #demo',
    },
    {
      id: 'v6-accept-sensor',
      when: {
        type: 'state',
        check: {
          steps: [
            {
              tool: 'gdocs_search_documents',
              args: { query: 'Acme Recommendation' },
              bind: 'doc_id',
            },
            {
              tool: 'gdocs_get_document_text',
              args: { documentId: '${doc_id}' },
              predicate: { regex: '(?i)\\$99\\s*(?:/|per\\s+)h(?:ou)?r' },
            },
          ],
        },
      },
      actions: [],
      notify: 'SENSOR: doc reflects the $99/hr rate',
    },
    {
      id: 'v6-marker-sensor',
      when: {
        type: 'state',
        check: {
          steps: [
            {
              tool: 'gdocs_search_documents',
              args: { query: 'Acme Recommendation' },
              bind: 'doc_id',
            },
            {
              tool: 'gdocs_get_document_text',
              args: { documentId: '${doc_id}' },
              predicate: { regex: '(?i)synthetic[\\s-]?marker' },
            },
          ],
        },
      },
      actions: [],
      notify: 'SENSOR: doc contains an synthetic marker section',
    },
    {
      id: 'v6-clock-correction',
      when: { type: 'time', after: 'v6-grant', offset: 'PT24H' },
      actions: [
        {
          type: 'permission',
          action: 'enable',
          role: 'default',
          tools: ['snowflake_submit_query'],
        },
      ],
      notify: 'virtual-clock reaction: +24h after first draft',
    },
    {
      id: 'tick',
      when: { type: 'time', every: 'PT30S', count: 3 },
      actions: [],
      notify: 'recurring scheduled marker — actionless but NOT a sensor',
    },
  ],
};

const AGENT_STEP: Record<string, unknown> = {
  id: 'demo-register-agent-triggers',
  type: 'register_agent_triggers',
  agent_name: 'stakeholders-a1b2c3',
  triggers: [
    {
      id: 're-sweep',
      when: {
        type: 'env_trigger',
        env_id: 'demo-triggers-env',
        trigger_id: 'v6-rate',
        status: 'fired',
      },
      actions: [{ type: 'say', text: '[Dana Reed]: do a final sweep.' }],
    },
    {
      id: 'marker',
      when: { type: 'step', turn: 2, cmp: 'gte' },
      actions: [
        {
          type: 'say',
          text: `[Dana Reed]: ${'you MUST add an synthetic marker section. '.repeat(
            16,
          )}`,
        },
      ],
    },
    {
      id: 'accept',
      when: {
        type: 'all',
        of: [
          {
            type: 'env_trigger',
            env_id: 'demo-triggers-env',
            trigger_id: 'v6-accept-sensor',
            status: 'fired',
          },
          { type: 'step', turn: 2, cmp: 'gte' },
          {
            type: 'env_trigger',
            env_id: 'demo-triggers-env',
            trigger_id: 'v6-marker-sensor',
            status: 'fired',
          },
        ],
      },
      actions: [
        { type: 'say', text: '[Dana Reed]: Looks good — sending it on.' },
        { type: 'end' },
      ],
    },
    {
      id: 'wrap-guard',
      when: { type: 'step', turn: 11, cmp: 'gte' },
      actions: [{ type: 'end' }],
    },
  ],
};

const OTHER_STEPS: Array<Record<string, unknown>> = [
  { id: 'deploy', type: 'deploy_env', env_id: 'demo-triggers-env' },
  { id: 'prompt', type: 'prompt_agent' },
];

function main(): void {
  {
    const graph = parseTriggerGraph([...OTHER_STEPS, ENV_STEP, AGENT_STEP]);
    assert(graph !== null, 'graph parses');
    if (!graph) return;

    const triggers = graph.nodes.filter(n => n.type === 'trigger');
    const anchors = graph.nodes.filter(n => n.type === 'anchor');
    assert(triggers.length === 10, `10 trigger nodes (got ${triggers.length})`);
    assert(anchors.length === 2, `2 anchor nodes (got ${anchors.length})`);
    assert(graph.triggerCount === 10, 'triggerCount is 10');
    assert(graph.groups.length === 2, '2 groups');

    const envGroup = graph.groups.find(gr => gr.kind === 'env');
    assert(
      envGroup?.executorAgentName === 'executor-a1b2c3' &&
        envGroup?.executorTimeoutSeconds === 180 &&
        envGroup?.watchRoles?.join(',') === 'default',
      'env group carries watch/executor/timeout',
    );

    const envAnchor = graph.nodes.find(
      n => n.id === 'anchor-env-demo-triggers-env',
    );
    const agentAnchor = graph.nodes.find(
      n => n.id === 'anchor-agent-stakeholders-a1b2c3',
    );
    assert(
      envAnchor?.anchor?.raw.env_id === 'demo-triggers-env' &&
        envAnchor?.anchor?.triggerCount === 6,
      'env anchor carries its registration step raw + trigger count',
    );
    assert(
      agentAnchor?.anchor?.raw.agent_name === 'stakeholders-a1b2c3' &&
        agentAnchor?.anchor?.triggerCount === 4,
      'agent anchor carries its registration step raw + trigger count',
    );

    const refEdges = graph.edges.filter(e => e.kind === 'env-ref');
    assert(refEdges.length === 4, `4 env-ref edges (got ${refEdges.length})`);
    const refPairs = refEdges.map(e => `${e.source}->${e.target}`).sort();
    assert(
      refPairs.includes(
        'env:demo-triggers-env:v6-rate->agent:stakeholders-a1b2c3:re-sweep',
      ),
      're-sweep edge from v6-rate',
    );
    assert(
      refPairs.includes(
        'env:demo-triggers-env:v6-accept-sensor->agent:stakeholders-a1b2c3:accept',
      ) &&
        refPairs.includes(
          'env:demo-triggers-env:v6-marker-sensor->agent:stakeholders-a1b2c3:accept',
        ),
      'accept edges from both sensors',
    );

    const accept = triggers.find(n => n.trigger?.triggerId === 'accept');
    assert(accept?.trigger?.when.leaves.length === 3, 'accept has 3 leaves');
    assert(
      accept?.trigger?.when.leaves.map(l => l.kind).join(',') ===
        'env_trigger,step,env_trigger',
      'accept leaves preserve authored order',
    );
    assert(accept?.trigger?.when.envRefs.length === 2, 'accept has 2 envRefs');
    assert(
      (accept?.height ?? 0) >
        (triggers.find(n => n.trigger?.triggerId === 'marker')?.height ?? 0),
      'composite node is taller than a leaf node',
    );

    const sensors = triggers.filter(n => n.trigger?.isSensor);
    assert(
      sensors.length === 2 &&
        sensors.every(
          n => n.trigger?.kind === 'env' && n.trigger?.when.kind === 'state',
        ),
      'actionless triggers flagged as sensors; time-kind exempt',
    );

    const clock = triggers.find(
      n => n.trigger?.triggerId === 'v6-clock-correction',
    );
    assert(
      clock?.trigger?.when.kind === 'time' &&
        clock?.trigger?.when.label === 'after v6-grant +24h',
      `time trigger labeled from schedule (got "${clock?.trigger?.when.label}")`,
    );
    const tick = triggers.find(n => n.trigger?.triggerId === 'tick');
    assert(
      tick?.trigger?.when.label === 'every 30s ×3' &&
        tick?.trigger?.isSensor === false,
      `recurring time label + actionless-not-sensor (got "${tick?.trigger?.when.label}")`,
    );
    const afterEdge = graph.edges.find(
      e =>
        e.kind === 'env-ref' &&
        e.source === 'env:demo-triggers-env:v6-grant' &&
        e.target === 'env:demo-triggers-env:v6-clock-correction',
    );
    assert(
      Boolean(afterEdge),
      'time after-dependency draws an env-ref edge to its anchor',
    );

    const rate = triggers.find(n => n.trigger?.triggerId === 'v6-rate');
    assert(
      rate?.trigger?.referencedBy.includes('re-sweep'),
      'v6-rate referencedBy re-sweep',
    );
    assert(
      rate?.trigger?.actions[0]?.kind === 'nl' &&
        rate?.trigger?.actions[0]?.hasVerify === true,
      'v6-rate nl action carries verify',
    );
    const grant = triggers.find(n => n.trigger?.triggerId === 'v6-grant');
    assert(
      grant?.trigger?.actions[0]?.label === 'enable 6 tools' &&
        grant?.trigger?.actions[0]?.hasVerify === false,
      'v6-grant permission chip label',
    );

    const maxEnvX = Math.max(
      ...triggers.filter(n => n.trigger?.kind === 'env').map(n => n.x),
    );
    const minAgentX = Math.min(
      ...triggers.filter(n => n.trigger?.kind === 'agent').map(n => n.x),
    );
    assert(
      maxEnvX < minAgentX,
      `agent lane strictly right of env lane (${maxEnvX} < ${minAgentX})`,
    );
  }

  {
    // Unknown kinds degrade, never throw (future barrier). `time`
    // became a known kind later, so the probe uses `barrier` now.
    const graph = parseTriggerGraph([
      {
        id: 'r',
        type: 'register_env_triggers',
        env_id: 'e1',
        triggers: [
          {
            id: 'future',
            when: { type: 'barrier', name: 'b1' },
            actions: [{ type: 'barrier', name: 'b1' }],
          },
        ],
      },
    ]);
    const node = graph?.nodes.find(n => n.trigger?.triggerId === 'future');
    assert(
      node?.trigger?.when.kind === 'barrier',
      'unknown when kind preserved',
    );
    assert(
      node?.trigger?.when.label === 'barrier',
      'unknown when labeled by kind',
    );
    assert(
      node?.trigger?.actions[0]?.kind === 'barrier',
      'unknown action kind preserved',
    );
  }

  {
    // time-when label grammar + duration humanizer.
    assert(humanizeDuration('PT24H') === '24h', 'PT24H → 24h');
    assert(humanizeDuration('P1DT2H') === '1d 2h', 'P1DT2H → 1d 2h');
    assert(humanizeDuration('PT0.5S') === '0.5s', 'PT0.5S → 0.5s');
    assert(
      humanizeDuration('2026-06-01T00:00:00Z') === '2026-06-01T00:00:00Z',
      'non-duration passes through raw',
    );
    assert(
      summarizeWhen({ type: 'time', at: 'PT1H' }).label === 'at t0+1h',
      'relative at mark',
    );
    assert(
      summarizeWhen({ type: 'time', at: '2026-06-02T00:00:00Z' }).label ===
        'at 2026-06-02T00:00:00Z',
      'absolute at mark',
    );
    assert(
      summarizeWhen({
        type: 'time',
        every: { dist: 'exp', mean: 'PT5M' },
        seed: 7,
        until: 'PT2H',
      }).label === 'every ~exp(5m) until t0+2h',
      'stochastic recurrence with until',
    );
    assert(
      summarizeWhen({ type: 'time', at: 'PT1H', every: 'PT30S' }).label ===
        'at t0+1h, every 30s',
      'start + recurrence compose',
    );
  }

  {
    // Dangling env_trigger reference: node kept, edge dropped.
    const graph = parseTriggerGraph([
      {
        id: 'r',
        type: 'register_agent_triggers',
        agent_name: 'a1',
        triggers: [
          {
            id: 'orphan',
            when: {
              type: 'env_trigger',
              env_id: 'missing-env',
              trigger_id: 'nope',
              status: 'fired',
            },
            actions: [{ type: 'say', text: 'x' }],
          },
        ],
      },
    ]);
    assert(
      graph?.nodes.some(n => n.trigger?.triggerId === 'orphan'),
      'dangling ref keeps the node',
    );
    assert(
      graph?.edges.filter(e => e.kind === 'env-ref').length === 0,
      'dangling ref drops the edge',
    );
  }

  {
    assert(
      parseTriggerGraph(OTHER_STEPS) === null,
      'no trigger steps → null (section unrendered)',
    );
    assert(
      summarizeWhen(undefined).kind === 'unknown',
      'non-dict when → unknown',
    );
  }

  {
    // Composite referencing the same env trigger twice (e.g. once per
    // status): one edge, deduped referencedBy; tools:'*' labels as all.
    const graph = parseTriggerGraph([
      {
        id: 'r1',
        type: 'register_env_triggers',
        env_id: 'e1',
        triggers: [
          {
            id: 't1',
            when: { type: 'action', tool: 'x' },
            actions: [{ type: 'permission', action: 'enable', tools: '*' }],
          },
        ],
      },
      {
        id: 'r2',
        type: 'register_agent_triggers',
        agent_name: 'a1',
        triggers: [
          {
            id: 'g1',
            when: {
              type: 'any',
              of: [
                {
                  type: 'env_trigger',
                  env_id: 'e1',
                  trigger_id: 't1',
                  status: 'fired',
                },
                {
                  type: 'env_trigger',
                  env_id: 'e1',
                  trigger_id: 't1',
                  status: 'failed',
                },
              ],
            },
            actions: [{ type: 'say', text: 'x' }],
          },
        ],
      },
    ]);
    const refEdges = graph?.edges.filter(e => e.kind === 'env-ref') ?? [];
    assert(refEdges.length === 1, 'duplicate env refs dedupe to one edge');
    const t1 = graph?.nodes.find(n => n.trigger?.triggerId === 't1');
    assert(
      t1?.trigger?.referencedBy.join(',') === 'g1',
      'referencedBy deduped',
    );
    assert(
      t1?.trigger?.actions[0]?.label === 'enable all tools',
      "tools:'*' labels as all tools",
    );
  }

  {
    // Many/long action chips grow the card instead of clipping.
    const mk = (actions: unknown[]) =>
      parseTriggerGraph([
        {
          id: 'r',
          type: 'register_env_triggers',
          env_id: 'e1',
          triggers: [{ id: 'a', when: { type: 'action', tool: 'x' }, actions }],
        },
      ]);
    const short = mk([{ type: 'end' }]);
    const wide = mk([
      { type: 'tool', tool: 'snowflake_submit_query' },
      { type: 'tool', tool: 'snowflake_query_history' },
      { type: 'tool', tool: 'snowflake_describe_table' },
    ]);
    const height = (g: ReturnType<typeof parseTriggerGraph>) =>
      g?.nodes.find(n => n.type === 'trigger')?.height ?? 0;
    assert(
      height(wide) > height(short),
      'wrapping action chips grow node height',
    );
  }

  {
    // Multiple envs and agents: one anchor/group per registration step,
    // ids namespaced per group, and env refs resolve to the RIGHT env
    // even when trigger ids collide across envs.
    const graph = parseTriggerGraph([
      {
        id: 'r1',
        type: 'register_env_triggers',
        env_id: 'env-a',
        triggers: [
          { id: 'shared', when: { type: 'action', tool: 'x' }, actions: [] },
        ],
      },
      {
        id: 'r2',
        type: 'register_env_triggers',
        env_id: 'env-b',
        triggers: [
          { id: 'shared', when: { type: 'action', tool: 'y' }, actions: [] },
        ],
      },
      {
        id: 'r3',
        type: 'register_agent_triggers',
        agent_name: 'agent-1',
        triggers: [
          {
            id: 'g1',
            when: {
              type: 'env_trigger',
              env_id: 'env-b',
              trigger_id: 'shared',
              status: 'fired',
            },
            actions: [{ type: 'say', text: 'x' }],
          },
        ],
      },
      {
        id: 'r4',
        type: 'register_agent_triggers',
        agent_name: 'agent-2',
        triggers: [
          {
            id: 'g2',
            when: { type: 'step', turn: 3, cmp: 'gte' },
            actions: [{ type: 'end' }],
          },
        ],
      },
    ]);
    assert(graph?.groups.length === 4, 'four groups for 2 envs + 2 agents');
    assert(
      graph?.nodes.filter(n => n.type === 'anchor').length === 4,
      'four anchor pills',
    );
    const refs = graph?.edges.filter(e => e.kind === 'env-ref') ?? [];
    assert(
      refs.length === 1 &&
        refs[0]?.source === 'env:env-b:shared' &&
        refs[0]?.target === 'agent:agent-1:g1',
      'cross-ref resolves to the right env despite trigger-id collision',
    );
    const envA = graph?.nodes.find(n => n.id === 'env:env-a:shared');
    const envB = graph?.nodes.find(n => n.id === 'env:env-b:shared');
    assert(
      envB?.trigger?.referencedBy.join(',') === 'g1' &&
        envA?.trigger?.referencedBy.length === 0,
      'referencedBy lands only on the referenced env trigger',
    );
    const maxEnvX = Math.max(
      ...(graph?.nodes.filter(n => n.trigger?.kind === 'env').map(n => n.x) ?? [
        0,
      ]),
    );
    const minAgentX = Math.min(
      ...(graph?.nodes
        .filter(n => n.trigger?.kind === 'agent')
        .map(n => n.x) ?? [0]),
    );
    assert(maxEnvX < minAgentX, 'lanes hold with multiple groups per side');
  }

  {
    // indexAuthoredTriggers: per-trigger config index for the
    // runtime badge popovers.
    const idx = indexAuthoredTriggers([...OTHER_STEPS, ENV_STEP, AGENT_STEP]);
    assert(idx.size === 10, `authored index has 10 entries (got ${idx.size})`);

    const grant = idx.get('env:demo-triggers-env:v6-grant');
    const grantWhen = grant?.raw.when as Record<string, unknown> | undefined;
    const grantActions = grant?.raw.actions as
      | Array<Record<string, unknown>>
      | undefined;
    assert(
      grant?.kind === 'env' &&
        grant?.groupKey === 'demo-triggers-env' &&
        grantWhen?.tool === 'gdocs_create_document' &&
        grantActions?.[0]?.type === 'permission' &&
        Array.isArray(grantActions?.[0]?.tools) &&
        typeof grant?.raw.notify === 'string',
      'authored index: v6-grant raw carries full when/actions/notify',
    );
    const rateActions = idx.get('env:demo-triggers-env:v6-rate')?.raw
      .actions as Array<Record<string, unknown>> | undefined;
    assert(
      rateActions?.[0]?.type === 'nl' &&
        typeof rateActions?.[0]?.instruction === 'string' &&
        rateActions?.[0]?.verify !== undefined,
      'authored index: v6-rate raw keeps nl instruction + verify',
    );
    const sensorActions = idx.get('env:demo-triggers-env:v6-accept-sensor')
      ?.raw.actions;
    assert(
      Array.isArray(sensorActions) && sensorActions.length === 0,
      'authored index: sensor has empty actions',
    );
    const acceptWhen = idx.get('agent:accept')?.raw.when as
      | Record<string, unknown>
      | undefined;
    assert(
      acceptWhen?.type === 'all' &&
        Array.isArray(acceptWhen?.of) &&
        acceptWhen.of.length === 3,
      'authored index: composite agent trigger keeps all-of leaves',
    );
    const resweepActions = idx.get('agent:re-sweep')?.raw.actions as
      | Array<Record<string, unknown>>
      | undefined;
    assert(
      resweepActions?.[0]?.type === 'say' &&
        typeof resweepActions?.[0]?.text === 'string',
      'authored index: agent say action keeps the say text',
    );

    assert(
      indexAuthoredTriggers(OTHER_STEPS).size === 0 &&
        indexAuthoredTriggers(undefined).size === 0 &&
        indexAuthoredTriggers([null, { id: 'slim', type: 'prompt_agent' }])
          .size === 0,
      'authored index: slim/absent steps → empty index',
    );
  }

  if (failures > 0) {
    console.error(`\n${failures} assertion(s) failed`);
    process.exit(1);
  }
  console.log('\nAll parse-triggers smoke assertions passed');
}

main();
