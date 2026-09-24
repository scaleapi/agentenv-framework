/**
 * Smoke test for the event-level trigger timeline parser.
 *
 * Runner: yarn test:smoke (auto-discovered), or:
 *   NODE_OPTIONS='--require ../../.pnp.cjs' npx tsx src/lib/parse-trigger-events.smoke.ts
 *
 * Fixture: a synthetic response in the shape of a completed instance
 * `task-demo-clock-d4e5f6-0a1b2c3d4e5f6a7b` — 5 env triggers spanning
 * action/state/time, 16 events including the anchored clock chain, 4 agent
 * firing-log events, and the 23-message conversation. Message text is truncated
 * to 70 chars; everything else is verbatim.
 */
import {
  highlightFor,
  parseTriggerTimeline,
  shortStamp,
  timelineFingerprint,
  type TriggerTimeline,
} from './parse-trigger-events';
import { countdownTo, humanizeMs, parseEnvClocks } from './trigger-clock';
import type { AuthoredTriggerDetail } from './parse-triggers';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

const ENV = 'demo-triggers-env';

const STATE: unknown = {
  'demo-triggers-env': {
    config: {
      watch_roles: ['default'],
      executor_configured: true,
    },
    triggers: [
      {
        id: 'v6-grant',
        type: 'action',
        when: {
          type: 'action',
          tool: 'gdocs_create_document',
        },
        status: 'fired',
        detected_at: '2026-08-06T17:58:01.523464+00:00',
        fired_at: '2026-08-06T17:58:01.524098+00:00',
        fire_count: 1,
        next_mark: null,
      },
      {
        id: 'v6-rate',
        type: 'action',
        when: {
          type: 'action',
          tool: 'gdocs_create_document',
        },
        status: 'fired',
        detected_at: '2026-08-06T17:58:01.523545+00:00',
        fired_at: '2026-08-06T17:58:11.229516+00:00',
        fire_count: 1,
        next_mark: null,
      },
      {
        id: 'v6-accept-sensor',
        type: 'state',
        when: {
          type: 'state',
        },
        status: 'armed',
        detected_at: null,
        fired_at: null,
        fire_count: 0,
        next_mark: null,
      },
      {
        id: 'v6-marker-sensor',
        type: 'state',
        when: {
          type: 'state',
        },
        status: 'armed',
        detected_at: null,
        fired_at: null,
        fire_count: 0,
        next_mark: null,
      },
      {
        id: 'v6-clock-correction',
        type: 'time',
        when: {
          type: 'time',
          after: 'v6-grant',
          offset: 'PT24H',
        },
        status: 'fired',
        detected_at: '2026-08-06T17:58:03.123816+00:00',
        fired_at: '2026-08-06T17:58:03.125008+00:00',
        fire_count: 1,
        next_mark: null,
      },
    ],
    events: [
      {
        seq: 1,
        ts: '2026-08-06T17:48:16.613273+00:00',
        virtual_time: '2026-12-03T07:51:29.490278Z',
        kind: 'added',
        trigger_id: 'v6-grant',
      },
      {
        seq: 2,
        ts: '2026-08-06T17:48:16.613390+00:00',
        virtual_time: '2026-12-03T07:51:40.262803Z',
        kind: 'added',
        trigger_id: 'v6-rate',
      },
      {
        seq: 3,
        ts: '2026-08-06T17:48:16.613428+00:00',
        virtual_time: '2026-12-03T07:51:43.562333Z',
        kind: 'added',
        trigger_id: 'v6-accept-sensor',
      },
      {
        seq: 4,
        ts: '2026-08-06T17:48:16.613465+00:00',
        virtual_time: '2026-12-03T07:51:46.734595Z',
        kind: 'added',
        trigger_id: 'v6-marker-sensor',
      },
      {
        seq: 5,
        ts: '2026-08-06T17:48:16.613495+00:00',
        virtual_time: '2026-12-03T07:51:49.362710Z',
        kind: 'added',
        trigger_id: 'v6-clock-correction',
      },
      {
        seq: 6,
        ts: '2026-08-06T17:58:01.523473+00:00',
        virtual_time: '2028-07-10T05:42:11.493101Z',
        kind: 'detected',
        trigger_id: 'v6-grant',
        provoking: {
          tool: 'gdocs_create_document',
          role: 'default',
        },
      },
      {
        seq: 7,
        ts: '2026-08-06T17:58:01.523552+00:00',
        virtual_time: '2028-07-10T05:42:18.212256Z',
        kind: 'detected',
        trigger_id: 'v6-rate',
        provoking: {
          tool: 'gdocs_create_document',
          role: 'default',
        },
      },
      {
        seq: 8,
        ts: '2026-08-06T17:58:01.524044+00:00',
        virtual_time: '2028-07-10T05:43:00.663600Z',
        kind: 'action_ok',
        trigger_id: 'v6-grant',
        action_index: 0,
        detail: {
          role: 'default',
          action: 'enable',
          tools: [
            'snowflake_submit_query',
            'snowflake_get_job_status',
            'snowflake_get_job_result',
            'snowflake_query_history',
            'snowflake_list_tables',
            'snowflake_describe_table',
          ],
        },
        changelog_id_before: 1,
        changelog_id_after: 1,
      },
      {
        seq: 9,
        ts: '2026-08-06T17:58:01.524104+00:00',
        virtual_time: '2028-07-10T05:43:05.962858Z',
        kind: 'fired',
        trigger_id: 'v6-grant',
        fire_count: 1,
      },
      {
        seq: 10,
        ts: '2026-08-06T17:58:01.524164+00:00',
        virtual_time: '2028-07-10T05:43:11.220038Z',
        kind: 'anchored',
        trigger_id: 'v6-clock-correction',
        anchor: 'v6-grant',
        mark: '2028-07-11T05:43:08.912294Z',
      },
      {
        seq: 11,
        ts: '2026-08-06T17:58:03.123848+00:00',
        virtual_time: '2028-07-11T20:06:43.844544Z',
        kind: 'detected',
        trigger_id: 'v6-clock-correction',
        provoking: {
          source: 'clock',
        },
        due: 1,
        capped: false,
        first_mark: '2028-07-11T05:43:08.912294Z',
        last_mark: '2028-07-11T05:43:08.912294Z',
      },
      {
        seq: 12,
        ts: '2026-08-06T17:58:03.124950+00:00',
        virtual_time: '2028-07-11T20:08:18.893270Z',
        kind: 'action_ok',
        trigger_id: 'v6-clock-correction',
        action_index: 0,
        detail: {
          role: 'default',
          action: 'enable',
          tools: ['snowflake_submit_query'],
        },
        changelog_id_before: 1,
        changelog_id_after: 1,
      },
      {
        seq: 13,
        ts: '2026-08-06T17:58:03.125016+00:00',
        virtual_time: '2028-07-11T20:08:24.792749Z',
        kind: 'fired',
        trigger_id: 'v6-clock-correction',
        fire_count: 1,
        mark: '2028-07-11T05:43:08.912294Z',
        capped: false,
      },
      {
        seq: 14,
        ts: '2026-08-06T17:58:11.228895+00:00',
        virtual_time: '2028-07-19T22:37:59.320070Z',
        kind: 'verify_ok',
        trigger_id: 'v6-rate',
        action_index: 0,
      },
      {
        seq: 15,
        ts: '2026-08-06T17:58:11.229453+00:00',
        virtual_time: '2028-07-19T22:38:47.876093Z',
        kind: 'action_ok',
        trigger_id: 'v6-rate',
        action_index: 0,
        detail: {
          nl: "Post exactly this message in the #demo Slack channel: 'Heads up team — Acme just told us their contract art ra",
        },
        changelog_id_before: 1,
        changelog_id_after: 2,
      },
      {
        seq: 16,
        ts: '2026-08-06T17:58:11.229523+00:00',
        virtual_time: '2028-07-19T22:38:54.150374Z',
        kind: 'fired',
        trigger_id: 'v6-rate',
        fire_count: 1,
      },
    ],
    events_dropped: 0,
  },
};

const STATE_META: unknown = {
  'demo-triggers-env': {
    clock: {
      armed: true,
      t0: '2026-06-01T00:00:00Z',
      virtual_seconds_per_real_second: 86400.0,
      virtual_time: '2031-10-22T14:04:54.968880Z',
      env_get_time_url: 'http://gateway:18765/clock/time',
    },
    clock_read_at_utc: '2026-08-06T18:18:00.845351+00:00',
  },
};

const AGENT_STATE: unknown = {
  'task-demo-agent-triggers-d4e5f6-prompt': [
    {
      seq: 1,
      ts: '2026-08-06T17:48:17.029101+00:00',
      kind: 'registered',
      detail: {
        added: ['re-sweep', 'marker', 'accept', 'wrap-guard'],
      },
    },
    {
      seq: 2,
      ts: '2026-08-06T17:56:37.957797+00:00',
      kind: 'fired',
      trigger_id: 'marker',
      turn: 2,
      context_id: '6d1fbbfd92b6438e9cf387e11f8ecf98',
    },
    {
      seq: 3,
      ts: '2026-08-06T18:00:06.736367+00:00',
      kind: 'fired',
      trigger_id: 're-sweep',
      turn: 4,
      context_id: '6d1fbbfd92b6438e9cf387e11f8ecf98',
    },
    {
      seq: 4,
      ts: '2026-08-06T18:18:00.173268+00:00',
      kind: 'fired',
      trigger_id: 'wrap-guard',
      turn: 11,
      context_id: '6d1fbbfd92b6438e9cf387e11f8ecf98',
    },
  ],
};

const CONVERSATIONS: unknown = [
  {
    conversation_id: '6d1fbbfd92b6438e9cf387e11f8ecf98',
    messages: [
      {
        role: 'user',
        ts: '2026-08-06T17:48:17.719119+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Dana Reed]: Can you look into Acme and let me know if we shou',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T17:50:06.130462+00:00',
        parts: [
          {
            kind: 'text',
            text: "I now have a thorough picture. Here's my full assessment, Dana:\n\n---\n",
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T17:50:30.656922+00:00',
        parts: [
          {
            kind: 'text',
            text: "[Dana Reed]: Okay, that's a lot of narrative but it's not the quest",
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T17:56:36.023076+00:00',
        parts: [
          {
            kind: 'text',
            text: 'I now have everything I need. Let me compile the full picture.\n\n---\n\n*',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T17:56:37.964208+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Dana Reed]: One more thing before this goes to Robin — you MUST a',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T17:56:58.559222+00:00',
        parts: [
          {
            kind: 'text',
            text: 'Dana, I have to stop you before this goes any further — I need to be ',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T17:57:20.279085+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Sam Patel]: IT is provisioning your warehouse access — it usually',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:00:05.165946+00:00',
        parts: [
          {
            kind: 'text',
            text: 'Doc skeleton is created: **"Acme Vendor Assessment — Northwind Delig',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:00:06.743052+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Dana Reed]: Before this goes to Robin — vendor news just landed i',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:01:30.529060+00:00',
        parts: [
          {
            kind: 'text',
            text: "All 5 replacements landed cleanly. Here's what the sweep found and wha",
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:02:35.438413+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Dana Reed]: Good — the $99/hr figure is exactly the kind of thing ',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:08:39.537534+00:00',
        parts: [
          {
            kind: 'text',
            text: "All 7 replacements landed. The doc is now complete. Here's what went i",
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:09:26.755657+00:00',
        parts: [
          {
            kind: 'text',
            text: "[Dana Reed]: One more thing before Robin sees it — he won't review",
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:13:03.962678+00:00',
        parts: [
          {
            kind: 'text',
            text: 'Dana — two things before I keep pushing on the vendor cap:\n\n**Section',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:13:19.607884+00:00',
        parts: [
          {
            kind: 'text',
            text: "[Dana Reed]: Fair enough on 5.2 — glad it's already there with the ",
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:15:17.352529+00:00',
        parts: [
          {
            kind: 'text',
            text: 'Done. Section 5.3 now reads:\n\n> **5.3 Budget Fit**\n> Acme previou',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:15:46.236413+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Dana Reed]: Budget Fit reads right now — inside the $10K cap, clea',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:16:06.866411+00:00',
        parts: [
          {
            kind: 'text',
            text: 'Neither is in the doc. I just read the full live text — the document e',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:16:19.600168+00:00',
        parts: [
          {
            kind: 'text',
            text: '[Dana Reed]: Those angles are exactly right — rate step-up, the Oct',
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:17:38.877834+00:00',
        parts: [
          {
            kind: 'text',
            text: "Both are in and correct. Here's what the live doc now shows at those t",
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:17:54.041456+00:00',
        parts: [
          {
            kind: 'text',
            text: "[Dana Reed]: That's the shape I needed — Risks, Cost Comparison, In",
          },
        ],
      },
      {
        role: 'agent',
        ts: '2026-08-06T18:17:58.893815+00:00',
        parts: [
          {
            kind: 'text',
            text: 'Understood — nothing to Acme, Robin owns that. Good luck with th',
          },
        ],
      },
      {
        role: 'user',
        ts: '2026-08-06T18:18:00.170469+00:00',
        parts: [
          {
            kind: 'text',
            text: '',
          },
        ],
      },
    ],
  },
];

/** The two agent triggers whose authored `when` references an env trigger. */
const AUTHORED = new Map<string, AuthoredTriggerDetail>([
  [
    'agent:re-sweep',
    {
      kind: 'agent',
      triggerId: 're-sweep',
      groupKey: 'stakeholders-d4e5f6',
      raw: {
        id: 're-sweep',
        when: {
          type: 'env_trigger',
          env_id: ENV,
          trigger_id: 'v6-rate',
          status: 'fired',
        },
      },
    },
  ],
  [
    'agent:accept',
    {
      kind: 'agent',
      triggerId: 'accept',
      groupKey: 'stakeholders-d4e5f6',
      raw: {
        id: 'accept',
        when: {
          type: 'all',
          of: [
            {
              type: 'env_trigger',
              env_id: ENV,
              trigger_id: 'v6-accept-sensor',
              status: 'fired',
            },
            { type: 'step', turn: 2, cmp: 'gte' },
            {
              type: 'env_trigger',
              env_id: ENV,
              trigger_id: 'v6-marker-sensor',
              status: 'fired',
            },
          ],
        },
      },
    },
  ],
]);

function parseAll(): TriggerTimeline {
  const timeline = parseTriggerTimeline({
    state: STATE,
    stateMeta: STATE_META,
    agentTriggerState: AGENT_STATE,
    conversations: CONVERSATIONS,
    authored: AUTHORED,
  });
  if (!timeline) throw new Error('fixture must produce a timeline');
  return timeline;
}

function main(): void {
  // Every persisted event reaches the timeline, across all three streams.
  {
    const t = parseAll();
    const env = t.rows.filter(r => r.stream === 'env');
    const agent = t.rows.filter(r => r.stream === 'agent');
    const conv = t.rows.filter(r => r.stream === 'conversation');
    assert(env.length === 16, `16 env event rows (got ${env.length})`);
    assert(agent.length === 4, `4 agent event rows (got ${agent.length})`);
    assert(conv.length === 23, `23 conversation rows (got ${conv.length})`);
    assert(
      t.envIds.length === 1 && t.envIds[0] === ENV,
      'single env id reported',
    );
    assert(t.eventsDropped === 0, 'nothing evicted on this run');
    assert(
      t.rows.every(r => r.kind !== 'gap'),
      'no truncation divider when seq is contiguous',
    );
  }

  // All 12 known kinds route through a specific summary, never the raw kind.
  {
    const t = parseAll();
    const kinds = new Set(
      t.rows.filter(r => r.stream === 'env').map(r => r.kind),
    );
    for (const kind of [
      'added',
      'detected',
      'action_ok',
      'fired',
      'anchored',
      'verify_ok',
    ]) {
      assert(kinds.has(kind), `fixture exercises ${kind}`);
    }
    const summaries = t.rows
      .filter(r => r.stream === 'env')
      .map(r => r.summary);
    assert(
      summaries.every(
        (s, i) => s !== t.rows.filter(r => r.stream === 'env')[i]?.kind,
      ),
      'no env row falls back to bare kind text',
    );
  }

  // Ordering: streams interleave on ts, and each stream stays monotonic in seq.
  {
    const t = parseAll();
    const envSeqs = t.rows.filter(r => r.stream === 'env').map(r => r.seq);
    assert(
      envSeqs.every((s, i) => i === 0 || (s ?? 0) > (envSeqs[i - 1] ?? 0)),
      'env rows stay in seq order after the merge',
    );
    const streams = t.rows.map(r => r.stream);
    assert(
      new Set(streams).size === 3,
      'all three streams interleave in one list',
    );
    // v6-rate fires at 17:58:11; the re-sweep agent firing lands at 18:00:06.
    const rate = t.rows.findIndex(
      r => r.groupKey === `env:${ENV}:v6-rate` && r.kind === 'fired',
    );
    const sweep = t.rows.findIndex(
      r => r.stream === 'agent' && r.triggerId === 're-sweep',
    );
    assert(
      rate >= 0 && sweep > rate,
      'env v6-rate fired sorts before the re-sweep agent firing',
    );
  }

  // The nl executor emits verify_ok *before* action_ok — same action block.
  {
    const t = parseAll();
    const block = t.rows.filter(
      r => r.groupKey === `env:${ENV}:v6-rate` && r.actionIndex === 0,
    );
    assert(
      block.length === 2,
      `v6-rate action #0 has 2 rows (got ${block.length})`,
    );
    assert(
      block[0]?.kind === 'verify_ok' && block[0]?.attempt === 1,
      'first attempt is verify_ok',
    );
    assert(block[1]?.kind === 'action_ok', 'action_ok closes the block');
    assert(
      block[1]?.summary.includes('changelog 1→2'),
      `nl action reports the env write (got "${block[1]?.summary}")`,
    );
  }

  // A recurrence re-runs the same action block; its verify is attempt 1 again.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'verify_failed',
              trigger_id: 'r',
              action_index: 0,
            },
            {
              seq: 2,
              ts: '2026-08-06T10:00:01+00:00',
              kind: 'verify_ok',
              trigger_id: 'r',
              action_index: 0,
            },
            {
              seq: 3,
              ts: '2026-08-06T10:00:02+00:00',
              kind: 'action_ok',
              trigger_id: 'r',
              action_index: 0,
            },
            {
              seq: 4,
              ts: '2026-08-06T10:00:03+00:00',
              kind: 'fired',
              trigger_id: 'r',
              fire_count: 1,
            },
            {
              seq: 5,
              ts: '2026-08-06T10:00:10+00:00',
              kind: 'verify_ok',
              trigger_id: 'r',
              action_index: 0,
            },
            {
              seq: 6,
              ts: '2026-08-06T10:00:11+00:00',
              kind: 'action_ok',
              trigger_id: 'r',
              action_index: 0,
            },
            {
              seq: 7,
              ts: '2026-08-06T10:00:12+00:00',
              kind: 'fired',
              trigger_id: 'r',
              fire_count: 2,
            },
          ],
        },
      },
    });
    const attempts = (t?.rows ?? [])
      .filter(r => r.kind.startsWith('verify_'))
      .map(r => r.attempt);
    assert(
      attempts.join(',') === '1,2,1',
      `the second firing's verify counts from 1 (got ${attempts.join(',')})`,
    );
  }

  // The permission action wrote nothing — the timeline says so explicitly.
  {
    const t = parseAll();
    const grant = t.rows.find(
      r => r.groupKey === `env:${ENV}:v6-grant` && r.kind === 'action_ok',
    );
    assert(
      grant?.summary.startsWith('enable snowflake_submit_query'),
      'permission tools listed',
    );
    assert(grant?.summary.includes('+3 more'), 'long tool lists are elided');
    assert(
      grant?.summary.includes('no env write'),
      'equal changelog ids read as no write',
    );
  }

  // groupKey IS the authored-index key, which is what lets the panel resolve
  // authored config without rebuilding the string itself.
  {
    const t = parseAll();
    const withTrigger = t.rows.filter(
      r => r.stream !== 'conversation' && r.triggerId,
    );
    assert(
      withTrigger.length === 19,
      `19 event rows carry a trigger id (got ${withTrigger.length})`,
    );
    assert(
      withTrigger.every(
        r =>
          r.groupKey ===
          (r.stream === 'env'
            ? `env:${r.envId}:${r.triggerId}`
            : `agent:${r.triggerId}`),
      ),
      "every event row keys on indexAuthoredTriggers' format",
    );
    const sweep = t.rows.find(
      r => r.stream === 'agent' && r.triggerId === 're-sweep',
    );
    assert(
      AUTHORED.get(sweep?.groupKey ?? '') !== undefined,
      'an agent row resolves its authored config straight off groupKey',
    );
  }

  // Edge 1 — co-detection: one gdocs_create_document call detected two triggers.
  {
    const t = parseAll();
    const co = t.edges.filter(e => e.kind === 'co-detection');
    assert(co.length === 1, `one co-detection edge (got ${co.length})`);
    assert(
      co[0]?.label === 'same gdocs_create_document call',
      'edge names the provoking tool',
    );
    const from = t.rows.find(r => r.id === co[0]?.from);
    const to = t.rows.find(r => r.id === co[0]?.to);
    assert(
      from?.triggerId === 'v6-grant' && to?.triggerId === 'v6-rate',
      'links v6-grant and v6-rate detections',
    );
  }

  // Edge 2 — anchor: v6-grant firing stamped the clock trigger's mark.
  {
    const t = parseAll();
    const anchor = t.edges.filter(e => e.kind === 'anchor');
    assert(anchor.length === 1, `one anchor edge (got ${anchor.length})`);
    const from = t.rows.find(r => r.id === anchor[0]?.from);
    const to = t.rows.find(r => r.id === anchor[0]?.to);
    assert(
      from?.kind === 'fired' && from?.triggerId === 'v6-grant',
      'anchor source is the firing',
    );
    assert(
      to?.kind === 'anchored' && to?.triggerId === 'v6-clock-correction',
      'anchor target',
    );
    // The mark stamped on `anchored` is the one the time `fired` later claims.
    const fired = t.rows.find(
      r =>
        r.groupKey === `env:${ENV}:v6-clock-correction` && r.kind === 'fired',
    );
    assert(
      fired?.raw.mark === to?.raw.mark,
      'anchored mark equals the fired mark',
    );
  }

  // Edge 3 — cross-engine: env v6-rate fired → agent re-sweep fired.
  {
    const t = parseAll();
    const cross = t.edges.filter(e => e.kind === 'cross-engine');
    assert(cross.length === 1, `one cross-engine edge (got ${cross.length})`);
    const from = t.rows.find(r => r.id === cross[0]?.from);
    const to = t.rows.find(r => r.id === cross[0]?.to);
    assert(
      from?.triggerId === 'v6-rate' && from?.kind === 'fired',
      'source is the env firing',
    );
    assert(
      to?.stream === 'agent' && to?.triggerId === 're-sweep',
      'target is the agent firing',
    );
    // `accept` never fired, and its sensors stayed armed — no phantom edge.
    assert(
      !t.edges.some(e => e.label.includes('accept')),
      'an unfired composite when yields no edge',
    );
  }

  // Firing reasons: the `fired` event carries no reason, so it comes off the
  // `detected` that opened the chain (or the clock mark, on the time path).
  {
    const t = parseAll();
    const reason = (tid: string) =>
      t.rows.find(
        r => r.stream === 'env' && r.kind === 'fired' && r.triggerId === tid,
      )?.reason;
    assert(
      reason('v6-grant') === 'the agent called gdocs_create_document (default)',
      `action firing names the call (got ${reason('v6-grant')})`,
    );
    assert(
      reason('v6-rate') === reason('v6-grant'),
      'both co-detected triggers cite the same call',
    );
    // The mark is the reason, so it must not also be duplicated into the summary.
    const clockFired = t.rows.find(
      r => r.kind === 'fired' && r.triggerId === 'v6-clock-correction',
    );
    assert(
      clockFired?.summary === 'fired (fire #1)',
      `time summary stays terse (got ${clockFired?.summary})`,
    );
    assert(
      reason('v6-clock-correction') ===
        'the virtual clock reached 2028-07-11 05:43:08Z',
      `time firing cites its mark (got ${reason('v6-clock-correction')})`,
    );
    const sweep = t.rows.find(
      r => r.stream === 'agent' && r.triggerId === 're-sweep',
    );
    assert(
      sweep?.reason === 'its condition was met: v6-rate fired',
      `an agent firing cites its authored condition (got ${sweep?.reason})`,
    );
    assert(
      t.rows.filter(r => r.kind !== 'fired').every(r => r.reason === undefined),
      'only fired rows carry a reason',
    );
  }

  // A state trigger's provoking tool prompted the re-evaluation; it is NOT what
  // the check matched, and the wording must not claim otherwise.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          triggers: [
            { id: 's', type: 'state' },
            { id: 'a', type: 'action' },
          ],
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'detected',
              trigger_id: 's',
              provoking: { tool: 'gdocs_create_document', role: 'default' },
            },
            {
              seq: 2,
              ts: '2026-08-06T10:00:01+00:00',
              kind: 'fired',
              trigger_id: 's',
              fire_count: 1,
            },
            {
              seq: 3,
              ts: '2026-08-06T10:00:02+00:00',
              kind: 'detected',
              trigger_id: 'a',
              provoking: { tool: 'gdocs_create_document', role: 'default' },
            },
            {
              seq: 4,
              ts: '2026-08-06T10:00:03+00:00',
              kind: 'fired',
              trigger_id: 'a',
              fire_count: 1,
            },
          ],
        },
      },
    });
    const reason = (tid: string) =>
      t?.rows.find(r => r.kind === 'fired' && r.triggerId === tid)?.reason;
    assert(
      reason('s') ===
        'its check matched, re-evaluated after gdocs_create_document (default)',
      `state firing is phrased as a re-evaluation (got ${reason('s')})`,
    );
    assert(
      reason('a') === 'the agent called gdocs_create_document (default)',
      'action firing differs',
    );
  }

  // An unknown trigger type must not fall through to the causal wording. The
  // spec comes from a different part of the payload than the events, so absence
  // is reachable — and "the agent called X" would be wrong for a state trigger.
  {
    const events = [
      {
        seq: 1,
        ts: '2026-08-06T10:00:00+00:00',
        kind: 'detected',
        trigger_id: 'x',
        provoking: { tool: 'gdocs_create_document', role: 'default' },
      },
      {
        seq: 2,
        ts: '2026-08-06T10:00:01+00:00',
        kind: 'fired',
        trigger_id: 'x',
        fire_count: 1,
      },
    ];
    const reasonWith = (triggers: unknown[]) =>
      parseTriggerTimeline({
        state: { 'env-x': { triggers, events } },
      })?.rows.find(r => r.kind === 'fired')?.reason;
    assert(
      reasonWith([]) ===
        'it was detected after gdocs_create_document (default)',
      `an unlisted trigger states the bare fact (got ${reasonWith([])})`,
    );
    assert(
      reasonWith([{ id: 'x' }]) ===
        'it was detected after gdocs_create_document (default)',
      'a spec with no type also stays neutral',
    );
    assert(
      reasonWith([{ id: 'x', type: 'action' }]) ===
        'the agent called gdocs_create_document (default)',
      'a known action type earns the causal wording',
    );
  }

  // Co-detection must survive another env's rows landing mid-run: seq contiguity
  // is promised per env, but rows are merged across envs by timestamp.
  {
    const det = (seq: number, tid: string) => ({
      seq,
      kind: 'detected',
      trigger_id: tid,
      provoking: { tool: 'gdocs_create_document', role: 'default' },
    });
    const t = parseTriggerTimeline({
      state: {
        'env-1': {
          events: [
            { ...det(6, 'a1'), ts: '2026-08-06T10:00:00+00:00' },
            { ...det(7, 'a2'), ts: '2026-08-06T10:00:02+00:00' },
          ],
        },
        // Sorts between env-1's two detections, and its seq is unrelated.
        'env-2': {
          events: [{ ...det(40, 'b1'), ts: '2026-08-06T10:00:01+00:00' }],
        },
      },
    });
    const order = (t?.rows ?? []).map(r => r.envId);
    assert(
      order.join(',') === 'env-1,env-2,env-1',
      `envs interleave (got ${order.join(',')})`,
    );
    const co = (t?.edges ?? []).filter(e => e.kind === 'co-detection');
    assert(
      co.length === 1,
      `env-1's run survives the interleave (got ${co.length} edges)`,
    );
    const from = t?.rows.find(r => r.id === co[0]?.from);
    const to = t?.rows.find(r => r.id === co[0]?.to);
    assert(
      from?.triggerId === 'a1' && to?.triggerId === 'a2',
      'and it links the two env-1 detections, not the env-2 one',
    );
  }

  // Unknowable reasons stay absent rather than being guessed.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          triggers: [{ id: 'a', type: 'action' }],
          events: [
            // The `detected` that explains this firing was evicted (seq gap).
            {
              seq: 900,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'fired',
              trigger_id: 'a',
              fire_count: 3,
            },
          ],
          events_dropped: 899,
        },
      },
      agentTriggerState: {
        'step-1': [
          {
            seq: 1,
            ts: '2026-08-06T10:00:01+00:00',
            kind: 'fired',
            trigger_id: 'unknown',
            turn: 1,
          },
        ],
      },
    });
    assert(
      t?.rows.find(r => r.stream === 'env' && r.kind === 'fired')?.reason ===
        undefined,
      'an evicted detection yields no reason rather than a guess',
    );
    assert(
      t?.rows.find(r => r.stream === 'agent')?.reason === undefined,
      'an agent firing with no authored config yields no reason',
    );
  }

  // A catch-up burst: every mark fires with its own reason, and capping is said.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          triggers: [{ id: 'r', type: 'time' }],
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'detected',
              trigger_id: 'r',
              provoking: { source: 'clock' },
              due: 2,
              capped: true,
              first_mark: '2028-01-01T00:00:00Z',
              last_mark: '2028-01-02T00:00:00Z',
            },
            {
              seq: 2,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'fired',
              trigger_id: 'r',
              fire_count: 1,
              mark: '2028-01-01T00:00:00Z',
              capped: true,
            },
            {
              seq: 3,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'fired',
              trigger_id: 'r',
              fire_count: 2,
              mark: '2028-01-02T00:00:00Z',
              capped: true,
            },
          ],
        },
      },
    });
    const reasons = (t?.rows ?? [])
      .filter(r => r.kind === 'fired')
      .map(r => r.reason);
    assert(
      reasons[0] ===
        'the virtual clock reached 2028-01-01 00:00:00Z (catch-up burst was capped)',
      `first mark (got ${reasons[0]})`,
    );
    assert(
      reasons[1]?.includes('2028-01-02 00:00:00Z'),
      'each firing in a burst cites its own mark, not the burst head',
    );
  }

  // Highlight: selecting a firing pulls in its chain and its edge partners.
  {
    const t = parseAll();
    const fired = t.rows.find(
      r => r.groupKey === `env:${ENV}:v6-rate` && r.kind === 'fired',
    );
    const lit = highlightFor(t, fired?.id ?? null);
    const ids = t.rows.filter(r => lit.ids.has(r.id));
    assert(
      ids.filter(r => r.groupKey === `env:${ENV}:v6-rate`).length === 5,
      'the whole v6-rate chain lights up, registration included',
    );
    assert(
      ids.some(r => r.stream === 'agent' && r.triggerId === 're-sweep'),
      'the cross-engine partner lights up',
    );
    assert(
      ids.some(r => r.triggerId === 'v6-grant' && r.kind === 'detected'),
      'the co-detected sibling lights up',
    );
    // Labels ride along with the ids so the panel needs no edge logic of its own.
    assert(
      lit.labels.get(fired?.id ?? '')?.join('|') === 'v6-rate fired → re-sweep',
      `the selected firing carries its edge label (got ${lit.labels
        .get(fired?.id ?? '')
        ?.join('|')})`,
    );
    assert(
      lit.labels.size === 4,
      `both ends of each in-selection edge are labelled (got ${lit.labels.size})`,
    );
    assert(
      highlightFor(t, null).ids.size === 0,
      'no selection highlights nothing',
    );
    assert(
      highlightFor(t, null).labels.size === 0,
      'no selection labels nothing',
    );
    assert(
      highlightFor(t, 'nope').ids.size === 0,
      'an unknown row id highlights nothing',
    );
  }

  // Clock header: the virtual axis the events are measured on.
  // Read straight off `state_meta` — the timeline carries events, not the axis,
  // because the axis moves every poll while the events sit still.
  {
    const t = parseAll();
    const clocks = parseEnvClocks(STATE_META);
    assert(clocks.length === 1, 'one clock reading');
    assert(
      clocks[0]?.armed && clocks[0]?.rate === 86400,
      'armed at 86400x real time',
    );
    assert(clocks[0]?.t0 === '2026-06-01T00:00:00Z', 't0 carried through');
    assert(
      clocks[0]?.virtualTime !== undefined &&
        clocks[0]?.readAtUtc !== undefined,
      'the reading and the time it was taken both survive',
    );
    assert(
      t.rows
        .filter(r => r.stream === 'env')
        .every(r => r.virtualTime !== undefined),
      'every env event carries virtual_time while the clock is armed',
    );
  }

  // Agent-stream kinds other than `fired` are summarized, not dropped.
  {
    const t = parseAll();
    const registered = t.rows.find(
      r => r.stream === 'agent' && r.kind === 'registered',
    );
    assert(
      registered?.summary.includes('4 agent triggers'),
      'registered names the count',
    );
    assert(
      registered?.groupKey === undefined,
      'registered has no trigger chain',
    );
    assert(
      t.rows.filter(r => r.stream === 'agent' && r.kind === 'fired').length ===
        3,
      'three agent firings',
    );
  }

  // Conversation turns: user messages open a turn, the agent reply stays on it.
  {
    const t = parseAll();
    const conv = t.rows.filter(r => r.stream === 'conversation');
    assert(
      conv[0]?.turn === 1 && conv[0]?.role === 'user',
      'first user message opens turn 1',
    );
    assert(
      conv[1]?.turn === 1 && conv[1]?.role === 'agent',
      'the reply stays on turn 1',
    );
    assert(conv[2]?.turn === 2, 'the next user message opens turn 2');
    assert(
      conv.every(r => r.summary.length > 0),
      'every message row has preview text',
    );
  }

  // Numeric summary fields survived the move off String(); hostile shapes read
  // as '?' rather than [object Object].
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'reanchored',
              trigger_id: 'r',
              generation: 7,
              dropped_mark: '2028-01-01T00:00:00Z',
            },
            {
              seq: 2,
              ts: '2026-08-06T10:00:01+00:00',
              kind: 'failed',
              trigger_id: 'r',
              action_index: 2,
            },
            {
              seq: 3,
              ts: '2026-08-06T10:00:02+00:00',
              kind: 'detected',
              trigger_id: 'r',
              provoking: { tool: { evil: 1 }, role: [] },
            },
          ],
        },
      },
      agentTriggerState: {
        'step-1': [
          {
            seq: 1,
            ts: '2026-08-06T10:00:03+00:00',
            kind: 'fired',
            trigger_id: 'g',
            turn: 4,
          },
        ],
      },
    });
    const summary = (seq: number) =>
      t?.rows.find(r => r.stream === 'env' && r.seq === seq)?.summary;
    assert(
      summary(1)?.includes('generation 7'),
      `reanchored keeps its generation (got ${summary(1)})`,
    );
    assert(
      summary(1)?.includes('2028-01-01 00:00:00Z'),
      'reanchored shows the dropped mark',
    );
    assert(
      summary(2) === 'action #2 rejected — trigger failed',
      `failed keeps its index (got ${summary(2)})`,
    );
    assert(
      summary(3) === 'condition matched on ? (?)',
      `a non-string provoking renders '?' (got ${summary(3)})`,
    );
    const agentFired = t?.rows.find(r => r.stream === 'agent');
    assert(
      agentFired?.summary.includes('turn 4'),
      `agent firing keeps its turn (got ${agentFired?.summary})`,
    );
  }

  // Unknown kinds (barriers, anything later) render, never crash.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T00:00:00+00:00',
              kind: 'barrier_timeout',
              trigger_id: 'b1',
            },
            {
              seq: 2,
              ts: '2026-08-06T00:00:01+00:00',
              kind: 'fired',
              trigger_id: 'b1',
            },
          ],
        },
      },
    });
    assert(t !== null, 'an unknown kind still yields a timeline');
    assert(
      t?.rows[0]?.kind === 'barrier_timeout',
      'unknown kind passes through verbatim',
    );
    assert(
      t?.rows[0]?.summary === 'barrier_timeout',
      'unknown kind falls back to its name',
    );
    assert(
      t?.rows[0]?.groupKey === 'env:env-x:b1',
      'unknown kind still joins its chain',
    );
  }

  // Truncation: a seq discontinuity becomes one divider carrying the count.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T00:00:00+00:00',
              kind: 'added',
              trigger_id: 'a',
            },
            {
              seq: 4200,
              ts: '2026-08-06T01:00:00+00:00',
              kind: 'fired',
              trigger_id: 'a',
            },
          ],
          events_dropped: 4198,
        },
      },
    });
    const gap = t?.rows.find(r => r.kind === 'gap');
    assert(
      gap?.dropped === 4198,
      `divider counts the hole (got ${gap?.dropped})`,
    );
    assert(t?.eventsDropped === 4198, 'the gateway total is reported too');
  }

  // Degradation and hostile shapes.
  {
    assert(parseTriggerTimeline({}) === null, 'no sources -> null');
    assert(parseTriggerTimeline({ state: {} }) === null, 'empty state -> null');
    assert(
      parseTriggerTimeline({ state: { 'env-x': { events: [] } } }) === null,
      'an env with no events -> null',
    );
    assert(
      parseTriggerTimeline({
        state: 'nope',
        agentTriggerState: 7,
        conversations: 'x',
      }) === null,
      'non-object sources -> null, not a throw',
    );
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [null, 3, { kind: 'added' }, { seq: 2, kind: 'fired' }],
        },
      },
      stateMeta: { 'env-x': { clock: null } },
      conversations: [null, { messages: [null, { role: 'user' }] }],
    });
    assert(
      t?.rows.filter(r => r.stream === 'env').length === 2,
      'junk events are skipped',
    );
    assert(
      parseEnvClocks({ 'env-1': { clock: null } }).length === 0,
      'a null clock yields no header',
    );
    assert(
      parseEnvClocks('nonsense').length === 0,
      'a non-object state_meta degrades rather than throwing',
    );
    assert(
      t?.rows.some(
        r => r.stream === 'conversation' && r.summary === '(no text)',
      ),
      'a message with no parts still renders',
    );
  }

  // Timestamp-less rows keep their position instead of sinking to the epoch.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'added',
              trigger_id: 'a',
            },
            { seq: 2, kind: 'detected', trigger_id: 'a' },
            {
              seq: 3,
              ts: '2026-08-06T10:00:05+00:00',
              kind: 'fired',
              trigger_id: 'a',
            },
          ],
        },
      },
    });
    assert(
      t?.rows.map(r => r.seq).join(',') === '1,2,3',
      `a missing ts holds its slot (got ${t?.rows.map(r => r.seq).join(',')})`,
    );
  }

  // A ts that regresses within one stream is clamped, never reordered.
  {
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:10+00:00',
              kind: 'added',
              trigger_id: 'a',
            },
            {
              seq: 2,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'fired',
              trigger_id: 'a',
            },
          ],
        },
      },
    });
    assert(
      t?.rows.map(r => r.seq).join(',') === '1,2',
      'seq order survives a ts regression',
    );
  }

  // Two conversations run on independent clocks and must still interleave.
  {
    const msg = (ts: string, role: string) => ({
      role,
      ts,
      parts: [{ kind: 'text', text: ts }],
    });
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'added',
              trigger_id: 'a',
            },
          ],
        },
      },
      conversations: [
        {
          conversation_id: 'c-A',
          messages: [
            msg('2026-08-06T10:00:01+00:00', 'user'),
            msg('2026-08-06T10:00:03+00:00', 'user'),
            msg('2026-08-06T10:00:05+00:00', 'user'),
          ],
        },
        {
          conversation_id: 'c-B',
          messages: [
            msg('2026-08-06T10:00:02+00:00', 'user'),
            msg('2026-08-06T10:00:04+00:00', 'user'),
            msg('2026-08-06T10:00:06+00:00', 'user'),
          ],
        },
      ],
    });
    const order = (t?.rows ?? [])
      .filter(r => r.stream === 'conversation')
      .map(r => r.conversationId);
    assert(
      order.join(',') === 'c-A,c-B,c-A,c-B,c-A,c-B',
      `two conversations interleave by ts (got ${order.join(',')})`,
    );
    assert(
      t?.conversationIds.join(',') === 'c-A,c-B',
      'both conversation ids reported',
    );
    // Turn numbering stays per-conversation, so each opens at turn 1.
    const firstOfB = (t?.rows ?? []).find(r => r.conversationId === 'c-B');
    assert(firstOfB?.turn === 1, 'each conversation numbers its own turns');
  }

  // Two id-less conversations must not collapse into one clamp either.
  {
    const msg = (ts: string) => ({ role: 'user', ts, parts: [] });
    const t = parseTriggerTimeline({
      state: {
        'env-x': {
          events: [
            {
              seq: 1,
              ts: '2026-08-06T10:00:00+00:00',
              kind: 'added',
              trigger_id: 'a',
            },
          ],
        },
      },
      conversations: [
        {
          messages: [
            msg('2026-08-06T10:00:01+00:00'),
            msg('2026-08-06T10:00:03+00:00'),
          ],
        },
        {
          messages: [
            msg('2026-08-06T10:00:02+00:00'),
            msg('2026-08-06T10:00:04+00:00'),
          ],
        },
      ],
    });
    const order = (t?.rows ?? [])
      .filter(r => r.stream === 'conversation')
      .map(r => r.conversationId);
    assert(
      order.join(',') === 'conv-0,conv-1,conv-0,conv-1',
      `index fallback keeps them apart (got ${order.join(',')})`,
    );
  }

  // A single conversation is unchanged by the per-conversation bucketing.
  {
    const t = parseAll();
    const conv = t.rows.filter(r => r.stream === 'conversation');
    assert(
      t.rows.length === 43,
      `reference run still merges to 43 rows (got ${t.rows.length})`,
    );
    assert(
      t.conversationIds.length === 1,
      'one conversation on the reference run',
    );
    assert(
      conv.every(r => r.conversationId === t.conversationIds[0]),
      'rows carry their conversation id',
    );
    assert(
      t.unavailableEnvs.length === 0,
      'an artifact-sourced run has no unavailable envs',
    );
  }

  // An unreachable live gateway is "unknown", not "no triggers".
  {
    const t = parseTriggerTimeline({
      state: {},
      stateSources: { 'env-x': 'unavailable', 'env-y': 'live' },
    });
    assert(t !== null, 'an unavailable env still yields a timeline');
    assert(
      t?.unavailableEnvs.join(',') === 'env-x',
      'only unavailable envs are reported',
    );
    assert(t?.rows.length === 0, 'no rows, but the caller can say why');
  }

  // …while sources that did resolve keep the zero-DOM degradation path.
  {
    assert(
      parseTriggerTimeline({
        state: {},
        stateSources: { 'env-x': 'artifact' },
      }) === null,
      'a resolved-but-eventless run still degrades to the v1 table',
    );
    assert(
      parseTriggerTimeline({ state: {}, stateSources: 'nonsense' }) === null,
      'a non-object stateSources degrades rather than throwing',
    );
  }

  {
    assert(
      shortStamp('2028-07-11T05:43:08.912294Z') === '2028-07-11 05:43:08Z',
      'stamp shortens',
    );
    assert(
      shortStamp('2026-08-06T17:48:16.613273+00:00') === '2026-08-06 17:48:16Z',
      'microsecond +00:00 stamps parse',
    );
    assert(
      shortStamp('2026-08-06T10:48:16-07:00') === '2026-08-06 17:48:16Z',
      'a non-UTC offset converts to UTC instead of being relabelled',
    );
    assert(shortStamp(undefined) === '—', 'a missing stamp renders as a dash');
    assert(
      shortStamp('not-a-date') === 'not-a-date',
      'an unparseable stamp passes through',
    );
  }

  // The live view's forward-looking half — armed marks and the virtual
  // countdown. Rate 86400 is the dev clock's: one real second is one virtual day.
  {
    const LIVE_STATE = {
      'env-1': {
        triggers: [
          {
            id: 't-recur',
            type: 'time',
            status: 'armed',
            fire_count: 2,
            next_mark: '2031-10-23T00:00:00Z',
          },
          {
            id: 't-firing',
            type: 'time',
            status: 'firing',
            fire_count: 3,
            next_mark: '2031-10-24T00:00:00Z',
          },
          {
            id: 't-done',
            type: 'action',
            status: 'fired',
            fire_count: 1,
            next_mark: null,
          },
          { id: 't-sensor', type: 'state', status: 'armed', fire_count: 0 },
        ],
        events: [],
      },
    };
    const LIVE_META = {
      'env-1': {
        clock: {
          armed: true,
          t0: '2026-06-01T00:00:00Z',
          virtual_seconds_per_real_second: 86400,
          virtual_time: '2031-10-22T00:00:00Z',
        },
        clock_read_at_utc: '2026-08-07T00:00:00Z',
      },
    };

    const live = parseTriggerTimeline({
      state: LIVE_STATE,
      stateMeta: LIVE_META,
    });
    assert(
      live !== null,
      'an armed mark with no events yet still yields a timeline',
    );
    const pending = live?.pending ?? [];
    assert(
      pending.length === 2,
      `only the two triggers with a next_mark are pending (${pending.length})`,
    );
    assert(
      pending.map(p => p.triggerId).join(',') === 't-recur,t-firing',
      'a mid-firing trigger keeps its mark — the gateway advances next_mark before settling status',
    );
    assert(
      pending[0]?.fireCount === 2,
      'fire_count rides along as the recurrence progress signal',
    );
    assert(
      pending[0]?.groupKey === 'env:env-1:t-recur',
      'pending keys match the authored index',
    );

    // The opening phase of a run whose triggers can't carry a mark at all: an
    // action trigger fires on a tool call, a state trigger on a row predicate,
    // and a time trigger has no mark until it is anchored. Gating the timeline
    // on `pending` alone hid the badge, clock and trigger rows for every such
    // run — and those are the runs where the per-turn table is empty too, so
    // the whole live view went blank.
    const armedOnly = parseTriggerTimeline({
      state: {
        'env-1': {
          triggers: [
            { id: 't-act', type: 'action', status: 'armed', fire_count: 0 },
            { id: 't-state', type: 'state', status: 'armed', fire_count: 0 },
            {
              id: 't-unanchored',
              type: 'time',
              status: 'armed',
              fire_count: 0,
              next_mark: null,
            },
          ],
          events: [],
        },
      },
      stateMeta: LIVE_META,
    });
    assert(
      armedOnly !== null,
      'armed triggers with no marks and no events still yield a timeline',
    );
    assert(
      (armedOnly?.pending ?? []).length === 0,
      'and none of them are pending — this passes on specs, not on a mark',
    );
    assert(
      armedOnly?.envIds.join(',') === 'env-1',
      'the env is still reported so the live badge and clock header render',
    );

    // The counterpart, pinned so the gate stays a gate: an env that registered
    // no triggers at all genuinely has nothing to show.
    assert(
      parseTriggerTimeline({
        state: { 'env-1': { triggers: [], events: [] } },
      }) === null,
      'an env with no triggers and no events -> null',
    );

    const clock = parseEnvClocks(LIVE_META)[0];
    assert(
      clock?.rate === 86400 && clock?.armed === true,
      'the live clock parses',
    );

    // Measured off the reading the clock vended, not off wall time — the value
    // trails by a poll, which is the deliberate trade.
    const ahead = clock ? countdownTo('2031-10-23T00:00:00Z', clock) : null;
    assert(
      ahead?.virtualMs === 86_400_000,
      'a mark one virtual day past the reading reads as 86.4e6 virtual ms',
    );
    assert(
      ahead?.realMs === 1000,
      'which is one real second away at rate 86400',
    );
    assert(ahead?.due === false, 'and is not yet due');

    const passed = clock ? countdownTo('2031-10-21T00:00:00Z', clock) : null;
    assert(
      passed?.due === true && passed?.virtualMs === -86_400_000,
      'a mark behind the reading is due rather than clamped',
    );

    assert(
      countdownTo('2031-10-23T00:00:00Z', {
        envId: 'e',
        armed: false,
        virtualTime: '2031-10-22T00:00:00Z',
        rate: 3600,
      }) === null,
      'an unarmed clock yields no countdown',
    );
    assert(
      countdownTo('2031-10-23T00:00:00Z', {
        envId: 'e',
        armed: true,
        virtualTime: '2031-10-22T00:00:00Z',
        rate: 0,
      })?.realMs === Infinity,
      'a stopped clock never reaches its mark',
    );
    assert(
      countdownTo('not-a-date', clock!) === null,
      'an unparseable mark yields no countdown rather than NaN',
    );

    // The fingerprint decides whether a poll re-parses the log at all, so what it
    // ignores matters as much as what it catches.
    {
      const base = {
        status: 'running',
        source: 'live',
        state: LIVE_STATE,
        state_meta: LIVE_META,
      };
      const fp = timelineFingerprint(base);
      assert(
        timelineFingerprint({ ...base, state_meta: LIVE_META }) === fp,
        'an identical payload fingerprints identically',
      );

      // The whole point: a live clock advances every poll while the log sits still.
      const moved = {
        ...base,
        state_meta: {
          'env-1': {
            clock: {
              ...LIVE_META['env-1'].clock,
              virtual_time: '2031-12-25T00:00:00Z',
            },
            clock_read_at_utc: '2026-08-07T01:00:00Z',
          },
        },
      };
      assert(
        timelineFingerprint(moved) === fp,
        'a clock that ticked on its own does not invalidate the log',
      );

      const withEvent = {
        ...base,
        state: {
          'env-1': {
            ...LIVE_STATE['env-1'],
            events: [{ seq: 1, kind: 'added', trigger_id: 't-recur' }],
          },
        },
      };
      assert(
        timelineFingerprint(withEvent) !== fp,
        'a new event does invalidate it',
      );

      const fired = {
        ...base,
        state: {
          'env-1': {
            ...LIVE_STATE['env-1'],
            triggers: LIVE_STATE['env-1'].triggers.map(t =>
              t.id === 't-recur' ? { ...t, fire_count: 3 } : t,
            ),
          },
        },
      };
      assert(
        timelineFingerprint(fired) !== fp,
        'so does a fire_count advancing',
      );
      assert(
        timelineFingerprint({ ...base, status: 'completed' }) !== fp,
        'so does the handoff to a terminal status',
      );
      assert(
        timelineFingerprint({
          ...base,
          state_sources: { 'env-1': 'unavailable' },
        }) !== fp,
        'so does an env going unreadable',
      );
      assert(
        timelineFingerprint(null) === 'empty' &&
          timelineFingerprint(null) !== fp,
        'a missing payload fingerprints to a value no real one collides with',
      );
    }

    assert(humanizeMs(86_400_000) === '1d', 'a day humanizes');
    assert(humanizeMs(3_720_000) === '1h 2m', 'compound spans keep two units');
    assert(humanizeMs(45_000) === '45s', 'sub-minute spans stay in seconds');
    assert(
      humanizeMs(Infinity) === '∞',
      'an unreachable mark renders as infinity',
    );
  }
}

main();

if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log('\nAll parse-trigger-events smoke assertions passed');
