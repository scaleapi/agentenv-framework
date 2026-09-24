/**
 * Smoke test for the runtime trigger-evidence parser.
 *
 * Runner: yarn test:smoke (auto-discovered), or:
 *   NODE_OPTIONS='--require ../../.pnp.cjs' npx tsx src/lib/parse-trigger-runtime.smoke.ts
 *
 * Fixture: a synthetic context.metadata in the shape a post-firing-log instance
 * `task-demo-triggers-e2e-67586890daa818e1` (4 env + 4 agent
 * triggers, 3 reaction turns: t1 all armed / LLM usersim, t2 re-sweep+
 * marker fired on the v6-grant/v6-rate flips, t3 accept fired on the
 * sensor flips).
 */
import {
  parseTriggerRuntime,
  triggerHistoryFor,
  turnRowFor,
} from './parse-trigger-runtime';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

const STEP = 'task-demo-triggers-e2e-prompt';
const ENV = 'demo-triggers-env';

const METADATA: Record<string, unknown> = {
  task_id: 'task-demo-triggers-e2e',
  env_trigger_registrations: [
    {
      step_id: 'task-demo-triggers-e2e-register-triggers',
      env_id: ENV,
      added: ['v6-grant', 'v6-rate', 'v6-accept-sensor', 'v6-marker-sensor'],
      executor_agent_name: 'executor-a1b2c3',
    },
  ],
  agent_trigger_registrations: [
    {
      step_id: 'task-demo-triggers-e2e-register-agent-triggers',
      agent_name: 'stakeholders-a1b2c3',
      added: ['re-sweep', 'marker', 'accept', 'wrap-guard'],
    },
  ],
  agent_trigger_firings: {
    [STEP]: [
      { turn: 1, fired: [] },
      { turn: 2, fired: ['re-sweep', 'marker'] },
      { turn: 3, fired: ['accept'] },
    ],
  },
  env_trigger_snapshots: {
    [STEP]: [
      {
        turn: 1,
        envs: {
          [ENV]: {
            'v6-grant': 'armed',
            'v6-rate': 'armed',
            'v6-accept-sensor': 'armed',
            'v6-marker-sensor': 'armed',
          },
        },
      },
      {
        turn: 2,
        envs: {
          [ENV]: {
            'v6-grant': 'fired',
            'v6-rate': 'fired',
            'v6-accept-sensor': 'armed',
            'v6-marker-sensor': 'armed',
          },
        },
      },
      {
        turn: 3,
        envs: {
          [ENV]: {
            'v6-grant': 'fired',
            'v6-rate': 'fired',
            'v6-accept-sensor': 'fired',
            'v6-marker-sensor': 'fired',
          },
        },
      },
    ],
  },
  usersim_turn_outputs: {
    [STEP]: [
      {
        turn: 1,
        fields: { milestone_id: 'qa', trigger_fired: null, speaker: 'dana' },
      },
    ],
  },
  env_trigger_state: {
    [ENV]: {
      capture_step_id: STEP,
      captured_at_utc: '2026-08-03T18:59:11.933099+00:00',
      event_count: 15,
      object_url: `obj://example-bucket/env_trigger_state/instance_id=x/${ENV}-56ba5cd1.json`,
      statuses: {
        'v6-accept-sensor': 'fired',
        'v6-grant': 'fired',
        'v6-marker-sensor': 'fired',
        'v6-rate': 'fired',
      },
      // capture projection — shape mirrors the real
      // post-0.9.1085 dev instance task-demo-clock-d4e5f6-zz00zz11.
      capture_is_final: true,
      triggers: {
        'v6-accept-sensor': {
          type: 'state',
          status: 'fired',
          fire_count: 1,
          next_mark: null,
        },
        'v6-grant': {
          type: 'action',
          status: 'fired',
          fire_count: 1,
          next_mark: null,
        },
        'v6-marker-sensor': {
          type: 'state',
          status: 'fired',
          fire_count: 1,
          next_mark: null,
        },
        'v6-rate': {
          type: 'action',
          status: 'fired',
          fire_count: 1,
          next_mark: null,
        },
      },
    },
  },
  agent_trigger_state: {
    [STEP]: [
      { seq: 1, ts: 't', kind: 'registered', detail: { added: ['re-sweep'] } },
      { seq: 2, ts: 't', kind: 'fired', trigger_id: 're-sweep', turn: 2 },
    ],
  },
};

function main(): void {
  {
    const rt = parseTriggerRuntime(METADATA);
    assert(rt !== null, 'runtime parses');
    if (!rt) return;

    assert(rt.registrations.length === 2, '2 registration rows');
    const envReg = rt.registrations.find(r => r.kind === 'env');
    assert(
      envReg?.groupKey === ENV &&
        envReg?.added.length === 4 &&
        envReg?.executorAgentName === 'executor-a1b2c3',
      'env registration row carries env/added/executor',
    );
    const agentReg = rt.registrations.find(r => r.kind === 'agent');
    assert(
      agentReg?.groupKey === 'stakeholders-a1b2c3' &&
        agentReg?.added.join(',') === 're-sweep,marker,accept,wrap-guard',
      'agent registration row carries agent/added',
    );

    assert(rt.envState.length === 1, 'one env state summary');
    assert(
      rt.envState[0]?.eventCount === 15 &&
        rt.envState[0]?.statuses['v6-rate'] === 'fired' &&
        Boolean(rt.envState[0]?.objectUrl) &&
        rt.envState[0]?.error === undefined,
      'env state summary parsed',
    );
    assert(
      rt.envState[0]?.captureIsFinal === true &&
        rt.envState[0]?.eventsDropped === undefined &&
        rt.envState[0]?.triggers?.['v6-grant']?.fireCount === 1 &&
        rt.envState[0]?.triggers?.['v6-grant']?.type === 'action' &&
        rt.envState[0]?.triggers?.['v6-grant']?.nextMark === undefined,
      'capture projection parsed (fire_count/type; null next_mark dropped)',
    );

    assert(rt.steps.length === 1 && rt.steps[0]?.stepId === STEP, 'one step');
    const step = rt.steps[0];
    assert(
      step?.firingLogAvailable === true && step?.hasPerTurnData === true,
      'firing log + per-turn data flags',
    );
    assert(step?.turns.length === 3, '3 turn rows');

    const [t1, t2, t3] = step?.turns ?? [];
    assert(
      t1?.fired.length === 0 &&
        t1?.deltas.length === 4 &&
        t1?.deltas.every(d => d.from === null && d.to === 'armed'),
      't1: no firings, initial snapshot as from=null deltas',
    );
    assert(
      t1?.usersim?.speaker === 'dana' && t1?.usersim?.milestone_id === 'qa',
      't1: usersim telemetry present',
    );
    assert(
      t2?.fired.join(',') === 're-sweep,marker' &&
        t2?.deltas.length === 2 &&
        t2?.deltas.every(d => d.from === 'armed' && d.to === 'fired') &&
        t2?.deltas
          .map(d => d.triggerId)
          .sort()
          .join(',') === 'v6-grant,v6-rate',
      't2: re-sweep+marker fired; v6-grant/v6-rate armed→fired deltas',
    );
    assert(
      t2?.usersim === undefined && t3?.usersim === undefined,
      't2/t3: usersim gaps mark trigger-injected turns',
    );
    assert(
      t3?.fired.join(',') === 'accept' &&
        t3?.deltas
          .map(d => d.triggerId)
          .sort()
          .join(',') === 'v6-accept-sensor,v6-marker-sensor',
      't3: accept fired; sensor flips',
    );
  }

  {
    // Pre-firing-log run: firings only, no snapshots/usersim; no firing log.
    const rt = parseTriggerRuntime({
      env_trigger_registrations: [
        { step_id: 'r', env_id: 'e1', added: ['t1'] },
      ],
      agent_trigger_firings: {
        p: [
          { turn: 1, fired: [] },
          { turn: 2, fired: ['g1'] },
        ],
      },
    });
    const step = rt?.steps[0];
    assert(
      step?.hasPerTurnData === false && step?.firingLogAvailable === false,
      'pre-firing-log: degraded flags set',
    );
    assert(
      step?.turns.length === 2 &&
        step?.turns.every(t => t.deltas.length === 0 && !t.hasSnapshot),
      'pre-firing-log: firings-only turn rows',
    );
  }

  {
    // Fail-open capture error surfaces.
    const rt = parseTriggerRuntime({
      env_trigger_state: {
        e1: { error: 'capture budget of 90s exceeded', capture_step_id: 'p' },
      },
    });
    assert(
      rt?.envState[0]?.error === 'capture budget of 90s exceeded' &&
        Object.keys(rt?.envState[0]?.statuses ?? { x: 1 }).length === 0,
      'env state error entry surfaced',
    );
    assert(
      rt?.envState[0]?.captureIsFinal === undefined &&
        rt?.envState[0]?.triggers === undefined &&
        rt?.envState[0]?.eventsDropped === undefined,
      'pre-capture-projection entry: new fields absent-safe',
    );
  }

  {
    // recurring-trigger capture: the trigger re-arms after every
    // arrival, so statuses alone read as never-fired — fire_count and
    // next_mark are the progress signals; the capture is not final.
    const rt = parseTriggerRuntime({
      env_trigger_registrations: [
        { step_id: 'r', env_id: 'e1', added: ['tick', 'once'] },
      ],
      env_trigger_state: {
        e1: {
          statuses: { tick: 'armed', once: 'fired' },
          capture_is_final: false,
          event_count: 12000,
          events_dropped: 2000,
          triggers: {
            tick: {
              type: 'time',
              status: 'armed',
              fire_count: 7,
              next_mark: '2026-06-08T00:00:00Z',
            },
            once: { type: 'time', status: 'fired', fire_count: 1 },
          },
        },
      },
    });
    const s = rt?.envState[0];
    assert(
      s?.captureIsFinal === false && s?.eventsDropped === 2000,
      'recurring: capture-not-final + dropped events parsed',
    );
    assert(
      s?.triggers?.tick?.fireCount === 7 &&
        s?.triggers?.tick?.nextMark === '2026-06-08T00:00:00Z',
      'recurring: fire_count + next_mark parsed',
    );
    if (rt) {
      const hist = triggerHistoryFor(rt, 'env', 'tick', 'e1');
      assert(
        hist.finalStatus === 'armed' &&
          hist.fireCount === 7 &&
          hist.nextMark === '2026-06-08T00:00:00Z',
        'triggerHistoryFor carries fireCount/nextMark for a re-armed recurrence',
      );
    }
  }

  {
    assert(
      parseTriggerRuntime({ a2a_conversations: {} }) === null,
      'no trigger keys → null',
    );
    assert(
      parseTriggerRuntime(undefined) === null,
      'undefined metadata → null',
    );
  }

  {
    // turnRowFor: the trajectory-strip join on the same fixture.
    const rt = parseTriggerRuntime(METADATA);
    const t2 = turnRowFor(rt, STEP, 2);
    assert(
      t2?.fired.join(',') === 're-sweep,marker' && t2?.deltas.length === 2,
      'turnRowFor: fixture turn 2 row joined',
    );
    const t1s = turnRowFor(rt, STEP, 1);
    assert(
      t1s?.usersim?.speaker === 'dana' && t1s?.deltas.length === 0,
      'turnRowFor: turn 1 keeps usersim, drops initial-armed deltas',
    );
    assert(
      turnRowFor(rt, 'other-step', 2) === null,
      'turnRowFor: unknown step → null',
    );
    assert(
      turnRowFor(rt, STEP, 99) === null,
      'turnRowFor: unknown turn → null',
    );
    assert(
      turnRowFor(rt, undefined, 2) === null &&
        turnRowFor(rt, STEP, undefined) === null &&
        turnRowFor(null, STEP, 2) === null,
      'turnRowFor: missing inputs → null',
    );

    // Quiet turns (no transitions, no firings, no user-sim — including a
    // turn whose only deltas are initial-armed registration state) are
    // filtered so the strip renders nothing for them.
    const quiet = parseTriggerRuntime({
      agent_trigger_firings: {
        p: [
          { turn: 1, fired: [] },
          { turn: 2, fired: [] },
        ],
      },
      env_trigger_snapshots: {
        p: [
          { turn: 1, envs: { e1: { t1: 'armed' } } },
          { turn: 2, envs: { e1: { t1: 'armed' } } },
        ],
      },
    });
    assert(
      quiet?.steps[0]?.turns.length === 2,
      'turnRowFor: quiet fixture parses 2 turn rows',
    );
    assert(
      turnRowFor(quiet, 'p', 1) === null,
      'turnRowFor: initial-armed-only turn → null',
    );
    assert(turnRowFor(quiet, 'p', 2) === null, 'turnRowFor: quiet turn → null');

    // A genuine turn-1 transition (first snapshot already non-armed) is
    // real activity and survives the initial-armed suppression.
    const early = parseTriggerRuntime({
      agent_trigger_firings: { p: [{ turn: 1, fired: [] }] },
      env_trigger_snapshots: {
        p: [{ turn: 1, envs: { e1: { t1: 'fired', t2: 'armed' } } }],
      },
    });
    const e1 = turnRowFor(early, 'p', 1);
    assert(
      e1?.deltas.length === 1 &&
        e1?.deltas[0]?.triggerId === 't1' &&
        e1?.deltas[0]?.to === 'fired',
      'turnRowFor: genuine turn-1 transition kept, initial armed dropped',
    );
  }

  {
    // triggerHistoryFor: one trigger's story for the popovers.
    const rt = parseTriggerRuntime(METADATA);
    if (!rt) return;
    const grant = triggerHistoryFor(rt, 'env', 'v6-grant', ENV);
    assert(
      grant.trail.length === 2 &&
        grant.trail[0]?.turn === 1 &&
        grant.trail[0]?.from === null &&
        grant.trail[0]?.to === 'armed' &&
        grant.trail[1]?.turn === 2 &&
        grant.trail[1]?.to === 'fired',
      'triggerHistoryFor: env trail is unfiltered (initial armed + flip)',
    );
    assert(
      grant.finalStatus === 'fired' &&
        grant.registeredBy ===
          'task-demo-triggers-e2e-register-triggers',
      'triggerHistoryFor: env final status + registering step',
    );
    const resweep = triggerHistoryFor(rt, 'agent', 're-sweep');
    assert(
      resweep.firedTurns.join(',') === '2' &&
        resweep.trail.length === 0 &&
        resweep.finalStatus === undefined &&
        resweep.registeredBy ===
          'task-demo-triggers-e2e-register-agent-triggers',
      'triggerHistoryFor: agent fired turns + registering step',
    );
    assert(
      triggerHistoryFor(rt, 'env', 'nope', ENV).trail.length === 0 &&
        triggerHistoryFor(rt, 'agent', 'nope').firedTurns.length === 0,
      'triggerHistoryFor: unknown trigger → empty history',
    );
  }

  if (failures > 0) {
    console.error(`\n${failures} assertion(s) failed`);
    process.exit(1);
  }
  console.log('\nAll parse-trigger-runtime smoke assertions passed');
}

main();
