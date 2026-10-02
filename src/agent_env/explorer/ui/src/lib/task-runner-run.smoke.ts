/**
 * Smoke test for the task runner's run lifecycle: a run starts, appears in the instance list,
 * and a refused launch surfaces the backend's reason instead of leaving the page on "Starting…".
 *
 * Runner: plain TS, throws on assertion failure. From this package:
 *   npx tsx src/lib/task-runner-run.smoke.ts
 */
import {
  allTerminal,
  buildRunBody,
  cancelRun,
  reconcileInstances,
  shouldPoll,
  startRun,
  workflowIdOf,
  type Fetch,
} from './task-runner-run';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

interface Call {
  url: string;
  init?: RequestInit;
}

function fakeFetch(status: number, body: unknown): { fetch: Fetch; calls: Call[] } {
  const calls: Call[] = [];
  const fetch: Fetch = async (url, init) => {
    calls.push({ url, init });
    const text = typeof body === 'string' ? body : JSON.stringify(body);
    return new Response(text, { status });
  };
  return { fetch, calls };
}

function onlyCall(calls: Call[]): Call {
  const [call, ...rest] = calls;
  if (!call || rest.length) throw new Error(`expected one request, got ${calls.length}`);
  return call;
}

async function rejects(p: Promise<unknown>): Promise<string> {
  try {
    await p;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  return '(resolved)';
}

async function main(): Promise<void> {
  const TASK = '@local/agentenv-framework/hello/hello';

  // The request a run sends.
  const body = buildRunBody({ start_step: 2, context_from_instance_id: 'i-1', context_json: { x: 1 } }, 3, '');
  assert(body.version === 3 && body.priority === 0, 'run body: task version, interactive priority');
  assert(body.start_step === 2, 'run body: start step');
  assert(body.context_from_instance_id === 'i-1' && !('context_json' in body), 'run body: instance context wins over inline context');
  assert(!('project_id' in body), 'run body: an empty projectId is not sent');
  assert(buildRunBody({ version: 5 }, 3, 'p-9').version === 5, 'run body: an explicit version wins');
  assert(buildRunBody(undefined, null, 'p-9').project_id === 'p-9', 'run body: projectId is forwarded');

  // A run starts.
  const ok = fakeFetch(200, { workflow_id: 'local-abc', instance_id: 'i-2', status: 'QUEUED' });
  const workflowId = await startRun(ok.fetch, 'http://h', TASK, body);
  const startCall = onlyCall(ok.calls);
  assert(workflowId === 'local-abc', 'start: resolves to the workflow id');
  assert(
    startCall.url === 'http://h/api/v1/tasks/%40local%2Fagentenv-framework%2Fhello%2Fhello/run',
    'start: POSTs to /run with the task id as one encoded segment',
  );
  assert(startCall.init?.method === 'POST', 'start: method is POST');
  assert(JSON.parse(String(startCall.init?.body)).start_step === 2, 'start: sends the run body');

  // A refused launch.
  const refused = fakeFetch(402, { detail: 'Budget has been exceeded!' });
  assert(
    (await rejects(startRun(refused.fetch, '', TASK, body))) === 'Budget has been exceeded!',
    "failed launch: throws the backend's detail",
  );
  const broken = fakeFetch(500, '<html>oops</html>');
  assert(
    (await rejects(startRun(broken.fetch, '', TASK, body))) === 'Failed to start run (500)',
    'failed launch: a body without detail falls back to the status',
  );

  // The started run appears in the list.
  const before = [{ instance_id: 'i-1', status: 'completed', current_step: 3, context: { a: 1 } }];
  const first = reconcileInstances(before, '', new Set(), false);
  assert(first !== null && first.startedInstanceId === null, 'poll: a first page selects nothing when no run is pending');
  assert(first?.latestCompleted?.instance_id === 'i-1', 'poll: the newest completed run seeds a re-run');
  assert(reconcileInstances(before, first!.snapshot, first!.ids, true) === null, 'poll: an unchanged page is ignored');
  const after = [{ instance_id: 'i-2', status: 'running', current_step: 0 }, ...before];
  const appeared = reconcileInstances(after, first!.snapshot, first!.ids, true);
  assert(appeared?.startedInstanceId === 'i-2', 'poll: the pending run is picked up once it appears');
  assert(appeared?.ids.has('i-2') && appeared.ids.has('i-1'), 'poll: known ids include the new run');
  const progressed = reconcileInstances(
    [{ instance_id: 'i-2', status: 'running', current_step: 1 }, ...before], appeared!.snapshot, appeared!.ids, false);
  assert(progressed !== null && progressed.startedInstanceId === null, 'poll: step progress updates without re-selecting');

  // When polling stops.
  const done = allTerminal(before);
  assert(done && !allTerminal([]) && !allTerminal(after), 'terminal: only a non-empty, all-finished list is terminal');
  assert(shouldPoll(TASK, 'idle', done, true), 'polling continues while a started run has not appeared');
  assert(!shouldPoll(TASK, 'idle', done, false), 'polling stops once every run is finished');
  assert(shouldPoll(TASK, 'idle', allTerminal(after), false), 'polling continues while a run is live');
  assert(shouldPoll(TASK, 'idle', allTerminal([]), false), 'polling continues before any run exists');
  for (const phase of ['initializing', 'error', 'failed'])
    assert(!shouldPoll(TASK, phase, false, true), `polling stops in phase ${phase}`);
  assert(!shouldPoll(null, 'idle', false, true), 'polling needs a task id');

  // Cancel.
  assert(workflowIdOf({ context: { metadata: { workflow_id: 'local-abc' } } }) === 'local-abc', 'cancel: reads the workflow id off the run');
  assert(workflowIdOf(undefined) === undefined, 'cancel: no run, no workflow id');
  const cancelled = fakeFetch(200, { canceled: true });
  await cancelRun(cancelled.fetch, '', TASK, 'local-a/b');
  assert(
    onlyCall(cancelled.calls).url === '/api/v1/tasks/%40local%2Fagentenv-framework%2Fhello%2Fhello/cancel-run?workflow_id=local-a%2Fb',
    'cancel: POSTs cancel-run with both ids encoded',
  );
  assert(
    (await rejects(cancelRun(fakeFetch(404, {}).fetch, '', TASK, 'x'))) === 'Cancel failed (404)',
    'cancel: a refused cancel throws',
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
