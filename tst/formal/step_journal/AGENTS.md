# AGENTS.md: model-checking the step journal's epoch fencing

How to check a change to the task-instance store's journal, attempt-epoch fencing or undo against the TLA+ model in
this directory. The repository-wide rules are in the root `AGENTS.md`.

`StepJournal.tla` models the instance document, the journal rows, and the scheduler that writes them. Each store call
is one atomic action. TLC, the TLA+ model checker, explores every interleaving of those calls across a retry, a
cancelled sibling whose `to_thread` write lands late, and a second process that re-registers the same `instance_id`.
It checks four invariants:

- `NoSupersededWriteLands`. A seed, completion, scheduler record, failure or cancel write from an attempt that an undo
  or a re-register retired never lands on the instance. If one lands, the run's stored context or `completed_steps`
  holds another attempt's results, and its dependents, its teardown and any resume trust them.
- `ReplayReproducesStored`. Replaying the live generation's journal over its seed reproduces the stored context,
  whenever no doc-first write is half done and the run is neither failed nor cancelled. If it does not,
  `undo_steps_sync` rebuilds a context the run never had, so a retry runs on the wrong state.
- `NoSandboxLeaks`. When a run ends, its live context names every sandbox its steps created, in a `deployed_*` record
  or a `retry_resets` marker. `teardown_run` terminates what `teardown.py:recorded_sandboxes` finds there, except the
  ids `TORN_DOWN_KEY` lists as terminated already, so a sandbox missing from both stays up after the run.
- `NoMarkerNamesLive`. A `retry_resets` marker never names a sandbox that a `deployed_*` record of the live context
  still names. If one does, the run's record says a retry rolled back a sandbox that is still in use.

`single.cfg` runs one process. `takeover.cfg` runs two processes on one `instance_id`, where the second re-registers it
while the first runs.

## When to run it

Run the check when a change touches a function in this table. Each action's comment in `StepJournal.tla` names the code
it stands for.

| Code | Model action |
| --- | --- |
| `store.py:upsert_instance` (the insert, or the re-register update) | `Register` |
| `store.py:seed_context` doc update and its `Lte(attempt_epoch)` filter | `SeedDoc` |
| `store.py:_journal_step` with `_bump_journal_seq` and its `attempt_epoch` check | `Bump`, `SeedBump`, `UndoSchedBump` (`FenceSeq` only under a mutation) |
| `store.py:_write_journal_row` | `RowLands` and `Upsert`, used by `Row`, `SeedRow`, `UndoSchedRow` |
| `store.py:record_step_complete_sync` `_mutate` and its CAS | `Cas` (`Declines`, `Complete`); its exact-seq delete is `Clean` |
| `store.py:undo_steps_sync` | `UndoRead` (the `find_one` and the journal read), `UndoWrite` (the CAS update), `UndoTrim`, `UndoSchedBump`, `UndoSchedRow` |
| `store.py:record_task_failure_sync`, `record_task_cancelled_sync` | `Terminal`, through `FailWrite` and `CancelWrite` |
| `document_store.py:compare_and_swap` | the atomicity of `Cas`, and the rev check in `UndoWrite` |
| `step_journal.py:commit_ordered`, `replay_context` | `Replay`, and the replayed context in `UndoRead` |
| `task.py:_run`: register, then seed under the registered `attempt_epoch` | `Register`, `SeedDoc` |
| `task.py:_drive_dag`: the dispatch loop, batch classification, loop exit, `except` branch | `Dispatch`, `Collect`, `Finish`, `CancelRun` |
| `task.py:_drive_dag._run_one`, `_heartbeat_step` | `ExecOk`, `ExecFail` |
| a deploy step's `execute` appending its record to the live context | `Make` |
| `task.py:_drive_dag._drain_in_flight` | `Drained` (in-flight writes become orphans that keep running) |
| `task.py:_roll_back_span` (the overlap gate, the undo, `state.epoch`) | the retry branch of `Collect`, then `UndoRead` through `Rearm` |
| `task.py:_roll_back_span._audit`, `_rolled_back_sandboxes`, `_sandbox_ids`, `_rebuild_context`, `teardown.py:env_record_ids` | the marker and the live context in `UndoWrite` |

The paths are under `src/agent_env/task/`, except `document_store.py`, which is
`src/agent_env/store/document_store/document_store.py`. A change elsewhere, such as the async wrappers' logging or retry
counts, does not need a run.

The model was last confirmed against the code at the commit that last changed this directory
(`git log -1 -- tst/formal/step_journal`). To list the covered changes since then, run:

```sh
git log "$(git log -1 --format=%H -- tst/formal/step_journal)"..HEAD -- src/agent_env/task/store.py \
  src/agent_env/task/task.py src/agent_env/task/step_journal.py src/agent_env/store/document_store/document_store.py
```

After you confirm that the model still describes a change to the covered code, touch this guide in the same commit.

## Run the check

```sh
tst/formal/step_journal/check.sh             # the baseline configs
tst/formal/step_journal/check.sh mutations   # each modeled code break must break its invariant
tst/formal/step_journal/check.sh one takeover.cfg ReplayReproducesStored   # one run, no expectation
```

The script needs Java 11 or later. The first run downloads `tla2tools.jar` v1.7.4 and checks its sha256. The script
exits nonzero when any run differs from its expected outcome, and prints the directory that holds each run's spec copy,
config and `tlc.out`. A passing baseline prints:

```
ok          single                             holds                              5333 distinct states found
ok          takeover                           holds                              3662428 distinct states found
TLC 1.7.4; configs and output in /tmp/step-journal-tlc.XXXXXX
```

The distinct-state count of a run that holds is a drift signal. It changes only when the model or the config changes,
so a different count on an unchanged model means TLC did not check the spec or the config you think. A run that
violates an invariant stops at the first counterexample, so its count varies from run to run.

Each mutation removes one guard from the model, and `check.sh mutations` expects it to break a named invariant. The
`MUTATIONS` comment at the top of `StepJournal.tla` lists them. `NoCasGenerationCheck` is expected to hold: every
re-register also increments `attempt_epoch`, so the epoch check in `record_step_complete_sync` already declines what
the generation check declines.

## Read a counterexample

For a violation, `tlc.out` holds the shortest path to the broken invariant, one state per step. Each state names the
action that produced it, such as `State 4: <SeedDoc line ... of module StepJournal>`, and lists the value of every
variable after that action. The label does not name the process; the per-process variable that changed, usually
`dpc`, does. Read the path from the first state. For each step, use the table to name the store call or scheduler step
the action stands for, and note which variables changed. Then write the path as numbered steps in the code's words.

## Tell a real bug from a modeling error

A deadlock or a TLC error is a modeling error. `Done` covers every correct end state.

For a violation, check each step of the path against the code:

- Two calls in one process with no `await` between them cannot interleave with another coroutine of that process.
  Only another process, or a `to_thread` worker that still runs after a cancel, can come between them.
- Each action must be exactly one store call. A call that the code makes as two calls must be two actions.
- A guard that the model checks must be the guard that the code checks, on the same document read.

If the code cannot take one of the steps, the model is wrong. Fix the action and cite, in its comment, the code that
rules the step out. If every step maps to the code, the bug is real. Reproduce it with a deterministic test on
`LocalSqliteDocumentStore`, as `tst/unit/task/seed_context_fence_test.py` does for a late seed, and report it with the
steps.

Do not weaken an invariant, narrow a config, or add a state to `Quiescent` to make a run pass. `Quiescent` lists only
the windows where the code is doc-first by design: from `register_task_instance` to the end of `seed_context`, and from
the undo CAS to its `__scheduler__` row.

## Update the model for an intended change

1. Map each hunk of the diff to its action with the table.
2. Edit the action so it does what the new code does, one store call per action, and name the code in its comment.
3. If the change adds or moves a guard, add a `Mut` value that removes it, list it in the `MUTATIONS` comment, and add
   a line to the `mutations` case of `check.sh` that names the invariant it must break.
4. Run the baseline and the mutations. Each line must print `ok`.
5. Update the table in this guide in the same commit.

## What the model does not cover

- Record shapes. The model names each sandbox by its id, as `_rolled_back_sandboxes` and `recorded_sandboxes` do, so
  it does not check how a record serializes.
- Records that name several sandboxes. Each record names one, so `teardown_test.py`, not the model, checks that a
  rolled-back env record sharing a survivor's sandboxes names only its own in the marker.
- Context contents. Each step writes only its own key, so the model does not check replay order, `rerecorded_steps`,
  `move_to_end` or the `add_to_sets` union. Re-records do not occur.
- Sandboxes other than `a`'s and `c`'s. Each creates one and records it in one step: a sandbox a cancel strands inside
  the provider's create call, a record appended before the step raises, and a `teardown_sandboxes` step are not
  modeled.
- Store failures: a failed journal write, a process that crashes mid-call, the `RuntimeError` paths, and the retries in
  the async wrappers.
- `start_step > 0` (`_journal_seeded_completions`), `end_step`, tolerant steps, `create_instance`, `clear_journal_sync`
  and `append_step_attempt_failure_sync`.
- The context that `record_task_failure_sync` writes. The model writes only the status.
- An empty seed. The code then skips the doc update, and the model always makes it.
- `cli/task/run.py:_stored`, which seeds without an epoch an instance under a fresh random id that no other process
  holds.
- Other shapes. The task is one DAG (`a`, then `b` and `c`) with one retry, at most two processes run, and in
  `takeover.cfg` only the first process retries and no run is cancelled.
