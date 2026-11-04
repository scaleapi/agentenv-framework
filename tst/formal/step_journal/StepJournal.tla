---------------------------- MODULE StepJournal ----------------------------
(***************************************************************************)
(* The task-instance store's step journal, attempt-epoch fencing and undo, *)
(* driven by Task._drive_dag. Each action is one store call, or a retry    *)
(* loop or read pair its comment argues atomic; no action writes both the  *)
(* instance document and the journal rows.                                 *)
(*                                                                         *)
(* Code: src/agent_env/task/store.py (TaskInstanceStore) and               *)
(*       src/agent_env/task/task.py (_drive_dag, _roll_back_span).         *)
(*                                                                         *)
(* The task is three steps: a, then b and c in parallel (both depend on    *)
(* a). b may fail and carries retry_config(max_retries=1,                  *)
(* retry_from_step_id=b). Each step writes only its own context key, so    *)
(* replay order does not matter.                                           *)
(*                                                                         *)
(* MUTATIONS. Mut names one code break; check.sh mutations runs each.      *)
(*   NoCasEpochFence      record_step_complete_sync._mutate drops its      *)
(*                        attempt_epoch check                              *)
(*   NoCasGenerationCheck _mutate drops its run_generation check           *)
(*   NoCas                compare_and_swap's update loses its rev          *)
(*                        precondition (read, then a blind write)          *)
(*   NoJournalSeqGuard    _write_journal_row replaces without Lte(seq - 1) *)
(*   FenceBeforeSeq       _journal_step reads the epoch, then bumps the    *)
(*                        seq in a later call                              *)
(*   NoUndoEpochBump      _roll_back_span calls undo_steps without         *)
(*                        bump_epoch                                       *)
(*   NoUndoFence          undo_steps_sync._mutate drops its attempt_epoch  *)
(*                        refusal                                          *)
(*   NoTerminalFence      record_task_failure_sync / _cancelled_sync drop  *)
(*                        the Lte(attempt_epoch) filter                    *)
(*   NoSeedDocFence       seed_context's doc update drops its            *)
(*                        Lte(attempt_epoch) filter                        *)
(*   NoSeedJournalFence   seed_context journals __seed__ with              *)
(*                        attempt_epoch=None                               *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Procs,        \* processes running Task.run on one caller-supplied instance_id
    RetryProcs,   \* processes whose step b may fail
    CancelProcs,  \* processes whose run may be cancelled (asyncio.CancelledError)
    Mut           \* "none", or the name of one code break (see MUTATIONS)

Steps == {"a", "b", "c"}
Deps  == [s \in Steps |-> IF s = "a" THEN {} ELSE {"a"}]
Keys  == Steps \cup {"seed", "sched"}        \* context keys and journal step_ids
KillKeys == Keys \cup {"status"}
Slots == {1, 2}

(* A write's token names the process that issued it and the ghost era of   *)
(* the attempt it belongs to. It is not stored by the code; it lets the    *)
(* invariants say whose write a value is.                                  *)
None  == [p |-> "none", born |-> 0]
NoRow == [seq |-> 0, gen |-> 0, val |-> None]  \* real seqs start at 1
EmptyCtx  == [k \in Keys |-> None]
EmptyDone == [s \in Steps |-> None]

IdleSlot == [pc |-> "idle", dep |-> 0, seq |-> 0, jg |-> 0, jok |-> FALSE,
             orphan |-> FALSE, born |-> 0, rd |-> EmptyDone, rdecl |-> FALSE]

VARIABLES
    doc,     \* the task_instances document
    jr,      \* task_step_journal rows, by step_id
    era,     \* ghost: bumped by every re-register and every landed undo
    killed,  \* ghost: per key, the era a write must be born at or after to be live
    aera,    \* ghost: per process, the era of the attempt it currently runs
    dpc,     \* per process: where Task._run / _drive_dag is
    ep,      \* per process: _SchedulerState.epoch
    sdone,   \* per process: _SchedulerState.completed
    retried, \* per process: retry budget of b spent
    ferr,    \* per process: first_error is set
    span,    \* per process: rollback_ids of the undo in progress
    uRev, uCtx, uDone, uSeqs, uRm, uEpoch, \* per process: undo_steps_sync's CAS read
    sSeq, sGen, \* per process: (seq, generation) a seed / scheduler journal bump returned
    thr      \* per process, step, slot: one _run_one task and its to_thread write

vars == <<doc, jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span,
          uRev, uCtx, uDone, uSeqs, uRm, uEpoch, sSeq, sGen, thr>>
undoVars == <<uRev, uCtx, uDone, uSeqs, uRm, uEpoch>>

Tok(p) == [p |-> p, born |-> aera[p]]
SlotTok(p, s, i) == [p |-> p, born |-> thr[p][s][i].born]

ActivePcs == {"exec", "failed", "bump", "fence", "row", "cas", "casw", "clean", "ok", "sup"}
WritePcs  == {"bump", "fence", "row", "cas", "casw", "clean"}
InFlight(p, s) == \E i \in Slots : ~thr[p][s][i].orphan /\ thr[p][s][i].pc \in ActivePcs
AnyInFlight(p) == \E s \in Steps : InFlight(p, s)

(* store.py:_write_journal_row: land the row unless one with a higher or   *)
(* equal seq exists (the Lte(seq - 1) replace, then find/insert). Its loop *)
(* decides each pass on one read, so it is one seq-max upsert; the         *)
(* RuntimeError after five contended passes is not modeled.                *)
RowLands(k, row) == jr[k] = NoRow \/ jr[k].seq < row.seq \/ Mut = "NoJournalSeqGuard"
Upsert(k, row) == IF RowLands(k, row) THEN [jr EXCEPT ![k] = row] ELSE jr

Init ==
    /\ doc = [exists |-> FALSE, rev |-> 0, gen |-> 0, epoch |-> 0, seq |-> 0,
              status |-> "none", ctx |-> EmptyCtx, done |-> EmptyDone, stTok |-> None]
    /\ jr = [k \in Keys |-> NoRow]
    /\ era = 0
    /\ killed = [k \in KillKeys |-> 0]
    /\ aera = [p \in Procs |-> 0]
    /\ dpc = [p \in Procs |-> "start"]
    /\ ep = [p \in Procs |-> 0]
    /\ sdone = [p \in Procs |-> {}]
    /\ retried = [p \in Procs |-> FALSE]
    /\ ferr = [p \in Procs |-> FALSE]
    /\ span = [p \in Procs |-> {}]
    /\ uRev = [p \in Procs |-> 0]
    /\ uCtx = [p \in Procs |-> EmptyCtx]
    /\ uDone = [p \in Procs |-> EmptyDone]
    /\ uSeqs = [p \in Procs |-> {}]
    /\ uRm = [p \in Procs |-> FALSE]
    /\ uEpoch = [p \in Procs |-> 0]
    /\ sSeq = [p \in Procs |-> 0]
    /\ sGen = [p \in Procs |-> 0]
    /\ thr = [p \in Procs |-> [s \in Steps |-> [i \in Slots |-> IdleSlot]]]

-----------------------------------------------------------------------------
(* Registration and seed: task.py:_run -> store.py:register_task_instance  *)

(* store.py:upsert_instance. Insert, or on DuplicateKeyError one atomic    *)
(* update that resets the run state and incs rev, run_generation and       *)
(* attempt_epoch (journal_seq is not reset). start_step = 0, so            *)
(* _journal_seeded_completions writes nothing.                             *)
Register(p) ==
    /\ dpc[p] = "start"
    /\ era' = era + 1
    /\ killed' = [k \in KillKeys |-> era + 1]
    /\ aera' = [aera EXCEPT ![p] = era + 1]
    /\ IF ~doc.exists
         THEN /\ doc' = [doc EXCEPT !.exists = TRUE, !.status = "running"]
              /\ ep' = [ep EXCEPT ![p] = 0]
         ELSE /\ doc' = [doc EXCEPT !.rev = @ + 1, !.gen = @ + 1, !.epoch = @ + 1,
                                    !.status = "running", !.ctx = EmptyCtx,
                                    !.done = EmptyDone, !.stTok = None]
              /\ ep' = [ep EXCEPT ![p] = doc.epoch + 1]
    /\ dpc' = [dpc EXCEPT ![p] = "seedDoc"]
    /\ UNCHANGED <<jr, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

(* store.py:seed_context, first call: store.update of the instance by      *)
(* instance_id and Lte(attempt_epoch), the epoch Task._run registered at.  *)
(* A filter that matches nothing is no error: the journal call still runs. *)
SeedDoc(p) ==
    /\ dpc[p] = "seedDoc"
    /\ IF doc.epoch > ep[p] /\ Mut # "NoSeedDocFence"
         THEN /\ dpc' = [dpc EXCEPT ![p] = "seedBump"]
              /\ UNCHANGED doc
         ELSE /\ doc' = [doc EXCEPT !.ctx["seed"] = Tok(p)]
              /\ dpc' = [dpc EXCEPT ![p] = "seedBump"]
    /\ UNCHANGED <<jr, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

(* store.py:_journal_step(__seed__, attempt_epoch) -> _bump_journal_seq:   *)
(* atomic inc of journal_seq, returns the doc; then the epoch check.       *)
SeedBump(p) ==
    /\ dpc[p] = "seedBump"
    /\ doc' = [doc EXCEPT !.seq = @ + 1]
    /\ IF doc.epoch > ep[p] /\ Mut # "NoSeedJournalFence"
         THEN /\ dpc' = [dpc EXCEPT ![p] = "run"]
              /\ UNCHANGED <<sSeq, sGen>>
         ELSE /\ sSeq' = [sSeq EXCEPT ![p] = doc.seq + 1]
              /\ sGen' = [sGen EXCEPT ![p] = doc.gen]
              /\ dpc' = [dpc EXCEPT ![p] = "seedRow"]
    /\ UNCHANGED <<jr, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, thr>>

(* store.py:_write_journal_row for __seed__.                               *)
SeedRow(p) ==
    /\ dpc[p] = "seedRow"
    /\ jr' = Upsert("seed", [seq |-> sSeq[p], gen |-> sGen[p], val |-> Tok(p)])
    /\ dpc' = [dpc EXCEPT ![p] = "run"]
    /\ UNCHANGED <<doc, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

-----------------------------------------------------------------------------
(* Steps: task.py:_drive_dag dispatch and _run_one                         *)

Ready(p) == {s \in Steps : s \notin sdone[p] /\ Deps[s] \subseteq sdone[p] /\ ~InFlight(p, s)}
NextSlot(p, s) == IF thr[p][s][1].pc = "idle" THEN 1 ELSE 2

(* _drive_dag's `while state.ready` loop creates a task for every ready    *)
(* step before it awaits, so they are dispatched together, all under       *)
(* state.epoch.                                                            *)
Dispatch(p) ==
    /\ dpc[p] = "run"
    /\ ~ferr[p]
    /\ Ready(p) # {}
    /\ \A s \in Ready(p) : thr[p][s][NextSlot(p, s)].pc = "idle"
    /\ thr' = [thr EXCEPT ![p] = [s \in Steps |->
                 IF s \in Ready(p)
                   THEN [thr[p][s] EXCEPT ![NextSlot(p, s)] =
                           [IdleSlot EXCEPT !.pc = "exec", !.dep = ep[p], !.born = aera[p]]]
                   ELSE thr[p][s]]]
    /\ UNCHANGED <<doc, jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* step.execute returns; _heartbeat_step -> record_step_complete ->        *)
(* asyncio.to_thread(record_step_complete_sync).                           *)
ExecOk(p, s, i) ==
    /\ thr[p][s][i].pc = "exec"
    /\ thr' = [thr EXCEPT ![p][s][i].pc = "bump"]
    /\ UNCHANGED <<doc, jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* step.execute raises; b is fatal, so no heartbeat is written.            *)
ExecFail(p, i) ==
    /\ p \in RetryProcs
    /\ thr[p]["b"][i].pc = "exec"
    /\ thr' = [thr EXCEPT ![p]["b"][i].pc = "failed"]
    /\ UNCHANGED <<doc, jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* store.py:_journal_step -> _bump_journal_seq, then the attempt_epoch     *)
(* check on the returned doc. A fenced write returns None (not journaled). *)
Bump(p, s, i) ==
    LET sl == thr[p][s][i] IN
    /\ sl.pc = "bump"
    /\ IF Mut = "FenceBeforeSeq"
         THEN /\ thr' = [thr EXCEPT ![p][s][i].pc = IF doc.epoch > sl.dep THEN "cas" ELSE "fence"]
              /\ UNCHANGED doc
         ELSE /\ doc' = [doc EXCEPT !.seq = @ + 1]
              /\ IF doc.epoch > sl.dep
                   THEN thr' = [thr EXCEPT ![p][s][i].pc = "cas"]
                   ELSE thr' = [thr EXCEPT ![p][s][i] = [sl EXCEPT !.pc = "row", !.seq = doc.seq + 1, !.jg = doc.gen]]
    /\ UNCHANGED <<jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* Mutation FenceBeforeSeq only: the seq is bumped after the fence read.   *)
FenceSeq(p, s, i) ==
    LET sl == thr[p][s][i] IN
    /\ sl.pc = "fence"
    /\ doc' = [doc EXCEPT !.seq = @ + 1]
    /\ thr' = [thr EXCEPT ![p][s][i] = [sl EXCEPT !.pc = "row", !.seq = doc.seq + 1, !.jg = doc.gen]]
    /\ UNCHANGED <<jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* store.py:_write_journal_row for the step.                               *)
Row(p, s, i) ==
    LET sl == thr[p][s][i]
        row == [seq |-> sl.seq, gen |-> sl.jg, val |-> SlotTok(p, s, i)] IN
    /\ sl.pc = "row"
    /\ jr' = Upsert(s, row)
    /\ thr' = [thr EXCEPT ![p][s][i] = [sl EXCEPT !.pc = "cas", !.jok = RowLands(s, row)]]
    /\ UNCHANGED <<doc, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* record_step_complete_sync._mutate's refusals.                           *)
Declines(d, sl) ==
    \/ sl.jok /\ d.gen # sl.jg /\ Mut # "NoCasGenerationCheck"
    \/ d.epoch > sl.dep /\ Mut # "NoCasEpochFence"

Complete(d, s, tok, base) ==
    LET nd == [base EXCEPT ![s] = tok] IN
    [d EXCEPT !.rev = @ + 1, !.ctx[s] = tok, !.done = nd,
              !.status = IF (\A x \in Steps : nd[x] # None) /\ d.status \notin {"failed", "cancelled"}
                           THEN "completed" ELSE @]

(* store.py:record_step_complete_sync -> document_store.compare_and_swap   *)
(* on rev. The CAS re-runs _mutate on a fresh read until the conditional   *)
(* update matches, so it is one atomic read-modify-write; the RuntimeError *)
(* after max_attempts is not modeled.                                      *)
Cas(p, s, i) ==
    LET sl == thr[p][s][i] IN
    /\ sl.pc = "cas"
    /\ IF Mut = "NoCas"
         THEN /\ thr' = [thr EXCEPT ![p][s][i] = [sl EXCEPT !.pc = "casw", !.rd = doc.done, !.rdecl = Declines(doc, sl)]]
              /\ UNCHANGED doc
         ELSE IF Declines(doc, sl)
           THEN /\ thr' = [thr EXCEPT ![p][s][i].pc = IF sl.jok THEN "clean" ELSE "sup"]
                /\ UNCHANGED doc
           ELSE /\ doc' = Complete(doc, s, SlotTok(p, s, i), doc.done)
                /\ thr' = [thr EXCEPT ![p][s][i].pc = "ok"]
    /\ UNCHANGED <<jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* Mutation NoCas only: the update lands with no rev precondition, built   *)
(* from the earlier read.                                                  *)
CasW(p, s, i) ==
    LET sl == thr[p][s][i] IN
    /\ sl.pc = "casw"
    /\ IF sl.rdecl
         THEN /\ thr' = [thr EXCEPT ![p][s][i].pc = IF sl.jok THEN "clean" ELSE "sup"]
              /\ UNCHANGED doc
         ELSE /\ doc' = Complete(doc, s, SlotTok(p, s, i), sl.rd)
              /\ thr' = [thr EXCEPT ![p][s][i].pc = "ok"]
    /\ UNCHANGED <<jr, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* record_step_complete_sync: a refused completion deletes its own row at  *)
(* its exact seq, then raises SupersededAttemptError.                      *)
Clean(p, s, i) ==
    LET sl == thr[p][s][i] IN
    /\ sl.pc = "clean"
    /\ jr' = IF jr[s] # NoRow /\ jr[s].seq = sl.seq THEN [jr EXCEPT ![s] = NoRow] ELSE jr
    /\ thr' = [thr EXCEPT ![p][s][i].pc = "sup"]
    /\ UNCHANGED <<doc, era, killed, aera, dpc, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

(* task.py:_drain_in_flight: cancel every in-flight task. A task still in  *)
(* step.execute stops with no write; one inside to_thread keeps running    *)
(* (orphan) and its write lands late.                                      *)
Drained(p) ==
    [thr EXCEPT ![p] = [s \in Steps |-> [i \in Slots |->
        LET sl == thr[p][s][i] IN
        IF sl.orphan \/ sl.pc \notin ActivePcs THEN sl
        ELSE IF sl.pc \in WritePcs THEN [sl EXCEPT !.orphan = TRUE]
        ELSE [sl EXCEPT !.pc = "done"]]]]

(* task.py:_drive_dag classifies one finished task (asyncio.wait).         *)
Collect(p, s, i) ==
    LET sl == thr[p][s][i]
        thr1 == [thr EXCEPT ![p][s][i].pc = "done"] IN
    /\ dpc[p] = "run"
    /\ ~sl.orphan
    /\ sl.pc \in {"ok", "sup", "failed"}
    /\ CASE sl.pc = "ok" ->
              /\ sdone' = [sdone EXCEPT ![p] = @ \cup {s}]
              /\ thr' = thr1
              /\ UNCHANGED <<dpc, retried, ferr, span>>
         [] sl.pc = "sup" \/ (sl.pc = "failed" /\ retried[p]) ->
              /\ ferr' = [ferr EXCEPT ![p] = TRUE]
              /\ thr' = thr1
              /\ UNCHANGED <<dpc, sdone, retried, span>>
         [] sl.pc = "failed" /\ ferr[p] ->
              /\ thr' = thr1
              /\ UNCHANGED <<dpc, sdone, retried, ferr, span>>
         [] OTHER ->
              \* b failed with retries left: _overlapping_survivors aborts when c completed while b ran.
              /\ retried' = [retried EXCEPT ![p] = TRUE]
              /\ IF "c" \in sdone[p]
                   THEN /\ ferr' = [ferr EXCEPT ![p] = TRUE]
                        /\ thr' = thr1
                        /\ UNCHANGED <<dpc, sdone, span>>
                   ELSE /\ span' = [span EXCEPT ![p] = {"b"} \cup {x \in Steps : x # "b" /\ InFlight(p, x)}]
                        /\ thr' = Drained(p)
                        /\ dpc' = [dpc EXCEPT ![p] = "undoRead"]
                        /\ UNCHANGED <<sdone, ferr>>
    /\ UNCHANGED <<doc, jr, era, killed, aera, ep, undoVars, sSeq, sGen>>

-----------------------------------------------------------------------------
(* Retry: task.py:_roll_back_span -> store.py:undo_steps_sync              *)

(* compare_and_swap's find_one and _mutate's journal read. A journal write *)
(* between the two reads is the same as one before both: it does not move  *)
(* rev, and the update below is conditional on rev.                        *)
UndoRead(p) ==
    LET g == doc.gen
        E == {k \in Keys : jr[k] # NoRow /\ jr[k].gen = g}
        keep == [s \in Steps |-> IF s \in span[p] THEN None ELSE doc.done[s]]
        unreplayable == {s \in Steps : doc.done[s] # None /\ s \notin span[p] /\ s \notin E}
        refuse == \/ doc.epoch > ep[p] /\ Mut # "NoUndoFence"
                  \/ "seed" \notin E
                  \/ unreplayable # {}
        replayed == [k \in Keys |->
                       IF k = "sched" THEN Tok(p)   \* extra_ops: the retry audit, applied over the replay
                       ELSE IF k \in E /\ (k = "seed" \/ (k \in Steps /\ keep[k] # None)) THEN jr[k].val
                       ELSE None] IN
    /\ dpc[p] = "undoRead"
    /\ IF refuse
         THEN /\ ferr' = [ferr EXCEPT ![p] = TRUE]
              /\ dpc' = [dpc EXCEPT ![p] = "run"]
              /\ UNCHANGED undoVars
         ELSE /\ uRev' = [uRev EXCEPT ![p] = doc.rev]
              /\ uCtx' = [uCtx EXCEPT ![p] = replayed]
              /\ uDone' = [uDone EXCEPT ![p] = keep]
              /\ uSeqs' = [uSeqs EXCEPT ![p] = {<<s, jr[s].seq>> : s \in span[p] \cap E}]
              /\ uRm' = [uRm EXCEPT ![p] = \E s \in span[p] : doc.done[s] # None]
              /\ dpc' = [dpc EXCEPT ![p] = "undoWrite"]
              /\ UNCHANGED <<ferr, uEpoch>>
    /\ UNCHANGED <<doc, jr, era, killed, aera, ep, sdone, retried, span, sSeq, sGen, thr>>

(* compare_and_swap's update_one_and_get with the rev precondition; on a   *)
(* miss it reads again. bump_epoch=True advances attempt_epoch.            *)
UndoWrite(p) ==
    LET newEpoch == IF Mut = "NoUndoEpochBump" THEN doc.epoch ELSE doc.epoch + 1 IN
    /\ dpc[p] = "undoWrite"
    /\ IF doc.rev = uRev[p]
         THEN /\ doc' = [doc EXCEPT !.rev = @ + 1, !.ctx = uCtx[p], !.done = uDone[p], !.epoch = newEpoch,
                                    !.status = IF uRm[p] /\ @ # "failed" THEN "running" ELSE @]
              /\ era' = era + 1
              /\ killed' = [k \in KillKeys |-> IF k \in span[p] THEN era + 1 ELSE killed[k]]
              /\ aera' = [aera EXCEPT ![p] = era + 1]
              /\ uEpoch' = [uEpoch EXCEPT ![p] = newEpoch]
              /\ dpc' = [dpc EXCEPT ![p] = "undoTrim"]
         ELSE /\ dpc' = [dpc EXCEPT ![p] = "undoRead"]
              /\ UNCHANGED <<doc, era, killed, aera, uEpoch>>
    /\ UNCHANGED <<jr, ep, sdone, retried, ferr, span, uRev, uCtx, uDone, uSeqs, uRm, sSeq, sGen, thr>>

(* undo_steps_sync: delete each undone row at the seq the CAS read.        *)
UndoTrim(p) ==
    /\ dpc[p] = "undoTrim"
    /\ jr' = [k \in Keys |-> IF k \in Steps /\ jr[k] # NoRow /\ <<k, jr[k].seq>> \in uSeqs[p] THEN NoRow ELSE jr[k]]
    /\ dpc' = [dpc EXCEPT ![p] = "undoSchedBump"]
    /\ UNCHANGED <<doc, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

(* _roll_back_span sets state.epoch from the undo's result; _drive_dag     *)
(* then drops the span from state.completed and calls _rearm.              *)
Rearm(p) ==
    /\ ep' = [ep EXCEPT ![p] = uEpoch[p]]
    /\ sdone' = [sdone EXCEPT ![p] = @ \ span[p]]
    /\ dpc' = [dpc EXCEPT ![p] = "run"]

(* undo_steps_sync: _journal_step(__scheduler__, attempt_epoch=result's).  *)
UndoSchedBump(p) ==
    /\ dpc[p] = "undoSchedBump"
    /\ doc' = [doc EXCEPT !.seq = @ + 1]
    /\ IF doc.epoch > uEpoch[p]
         THEN /\ Rearm(p)
              /\ UNCHANGED <<sSeq, sGen>>
         ELSE /\ sSeq' = [sSeq EXCEPT ![p] = doc.seq + 1]
              /\ sGen' = [sGen EXCEPT ![p] = doc.gen]
              /\ dpc' = [dpc EXCEPT ![p] = "undoSchedRow"]
              /\ UNCHANGED <<ep, sdone>>
    /\ UNCHANGED <<jr, era, killed, aera, retried, ferr, span, undoVars, thr>>

UndoSchedRow(p) ==
    /\ dpc[p] = "undoSchedRow"
    /\ jr' = Upsert("sched", [seq |-> sSeq[p], gen |-> sGen[p], val |-> uCtx[p]["sched"]])
    /\ Rearm(p)
    /\ UNCHANGED <<doc, era, killed, aera, retried, ferr, span, undoVars, sSeq, sGen, thr>>

-----------------------------------------------------------------------------
(* Ending the run                                                          *)

(* _drive_dag's loop exits once nothing is in flight.                      *)
Finish(p) ==
    /\ dpc[p] = "run"
    /\ ~AnyInFlight(p)
    /\ ferr[p] \/ sdone[p] = Steps
    /\ dpc' = [dpc EXCEPT ![p] = IF ferr[p] THEN "fail" ELSE "end"]
    /\ UNCHANGED <<doc, jr, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

(* The terminal write: one conditional update on status != completed and,  *)
(* given attempt_epoch, attempt_epoch <= it.                               *)
Terminal(p, status) ==
    /\ doc' = IF doc.status # "completed" /\ (doc.epoch <= ep[p] \/ Mut = "NoTerminalFence")
                THEN [doc EXCEPT !.rev = @ + 1, !.status = status, !.stTok = Tok(p)]
                ELSE doc
    /\ dpc' = [dpc EXCEPT ![p] = "end"]

(* task.py:_handle_failure -> store.py:record_task_failure_sync.           *)
FailWrite(p) ==
    /\ dpc[p] = "fail"
    /\ Terminal(p, "failed")
    /\ UNCHANGED <<jr, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

(* A CancelledError reaches _drive_dag: _drain_in_flight, then            *)
(* store.py:record_task_cancelled_sync(attempt_epoch=state.epoch).         *)
CancelRun(p) ==
    /\ p \in CancelProcs
    /\ dpc[p] = "run"
    /\ ~ferr[p]
    /\ thr' = Drained(p)
    /\ dpc' = [dpc EXCEPT ![p] = "cancelWrite"]
    /\ UNCHANGED <<doc, jr, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen>>

CancelWrite(p) ==
    /\ dpc[p] = "cancelWrite"
    /\ Terminal(p, "cancelled")
    /\ UNCHANGED <<jr, era, killed, aera, ep, sdone, retried, ferr, span, undoVars, sSeq, sGen, thr>>

Done ==
    /\ \A p \in Procs : dpc[p] = "end"
    /\ \A p \in Procs, s \in Steps, i \in Slots : thr[p][s][i].pc \in {"idle", "done", "ok", "sup"}
    /\ UNCHANGED vars

Next ==
    \/ \E p \in Procs :
         \/ Register(p) \/ SeedDoc(p) \/ SeedBump(p) \/ SeedRow(p)
         \/ UndoRead(p) \/ UndoWrite(p) \/ UndoTrim(p) \/ UndoSchedBump(p) \/ UndoSchedRow(p)
         \/ Finish(p) \/ FailWrite(p) \/ CancelRun(p) \/ CancelWrite(p)
         \/ Dispatch(p)
         \/ \E i \in Slots : ExecFail(p, i)
         \/ \E s \in Steps, i \in Slots :
              ExecOk(p, s, i) \/ Bump(p, s, i) \/ FenceSeq(p, s, i) \/ Row(p, s, i)
              \/ Cas(p, s, i) \/ CasW(p, s, i) \/ Clean(p, s, i) \/ Collect(p, s, i)
    \/ Done

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* INVARIANTS                                                              *)

(* A write from a rolled-back or superseded attempt never lands: every     *)
(* context value, every completed_steps entry, and a failed/cancelled      *)
(* status on the instance was written by an attempt that no undo or        *)
(* re-register has since retired.                                          *)
NoSupersededWriteLands ==
    /\ \A k \in Keys : doc.ctx[k] # None => doc.ctx[k].born >= killed[k]
    /\ \A s \in Steps : doc.done[s] # None => doc.done[s].born >= killed[s]
    /\ doc.status \in {"failed", "cancelled"} => doc.stTok.born >= killed["status"]

(* replay_context(commit_ordered(entries of the live generation)): the     *)
(* seed, the scheduler record, and each completed step's row.              *)
Replay == [k \in Keys |->
             IF jr[k] # NoRow /\ jr[k].gen = doc.gen
                /\ (k \in {"seed", "sched"} \/ (k \in Steps /\ doc.done[k] # None))
             THEN jr[k].val ELSE None]

(* Between a doc-first write and its journal row (seed_context, the        *)
(* scheduler record) the two are knowingly apart.                          *)
Quiescent == \A p \in Procs : dpc[p] \notin {"seedDoc", "seedBump", "seedRow", "undoTrim", "undoSchedBump", "undoSchedRow"}

(* Replaying the live generation's journal over the seed reproduces the    *)
(* stored context, whenever no seed or scheduler write is half done and    *)
(* the run is not failed or cancelled (record_task_failure_sync writes the *)
(* in-memory context unjournaled, by design).                              *)
ReplayReproducesStored ==
    (Quiescent /\ doc.exists /\ doc.status \notin {"failed", "cancelled"}) => Replay = doc.ctx

=============================================================================
