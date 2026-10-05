/** Runtime Triggers panel for the instance viewer. Renders the trigger ledger a run wrote into
 *  context.metadata — registrations, final env-trigger statuses, and the per-turn record. Presentation-only
 *  (the viewer parses via parseTriggerRuntime). */
import React from 'react';
import { Badge, Callout, Popover, Table, Text } from '@radix-ui/themes';
import { AlertTriangle, Zap } from 'lucide-react';
import { ConfigDetail, CopyButton, truncate } from './flow-shared';
import { ENV_COLOR, AGENT_COLOR } from './triggers-graph';
import { TriggerTimeline } from './trigger-timeline';
import { triggerHistoryFor, turnRowFor } from '../lib/parse-trigger-runtime';
import type {
  TriggerRuntime,
  TriggerRegistrationRow,
  EnvStateSummary,
  StepTriggerRuntime,
  TurnDelta,
} from '../lib/parse-trigger-runtime';
import type { AuthoredTriggerDetail } from '../lib/parse-triggers';

type BadgeColor = React.ComponentProps<typeof Badge>['color'];

const STATUS_COLORS: Record<string, BadgeColor> = {
  armed: 'blue',
  firing: 'amber',
  fired: 'green',
  failed: 'red',
  disabled: 'gray',
  removed: 'gray',
};

function statusColor(status: string): BadgeColor {
  return STATUS_COLORS[status] ?? 'gray';
}

/** Whether a fire count is worth showing beside a status. ×1 on a terminal `fired` is noise, but a re-armed
 *  recurrence returns to `armed` after each arrival, so any count on a non-`fired` status is load-bearing. */
function showFireCount(
  fireCount: number | undefined,
  status: string | undefined,
): fireCount is number {
  return (
    fireCount !== undefined &&
    fireCount >= 1 &&
    (fireCount > 1 || status !== 'fired')
  );
}

function KindDot({ kind }: { kind: 'env' | 'agent' }) {
  return (
    <span
      className="w-2 h-2 rounded-full flex-shrink-0 inline-block"
      style={{ backgroundColor: kind === 'env' ? ENV_COLOR : AGENT_COLOR }}
    />
  );
}

function RegistrationsTable({ rows }: { rows: TriggerRegistrationRow[] }) {
  if (rows.length === 0) return null;
  return (
    <Table.Root variant="ghost" size="1">
      <Table.Header>
        <Table.Row>
          <Table.ColumnHeaderCell>Kind</Table.ColumnHeaderCell>
          <Table.ColumnHeaderCell>Source</Table.ColumnHeaderCell>
          <Table.ColumnHeaderCell>Triggers</Table.ColumnHeaderCell>
          <Table.ColumnHeaderCell>Executor</Table.ColumnHeaderCell>
          <Table.ColumnHeaderCell>Registered by</Table.ColumnHeaderCell>
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {rows.map((r, i) => (
          <Table.Row key={i}>
            <Table.Cell>
              <span className="flex items-center gap-1.5">
                <KindDot kind={r.kind} />
                {r.kind}
              </span>
            </Table.Cell>
            <Table.Cell>
              <span className="font-mono text-[11px]">{r.groupKey}</span>
            </Table.Cell>
            <Table.Cell>
              <span className="flex flex-wrap gap-1">
                {r.added.map(id => (
                  <Badge
                    key={id}
                    color={r.kind === 'env' ? 'teal' : 'pink'}
                    variant="soft"
                    size="1"
                  >
                    {id}
                  </Badge>
                ))}
              </span>
            </Table.Cell>
            <Table.Cell>
              <span className="font-mono text-[11px]">
                {r.executorAgentName ?? '—'}
              </span>
            </Table.Cell>
            <Table.Cell>
              <span className="font-mono text-[10px] text-[var(--muted-foreground)]">
                {truncate(r.stepId, 44)}
              </span>
            </Table.Cell>
          </Table.Row>
        ))}
      </Table.Body>
    </Table.Root>
  );
}

function EnvStateSection({ envState }: { envState: EnvStateSummary[] }) {
  if (envState.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      <Text size="1" weight="medium">
        Final env-trigger state
      </Text>
      {envState.map(s => (
        <div
          key={s.envId}
          className="rounded border border-[var(--border)] px-3 py-2 flex flex-col gap-1.5"
        >
          <div className="flex items-center gap-2 flex-wrap">
            <KindDot kind="env" />
            <span className="font-mono text-[11px]">{s.envId}</span>
            {s.eventCount != null && (
              <span className="text-[10px] text-[var(--muted-foreground)]">
                {s.eventCount} events
                {s.eventsDropped != null && ` (${s.eventsDropped} dropped)`}
              </span>
            )}
            {s.capturedAtUtc && (
              <span className="text-[10px] text-[var(--muted-foreground)]">
                captured {s.capturedAtUtc}
              </span>
            )}
            {s.captureIsFinal === false && (
              <Badge color="amber" variant="soft" size="1">
                capture not final
              </Badge>
            )}
          </div>
          {s.error ? (
            <Callout.Root color="red" size="1">
              <Callout.Icon>
                <AlertTriangle size={13} />
              </Callout.Icon>
              <Callout.Text>capture failed: {s.error}</Callout.Text>
            </Callout.Root>
          ) : (
            <div className="flex flex-wrap gap-1">
              {Object.entries(s.statuses).map(([id, status]) => {
                const n = s.triggers?.[id]?.fireCount;
                return (
                  <Badge
                    key={id}
                    color={statusColor(status)}
                    variant="soft"
                    size="1"
                  >
                    {id}: {status}
                    {showFireCount(n, status) && ` ×${n}`}
                  </Badge>
                );
              })}
            </div>
          )}
          {s.objectUrl && (
            <div className="flex items-center gap-1">
              <span className="font-mono text-[10px] text-[var(--muted-foreground)] break-all">
                {s.objectUrl}
              </span>
              <CopyButton text={s.objectUrl} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** Data the badge popovers need: the parsed runtime ledger plus the
 *  authored-config index (absent → runtime-only detail). */
interface TriggerDetailCtx {
  runtime: TriggerRuntime;
  authored?: Map<string, AuthoredTriggerDetail>;
}

function TriggerBadgePopover({
  ctx,
  kind,
  triggerId,
  envId,
  children,
}: {
  ctx: TriggerDetailCtx;
  kind: 'env' | 'agent';
  triggerId: string;
  envId?: string;
  children: React.ReactNode;
}) {
  const authored = ctx.authored?.get(
    kind === 'env' ? `env:${envId ?? ''}:${triggerId}` : `agent:${triggerId}`,
  );
  const history = triggerHistoryFor(ctx.runtime, kind, triggerId, envId);
  return (
    <Popover.Root>
      <Popover.Trigger>
        <button
          type="button"
          className="appearance-none border-0 bg-transparent p-0 m-0 cursor-pointer"
          aria-label={`Trigger ${triggerId} details`}
        >
          {children}
        </button>
      </Popover.Trigger>
      <Popover.Content size="1" style={{ maxWidth: 460 }}>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1.5 flex-wrap">
            <KindDot kind={kind} />
            <span className="font-mono text-[12px] font-semibold">
              {triggerId}
            </span>
            <span className="font-mono text-[10px] text-[var(--muted-foreground)]">
              {kind} · {authored?.groupKey ?? envId}
            </span>
            {authored && (
              <CopyButton text={JSON.stringify(authored.raw, null, 2)} />
            )}
          </div>
          {authored ? (
            <div className="max-h-80 overflow-y-auto overflow-x-auto pr-1">
              <ConfigDetail obj={authored.raw} omitKeys={['id']} />
            </div>
          ) : (
            <Text size="1" color="gray">
              Authored config not available on this surface — see the trigger
              graph on the task page.
            </Text>
          )}
          {(history.trail.length > 0 ||
            history.firedTurns.length > 0 ||
            history.finalStatus ||
            history.registeredBy) && (
            <div className="flex flex-col gap-1 border-t border-[var(--border)] pt-2">
              {history.trail.length > 0 && (
                <span className="flex items-center gap-1 flex-wrap">
                  {history.trail.map((h, i) => (
                    <Badge
                      key={i}
                      color={statusColor(h.to)}
                      variant="soft"
                      size="1"
                    >
                      <span className="font-mono">
                        t{h.turn} {h.from ? `${h.from}→${h.to}` : h.to}
                      </span>
                    </Badge>
                  ))}
                </span>
              )}
              {history.firedTurns.length > 0 && (
                <Text size="1">
                  fired on turn{history.firedTurns.length > 1 ? 's' : ''}{' '}
                  {history.firedTurns.join(', ')}
                </Text>
              )}
              {history.finalStatus && (
                <span className="flex items-center gap-1">
                  <Text size="1" color="gray">
                    final captured status
                  </Text>
                  <Badge
                    color={statusColor(history.finalStatus)}
                    variant="soft"
                    size="1"
                  >
                    {history.finalStatus}
                    {showFireCount(history.fireCount, history.finalStatus) &&
                      ` ×${history.fireCount}`}
                  </Badge>
                </span>
              )}
              {history.nextMark && (
                <Text size="1" color="gray">
                  next mark (virtual clock){' '}
                  <span className="font-mono text-[11px]">
                    {history.nextMark}
                  </span>
                </Text>
              )}
              {history.registeredBy && (
                <span className="font-mono text-[10px] text-[var(--muted-foreground)]">
                  registered by {truncate(history.registeredBy, 44)}
                </span>
              )}
            </div>
          )}
        </div>
      </Popover.Content>
    </Popover.Root>
  );
}

function DeltaBadges({
  deltas,
  ctx,
}: {
  deltas: TurnDelta[];
  ctx?: TriggerDetailCtx;
}) {
  return (
    <span className="flex flex-wrap gap-1">
      {deltas.map((d, i) => {
        const badge = (
          <Badge color={statusColor(d.to)} variant="soft" size="1">
            <span className="font-mono">
              {d.triggerId} {d.from ? `${d.from}→${d.to}` : d.to}
            </span>
          </Badge>
        );
        return ctx ? (
          <TriggerBadgePopover
            key={i}
            ctx={ctx}
            kind="env"
            triggerId={d.triggerId}
            envId={d.envId}
          >
            {badge}
          </TriggerBadgePopover>
        ) : (
          <React.Fragment key={i}>{badge}</React.Fragment>
        );
      })}
    </span>
  );
}

function FiredBadges({
  fired,
  ctx,
}: {
  fired: string[];
  ctx?: TriggerDetailCtx;
}) {
  return (
    <span className="flex flex-wrap gap-1">
      {fired.map(id => {
        const badge = (
          <Badge color="violet" variant="solid" size="1">
            {id}
          </Badge>
        );
        return ctx ? (
          <TriggerBadgePopover key={id} ctx={ctx} kind="agent" triggerId={id}>
            {badge}
          </TriggerBadgePopover>
        ) : (
          <React.Fragment key={id}>{badge}</React.Fragment>
        );
      })}
    </span>
  );
}

function UsersimChips({ fields }: { fields: Record<string, unknown> }) {
  return (
    <span className="flex flex-wrap gap-1">
      {Object.entries(fields).map(([k, v]) => (
        <span
          key={k}
          className="inline-block px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[10px]"
        >
          <span className="text-[var(--muted-foreground)]">{k}:</span>{' '}
          {String(v ?? '—')}
        </span>
      ))}
    </span>
  );
}

function PerTurnTable({
  step,
  ctx,
}: {
  step: StepTriggerRuntime;
  ctx: TriggerDetailCtx;
}) {
  if (step.turns.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <Text size="1" weight="medium">
          Per-turn record
        </Text>
        <span className="font-mono text-[10px] text-[var(--muted-foreground)]">
          {truncate(step.stepId, 44)}
        </span>
      </div>
      {!step.hasPerTurnData && (
        <Text size="1" color="gray">
          Env-trigger snapshots unavailable for this run — showing agent firings
          only.
        </Text>
      )}
      {!step.firingLogAvailable && (
        <Text size="1" color="gray">
          Agent firing log unavailable (end-of-run readback did not complete).
        </Text>
      )}
      <Table.Root variant="ghost" size="1">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeaderCell>Turn</Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>Env trigger changes</Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>
              Agent triggers fired
            </Table.ColumnHeaderCell>
            <Table.ColumnHeaderCell>User-sim</Table.ColumnHeaderCell>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {step.turns.map(t => (
            <Table.Row key={t.turn}>
              <Table.Cell>{t.turn}</Table.Cell>
              <Table.Cell>
                {t.deltas.length > 0 ? (
                  <DeltaBadges deltas={t.deltas} ctx={ctx} />
                ) : (
                  <span className="text-[var(--muted-foreground)]">—</span>
                )}
              </Table.Cell>
              <Table.Cell>
                {t.fired.length > 0 ? (
                  <FiredBadges fired={t.fired} ctx={ctx} />
                ) : (
                  <span className="text-[var(--muted-foreground)]">—</span>
                )}
              </Table.Cell>
              <Table.Cell>
                {t.usersim ? (
                  <UsersimChips fields={t.usersim} />
                ) : (
                  <span className="text-[var(--muted-foreground)]">—</span>
                )}
              </Table.Cell>
            </Table.Row>
          ))}
        </Table.Body>
      </Table.Root>
    </div>
  );
}

/** Quoted, clickable preview of the reply authored after a turn (user-sim message or trigger-injected `say`). The popover shows the full text + telemetry. */
function ReplyPopover({
  title,
  reply,
  turn,
  fields,
  note,
}: {
  title: string;
  reply: string;
  turn: number;
  fields?: Record<string, unknown>;
  note?: string;
}) {
  return (
    <Popover.Root>
      <Popover.Trigger>
        <button
          type="button"
          className="appearance-none border-0 bg-transparent p-0 m-0 cursor-pointer text-left"
          aria-label={`${title} after turn ${turn}`}
        >
          <span className="text-[11px] italic text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors">
            “{truncate(reply.replace(/\s+/g, ' '), 88)}”
          </span>
        </button>
      </Popover.Trigger>
      <Popover.Content size="1" style={{ maxWidth: 460 }}>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[12px] font-semibold">{title}</span>
            <span className="text-[10px] text-[var(--muted-foreground)]">
              after turn {turn}
            </span>
            <CopyButton text={reply} />
          </div>
          {note && (
            <Text size="1" color="gray">
              {note}
            </Text>
          )}
          <p className="text-[12px] whitespace-pre-wrap max-h-80 overflow-y-auto pr-1">
            {reply}
          </p>
          {fields && (
            <div className="border-t border-[var(--border)] pt-2">
              <UsersimChips fields={fields} />
            </div>
          )}
        </div>
      </Popover.Content>
    </Popover.Root>
  );
}

/** Compact trigger summary for one trajectory turn section. `turnIndex` is 0-based; the ledger's reaction
 *  turns are 1-based, and the +1 lives here only. Null when the join misses. `nextPromptText` is the reply authored after this turn. */
export function TriggerTurnStrip({
  runtime,
  stepId,
  turnIndex,
  authored,
  nextPromptText,
}: {
  runtime: TriggerRuntime | null;
  stepId?: string;
  turnIndex?: number;
  authored?: Map<string, AuthoredTriggerDetail>;
  nextPromptText?: string;
}) {
  const row = turnRowFor(
    runtime,
    stepId,
    turnIndex === undefined ? undefined : turnIndex + 1,
  );
  if (!row || !runtime) return null;
  const ctx: TriggerDetailCtx = { runtime, authored };
  const reply = nextPromptText?.trim() ? nextPromptText.trim() : undefined;
  const speaker =
    typeof row.usersim?.speaker === 'string' ? row.usersim.speaker : undefined;
  return (
    <div className="mt-3 rounded-md border border-[var(--border)] px-3 py-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <span className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
        <Zap size={11} />
        Turn {row.turn}
      </span>
      {row.deltas.length > 0 && (
        <span className="flex items-center gap-1.5 flex-wrap">
          <KindDot kind="env" />
          <span className="text-[10px] text-[var(--muted-foreground)]">
            env · during turn
          </span>
          <DeltaBadges deltas={row.deltas} ctx={ctx} />
        </span>
      )}
      {row.fired.length > 0 && (
        <span className="flex items-center gap-1.5 flex-wrap">
          <KindDot kind="agent" />
          <span className="text-[10px] text-[var(--muted-foreground)]">
            agent · after turn
          </span>
          <FiredBadges fired={row.fired} ctx={ctx} />
        </span>
      )}
      {row.usersim && (
        <span className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] text-[var(--muted-foreground)]">
            user-sim{speaker ? ` (${speaker})` : ''} replied
          </span>
          {reply ? (
            <ReplyPopover
              title="User-sim reply"
              reply={reply}
              turn={row.turn}
              fields={row.usersim}
            />
          ) : (
            <UsersimChips fields={row.usersim} />
          )}
        </span>
      )}
      {!row.usersim && row.fired.length > 0 && reply && (
        <span className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] text-[var(--muted-foreground)]">
            injected reply
          </span>
          <ReplyPopover
            title="Trigger-injected reply"
            reply={reply}
            turn={row.turn}
            note="Typed into the conversation by a fired trigger's say action — not an LLM user-sim reply."
          />
        </span>
      )}
    </div>
  );
}

export function TriggersPanel({
  runtime,
  authored,
  taskId,
  instanceId,
}: {
  runtime: TriggerRuntime;
  authored?: Map<string, AuthoredTriggerDetail>;
  taskId?: string;
  instanceId?: string;
}) {
  const ctx: TriggerDetailCtx = { runtime, authored };
  return (
    <div className="flex flex-col gap-4">
      <Text size="1" color="gray">
        Runtime trigger evidence recorded by this run. Turns are 1-based
        reaction turns: env-trigger changes happen during the turn (provoked by
        the agent&apos;s tool calls), while agent firings and the reply that opens
        the next turn happen after it. A missing user-sim entry on a turn with
        firings marks a trigger-injected (typed) message rather than an LLM
        reply. Click a trigger badge for its condition, actions, and status
        history.
      </Text>
      <RegistrationsTable rows={runtime.registrations} />
      <EnvStateSection envState={runtime.envState} />
      {taskId && instanceId && (
        <TriggerTimeline taskId={taskId} instanceId={instanceId} authored={authored} />
      )}
      {runtime.steps.map(step => (
        <PerTurnTable key={step.stepId} step={step} ctx={ctx} />
      ))}
    </div>
  );
}
