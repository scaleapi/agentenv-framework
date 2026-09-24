/**
 * Event-level env/agent trigger timeline. `useLiveTriggers` fetches and keeps it
 * current while the run is in flight; renders nothing when the run had no events.
 */
'use client';

import React, { useMemo, useState } from 'react';
import { Badge, Callout, Popover, Text } from '@radix-ui/themes';
import {
  AlertTriangle,
  Anchor,
  CheckCircle2,
  MinusCircle,
  PlusCircle,
  Clock,
  HelpCircle,
  Link2,
  Loader2,
  MessageCircle,
  MoreHorizontal,
  Radar,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  XCircle,
  Zap,
} from 'lucide-react';
import { ConfigDetail, CopyButton, truncate } from './flow-shared';
import { AGENT_COLOR, ENV_COLOR } from './triggers-graph';
import { SECTION_HEADER_CLASS } from './shared';
import {
  highlightFor,
  parseTriggerTimeline,
  shortStamp,
  type PendingTrigger,
  type TimelineHighlight,
  type TimelineRow,
  type TriggerTimeline,
} from '../lib/parse-trigger-events';
import { countdownTo, humanizeMs, type EnvClock } from '../lib/trigger-clock';
import { useLiveTriggers } from '../hooks/useLiveTriggers';
import type { AuthoredTriggerDetail } from '../lib/parse-triggers';

type BadgeColor = React.ComponentProps<typeof Badge>['color'];
type LucideIcon = typeof Zap;

const KINDS: Record<string, { color: BadgeColor; Icon: LucideIcon }> = {
  added: { color: 'gray', Icon: PlusCircle },
  removed: { color: 'gray', Icon: MinusCircle },
  detected: { color: 'amber', Icon: Radar },
  anchored: { color: 'cyan', Icon: Anchor },
  reanchored: { color: 'cyan', Icon: RotateCcw },
  action_ok: { color: 'green', Icon: CheckCircle2 },
  action_failed: { color: 'red', Icon: XCircle },
  verify_ok: { color: 'teal', Icon: ShieldCheck },
  verify_failed: { color: 'orange', Icon: ShieldAlert },
  fired: { color: 'violet', Icon: Zap },
  failed: { color: 'red', Icon: XCircle },
  eval_error: { color: 'red', Icon: AlertTriangle },
  registered: { color: 'gray', Icon: PlusCircle },
  reset: { color: 'gray', Icon: RotateCcw },
};

const UNKNOWN = { color: 'gray' as BadgeColor, Icon: HelpCircle };

function ClockHeader({ clocks, live }: { clocks: EnvClock[]; live: boolean }) {
  if (clocks.length === 0) return null;
  return (
    <div className="flex flex-col gap-1 rounded border border-[var(--border)] px-3 py-2">
      {clocks.map(c => (
        <div key={c.envId} className="flex flex-wrap items-center gap-2">
          <Clock size={12} className="text-[var(--muted-foreground)]" />
          {clocks.length > 1 && (
            <span className="font-mono text-[10px]">{c.envId}</span>
          )}
          {c.armed ? (
            <Text size="1" color="gray">
              virtual clock at{' '}
              <span className="font-mono">{c.rate ?? '?'}×</span> real time · t0{' '}
              <span className="font-mono">{shortStamp(c.t0)}</span> · virtual
              time {live ? 'at last read' : 'at capture'}{' '}
              <span className="font-mono">{shortStamp(c.virtualTime)}</span>
            </Text>
          ) : (
            <Text size="1" color="gray">
              virtual clock not armed — events carry no virtual time
            </Text>
          )}
        </div>
      ))}
    </div>
  );
}

/** Armed triggers with a due instant ahead of them. A recurrence re-arms after
 *  every arrival, so this and `fire_count` are the only forward-looking signals
 *  a live run has. */
function PendingMarks({
  pending,
  clocks,
  live,
}: {
  pending: PendingTrigger[];
  clocks: EnvClock[];
  live: boolean;
}) {
  if (pending.length === 0) return null;
  const byEnv = new Map(clocks.map(c => [c.envId, c]));
  return (
    <div className="flex flex-col gap-1 rounded border border-[var(--border)] px-3 py-2">
      {pending.map(p => {
        // The mark is worth showing on a finished run — what it was waiting for
        // is part of the record. A countdown is not: a run cancelled with a
        // recurrence still armed would read as though it were about to fire.
        const clock = live ? byEnv.get(p.envId) : undefined;
        const left = clock ? countdownTo(p.nextMark, clock) : null;
        return (
          <div
            // The mark is in the key so a re-anchor mounts a fresh row rather than
            // letting the old countdown tick on toward a dropped instant.
            key={`${p.groupKey}:${p.nextMark}`}
            className="flex flex-wrap items-center gap-2 text-[11px]"
          >
            <Clock size={12} className="text-[var(--muted-foreground)]" />
            <span className="font-mono">{p.triggerId}</span>
            <Text size="1" color="gray">
              {live ? 'next firing at ' : 'was next due at '}
              <span className="font-mono">{shortStamp(p.nextMark)}</span>
            </Text>
            {left && (
              <Badge
                color={left.due ? 'amber' : 'cyan'}
                variant="soft"
                size="1"
              >
                {left.due
                  ? 'due — arriving'
                  : `in ${humanizeMs(left.virtualMs)} virtual · ~${humanizeMs(
                      left.realMs,
                    )} real`}
              </Badge>
            )}
            {p.status && p.status !== 'armed' && (
              <Badge color="violet" variant="soft" size="1">
                {p.status}
              </Badge>
            )}
            {p.fireCount !== undefined && p.fireCount > 0 && (
              <Text size="1" color="gray">
                fired {p.fireCount}× so far
              </Text>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** The whole log, memoized. The clock reading lands every few seconds and
 *  re-renders the panel around it; without this, each of those ticks would
 *  reconcile every row — up to the gateway's 10k-event cap. */
const TimelineRows = React.memo(function TimelineRows({
  rows,
  highlight,
  selected,
  onSelect,
  authored,
  showVirtual,
  showEnv,
  showConversation,
}: {
  rows: TimelineRow[];
  highlight: TimelineHighlight;
  selected: string | null;
  onSelect: (id: string | null) => void;
  authored?: Map<string, AuthoredTriggerDetail>;
  showVirtual: boolean;
  showEnv: boolean;
  showConversation: boolean;
}) {
  return (
    <ol className="flex flex-col gap-0.5">
      {rows.map(row => {
        const dimmed = selected !== null && !highlight.ids.has(row.id);
        if (row.kind === 'gap') return <GapRow key={row.id} row={row} />;
        if (row.stream === 'conversation') {
          return (
            <ConversationRow
              key={row.id}
              row={row}
              dimmed={dimmed}
              showConversation={showConversation}
            />
          );
        }
        return (
          <EventRow
            key={row.id}
            row={row}
            authored={row.groupKey ? authored?.get(row.groupKey) : undefined}
            dimmed={dimmed}
            selected={selected === row.id}
            edgeLabels={highlight.labels.get(row.id) ?? []}
            onSelect={() => onSelect(selected === row.id ? null : row.id)}
            showVirtual={showVirtual}
            showEnv={showEnv}
          />
        );
      })}
    </ol>
  );
});

function RowStamps({
  row,
  showVirtual,
}: {
  row: TimelineRow;
  showVirtual: boolean;
}) {
  return (
    <span className="flex w-44 shrink-0 flex-col text-right font-mono text-[10px] text-[var(--muted-foreground)]">
      <span>{shortStamp(row.ts)}</span>
      {showVirtual && row.virtualTime && (
        <span className="text-[var(--muted-foreground)] opacity-70">
          v {shortStamp(row.virtualTime)}
        </span>
      )}
    </span>
  );
}

function ConversationRow({
  row,
  dimmed,
  showConversation,
}: {
  row: TimelineRow;
  dimmed: boolean;
  showConversation: boolean;
}) {
  return (
    <li
      className={`flex items-start gap-2 border-t border-dashed border-[var(--border)] px-2 pt-2 ${
        dimmed ? 'opacity-30' : ''
      }`}
    >
      <RowStamps row={row} showVirtual={false} />
      <Badge
        color={row.role === 'user' ? 'blue' : 'violet'}
        variant="soft"
        size="1"
      >
        <MessageCircle size={11} />
        turn {row.turn} · {row.role}
      </Badge>
      {showConversation && row.conversationId && (
        <span className="font-mono text-[10px] text-[var(--muted-foreground)]">
          {truncate(row.conversationId, 8)}
        </span>
      )}
      <span className="flex-1 text-[11px] italic text-[var(--muted-foreground)]">
        “{truncate(row.summary, 110)}”
      </span>
    </li>
  );
}

function GapRow({ row }: { row: TimelineRow }) {
  return (
    <li className="flex items-center gap-2 px-2 py-1">
      <span className="w-44 shrink-0" />
      <Badge color="gray" variant="outline" size="1">
        <MoreHorizontal size={11} />
        {row.summary}
      </Badge>
    </li>
  );
}

function EventRow({
  row,
  authored,
  dimmed,
  selected,
  edgeLabels,
  onSelect,
  showVirtual,
  showEnv,
}: {
  row: TimelineRow;
  authored?: AuthoredTriggerDetail;
  dimmed: boolean;
  selected: boolean;
  edgeLabels: string[];
  onSelect: () => void;
  showVirtual: boolean;
  showEnv: boolean;
}) {
  const { color, Icon } = KINDS[row.kind] ?? UNKNOWN;
  return (
    <li
      className={`flex items-start gap-2 rounded px-2 py-1 transition-opacity ${
        selected ? 'bg-[var(--secondary)]' : ''
      } ${dimmed ? 'opacity-30' : ''}`}
    >
      <RowStamps row={row} showVirtual={showVirtual} />
      <span
        className="mt-1.5 inline-block h-2 w-2 flex-shrink-0 rounded-full"
        style={{
          backgroundColor: row.stream === 'env' ? ENV_COLOR : AGENT_COLOR,
        }}
        title={row.stream === 'env' ? 'env trigger' : 'agent trigger'}
      />
      <button
        type="button"
        onClick={onSelect}
        className="m-0 flex flex-1 cursor-pointer appearance-none items-start gap-2 border-0 bg-transparent p-0 text-left"
        aria-label={`Highlight ${row.triggerId ?? row.kind}`}
      >
        <Badge color={color} variant="soft" size="1">
          <Icon size={11} />
          {row.kind}
        </Badge>
        {showEnv && row.envId && (
          <span className="font-mono text-[10px] text-[var(--muted-foreground)]">
            {row.envId}
          </span>
        )}
        {row.triggerId && (
          <span className="font-mono text-[11px]">
            {truncate(row.triggerId, 22)}
          </span>
        )}
        {row.actionIndex !== undefined && (
          <Badge color="gray" variant="outline" size="1">
            action #{row.actionIndex}
            {row.attempt !== undefined && row.attempt > 1
              ? ` · retry ${row.attempt - 1}`
              : ''}
          </Badge>
        )}
        <span className="flex-1 text-[11px] text-[var(--foreground)]">
          {row.summary}
          {row.reason && (
            <span className="italic text-[var(--muted-foreground)]">
              {' '}
              — because {row.reason}
            </span>
          )}
        </span>
        {edgeLabels.map(label => (
          <Badge
            key={label}
            color="violet"
            variant="outline"
            size="1"
            title={label}
          >
            <Link2 size={11} />
            {truncate(label, 34)}
          </Badge>
        ))}
      </button>
      <Popover.Root>
        <Popover.Trigger>
          <button
            type="button"
            className="m-0 cursor-pointer appearance-none border-0 bg-transparent p-0 font-mono text-[10px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            aria-label={`Raw payload for ${row.kind}`}
          >
            {row.seq === undefined ? 'raw' : `#${row.seq}`}
          </button>
        </Popover.Trigger>
        <Popover.Content size="1" style={{ maxWidth: 460 }}>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <Text size="1" weight="medium">
                {row.kind} · {row.triggerId ?? row.stream}
              </Text>
              <CopyButton text={JSON.stringify(row.raw, null, 2)} />
            </div>
            <div className="max-h-80 overflow-y-auto overflow-x-auto pr-1">
              <ConfigDetail obj={row.raw} />
            </div>
            {authored && (
              <div className="flex flex-col gap-1 border-t border-[var(--border)] pt-2">
                <Text size="1" weight="medium">
                  Authored config
                </Text>
                <div className="max-h-60 overflow-y-auto overflow-x-auto pr-1">
                  <ConfigDetail obj={authored.raw} omitKeys={['id']} />
                </div>
              </div>
            )}
          </div>
        </Popover.Content>
      </Popover.Root>
    </li>
  );
}

export function TriggerTimeline({
  taskId,
  instanceId,
  authored,
}: {
  taskId: string;
  instanceId: string;
  authored?: Map<string, AuthoredTriggerDetail>;
}) {
  const { feed, clocks, conversations } = useLiveTriggers(taskId, instanceId);
  const [selected, setSelected] = useState<string | null>(null);
  const payload = 'payload' in feed ? feed.payload : null;

  // The hook holds payload identity steady while nothing the timeline draws has
  // changed, so this recomputes when the log does and not when the clock ticks.
  const timeline: TriggerTimeline | null = useMemo(
    () =>
      payload
        ? parseTriggerTimeline({
            state: payload.state,
            stateMeta: payload.state_meta,
            stateSources: payload.state_sources,
            agentTriggerState: payload.agent_trigger_state,
            conversations,
            authored,
          })
        : null,
    [payload, conversations, authored],
  );

  const highlight = useMemo(
    () =>
      timeline
        ? highlightFor(timeline, selected)
        : { ids: new Set<string>(), labels: new Map<string, string[]>() },
    [timeline, selected],
  );

  // Only `failed` has nothing to keep on screen; the degraded states still render
  // what they last had, behind a badge saying it is no longer current.
  if (feed.kind === 'failed') {
    return (
      <Text size="1" color="gray">
        Event timeline unavailable — {feed.error}.
      </Text>
    );
  }
  if (!payload) {
    return (
      <Text size="1" color="gray">
        <Loader2 size={12} className="mr-1 inline animate-spin" />
        Loading trigger events…
      </Text>
    );
  }
  if (!timeline) return null;

  const live = payload.status === 'running';

  const showVirtual = timeline.rows.some(r => r.virtualTime);
  const showEnv = timeline.envIds.length > 1;
  const showConversation = timeline.conversationIds.length > 1;
  const source = String(payload.source ?? 'metadata');

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className={SECTION_HEADER_CLASS}>Event timeline</span>
        <Badge
          color={source === 'live' ? 'green' : 'gray'}
          variant="soft"
          size="1"
        >
          {source === 'live' && (
            <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
          )}
          {source}
        </Badge>
        {(feed.kind === 'retrying' || feed.kind === 'stopped') && (
          <Badge
            color={feed.kind === 'stopped' ? 'red' : 'amber'}
            variant="soft"
            size="1"
            title={feed.error}
          >
            {feed.kind === 'stopped'
              ? 'not updating — showing last read'
              : 'reconnecting — showing last read'}
          </Badge>
        )}
        {timeline.eventsDropped > 0 && (
          <Badge color="amber" variant="soft" size="1">
            {timeline.eventsDropped} dropped
          </Badge>
        )}
        {selected && (
          <button
            type="button"
            onClick={() => setSelected(null)}
            className="cursor-pointer appearance-none border-0 bg-transparent p-0 text-[11px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
          >
            clear highlight
          </button>
        )}
      </div>
      <Text size="1" color="gray">
        Every persisted trigger event in order, interleaved with the
        conversation. Env and agent events come from different machines, so rows
        are placed by timestamp while each stream keeps its own{' '}
        <span className="font-mono">seq</span> order. Click a row to highlight
        its chain and everything causally linked to it; click{' '}
        <span className="font-mono">#seq</span> for the raw payload.
      </Text>
      {timeline.unavailableEnvs.length > 0 && (
        <Callout.Root color="amber" size="1">
          <Callout.Icon>
            <AlertTriangle size={13} />
          </Callout.Icon>
          <Callout.Text>
            Could not read live trigger state for{' '}
            <span className="font-mono">
              {timeline.unavailableEnvs.join(', ')}
            </span>{' '}
            — events from
            {timeline.unavailableEnvs.length > 1
              ? ' those envs'
              : ' that env'}{' '}
            are missing below, not absent.
          </Callout.Text>
        </Callout.Root>
      )}
      <ClockHeader clocks={clocks} live={live} />
      <PendingMarks pending={timeline.pending} clocks={clocks} live={live} />
      <TimelineRows
        rows={timeline.rows}
        highlight={highlight}
        selected={selected}
        onSelect={setSelected}
        authored={authored}
        showVirtual={showVirtual}
        showEnv={showEnv}
        showConversation={showConversation}
      />
    </div>
  );
}
