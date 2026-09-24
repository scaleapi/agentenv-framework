import React from 'react';
import {
  AlertTriangle,
  Check,
  Clock,
  MinusCircle,
  User,
  X,
} from 'lucide-react';
import { Badge, Callout, Card, Flex, Table, Text } from '@radix-ui/themes';

/** One directive applied by the `apply_server_config` step, from context.metadata.server_config_changes[]:
 *  the extension invoked against a backing MCP server at setup and the server's reply. */
export type ServerConfigChange = {
  step_id?: string;
  env_id?: string;
  service: string;
  uri: string;
  args: Record<string, unknown>;
  result?: {
    ok?: boolean;
    wait_range_seconds?: number[];
    [k: string]: unknown;
  };
};

/** A failed apply_server_config step (context.metadata.failed_steps[]). The step raises on the first bad
 *  directive, so only the error string (with the directive uri + service) is captured. */
export type ServerConfigFailure = {
  error: string;
  error_type?: string;
  step_id?: string;
};

/** A directive skipped because the server didn't advertise the extension (ran with `tolerate_unadvertised`),
 *  from context.metadata.server_config_skipped[]. Not a failure — a no-op for that server. */
export type ServerConfigSkip = {
  step_id?: string;
  env_id?: string;
  service: string;
  uri: string;
  reason?: string;
};

/** Best-effort structured view of a failure parsed from its error string. */
function parseFailure(f: ServerConfigFailure): {
  uri?: string;
  service?: string;
  reason: string;
} {
  const uri = f.error.match(/urn:agentenv:[^'"\s)]+/)?.[0];
  const service = f.error.match(/service ['"]([^'"]+)['"]/)?.[1];
  const idx = f.error.indexOf('failed:');
  const reason =
    idx >= 0 ? f.error.slice(idx + 'failed:'.length).trim() : f.error;
  return { uri, service, reason };
}

type BadgeColor = 'amber' | 'blue' | 'gray' | 'indigo' | 'violet';

const BEHAVIOR: Record<
  string,
  { label: string; color: BadgeColor; icon: React.ReactNode }
> = {
  'urn:agentenv:set-errors/v1': {
    label: 'Forced errors',
    color: 'amber',
    icon: <AlertTriangle size={13} aria-hidden />,
  },
  'urn:agentenv:set-async-wait/v1': {
    label: 'Async wait',
    color: 'blue',
    icon: <Clock size={13} aria-hidden />,
  },
  'urn:agentenv:set-acting-user/v1': {
    label: 'Acting persona',
    color: 'violet',
    icon: <User size={13} aria-hidden />,
  },
};

function ordinal(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
}

/** Human-readable summary of a directive's args (e.g. "every 3rd call · rate_limit"). */
function describeSettings(c: ServerConfigChange): string {
  const a = c.args ?? {};
  if (c.uri.includes('set-errors')) {
    const parts: string[] = [];
    if (a.every_nth != null)
      parts.push(`every ${ordinal(Number(a.every_nth))} call`);
    if (a.error_rate != null)
      parts.push(`rate ${Math.round(Number(a.error_rate) * 100)}%`);
    if (a.transient) parts.push('transient');
    const label = parts.join(' · ') || 'error';
    const kind = a.error_type as string | undefined;
    return kind ? `${label} · ${kind}` : label;
  }
  if (c.uri.includes('set-async-wait')) {
    const wr = c.result?.wait_range_seconds ?? [
      a.min_seconds as number,
      a.max_seconds as number,
    ];
    const lo = Number(wr?.[0]);
    const hi = Number(wr?.[1] ?? wr?.[0]);
    if (Number.isNaN(lo)) return 'async';
    return lo === hi
      ? `wait ${lo}s (real time)`
      : `wait ${lo}-${hi}s (real time)`;
  }
  if (c.uri.includes('set-acting-user')) {
    return a.user_email ? `as ${a.user_email}` : 'acting user';
  }
  return JSON.stringify(a);
}

/** Renders the behavior config a task armed on its backing servers at setup (context.metadata.server_config_changes). Read-only; the agent never sees these directives. */
export function ServerConfigPanel({
  changes,
  failures = [],
  skipped = [],
}: {
  changes: ServerConfigChange[];
  failures?: ServerConfigFailure[];
  skipped?: ServerConfigSkip[];
}) {
  if (!changes?.length && !failures.length && !skipped.length) {
    return (
      <Text size="2" color="gray">
        No server configuration was applied for this run.
      </Text>
    );
  }
  return (
    <Flex direction="column" gap="3">
      <Text size="2" color="gray">
        Behavior config applied at setup by the <code>apply_server_config</code>{' '}
        step. The agent never sees these directives; they shape how the backing
        servers respond during the run.
      </Text>
      {failures.length > 0 && (
        <Callout.Root color="red" size="1">
          <Callout.Icon>
            <AlertTriangle size={15} aria-hidden />
          </Callout.Icon>
          <Callout.Text>
            {failures.length} directive{failures.length > 1 ? 's' : ''} failed
            to apply. Directives listed as applied (if any) took effect before
            the failure; the step records each on success, so this reflects the
            server&apos;s actual state.
          </Callout.Text>
        </Callout.Root>
      )}
      <Card>
        <Table.Root variant="ghost" size="1">
          <Table.Header>
            <Table.Row>
              <Table.ColumnHeaderCell>Service</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Behavior</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Tool</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Settings</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Status</Table.ColumnHeaderCell>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {changes.map((c, i) => {
              const b = BEHAVIOR[c.uri] ?? {
                label: c.uri,
                color: 'gray' as BadgeColor,
                icon: null,
              };
              const ok = c.result?.ok !== false;
              return (
                <Table.Row
                  key={`${c.uri}-${String(c.args?.tool_name ?? '')}-${i}`}
                >
                  <Table.Cell>
                    <Badge color="indigo" variant="soft">
                      {c.service}
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    <Badge color={b.color} variant="soft">
                      <Flex align="center" gap="1">
                        {b.icon}
                        {b.label}
                      </Flex>
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    <Text
                      size="2"
                      style={{
                        fontFamily: 'var(--code-font-family, monospace)',
                      }}
                    >
                      {String(c.args?.tool_name ?? '')}
                    </Text>
                  </Table.Cell>
                  <Table.Cell>
                    <Text size="2">{describeSettings(c)}</Text>
                  </Table.Cell>
                  <Table.Cell>
                    {ok ? (
                      <Flex
                        align="center"
                        gap="1"
                        style={{ color: 'var(--green-11)' }}
                      >
                        <Check size={14} aria-hidden /> applied
                      </Flex>
                    ) : (
                      <Flex
                        align="center"
                        gap="1"
                        style={{ color: 'var(--red-11)' }}
                      >
                        <X size={14} aria-hidden /> failed
                      </Flex>
                    )}
                  </Table.Cell>
                </Table.Row>
              );
            })}
            {failures.map((f, i) => {
              const p = parseFailure(f);
              const b = (p.uri && BEHAVIOR[p.uri]) || {
                label: p.uri ?? 'Config',
                color: 'gray' as BadgeColor,
                icon: null,
              };
              return (
                <Table.Row
                  key={`fail-${i}`}
                  style={{ background: 'var(--red-2)' }}
                >
                  <Table.Cell>
                    <Badge color="indigo" variant="soft">
                      {p.service ?? '-'}
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    <Badge color={b.color} variant="soft">
                      <Flex align="center" gap="1">
                        {b.icon}
                        {b.label}
                      </Flex>
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    <Text size="2" color="gray">
                      -
                    </Text>
                  </Table.Cell>
                  <Table.Cell>
                    <Text size="2" color="red" title={f.error}>
                      {p.reason}
                    </Text>
                  </Table.Cell>
                  <Table.Cell>
                    <Flex
                      align="center"
                      gap="1"
                      style={{ color: 'var(--red-11)' }}
                    >
                      <X size={14} aria-hidden /> failed
                    </Flex>
                  </Table.Cell>
                </Table.Row>
              );
            })}
            {skipped.map((s, i) => {
              const b = BEHAVIOR[s.uri] ?? {
                label: s.uri,
                color: 'gray' as BadgeColor,
                icon: null,
              };
              const reason =
                s.reason === 'extension_not_advertised'
                  ? 'extension not advertised'
                  : s.reason ?? '-';
              return (
                <Table.Row key={`skip-${i}`}>
                  <Table.Cell>
                    <Badge color="indigo" variant="soft">
                      {s.service ?? '-'}
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    <Badge color={b.color} variant="soft">
                      <Flex align="center" gap="1">
                        {b.icon}
                        {b.label}
                      </Flex>
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    <Text size="2" color="gray">
                      -
                    </Text>
                  </Table.Cell>
                  <Table.Cell>
                    <Text size="2" color="gray">
                      {reason}
                    </Text>
                  </Table.Cell>
                  <Table.Cell>
                    <Flex
                      align="center"
                      gap="1"
                      style={{ color: 'var(--gray-11)' }}
                    >
                      <MinusCircle size={14} aria-hidden /> skipped
                    </Flex>
                  </Table.Cell>
                </Table.Row>
              );
            })}
          </Table.Body>
        </Table.Root>
      </Card>
    </Flex>
  );
}
