'use client';

import React, {
  useRef,
  useState,
  useEffect,
  useCallback,
  createContext,
  useContext,
} from 'react';
import * as Accordion from '@radix-ui/react-accordion';
import { cn } from '../lib/utils';
import { objectContentUrl } from './shared';
import {
  type ParsedTrajectory,
  type SubAgentSummary,
  type ToolCallEvent,
  type TrajectoryEvent,
  type TrajectoryStep,
  parseToolName,
  formatDuration,
} from '../lib/parse-trajectory';
import {
  Copy,
  Check,
  Terminal,
  Mail,
  Users,
  Calendar,
  Clock,
  MapPin,
  MessageSquare,
  Search,
  Hash,
  Plane,
  Home,
  Car,
  Bell,
  Database,
  BarChart3,
  Wrench,
  Maximize2,
  Minimize2,
  ChevronRight,
  Phone,
  Building2,
  DollarSign,
  FileText,
  Receipt,
  type LucideIcon,
} from 'lucide-react';

// Screenshot-trimmed trajectories replace each base64 frame with a `__AEHIMG__<start>_<end>` placeholder
// (byte range), resolved to `/trajectory-image` for lazy loading. Small trajectories carry raw base64.
const SCREENSHOT_PLACEHOLDER_PREFIX = '__AEHIMG__';

// The trajectory's object URL — lets descendants build lazy-image URLs without
// prop-drilling through every nested component.
const ScreenshotBaseUriContext = createContext<string | undefined>(undefined);

// Screenshot value → <img> src: raw base64 → data: URL; `__AEHIMG__` placeholder
// → lazy `/trajectory-image` URL. Null when there's nothing to show.
function screenshotImgSrc(
  shot: string | undefined,
  baseUri: string | undefined,
): string | null {
  if (!shot) return null;
  if (shot.startsWith(SCREENSHOT_PLACEHOLDER_PREFIX)) {
    if (!baseUri) return null;
    const [start, end] = shot
      .slice(SCREENSHOT_PLACEHOLDER_PREFIX.length)
      .split('_');
    if (!start || !end) return null;
    return `${objectContentUrl(baseUri)}&start=${start}&end=${end}`;
  }
  return `data:image/png;base64,${shot}`;
}

// --- Simple markdown renderer ---

function isTableRow(line: string): boolean {
  return line.trimStart().startsWith('|') && line.trimEnd().endsWith('|');
}

function isSeparatorRow(line: string): boolean {
  return /^\|[\s:|-]+\|$/.test(line.trim());
}

function parseTableCells(line: string): string[] {
  return line
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(c => c.trim());
}

function renderTable(tableLines: string[], startKey: number): React.ReactNode {
  const headerLine = tableLines[0];
  if (!headerLine) return null;
  const headers = parseTableCells(headerLine);

  // Skip separator row (index 1), body starts at index 2
  const bodyLines = tableLines.slice(2);

  return (
    <div key={startKey} className="overflow-x-auto my-2">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            {headers.map((h, j) => (
              <th
                key={j}
                className="border border-[var(--border)] bg-[var(--muted)] px-2 py-1 text-left font-semibold"
              >
                {renderInline(h)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bodyLines.map((row, j) => (
            <tr key={j}>
              {parseTableCells(row).map((cell, k) => (
                <td key={k} className="border border-[var(--border)] px-2 py-1">
                  {renderInline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SimpleMarkdown({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let keyCounter = 0;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? '';
    const k = keyCounter++;

    // Detect table: current line is a table row, next line is a separator
    if (
      isTableRow(line) &&
      i + 1 < lines.length &&
      isSeparatorRow(lines[i + 1] ?? '')
    ) {
      const tableLines: string[] = [];
      while (i < lines.length && isTableRow(lines[i] ?? '')) {
        tableLines.push(lines[i] ?? '');
        i++;
      }
      // Need at least header + separator + one body row
      if (tableLines.length >= 3) {
        elements.push(renderTable(tableLines, k));
      }
      continue;
    }

    if (line.match(/^---+$/)) {
      elements.push(<hr key={k} className="my-3 border-[var(--border)]" />);
      i++;
      continue;
    }

    const h3Match = line.match(/^###\s+(.+)/);
    if (h3Match) {
      elements.push(
        <h4 key={k} className="text-sm font-semibold mt-3 mb-1">
          {renderInline(h3Match[1] ?? '')}
        </h4>,
      );
      i++;
      continue;
    }

    const h2Match = line.match(/^##\s+(.+)/);
    if (h2Match) {
      elements.push(
        <h3 key={k} className="text-base font-semibold mt-4 mb-1">
          {renderInline(h2Match[1] ?? '')}
        </h3>,
      );
      i++;
      continue;
    }

    const h1Match = line.match(/^#\s+(.+)/);
    if (h1Match) {
      elements.push(
        <h2 key={k} className="text-lg font-semibold mt-4 mb-2">
          {renderInline(h1Match[1] ?? '')}
        </h2>,
      );
      i++;
      continue;
    }

    if (line.match(/^[-*]\s/)) {
      elements.push(
        <div key={k} className="flex gap-2 ml-2">
          <span className="text-[var(--muted-foreground)]">•</span>
          <span>{renderInline(line.replace(/^[-*]\s/, ''))}</span>
        </div>,
      );
      i++;
      continue;
    }

    if (line.trim() === '') {
      elements.push(<div key={k} className="h-2" />);
    } else {
      elements.push(<p key={k}>{renderInline(line)}</p>);
    }
    i++;
  }

  return <div className="text-sm leading-relaxed">{elements}</div>;
}

function renderInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    if (boldMatch && boldMatch.index !== undefined) {
      if (boldMatch.index > 0) {
        parts.push(remaining.slice(0, boldMatch.index));
      }
      parts.push(<strong key={key++}>{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldMatch.index + boldMatch[0].length);
    } else {
      parts.push(remaining);
      break;
    }
  }

  return parts.length === 1 ? parts[0] : <>{parts}</>;
}

// --- Service icons ---

const SERVICE_ICONS: Record<string, LucideIcon> = {
  Code: Terminal,
  Linear: BarChart3,
  Slack: Hash,
  Email: Mail,
  Contacts: Users,
  Calendar: Calendar,
  CRM: Database,
  Rides: Car,
  Flights: Plane,
  Airbnb: Home,
  Messaging: MessageSquare,
  Reminders: Bell,
  Tool: Wrench,
};

// --- Sub-components ---

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-[var(--muted-foreground)] hover:bg-[var(--accent)] transition-colors"
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

function ServiceBadge({ service, color }: { service: string; color: string }) {
  const Icon = SERVICE_ICONS[service] ?? Wrench;
  return (
    <span
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white"
      style={{ background: color }}
    >
      <Icon className="w-3 h-3" />
      {service}
    </span>
  );
}

function InputTable({ input }: { input: Record<string, unknown> }) {
  const entries = Object.entries(input);
  if (entries.length === 0) {
    return (
      <span className="text-xs italic text-[var(--muted-foreground)]">
        No parameters
      </span>
    );
  }

  return (
    <table className="w-full text-xs">
      <tbody>
        {entries.map(([key, val]) => {
          const display =
            typeof val === 'string' ? val : JSON.stringify(val, null, 2);
          const isLong = display.length > 80;

          return (
            <tr
              key={key}
              className="border-b border-[var(--border)] last:border-b-0"
            >
              <td className="py-1 pr-2 font-semibold text-[var(--muted-foreground)] whitespace-nowrap align-top w-[1%]">
                {key}
              </td>
              <td className="py-1">
                {isLong ? (
                  <pre className="whitespace-pre-wrap break-all text-xs bg-[var(--muted)] rounded p-1.5 max-h-[200px] overflow-y-auto m-0">
                    {display}
                  </pre>
                ) : (
                  <span>{display}</span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ExpandableScrollArea({ children }: { children: React.ReactNode }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="relative">
      <div
        className={cn(
          'overflow-y-auto transition-[max-height] duration-200',
          expanded ? 'max-h-[none]' : 'max-h-[400px]',
        )}
      >
        {children}
      </div>
      <button
        onClick={() => setExpanded(prev => !prev)}
        className="flex items-center gap-1 mt-1 px-1.5 py-0.5 rounded text-[10px] text-[var(--muted-foreground)] hover:bg-[var(--accent)] transition-colors"
      >
        {expanded ? (
          <Minimize2 className="w-3 h-3" />
        ) : (
          <Maximize2 className="w-3 h-3" />
        )}
        {expanded ? 'Collapse' : 'Expand'}
      </button>
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <div className="rounded-lg overflow-hidden border border-[#333] mb-1">
      {/* Terminal title bar */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#2d2d2d]">
        <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]" />
        <span className="w-2.5 h-2.5 rounded-full bg-[#febc2e]" />
        <span className="w-2.5 h-2.5 rounded-full bg-[#28c840]" />
        <span className="flex-1 text-center text-[10px] text-[#888]">bash</span>
        <CopyButton text={code} />
      </div>
      {/* Terminal body */}
      <div className="bg-[#1a1a1a] px-3 py-2.5">
        <pre className="text-xs text-[#e0e0e0] font-mono whitespace-pre-wrap break-all leading-relaxed overflow-y-auto max-h-[300px]">
          <span className="text-[#6bc26b] select-none">$ </span>
          {code}
        </pre>
      </div>
    </div>
  );
}

// Fields that get special rendering in EmailCard
const EMAIL_ADDRESS_FIELDS = new Set([
  'recipients',
  'to',
  'cc',
  'bcc',
  'from',
  'sender',
]);
const EMAIL_BODY_FIELDS = new Set(['content', 'body', 'text', 'html']);
const EMAIL_SUBJECT_FIELDS = new Set(['subject', 'title']);
const EMAIL_HIDDEN_FIELDS = new Set(['email_id', 'id', 'message_id']);

function EmailCard({ input }: { input: Record<string, unknown> }) {
  // Find subject
  const subjectKey = Object.keys(input).find(k => EMAIL_SUBJECT_FIELDS.has(k));
  const subject = subjectKey ? String(input[subjectKey] ?? '') : '';

  // Find body
  const bodyKey = Object.keys(input).find(k => EMAIL_BODY_FIELDS.has(k));
  const body = bodyKey ? String(input[bodyKey] ?? '') : '';

  // Collect address fields
  const addressFields: { label: string; addresses: string[] }[] = [];
  for (const [key, val] of Object.entries(input)) {
    if (EMAIL_ADDRESS_FIELDS.has(key) && val) {
      const addrs = Array.isArray(val) ? val.map(String) : [String(val)];
      addressFields.push({ label: formatFieldLabel(key), addresses: addrs });
    }
  }

  // Collect remaining fields
  const renderedKeys = new Set([
    subjectKey,
    bodyKey,
    ...Object.keys(input).filter(
      k => EMAIL_ADDRESS_FIELDS.has(k) || EMAIL_HIDDEN_FIELDS.has(k),
    ),
  ]);
  const extraFields = Object.entries(input)
    .filter(([k, v]) => !renderedKeys.has(k) && v !== null && v !== undefined)
    .map(([k, v]) => ({
      label: formatFieldLabel(k),
      value: typeof v === 'object' ? JSON.stringify(v) : String(v),
    }));

  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden bg-[var(--background)]">
      {/* Subject bar */}
      {subject && (
        <div className="px-4 py-2.5 border-b border-[var(--border)]">
          <div className="text-sm font-semibold">{subject}</div>
        </div>
      )}

      {/* Address fields */}
      {addressFields.length > 0 && (
        <div className="px-4 py-2 border-b border-[var(--border)] text-xs space-y-1.5">
          {addressFields.map(({ label, addresses }) => (
            <div key={label} className="flex items-start gap-2">
              <span className="text-[var(--muted-foreground)] w-8 flex-shrink-0 pt-0.5">
                {label}
              </span>
              <div className="flex flex-wrap gap-1">
                {addresses.map((addr, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 bg-[var(--muted)] rounded-full px-2 py-0.5"
                  >
                    <Mail className="w-2.5 h-2.5 text-[var(--muted-foreground)]" />
                    {addr}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Extra fields */}
      {extraFields.length > 0 && (
        <div className="px-4 py-1.5 border-b border-[var(--border)] text-xs">
          {extraFields.map(({ label, value }) => (
            <div key={label} className="flex gap-2 py-0.5">
              <span className="text-[var(--muted-foreground)] font-medium">
                {label}
              </span>
              <span>{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Body */}
      {body && (
        <div className="px-4 py-3 text-xs leading-relaxed max-h-[400px] overflow-y-auto">
          <SimpleMarkdown text={body} />
        </div>
      )}

      {/* Sent indicator */}
      <div className="px-4 py-2 border-t border-[var(--border)] bg-[var(--muted)]">
        <div className="flex items-center gap-1.5 text-[10px] text-green-500 font-semibold">
          <Check className="w-3 h-3" />
          Sent
        </div>
      </div>
    </div>
  );
}

type EmailResultObject = Record<string, unknown>;
type ContactResultObject = Record<string, unknown>;

function isEmailArray(arr: unknown[]): arr is EmailResultObject[] {
  if (arr.length === 0) return false;
  const first = arr[0];
  if (typeof first !== 'object' || first === null) return false;
  const keys = Object.keys(first);
  const hasSubject = keys.some(k => EMAIL_SUBJECT_FIELDS.has(k));
  const hasSender = keys.some(k => k === 'sender' || k === 'from');
  return hasSubject && hasSender;
}

function tryParseEmailResults(output: string): EmailResultObject[] | null {
  try {
    const parsed = JSON.parse(output);
    // Direct array
    if (Array.isArray(parsed) && isEmailArray(parsed)) {
      return parsed;
    }
    // Wrapped: { emails: [...], ... } or { messages: [...], ... }
    if (typeof parsed === 'object' && parsed !== null) {
      for (const val of Object.values(parsed)) {
        if (Array.isArray(val) && isEmailArray(val)) {
          return val;
        }
      }
    }
  } catch {
    /* not JSON */
  }
  return null;
}

function formatTimestamp(ts: number): string {
  try {
    const date = new Date(ts * 1000);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return '';
  }
}

function senderInitials(sender: string): string {
  // "marta.kowalczyk@..." → "MK", "John Smith" → "JS"
  const name = sender.split('@')[0] ?? sender;
  const parts = name.split(/[.\s_-]/).filter(Boolean);
  return parts
    .slice(0, 2)
    .map(p => (p[0] ?? '').toUpperCase())
    .join('');
}

const EMAIL_RESULT_HIDDEN_FIELDS = new Set([
  'email_id',
  'id',
  'message_id',
  'sender',
  'from',
  ...EMAIL_SUBJECT_FIELDS,
  ...EMAIL_BODY_FIELDS,
  ...EMAIL_ADDRESS_FIELDS,
  'timestamp',
  'is_read',
]);

function EmailResultList({ emails }: { emails: EmailResultObject[] }) {
  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden mt-1 divide-y divide-[var(--border)]">
      {emails.map((email, i) => {
        const sender = String(email.sender ?? email.from ?? '');
        const subject = String(
          Object.entries(email).find(([k]) =>
            EMAIL_SUBJECT_FIELDS.has(k),
          )?.[1] ?? '',
        );
        const body = String(
          Object.entries(email).find(([k]) => EMAIL_BODY_FIELDS.has(k))?.[1] ??
            '',
        );
        const timestamp =
          typeof email.timestamp === 'number' ? email.timestamp : null;
        const isRead = email.is_read;
        const recipients = Array.isArray(email.recipients)
          ? (email.recipients as string[])
          : email.to
          ? Array.isArray(email.to)
            ? (email.to as string[])
            : [String(email.to)]
          : [];

        return (
          <div
            key={String(email.email_id ?? email.id ?? i)}
            className={cn(
              'flex gap-3 px-3 py-2.5 text-xs transition-colors hover:bg-[var(--accent)]',
              isRead === false ? 'bg-[var(--background)]' : 'opacity-70',
            )}
          >
            <div className="w-8 h-8 rounded-full bg-[#2563EB] flex items-center justify-center text-white text-[10px] font-bold flex-shrink-0 mt-0.5">
              {senderInitials(sender)}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span
                  className={cn(
                    'flex-1 truncate',
                    isRead === false ? 'font-bold' : 'font-semibold',
                  )}
                >
                  {sender}
                </span>
                {timestamp && (
                  <span className="text-[var(--muted-foreground)] flex-shrink-0 text-[10px]">
                    {formatTimestamp(timestamp)}
                  </span>
                )}
                {isRead === false && (
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                )}
              </div>
              <div
                className={cn(
                  'truncate',
                  isRead === false ? 'font-semibold' : '',
                )}
              >
                {subject}
              </div>
              {body && (
                <div className="text-[var(--muted-foreground)] mt-0.5 line-clamp-2 leading-relaxed">
                  {body.slice(0, 200)}
                </div>
              )}
              {recipients.length > 0 && (
                <div className="flex items-center gap-1 text-[var(--muted-foreground)] mt-1.5">
                  <Mail className="w-2.5 h-2.5 flex-shrink-0" />
                  <span className="truncate">{recipients.join(', ')}</span>
                </div>
              )}
              <div className="mt-1.5">
                <DynamicFieldGrid
                  data={email}
                  hiddenFields={EMAIL_RESULT_HIDDEN_FIELDS}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

type ContactObject = Record<string, unknown>;

const CONTACT_HIDDEN_FIELDS = new Set([
  'contact_id',
  'first_name',
  'last_name',
]);

function isContactArray(arr: unknown[]): arr is ContactObject[] {
  if (arr.length === 0) return false;
  const first = arr[0];
  return (
    typeof first === 'object' &&
    first !== null &&
    'first_name' in first &&
    'last_name' in first
  );
}

function tryParseContactResults(output: string): ContactObject[] | null {
  try {
    const parsed = JSON.parse(output);
    // Direct array: [{ first_name, last_name, ... }]
    if (Array.isArray(parsed) && isContactArray(parsed)) {
      return parsed;
    }
    // Wrapped: { contacts: [...], metadata: ... }
    if (typeof parsed === 'object' && parsed !== null) {
      for (const val of Object.values(parsed)) {
        if (Array.isArray(val) && isContactArray(val)) {
          return val;
        }
      }
    }
  } catch {
    /* not JSON */
  }
  return null;
}

// --- Calendar event results ---

type CalendarEventObject = Record<string, unknown>;

const CALENDAR_HIDDEN_FIELDS = new Set([
  'event_id',
  'title',
  'start_datetime',
  'end_datetime',
  'description',
  'location',
  'attendees',
  'tag',
]);

function isCalendarEventArray(arr: unknown[]): arr is CalendarEventObject[] {
  if (arr.length === 0) return false;
  const first = arr[0];
  if (typeof first !== 'object' || first === null) return false;
  const f = first as Record<string, unknown>;
  return 'title' in f && ('start_datetime' in f || 'event_id' in f);
}

function tryParseCalendarResults(output: string): CalendarEventObject[] | null {
  try {
    const parsed = JSON.parse(output);
    if (Array.isArray(parsed) && isCalendarEventArray(parsed)) return parsed;
    if (typeof parsed === 'object' && parsed !== null) {
      for (const val of Object.values(parsed)) {
        if (Array.isArray(val) && isCalendarEventArray(val)) return val;
      }
    }
  } catch {
    /* not JSON */
  }
  return null;
}

function formatEventTime(ts: unknown): string {
  if (typeof ts !== 'number') return '';
  try {
    return new Date(ts * 1000).toLocaleString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  } catch {
    return '';
  }
}

function formatEventDuration(start: unknown, end: unknown): string {
  if (typeof start !== 'number' || typeof end !== 'number') return '';
  const mins = Math.round((end - start) / 60);
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

const TAG_COLORS: Record<string, string> = {
  meeting: '#EA580C',
  deadline: '#DC2626',
  reminder: '#DB2777',
  personal: '#2563EB',
  travel: '#0284C7',
  social: '#0D9488',
};

function CalendarEventList({ events }: { events: CalendarEventObject[] }) {
  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden mt-1 divide-y divide-[var(--border)]">
      {events.map((evt, i) => {
        const title = String(evt.title ?? '');
        const startTs = evt.start_datetime;
        const endTs = evt.end_datetime;
        const location = evt.location ? String(evt.location) : '';
        const description = evt.description ? String(evt.description) : '';
        const tag = evt.tag ? String(evt.tag) : '';
        const attendees = Array.isArray(evt.attendees)
          ? (evt.attendees as string[])
          : [];
        const tagColor = TAG_COLORS[tag.toLowerCase()] ?? '#6B7280';
        const duration = formatEventDuration(startTs, endTs);

        return (
          <div
            key={String(evt.event_id ?? i)}
            className="flex gap-3 px-3 py-2.5 text-xs"
            style={{ borderLeftWidth: '3px', borderLeftColor: tagColor }}
          >
            {/* Time column */}
            <div className="flex flex-col items-center flex-shrink-0 w-14 pt-0.5">
              <Calendar className="w-3.5 h-3.5 text-[var(--muted-foreground)] mb-1" />
              {typeof startTs === 'number' && (
                <span className="text-[10px] text-[var(--muted-foreground)] text-center leading-tight">
                  {new Date(startTs * 1000).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                  })}
                </span>
              )}
              {duration && (
                <span className="text-[10px] font-medium text-[var(--muted-foreground)]">
                  {duration}
                </span>
              )}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-semibold text-sm truncate">{title}</span>
                {tag && (
                  <span
                    className="inline-flex items-center rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-white flex-shrink-0"
                    style={{ background: tagColor }}
                  >
                    {tag}
                  </span>
                )}
              </div>

              {/* Time range */}
              {typeof startTs === 'number' && (
                <div className="flex items-center gap-1.5 text-[var(--muted-foreground)] mb-1">
                  <Clock className="w-3 h-3 flex-shrink-0" />
                  <span>
                    {formatEventTime(startTs)}
                    {typeof endTs === 'number' && (
                      <> — {formatEventTime(endTs)}</>
                    )}
                  </span>
                </div>
              )}

              {/* Location */}
              {location && (
                <div className="flex items-center gap-1.5 text-[var(--muted-foreground)] mb-1">
                  <MapPin className="w-3 h-3 flex-shrink-0" />
                  <span>{location}</span>
                </div>
              )}

              {/* Attendees */}
              {attendees.length > 0 && (
                <div className="flex items-center gap-1.5 text-[var(--muted-foreground)] mb-1">
                  <Users className="w-3 h-3 flex-shrink-0" />
                  <div className="flex flex-wrap gap-1">
                    {attendees.map((a, j) => (
                      <span
                        key={j}
                        className="inline-flex items-center bg-[var(--muted)] rounded-full px-1.5 py-0.5 text-[10px]"
                      >
                        {a}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Description */}
              {description && (
                <div className="text-[var(--muted-foreground)] mt-1.5 leading-relaxed line-clamp-3">
                  {description}
                </div>
              )}

              {/* Extra fields */}
              <div className="mt-1">
                <DynamicFieldGrid
                  data={evt}
                  hiddenFields={CALENDAR_HIDDEN_FIELDS}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// --- QuickBooks entity results ---

type QBEntity = Record<string, unknown>;

interface QBParseResult {
  type: 'customers' | 'vendors' | 'accounts' | 'invoices' | 'bills';
  message: string;
  entities: QBEntity[];
  total?: number;
}

const QB_ENTITY_KEYS = [
  'customers',
  'vendors',
  'accounts',
  'invoices',
  'bills',
] as const;

function tryParseQuickBooksResults(output: string): QBParseResult | null {
  try {
    const parsed = JSON.parse(output);
    if (typeof parsed !== 'object' || parsed === null) return null;
    for (const key of QB_ENTITY_KEYS) {
      if (Array.isArray(parsed[key]) && parsed[key].length > 0) {
        return {
          type: key,
          message: typeof parsed.message === 'string' ? parsed.message : '',
          entities: parsed[key] as QBEntity[],
          total: typeof parsed.total === 'number' ? parsed.total : undefined,
        };
      }
    }
  } catch {
    /* not JSON */
  }
  return null;
}

function formatCurrency(val: unknown): string {
  if (typeof val !== 'number') return '';
  return val.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  });
}

function extractNestedString(obj: unknown, ...keys: string[]): string {
  if (typeof obj !== 'object' || obj === null) return '';
  const o = obj as Record<string, unknown>;
  for (const k of keys) {
    if (typeof o[k] === 'string') return o[k] as string;
  }
  return '';
}

function formatAddress(addr: unknown): string {
  if (typeof addr !== 'object' || addr === null) return '';
  const a = addr as Record<string, unknown>;
  const parts = [
    a.Line1,
    a.Line2,
    [a.City, a.CountrySubDivisionCode, a.PostalCode].filter(Boolean).join(', '),
  ].filter(Boolean);
  return parts.join(', ');
}

const QB_TYPE_CONFIG: Record<
  string,
  { icon: LucideIcon; color: string; label: string }
> = {
  customers: { icon: Users, color: '#2563EB', label: 'Customer' },
  vendors: { icon: Building2, color: '#7C3AED', label: 'Vendor' },
  accounts: { icon: BarChart3, color: '#0D9488', label: 'Account' },
  invoices: { icon: FileText, color: '#EA580C', label: 'Invoice' },
  bills: { icon: Receipt, color: '#DC2626', label: 'Bill' },
};

function QuickBooksResultList({ result }: { result: QBParseResult }) {
  const config = QB_TYPE_CONFIG[result.type] ?? {
    icon: Database,
    color: '#6B7280',
    label: result.type,
  };
  const Icon = config.icon;

  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden mt-1">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 bg-[var(--muted)] border-b border-[var(--border)]">
        <Icon className="w-3.5 h-3.5" style={{ color: config.color }} />
        <span className="text-xs font-semibold">
          {result.entities.length}{' '}
          {result.entities.length === 1 ? config.label : config.label + 's'}
        </span>
        {result.message && (
          <span className="text-[10px] text-[var(--muted-foreground)]">
            — {result.message}
          </span>
        )}
      </div>

      {/* Entities */}
      <div className="divide-y divide-[var(--border)]">
        {result.entities.map((entity, i) => {
          const displayName = String(
            entity.DisplayName ??
              entity.Name ??
              entity.DocNumber ??
              `#${i + 1}`,
          );
          const companyName = entity.CompanyName
            ? String(entity.CompanyName)
            : '';
          const balance =
            typeof entity.Balance === 'number'
              ? entity.Balance
              : typeof entity.TotalAmt === 'number'
              ? entity.TotalAmt
              : typeof entity.CurrentBalance === 'number'
              ? entity.CurrentBalance
              : null;
          const email = extractNestedString(entity.PrimaryEmailAddr, 'Address');
          const phone = extractNestedString(
            entity.PrimaryPhone,
            'FreeFormNumber',
          );
          const addr = formatAddress(entity.BillAddr ?? entity.ShipAddr);
          const active = entity.Active;
          const accountType =
            typeof entity.AccountType === 'string'
              ? entity.AccountType
              : typeof entity.AccountSubType === 'string'
              ? entity.AccountSubType
              : '';

          const initials = displayName
            .split(/[\s-]+/)
            .slice(0, 2)
            .map(w => (w[0] ?? '').toUpperCase())
            .join('');

          return (
            <div
              key={String(entity.Id ?? i)}
              className="flex gap-3 px-3 py-2.5 text-xs"
            >
              {/* Avatar */}
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-white text-[10px] font-bold flex-shrink-0 mt-0.5"
                style={{ background: config.color }}
              >
                {initials}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="font-semibold text-sm truncate">
                    {displayName}
                  </span>
                  {active === false && (
                    <span className="text-[9px] font-bold uppercase px-1 py-0.5 rounded bg-red-100 text-red-600">
                      Inactive
                    </span>
                  )}
                  {accountType && (
                    <span className="text-[9px] font-medium uppercase px-1 py-0.5 rounded bg-[var(--muted)] text-[var(--muted-foreground)]">
                      {accountType}
                    </span>
                  )}
                </div>

                {companyName && companyName !== displayName && (
                  <div className="flex items-center gap-1.5 text-[var(--muted-foreground)]">
                    <Building2 className="w-3 h-3 flex-shrink-0" />
                    <span>{companyName}</span>
                  </div>
                )}

                <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-1 text-[var(--muted-foreground)]">
                  {balance !== null && (
                    <span className="inline-flex items-center gap-1">
                      <DollarSign className="w-3 h-3" />
                      <span className="font-medium text-[var(--foreground)]">
                        {formatCurrency(balance)}
                      </span>
                    </span>
                  )}
                  {email && (
                    <span className="inline-flex items-center gap-1">
                      <Mail className="w-3 h-3" />
                      {email}
                    </span>
                  )}
                  {phone && (
                    <span className="inline-flex items-center gap-1">
                      <Phone className="w-3 h-3" />
                      {phone}
                    </span>
                  )}
                  {addr && (
                    <span className="inline-flex items-center gap-1">
                      <MapPin className="w-3 h-3" />
                      {addr}
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatFieldLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatFieldValue(val: unknown): string {
  if (val === null || val === undefined) return '';
  if (typeof val === 'object') return JSON.stringify(val);
  return String(val);
}

function DynamicFieldGrid({
  data,
  hiddenFields,
}: {
  data: Record<string, unknown>;
  hiddenFields: Set<string>;
}) {
  const fields = Object.entries(data)
    .filter(
      ([k, v]) =>
        !hiddenFields.has(k) && v !== null && v !== undefined && v !== '',
    )
    .map(([k, v]) => ({
      label: formatFieldLabel(k),
      value: formatFieldValue(v),
    }));

  if (fields.length === 0) return null;

  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[var(--muted-foreground)] text-xs">
      {fields.map(({ label, value }) => (
        <React.Fragment key={label}>
          <span className="font-medium">{label}</span>
          <span className="truncate">{value}</span>
        </React.Fragment>
      ))}
    </div>
  );
}

function ContactResultList({ contacts }: { contacts: ContactObject[] }) {
  return (
    <div className="space-y-1.5 mt-1">
      {contacts.map((contact, i) => {
        const firstName = String(contact.first_name ?? '');
        const lastName = String(contact.last_name ?? '');
        const name = [firstName, lastName].filter(Boolean).join(' ');
        const initials = [firstName[0], lastName[0]]
          .filter(Boolean)
          .join('')
          .toUpperCase();

        return (
          <div
            key={String(contact.contact_id ?? i)}
            className="rounded border border-[var(--border)] px-3 py-2 text-xs"
          >
            <div className="flex items-center gap-3 mb-1.5">
              <div className="w-8 h-8 rounded-full bg-[#0D9488] flex items-center justify-center text-white text-[10px] font-bold flex-shrink-0">
                {initials}
              </div>
              <div className="font-semibold text-sm">{name}</div>
            </div>
            <div className="ml-11">
              <DynamicFieldGrid
                data={contact}
                hiddenFields={CONTACT_HIDDEN_FIELDS}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function detectCodeInput(input: Record<string, unknown>): string | null {
  // Look for a single long string field that looks like code
  for (const key of ['command', 'cmd', 'script', 'code']) {
    const val = input[key];
    if (typeof val === 'string' && val.length > 0) return val;
  }
  return null;
}

type ChannelObject = Record<string, unknown>;

function isChannelArray(arr: unknown[]): arr is ChannelObject[] {
  if (arr.length === 0) return false;
  const first = arr[0];
  return (
    typeof first === 'object' &&
    first !== null &&
    'name' in first &&
    ('is_channel' in first || 'num_members' in first)
  );
}

function tryParseChannelResults(output: string): ChannelObject[] | null {
  try {
    const parsed = JSON.parse(output);
    if (Array.isArray(parsed) && isChannelArray(parsed)) return parsed;
    if (typeof parsed === 'object' && parsed !== null) {
      for (const val of Object.values(parsed)) {
        if (Array.isArray(val) && isChannelArray(val)) return val;
      }
    }
  } catch {
    /* not JSON */
  }
  return null;
}

const CHANNEL_HIDDEN_FIELDS = new Set(['id', 'name']);

function ChannelResultList({ channels }: { channels: ChannelObject[] }) {
  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden mt-1 divide-y divide-[var(--border)]">
      {channels.map((ch, i) => {
        const name = String(ch.name ?? '');

        return (
          <div key={String(ch.id ?? i)} className="px-3 py-2 text-xs">
            <div className="flex items-center gap-2 mb-1">
              <Hash className="w-4 h-4 text-[var(--muted-foreground)] flex-shrink-0" />
              <span className="font-semibold">{name}</span>
            </div>
            <div className="ml-6">
              <DynamicFieldGrid
                data={ch}
                hiddenFields={CHANNEL_HIDDEN_FIELDS}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

type MessageObject = Record<string, unknown>;

function isMessageArray(arr: unknown[]): arr is MessageObject[] {
  if (arr.length === 0) return false;
  const first = arr[0];
  if (typeof first !== 'object' || first === null) return false;
  const f = first as Record<string, unknown>;
  const hasTextAndUser = 'text' in f && 'user' in f;
  const hasTimestamp = 'ts' in f || 'timestamp' in f;
  const hasTypeMessage = f.type === 'message';
  // Require text+user AND at least one of: ts/timestamp or type=message
  return hasTextAndUser && (hasTimestamp || hasTypeMessage);
}

function tryParseMessageResults(output: string): MessageObject[] | null {
  try {
    const parsed = JSON.parse(output);
    if (Array.isArray(parsed) && isMessageArray(parsed)) return parsed;
    if (typeof parsed === 'object' && parsed !== null) {
      for (const val of Object.values(parsed)) {
        if (Array.isArray(val) && isMessageArray(val)) return val;
      }
    }
  } catch {
    /* not JSON */
  }
  return null;
}

const MESSAGE_HIDDEN_FIELDS = new Set([
  'text',
  'user',
  'ts',
  'timestamp',
  'type',
]);

function MessageResultList({ messages }: { messages: MessageObject[] }) {
  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden mt-1 divide-y divide-[var(--border)]">
      {messages.map((msg, i) => {
        const user = String(msg.user ?? '');
        const text = String(msg.text ?? '');
        const ts =
          typeof msg.ts === 'string'
            ? parseFloat(msg.ts)
            : typeof msg.ts === 'number'
            ? msg.ts
            : null;
        const initials = user
          .replace(/^(npc_|persona_)/, '')
          .slice(0, 2)
          .toUpperCase();

        return (
          <div
            key={String(msg.ts ?? i)}
            className="flex gap-2.5 px-3 py-2 text-xs"
          >
            <div className="w-7 h-7 rounded bg-[#4A154B] flex items-center justify-center text-white text-[10px] font-bold flex-shrink-0 mt-0.5">
              {initials}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-semibold">{user}</span>
                {ts && (
                  <span className="text-[var(--muted-foreground)] text-[10px]">
                    {formatTimestamp(ts)}
                  </span>
                )}
              </div>
              <div className="leading-relaxed whitespace-pre-wrap">{text}</div>
              <div className="mt-1">
                <DynamicFieldGrid
                  data={msg}
                  hiddenFields={MESSAGE_HIDDEN_FIELDS}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

type GenericRecord = Record<string, unknown>;

const GENERIC_RECORD_HIDDEN_FIELDS = new Set(['id']);

function tryParseGenericRecordList(output: string): GenericRecord[] | null {
  try {
    const parsed = JSON.parse(output);
    let arr: unknown[] | null = null;
    if (Array.isArray(parsed) && parsed.length > 0) {
      arr = parsed;
    } else if (typeof parsed === 'object' && parsed !== null) {
      for (const val of Object.values(parsed)) {
        if (Array.isArray(val) && val.length > 0) {
          arr = val;
          break;
        }
      }
    }
    if (!arr || arr.length === 0) return null;
    const first = arr[0];
    if (typeof first === 'object' && first !== null && 'name' in first) {
      return arr as GenericRecord[];
    }
  } catch {
    /* not JSON */
  }
  return null;
}

function GenericRecordList({ records }: { records: GenericRecord[] }) {
  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden mt-1 divide-y divide-[var(--border)]">
      {records.map((record, i) => {
        const name = String(record.name ?? '');

        return (
          <div key={String(record.id ?? i)} className="px-3 py-2 text-xs">
            <div className="font-semibold text-sm mb-1">{name}</div>
            <DynamicFieldGrid
              data={record}
              hiddenFields={GENERIC_RECORD_HIDDEN_FIELDS}
            />
          </div>
        );
      })}
    </div>
  );
}

function detectEmailInput(input: Record<string, unknown>): boolean {
  const keys = Object.keys(input);
  const hasRecipients = keys.some(k => EMAIL_ADDRESS_FIELDS.has(k));
  const hasBody = keys.some(k => EMAIL_BODY_FIELDS.has(k));
  return hasRecipients && hasBody;
}

function stripMcpPrefix(name: string): string {
  const parts = name.split('__');
  if (parts.length >= 3 && parts[0] === 'mcp') {
    return parts.slice(2).join('__');
  }
  return name;
}

function SubAgentSummaryPanel({
  summary,
  index,
}: {
  summary: SubAgentSummary;
  index: number;
}) {
  const promptValue = `subagent-prompt-${index}`;
  return (
    <div className="mb-2 mt-1 rounded-md border border-[var(--border)] bg-[var(--muted)]/40 p-2.5 text-xs">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="font-semibold text-[var(--foreground)]">
          Sub-agent
        </span>
        {summary.taskType && (
          <span className="rounded bg-[var(--muted)] px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-[var(--muted-foreground)]">
            {summary.taskType}
          </span>
        )}
      </div>
      {summary.description && (
        <div className="text-[var(--foreground)] mb-1.5">
          {summary.description}
        </div>
      )}
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[var(--muted-foreground)] mb-1.5">
        {summary.toolUses > 0 && (
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              {summary.toolUses}
            </span>{' '}
            tool{summary.toolUses === 1 ? '' : 's'}
          </span>
        )}
        {summary.totalTokens > 0 && (
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              {summary.totalTokens.toLocaleString()}
            </span>{' '}
            tokens
          </span>
        )}
        {summary.durationMs > 0 && (
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              {formatDuration(summary.durationMs)}
            </span>
          </span>
        )}
        {summary.lastToolName && (
          <span>
            last:{' '}
            <span className="font-mono text-[var(--foreground)]">
              {summary.lastToolName}
            </span>
          </span>
        )}
      </div>
      {summary.prompt && (
        <Accordion.Root type="multiple">
          <Accordion.Item value={promptValue}>
            <Accordion.Header>
              <Accordion.Trigger className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors">
                <ChevronRight className="w-3 h-3 transition-transform [[data-state=open]>&]:rotate-90" />
                View prompt
              </Accordion.Trigger>
            </Accordion.Header>
            <Accordion.Content>
              <pre className="mt-1 text-xs whitespace-pre-wrap break-all bg-[var(--card)] rounded p-2 border border-[var(--border)]">
                {summary.prompt}
              </pre>
            </Accordion.Content>
          </Accordion.Item>
        </Accordion.Root>
      )}
      {summary.events.length > 0 && (
        <SubAgentEventsTimeline events={summary.events} parentIndex={index} />
      )}
    </div>
  );
}

/** Inline timeline of the sub-agent's own events, in a collapsible accordion so it doesn't dominate the parent view. Nested sub-agents recurse through SubAgentSummaryPanel. */
function SubAgentEventsTimeline({
  events,
  parentIndex,
}: {
  events: TrajectoryEvent[];
  parentIndex: number;
}) {
  const timelineValue = `subagent-timeline-${parentIndex}`;
  const toolCount = events.filter(e => e.type === 'tool_call').length;
  let thinkingCounter = 0;
  let toolCounter = 0;
  return (
    <Accordion.Root type="multiple" className="mt-2">
      <Accordion.Item value={timelineValue}>
        <Accordion.Header>
          <Accordion.Trigger className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors">
            <ChevronRight className="w-3 h-3 transition-transform [[data-state=open]>&]:rotate-90" />
            View sub-agent trajectory
            <span className="font-normal text-[var(--muted-foreground)]">
              {events.length} events · {toolCount} tool
              {toolCount === 1 ? '' : 's'}
            </span>
          </Accordion.Trigger>
        </Accordion.Header>
        <Accordion.Content>
          <div className="mt-2 pl-2 border-l-2 border-[var(--border)]">
            <Accordion.Root type="multiple">
              {events.map((event, i) => {
                if (event.type === 'thinking') {
                  const id = `sub-${parentIndex}-thinking-${thinkingCounter++}`;
                  return (
                    <Accordion.Item key={id} value={id} className="mb-2">
                      <Accordion.Header>
                        <Accordion.Trigger className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1.5 rounded-md bg-[var(--muted)] text-[var(--muted-foreground)] hover:bg-[var(--accent)] transition-colors">
                          <ChevronRight className="w-3 h-3 transition-transform [[data-state=open]>&]:rotate-90" />
                          View Thinking
                        </Accordion.Trigger>
                      </Accordion.Header>
                      <Accordion.Content>
                        <pre className="text-xs whitespace-pre-wrap leading-relaxed text-[var(--muted-foreground)] bg-[var(--muted)] border-l-2 border-[var(--border)] rounded-r p-3 mt-1 max-h-[400px] overflow-y-auto">
                          {event.text}
                        </pre>
                      </Accordion.Content>
                    </Accordion.Item>
                  );
                }
                if (event.type === 'text') {
                  return (
                    <div
                      key={`sub-${parentIndex}-text-${i}`}
                      className="text-xs text-[var(--foreground)] mb-2 whitespace-pre-wrap"
                    >
                      {event.text}
                    </div>
                  );
                }
                if (event.type === 'tool_call') {
                  const idx = toolCounter++;
                  return (
                    <ToolCard
                      key={event.id || `sub-${parentIndex}-tool-${idx}`}
                      event={event}
                      // Offset to avoid Accordion id collisions with the
                      // parent timeline's tool cards.
                      index={parentIndex * 1000 + idx}
                    />
                  );
                }
                return null;
              })}
            </Accordion.Root>
          </div>
        </Accordion.Content>
      </Accordion.Item>
    </Accordion.Root>
  );
}

function ToolCard({
  event,
  index,
  hideScreenshot = false,
}: {
  event: ToolCallEvent;
  index: number;
  hideScreenshot?: boolean;
}) {
  const { service, color, action } = parseToolName(event.name);
  const isError = event.result?.isError ?? false;
  const codeInput = detectCodeInput(event.input);
  const isEmail = detectEmailInput(event.input);
  const screenshotBaseUri = useContext(ScreenshotBaseUriContext);
  const eventShotSrc = screenshotImgSrc(
    event.result?.screenshot,
    screenshotBaseUri,
  );

  return (
    <div
      className={cn(
        'rounded-lg border bg-[var(--card)] p-3 mb-2',
        isError ? 'border-red-500/50' : 'border-[var(--border)]',
      )}
      style={{
        borderLeftWidth: '4px',
        borderLeftColor: isError ? '#EF4444' : color,
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <ServiceBadge service={service} color={isError ? '#EF4444' : color} />
        <span className="text-sm font-semibold flex-1">{action}</span>
        {!codeInput && (
          <CopyButton
            text={JSON.stringify(
              { tool: stripMcpPrefix(event.name), input: event.input },
              null,
              2,
            )}
          />
        )}
        {isError && (
          <span className="text-[10px] font-semibold text-red-500 uppercase">
            Error
          </span>
        )}
      </div>

      <div className="mb-1">
        {codeInput ? (
          <CodeBlock code={codeInput} />
        ) : isEmail ? (
          <EmailCard input={event.input} />
        ) : (
          <InputTable input={event.input} />
        )}
      </div>

      {event.subAgentSummary && (
        <SubAgentSummaryPanel summary={event.subAgentSummary} index={index} />
      )}

      {event.result &&
        (() => {
          const emailResults = tryParseEmailResults(event.result.output);
          const contactResults = tryParseContactResults(event.result.output);
          const calendarResults =
            !emailResults && !contactResults
              ? tryParseCalendarResults(event.result.output)
              : null;
          const qbResults =
            !emailResults && !contactResults && !calendarResults
              ? tryParseQuickBooksResults(event.result.output)
              : null;
          const channelResults = !qbResults
            ? tryParseChannelResults(event.result.output)
            : null;
          const messageResults =
            !emailResults && !qbResults
              ? tryParseMessageResults(event.result.output)
              : null;
          const genericRecords =
            !emailResults &&
            !contactResults &&
            !calendarResults &&
            !qbResults &&
            !channelResults &&
            !messageResults
              ? tryParseGenericRecordList(event.result.output)
              : null;
          const resultLabel = emailResults
            ? ` · ${emailResults.length} email${
                emailResults.length !== 1 ? 's' : ''
              }`
            : contactResults
            ? ` · ${contactResults.length} contact${
                contactResults.length !== 1 ? 's' : ''
              }`
            : calendarResults
            ? ` · ${calendarResults.length} event${
                calendarResults.length !== 1 ? 's' : ''
              }`
            : qbResults
            ? ` · ${qbResults.entities.length} ${qbResults.type.replace(
                /s$/,
                '',
              )}${qbResults.entities.length !== 1 ? 's' : ''}`
            : channelResults
            ? ` · ${channelResults.length} channel${
                channelResults.length !== 1 ? 's' : ''
              }`
            : messageResults
            ? ` · ${messageResults.length} message${
                messageResults.length !== 1 ? 's' : ''
              }`
            : genericRecords
            ? ` · ${genericRecords.length} record${
                genericRecords.length !== 1 ? 's' : ''
              }`
            : '';
          const resultValue = `tool-result-${index}`;
          return (
            <Accordion.Root type="multiple">
              <Accordion.Item value={resultValue}>
                <Accordion.Header>
                  <Accordion.Trigger className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1.5 rounded-md bg-[var(--muted)] text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors">
                    <ChevronRight className="w-3 h-3 transition-transform [[data-state=open]>&]:rotate-90" />
                    View Result
                    <span className="font-normal text-[var(--muted-foreground)]">
                      {event.result!.durationMs}ms{resultLabel}
                    </span>
                  </Accordion.Trigger>
                </Accordion.Header>
                <Accordion.Content>
                  <div className="mt-1">
                    <div className="flex justify-end mb-1">
                      <CopyButton text={event.result!.output} />
                    </div>
                    <ExpandableScrollArea>
                      {emailResults ? (
                        <EmailResultList emails={emailResults} />
                      ) : contactResults ? (
                        <ContactResultList contacts={contactResults} />
                      ) : calendarResults ? (
                        <CalendarEventList events={calendarResults} />
                      ) : qbResults ? (
                        <QuickBooksResultList result={qbResults} />
                      ) : channelResults ? (
                        <ChannelResultList channels={channelResults} />
                      ) : messageResults ? (
                        <MessageResultList messages={messageResults} />
                      ) : genericRecords ? (
                        <GenericRecordList records={genericRecords} />
                      ) : (
                        <pre className="text-xs whitespace-pre-wrap break-all bg-[var(--muted)] rounded p-2">
                          {event.result!.output}
                        </pre>
                      )}
                    </ExpandableScrollArea>
                  </div>
                </Accordion.Content>
              </Accordion.Item>
            </Accordion.Root>
          );
        })()}
      {!hideScreenshot && eventShotSrc && (
        <img
          src={eventShotSrc}
          loading="lazy"
          decoding="async"
          alt="Screenshot after action"
          className="mt-2 rounded border border-[var(--border)] max-w-full"
          style={{ maxHeight: '400px', objectFit: 'contain' }}
        />
      )}
    </div>
  );
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '…';
}

// --- Step events renderer ---

function StepEvents({
  step,
  stepIndex,
  hideScreenshots = false,
}: {
  step: TrajectoryStep;
  stepIndex: number;
  hideScreenshots?: boolean;
}) {
  let thinkingCounter = 0;
  let toolCounter = 0;

  return (
    <Accordion.Root type="multiple">
      {step.events.map((event, i) => {
        if (event.type === 'thinking') {
          const id = `step-${stepIndex}-thinking-${thinkingCounter++}`;
          return (
            <Accordion.Item key={id} value={id} className="mb-2">
              <Accordion.Header>
                <Accordion.Trigger className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1.5 rounded-md bg-[var(--muted)] text-[var(--muted-foreground)] hover:bg-[var(--accent)] transition-colors">
                  <ChevronRight className="w-3 h-3 transition-transform [[data-state=open]>&]:rotate-90" />
                  View Thinking
                </Accordion.Trigger>
              </Accordion.Header>
              <Accordion.Content>
                <pre className="text-xs whitespace-pre-wrap leading-relaxed text-[var(--muted-foreground)] bg-[var(--muted)] border-l-2 border-[var(--border)] rounded-r p-3 mt-1 max-h-[400px] overflow-y-auto">
                  {event.text}
                </pre>
              </Accordion.Content>
            </Accordion.Item>
          );
        }

        if (event.type === 'tool_call') {
          const idx = toolCounter++;
          return (
            <ToolCard
              key={event.id || `tool-${stepIndex}-${idx}`}
              event={event}
              index={stepIndex * 100 + idx}
              hideScreenshot={hideScreenshots}
            />
          );
        }

        return null;
      })}
    </Accordion.Root>
  );
}

// --- Main component ---

interface TrajectoryViewerProps {
  trajectory: ParsedTrajectory;
  timelineScrollRef?: React.MutableRefObject<HTMLDivElement | null>;
  isInProgress?: boolean;
  // Object URL of this trajectory; only used to lazy-load screenshot-trimmed frames.
  screenshotBaseUri?: string;
}

export function TrajectoryViewer({
  trajectory,
  timelineScrollRef,
  isInProgress = false,
  screenshotBaseUri,
}: TrajectoryViewerProps) {
  const {
    model,
    userPrompt,
    finalResponse,
    steps,
    events,
    totalDurationMs,
    numTurns,
    toolCallCount,
    serviceCounts,
  } = trajectory;

  const [activeStep, setActiveStep] = useState(0);

  const internalTimelineRef = useRef<HTMLDivElement | null>(null);
  const setTimelineNode = useCallback(
    (node: HTMLDivElement | null) => {
      internalTimelineRef.current = node;
      if (timelineScrollRef) {
        timelineScrollRef.current = node;
      }
    },
    [timelineScrollRef],
  );

  const hoveredRef = useRef(false);

  const userNavigatedRef = useRef(false);

  const goToStep = useCallback(
    (index: number, fromUser = true) => {
      if (index < 0 || index >= steps.length) return;
      if (fromUser) userNavigatedRef.current = true;
      setActiveStep(index);
      const el = internalTimelineRef.current;
      if (el) el.scrollTo({ top: 0, behavior: 'auto' });
    },
    [steps.length],
  );

  useEffect(() => {
    setActiveStep(prev =>
      steps.length === 0 ? 0 : Math.min(prev, steps.length - 1),
    );
  }, [steps.length]);

  useEffect(() => {
    if (!isInProgress || userNavigatedRef.current || steps.length === 0) return;
    setActiveStep(steps.length - 1);
  }, [steps.length, isInProgress]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!hoveredRef.current) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        goToStep(activeStep + 1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goToStep(activeStep - 1);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [activeStep, goToStep]);

  // The frame the agent saw before the step's action: the last screenshot from an earlier step.
  const stepScreenshot = useCallback(
    (step: TrajectoryStep) => {
      const idx = steps.indexOf(step);
      for (let s = idx - 1; s >= 0; s--) {
        const evs = steps[s]?.events ?? [];
        for (let i = evs.length - 1; i >= 0; i--) {
          const e = evs[i];
          if (e && e.type === 'tool_call' && e.result?.screenshot) {
            return e.result.screenshot;
          }
        }
      }
      return undefined;
    },
    [steps],
  );

  return (
    <ScreenshotBaseUriContext.Provider value={screenshotBaseUri}>
      <div
        className="flex flex-col gap-4"
        onMouseEnter={() => {
          hoveredRef.current = true;
        }}
        onMouseLeave={() => {
          hoveredRef.current = false;
        }}
      >
        <div className="flex flex-wrap items-center gap-3 text-xs text-[var(--muted-foreground)] py-2 border-b border-[var(--border)] flex-shrink-0">
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              Model:
            </span>{' '}
            {model}
          </span>
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              Turns:
            </span>{' '}
            {numTurns}
          </span>
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              Tool Calls:
            </span>{' '}
            {toolCallCount}
          </span>
          <span>
            <span className="font-semibold text-[var(--foreground)]">
              Duration:
            </span>{' '}
            {formatDuration(totalDurationMs)}
          </span>
          <div className="flex gap-1 flex-wrap">
            {Object.entries(serviceCounts)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([svc, count]) => {
                const matchingEvent = events.find(
                  e =>
                    e.type === 'tool_call' &&
                    parseToolName(e.name).service === svc,
                ) as ToolCallEvent | undefined;
                const color = matchingEvent
                  ? parseToolName(matchingEvent.name).color
                  : '#6B7280';
                return (
                  <span
                    key={svc}
                    className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold text-white"
                    style={{ background: color }}
                  >
                    {svc} ({count})
                  </span>
                );
              })}
          </div>
        </div>

        {userPrompt && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] flex-shrink-0">
            <div className="p-3">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                User Prompt
              </div>
              <pre className="text-sm whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                {userPrompt}
              </pre>
            </div>
          </div>
        )}

        {steps.length > 0 &&
          (() => {
            const idx = Math.min(activeStep, steps.length - 1);
            const step = steps[idx];
            if (!step) return null;
            const shot = stepScreenshot(step);
            const shotSrc = screenshotImgSrc(shot, screenshotBaseUri);
            const progressPct =
              steps.length > 1 ? (idx / (steps.length - 1)) * 100 : 100;

            const stepDetail = (
              <>
                <div className="flex items-baseline gap-2 mb-3">
                  <span className="text-xs font-bold text-[var(--muted-foreground)] flex-shrink-0">
                    Step {idx + 1}
                  </span>
                  {step.events.length > 0 && (
                    <span className="text-sm leading-relaxed">
                      {step.label}
                    </span>
                  )}
                </div>
                {step.events.length > 0 ? (
                  <StepEvents
                    step={step}
                    stepIndex={idx}
                    // No later step shows the last action's resulting frame, so it stays inline.
                    hideScreenshots={!!shotSrc && idx < steps.length - 1}
                  />
                ) : (
                  // Narration-only step: the agent emitted text between tool calls but the next action was reasoning, not a tool. Render as a substantive card, not a bare stub.
                  <div className="rounded-md border border-[var(--border)] bg-[var(--card)] p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1.5">
                      Agent message
                    </div>
                    <div className="text-sm leading-relaxed">
                      <SimpleMarkdown text={step.label} />
                    </div>
                  </div>
                )}
              </>
            );

            return (
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-3 flex-shrink-0">
                  <button
                    onClick={() => goToStep(idx - 1)}
                    disabled={idx <= 0}
                    className={cn(
                      'inline-flex items-center gap-1.5 rounded-lg border px-4 py-2 text-xs font-semibold transition-colors',
                      'border-[var(--border)] bg-[var(--card)] text-[var(--foreground)]',
                      'hover:enabled:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed',
                    )}
                  >
                    ← Prev
                  </button>
                  <div className="flex-1 text-center">
                    <div className="text-xs font-semibold text-[var(--foreground)]">
                      Step{' '}
                      <span className="text-lg font-bold text-[#3B82F6]">
                        {idx + 1}
                      </span>{' '}
                      / {steps.length}
                    </div>
                    <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-[var(--muted)]">
                      <div
                        className="h-full rounded-full bg-[#3B82F6] transition-all duration-200"
                        style={{ width: `${progressPct}%` }}
                      />
                    </div>
                  </div>
                  <button
                    onClick={() => goToStep(idx + 1)}
                    disabled={idx >= steps.length - 1}
                    className={cn(
                      'inline-flex items-center gap-1.5 rounded-lg border border-transparent px-4 py-2 text-xs font-semibold text-white transition-opacity',
                      'bg-[#3B82F6] hover:enabled:opacity-80 disabled:opacity-40 disabled:cursor-not-allowed',
                    )}
                  >
                    Next →
                  </button>
                </div>

                <div className="flex flex-wrap gap-1 flex-shrink-0">
                  {steps.map((s, i) => {
                    const isFinal = i === steps.length - 1;
                    const isActive = i === idx;
                    return (
                      <button
                        key={i}
                        onClick={() => goToStep(i)}
                        title={`Step ${i + 1}: ${truncate(s.label, 60)}`}
                        className={cn(
                          'h-3.5 w-3.5 rounded-sm border transition-transform hover:scale-125',
                          isActive
                            ? 'scale-110 border-[#3B82F6] bg-[#3B82F6]'
                            : isFinal
                            ? 'border-[#10B981] bg-[#10B981]'
                            : 'border-[var(--border)] bg-[var(--muted)]',
                        )}
                      />
                    );
                  })}
                </div>

                <div
                  ref={setTimelineNode}
                  className="flex flex-col gap-4 rounded-lg border border-[var(--border)] bg-[var(--card)] p-4"
                >
                  <div>{stepDetail}</div>
                  {shotSrc && (
                    <img
                      src={shotSrc}
                      loading="lazy"
                      decoding="async"
                      alt={`Step ${idx + 1} screenshot`}
                      className="max-h-[360px] max-w-full w-auto self-start flex-shrink-0 rounded-lg border border-[var(--border)] bg-[var(--muted)] cursor-zoom-in object-contain"
                      onClick={() => window.open(shotSrc, '_blank')}
                    />
                  )}
                </div>
              </div>
            );
          })()}

        {finalResponse && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] flex-shrink-0">
            <div
              className="p-3"
              style={{
                borderLeftWidth: '4px',
                borderLeftColor: isInProgress ? '#3B82F6' : '#10B981',
              }}
            >
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2 flex items-center gap-2">
                {isInProgress ? 'Latest Response' : 'Final Response'}
                {isInProgress && (
                  <span className="inline-flex items-center gap-1 normal-case tracking-normal text-[10px] font-normal text-blue-500">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                    in progress
                  </span>
                )}
              </div>
              <div className="max-h-48 overflow-y-auto">
                <SimpleMarkdown text={finalResponse} />
              </div>
            </div>
          </div>
        )}
      </div>
    </ScreenshotBaseUriContext.Provider>
  );
}
