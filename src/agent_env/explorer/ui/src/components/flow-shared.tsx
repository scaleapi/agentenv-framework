/** Shared React Flow scaffolding + generic config renderer, used by both the
 *  steps pipeline and the triggers graph. */
import React, {
  useState,
  useMemo,
  useCallback,
  useEffect,
  useRef,
} from 'react';
import { Copy, Check } from 'lucide-react';
import {
  ReactFlow,
  Background,
  MarkerType,
  type Node,
  type Edge,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [text]);
  return (
    <button
      onClick={handleCopy}
      className="p-1 rounded hover:bg-[var(--accent)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
      title="Copy JSON"
    >
      {copied ? (
        <Check size={14} className="text-green-500" />
      ) : (
        <Copy size={14} />
      )}
    </button>
  );
}

/* ---- Dynamic config detail renderer ---- */

export function formatFieldLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\bid\b/gi, 'ID')
    .replace(/\bttl\b/gi, 'TTL')
    .replace(/\benv\b/gi, 'Env')
    .replace(/\bdb\b/gi, 'DB')
    .replace(/\bgb\b/gi, 'GB')
    .replace(/\burl\b/gi, 'URL');
}

function ObjectCard({
  obj,
  index,
}: {
  obj: Record<string, unknown>;
  index?: number;
}) {
  // Promote title/name/id to the heading only when it's a real string — a nested object under `title` (e.g. {title:{regex:...}}) must stay in the body, not render "[object Object]".
  const rawTitle = obj.title ?? obj.name ?? obj.id;
  const title =
    typeof rawTitle === 'string' || typeof rawTitle === 'number'
      ? String(rawTitle)
      : '';
  const entries = Object.entries(obj)
    .filter(([k]) => !(title !== '' && (k === 'title' || k === 'name')))
    .sort(([a], [b]) => a.localeCompare(b));
  return (
    <div className="rounded border border-[var(--border)] px-2.5 py-2 text-[11px]">
      {title && (
        <div className="font-medium text-[var(--foreground)] mb-1">
          {index != null && (
            <span className="text-[var(--muted-foreground)] mr-1">
              #{index + 1}
            </span>
          )}
          {String(title)}
        </div>
      )}
      {entries.map(([k, v]) => {
        if (v == null) return null;
        const isBlock =
          Array.isArray(v) ||
          (typeof v === 'object' &&
            !Object.values(v as Record<string, unknown>).every(
              x => x == null || typeof x !== 'object',
            ));
        return (
          <div
            key={k}
            className={`gap-2 py-0.5 ${isBlock ? 'flex flex-col' : 'flex'}`}
          >
            <span className="text-[var(--muted-foreground)] flex-shrink-0 capitalize">
              {formatFieldLabel(k)}
            </span>
            {isBlock ? (
              <div className="text-[var(--foreground)]">
                {formatFieldValue(k, v)}
              </div>
            ) : (
              <span
                className={`text-[var(--foreground)] break-all ${
                  isMonoField(k) ? 'font-mono' : ''
                }`}
              >
                {formatFieldValue(k, v)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function formatFieldValue(key: string, value: unknown): React.ReactNode {
  if (value == null)
    return <span className="text-[var(--muted-foreground)] italic">null</span>;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (key.includes('seconds')) return `${value}s`;
    if (key.includes('gb') || key.includes('size')) return `${value} GB`;
    return String(value);
  }
  if (typeof value === 'string') {
    if (value.length > 200)
      return <span className="whitespace-pre-wrap">{value}</span>;
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length === 0)
      return (
        <span className="text-[var(--muted-foreground)] italic">empty</span>
      );
    if (value.every(v => typeof v === 'string')) {
      return (
        <span className="flex flex-wrap gap-1">
          {value.map((v, i) => (
            <span
              key={i}
              className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium font-mono bg-[var(--secondary)] text-[var(--foreground)]"
            >
              {v}
            </span>
          ))}
        </span>
      );
    }
    if (
      value.every(v => typeof v === 'object' && v !== null && !Array.isArray(v))
    ) {
      return (
        <div className="flex flex-col gap-1.5 w-full">
          {value.map((item, i) => (
            <ObjectCard
              key={i}
              obj={item as Record<string, unknown>}
              index={i}
            />
          ))}
        </div>
      );
    }
    return (
      <pre className="text-[10px] font-mono text-[var(--muted-foreground)] overflow-x-auto max-h-48 overflow-y-auto">
        {JSON.stringify(value, null, 2)}
      </pre>
    );
  }
  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const flat = Object.values(obj).every(
      v => v == null || typeof v !== 'object',
    );
    if (flat) {
      return (
        <span className="flex flex-wrap gap-1">
          {Object.entries(obj).map(([k, v]) => (
            <span
              key={k}
              className="inline-block px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[10px]"
            >
              <span className="text-[var(--muted-foreground)]">
                {formatFieldLabel(k)}:
              </span>{' '}
              <span className="text-[var(--foreground)]">{String(v)}</span>
            </span>
          ))}
        </span>
      );
    }
    return <ObjectCard obj={obj} />;
  }
  return String(value);
}

function isMonoField(key: string): boolean {
  return (
    key.endsWith('_id') ||
    key === 'id' ||
    key === 'prompt_id' ||
    key === 'verifier_id' ||
    key.includes('model') ||
    key.includes('artifact')
  );
}

export function ConfigDetail({
  obj,
  omitKeys = [],
}: {
  obj: Record<string, unknown>;
  omitKeys?: string[];
}) {
  const entries = Object.entries(obj)
    .filter(([k]) => !omitKeys.includes(k))
    .sort(([a], [b]) => a.localeCompare(b));
  return (
    <div>
      {entries.map(([key, value]) => (
        <div key={key} className="flex gap-3 py-1.5">
          <span className="text-[11px] text-[var(--muted-foreground)] w-36 flex-shrink-0 text-right capitalize">
            {formatFieldLabel(key)}
          </span>
          <span
            className={`text-[11px] text-[var(--foreground)] break-all ${
              isMonoField(key) ? 'font-mono' : ''
            }`}
          >
            {formatFieldValue(key, value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ---- Canvas scaffolding ---- */

export const EDGE_STYLE = {
  stroke: 'var(--muted-foreground)',
  strokeWidth: 1.5,
  opacity: 0.4,
};
export const EDGE_MARKER = {
  type: MarkerType.ArrowClosed,
  width: 16,
  height: 16,
  color: 'var(--muted-foreground)',
};

export function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

export function FlowCanvas({
  nodes,
  edges,
  nodeTypes,
  onNodeClick,
  onPaneClick,
  height,
}: {
  nodes: Node[];
  edges: Edge[];
  nodeTypes: NodeTypes;
  onNodeClick: (event: React.MouseEvent, node: Node) => void;
  onPaneClick: () => void;
  height: number;
}) {
  // Modifier-gated wheel zoom: without the modifier the wheel scrolls the page; with it, ReactFlow zooms. Drag always pans. A ~1.2s overlay flashes on a bare wheel.
  const isMac = useMemo(() => {
    if (typeof navigator === 'undefined') return false;
    // `userAgentData.platform` when available, else `platform`. A wrong glyph is
    // only cosmetic.
    const ua =
      (navigator as { userAgentData?: { platform?: string } }).userAgentData
        ?.platform ??
      navigator.platform ??
      '';
    return /mac|iphone|ipad|ipod/i.test(ua);
  }, []);
  const modifierLabel = isMac ? '⌘' : 'Ctrl';
  const [showZoomHint, setShowZoomHint] = useState(false);
  const hintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (hintTimerRef.current) clearTimeout(hintTimerRef.current);
    };
  }, []);

  // Native capture-phase wheel listener: ReactFlow's d3-zoom registers its own native listener, which synthetic
  // React handlers can't stop. A capture:true native listener runs before d3, and stopImmediatePropagation blocks
  // it. We don't preventDefault — page scroll is exactly what we want without the modifier.
  const wheelGateRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = wheelGateRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        // Modifier held → user wants to zoom; let d3-zoom + ReactFlow
        // handle the wheel and keep the hint suppressed.
        if (hintTimerRef.current) {
          clearTimeout(hintTimerRef.current);
          hintTimerRef.current = null;
        }
        setShowZoomHint(false);
        return;
      }
      // No modifier → block d3-zoom from consuming the wheel and
      // flash the hint. Default action (page scroll) still fires.
      e.stopImmediatePropagation();
      setShowZoomHint(true);
      if (hintTimerRef.current) clearTimeout(hintTimerRef.current);
      hintTimerRef.current = setTimeout(() => setShowZoomHint(false), 1200);
    };
    // capture:true so we run before descendant listeners; passive:true so we can't block scroll perf.
    el.addEventListener('wheel', handler, { capture: true, passive: true });
    return () =>
      el.removeEventListener('wheel', handler, { capture: true } as
        | EventListenerOptions
        | boolean);
  }, []);

  return (
    <div className="relative" style={{ height }} ref={wheelGateRef}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.3}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        // panOnScroll off so wheel events bubble to the page unless the modifier is held. Drag still pans.
        panOnScroll={false}
        // Zoom on wheel only with the modifier — 'Meta' (Cmd/macOS) and 'Control' (Win/Linux), both accepted.
        zoomOnScroll
        zoomActivationKeyCode={['Meta', 'Control']}
      >
        <Background
          gap={16}
          size={1}
          color="var(--muted-foreground)"
          style={{ opacity: 0.15 }}
        />
      </ReactFlow>
      {showZoomHint && (
        <div
          // Pointer-events off so the hint never blocks canvas clicks/drags. aria-live="polite" so screen readers announce it non-intrusively.
          aria-live="polite"
          className="absolute inset-0 flex items-center justify-center pointer-events-none bg-black/45 text-white text-xs font-medium transition-opacity"
        >
          Hold {modifierLabel} to zoom the workflow
        </div>
      )}
    </div>
  );
}
