import React, { useState, useMemo, useCallback } from 'react';
import { ChevronUp } from 'lucide-react';
import {
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  ReactFlowProvider,
} from '@xyflow/react';
import dagre from '@dagrejs/dagre';
import { STEP_COLORS } from './task-steps-shared';
import {
  CopyButton,
  ConfigDetail,
  EDGE_STYLE,
  EDGE_MARKER,
  FlowCanvas,
  truncate,
} from './flow-shared';

const DEFAULT_COLOR = '#6b7280';

const COMMON_KEYS = ['id', 'type', 'version'];

const NODE_W = 200;
const NODE_H = 68;
const TERMINAL_W = 72;
const TERMINAL_H = 36;
// Dagre layout spacing: RANK_SEP = horizontal gap between depth levels,
// NODE_SEP = vertical gap between siblings sharing a rank.
const RANK_SEP = 80;
const NODE_SEP = 40;

interface Step {
  id?: string;
  type?: string;
  version?: number;
  /** Upstream dependencies (`{task_step_id}[]`). A missing field = "no deps recorded", an implicit dep on the prior entry; `[]` = root. */
  depends_on?: Array<{ task_step_id: string }>;
  [key: string]: unknown;
}

interface StepsPipelineProps {
  steps: Step[];
  selectedIndex?: number | null;
  onStepSelect?: (index: number | null) => void;
}

function TerminalNode({ data }: NodeProps) {
  const { label } = data as { label: string };
  return (
    <div
      className="flex items-center justify-center rounded-full bg-[var(--background)] border border-[var(--border)] text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]"
      style={{ width: TERMINAL_W, height: TERMINAL_H }}
    >
      {label}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-0 !h-0 !border-0 !bg-transparent"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-0 !h-0 !border-0 !bg-transparent"
      />
    </div>
  );
}

function StepNode({ data }: NodeProps) {
  const { step, color, isSelected } = data as {
    step: Step;
    color: string;
    isSelected: boolean;
  };

  return (
    <div
      className="relative rounded-lg bg-[var(--background)] border overflow-hidden"
      style={{
        width: NODE_W,
        height: NODE_H,
        borderColor: isSelected ? color : 'var(--border)',
        borderWidth: isSelected ? 2 : 1,
      }}
    >
      {/* Color accent bar */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1 rounded-l"
        style={{ backgroundColor: color }}
      />

      <div className="pl-3 pr-2.5 py-2 h-full flex flex-col justify-center">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-semibold text-[var(--foreground)]">
            {String(step.type)}
          </span>
          {step.version != null && (
            <span className="ml-auto text-[9px] text-[var(--muted-foreground)]">
              v{String(step.version)}
            </span>
          )}
        </div>
        <span className="text-[9px] font-mono text-[var(--muted-foreground)] mt-1 block">
          {truncate(String(step.id ?? ''), 28)}
        </span>
      </div>

      {/* Handles for edges */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-0 !h-0 !border-0 !bg-transparent"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-0 !h-0 !border-0 !bg-transparent"
      />
    </div>
  );
}

const nodeTypes = { stepNode: StepNode, terminalNode: TerminalNode };

function PipelineFlow({
  steps,
  selectedIndex: externalSelected,
  onStepSelect,
}: StepsPipelineProps) {
  const isControlled = onStepSelect !== undefined;
  const [internalSelected, setInternalSelected] = useState<number | null>(null);
  const selected = isControlled ? externalSelected ?? null : internalSelected;
  const setSelected = isControlled ? onStepSelect : setInternalSelected;

  // Build nodes + edges from each step's `depends_on` and lay out with dagre (one fused memo — dagre needs the
  // whole graph). 1: step.id → node-id map. 2: one edge per dep; missing `depends_on` = implicit dep on the prior
  // entry, `[]` = root. 3: no incoming edge → Start, no outgoing → End. 4: dagre left-to-right, shift to top-left.
  // Selection state stays out of the deps so clicking a node doesn't re-run layout (overlaid in a thin memo below).
  const {
    nodes: layoutNodes,
    edges,
    graphHeight,
  } = useMemo(() => {
    const rawNodes: Node[] = [
      {
        id: 'start',
        type: 'terminalNode',
        position: { x: 0, y: 0 },
        data: { label: 'Start' },
        draggable: false,
      },
      ...steps.map((step, i) => ({
        id: `step-${i}`,
        type: 'stepNode',
        position: { x: 0, y: 0 },
        data: {
          step,
          color: STEP_COLORS[String(step.type)] || DEFAULT_COLOR,
          isSelected: false,
        },
        draggable: false,
      })),
      {
        id: 'end',
        type: 'terminalNode',
        position: { x: 0, y: 0 },
        data: { label: 'End' },
        draggable: false,
      },
    ];

    const idToNodeId = new Map<string, string>();
    steps.forEach((s, i) => {
      if (s.id) idToNodeId.set(s.id, `step-${i}`);
    });

    const rawEdges: Edge[] = [];
    const hasIncoming = new Set<string>();
    const hasOutgoing = new Set<string>();

    const pushEdge = (source: string, target: string) => {
      rawEdges.push({
        id: `e-${source}-${target}`,
        source,
        target,
        animated: true,
        style: EDGE_STYLE,
        markerEnd: EDGE_MARKER,
      });
      hasIncoming.add(target);
      hasOutgoing.add(source);
    };

    steps.forEach((s, i) => {
      const target = `step-${i}`;
      if (s.depends_on === undefined) {
        // Legacy: implicit dep on the prior step (step 0 stays a root). Chain by node-id (`step-${i-1}`) so it works even when steps lack an `id`.
        if (i > 0) pushEdge(`step-${i - 1}`, target);
        return;
      }
      const deps = Array.isArray(s.depends_on) ? s.depends_on : [];
      for (const d of deps) {
        const sourceId = idToNodeId.get(d?.task_step_id);
        // Dangling dep (referenced step not in this task) — skip the
        // edge; the synthetic start-edge below will re-root the orphan.
        if (!sourceId) continue;
        pushEdge(sourceId, target);
      }
    });

    steps.forEach((_, i) => {
      const id = `step-${i}`;
      if (!hasIncoming.has(id)) {
        rawEdges.push({
          id: `e-start-${id}`,
          source: 'start',
          target: id,
          animated: true,
          style: EDGE_STYLE,
          markerEnd: EDGE_MARKER,
        });
      }
      if (!hasOutgoing.has(id)) {
        rawEdges.push({
          id: `e-${id}-end`,
          source: id,
          target: 'end',
          animated: true,
          style: EDGE_STYLE,
          markerEnd: EDGE_MARKER,
        });
      }
    });

    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'LR', nodesep: NODE_SEP, ranksep: RANK_SEP });
    g.setDefaultEdgeLabel(() => ({}));
    rawNodes.forEach(n => {
      const isTerminal = n.id === 'start' || n.id === 'end';
      g.setNode(n.id, {
        width: isTerminal ? TERMINAL_W : NODE_W,
        height: isTerminal ? TERMINAL_H : NODE_H,
      });
    });
    rawEdges.forEach(e => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const positionedNodes: Node[] = rawNodes.map(n => {
      const p = g.node(n.id);
      const isTerminal = n.id === 'start' || n.id === 'end';
      const w = isTerminal ? TERMINAL_W : NODE_W;
      const h = isTerminal ? TERMINAL_H : NODE_H;
      return { ...n, position: { x: p.x - w / 2, y: p.y - h / 2 } };
    });

    const { height } = g.graph();
    return {
      nodes: positionedNodes,
      edges: rawEdges,
      graphHeight: height ?? NODE_H,
    };
  }, [steps]);

  // Selection overlay: clone step nodes with the current `isSelected` flag — a cheap O(n) pass, no dagre, so clicking only re-runs this. Terminal nodes returned by identity.
  const nodes = useMemo<Node[]>(() => {
    return layoutNodes.map(n => {
      if (n.type !== 'stepNode') return n;
      const idx = parseInt(n.id.replace('step-', ''), 10);
      const isSelected = selected === idx;
      const prev = n.data as { isSelected: boolean };
      if (prev.isSelected === isSelected) return n;
      return { ...n, data: { ...prev, isSelected } };
    });
  }, [layoutNodes, selected]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (!node.id.startsWith('step-')) return;
      const idx = parseInt(node.id.replace('step-', ''), 10);
      if (isControlled) {
        setSelected(selected === idx ? null : idx);
      } else {
        (setSelected as React.Dispatch<React.SetStateAction<number | null>>)(
          prev => (prev === idx ? null : idx),
        );
      }
    },
    [isControlled, selected, setSelected],
  );

  const onPaneClick = useCallback(() => {
    setSelected(null);
  }, [setSelected]);

  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden">
      {/* Grow vertically to accommodate parallel branches that stack
          into multiple dagre rows. Floor at 160 px so linear / single-
          row DAGs keep today's visual density. `+32` is ~16 px top +
          16 px bottom breathing room around dagre's computed bounds. */}
      <FlowCanvas
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        height={Math.max(160, graphHeight + 32)}
      />

      {/* Detail panel (only in uncontrolled/read-only mode) */}
      {!isControlled && selected !== null && steps[selected] && (
        <>
          <div className="flex items-center justify-between px-3 py-2 border-t border-[var(--border)] bg-[var(--secondary)]">
            <div className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{
                  backgroundColor:
                    STEP_COLORS[String(steps[selected].type)] || DEFAULT_COLOR,
                }}
              />
              <span className="text-xs font-semibold text-[var(--foreground)]">
                {String(steps[selected].type)}
              </span>
              <span className="text-xs text-[var(--muted-foreground)] font-mono">
                {String(steps[selected].id ?? '')}
              </span>
            </div>
          </div>
          <div className="relative p-3 overflow-x-auto max-h-96 overflow-y-auto border-t border-[var(--border)]">
            <div className="absolute top-2 left-2">
              <CopyButton text={JSON.stringify(steps[selected], null, 2)} />
            </div>
            <ConfigDetail obj={steps[selected]} omitKeys={COMMON_KEYS} />
          </div>
          <div className="flex justify-center border-t border-[var(--border)]">
            <button
              onClick={() => setSelected(null)}
              className="w-full py-1.5 flex items-center justify-center text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors"
              title="Collapse"
            >
              <ChevronUp size={16} />
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export function StepsPipeline({
  steps,
  selectedIndex,
  onStepSelect,
}: StepsPipelineProps) {
  if (steps.length === 0) {
    return (
      <p className="text-sm text-[var(--muted-foreground)]">
        No steps in this task
      </p>
    );
  }

  return (
    <ReactFlowProvider>
      <PipelineFlow
        steps={steps}
        selectedIndex={selectedIndex}
        onStepSelect={onStepSelect}
      />
    </ReactFlowProvider>
  );
}
