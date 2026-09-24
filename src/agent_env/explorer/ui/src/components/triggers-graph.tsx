/**
 * Authored trigger graph for the task detail page. Renders env/agent triggers from
 * `register_env_triggers` / `register_agent_triggers` as a left-to-right bipartite
 * React Flow canvas. Presentational only (parses the fetched task JSON, no extra
 * requests); returns null when no triggers are registered.
 */
import React, { useState, useMemo, useCallback } from 'react';
import { ChevronUp, ShieldCheck } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import {
  Zap,
  Radar,
  Footprints,
  Link2,
  Braces,
  Clock,
  MessageCircle,
  MessageSquare,
  KeyRound,
  Square,
  Wrench,
  HelpCircle,
} from 'lucide-react';
import { Badge } from '@radix-ui/themes';
import {
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  ReactFlowProvider,
  MarkerType,
} from '@xyflow/react';
import { SECTION_HEADER_CLASS } from './shared';
import { CopyButton, ConfigDetail, FlowCanvas, truncate } from './flow-shared';
import {
  parseTriggerGraph,
  type TriggerNodeData,
  type AnchorNodeData,
  type WhenLeaf,
} from '../lib/parse-triggers';

export const ENV_COLOR = '#0d9488';
export const AGENT_COLOR = '#c026d3';

type BadgeColor = React.ComponentProps<typeof Badge>['color'];

const WHEN_KINDS: Record<string, { color: BadgeColor; Icon: LucideIcon }> = {
  action: { color: 'amber', Icon: Zap },
  state: { color: 'teal', Icon: Radar },
  step: { color: 'blue', Icon: Footprints },
  time: { color: 'cyan', Icon: Clock },
  env_trigger: { color: 'violet', Icon: Link2 },
  all: { color: 'indigo', Icon: Braces },
  any: { color: 'indigo', Icon: Braces },
  conversational: { color: 'pink', Icon: MessageCircle },
};

const ACTION_KINDS: Record<string, { color: BadgeColor; Icon: LucideIcon }> = {
  permission: { color: 'green', Icon: KeyRound },
  nl: { color: 'amber', Icon: MessageSquare },
  say: { color: 'sky', Icon: MessageCircle },
  end: { color: 'red', Icon: Square },
  tool: { color: 'orange', Icon: Wrench },
};

const UNKNOWN_KIND = { color: 'gray' as BadgeColor, Icon: HelpCircle };

function WhenBadge({ kind, label }: WhenLeaf) {
  const { color, Icon } = WHEN_KINDS[kind] ?? UNKNOWN_KIND;
  return (
    <Badge color={color} variant="soft" size="1" title={label}>
      <Icon size={11} />
      <span className="font-mono">{truncate(label, 30)}</span>
    </Badge>
  );
}

function TriggerNode({ data }: NodeProps) {
  const { trigger, width, height, isSelected } = data as {
    trigger: TriggerNodeData;
    width: number;
    height: number;
    isSelected: boolean;
  };
  const accent = trigger.kind === 'env' ? ENV_COLOR : AGENT_COLOR;

  return (
    <div
      className="relative rounded-lg bg-[var(--background)] border overflow-hidden"
      style={{
        width,
        height,
        borderColor: isSelected ? accent : 'var(--border)',
        borderWidth: isSelected ? 2 : 1,
      }}
    >
      <div
        className="absolute left-0 top-0 bottom-0 w-1 rounded-l"
        style={{ backgroundColor: accent }}
      />
      <div className="pl-3 pr-2.5 py-1.5 h-full flex flex-col justify-center gap-1">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-semibold font-mono text-[var(--foreground)]">
            {truncate(trigger.triggerId, 22)}
          </span>
          {trigger.isSensor && (
            <Badge color="gray" variant="outline" size="1">
              sensor
            </Badge>
          )}
        </div>
        <div className="flex items-center">
          <WhenBadge kind={trigger.when.kind} label={trigger.when.label} />
        </div>
        {trigger.when.leaves.map((leaf, i) => (
          <div key={i} className="flex items-center pl-3">
            <WhenBadge kind={leaf.kind} label={leaf.label} />
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-1">
          {trigger.actions.map((a, i) => {
            const { color, Icon } = ACTION_KINDS[a.kind] ?? UNKNOWN_KIND;
            return (
              <Badge key={i} color={color} variant="surface" size="1">
                <Icon size={11} />
                {truncate(a.label, 18)}
                {a.hasVerify && <ShieldCheck size={11} />}
              </Badge>
            );
          })}
          {trigger.actions.length === 0 && (
            <span className="text-[9px] text-[var(--muted-foreground)] italic">
              {trigger.isSensor
                ? 'no actions — state only'
                : trigger.notify
                ? 'no actions — notify only'
                : 'no actions'}
            </span>
          )}
        </div>
      </div>
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

function AnchorNode({ data }: NodeProps) {
  const { anchor, width, height, isSelected } = data as {
    anchor: AnchorNodeData;
    width: number;
    height: number;
    isSelected: boolean;
  };
  const accent = anchor.kind === 'env' ? ENV_COLOR : AGENT_COLOR;
  return (
    <div
      className="flex items-center justify-center gap-1 rounded-full bg-[var(--background)] border text-[10px] font-semibold font-mono text-[var(--muted-foreground)]"
      style={{
        width,
        height,
        borderColor: accent,
        borderWidth: isSelected ? 2 : 1,
      }}
    >
      {truncate(anchor.label, 28)}
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

const nodeTypes = { triggerNode: TriggerNode, anchorNode: AnchorNode };

const REF_EDGE_COLOR = '#8b5cf6';

function TriggersFlow({ steps }: { steps: Array<Record<string, unknown>> }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Layout memo: parse + dagre run only when the steps change. Selection
  // is overlaid in the thin second memo below so node clicks never
  // re-run layout (same two-memo pattern as StepsPipeline).
  const graph = useMemo(() => parseTriggerGraph(steps), [steps]);

  const { layoutNodes, flowEdges } = useMemo(() => {
    if (!graph) return { layoutNodes: [] as Node[], flowEdges: [] as Edge[] };
    const nodes: Node[] = graph.nodes.map(n => ({
      id: n.id,
      type: n.type === 'anchor' ? 'anchorNode' : 'triggerNode',
      position: { x: n.x, y: n.y },
      data:
        n.type === 'anchor'
          ? {
              anchor: n.anchor,
              width: n.width,
              height: n.height,
              isSelected: false,
            }
          : {
              trigger: n.trigger,
              width: n.width,
              height: n.height,
              isSelected: false,
            },
      draggable: false,
    }));
    const edges: Edge[] = graph.edges.map(e =>
      e.kind === 'env-ref'
        ? {
            id: e.id,
            source: e.source,
            target: e.target,
            animated: true,
            style: { stroke: REF_EDGE_COLOR, strokeWidth: 1.8, opacity: 0.75 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              width: 16,
              height: 16,
              color: REF_EDGE_COLOR,
            },
          }
        : {
            id: e.id,
            source: e.source,
            target: e.target,
            style: {
              stroke: 'var(--muted-foreground)',
              strokeWidth: 1,
              opacity: 0.25,
              strokeDasharray: '4 3',
            },
          },
    );
    return { layoutNodes: nodes, flowEdges: edges };
  }, [graph]);

  const nodes = useMemo<Node[]>(() => {
    return layoutNodes.map(n => {
      const isSelected = n.id === selectedId;
      const prev = n.data as { isSelected: boolean };
      if (prev.isSelected === isSelected) return n;
      return { ...n, data: { ...prev, isSelected } };
    });
  }, [layoutNodes, selectedId]);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedId(prev => (prev === node.id ? null : node.id));
  }, []);

  const onPaneClick = useCallback(() => setSelectedId(null), []);

  if (!graph) return null;

  const selectedNode = selectedId
    ? graph.nodes.find(n => n.id === selectedId)
    : undefined;
  const selected = selectedNode?.trigger;
  const selectedAnchor = selectedNode?.anchor;

  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-2">
        <h3 className={SECTION_HEADER_CLASS}>
          Triggers ({graph.triggerCount})
        </h3>
      </div>
      <div className="rounded-lg border border-[var(--border)] overflow-hidden">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 border-b border-[var(--border)] bg-[var(--secondary)] text-[11px] text-[var(--muted-foreground)]">
          {graph.groups.map((gr, i) => (
            <span
              key={`${gr.kind}:${gr.key}:${i}`}
              className="flex items-center gap-1.5"
            >
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{
                  backgroundColor: gr.kind === 'env' ? ENV_COLOR : AGENT_COLOR,
                }}
              />
              <span className="font-mono text-[var(--foreground)]">
                {gr.key}
              </span>
              {gr.watchRoles && <span>watch: {gr.watchRoles.join(', ')}</span>}
              {gr.executorAgentName && (
                <span>
                  executor: {gr.executorAgentName}
                  {gr.executorTimeoutSeconds != null &&
                    ` (${gr.executorTimeoutSeconds}s)`}
                </span>
              )}
            </span>
          ))}
        </div>

        <FlowCanvas
          nodes={nodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          height={Math.max(220, graph.height + 40)}
        />

        {selected && (
          <>
            <div className="flex items-center justify-between px-3 py-2 border-t border-[var(--border)] bg-[var(--secondary)]">
              <div className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor:
                      selected.kind === 'env' ? ENV_COLOR : AGENT_COLOR,
                  }}
                />
                <span className="text-xs font-semibold font-mono text-[var(--foreground)]">
                  {selected.triggerId}
                </span>
                <span className="text-xs text-[var(--muted-foreground)] font-mono">
                  {selected.kind === 'env' ? 'env' : 'agent'} ·{' '}
                  {selected.groupKey}
                </span>
              </div>
              {selected.referencedBy.length > 0 && (
                <div className="flex items-center gap-1">
                  {selected.referencedBy.map(id => (
                    <Badge key={id} color="violet" variant="soft" size="1">
                      <Link2 size={11} />
                      referenced by {id}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
            <div className="relative p-3 overflow-x-auto max-h-96 overflow-y-auto border-t border-[var(--border)]">
              <div className="absolute top-2 left-2">
                <CopyButton text={JSON.stringify(selected.raw, null, 2)} />
              </div>
              <ConfigDetail obj={selected.raw} omitKeys={['id']} />
            </div>
            <div className="flex justify-center border-t border-[var(--border)]">
              <button
                onClick={() => setSelectedId(null)}
                className="w-full py-1.5 flex items-center justify-center text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors"
                title="Collapse"
              >
                <ChevronUp size={16} />
              </button>
            </div>
          </>
        )}

        {selectedAnchor && (
          <>
            <div className="flex items-center justify-between px-3 py-2 border-t border-[var(--border)] bg-[var(--secondary)]">
              <div className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor:
                      selectedAnchor.kind === 'env' ? ENV_COLOR : AGENT_COLOR,
                  }}
                />
                <span className="text-xs font-semibold font-mono text-[var(--foreground)]">
                  {selectedAnchor.label}
                </span>
                <span className="text-xs text-[var(--muted-foreground)]">
                  {selectedAnchor.kind === 'env'
                    ? 'env trigger registration'
                    : 'agent trigger registration'}
                </span>
              </div>
              <Badge
                color={selectedAnchor.kind === 'env' ? 'teal' : 'pink'}
                variant="soft"
                size="1"
              >
                {selectedAnchor.triggerCount} trigger
                {selectedAnchor.triggerCount === 1 ? '' : 's'}
              </Badge>
            </div>
            <div className="relative p-3 overflow-x-auto max-h-96 overflow-y-auto border-t border-[var(--border)]">
              <div className="absolute top-2 left-2">
                <CopyButton
                  text={JSON.stringify(selectedAnchor.raw, null, 2)}
                />
              </div>
              <ConfigDetail
                obj={selectedAnchor.raw}
                omitKeys={['type', 'version']}
              />
            </div>
            <div className="flex justify-center border-t border-[var(--border)]">
              <button
                onClick={() => setSelectedId(null)}
                className="w-full py-1.5 flex items-center justify-center text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors"
                title="Collapse"
              >
                <ChevronUp size={16} />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function TriggersGraph({
  steps,
}: {
  steps: Array<Record<string, unknown>>;
}) {
  const hasTriggerSteps = steps.some(
    s =>
      s.type === 'register_env_triggers' ||
      s.type === 'register_agent_triggers',
  );
  if (!hasTriggerSteps) return null;
  return (
    <ReactFlowProvider>
      <TriggersFlow steps={steps} />
    </ReactFlowProvider>
  );
}
