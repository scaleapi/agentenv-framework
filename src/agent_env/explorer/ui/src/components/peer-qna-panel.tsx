import React, { useMemo } from 'react';
import { Badge, Box, Card, Flex, ScrollArea, Text } from '@radix-ui/themes';
import type { ParsedTrajectory, ToolCallEvent } from '../lib/parse-trajectory';

// Renders the solver's peer-agent Q&A (peer_send_message / peer_start_task / peer_get_task) from the
// trajectory. Unlike the Conversations or Human Input tabs, peer exchanges otherwise only live inline in the raw trajectory.

type PeerTurn = {
  kind: 'ask' | 'answer';
  peer: string;
  text: string;
  state?: string;
  isError: boolean;
};

// The peer-agents MCP tools, matched exactly or as a suffix behind a harness's server prefix (separator
// varies), requiring a non-alphanumeric boundary before the needle:
//   claude-code-cli  mcp__peer_agents__peer_send_message
//   opencode-cli     peeragentsmcp_peer_send_message
//   gemini-cli       mcp_peeragentsmcp_peer_send_message
//   codex-cli        <server>::peer_send_message
// The boundary keeps `list_peer_send_messages` / `xpeer_send_message` from matching.
const ASK_TOOLS = ['peer_send_message', 'peer_start_task'];
const GET_TOOL = 'peer_get_task';

function matches(name: string, needle: string): boolean {
  if (name === needle) return true;
  if (!name.endsWith(needle)) return false;
  const boundary = name[name.length - needle.length - 1];
  return boundary !== undefined && !/[a-zA-Z0-9]/.test(boundary);
}

function partsToText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts
    .filter(
      (p): p is { kind?: string; text?: string } =>
        !!p &&
        typeof p === 'object' &&
        typeof (p as { text?: unknown }).text === 'string',
    )
    .map(p => p.text!)
    .join('\n');
}

function questionOf(input: Record<string, unknown>): string {
  if (typeof input.prompt === 'string' && input.prompt.trim())
    return input.prompt;
  const fromParts = partsToText(input.parts);
  return fromParts || '(no message text)';
}

function strOr(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

function fields(o: Record<string, unknown>): {
  response?: string;
  state?: string;
  taskId?: string;
} {
  return {
    response: strOr(o.response),
    state: strOr(o.state),
    taskId: strOr(o.task_id),
  };
}

// A peer tool's result carries {task_id, context_id, state, response?}, but harnesses serialize it differently:
//   claude-code-cli  content-block array   [{ "type":"text", "text":"<json>" }]
//   opencode/gemini  the object directly    { "task_id", "state", "response" }
//   codex-cli        MCP envelope          { "content": [{ "type":"text", "text":"<json>" }] }
// Unwrap content blocks, parse the inner JSON; fall back to the joined text, then raw output.
function blocksToFields(blocks: unknown[]): {
  response?: string;
  state?: string;
  taskId?: string;
} {
  const text = blocks
    .map(p =>
      p &&
      typeof p === 'object' &&
      typeof (p as { text?: unknown }).text === 'string'
        ? (p as { text: string }).text
        : '',
    )
    .filter(Boolean)
    .join('\n');
  try {
    const inner = JSON.parse(text) as Record<string, unknown>;
    if (inner && typeof inner === 'object' && !Array.isArray(inner))
      return fields(inner);
  } catch {
    /* inner isn't JSON — treat the joined text as the answer */
  }
  return { response: text || undefined };
}

function parseResult(output: string | undefined): {
  response?: string;
  state?: string;
  taskId?: string;
} {
  if (!output) return {};
  let root: unknown;
  try {
    root = JSON.parse(output);
  } catch {
    return { response: output };
  }
  if (Array.isArray(root)) return blocksToFields(root); // claude: bare block array
  if (root && typeof root === 'object') {
    const content = (root as { content?: unknown }).content;
    if (Array.isArray(content)) return blocksToFields(content); // codex: {content:[blocks]}
    return fields(root as Record<string, unknown>); // opencode / gemini: object directly
  }
  return { response: output };
}

function extractPeerTurns(traj: ParsedTrajectory): PeerTurn[] {
  const turns: PeerTurn[] = [];
  const taskIdToPeer: Record<string, string> = {};
  // Task IDs whose answer we've already emitted inline (a synchronous send/start),
  // so a later defensive peer_get_task for the same task doesn't duplicate it.
  const answered = new Set<string>();
  const toolCalls = traj.events.filter(
    (e): e is ToolCallEvent => e.type === 'tool_call',
  );
  for (const call of toolCalls) {
    const isAsk = ASK_TOOLS.some(t => matches(call.name, t));
    const isGet = matches(call.name, GET_TOOL);
    if (!isAsk && !isGet) continue;

    const res = parseResult(call.result?.output);
    const isError = !!call.result?.isError;

    if (isAsk) {
      const peer =
        typeof call.input.name === 'string' ? call.input.name : 'peer';
      if (res.taskId) taskIdToPeer[res.taskId] = peer;
      turns.push({
        kind: 'ask',
        peer,
        text: questionOf(call.input),
        isError: false,
      });
      if (res.response) {
        turns.push({
          kind: 'answer',
          peer,
          text: res.response,
          state: res.state,
          isError,
        });
        if (res.taskId) answered.add(res.taskId);
      } else if (res.state) {
        turns.push({
          kind: 'answer',
          peer,
          text: `(no reply yet — state: ${res.state})`,
          state: res.state,
          isError,
        });
      }
    } else if (isGet && res.response) {
      const taskId =
        typeof call.input.task_id === 'string' ? call.input.task_id : '';
      // Skip if the ask already surfaced this task's answer inline.
      if (taskId && answered.has(taskId)) continue;
      if (taskId) answered.add(taskId);
      turns.push({
        kind: 'answer',
        peer: taskIdToPeer[taskId] ?? 'peer',
        text: res.response,
        state: res.state,
        isError,
      });
    }
  }
  return turns;
}

export function PeerQnAPanel({
  trajectories,
}: {
  trajectories: { label: string; trajectory?: ParsedTrajectory }[];
}) {
  const sections = useMemo(
    () =>
      trajectories
        .map(t => ({
          label: t.label,
          turns: t.trajectory ? extractPeerTurns(t.trajectory) : [],
        }))
        .filter(s => s.turns.length > 0),
    [trajectories],
  );

  const anyLoaded = trajectories.some(t => !!t.trajectory);

  if (!anyLoaded) {
    return (
      <Box p="4">
        <Text size="2" color="gray">
          Open the Trajectory Viewer tab to load the trajectory, then return
          here to see the peer Q&amp;A.
        </Text>
      </Box>
    );
  }
  if (sections.length === 0) {
    return (
      <Box p="4">
        <Text size="2" color="gray">
          No peer messages in this run. When the solver calls a peer (e.g.
          `peer_send_message`), the question and the peer&apos;s answer will
          appear here.
        </Text>
      </Box>
    );
  }

  return (
    <ScrollArea type="auto" scrollbars="vertical" style={{ maxHeight: 640 }}>
      <Flex direction="column" gap="4" p="3">
        {sections.map((section, si) => (
          <Flex key={si} direction="column" gap="2">
            {sections.length > 1 && (
              <Text size="1" color="gray" weight="bold">
                {section.label}
              </Text>
            )}
            {section.turns.map((turn, ti) => {
              const isAsk = turn.kind === 'ask';
              return (
                <Flex key={ti} justify={isAsk ? 'end' : 'start'}>
                  <Card
                    size="1"
                    variant="surface"
                    style={{
                      maxWidth: '85%',
                      backgroundColor: turn.isError
                        ? 'var(--red-3)'
                        : isAsk
                        ? 'var(--blue-3)'
                        : 'var(--violet-4)',
                      borderColor: turn.isError
                        ? 'var(--red-6)'
                        : isAsk
                        ? 'var(--blue-6)'
                        : 'var(--violet-7)',
                    }}
                  >
                    <Flex direction="column" gap="1">
                      <Flex gap="2" align="center">
                        <Text
                          size="1"
                          color="gray"
                          weight="bold"
                          style={{
                            letterSpacing: '0.05em',
                            textTransform: 'uppercase',
                          }}
                        >
                          {isAsk
                            ? `agent → ${turn.peer}`
                            : `${turn.peer} → agent`}
                        </Text>
                        {turn.state && !isAsk && (
                          <Badge
                            size="1"
                            variant="soft"
                            color={
                              turn.state === 'completed' ? 'green' : 'amber'
                            }
                          >
                            {turn.state}
                          </Badge>
                        )}
                      </Flex>
                      <Text size="2" style={{ whiteSpace: 'pre-wrap' }}>
                        {turn.text}
                      </Text>
                    </Flex>
                  </Card>
                </Flex>
              );
            })}
          </Flex>
        ))}
      </Flex>
    </ScrollArea>
  );
}
