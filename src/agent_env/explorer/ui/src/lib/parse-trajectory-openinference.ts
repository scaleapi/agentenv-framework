/**
 * OpenInference → OTEL GenAI semconv adapter.
 *
 * The openai_agents_sdk A2A agent emits OTEL spans via OpenInference's
 * SDK instrumentor. OpenInference uses its own attribute scheme:
 *
 *   openinference.span.kind = AGENT | CHAIN | TOOL | LLM
 *   llm.model_name, llm.input_messages.<i>.message.{role,content}
 *   llm.output_messages.<i>.message.{content, tool_calls.<j>.tool_call.*}
 *   llm.token_count.{prompt,completion}
 *   tool.name, input.value, output.value
 *
 * `parse-trajectory.ts` keys off the OTEL GenAI semantic conventions
 * (`gen_ai.*`) that claude_code and codex emit. Rather than fork the
 * parser, this module normalizes each OpenInference-shaped span by
 * synthesizing equivalent `gen_ai.*` attributes; both schemes coexist and
 * downstream classification + extraction logic stays unchanged.
 */
import type { OtelSpan } from './parse-trajectory';

const OPENINFERENCE_KIND_TO_GEN_AI_OP: Record<string, string> = {
  AGENT: 'invoke_agent',
  CHAIN: 'chain',
  TOOL: 'execute_tool',
  LLM: 'chat',
};

/**
 * Mutates `span.attributes` in place: if the span carries an
 * `openinference.span.kind` attribute, fills in the equivalent `gen_ai.*`
 * attributes that the main parser expects. Existing `gen_ai.*` values are
 * never overwritten — a span that already carries OTEL semconv (e.g. from
 * claude_code or codex) is left untouched.
 *
 * Returns the same span for chaining convenience.
 */
export function normalizeOpenInferenceSpan(span: OtelSpan): OtelSpan {
  const attrs = span.attributes;
  // Defensive: an unknown trajectory format (e.g. Claude Code CLI's
  // stream-json events, which carry no `attributes` field) would otherwise
  // crash here with "Cannot read properties of undefined (reading ...)".
  // The main entry point dispatches such formats to dedicated parsers
  // before this runs, but this guard prevents a hard crash if a new
  // format ever slips through detection.
  if (!attrs) return span;
  const kind = attrs['openinference.span.kind'];
  if (!kind) return span;

  if (
    !attrs['gen_ai.operation.name'] &&
    OPENINFERENCE_KIND_TO_GEN_AI_OP[kind]
  ) {
    attrs['gen_ai.operation.name'] = OPENINFERENCE_KIND_TO_GEN_AI_OP[kind];
  }
  if (!attrs['gen_ai.request.model'] && attrs['llm.model_name']) {
    attrs['gen_ai.request.model'] = attrs['llm.model_name'];
  }
  if (!attrs['gen_ai.tool.name'] && attrs['tool.name']) {
    attrs['gen_ai.tool.name'] = attrs['tool.name'];
  }
  if (!attrs['gen_ai.usage.input_tokens'] && attrs['llm.token_count.prompt']) {
    attrs['gen_ai.usage.input_tokens'] = attrs['llm.token_count.prompt'];
  }
  if (
    !attrs['gen_ai.usage.output_tokens'] &&
    attrs['llm.token_count.completion']
  ) {
    attrs['gen_ai.usage.output_tokens'] = attrs['llm.token_count.completion'];
  }

  if (kind === 'LLM' && !attrs['gen_ai.completion']) {
    attrs['gen_ai.completion'] = JSON.stringify({
      content: extractLlmCompletionContent(attrs),
    });
  }

  if (kind === 'LLM' && !attrs['gen_ai.prompt']) {
    const messages = extractLlmInputMessages(attrs);
    if (messages.length > 0) {
      attrs['gen_ai.prompt'] = JSON.stringify({ messages });
    }
  }

  if (kind === 'TOOL' && !attrs['gen_ai.completion'] && attrs['output.value']) {
    attrs['gen_ai.completion'] = JSON.stringify({
      output: attrs['output.value'],
    });
  }

  return span;
}

/**
 * Pull `llm.output_messages.<i>.message.{content, tool_calls.<j>.tool_call.*}`
 * out of a flat OpenInference attribute bag and pack them into the
 * Anthropic-style content-block array the parser expects under
 * `gen_ai.completion.content`. Each text content becomes a `text` block;
 * each tool_call becomes a `tool_use` block.
 */
// Defensive cap. A misshaped span that fails the `!== undefined` exit
// (e.g. attribute keys with a different separator we didn't anticipate)
// would otherwise spin until OOM. Real LLM responses don't approach
// these counts.
const MAX_OUTPUT_MESSAGES = 256;
const MAX_TOOL_CALLS_PER_MESSAGE = 256;

function extractLlmCompletionContent(
  attrs: Record<string, string>,
): Array<Record<string, unknown>> {
  const content: Array<Record<string, unknown>> = [];
  let i = 0;
  while (
    i < MAX_OUTPUT_MESSAGES &&
    attrs[`llm.output_messages.${i}.message.role`] !== undefined
  ) {
    const text = attrs[`llm.output_messages.${i}.message.content`];
    if (text) content.push({ type: 'text', text });

    let j = 0;
    // Re-compute the lookup key each iteration so the while-condition actually advances past j=0.
    while (
      j < MAX_TOOL_CALLS_PER_MESSAGE &&
      attrs[
        `llm.output_messages.${i}.message.tool_calls.${j}.tool_call.function.name`
      ] !== undefined
    ) {
      const name =
        attrs[
          `llm.output_messages.${i}.message.tool_calls.${j}.tool_call.function.name`
        ] || '';
      const argsStr =
        attrs[
          `llm.output_messages.${i}.message.tool_calls.${j}.tool_call.function.arguments`
        ] || '{}';
      const id =
        attrs[
          `llm.output_messages.${i}.message.tool_calls.${j}.tool_call.id`
        ] || '';
      let input: unknown = {};
      try {
        input = JSON.parse(argsStr);
      } catch {
        /* leave as empty object */
      }
      content.push({ type: 'tool_use', id, name, input });
      j++;
    }
    i++;
  }
  return content;
}

// Mirror MAX_OUTPUT_MESSAGES — same defensive cap reasoning.
const MAX_INPUT_MESSAGES = 256;

function extractLlmInputMessages(
  attrs: Record<string, string>,
): Array<{ role: string; content: string }> {
  const messages: Array<{ role: string; content: string }> = [];
  let i = 0;
  while (
    i < MAX_INPUT_MESSAGES &&
    attrs[`llm.input_messages.${i}.message.role`] !== undefined
  ) {
    messages.push({
      role: attrs[`llm.input_messages.${i}.message.role`] || '',
      content: attrs[`llm.input_messages.${i}.message.content`] || '',
    });
    i++;
  }
  return messages;
}
