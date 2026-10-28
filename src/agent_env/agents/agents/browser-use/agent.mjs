// One turn of Browser Use Pi: reads the turn as JSON on stdin, prints each event, then the result, as a JSON line.
import { Browser, BrowserUse, Type } from '@browser_use/pi';
import { createModels, createProvider, envApiKeyAuth } from '@earendil-works/pi-ai';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { existsSync } from 'node:fs';
import { text } from 'node:stream/consumers';

const turn = JSON.parse(await text(process.stdin));
const print = (line) => process.stdout.write(JSON.stringify(line) + '\n');

// The turn's model, served by agent-env's OpenAI-compatible endpoint.
const models = createModels();
models.setProvider(createProvider({
  id: 'agentenv',
  baseUrl: process.env.LITELLM_BASE_URL,
  auth: { apiKey: envApiKeyAuth('agent-env model key', ['LITELLM_API_KEY']) },
  models: [{
    id: turn.model, name: turn.model, api: 'openai-completions', provider: 'agentenv',
    baseUrl: process.env.LITELLM_BASE_URL, headers: JSON.parse(process.env.LITELLM_EXTRA_HEADERS || '{}'), reasoning: Boolean(turn.effort), input: ['text', 'image'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 200_000, maxTokens: 32_000,
  }],
  api: openAICompletionsApi(),
}));

// Each env's MCP tools become the agent's tools, named <server>__<tool>.
const clients = [];
const tools = [];
for (const [server, { url, headers }] of Object.entries(turn.mcpServers)) {
  const client = new Client({ name: 'browser-use', version: '1.0.0' });
  await client.connect(new StreamableHTTPClientTransport(new URL(url), { requestInit: { headers } }));
  clients.push(client);
  let page = {};
  do {
    page = await client.listTools({ cursor: page.nextCursor });
    for (const tool of page.tools) addTool(client, server, tool);
  } while (page.nextCursor);
}

function addTool(client, server, tool) {
  const base = `${server}__${tool.name}`.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 60);
  let name = base;
  for (let n = 2; tools.some((t) => t.name === name); n++) name = `${base}_${n}`;
  tools.push({
    name, label: name, description: tool.description ?? '', parameters: Type.Unsafe(tool.inputSchema),
    execute: async (_id, args, signal) => {
      const result = await client.callTool({ name: tool.name, arguments: args }, undefined, { signal });
      const content = result.content.map((part) => part.type === 'image' ? part : { type: 'text', text: part.text ?? JSON.stringify(part) });
      if (result.isError) throw new Error(content.map((part) => part.text ?? '').join('\n'));
      return { content, details: result.structuredContent ?? null };
    },
  });
}

const agent = await BrowserUse.create({
  model: `agentenv/${turn.model}`,
  models,
  reasoning: turn.effort,
  instructions: turn.systemPrompt,
  tools,
  browser: turn.cdpUrl ? Browser.chrome({ cdpUrl: turn.cdpUrl }) : Browser.cloud({ apiKey: process.env.BROWSER_USE_API_KEY, timeoutMinutes: Math.ceil(turn.timeoutMs / 60_000) + 1 }),
  workspace: turn.workspace,
  historyFile: existsSync(turn.historyFile) ? turn.historyFile : undefined,
  telemetry: false,
});
// A cancelled turn stops its browser before it goes.
process.once('SIGTERM', () => agent.close().finally(() => process.exit(143)));
try {
  const options = { maxSteps: turn.maxSteps, timeoutMs: turn.timeoutMs, onEvent: (event) => print({ type: 'event', event }) };
  const result = existsSync(turn.historyFile) ? await agent.followUp(turn.prompt, options) : await agent.run(turn.prompt, options);
  await agent.saveHistory(turn.historyFile);
  print({ type: 'result', result });
} finally {
  await agent.close();
  await Promise.allSettled(clients.map((client) => client.close()));
}
