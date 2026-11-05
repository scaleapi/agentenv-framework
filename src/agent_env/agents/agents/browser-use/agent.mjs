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

// A cancelled turn lets go of its browser before it goes, one it is still connecting to included.
let agent;
let cancelled = false;
process.once('SIGTERM', () => {
  cancelled = true;
  if (agent) agent.close().finally(() => process.exit(143));
});

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

// The browser's one open tab, which a browser computer shows as its screen, so the agent works where it's seen; with
// none or several, or one it can't list, Browser Use opens a tab of its own.
const TAB_LOOKUP_MS = 15_000;
async function onlyTab(cdpUrl) {
  try {
    let endpoint = cdpUrl;
    if (/^https?:/.test(endpoint)) {
      const url = new URL(endpoint);
      url.pathname = `${url.pathname.replace(/\/$/, '')}/json/version`;
      endpoint = (await (await fetch(url, { signal: AbortSignal.timeout(TAB_LOOKUP_MS) })).json()).webSocketDebuggerUrl;
    }
    const socket = new WebSocket(endpoint);
    let timer;
    const targets = await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error('Target.getTargets timed out')), TAB_LOOKUP_MS);
      socket.onopen = () => socket.send(JSON.stringify({ id: 1, method: 'Target.getTargets' }));
      socket.onmessage = ({ data }) => {
        const message = JSON.parse(data);
        if (message.id === 1) resolve(message.result?.targetInfos ?? []);
      };
      socket.onerror = socket.onclose = () => reject(new Error('the browser closed the connection'));
    }).finally(() => {
      clearTimeout(timer);
      socket.close();
    });
    const pages = targets.filter((target) => target.type === 'page');
    return pages.length === 1 ? pages[0].targetId : undefined;
  } catch {
    return undefined;
  }
}

const targetId = await onlyTab(turn.cdpUrl);
if (cancelled) process.exit(143);
agent = await BrowserUse.create({
  model: `agentenv/${turn.model}`,
  models,
  reasoning: turn.effort,
  instructions: turn.systemPrompt,
  tools,
  browser: Browser.chrome({ cdpUrl: turn.cdpUrl, ...(targetId ? { targetId } : {}) }),
  workspace: turn.workspace,
  historyFile: existsSync(turn.historyFile) ? turn.historyFile : undefined,
  telemetry: false,
});
if (cancelled) await agent.close().finally(() => process.exit(143));
try {
  const options = { maxSteps: turn.maxSteps, timeoutMs: turn.timeoutMs, onEvent: (event) => print({ type: 'event', event }) };
  const result = existsSync(turn.historyFile) ? await agent.followUp(turn.prompt, options) : await agent.run(turn.prompt, options);
  await agent.saveHistory(turn.historyFile);
  print({ type: 'result', result });
} finally {
  await agent.close();
  await Promise.allSettled(clients.map((client) => client.close()));
}
