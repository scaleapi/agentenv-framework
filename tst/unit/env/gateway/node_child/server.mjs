// A child MCP server that is not FastMCP and not Python: plain Node, no dependencies, MCP's streamable HTTP transport
// answered with JSON bodies (stateless). Its one tool validates strictly, as a hand-written or TypeScript server does:
// an undeclared key or a wrong type is refused with invalid_arguments naming the keys, and it never re-parses a JSON
// string. A valid call answers with the arguments exactly as they arrived. Prints its port on stdout once listening.
import http from "node:http";

const TOOL = {
  name: "find_messages",
  description: "Find messages.",
  inputSchema: {
    type: "object",
    properties: { q: { type: "string" }, labels: { type: "array", items: { type: "string" } } },
    additionalProperties: false,
  },
};

function invalidFields(args) {
  const bad = [];
  for (const [key, value] of Object.entries(args)) {
    if (!(key in TOOL.inputSchema.properties)) bad.push(key);
    else if (key === "q" && typeof value !== "string") bad.push(key);
    else if (key === "labels" && !(Array.isArray(value) && value.every((v) => typeof v === "string"))) bad.push(key);
  }
  return bad.sort();
}

function answer(msg) {
  const ok = (result) => ({ jsonrpc: "2.0", id: msg.id, result });
  switch (msg.method) {
    case "initialize":
      return ok({ protocolVersion: msg.params.protocolVersion, capabilities: { tools: {} },
                  serverInfo: { name: "node-child", version: "1.0.0" } });
    case "ping":
      return ok({});
    case "tools/list":
      return ok({ tools: [TOOL] });
    case "tools/call": {
      const args = msg.params.arguments ?? {};
      const bad = msg.params.name === TOOL.name ? invalidFields(args) : ["name"];
      const text = bad.length ? JSON.stringify({ error: "invalid_arguments", invalid_fields: bad }) : JSON.stringify(args);
      return ok({ isError: bad.length > 0, content: [{ type: "text", text }] });
    }
    default:
      return { jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: `method not found: ${msg.method}` } };
  }
}

const server = http.createServer((req, res) => {
  if (req.method !== "POST") {
    res.writeHead(405).end();
    return;
  }
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", () => {
    const msg = JSON.parse(body);
    if (msg.id === undefined) {  // a notification
      res.writeHead(202).end();
      return;
    }
    res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(answer(msg)));
  });
});
server.listen(0, "127.0.0.1", () => console.log(server.address().port));
