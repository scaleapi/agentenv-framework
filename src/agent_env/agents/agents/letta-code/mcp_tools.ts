/** Expose the registered MCP schemas as native Letta Code tools through its public mods API. */
import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// How long one `letta mcp call` may run, as long as the bridge gives its other CLI commands (_CLI_TIMEOUT_S).
const MCP_CALL_TIMEOUT_MS = 120_000;
// The most output one `letta mcp call` may print; a larger tool result fails the call.
const MCP_CALL_MAX_OUTPUT_BYTES = 16 * 1024 * 1024;

export default function activate(letta) {
  const { agentId, tools } = JSON.parse(
    readFileSync(join(homedir(), ".letta", "agentenv-mcp-tools.json"), "utf8"),
  );
  const dispose = tools.map((tool) => letta.tools.register({
    name: tool.name,
    description: tool.description || tool.name,
    parameters: tool.inputSchema,
    requiresApproval: false,
    parallelSafe: false,
    run(ctx) {
      return new Promise((resolve) => {
        execFile("letta", ["--backend", "local", "mcp", "call", tool.name,
          "--agent", agentId, "--args", JSON.stringify(ctx.args)],
        { signal: ctx.signal, timeout: MCP_CALL_TIMEOUT_MS, maxBuffer: MCP_CALL_MAX_OUTPUT_BYTES },
        (error, stdout, stderr) => {
          let result;
          try { result = JSON.parse(stdout); } catch { /* CLI startup errors have only stderr. */ }
          resolve({
            content: result ? JSON.stringify(result) : (stderr || error?.message || stdout),
            isError: Boolean(error) || result?.isError === true,
          });
        });
      });
    },
  }));
  return () => dispose.forEach((remove) => remove());
}
