import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';


// Deterministic buildId so all replicas in a single deploy agree on
// /_next/static/{buildId}/... paths. Otherwise Next.js's default
// random-per-build hash makes replica B return 200+HTML when asked for
// replica A's asset, Cloudflare caches the HTML, and every user gets the
// MIME-type error until the cache TTL expires.
const PACKAGE_ROOT = fileURLToPath(new URL('.', import.meta.url));
const HASH_ROOT_FILES = [
  'package.json',
  'next.config.mjs',
  'tsconfig.json',
  'tailwind.config.ts',
  'postcss.config.mjs',
];

/** @param {import('node:crypto').Hash} hash @param {string} absPath */
function hashFile(hash, absPath) {
  hash.update(relative(PACKAGE_ROOT, absPath));
  hash.update('\0');
  hash.update(readFileSync(absPath));
  hash.update('\0');
}

/** @param {import('node:crypto').Hash} hash @param {string} dir */
function hashDir(hash, dir) {
  const entries = readdirSync(dir).sort();
  for (const name of entries) {
    const abs = join(dir, name);
    const st = statSync(abs);
    if (st.isDirectory()) hashDir(hash, abs);
    else if (st.isFile()) hashFile(hash, abs);
  }
}

function computeSourceBuildId() {
  const hash = createHash('sha256');
  hashDir(hash, join(PACKAGE_ROOT, 'src'));
  for (const f of HASH_ROOT_FILES) {
    try {
      hashFile(hash, join(PACKAGE_ROOT, f));
    } catch {
      /* file optional */
    }
  }
  return hash.digest('hex').slice(0, 20);
}

function resolveBuildId() {
  return (
    process.env.BUILD_ID ||
    process.env.CIRCLE_SHA1 ||
    process.env.GIT_SHA ||
    computeSourceBuildId()
  );
}

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: true,

  // Static export: the hub can then serve the built assets itself
  // (`agent-env up --ui ./ui/out`), so the whole stack is one process on one port.
  // Safe here because every route is client-rendered behind the [[...slug]] catch-all;
  // `next dev` still runs the normal server, and rewrites below still apply there.
  output: process.env.NEXT_STATIC_EXPORT === '1' ? 'export' : undefined,
  images: { unoptimized: true },
  generateBuildId: () => resolveBuildId(),

  // Proxy API calls to the backend in local dev to avoid CORS issues.
  // Proxy API calls to the hub so the browser makes same-origin requests (no CORS).
  //
  // Defaults to the address `agent-env up` binds, so a fresh clone works with no .env at
  // all — .env is gitignored, and when this was undefined the rewrites silently went
  // away and every /api/v1 call fell through to the catch-all route as 200 text/html.
  // Override AGENT_ENV_HUB_BACKEND_PROXY_URL to point at a hub elsewhere.
  async rewrites() {
    const proxyUrl =
      process.env.AGENT_ENV_HUB_BACKEND_PROXY_URL ?? 'http://127.0.0.1:8234';
    return {
      beforeFiles: [
        {
          // beforeFiles rewrites run before Next.js page/api routes.
          // Any pages/api/v1/* handlers are unreachable when active —
          // intentional; the backend owns /api/v1/*.
          source: '/api/v1/:path*',
          destination: `${proxyUrl}/api/v1/:path*`,
        },
        {
          // /health belongs to the backend too. Without this the catch-all
          // [[...slug]] route answers it with 200 text/html, which makes a
          // liveness probe report a healthy backend when none is running.
          source: '/health',
          destination: `${proxyUrl}/health`,
        },
        {
          // Same trap: the client reads /openapi.json to learn which run-start
          // operations require a LiteLLM key. Unproxied, the catch-all answers with
          // 200 HTML, res.json() throws, and the check fails *closed* — every key
          // field becomes mandatory.
          source: '/openapi.json',
          destination: `${proxyUrl}/openapi.json`,
        },
      ],
      afterFiles: [],
      fallback: [],
    };
  },

};

export default config; 