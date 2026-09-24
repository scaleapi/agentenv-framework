'use client';

import { Flex, Text } from '@radix-ui/themes';
import { useEffect, useState } from 'react';
import { ExclamationTriangleIcon } from '@radix-ui/react-icons';
import { BACKEND_URL } from './shared';

/**
 * Tells the user when the hub is unreachable instead of showing a blank page.
 * Pages swallow fetch failures into `loading: false`, so "ran `npm run dev` but not
 * `agent-env up`" otherwise renders as an empty content area. One session-level
 * global banner (no per-page wiring); invisible when the hub answers.
 */
const POLL_MS = 5000;

export default function BackendStatusBanner() {
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const base = BACKEND_URL?.replace(/\/$/, '') ?? '';

    async function probe() {
      try {
        const res = await fetch(`${base}/health`, { cache: 'no-store' });
        // A 200 is not enough: it has to be the *API's* 200. Next's catch-all
        // route will happily answer /health with 200 text/html when the proxy
        // rewrite is missing or misconfigured, which would report a healthy
        // backend with nothing running behind it.
        const isJson = (res.headers.get('content-type') ?? '').includes('application/json');
        if (!cancelled) setReachable(res.ok && isJson);
      } catch {
        if (!cancelled) setReachable(false);
      }
    }

    probe();
    const timer = setInterval(probe, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // `null` = the first probe hasn't resolved; don't flash a warning on a healthy load.
  if (reachable !== false) return null;

  return (
    <Flex
      gap="3"
      px="4"
      py="2"
      align="center"
      className="fixed bottom-0 left-0 right-0 z-[1000] border-t border-[var(--border)] bg-[#fbbf24] text-black"
    >
      <ExclamationTriangleIcon />
      <Text size="2" weight="medium">
        Can&apos;t reach the agent-env hub.
      </Text>
      <Text size="2">
        Start it with <code className="rounded bg-black/10 px-1.5 py-0.5">agent-env up</code>
        {' '}— the UI proxies <code className="rounded bg-black/10 px-1.5 py-0.5">/api/v1</code> to it.
      </Text>
    </Flex>
  );
}
