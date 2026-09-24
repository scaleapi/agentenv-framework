import { useEffect, useState } from 'react';

import { apiFetch, BACKEND_URL } from './shared';

// A2A conversation transcript from the doc store's `agent_env_a2a_conversations`
// collection via `/task-instances/{id}/conversations`. Read-only.

interface ConversationMessage {
  role: string; // "user" = the source/initiator side, "agent" = the target/responder
  parts?: { kind?: string; text?: string }[];
  ts?: string;
}

interface Conversation {
  conversation_id: string;
  source_agent_name?: string;
  target_agent_name?: string;
  status?: string;
  messages?: ConversationMessage[];
}

function partsToText(parts?: ConversationMessage['parts']): string {
  return (parts ?? [])
    .map(p => p?.text ?? '')
    .join('\n')
    .trim();
}

export function ConversationsPanel({ instanceId }: { instanceId: string }) {
  const [conversations, setConversations] = useState<Conversation[] | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setConversations(null);
    setError(null);
    apiFetch(
      `${BACKEND_URL}/api/v1/task-instances/${encodeURIComponent(
        instanceId,
      )}/conversations`,
    )
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { conversations?: Conversation[] };
        if (!cancelled) setConversations(data.conversations ?? []);
      })
      .catch(e => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : 'failed to load');
      });
    return () => {
      cancelled = true;
    };
  }, [instanceId]);

  if (error)
    return (
      <p className="p-4 min-h-[400px] text-sm text-red-500">
        Failed to load conversations: {error}
      </p>
    );
  if (!conversations)
    return (
      <p className="p-4 min-h-[400px] text-sm text-[var(--muted-foreground)]">
        Loading…
      </p>
    );
  if (conversations.length === 0)
    return (
      <p className="p-4 min-h-[400px] text-sm text-[var(--muted-foreground)]">
        No conversations recorded for this run.
      </p>
    );

  return (
    <div className="p-4 space-y-6 min-h-[400px]">
      {conversations.map(conv => {
        const src = conv.source_agent_name ?? 'user';
        const tgt = conv.target_agent_name ?? 'agent';
        const messages = conv.messages ?? [];
        return (
          <div
            key={conv.conversation_id}
            className="rounded-lg border border-[var(--border)]"
          >
            <div className="px-3 py-2 border-b border-[var(--border)] text-xs text-[var(--muted-foreground)] flex items-center gap-2 flex-wrap">
              <span className="font-mono font-semibold">
                {src} → {tgt}
              </span>
              <span>· {messages.length} messages</span>
              {conv.status && <span>· {conv.status}</span>}
            </div>
            <div className="p-3 space-y-3">
              {messages.map((m, i) => {
                const isAgent = m.role === 'agent';
                const who = isAgent ? tgt : src;
                const text = partsToText(m.parts);
                return (
                  <div
                    key={i}
                    className={`flex ${
                      isAgent ? 'justify-start' : 'justify-end'
                    }`}
                  >
                    <div
                      className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                        isAgent
                          ? 'bg-[var(--secondary)]'
                          : 'bg-[var(--accent)]'
                      }`}
                    >
                      <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1 font-mono">
                        {who}
                      </div>
                      {text || (
                        <span className="italic text-[var(--muted-foreground)]">
                          (no text)
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
