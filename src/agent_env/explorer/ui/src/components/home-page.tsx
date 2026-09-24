import { useMemo } from 'react';
import {
  Server,
  Globe,
  FlaskConical,
  ClipboardList,
  Plus,
  Bot,
  Wand2,
  BookOpen,
} from 'lucide-react';
import Link from 'next/link';
import { type Page, type HomeCardGroup, PAGE_PATH } from './shared';

export const HOME_CARD_GROUPS: HomeCardGroup[] = [
  {
    label: 'Platform',
    cards: [
      {
        page: 'docs',
        icon: BookOpen,
        title: 'Docs',
        description: 'API and AgentEnv reference',
      },
    ],
  },
  {
    label: 'Lab',
    cards: [
      {
        page: 'environments',
        icon: Server,
        title: 'Environments Hub',
        description: 'Discover & Curate RL environments',
      },
      {
        page: 'universes',
        icon: Globe,
        title: 'Universes Hub',
        description: 'Explore the Worlds Underlying RL Envs',
      },
    ],
  },
  {
    label: 'Evaluations',
    cards: [
      {
        page: 'tasks',
        icon: ClipboardList,
        title: 'Tasks Hub',
        description: 'Browse Agent Tasks & Step Sequences',
      },
    ],
  },
  {
    label: 'Agents',
    cards: [
      {
        page: 'agents',
        icon: Bot,
        title: 'Agents Hub',
        description: 'Browse & Deploy A2A Agents',
      },
    ],
  },
];

export function HomePage({
  onNavigate,
  hostRestricted = false,
}: {
  onNavigate: (page: Page) => void;
  /** Drops cards for sections this host 403s, mirroring the sidebar filter. */
  hostRestricted?: boolean;
}) {
  const groups = useMemo(() => {
    if (!hostRestricted) return HOME_CARD_GROUPS;
    return HOME_CARD_GROUPS.map(group => ({
      ...group,
      cards: group.cards,
    })).filter(group => group.cards.length > 0);
  }, [hostRestricted]);

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-8">Home</h1>
      {groups.map(group => (
        <div key={group.label} className="mb-8">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
            {group.label}
          </h2>
          <div className="grid grid-cols-2 gap-6 max-w-4xl">
            {group.cards.map(({ page, icon: Icon, title, description }) => (
              <Link
                key={page}
                href={PAGE_PATH[page] ?? '/'}
                onClick={e => {
                  if (e.metaKey || e.ctrlKey || e.button === 1) return;
                  e.preventDefault();
                  onNavigate(page);
                }}
                className="flex flex-col gap-4 p-8 rounded-lg border border-[var(--border)] text-left hover:bg-[var(--accent)] transition-colors"
              >
                <Icon size={32} className="text-[var(--muted-foreground)]" />
                <div>
                  <div className="text-base font-semibold text-[var(--foreground)]">
                    {title}
                  </div>
                  <div className="text-sm text-[var(--muted-foreground)] mt-1">
                    {description}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
