import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import Image from 'next/image';
import {
  Home as HomeIcon,
  Server,
  Globe,
  FlaskConical,
  ClipboardList,
  BarChart3,
  Plus,
  Bot,
  Wand2,
  Wrench,
  Sprout,
  Sparkles,
  BookOpen,
  Smartphone,
  Package,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import { type Page, type NavGroup, pageToPath } from '../components/shared';
import { HomePage } from '../components/home-page';
import { EnvironmentsPage } from '../components/environments-page';
import { EnvDetailPage } from '../components/env-detail-page';
import { UniversesPage } from '../components/universes-page';
import { UniverseDetailPage } from '../components/universe-detail-page';
import { TasksHubPage } from '../components/tasks-hub-page';
import { TaskDetailPage } from '../components/task-detail-page';
import { AgentsHubPage } from '../components/agents-hub-page';
import { AgentDetailPage } from '../components/agent-detail-page';
import { TaskRunnerPage } from '../components/task-runner-page';
import { DocsPage } from '../components/docs-page';

/* ------------------------------------------------------------------ */
/*  URL <-> Page mapping                                               */
/* ------------------------------------------------------------------ */

const SECTION_MAP: Record<string, { list: Page; detail: Page }> = {
  environments: { list: 'environments', detail: 'env-detail' },
  universes: { list: 'universes', detail: 'universe-detail' },
  tasks: { list: 'tasks', detail: 'task-detail' },
  agents: { list: 'agents', detail: 'agent-detail' },
  'task-runner': { list: 'task-runner', detail: 'task-runner' },
  docs: { list: 'docs', detail: 'docs' },
};

function slugToRoute(segments: string[]): {
  page: Page;
  entityId: string | null;
} {
  if (!segments || segments.length === 0)
    return { page: 'home', entityId: null };
  const section = segments[0] as string;
  const entityId = segments[1] ? decodeURIComponent(segments[1]) : null;
  const mapping = section ? SECTION_MAP[section] : undefined;
  if (!mapping) return { page: 'home', entityId: null };
  return { page: entityId ? mapping.detail : mapping.list, entityId };
}

function currentRoute(): { page: Page; entityId: string | null } {
  if (typeof window === 'undefined') return { page: 'home', entityId: null };
  return slugToRoute(window.location.pathname.split('/').filter(Boolean));
}

/* ------------------------------------------------------------------ */
/*  Static data                                                        */
/* ------------------------------------------------------------------ */

const DETAIL_TO_PARENT: Partial<Record<Page, Page>> = {
  'env-detail': 'environments',
  'universe-detail': 'universes',
  'task-detail': 'tasks',
  'agent-detail': 'agents',
};

const NAV_GROUPS: NavGroup[] = [
  {
    label: null,
    items: [
      { page: 'home', label: 'Home', icon: HomeIcon },
      { page: 'docs', label: 'Docs', icon: BookOpen },
    ],
  },
  {
    label: 'Lab',
    items: [
      { page: 'environments', label: 'Environments Hub', icon: Server },
      { page: 'universes', label: 'Universes Hub', icon: Globe },
    ],
  },
  {
    label: 'Evaluations',
    items: [
      { page: 'tasks', label: 'Tasks Hub', icon: ClipboardList },
    ],
  },
  {
    label: 'Agents',
    items: [
      { page: 'agents', label: 'Agents Hub', icon: Bot },
    ],
  },
];

/* ------------------------------------------------------------------ */
/*  App component                                                      */
/* ------------------------------------------------------------------ */

export default function App() {
  const router = useRouter();
  const isProgrammaticNav = useRef(false);

  /* --- state (defaults match server render to avoid hydration mismatch) --- */
  const [initialized, setInitialized] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activePage, setActivePage] = useState<Page>('home');
  const [selectedEnvId, setSelectedEnvId] = useState<string | null>(null);
  const [envHistory, setEnvHistory] = useState<string[]>([]);
  const [selectedUniverseId, setSelectedUniverseId] = useState<string | null>(
    null,
  );
  const [selectedUniverseVersion, setSelectedUniverseVersion] = useState<
    number | undefined
  >(undefined);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [editTaskId, setEditTaskId] = useState<string | null>(null);
  const [cloneTaskId, setCloneTaskId] = useState<string | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedSkillId, setSelectedSkillId] = useState<string | null>(null);
  const [selectedDeliveryId, setSelectedDeliveryId] = useState<string | null>(
    null,
  );
  const [playgroundInitEnvId, setPlaygroundInitEnvId] = useState<string | null>(
    null,
  );
  const [playgroundInitUniverseId, setPlaygroundInitUniverseId] = useState<
    string | null
  >(null);
  const [playgroundInitEnvType, setPlaygroundInitEnvType] = useState<
    string | null
  >(null);
  const [playgroundInstanceId, setPlaygroundInstanceId] = useState<
    string | null
  >(null);
  const [taskRunnerTaskId, setTaskRunnerTaskId] = useState<string | null>(null);
  const [evaluatorOnlyTaskId, setEvaluatorOnlyTaskId] = useState<string | null>(
    null,
  );
  const [rerunFromStep, setRerunFromStep] = useState<number | null>(null);
  const [savedContextJson, setSavedContextJson] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [isEmbedded, setIsEmbedded] = useState(false);

  /* --- Detect iframe / host after mount to avoid SSR hydration mismatch --- */
  useEffect(() => {
    setIsEmbedded(window.parent !== window);
  }, []);

  /* --- Initialize state from URL on mount --- */
  useEffect(() => {
    const { page, entityId } = currentRoute();
    setActivePage(page);
    if (page === 'env-detail') setSelectedEnvId(entityId);
    if (page === 'universe-detail') setSelectedUniverseId(entityId);
    if (page === 'task-detail') setSelectedTaskId(entityId);
    if (page === 'agent-detail') setSelectedAgentId(entityId);
    if (page === 'task-runner') setTaskRunnerTaskId(entityId);
    setInitialized(true);
  }, []);

  /* --- Sync state from URL on browser back/forward --- */
  useEffect(() => {
    const handleRouteChange = (url: string) => {
      if (isProgrammaticNav.current) {
        isProgrammaticNav.current = false;
        return;
      }
      // Strip query (?…) and fragment (#…) so a deep-link hash isn't folded
      // into the entity id on browser back/forward.
      const path = (url.split('#')[0] ?? '').split('?')[0] ?? '';
      const segments = path.split('/').filter(Boolean);
      const { page, entityId } = slugToRoute(segments);
      setActivePage(page);
      setSelectedEnvId(page === 'env-detail' ? entityId : null);
      setSelectedUniverseId(page === 'universe-detail' ? entityId : null);
      if (page !== 'universe-detail') setSelectedUniverseVersion(undefined);
      setSelectedTaskId(page === 'task-detail' ? entityId : null);
      setCloneTaskId(null);
      setEvaluatorOnlyTaskId(null);
      setRerunFromStep(null);
      setSelectedAgentId(page === 'agent-detail' ? entityId : null);
      setTaskRunnerTaskId(page === 'task-runner' ? entityId : null);
      setEnvHistory([]);
      setSavedContextJson(null);
    };
    router.events.on('routeChangeComplete', handleRouteChange);
    return () => router.events.off('routeChangeComplete', handleRouteChange);
  }, [router.events]);

  /* --- URL push helper --- */
  const pushRoute = useCallback(
    (page: Page, entityId?: string | null) => {
      isProgrammaticNav.current = true;
      router.push(pageToPath(page, entityId), undefined, { shallow: true });
    },
    [router],
  );

  /* --- Navigation callbacks --- */

  const navigateToEnvDetail = useCallback(
    (envId: string) => {
      setSelectedEnvId(prev => {
        if (prev && activePage === 'env-detail')
          setEnvHistory(h => [...h, prev]);
        return envId;
      });
      setActivePage('env-detail');
      pushRoute('env-detail', envId);
    },
    [activePage, pushRoute],
  );

  const handleEnvBack = useCallback(() => {
    if (envHistory.length > 0) {
      const prev = envHistory[envHistory.length - 1] ?? null;
      setEnvHistory(h => h.slice(0, -1));
      setSelectedEnvId(prev);
      pushRoute('env-detail', prev);
    } else {
      setActivePage('environments');
      pushRoute('environments');
    }
  }, [envHistory, pushRoute]);

  const navigateToUniverseDetail = useCallback(
    (universeId: string, version?: number) => {
      setSelectedUniverseId(universeId);
      setSelectedUniverseVersion(version);
      setActivePage('universe-detail');
      pushRoute('universe-detail', universeId);
    },
    [pushRoute],
  );

  const navigateToTaskDetail = useCallback(
    (taskId: string) => {
      setSelectedTaskId(taskId);
      setEditTaskId(null);
      setCloneTaskId(null);
      setActivePage('task-detail');
      pushRoute('task-detail', taskId);
    },
    [pushRoute],
  );

  const navigateToAgentDetail = useCallback(
    (agentId: string) => {
      setSelectedAgentId(agentId);
      setActivePage('agent-detail');
      pushRoute('agent-detail', agentId);
    },
    [pushRoute],
  );





  const sidebarActivePage = DETAIL_TO_PARENT[activePage] ?? activePage;
  const hideSidebar = isEmbedded;

  const navGroups = NAV_GROUPS;

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      {!hideSidebar && (
        <nav
          className={`${
            sidebarOpen ? 'w-[220px]' : 'w-[56px]'
          } flex-shrink-0 border-r border-[var(--border)] flex flex-col transition-[width] duration-200`}
        >
          <div
            className={`flex items-center ${
              sidebarOpen ? 'px-5' : 'justify-center'
            } py-5`}
          >
            {sidebarOpen && (
              <Image
                src="/logos/agent-env-logo.svg"
                alt="AgentEnvExplorer"
                width={664}
                height={62}
                className="h-3.5 w-auto object-contain"
                priority
              />
            )}
          </div>
          <div
            className={`flex flex-col gap-4 ${sidebarOpen ? 'px-3' : 'px-1.5'}`}
          >
            {navGroups.map((group, gi) => (
              <div key={gi} className="flex flex-col gap-0.5">
                {group.label && sidebarOpen && (
                  <div className="px-3 pb-1 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                    {group.label}
                  </div>
                )}
                {group.items.map(({ page, label, icon: Icon }) => (
                  <Link
                    key={page}
                    href={pageToPath(page)}
                    onClick={e => {
                      if (e.metaKey || e.ctrlKey || e.button === 1) return;
                      e.preventDefault();
                      setActivePage(page);
                      setEnvHistory([]);
                      setSelectedEnvId(null);
                      setSelectedUniverseId(null);
                      setSelectedUniverseVersion(undefined);
                      setSelectedTaskId(null);
                      setEditTaskId(null);
                      setCloneTaskId(null);
                      setSelectedAgentId(null);
                      setSelectedSkillId(null);
                      setPlaygroundInitEnvId(null);
                      setPlaygroundInitUniverseId(null);
                      setPlaygroundInstanceId(null);
                      setTaskRunnerTaskId(null);
                      setEvaluatorOnlyTaskId(null);
                      setRerunFromStep(null);
                      setSelectedDeliveryId(null);
                      setSavedContextJson(null);
                      pushRoute(page);
                    }}
                    title={sidebarOpen ? undefined : label}
                    className={`flex items-center ${
                      sidebarOpen ? 'gap-2.5 px-3' : 'justify-center px-0'
                    } py-2 rounded-md text-sm transition-colors text-left ${
                      sidebarActivePage === page
                        ? 'bg-[var(--secondary)] text-[var(--foreground)] font-medium'
                        : 'text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)]'
                    }`}
                  >
                    <Icon size={16} className="flex-shrink-0" />
                    {sidebarOpen && label}
                  </Link>
                ))}
              </div>
            ))}
          </div>
          <div
            className={`mt-auto p-3 flex ${
              sidebarOpen ? 'justify-end' : 'justify-center'
            }`}
          >
            <button
              onClick={() => setSidebarOpen(prev => !prev)}
              className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              {sidebarOpen ? (
                <PanelLeftClose size={18} />
              ) : (
                <PanelLeftOpen size={18} />
              )}
            </button>
          </div>
        </nav>
      )}

      {/* Main content — gate on initialized to avoid flashing home before URL is read */}
      <main className="flex-1 overflow-auto">
        <>
            {activePage === 'home' && initialized && (
              <HomePage
                onNavigate={page => {
                  setActivePage(page);
                  pushRoute(page);
                }}
              />
            )}
            {activePage === 'environments' && (
              <EnvironmentsPage onSelectEnv={navigateToEnvDetail} />
            )}
            {activePage === 'env-detail' && selectedEnvId && (
              <EnvDetailPage
                envId={selectedEnvId}
                onBack={handleEnvBack}
                onNavigateToEnv={navigateToEnvDetail}
                onNavigateToUniverse={navigateToUniverseDetail}
              />
            )}
            {activePage === 'universes' && (
              <UniversesPage onSelectUniverse={navigateToUniverseDetail} />
            )}
            {activePage === 'universe-detail' && selectedUniverseId && (
              <UniverseDetailPage
                universeId={selectedUniverseId}
                initialVersion={selectedUniverseVersion}
                onBack={() => {
                  setActivePage('universes');
                  pushRoute('universes');
                }}
              />
            )}
            {activePage === 'tasks' && (
              <TasksHubPage onSelectTask={navigateToTaskDetail} />
            )}
            {activePage === 'task-detail' && selectedTaskId && (
              <TaskDetailPage
                taskId={selectedTaskId}
                onBack={() => {
                  setActivePage('tasks');
                  pushRoute('tasks');
                }}
              />
            )}
            {activePage === 'docs' && <DocsPage />}
            {activePage === 'agents' && (
              <AgentsHubPage onSelectAgent={navigateToAgentDetail} />
            )}
            {activePage === 'agent-detail' && selectedAgentId && (
              <AgentDetailPage
                agentId={selectedAgentId}
                onBack={() => {
                  setActivePage('agents');
                  pushRoute('agents');
                }}
              />
            )}
            {activePage === 'task-runner' && (
              <TaskRunnerPage
                taskId={taskRunnerTaskId}
                savedContextJson={savedContextJson}
                onContextCaptured={setSavedContextJson}
              />
            )}
        </>
      </main>
    </div>
  );
}
