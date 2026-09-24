/**
 * agent-env explorer design tokens — the single source of truth for the UI theme.
 *
 * Values follow the agent-env explorer design system (color, typography,
 * spacing and component scales).
 *
 * These objects are consumed by `tailwind.config.ts` (colors, fontFamily,
 * fontSize, fontWeight, borderRadius). The semantic colors are also mirrored as
 * CSS custom properties in `src/styles/globals.css` (`--background`,
 * `--foreground`, …) — that is what the ~1000 existing `var(--…)` call sites
 * consume, so updating them there performs the actual visual swap. The Tailwind
 * `colors.*` aliases below simply point at those same variables so both
 * `text-[var(--foreground)]` and `text-foreground` resolve identically.
 */

/** Neutral gray ramp (Color frame). */
export const gray = {
  50: '#fafafa',
  100: '#f5f5f5',
  200: '#ededed',
  300: '#e5e5e5',
  400: '#d9d9d9',
  500: '#a3a3a3',
  600: '#737373',
  700: '#525252',
  800: '#333333',
  900: '#1a1a1a',
  950: '#111111',
} as const;

/**
 * Per-provider brand colors (24) for model / leaderboard visualizations.
 * Use as `bg-brand-anthropic`, `text-brand-openai`, etc.
 */
export const brand = {
  anthropic: '#e27553',
  openai: '#65c2b9',
  google: '#598dd2',
  amazon: '#e09a3b',
  'moonshot-ai': '#a37cad',
  qwen: '#e072af',
  meta: '#6fcbef',
  xai: '#d1b745',
  mistral: '#79c96e',
  minimax: '#8d86e8',
  'microsoft-phi': '#d95358',
  deepseek: '#e39cab',
  xiaomi: '#bd836f',
  'liquid-ai': '#a69e66',
  manus: '#62a16e',
  nous: '#75a4b5',
  perplexity: '#817ead',
  writer: '#d97d7c',
  zhipu: '#edcca3',
  together: '#c7a678',
  nvidia: '#9ab896',
  samsung: '#8fb6eb',
  alibaba: '#d1b4c6',
  ai2: '#ddb5eb',
} as const;

/** Fixed accent hues (Color frame — accent/*). */
export const scale = {
  black: '#000000',
  red: '#e27553',
  teal: '#65c2b9',
  blue: '#598dd2',
  cyan: '#6fcbef',
} as const;

/**
 * Semantic color aliases, backed by the CSS custom properties declared in
 * `globals.css`. Kept in sync with the design system's semantic tokens:
 *   text/primary #111 · secondary #525252 · tertiary #737373 · muted #a3a3a3
 *   background/primary #fff · secondary #f5f5f5 · tertiary #fafafa
 *   surface/default #fff · muted #f5f5f5
 *   border/default #d9d9d9 · subtle #e5e5e5 · muted #737373
 *   accent/primary #000
 */
export const semantic = {
  background: 'var(--background)',
  'background-tertiary': 'var(--background-tertiary)',
  foreground: 'var(--foreground)',
  card: 'var(--card)',
  'card-foreground': 'var(--card-foreground)',
  popover: 'var(--popover)',
  'popover-foreground': 'var(--popover-foreground)',
  surface: 'var(--surface)',
  'surface-muted': 'var(--surface-muted)',
  primary: 'var(--primary)',
  'primary-foreground': 'var(--primary-foreground)',
  secondary: 'var(--secondary)',
  'secondary-foreground': 'var(--secondary-foreground)',
  muted: 'var(--muted)',
  'muted-foreground': 'var(--muted-foreground)',
  accent: 'var(--accent)',
  'accent-foreground': 'var(--accent-foreground)',
  destructive: 'var(--destructive)',
  border: 'var(--border)',
  'border-subtle': 'var(--border-subtle)',
  'border-strong': 'var(--border-strong)',
  input: 'var(--input)',
  ring: 'var(--ring)',
} as const;

/** Font stacks. `--font-inter` / `--font-geist-mono` are wired in `_app.tsx`. */
export const fontFamily = {
  sans: [
    'var(--font-inter)',
    'ui-sans-serif',
    'system-ui',
    '-apple-system',
    'Segoe UI',
    'Roboto',
    'Helvetica Neue',
    'Arial',
    'sans-serif',
  ],
  mono: [
    'var(--font-geist-mono)',
    'ui-monospace',
    'SFMono-Regular',
    'Menlo',
    'Monaco',
    'Consolas',
    'Liberation Mono',
    'Courier New',
    'monospace',
  ],
} as const;

/**
 * Type scale (Typography frame) as Tailwind `[fontSize, { lineHeight,
 * fontWeight }]` tuples. Inter for UI, Geist Mono for data/code.
 */
export const fontSize = {
  'display-lg': ['60px', { lineHeight: '75px', fontWeight: '300' }],
  'display-sm': ['36px', { lineHeight: '45px', fontWeight: '600' }],
  h1: ['32px', { lineHeight: '40px', fontWeight: '600' }],
  h2: ['30px', { lineHeight: '37.5px', fontWeight: '600' }],
  h3: ['28px', { lineHeight: '35px', fontWeight: '600' }],
  h4: ['20px', { lineHeight: '27.5px', fontWeight: '600' }],
  h5: ['18px', { lineHeight: '24.75px', fontWeight: '500' }],
  'body-lg': ['16px', { lineHeight: '26.4px', fontWeight: '400' }],
  body: ['14px', { lineHeight: '25.2px', fontWeight: '400' }],
  label: ['14px', { lineHeight: '21px', fontWeight: '500' }],
  'label-sm': ['12px', { lineHeight: '18px', fontWeight: '500' }],
  caption: ['12px', { lineHeight: '18px', fontWeight: '400' }],
  mono: ['12px', { lineHeight: '18px', fontWeight: '400' }],
  'mono-md': ['12px', { lineHeight: '18px', fontWeight: '500' }],
} as const;

/** Font weights (Typography frame). */
export const fontWeight = {
  light: '300',
  normal: '400',
  medium: '500',
  semibold: '600',
} as const;

/** Corner radii (Spacing frame — radius/*). Absolute px per the design system. */
export const borderRadius = {
  none: '0px',
  sm: '4px',
  md: '8px',
  lg: '12px',
  xl: '16px',
  full: '9999px',
} as const;

/**
 * Spacing scale (Spacing frame), in px. Exposed as CSS custom properties
 * (`--space-*`) in `globals.css`. NOT spread onto Tailwind's numeric spacing
 * scale on purpose: this app pins `html { font-size: 14px }`, so Tailwind's
 * rem-based spacing already renders on a 0.875× rhythm the whole layout relies
 * on. Overriding those keys with absolute px would shift every margin/pad. The
 * default Tailwind numeric scale (0,0.5,1,1.5,2,3,4,5,6,8,10,12,14,16,20,24,32,36)
 * already maps 1:1 to these px values at a 16px root, so the scale itself is
 * unchanged — this is the documented token record of it.
 */
export const spacing = [
  0, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 128, 144,
] as const;
