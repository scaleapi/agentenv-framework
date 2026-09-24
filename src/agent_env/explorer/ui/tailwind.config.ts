import type { Config } from 'tailwindcss';
import {
  borderRadius,
  brand,
  fontFamily,
  fontSize,
  fontWeight,
  gray,
  scale,
  semantic,
} from './src/styles/tokens';

/**
 * Tailwind theme, extended with the explorer's design tokens.
 *
 * Source of truth lives in `src/styles/tokens.ts`; the semantic color aliases
 * point at the CSS custom properties declared in `src/styles/globals.css`
 * (`--foreground`, `--border`, …), which is what the app's ~1000 existing
 * `text-[var(--…)]` / `border-[var(--…)]` call sites consume. The extension
 * here adds first-class utilities on top of that: the neutral `gray` ramp, the
 * 24 `brand-*` model colors, the `scale-*` accent hues, the Inter/Geist-Mono
 * font stacks, the named type scale (`text-h1`, `text-body`, `text-mono`, …),
 * and the design-system corner radii.
 *
 * The component-library styling comes from `@radix-ui/themes/styles.css`
 * (imported in `_app.tsx`), themed via the `<Theme>` props there.
 */
export default {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ...semantic,
        gray,
        brand,
        scale,
      },
      fontFamily: fontFamily as unknown as Record<string, string[]>,
      // Cast through `unknown`: the token tuples are readonly (`as const`),
      // while Tailwind's `fontSize` expects mutable `[size, { … }]` entries.
      fontSize: fontSize as unknown as Record<
        string,
        [string, { lineHeight: string; fontWeight: string }]
      >,
      fontWeight,
      borderRadius,
    },
  },
  plugins: [],
} satisfies Config;
